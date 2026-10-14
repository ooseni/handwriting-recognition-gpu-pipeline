"""The AWS port: the S3 store, the EC2 checker, the boot script and the launcher.

The GCP pipeline's tests cover the job runner and checker logic; these cover what is
AWS-specific, with the aws CLI faked -- before an L40S hour is spent.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path.parent))
    return mod


job = load("job", ROOT / "cloud/gcp/job.py")
checker = load("checker", ROOT / "cloud/gcp/checker.py")
aws_launch = load("aws_launch", ROOT / "cloud/aws/launch.py")


class FakeS3:
    """Records aws CLI calls and answers them from a dict, as S3 would."""

    def __init__(self):
        self.objects, self.calls = {}, []

    def __call__(self, cmd, capture_output=True, text=True):
        self.calls.append(cmd)
        if cmd[1:3] == ["s3api", "head-object"]:
            key = cmd[cmd.index("--key") + 1]
            return subprocess.CompletedProcess(cmd, 0 if key in self.objects else 255, "", "")
        if cmd[1:3] == ["s3", "cp"] and cmd[-1] == "-":
            key = cmd[-2].split("/", 3)[3]
            ok = key in self.objects
            return subprocess.CompletedProcess(cmd, 0 if ok else 1, self.objects.get(key, ""), "")
        if cmd[1:3] == ["s3", "cp"]:
            src, dest = cmd[-2], cmd[-1]
            self.objects[dest.split("/", 3)[3]] = Path(src).read_text(encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")


def test_the_s3_store_speaks_aws_s3(monkeypatch, tmp_path):
    fake = FakeS3()
    monkeypatch.setattr(job.subprocess, "run", fake)
    st = job.Store("s3://b", aws="aws")
    assert st.s3 and not st.gcs
    st.put_text("status/j/status.json", '{"state": "running"}')
    assert st.get_text("status/j/status.json") == '{"state": "running"}'
    assert st.exists("status/j/status.json") and not st.exists("status/j/DONE")
    assert st.get_text("nope") is None
    (tmp_path / "run").mkdir()
    st.sync_up(tmp_path / "run", "runs/r")
    st.sync_down("runs/r", tmp_path / "back")
    sync = [c for c in fake.calls if c[1:3] == ["s3", "sync"]]
    assert sync[0][-2:] == [str(tmp_path / "run"), "s3://b/runs/r"]
    assert sync[1][-2:] == ["s3://b/runs/r", str(tmp_path / "back")]
    assert not any("--recursive" in c for c in sync)            # sync always recurses
    assert not any(c[1:3] == ["s3", "ls"] for c in fake.calls)   # ls matches prefixes


def test_the_s3_listing_trusts_an_etag_only_when_it_is_an_md5(monkeypatch):
    listing = {"Contents": [
        {"Key": "runs/r/log.csv", "Size": 102, "ETag": '"0cc175b9c0f1b6a831c399e269772661"'},
        {"Key": "runs/r/adapter/adapter_model.safetensors", "Size": 322279296,
         "ETag": '"6d0f5e3b5a1f4e1a9c1d2b3c4d5e6f70-39"'}]}
    monkeypatch.setattr(checker.subprocess, "run", lambda cmd, **k: subprocess.CompletedProcess(
        cmd, 0, json.dumps(listing), ""))
    got = checker.listing(job.Store("s3://b"), "runs/r")
    assert got["runs/r/log.csv"] == {"size": 102, "md5": "0cc175b9c0f1b6a831c399e269772661"}
    assert got["runs/r/adapter/adapter_model.safetensors"] == {"size": 322279296, "md5": None}


@pytest.mark.parametrize("ec2, ours", [("running", "RUNNING"), ("stopped", "TERMINATED"),
                                       ("terminated", "GONE"), ("pending", "STAGING")])
def test_ec2_states_read_as_the_checker_expects(monkeypatch, ec2, ours):
    monkeypatch.setattr(checker.subprocess, "run", lambda cmd, **k: subprocess.CompletedProcess(
        cmd, 0, ec2 + "\n", ""))
    assert checker.AwsCompute("us-east-1").state("i-1", "us-east-1a") == ours


def test_a_missing_instance_reads_as_gone(monkeypatch):
    monkeypatch.setattr(checker.subprocess, "run", lambda cmd, **k: subprocess.CompletedProcess(
        cmd, 254, "", "InvalidInstanceID.NotFound"))
    assert checker.AwsCompute("us-east-1").state("i-1", "z") == "GONE"


def test_each_experiment_changes_one_thing_from_the_plain_recipe():
    specs = aws_launch.job_specs()
    assert set(specs) == {"qwen4b_all_stackmix", "qwen4b_all_5ep", "qwen4b_all_seed7"}
    for run, spec in specs.items():
        train = next(s for s in spec["steps"] if s["name"] == "train")["cmd"]
        assert train[:7] == ["python", "-m", "road.qwenvl", "--run", run, "--fold", "all"]
        assert train[7:] == [*aws_launch.EXPERIMENTS[run], "--beams", "8", "--n-best", "5"]
        assert any(s["name"] == "stackmix" for s in spec["steps"]) == (run == "qwen4b_all_stackmix")
        pre = next(s for s in spec["steps"] if s["name"] == "preflight")["cmd"]
        assert "--smoke" in pre and pre[pre.index("--smoke") + 1:] == train[7:]


def test_the_boot_script_is_filled_in_and_unix(tmp_path):
    text = aws_launch.render_boot("qwen4b_all_5ep", cpu=False, idle=45, max_hours=10)
    assert "\r" not in text and not re.findall(r"__[A-Z][A-Z_]*__", text)
    assert 'BUCKET="road-htr-123456789012"' in text and 'JOB="qwen4b_all_5ep"' in text
    assert 'MAX_HOURS="10"' in text and 'export AWS_DEFAULT_REGION="us-east-1"' in text
    assert "shutdown -h \"+$((MAX_HOURS * 60))\"" in text                # the hard cap
    assert "systemctl enable road.service" in text                      # every boot
    assert b"\r" not in (ROOT / "cloud/aws/boot.sh").read_bytes()


class FakeAws:
    def __init__(self, run_errors=(), vm=None, state="stopped"):
        self.errors, self.calls, self.vm, self.state = list(run_errors), [], vm, state
        self.user_data = None

    def __call__(self, *args, check=True):
        self.calls.append(args)
        out, code, err = "", 0, ""
        if args[:2] == ("s3", "cp") and args[-1] == "-":
            code, out = (0, json.dumps(self.vm)) if self.vm else (1, "")
        elif args[:2] == ("ec2", "describe-instances"):
            out = self.state
        elif args[:2] == ("ec2", "describe-subnets"):
            out = json.dumps([["us-east-1a", "subnet-a"], ["us-east-1b", "subnet-b"]])
        elif args[:2] == ("ec2", "run-instances"):
            err = self.errors.pop(0) if self.errors else ""
            code, out = (1, "") if err else (0, "i-0abc")
            path = args[args.index("--user-data") + 1].removeprefix("file://")
            self.user_data = Path(path).read_text(encoding="utf-8")
        return subprocess.CompletedProcess(args, code, out, err)


def test_instances_stop_on_power_off_use_imdsv2_and_only_the_role():
    fake = FakeAws(run_errors=["InsufficientInstanceCapacity: none in us-east-1a"])
    assert aws_launch.create(fake, "qwen4b_all_5ep", "g6e.xlarge", cpu=False, idle=45,
                             max_hours=10) == "i-0abc"
    runs = [c for c in fake.calls if c[:2] == ("ec2", "run-instances")]
    assert [c[c.index("--subnet-id") + 1] for c in runs] == ["subnet-a", "subnet-b"]
    flags = runs[-1]
    assert flags[flags.index("--instance-initiated-shutdown-behavior") + 1] == "stop"
    assert flags[flags.index("--metadata-options") + 1].startswith("HttpTokens=required")
    assert flags[flags.index("--iam-instance-profile") + 1] == "Name=road-htr-vm"
    assert "VolumeType=gp3" in flags[flags.index("--block-device-mappings") + 1]
    assert 'JOB="qwen4b_all_5ep"' in fake.user_data


@pytest.mark.parametrize("state, replaced", [("stopped", True), ("running", False)])
def test_a_relaunch_replaces_a_stopped_instance_never_a_live_one(state, replaced):
    fake = FakeAws(vm={"instance": "i-old", "zone": "us-east-1a"}, state=state)
    if replaced:
        aws_launch.create(fake, "qwen4b_all_seed7", "g6e.xlarge", cpu=False, idle=45, max_hours=8)
        verbs = [c[:2] for c in fake.calls]
        assert verbs.index(("ec2", "terminate-instances")) < verbs.index(("ec2", "run-instances"))
    else:
        with pytest.raises(SystemExit, match="not replacing a live VM"):
            aws_launch.create(fake, "qwen4b_all_seed7", "g6e.xlarge", cpu=False, idle=45,
                              max_hours=8)


def test_the_role_reaches_only_the_bucket():
    fake = FakeAws()
    fake.__call__ = None   # unused; setup() goes through the instance below
    calls = []

    def recorder(*args, check=True):
        calls.append(args)
        return subprocess.CompletedProcess(args, 1 if args[1].startswith(("get-", "head-"))
                                           else 0, "", "")
    aws_launch.setup(recorder)
    policy = next(json.loads(c[c.index("--policy-document") + 1]) for c in calls
                  if c[:2] == ("iam", "put-role-policy"))
    s3 = [s for s in policy["Statement"] if s["Resource"].startswith("arn:aws:s3")]
    assert {s["Resource"] for s in s3} == {"arn:aws:s3:::road-htr-123456789012",
                                            "arn:aws:s3:::road-htr-123456789012/*"}
    actions = {a for s in s3 for a in
               ([s["Action"]] if isinstance(s["Action"], str) else s["Action"])}
    assert actions == {"s3:ListBucket", "s3:GetObject", "s3:PutObject"}
    # the one EC2 right: terminating a VM -- and only one tagged as ours
    ec2 = [s for s in policy["Statement"] if not s["Resource"].startswith("arn:aws:s3")]
    assert len(ec2) == 1 and ec2[0]["Action"] == "ec2:TerminateInstances"
    assert ec2[0]["Condition"] == {"StringEquals": {"aws:ResourceTag/purpose": "road-htr"}}
    block = next(c for c in calls if c[:2] == ("s3api", "put-public-access-block"))
    assert block[-1].count("=true") == 4


def test_a_type_with_no_capacity_falls_back_to_the_next_within_the_quota(monkeypatch):
    """No zone had a g6e.xlarge on 2026-10-02: the job must try the next type, and a
    24 GB card must decode in smaller batches."""
    started = []

    def create(aws, job, itype, cpu, idle, max_hours):
        if itype == "g6e.xlarge":
            raise aws_launch.NoCapacity("none")
        started.append((itype, max_hours))
        return "i-1"

    monkeypatch.setattr(aws_launch, "create", create)
    uploads = []
    rec = lambda *a, check=True: uploads.append(a) or subprocess.CompletedProcess(a, 0, "", "")
    assert aws_launch.launch_gpu(rec, "qwen4b_all_seed7", budget=12, idle=45) == 8
    assert started == [("g6e.2xlarge", 8)]                   # same L40S, more CPU
    started.clear()
    assert aws_launch.launch_gpu(rec, "qwen4b_all_seed7", budget=4, idle=45) == 4
    assert started == [("g5.xlarge", 16)]                    # 8 vCPUs no longer fit
    with pytest.raises(aws_launch.NoCapacity):
        aws_launch.launch_gpu(rec, "qwen4b_all_seed7", budget=0, idle=45)


def test_a_24_gb_card_decodes_four_lines_at_a_time():
    big, small = aws_launch.spec_for("qwen4b_all_5ep", 48), aws_launch.spec_for("qwen4b_all_5ep", 24)
    train = lambda s: next(x for x in s["steps"] if x["name"] == "train")["cmd"]
    assert "--decode-batch-size" not in train(big)
    assert train(small) == train(big) + ["--decode-batch-size", "4"]


def test_the_judge_matrix_jobs_share_one_pool_and_cover_every_judge():
    """Three GPUs split the judges; the offline analysis merges their score columns, which
    only works if every job pooled exactly the same candidates."""
    specs = {j: aws_launch.matrix_spec(j, 48) for j in aws_launch.MATRIX}
    judged, pools = [], []
    for job, spec in specs.items():
        cmd = next(s for s in spec["steps"] if s["name"] == "score")["cmd"]
        i, j, k = cmd.index("--runs"), cmd.index("--candidates-from"), cmd.index("--out")
        judged += cmd[i + 1:j]
        pools.append(set(cmd[i + 1:j]) | set(cmd[j + 1:k]))
        assert "--oof" in cmd and cmd[cmd.index("--batch-size") + 1] == "16"
        assert set(aws_launch.F0_RUNS) <= {p.removeprefix("runs/") for p in spec["pull"]}
    assert sorted(judged) == sorted(aws_launch.F0_RUNS)            # each judge exactly once
    assert all(p == set(aws_launch.F0_RUNS) for p in pools)       # one shared pool
    small = next(s for s in aws_launch.matrix_spec("oof_matrix_b", 24)["steps"]
                 if s["name"] == "score")["cmd"]
    assert small[small.index("--batch-size") + 1] == "4"


def test_each_matrix_job_waits_for_its_own_judges_weights_before_scoring():
    spec = aws_launch.matrix_spec("oof_matrix_a", 48)
    names = [s["name"] for s in spec["steps"]]
    assert names[0] == "data" and names[-1] == "score"
    waits = {s["name"].removeprefix("wait-"): s for s in spec["steps"] if "wait_for" in s}
    assert set(waits) == set(aws_launch.MATRIX["oof_matrix_a"])
    assert waits["qwen4b_f0"]["wait_for"] == "runs/qwen4b_f0/adapter/adapter_model.safetensors"
    assert waits["trocr_f0"]["wait_for"] == "runs/trocr_f0/model/model.safetensors"
    assert all(w["pull"] == [f"runs/{j}"] for j, w in waits.items())


def test_an_ended_job_terminates_its_vm_only_once_its_results_are_reported():
    """The VM that idled six hours: after its job ends it must be gone within minutes,
    terminating only when DONE or FAILED is in the bucket, stopping otherwise."""
    text = aws_launch.render_boot("j", cpu=False, idle=2, max_hours=8)
    body = text[text.index("idle_then_poweroff() {"):text.index("fail() {")]
    assert 'IDLE="2"' in text
    assert "if exists DONE || exists FAILED; then" in body
    assert 'aws ec2 terminate-instances --instance-ids "$id"' in body
    assert body.index("terminate-instances") < body.index("shutdown -h now")   # stop = fallback
    assert "X-aws-ec2-metadata-token" in text                                  # IMDSv2


def test_the_launcher_will_not_start_a_job_before_its_inputs_are_uploaded(monkeypatch):
    monkeypatch.setattr(aws_launch, "create", lambda *a, **k: "i-1")
    present = {"runs/qwen4b_f0/adapter/adapter_model.safetensors"}

    def s3(*args, check=True):
        if args[:2] == ("s3api", "head-object"):
            key = args[args.index("--key") + 1]
            return subprocess.CompletedProcess(args, 0 if key in present else 255, "", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    spec = lambda gb: aws_launch.matrix_spec("oof_matrix_a", gb)
    with pytest.raises(SystemExit, match="trocr_f0/model/model.safetensors"):
        aws_launch.launch_gpu(s3, "oof_matrix_a", 12, 2, spec_fn=spec, max_hours=4)
    assert aws_launch.launch_gpu(s3, "oof_matrix_a", 12, 2, spec_fn=spec, max_hours=4,
                                 allow_pending=True) == 4
    present |= {f"runs/{j}/adapter/adapter_model.safetensors" for j in
                ("qwen4b_f0_tall", "qwen4b_f0_v2")} | {"runs/trocr_f0/model/model.safetensors"}
    assert aws_launch.launch_gpu(s3, "oof_matrix_a", 12, 2, spec_fn=spec, max_hours=4) == 4


def test_vms_end_two_minutes_after_their_job_by_default():
    for path in ("cloud/aws/launch.py", "cloud/gcp/launch.py"):
        assert '"--idle-minutes", type=int, default=2' in (ROOT / path).read_text(encoding="utf-8")
    startup = (ROOT / "cloud/gcp/startup.sh").read_text(encoding="utf-8")
    assert "road-idle-minutes || echo 2)" in startup


def test_the_final_jobs_share_one_pool_and_cover_all_nine_judges():
    specs = {j: aws_launch.final_spec(j, 48) for j in aws_launch.FINAL_JUDGES}
    judged, pools = [], []
    for job, spec in specs.items():
        cmd = next(s for s in spec["steps"] if s["name"] == "score")["cmd"]
        i, j, k = cmd.index("--runs"), cmd.index("--candidates-from"), cmd.index("--out")
        judged += cmd[i + 1:j]
        pools.append(set(cmd[i + 1:j]) | set(cmd[j + 1:k]))
        assert "--oof" not in cmd and set(aws_launch.FINAL_REUSE) <= set(cmd[cmd.index("--reuse") + 1:])
        waits = [s for s in spec["steps"] if "wait_for" in s]
        assert all(w["timeout_minutes"] <= 30 for w in waits)
    assert len(judged) == len(set(judged)) == 9 and "trocr_all" in judged
    assert all(p == pools[0] for p in pools)                        # one shared pool
