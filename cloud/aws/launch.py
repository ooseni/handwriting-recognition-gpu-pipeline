"""Run ROAD jobs on EC2: the GCP pipeline's job runner and checker, on AWS.

Run inside WSL, where the aws CLI is logged in (aws sso login):

    python3 cloud/aws/launch.py setup        # private bucket + an IAM role limited to it
    python3 cloud/aws/launch.py stage        # code, CSVs, job specs -> s3
    python3 cloud/aws/launch.py rehearse     # one CPU VM at smoke size, ~$0.05
    python3 cloud/aws/launch.py create       # the three experiments, one L40S each
    ... --dry-run                            # print every aws command, run none

The three jobs each change ONE thing from qwen4b_all's recipe (today's defaults on all
data -- every setting added since defaults to the old behaviour), decoded like
qwen4b_all_8beams.csv (0.9066 public), so each gets a clean leaderboard read:

    qwen4b_all_stackmix   + 2,500 StackMix lines (no heavy augmentation, jitter 0.25)
    qwen4b_all_5ep        5 epochs instead of 3
    qwen4b_all_seed7      seed 7 instead of 42: the leaderboard's noise between two
                          identical recipes, which every other comparison needs

Cost guards (see boot.sh): each instance powers itself off MAX_HOURS after every boot,
and is launched to stop, never terminate, on that hard cap. IDLE minutes after its job
ends it terminates itself once the job's DONE or FAILED is in the bucket, or stops with
its disk kept if not, so an ended job never sits billing and results are never lost.

Configuration comes from the environment: ROAD_AWS_ACCOUNT (your 12-digit account ID)
and, optionally, ROAD_AWS_REGION.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGION = os.environ.get("ROAD_AWS_REGION", "us-east-1")
ACCOUNT = os.environ.get("ROAD_AWS_ACCOUNT", "123456789012")  # placeholder: set your own
BUCKET = f"road-htr-{ACCOUNT}"
ROLE = "road-htr-vm"                       # also the instance profile's name
CPU_TYPE = "t3.xlarge"                     # the rehearsal: 4 vCPU, no GPU
# In order of preference: (type, vCPUs, GPU GB). Capacity is per type and zone -- on
# 2026-10-02 no us-east-1 zone had a g6e.xlarge -- and the G/VT quota is 12 vCPUs.
GPU_TYPES = [("g6e.xlarge", 4, 48),        # 1 x L40S
             ("g6e.2xlarge", 8, 48),       # the same GPU, more CPU
             ("g5.xlarge", 4, 24),         # 1 x A10G
             ("g6.xlarge", 4, 24)]         # 1 x L4, as on GCP
VCPU_QUOTA = 12
AMI_PARAM = ("/aws/service/deeplearning/ami/x86_64/"
             "base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id")
ZONES = ["us-east-1a", "us-east-1b", "us-east-1c", "us-east-1d"]
CSV_SRC = "/opt/road/csv"
CAPACITY_ERRORS = ("InsufficientInstanceCapacity", "Unsupported")

DECODE = ["--beams", "8", "--n-best", "5"]      # as qwen4b_all_8beams.csv was decoded
SYNTH = ["--synth", "data/synth/stackmix.csv"]
EXPERIMENTS = {
    "qwen4b_all_stackmix": SYNTH,
    "qwen4b_all_5ep": ["--epochs", "5"],
    "qwen4b_all_seed7": ["--seed", "7"],
}
MAX_HOURS = {"qwen4b_all_stackmix": 8, "qwen4b_all_5ep": 10, "qwen4b_all_seed7": 8}


def qwen(*args: str) -> list[str]:
    return ["python", "-m", "road.qwenvl", *args]


def steps(run: str, change: list[str], smoke: bool = False) -> list[dict]:
    flags = [*change, *DECODE]
    tiny = ["--model", "tiny", "--workers", "0"] if smoke else []
    out = [{"name": "data", "cmd": ["python", "-m", "road.setup_data", "--csv-src", CSV_SRC]}]
    if "--synth" in change:
        out.append({"name": "stackmix", "cmd": ["python", "-m", "road.stackmix", "--n",
                                                "24" if smoke else "2500"]})
    out += [
        {"name": "preflight", "cmd": qwen("--run", f"preflight_{run}", "--smoke", *flags, *tiny)},
        {"name": "train", "cmd": qwen("--run", run, "--fold", "all", *flags,
                                      *(["--smoke", *tiny, "--keep-weights"] if smoke else []))},
    ]
    return out


def job_specs() -> dict[str, dict]:
    return {run: {"job": run, "steps": steps(run, change), "sync_minutes": 5,
                  "outputs": [run, f"preflight_{run}"]}
            for run, change in EXPERIMENTS.items()}


def rehearsal_specs() -> dict[str, dict]:
    """The StackMix job at smoke size on a CPU, plus an ensemble of it."""
    run = "aws_rehearse"
    body = steps(run, SYNTH, smoke=True) + [{"name": "ens", "cmd": [
        "python", "-m", "road.ensemble", "--runs", run, run, "--candidates-from", run,
        "--out", "aws_rehearse_ens", "--beams", "3", "--partial"]}]
    return {run: {"job": run, "steps": body, "sync_minutes": 2,
                  "outputs": [run, f"preflight_{run}", "aws_rehearse_ens"]}}


def render_boot(job: str, cpu: bool, idle: int, max_hours: int) -> str:
    text = (ROOT / "cloud/aws/boot.sh").read_text(encoding="utf-8").replace("\r\n", "\n")
    for key, value in {"BUCKET": BUCKET, "JOB": job, "IDLE": str(idle),
                       "CPU": "1" if cpu else "0", "MAX_HOURS": str(max_hours),
                       "REGION": REGION}.items():
        text = text.replace(f"__{key}__", value)
    if left := re.findall(r"__[A-Z][A-Z_]*__", text):
        raise SystemExit(f"boot.sh: placeholders left unfilled: {left}")
    return text


class Aws:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

    def __call__(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        cmd = ["aws", *args]
        print("$ " + " ".join(shlex.quote(a) for a in cmd), flush=True)
        if self.dry_run:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if check and r.returncode:
            raise SystemExit(f"aws failed ({r.returncode}):\n{r.stderr.strip()[-2000:]}")
        return r


def setup(aws: Aws) -> None:
    """Idempotent: the private bucket, and a role that can reach only that bucket."""
    if aws("s3api", "head-bucket", "--bucket", BUCKET, check=False).returncode:
        aws("s3api", "create-bucket", "--bucket", BUCKET, "--region", REGION)
    aws("s3api", "put-public-access-block", "--bucket", BUCKET, "--public-access-block-"
        "configuration", "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,"
        "RestrictPublicBuckets=true")
    aws("s3api", "put-bucket-tagging", "--bucket", BUCKET, "--tagging",
        "TagSet=[{Key=purpose,Value=road-htr}]")
    if aws("iam", "get-role", "--role-name", ROLE, check=False).returncode:
        trust = {"Version": "2012-10-17", "Statement": [{
            "Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole"}]}
        aws("iam", "create-role", "--role-name", ROLE, "--assume-role-policy-document",
            json.dumps(trust), "--tags", "Key=purpose,Value=road-htr")
    access = {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": "s3:ListBucket", "Resource": f"arn:aws:s3:::{BUCKET}"},
        {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"],
         "Resource": f"arn:aws:s3:::{BUCKET}/*"},
        # so an ended job's VM can terminate itself -- and only our own VMs, by tag
        {"Effect": "Allow", "Action": "ec2:TerminateInstances",
         "Resource": f"arn:aws:ec2:{REGION}:{ACCOUNT}:instance/*",
         "Condition": {"StringEquals": {"aws:ResourceTag/purpose": "road-htr"}}}]}
    aws("iam", "put-role-policy", "--role-name", ROLE, "--policy-name", "road-htr-bucket",
        "--policy-document", json.dumps(access))
    if aws("iam", "get-instance-profile", "--instance-profile-name", ROLE,
           check=False).returncode:
        aws("iam", "create-instance-profile", "--instance-profile-name", ROLE)
        aws("iam", "add-role-to-instance-profile", "--instance-profile-name", ROLE,
            "--role-name", ROLE)


def code_tarball(dest: Path) -> int:
    """The working tree as git sees it, as on GCP. safe.directory: WSL's git does not
    own a repository on the Windows drive."""
    out = subprocess.run(["git", "-c", f"safe.directory={ROOT}", "ls-files", "-co",
                          "--exclude-standard"], cwd=ROOT, capture_output=True, text=True,
                         check=True).stdout.split("\n")
    files = [f for f in out if f and (ROOT / f).is_file()]
    with tarfile.open(dest, "w:gz") as tar:
        for f in files:
            tar.add(ROOT / f, arcname=f)
    return len(files)


def stage(aws: Aws, specs: dict[str, dict]) -> None:
    store = f"s3://{BUCKET}"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        print(f"code: {code_tarball(tmp / 'repo.tar.gz')} files", flush=True)
        aws("s3", "cp", "--only-show-errors", str(tmp / "repo.tar.gz"), f"{store}/code/repo.tar.gz")
        for name in ("Train.csv", "Test.csv", "SampleSubmission.csv"):
            aws("s3", "cp", "--only-show-errors", str(ROOT / "data" / name), f"{store}/csv/{name}")
        for name, spec in specs.items():
            (tmp / f"{name}.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
            aws("s3", "cp", "--only-show-errors", str(tmp / f"{name}.json"),
                f"{store}/jobs/{name}.json")


def create(aws: Aws, job: str, itype: str, cpu: bool, idle: int, max_hours: int) -> str:
    """One instance for one job, trying zones in order. Returns the instance id."""
    store = f"s3://{BUCKET}"
    old = aws("s3", "cp", "--only-show-errors", f"{store}/vms/{job}.json", "-", check=False)
    if old.returncode == 0 and old.stdout.strip():
        prev = json.loads(old.stdout)
        st = aws("ec2", "describe-instances", "--region", REGION, "--instance-ids",
                 prev["instance"], "--query", "Reservations[0].Instances[0].State.Name",
                 "--output", "text", check=False).stdout.strip()
        if st in ("pending", "running", "stopping", "shutting-down"):
            raise SystemExit(f"{prev['instance']} ({job}) is {st} -- not replacing a live VM")
        if st == "stopped":
            print(f"replacing stopped {prev['instance']}", flush=True)
            aws("ec2", "terminate-instances", "--region", REGION, "--instance-ids",
                prev["instance"])
    aws("s3", "rm", "--only-show-errors", "--recursive", f"{store}/status/{job}/", check=False)

    ami = aws("ssm", "get-parameter", "--region", REGION, "--name", AMI_PARAM, "--query",
              "Parameter.Value", "--output", "text").stdout.strip() or "ami-DRYRUN"
    root_dev = aws("ec2", "describe-images", "--region", REGION, "--image-ids", ami, "--query",
                   "Images[0].RootDeviceName", "--output", "text",
                   check=False).stdout.strip() or "/dev/sda1"
    subnets = json.loads(aws("ec2", "describe-subnets", "--region", REGION, "--filters",
                             "Name=default-for-az,Values=true", "--query",
                             "Subnets[].[AvailabilityZone,SubnetId]", "--output",
                             "json").stdout or "[]")
    by_zone = dict(subnets) or {z: "subnet-DRYRUN" for z in ZONES}
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, encoding="utf-8",
                                     newline="\n") as f:
        f.write(render_boot(job, cpu, idle, max_hours))
    tags = (f"ResourceType=instance,Tags=[{{Key=Name,Value=road-{job.replace('_', '-')}}},"
            f"{{Key=purpose,Value=road-htr}},{{Key=job,Value={job}}}]")
    tried = []
    for zone in [z for z in ZONES if z in by_zone]:
        r = aws("ec2", "run-instances", "--region", REGION, "--image-id", ami,
                "--instance-type", itype, "--subnet-id", by_zone[zone],
                "--iam-instance-profile", f"Name={ROLE}",
                "--block-device-mappings",
                f"DeviceName={root_dev},Ebs={{VolumeSize=200,VolumeType=gp3,"
                f"DeleteOnTermination=true}}",
                "--metadata-options", "HttpTokens=required,HttpEndpoint=enabled",
                "--instance-initiated-shutdown-behavior", "stop",
                "--user-data", f"file://{f.name}", "--tag-specifications", tags,
                "--count", "1", "--query", "Instances[0].InstanceId", "--output", "text",
                check=False)
        if r.returncode == 0:
            instance = r.stdout.strip() or "i-DRYRUN"
            Path(f.name).unlink()
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as v:
                json.dump({"instance": instance, "zone": zone, "cloud": "aws",
                           "type": itype}, v)
            aws("s3", "cp", "--only-show-errors", v.name, f"{store}/vms/{job}.json")
            Path(v.name).unlink()
            print(f"{job}: {instance} ({itype}) started in {zone}", flush=True)
            return instance
        if any(e in r.stderr for e in CAPACITY_ERRORS):
            tried.append(f"{zone}: no capacity")
            print(f"  {zone}: no {itype} capacity -- trying the next zone", flush=True)
            continue
        Path(f.name).unlink()
        raise SystemExit(f"could not start {job} in {zone}:\n{r.stderr.strip()[-2000:]}")
    Path(f.name).unlink()
    raise NoCapacity(f"no zone had {itype} capacity:\n  " + "\n  ".join(tried))


class NoCapacity(SystemExit):
    """Every zone refused this instance type: worth trying another type, not a fault."""


def spec_for(job: str, gpu_gb: int) -> dict:
    """The job's spec for a GPU of this size. A 24 GB card decodes 4 lines at a time,
    as the L4 had to on GCP; that changes how lines are batched, not the model."""
    spec = job_specs()[job]
    if gpu_gb < 40:
        for step in spec["steps"]:
            if step["name"] in ("preflight", "train"):
                step["cmd"] = [*step["cmd"], "--decode-batch-size", "4"]
    return spec


# --- the fold-0 judge matrix ------------------------------------------------------------
# Every fold-0 model judges one shared pool of fold-0 candidates; scripts/judge_matrix.py
# then evaluates every subset of judges offline. Split three ways so three GPUs share the
# scoring; every job pools the same eight runs, so the pools are identical.
F0_RUNS = ["qwen4b_f0", "qwen4b_f0_tall", "qwen4b_f0_v2", "qwen4b_f0_jitter",
           "qwen4b_f0_combo", "qwen4b_f0_stackmix", "qwen4b_f0_heavy", "trocr_f0"]
MATRIX = {"oof_matrix_a": ["qwen4b_f0", "qwen4b_f0_tall", "qwen4b_f0_v2", "trocr_f0"],
          "oof_matrix_b": ["qwen4b_f0_jitter", "qwen4b_f0_combo"],
          "oof_matrix_c": ["qwen4b_f0_stackmix", "qwen4b_f0_heavy"]}
MATRIX_REUSE = ["oof_2judge", "oof_3judge"]      # September's fold-0 scores
WEIGHTS = {False: "adapter/adapter_model.safetensors",   # a Qwen LoRA run
           True: "model/model.safetensors"}               # a TrOCR run


def matrix_spec(job: str, gpu_gb: int) -> dict:
    judges = MATRIX[job]
    others = [r for r in F0_RUNS if r not in judges]
    batch = "16" if gpu_gb >= 40 else "4"
    cmd = ["python", "-m", "road.ensemble", "--runs", *judges, "--candidates-from", *others,
           "--out", job, "--oof", "--beams", "3", "--batch-size", batch,
           "--reuse", *MATRIX_REUSE]
    # The judges' weights reach S3 over a home connection while the VM installs, so the
    # job pulls whatever is there at the start (the predictions, uploaded first) and then
    # waits for each of its own judges' weights; an S3 object appears only once complete.
    waits = [{"name": f"wait-{j}", "wait_for": f"runs/{j}/{WEIGHTS[j.startswith('trocr')]}",
              "timeout_minutes": 300, "poll_seconds": 60, "pull": [f"runs/{j}"]}
             for j in judges]
    return {"job": job, "sync_minutes": 5, "outputs": [job],
            "pull": [f"runs/{r}" for r in F0_RUNS + MATRIX_REUSE],
            "steps": [{"name": "data", "cmd": ["python", "-m", "road.setup_data",
                                               "--csv-src", CSV_SRC]},
                      *waits,
                      {"name": "score", "cmd": cmd}]}


def missing_inputs(aws: Aws, spec: dict) -> list[str]:
    """Objects the job's wait steps would wait for that are not in S3 yet."""
    return [s["wait_for"] for s in spec["steps"] if "wait_for" in s and aws(
        "s3api", "head-object", "--bucket", BUCKET, "--key", s["wait_for"],
        check=False).returncode]


# --- the final test ensemble ------------------------------------------------------------
# The judge matrix on fold 0: all eight fold-0 judges beat the two-judge baseline by
# +0.0116 (5.1 sd), TrOCR among them carrying most of it, so the test set gets every
# all-data judge we have. Three GPUs split the scoring over one shared pool; every earlier
# test ensemble's scores are reused, so the old judges only score new candidates.
# scripts/combine_judges.py then sums all nine.
FINAL_JUDGES = {
    "test_matrix_a": ["qwen4b_all_seed7", "qwen4b_all_5ep", "qwen4b_all"],
    "test_matrix_b": ["qwen4b_all_stackmix", "qwen4b_all_jitter", "trocr_all"],
    "test_matrix_c": ["qwen8b_all", "qwen8b_all_combo", "qwen4b_all_combo"],
}
FINAL_SOURCES = ["qwen4b_combined", "trocr_f0", "qwen4b_f0", "qwen4b_f0_v2", "qwen4b_f0_tall",
                 "qwen4b_f0_jitter", "qwen4b_f0_combo", "qwen4b_f0_stackmix", "qwen4b_f0_heavy"]
FINAL_REUSE = ["ens_nbest", "ens_final", "ens_trocr_judge", "ens_combo", "ens_combo_3j",
               "ens_combo_swap", "ens_combo_pair"]


def final_spec(job: str, gpu_gb: int) -> dict:
    judges = FINAL_JUDGES[job]
    everyone = [j for js in FINAL_JUDGES.values() for j in js] + FINAL_SOURCES
    others = [r for r in everyone if r not in judges]
    batch = "16" if gpu_gb >= 40 else "4"
    cmd = ["python", "-m", "road.ensemble", "--runs", *judges, "--candidates-from", *others,
           "--out", job, "--beams", "3", "--batch-size", batch, "--reuse", *FINAL_REUSE]
    waits = [{"name": f"wait-{j}", "wait_for": f"runs/{j}/{WEIGHTS[j.startswith('trocr')]}",
              "timeout_minutes": 15, "poll_seconds": 60, "pull": [f"runs/{j}"]}
             for j in judges]
    return {"job": job, "sync_minutes": 5, "outputs": [job],
            "pull": [f"runs/{r}" for r in everyone + FINAL_REUSE],
            "steps": [{"name": "data", "cmd": ["python", "-m", "road.setup_data",
                                               "--csv-src", CSV_SRC]},
                      *waits,
                      {"name": "score", "cmd": cmd}]}


def launch_gpu(aws: Aws, job: str, budget: int, idle: int, spec_fn=None,
               max_hours: int | None = None, allow_pending: bool = False) -> int:
    """Start the job on the first GPU type with capacity that fits the vCPU budget left.
    Returns the vCPUs it took. spec_fn(gpu_gb) builds the job's spec for that card.

    Refuses while any input the job waits for is still missing from S3: launching ahead
    of an upload is how a VM once sat idle for six hours."""
    if not allow_pending:
        probe = spec_fn(48) if spec_fn else spec_for(job, 48)
        if missing := missing_inputs(aws, probe):
            raise SystemExit(f"{job}: not launching -- inputs not in S3 yet: {missing} "
                             "(upload them first, or pass --allow-pending)")
    tried = []
    for itype, vcpus, gpu_gb in GPU_TYPES:
        if vcpus > budget:
            continue
        spec = spec_fn(gpu_gb) if spec_fn else spec_for(job, gpu_gb)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / f"{job}.json"
            p.write_text(json.dumps(spec, indent=1), encoding="utf-8")
            aws("s3", "cp", "--only-show-errors", str(p), f"s3://{BUCKET}/jobs/{job}.json")
        hours = (max_hours or MAX_HOURS[job]) * (1 if gpu_gb >= 40 else 2)   # 24 GB: slower
        try:
            create(aws, job, itype, cpu=False, idle=idle, max_hours=hours)
            return vcpus
        except NoCapacity:
            tried.append(itype)
            print(f"{job}: no {itype} anywhere in {REGION} -- trying the next type", flush=True)
    raise NoCapacity(f"{job}: none of {tried} had capacity within {budget} vCPUs")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["setup", "stage", "rehearse", "create", "matrix",
                                       "final"],
                    help="matrix: the fold-0 judge matrix; final: the 9-judge test ensemble "
                         "(both scoring only, no training)")
    ap.add_argument("jobs", nargs="*", help="create: which experiments (default: all three)")
    ap.add_argument("--idle-minutes", type=int, default=2,
                    help="minutes after its job ends before a VM terminates itself")
    ap.add_argument("--allow-pending", action="store_true",
                    help="launch even if inputs the job waits for are not in S3 yet")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    aws = Aws(args.dry_run)
    if args.action == "setup":
        setup(aws)
    elif args.action == "stage":
        stage(aws, job_specs())
    elif args.action in ("matrix", "final"):
        table, build = (MATRIX, matrix_spec) if args.action == "matrix" else \
            (FINAL_JUDGES, final_spec)
        jobs = args.jobs or list(table)
        if bad := [j for j in jobs if j not in table]:
            ap.error(f"unknown {args.action} job(s) {bad}: pick from {list(table)}")
        stage(aws, {j: build(j, 48) for j in jobs})
        budget, waiting = VCPU_QUOTA, []
        for job in jobs:
            try:
                budget -= launch_gpu(aws, job, budget, args.idle_minutes,
                                     spec_fn=lambda gb, job=job: build(job, gb),
                                     max_hours=4, allow_pending=args.allow_pending)
            except NoCapacity as e:
                print(f"!! {e}", flush=True)
                waiting.append(job)
        if waiting:
            print(f"not started for lack of capacity: {waiting}", flush=True)
            return 1
    elif args.action == "rehearse":
        stage(aws, rehearsal_specs())
        for job in rehearsal_specs():
            create(aws, job, CPU_TYPE, cpu=True, idle=2, max_hours=2)
    else:
        jobs = args.jobs or list(EXPERIMENTS)
        if bad := [j for j in jobs if j not in EXPERIMENTS]:
            ap.error(f"unknown job(s) {bad}: pick from {list(EXPERIMENTS)}")
        stage(aws, {j: s for j, s in job_specs().items() if j in jobs})
        budget, waiting = VCPU_QUOTA, []
        for job in jobs:
            try:
                budget -= launch_gpu(aws, job, budget, args.idle_minutes,
                                     allow_pending=args.allow_pending)
            except NoCapacity as e:
                print(f"!! {e}", flush=True)
                waiting.append(job)
        if waiting:
            print(f"not started for lack of capacity: {waiting} -- rerun "
                  f"`create {' '.join(waiting)}` later", flush=True)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
