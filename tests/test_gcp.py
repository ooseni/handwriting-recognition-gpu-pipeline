"""The GCP pipeline, with a directory standing in for the bucket and a fake for the VMs.

What must hold before a GPU hour is spent: a job's results reach the bucket even when a
step fails; a reboot resumes rather than repeats; a VM waiting on another ends cleanly
if that one fails; the checker deletes a VM only after every file verified; and the
cloud jobs run exactly the flags the notebooks were tested with.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GCP = ROOT / "cloud" / "gcp"


def load(name):
    spec = importlib.util.spec_from_file_location(name, GCP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(GCP))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(GCP))
    return mod


job = load("job")
checker = load("checker")
launch = load("launch")

PY = sys.executable


def py(code: str) -> list[str]:
    return ["python", "-c", code]


def make_job(tmp_path, steps, outputs=("out",), pull=()):
    spec = {"job": "j", "steps": steps, "outputs": list(outputs), "pull": list(pull),
            "sync_minutes": 0.001}
    bucket = tmp_path / "bucket"
    bucket.mkdir(exist_ok=True)
    runs = tmp_path / "vm" / "runs"
    return job.Job(spec, job.Store(str(bucket)), ROOT, runs, tmp_path / "vm" / "j.log",
                   heartbeat=0.05), bucket, runs


WRITE_OUT = ("import os, pathlib; p = pathlib.Path(os.environ['ROAD_RUNS'], 'out'); "
             "p.mkdir(parents=True, exist_ok=True); (p / 'pred.csv').write_text('ID,Target\\n')")


# --- job.py ---------------------------------------------------------------------------------

def test_a_job_syncs_its_outputs_then_marks_done(tmp_path):
    j, bucket, _ = make_job(tmp_path, [
        {"name": "train", "cmd": py("print('3 epochs x 4 batches'); print('    step 2/4'); "
                                    "print('  epoch  1  loss 0.5'); " + WRITE_OUT)},
        {"name": "ens", "cmd": py("print('ok')")}])
    assert j.run() == 0
    status = bucket / "status" / "j"
    done = json.loads((status / "DONE").read_text())
    assert done["state"] == "done" and done["done"] == ["train", "ens"]
    assert (bucket / "runs/out/pred.csv").read_text() == "ID,Target\n"
    manifest = json.loads((status / "manifest.json").read_text())
    assert manifest["runs/out/pred.csv"]["md5"] == job.md5(bucket / "runs/out/pred.csv")
    assert "step 2/4" in (status / "job.log").read_text()


def test_a_failing_step_still_saves_its_outputs_and_says_why(tmp_path):
    j, bucket, _ = make_job(tmp_path, [
        {"name": "train", "cmd": py(WRITE_OUT + "; print('CUDA out of memory'); "
                                                "raise SystemExit(3)")},
        {"name": "never", "cmd": py("raise SystemExit('must not run')")}])
    assert j.run() == 3
    failed = json.loads((bucket / "status/j/FAILED").read_text())
    assert failed["failed_step"] == "train" and failed["exit_code"] == 3
    assert any("CUDA out of memory" in line for line in failed["log_tail"])
    assert (bucket / "runs/out/pred.csv").exists()          # the work is not lost
    assert not (bucket / "status/j/DONE").exists()


def test_a_reboot_resumes_and_a_finished_job_never_reruns(tmp_path):
    j, bucket, _ = make_job(tmp_path, [
        {"name": "a", "cmd": py("raise SystemExit('already done -- must be skipped')")},
        {"name": "b", "cmd": py(WRITE_OUT)}])
    (bucket / "status/j").mkdir(parents=True)
    (bucket / "status/j/status.json").write_text(json.dumps({"done": ["a"]}))
    assert j.run() == 0 and (bucket / "status/j/DONE").exists()

    again, _, _ = make_job(tmp_path, [{"name": "x", "cmd": py("raise SystemExit(9)")}])
    assert again.run() == 0                                  # DONE: nothing reruns


def test_a_wait_pulls_what_it_waited_for(tmp_path):
    j, bucket, runs = make_job(tmp_path, [
        {"name": "wait", "wait_for": "status/other/DONE", "timeout_minutes": 1,
         "poll_seconds": 0.05, "pull": ["runs/theirs"]},
        {"name": "use", "cmd": py("import os, pathlib; "
                                  "assert pathlib.Path(os.environ['ROAD_RUNS'], 'theirs', "
                                  "'adapter.bin').read_text() == 'w'")}])
    (bucket / "status/other").mkdir(parents=True)
    (bucket / "status/other/DONE").write_text("{}")
    (bucket / "runs/theirs").mkdir(parents=True)
    (bucket / "runs/theirs/adapter.bin").write_text("w")
    assert j.run() == 0
    assert json.loads((bucket / "status/j/DONE").read_text())["skipped"] == []


def test_a_wait_on_a_failed_job_ends_this_one_cleanly(tmp_path):
    """The 8B's own run must not be lost because the 4B it waits on failed."""
    j, bucket, _ = make_job(tmp_path, [
        {"name": "train", "cmd": py(WRITE_OUT)},
        {"name": "wait", "wait_for": "status/other/DONE", "abort_if": "status/other/FAILED",
         "timeout_minutes": 1, "poll_seconds": 0.05},
        {"name": "pair", "cmd": py("raise SystemExit('must not run')")}])
    (bucket / "status/other").mkdir(parents=True)
    (bucket / "status/other/FAILED").write_text("{}")
    assert j.run() == 0
    done = json.loads((bucket / "status/j/DONE").read_text())
    assert done["skipped"] == ["wait", "pair"] and (bucket / "runs/out/pred.csv").exists()


def test_inputs_are_pulled_and_only_outputs_are_pushed(tmp_path):
    j, bucket, runs = make_job(tmp_path, [{"name": "s", "cmd": py(WRITE_OUT)}],
                               pull=["runs/judge"])
    (bucket / "runs/judge").mkdir(parents=True)
    (bucket / "runs/judge/scores.csv").write_text("s")
    assert j.run() == 0
    assert (runs / "judge/scores.csv").exists()
    assert set(json.loads((bucket / "status/j/manifest.json").read_text())) == \
        {"runs/out/pred.csv"}


def test_progress_reads_the_trainers_lines():
    p = job.Progress("j", [{"name": "train"}])
    p.begin("train")
    for text in ["qwen4b: fit 6598 lines, 3 epochs x 825 batches, 80.5M trainable",
                 "    step  400/825  loss 0.7", "  epoch  1  loss 0.63  held-out ...",
                 "    step  412/825  loss 0.3"]:
        p.line(text)
    s = p.snapshot()
    assert (s["epoch"], s["batch"], s["epochs"], s["batches"]) == (1, 412, 3, 825)
    assert s["train_fraction"] == pytest.approx((1 + 412 / 825) / 3, abs=1e-4)
    p.line("test: 1374 lines decoded (1225s) -> x.csv")
    assert p.snapshot()["phase"] == "decoding"


# --- checker.py -----------------------------------------------------------------------------

class FakeCompute:
    def __init__(self, state="RUNNING"):
        self.calls, self._state = [], state

    def state(self, name, zone):
        return self._state

    def delete(self, name, zone):
        self.calls.append(("delete", name))
        return True

    def stop(self, name, zone):
        self.calls.append(("stop", name))
        return True

    def start(self, name, zone):
        self.calls.append(("start", name))
        self._state = "RUNNING"
        return True


def finished_bucket(tmp_path, state="DONE"):
    j, bucket, _ = make_job(tmp_path, [{"name": "s", "cmd": py(
        WRITE_OUT + ("" if state == "DONE" else "; raise SystemExit(2)"))}])
    j.run()
    (bucket / "jobs").mkdir(exist_ok=True)
    (bucket / "jobs/j.json").write_text(json.dumps(j.spec))
    (bucket / "vms").mkdir(exist_ok=True)
    (bucket / "vms/j.json").write_text(json.dumps({"instance": "road-j", "zone": "z"}))
    return bucket


def test_checker_deletes_the_vm_only_after_every_file_verifies(tmp_path):
    bucket = finished_bucket(tmp_path)
    fake, dest = FakeCompute(), tmp_path / "home"
    c = checker.Checker(job.Store(str(bucket)), fake, dest, ["j"])
    assert c.run(every=0, once=True) == 0
    assert (dest / "out/pred.csv").exists() and fake.calls == [("delete", "road-j")]
    receipt = json.loads((dest / "_gcp/j/receipt.json").read_text())
    assert receipt["problems"] == [] and receipt["state"] == "done"


def test_checker_keeps_the_vm_when_the_bucket_lacks_what_the_vm_had(tmp_path):
    bucket = finished_bucket(tmp_path)
    manifest = json.loads((bucket / "status/j/manifest.json").read_text())
    manifest["runs/out/adapter.safetensors"] = {"size": 5, "md5": "0" * 32}
    (bucket / "status/j/manifest.json").write_text(json.dumps(manifest))
    fake = FakeCompute()
    checker.Checker(job.Store(str(bucket)), fake, tmp_path / "home", ["j"]).run(0, True)
    assert fake.calls == []                                   # nothing deleted on a doubt
    receipt = json.loads((tmp_path / "home/_gcp/j/receipt.json").read_text())
    assert any("adapter.safetensors" in p for p in receipt["problems"])


def test_checker_stops_a_failed_vm_rather_than_deleting_it(tmp_path):
    bucket = finished_bucket(tmp_path, state="FAILED")
    fake = FakeCompute()
    checker.Checker(job.Store(str(bucket)), fake, tmp_path / "home", ["j"]).run(0, True)
    assert fake.calls == [("stop", "road-j")]
    assert (tmp_path / "home/out/pred.csv").exists()          # partial results still home


def test_checker_restarts_a_stopped_vm_at_most_twice(tmp_path):
    bucket = tmp_path / "bucket"
    (bucket / "vms").mkdir(parents=True)
    (bucket / "jobs").mkdir()
    (bucket / "jobs/j.json").write_text("{}")
    (bucket / "vms/j.json").write_text(json.dumps({"instance": "road-j", "zone": "z"}))
    fake = FakeCompute(state="TERMINATED")
    c = checker.Checker(job.Store(str(bucket)), fake, tmp_path / "home", ["j"])
    for _ in range(4):
        fake._state = "TERMINATED"
        c.tick()
    assert fake.calls == [("start", "road-j"), ("start", "road-j")]


def test_an_unreachable_bucket_is_reported_and_nothing_is_touched(tmp_path, capsys):
    """A dropped connection once made a training VM read as missing. With the bucket
    unreadable the checker must say so and act on nothing."""
    fake = FakeCompute(state="TERMINATED")
    c = checker.Checker(job.Store(str(tmp_path / "nowhere")), fake, tmp_path / "home", ["j"])
    c.tick()
    assert fake.calls == [] and "UNREACHABLE" in capsys.readouterr().out


# --- launch.py ------------------------------------------------------------------------------

def notebook_config(name, cell, edits, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    monkeypatch.setitem(sys.modules, "google.colab", types.ModuleType("google.colab"))
    nb = nbformat.read(ROOT / "notebooks" / name, as_version=4)
    src = next(c for c in nb.cells if c.id == cell).source
    for old, new in edits:
        assert old in src
        src = src.replace(old, new, 1)
    ns = {}
    exec(compile(src, cell, "exec"), ns)
    return ns


@pytest.mark.parametrize("size", ["4b", "8b"])
def test_cloud_training_runs_the_notebooks_flags(size, monkeypatch):
    nb = notebook_config("02_qwenvl.ipynb", "14fb6199",
                         [("MODE = 'combo'\n", "MODE = 'winner'\n"),
                          ("SIZE = '4b'", f"SIZE = '{size}'")], monkeypatch)
    run = launch.run_name(size)
    steps = {s["name"]: s for s in launch.job_specs()[run]["steps"]}
    assert steps["train"]["cmd"] == ["python", "-m", "road.qwenvl", "--run", nb["RUN"],
                                     "--fold", "all", *nb["EXTRA"]]
    assert steps["preflight"]["cmd"][5:] == ["--smoke", *nb["EXTRA"]]
    assert steps["stackmix"]["cmd"][3:] == nb["PRE"][0][1]


def test_cloud_ensembles_are_the_notebooks_entries(monkeypatch):
    nb = notebook_config("03_ensemble.ipynb", "ba95c36b", [], monkeypatch)
    specs = launch.job_specs()
    steps = {s["name"]: s["cmd"] for spec in specs.values() for s in spec["steps"]
             if s["name"].startswith("ens_")}
    assert set(steps) == {out for out, _, _ in nb["JOBS"]}
    for out, judges, sources in nb["JOBS"]:
        cmd = steps[out]
        i, j, k = cmd.index("--runs"), cmd.index("--candidates-from"), cmd.index("--out")
        assert cmd[i + 1:j] == judges and cmd[j + 1:k] == sources, out
        assert cmd[cmd.index("--beams") + 1] == str(nb["BEAMS"])
        assert set(nb["REUSE"]) <= set(cmd[cmd.index("--reuse") + 1:])


def test_every_ensemble_reuses_the_entries_before_it():
    four = launch.job_specs()["qwen4b_all_combo"]["steps"]
    ens = [s for s in four if s["name"].startswith("ens_")]
    for i, s in enumerate(ens):
        reused = s["cmd"][s["cmd"].index("--reuse") + 1:]
        assert all(e["name"] in reused for e in ens[:i])


def test_the_rehearsal_has_the_production_shape():
    specs = launch.rehearsal_specs()
    a, b = specs["rehearse_a"]["steps"], specs["rehearse_b"]["steps"]
    train = next(s for s in a if s["name"] == "train")["cmd"]
    assert all(f in train for f in launch.WINNER_FLAGS) and "tiny" in train
    assert any("wait_for" in s for s in b)
    assert "--reuse" in b[-1]["cmd"]


class FakeGcloud:
    def __init__(self, errors):
        self.errors, self.created = list(errors), []

    def __call__(self, *args, check=True):
        import subprocess
        if args[:3] == ("compute", "instances", "create"):
            zone = next(a for a in args if a.startswith("--zone=")).split("=")[1]
            err = self.errors.pop(0) if self.errors else ""
            if not err:
                self.created.append(zone)
            return subprocess.CompletedProcess(args, 1 if err else 0, "", err)
        return subprocess.CompletedProcess(args, 0, "", "")


def test_create_moves_on_from_a_full_zone_but_not_from_the_global_quota():
    gc = FakeGcloud(["ZONE_RESOURCE_POOL_EXHAUSTED in us-central1-a", ""])
    zone = launch.create(gc, "j", "a2-highgpu-1g", ["us-central1-a", "us-central1-b"],
                         cpu=False, idle=45, max_hours=10)
    assert zone == "us-central1-b" and gc.created == ["us-central1-b"]

    gc = FakeGcloud(["Quota 'GPUS_ALL_REGIONS' exceeded. Limit: 0.0 globally."])
    with pytest.raises(SystemExit, match="GPUS_ALL_REGIONS"):
        launch.create(gc, "j", "a2-highgpu-1g", ["us-central1-a", "us-central1-b"],
                      cpu=False, idle=45, max_hours=10)


def test_the_code_tarball_carries_code_and_never_data(tmp_path):
    files = launch.code_tarball(tmp_path / "repo.tar.gz")
    with tarfile.open(tmp_path / "repo.tar.gz") as tar:
        names = set(tar.getnames())
    assert {"src/road/qwenvl.py", "cloud/gcp/job.py", "cloud/gcp/startup.sh",
            "requirements-colab-test.txt"} <= names
    assert not any(n.startswith(("data/", "runs/")) or n.endswith((".zip", ".jpg"))
                   for n in names), [n for n in names if n.startswith(("data/", "runs/"))]
    assert len(files) == len(names)


def test_the_startup_script_has_unix_line_endings():
    assert b"\r" not in (GCP / "startup.sh").read_bytes()


def test_the_chain_starts_the_next_job_only_once_the_last_is_home(tmp_path):
    """One GPU of quota: the 8B must not be launched while the 4B still holds it."""
    bucket = finished_bucket(tmp_path)
    launched = []
    c = checker.Checker(job.Store(str(bucket)), FakeCompute(), tmp_path / "home", ["j"])

    def launch_next(name):
        launched.append((name, list(c.home)))
        # the new job finishes at once, so the loop can end
        (bucket / "status" / name).mkdir(parents=True)
        for f in ("DONE", "status.json"):
            (bucket / "status" / name / f).write_text(
                json.dumps({"state": "done", "done": [], "steps": []}))
        (bucket / "status" / name / "manifest.json").write_text("{}")   # no files
        (bucket / "jobs" / f"{name}.json").write_text(json.dumps({"outputs": []}))

    assert c.run(every=0, once=False, chain=["k"], launch_next=launch_next) == 0
    assert launched == [("k", ["j"])] and c.home == {"j", "k"}


def test_on_an_l4_only_the_batches_change():
    """Training as on the A100 (the 8B at notebook 02's half batch); decoding in smaller
    batches, because 8 lines x 8 beams ran an L4 out of memory; scoring likewise."""
    l4, a100 = launch.job_specs("l4"), launch.job_specs("a100")
    train = lambda specs, run: next(s for s in specs[run]["steps"] if s["name"] == "train")["cmd"]
    assert train(l4, "qwen4b_all_combo") == train(a100, "qwen4b_all_combo") + \
        ["--decode-batch-size", "4"]
    assert train(l4, "qwen8b_all_combo") == train(a100, "qwen8b_all_combo") + \
        ["--batch-size", "2", "--grad-accum", "8", "--decode-batch-size", "1"]
    for spec in l4.values():
        for step in spec["steps"]:
            if step["name"].startswith("ens_"):
                cmd = step["cmd"]
                assert cmd[cmd.index("--beams") + 2:][:2] == ["--batch-size", "2"]


def test_the_finish_job_decodes_the_saved_adapter_then_builds_the_entries():
    """The recovery after the L4's decode ran out of memory: no retraining."""
    (name, spec), = launch.finish_specs("l4").items()
    assert name == "qwen4b_all_combo_finish"
    assert [s["name"] for s in spec["steps"]] == ["data", "decode", "ens_combo",
                                                  "ens_combo_3j", "ens_combo_swap"]
    decode = spec["steps"][1]["cmd"]
    assert decode[:5] == ["python", "-m", "road.qwenvl", "--run", "qwen4b_all_combo"]
    assert {"--predict-only", "--decode-batch-size"} <= set(decode)
    assert decode[decode.index("--beams") + 1] == "8"
    assert decode[decode.index("--n-best") + 1] == "5"
    assert "runs/qwen4b_all_combo" in spec["pull"]
    full = launch.job_specs("l4")["qwen4b_all_combo"]["steps"]
    assert spec["steps"][2:] == [s for s in full if s["name"].startswith("ens_")]


def test_the_8b_can_wait_on_the_recovery_job():
    eight = launch.job_specs("l4", wait_on="qwen4b_all_combo_finish")["qwen8b_all_combo"]
    wait = next(s for s in eight["steps"] if "wait_for" in s)
    assert wait["wait_for"] == "status/qwen4b_all_combo_finish/DONE"
    assert wait["abort_if"] == "status/qwen4b_all_combo_finish/FAILED"


def test_the_chain_stops_after_a_failure(tmp_path):
    bucket = finished_bucket(tmp_path, state="FAILED")
    launched = []
    c = checker.Checker(job.Store(str(bucket)), FakeCompute(), tmp_path / "home", ["j"])
    assert c.run(every=0, once=False, chain=["k"], launch_next=launched.append) == 1
    assert launched == [] and c.failed == {"j"}


def test_a_console_that_cannot_print_a_progress_bar_does_not_end_the_job(tmp_path):
    """tqdm draws with block characters; a cp1252 console once killed a rehearsal here."""
    import os
    import subprocess
    bucket = tmp_path / "bucket"
    bucket.mkdir()
    spec = tmp_path / "j.json"
    spec.write_text(json.dumps({"job": "j", "outputs": [], "steps": [
        {"name": "bar", "cmd": py("print('Writing shards: \u2588\u2588\u258c 50%')")}]}))
    r = subprocess.run([PY, str(GCP / "job.py"), "--spec", str(spec), "--store", str(bucket),
                        "--repo", str(ROOT), "--runs", str(tmp_path / "runs")],
                       env={**os.environ, "PYTHONIOENCODING": "cp1252"},
                       capture_output=True)
    assert r.returncode == 0, r.stderr.decode(errors="replace")[-800:]
    assert (bucket / "status/j/DONE").exists()
    assert "█" in (tmp_path / "j.log").read_text(encoding="utf-8")   # the log keeps it


def test_gpu_vms_terminate_on_maintenance_and_cpu_vms_migrate():
    """GCE requires TERMINATE for a GPU VM and refuses it for an e2 VM -- the GCP
    rehearsal's first finding."""
    seen = {}

    class Record(FakeGcloud):
        def __call__(self, *args, check=True):
            if args[:3] == ("compute", "instances", "create"):
                seen[args[3]] = [a for a in args if a.startswith("--maintenance-policy")]
            return super().__call__(*args, check=check)

    launch.create(Record([]), "gpu_job", "g2-standard-12", ["z"], cpu=False, idle=45, max_hours=16)
    launch.create(Record([]), "cpu_job", "e2-standard-4", ["z"], cpu=True, idle=10, max_hours=3)
    assert seen == {"road-gpu-job": ["--maintenance-policy=TERMINATE"],
                    "road-cpu-job": ["--maintenance-policy=MIGRATE"]}


@pytest.mark.parametrize("state, replaced", [("TERMINATED", True), ("RUNNING", False)])
def test_a_relaunch_replaces_a_stopped_vm_but_never_a_running_one(state, replaced):
    import subprocess
    calls = []

    class Existing(FakeGcloud):
        def __call__(self, *args, check=True):
            calls.append(args[:3])
            if args[:2] == ("storage", "cat"):
                return subprocess.CompletedProcess(args, 0, json.dumps(
                    {"instance": "road-j", "zone": "z"}), "")
            if args[:3] == ("compute", "instances", "describe"):
                return subprocess.CompletedProcess(args, 0, state + "\n", "")
            return super().__call__(*args, check=check)

    if replaced:
        launch.create(Existing([]), "j", "g2-standard-12", ["z"], cpu=False, idle=45,
                      max_hours=16)
        assert calls.index(("compute", "instances", "delete")) < \
            calls.index(("compute", "instances", "create"))
    else:
        with pytest.raises(SystemExit, match="not replacing a live VM"):
            launch.create(Existing([]), "j", "g2-standard-12", ["z"], cpu=False, idle=45,
                          max_hours=16)
        assert ("compute", "instances", "delete") not in calls


def test_no_heartbeat_lands_after_the_final_status(tmp_path):
    """A late beat saying 'running' must never overwrite 'done'."""
    j, bucket, _ = make_job(tmp_path, [{"name": "s", "cmd": py(WRITE_OUT)}])
    j.heartbeat = 0.001
    assert j.run() == 0
    assert not j.beat.is_alive()
    import time
    time.sleep(0.2)
    assert json.loads((bucket / "status/j/status.json").read_text())["state"] == "done"


def test_a_restarted_checker_neither_relaunches_nor_redeletes(tmp_path):
    """The checker outlives no session cap, so it gets restarted: by then the 4B's VM is
    gone and the 8B is running. Neither may be touched twice."""
    bucket = finished_bucket(tmp_path)
    (bucket / "vms/k.json").write_text(json.dumps({"instance": "road-k", "zone": "z"}))

    class Mixed(FakeCompute):
        def state(self, name, zone):
            return "GONE" if name == "road-j" else "RUNNING"

    fake, launched = Mixed(), []
    c = checker.Checker(job.Store(str(bucket)), fake, tmp_path / "home", ["j"])
    c.tick()                                  # j: home again, its VM already gone
    assert c.home == {"j"} and fake.calls == []
    assert c.started("k") and launched == []  # k's VM is live: not launched again


def test_a_full_region_moves_on_but_a_global_limit_stops():
    """The 8B must skip the region whose one L4 the 4B holds; the GPU-wide limit must not
    be retried zone by zone."""
    region = "Quota 'NVIDIA_L4_GPUS' exceeded.  Limit: 1.0 in region us-central1."
    gc = FakeGcloud([region, ""])
    assert launch.create(gc, "j", "g2-standard-12", ["us-central1-a", "us-east1-b"],
                         cpu=False, idle=45, max_hours=20) == "us-east1-b"
    other = "Quota 'SOME_NEW_GPU_METRIC' exceeded.  Limit: 0.0 in region us-central1."
    assert launch.create(FakeGcloud([other, ""]), "j", "g2-standard-12",
                         ["us-central1-a", "us-east1-b"], cpu=False, idle=45,
                         max_hours=20) == "us-east1-b"
    with pytest.raises(SystemExit, match="globally"):
        launch.create(FakeGcloud(["Quota 'GPUS_ALL_REGIONS' exceeded.  Limit: 1.0 globally."]),
                      "j", "g2-standard-12", ["us-central1-a", "us-east1-b"], cpu=False,
                      idle=45, max_hours=20)


def test_no_wait_outlasts_thirty_minutes_whatever_the_spec_says(tmp_path, monkeypatch):
    """A VM once waited six hours for an upload that had failed: an idle VM still bills."""
    clock = [0.0]
    monkeypatch.setattr(job.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(job.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    j, bucket, _ = make_job(tmp_path, [])
    assert j.wait({"wait_for": "never/there", "timeout_minutes": 300, "poll_seconds": 60}) is False
    assert 30 * 60 <= clock[0] <= 31 * 60
    clock[0] = 0.0
    assert j.wait({"wait_for": "never/there", "poll_seconds": 60}) is False   # the default
    assert 15 * 60 <= clock[0] <= 16 * 60
