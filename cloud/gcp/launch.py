"""Stage inputs in the bucket and start the VMs that run the winner jobs.

    python cloud/gcp/launch.py stage                 # code, CSVs, ensemble inputs, job specs
    python cloud/gcp/launch.py create 4b             # one L4 VM (the quota GCP granted)
    python cloud/gcp/launch.py create 4b 8b --gpu a100   # both at once, given A100 quota
    python cloud/gcp/launch.py rehearse              # two CPU VMs, smoke-sized, ~$0.10
    ... --dry-run                                    # print every gcloud command, run none

Nothing here deletes anything; checker.py deletes a VM only after its results are
downloaded and verified. A VM carries three cost guards of its own: GCE stops it at
--max-run-duration, the startup script powers it off after the job, and a VM whose
job is already DONE or FAILED never reruns it.

The jobs mirror notebook 02's MODE='winner' and notebook 03's entries exactly -- a test
executes the notebooks' config cells and compares the flags -- so what runs here is
what was tested on fold 0 and what Colab would have run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = os.environ.get("ROAD_GCP_PROJECT", "your-gcp-project")   # placeholder: set your own
BUCKET = f"{PROJECT}-road-htr"
DRIVE_RUNS = Path(os.environ.get("ROAD_RUNS", "runs"))      # where finished runs are copied
VM_ROOT = "/opt/road"                 # startup.sh's layout on the VM
CSV_SRC = f"{VM_ROOT}/csv"

# --- the recipe: notebook 02, MODE='winner', WINNER='combo' ----------------------------
SYNTH = ["--synth", "data/synth/stackmix.csv"]
WINNER_FLAGS = ["--jitter", "0.5", "--aug-heavy", *SYNTH, "--beams", "8", "--n-best", "5"]
MODELS = {"4b": [], "8b": ["--model", "Qwen/Qwen3-VL-8B-Instruct"]}
STACKMIX_N = "2500"

# --- the ensemble: notebook 03 -------------------------------------------------------------
BEAMS = "3"
POOL = ["trocr_all", "qwen4b_combined", "trocr_f0", "qwen4b_f0", "qwen4b_f0_v2",
        "qwen4b_f0_tall"]
OLD_JUDGES = ["qwen4b_all", "qwen8b_all"]
REUSE = ["ens_nbest", "ens_final", "ens_trocr_judge"]

# --- machines ------------------------------------------------------------------------------
# (machine type, zones in the order tried, extra run flags per size, scoring flags)
GPUS = {
    "a100": ("a2-highgpu-1g", ["us-central1-a", "us-central1-b", "us-central1-c",
                               "us-central1-f"], {"4b": [], "8b": []}, []),
    # 22 GB usable. Training fits (the 4B peaked at 13.9 GiB), but decoding 8 lines x 8
    # beams ran out of memory on 2026-10-02 after all three epochs, so lines decode in
    # smaller batches; the 8B trains at batch 2 x 8 accumulated (the same 16 per step:
    # at batch 4 its preflight peaked at 20.1 GiB) and the ensembles score in small batches.
    "l4": ("g2-standard-12", ["us-central1-a", "us-central1-b", "us-central1-c",
                              "us-east1-b", "us-east1-c", "us-east1-d", "us-east4-a",
                              "us-east4-c", "us-west1-a", "us-west1-b"],
           {"4b": ["--decode-batch-size", "4"],
            "8b": ["--batch-size", "2", "--grad-accum", "8", "--decode-batch-size", "1"]},
           ["--batch-size", "2"]),
}
# GCE stops a VM at this age whatever it is doing -- the backstop against a hung job.
# Set well past the expected run (A100 fold-0 timings x ~3 for an L4), because a stop
# mid-training restarts that step from scratch.
MAX_HOURS = {"a100": {"4b": 8, "8b": 10}, "l4": {"4b": 16, "8b": 28}}
IMAGE = ("deeplearning-platform-release", "common-cu129-ubuntu-2204-nvidia-580")
CAPACITY_ERRORS = ("ZONE_RESOURCE_POOL_EXHAUSTED", "does not have enough resources",
                   "resource pool exhausted", "currently unavailable")
# a full regional quota is worth another region; the global GPU quota is not
REGIONAL_QUOTAS = ("Quota 'NVIDIA_L4_GPUS' exceeded", "Quota 'NVIDIA_A100_GPUS' exceeded",
                   "Quota 'A2_CPUS' exceeded", "Quota 'G2_CPUS' exceeded")
REGIONAL_QUOTA_RE = re.compile(r"Quota '[A-Z0-9_]+' exceeded\.\s+Limit: [\d.]+ in region [a-z0-9-]+")


def run_name(size: str) -> str:
    return f"qwen{size}_all_combo"


def instance_name(job: str) -> str:
    return "road-" + job.replace("_", "-")


def qwen(*args: str) -> list[str]:
    return ["python", "-m", "road.qwenvl", *args]


def ensemble(out: str, judges: list[str], sources: list[str], reuse: list[str],
             extra: list[str] = ()) -> dict:
    cmd = ["python", "-m", "road.ensemble", "--runs", *judges, "--candidates-from", *sources,
           "--out", out, "--beams", BEAMS, *extra]
    if reuse:
        cmd += ["--reuse", *reuse]
    return {"name": out, "cmd": cmd}


def training_steps(run: str, size: str, gpu: str, smoke: bool = False) -> list[dict]:
    """Notebook 02's preflight and train cells, as commands. smoke=True is the CPU
    rehearsal: the same flags on the tiny model, which is what the notebook's own local
    test does."""
    fit = [*WINNER_FLAGS, *MODELS[size], *GPUS[gpu][2][size]]
    tiny = ["--model", "tiny", "--workers", "0"] if smoke else []
    return [
        {"name": "data", "cmd": ["python", "-m", "road.setup_data", "--csv-src", CSV_SRC]},
        {"name": "stackmix", "cmd": ["python", "-m", "road.stackmix", "--n",
                                     "24" if smoke else STACKMIX_N]},
        {"name": "preflight", "cmd": qwen("--run", f"preflight_{run}", "--smoke", *fit, *tiny)},
        {"name": "train", "cmd": qwen("--run", run, "--fold", "all", *fit,
                                      *(["--smoke", *tiny, "--keep-weights"] if smoke else []))},
    ]


ENTRIES = ["ens_combo", "ens_combo_3j", "ens_combo_swap"]
INPUTS = [f"runs/{r}" for r in POOL + OLD_JUDGES + REUSE]


def four_entries(gpu: str) -> list[dict]:
    """Notebook 03's three entries for the 4B, each reusing the scores before it."""
    w4, score = run_name("4b"), GPUS[gpu][3]
    pool = POOL + [w4]
    return [
        ensemble("ens_combo", OLD_JUDGES, pool, REUSE, score),
        ensemble("ens_combo_3j", OLD_JUDGES + [w4], pool, REUSE + ["ens_combo"], score),
        ensemble("ens_combo_swap", [w4, "qwen8b_all"], pool + ["qwen4b_all"],
                 REUSE + ["ens_combo", "ens_combo_3j"], score),
    ]


def job_specs(gpu: str = "a100", wait_on: str | None = None) -> dict[str, dict]:
    """The two production jobs. The 4B also builds three entries; the 8B the fourth,
    once the job named by wait_on (the 4B by default) is DONE."""
    w4, w8 = run_name("4b"), run_name("8b")
    pool = POOL + [w4]
    entries = ENTRIES
    four = training_steps(w4, "4b", gpu) + four_entries(gpu)
    wait_on = wait_on or w4
    eight = training_steps(w8, "8b", gpu) + [
        {"name": "wait-4b", "wait_for": f"status/{wait_on}/DONE",
         "abort_if": f"status/{wait_on}/FAILED", "timeout_minutes": 900,
         "pull": [f"runs/{w4}", *(f"runs/{e}" for e in entries)]},
        ensemble("ens_combo_pair", [w4, w8], pool + OLD_JUDGES, REUSE + entries, GPUS[gpu][3]),
    ]
    inputs = INPUTS
    return {
        w4: {"job": w4, "steps": four, "pull": inputs, "sync_minutes": 5,
             "outputs": [w4, f"preflight_{w4}", *entries]},
        w8: {"job": w8, "steps": eight, "pull": inputs, "sync_minutes": 5,
             "outputs": [w8, f"preflight_{w8}", "ens_combo_pair"]},
    }


def finish_specs(gpu: str = "l4") -> dict[str, dict]:
    """Recovery for a 4B whose training finished and whose decode did not: decode the
    saved adapter at this GPU's decode batch size, then build the three entries. No
    retraining -- the adapter is already in the bucket."""
    w4 = run_name("4b")
    job = f"{w4}_finish"
    steps = [
        {"name": "data", "cmd": ["python", "-m", "road.setup_data", "--csv-src", CSV_SRC]},
        {"name": "decode", "cmd": qwen("--run", w4, "--fold", "all", "--predict-only",
                                       "--beams", "8", "--n-best", "5", *GPUS[gpu][2]["4b"])},
        *four_entries(gpu),
    ]
    return {job: {"job": job, "steps": steps, "pull": INPUTS + [f"runs/{w4}"],
                  "sync_minutes": 5, "outputs": [w4, *ENTRIES]}}


def rehearsal_specs() -> dict[str, dict]:
    """The production shape at smoke size on two CPU VMs: the same training flags, a
    cross-VM wait and pull, and ensembles with --reuse -- everything but the GPU."""
    a, b = "rehearse_a", "rehearse_b"
    steps_a = training_steps(a, "4b", "a100", smoke=True) + [
        ensemble("rehearse_ens", [a, a], [a], [], ["--partial"])]
    steps_b = training_steps(b, "8b", "a100", smoke=True) + [
        {"name": "wait-a", "wait_for": f"status/{a}/DONE", "abort_if": f"status/{a}/FAILED",
         "timeout_minutes": 90, "poll_seconds": 30, "pull": [f"runs/{a}", "runs/rehearse_ens"]},
        ensemble("rehearse_pair", [a, b], [a, b], ["rehearse_ens"], ["--partial"]),
    ]
    return {
        a: {"job": a, "steps": steps_a, "sync_minutes": 2,
            "outputs": [a, f"preflight_{a}", "rehearse_ens"]},
        b: {"job": b, "steps": steps_b, "sync_minutes": 2,
            "outputs": [b, f"preflight_{b}", "rehearse_pair"]},
    }


# --- gcloud ----------------------------------------------------------------------------------
# on Windows gcloud is a .cmd wrapper: its full path runs without a shell, so arguments
# are passed exactly as written
GCLOUD = shutil.which("gcloud") or "gcloud"


class Gcloud:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

    def __call__(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        cmd = ["gcloud", *args]
        print("$ " + " ".join(shlex.quote(a) for a in cmd), flush=True)
        if self.dry_run:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        r = subprocess.run([GCLOUD, *args], capture_output=True, text=True)
        if check and r.returncode:
            raise SystemExit(f"gcloud failed ({r.returncode}):\n{r.stderr.strip()[-2000:]}")
        return r


def code_tarball(dest: Path) -> list[str]:
    """Every file git tracks or would track -- the working tree, uncommitted edits
    included, ignored files (data, runs) never."""
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.split("\n")
    files = [f for f in out if f and (ROOT / f).is_file()]
    with tarfile.open(dest, "w:gz") as tar:
        for f in files:
            tar.add(ROOT / f, arcname=f)
    return files


def stage_runs(dest: Path, drive: Path = DRIVE_RUNS) -> list[str]:
    """Only what the ensembles read: every pool run's predictions, the old judges'
    adapters, and the earlier ensembles' scores. Weights nobody loads stay home."""
    copied = []

    def take(run: str, names: list[str], dirs: list[str] = ()) -> None:
        for n in names:
            src = drive / run / n
            if src.exists():
                (dest / run).mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest / run / n)
                copied.append(f"{run}/{n}")
        for d in dirs:
            shutil.copytree(drive / run / d, dest / run / d, dirs_exist_ok=True)
            copied.append(f"{run}/{d}/")

    for run in POOL:
        take(run, [f"{run}.csv", "nbest.csv"])
    for run in OLD_JUDGES:
        take(run, [f"{run}.csv", "nbest.csv", "config.json"], ["adapter"])
    for ens in REUSE:
        take(ens, ["scores.csv"])
    missing = [r for r in POOL + OLD_JUDGES if not (dest / r / f"{r}.csv").exists()]
    if missing:
        raise SystemExit(f"predictions missing on Drive for {missing}")
    return copied


def stage(gc: Gcloud, rehearsal: bool) -> None:
    store = f"gs://{BUCKET}"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        files = code_tarball(tmp / "repo.tar.gz")
        print(f"code: {len(files)} files", flush=True)
        gc("storage", "cp", str(tmp / "repo.tar.gz"), f"{store}/code/repo.tar.gz")
        gc("storage", "cp", *(str(ROOT / "data" / n) for n in
                              ("Train.csv", "Test.csv", "SampleSubmission.csv")), f"{store}/csv/")
        specs = rehearsal_specs() if rehearsal else job_specs()
        for name, spec in specs.items():
            (tmp / f"{name}.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
            gc("storage", "cp", str(tmp / f"{name}.json"), f"{store}/jobs/{name}.json")
        if not rehearsal:
            copied = stage_runs(tmp / "runs")
            print(f"ensemble inputs: {len(copied)} files/dirs", flush=True)
            gc("storage", "rsync", "--recursive", str(tmp / "runs"), f"{store}/runs")


def create(gc: Gcloud, job: str, machine: str, zones: list[str], cpu: bool,
           idle: int, max_hours: int) -> str:
    """Start one VM, trying zones in order until one has capacity. Returns the zone."""
    store = f"gs://{BUCKET}"
    script = ROOT / "cloud/gcp/startup.sh"
    if b"\r" in script.read_bytes():   # bash on the VM would choke on CRLF endings
        raise SystemExit(f"{script} has CRLF line endings -- see .gitattributes")
    # A relaunch replaces the stopped VM a FAILED job leaves behind -- the checker has
    # already brought its logs and partial results home -- but never a running one.
    old = gc("storage", "cat", f"{store}/vms/{job}.json", check=False)
    if old.returncode == 0 and old.stdout.strip():
        prev = json.loads(old.stdout)
        state = gc("compute", "instances", "describe", prev["instance"], f"--project={PROJECT}",
                   f"--zone={prev['zone']}", "--format=value(status)", check=False)
        if state.returncode == 0 and state.stdout.strip():
            if state.stdout.strip() in ("RUNNING", "STAGING", "PROVISIONING", "STOPPING"):
                raise SystemExit(f"{prev['instance']} is {state.stdout.strip()} -- not replacing "
                                 "a live VM")
            print(f"replacing stopped {prev['instance']} ({state.stdout.strip()})", flush=True)
            gc("compute", "instances", "delete", prev["instance"], f"--project={PROJECT}",
               f"--zone={prev['zone']}", "--quiet")
    # a relaunch must not inherit an old run's DONE/FAILED or its finished steps
    gc("storage", "rm", "--recursive", f"{store}/status/{job}", check=False)
    meta = f"road-bucket={BUCKET},road-job={job},road-idle-minutes={idle},road-cpu={int(cpu)}"
    # a GPU VM cannot live-migrate, so GCE requires TERMINATE for it; an e2 VM refuses
    # TERMINATE unless preemptible, so it keeps the default (MIGRATE)
    host = ["--maintenance-policy=MIGRATE"] if cpu else \
        ["--maintenance-policy=TERMINATE", "--restart-on-failure"]
    if not cpu:
        meta += ",install-nvidia-driver=True"
    tried = []
    for zone in zones:
        r = gc("compute", "instances", "create", instance_name(job), f"--project={PROJECT}",
               f"--zone={zone}", f"--machine-type={machine}",
               f"--image-project={IMAGE[0]}", f"--image-family={IMAGE[1]}",
               "--boot-disk-size=200GB", "--boot-disk-type=pd-balanced",
               *host, "--scopes=storage-rw,logging-write",
               f"--max-run-duration={max_hours}h", "--instance-termination-action=STOP",
               f"--metadata={meta}",
               f"--metadata-from-file=startup-script={script}",
               f"--labels=purpose=road-htr,job={job.replace('_', '-')}", check=False)
        if r.returncode == 0:
            vm = {"instance": instance_name(job), "zone": zone, "project": PROJECT,
                  "machine": machine}
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                json.dump(vm, f)
            gc("storage", "cp", f.name, f"{store}/vms/{job}.json")
            Path(f.name).unlink()
            print(f"{instance_name(job)} started in {zone}", flush=True)
            return zone
        reason = next((e for e in CAPACITY_ERRORS + REGIONAL_QUOTAS if e in r.stderr), None)
        if reason is None and (m := REGIONAL_QUOTA_RE.search(r.stderr)):
            reason = m[0]   # any per-region quota; a global one says "globally" instead
        if reason is None:   # anything else (global quota, bad flag) will not improve
            raise SystemExit(f"could not create {instance_name(job)} in {zone}:\n"
                             f"{r.stderr.strip()[-2000:]}")
        tried.append(f"{zone}: {reason}")
        print(f"  {zone}: {reason} -- trying the next zone", flush=True)
    raise SystemExit(f"no zone could take {machine}:\n  " + "\n  ".join(tried))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["stage", "create", "rehearse", "finish"],
                    help="finish: decode the saved 4B and build its three entries")
    ap.add_argument("sizes", nargs="*", help="4b and/or 8b (create only; default 4b)")
    ap.add_argument("--gpu", choices=sorted(GPUS), default="l4",
                    help="l4 by default: on 2026-10-02 GCP granted 1 GPU and denied A100s")
    ap.add_argument("--zone", default=None, help="try only this zone")
    ap.add_argument("--idle-minutes", type=int, default=2,
                    help="minutes after its job ends before a VM powers itself off")
    ap.add_argument("--max-hours", type=int, default=None,
                    help="GCE stops the VM after this; default per GPU and size")
    ap.add_argument("--wait-on", default=None,
                    help="job whose DONE the 8B waits for before its ensemble "
                         "(default: the 4B; after a recovery: qwen4b_all_combo_finish)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    # checked here, not by argparse: a list default on a positional with choices is
    # itself rejected as an invalid choice
    args.sizes = args.sizes or ["4b"]
    if bad := [s for s in args.sizes if s not in MODELS]:
        ap.error(f"unknown size(s) {bad}: pick from {sorted(MODELS)}")
    gc = Gcloud(args.dry_run)

    if args.action == "stage":
        stage(gc, rehearsal=False)
    elif args.action == "finish":
        upload(gc, finish_specs(args.gpu), code=True)
        machine, zones = GPUS[args.gpu][:2]
        for job in finish_specs(args.gpu):
            create(gc, job, machine, [args.zone] if args.zone else zones, cpu=False,
                   idle=args.idle_minutes, max_hours=args.max_hours or 8)
    elif args.action == "rehearse":
        stage(gc, rehearsal=True)
        for job in rehearsal_specs():
            create(gc, job, "e2-standard-4", [args.zone or "us-central1-a"], cpu=True,
                   idle=2, max_hours=3)
    else:
        start(gc, args.sizes, args.gpu, args.zone, args.idle_minutes, args.max_hours,
              args.wait_on)
    return 0


def upload(gc: Gcloud, specs: dict[str, dict], code: bool = False) -> None:
    """Job specs to the bucket -- and, with code=True, the working tree as the VM's code."""
    with tempfile.TemporaryDirectory() as tmp:
        if code:
            files = code_tarball(Path(tmp) / "repo.tar.gz")
            print(f"code: {len(files)} files", flush=True)
            gc("storage", "cp", str(Path(tmp) / "repo.tar.gz"), f"gs://{BUCKET}/code/repo.tar.gz")
        for name, spec in specs.items():
            p = Path(tmp) / f"{name}.json"
            p.write_text(json.dumps(spec, indent=1), encoding="utf-8")
            gc("storage", "cp", str(p), f"gs://{BUCKET}/jobs/{name}.json")


def start(gc: Gcloud, sizes: list[str], gpu: str, zone: str | None = None, idle: int = 2,
          max_hours: int | None = None, wait_on: str | None = None) -> None:
    """Upload the job specs for this GPU (the batch flags depend on it) and create one
    VM per size. checker.py --chain calls this to start the 8B once the 4B is home."""
    machine, zones = GPUS[gpu][:2]
    upload(gc, job_specs(gpu, wait_on))
    for size in sizes:
        create(gc, run_name(size), machine, [zone] if zone else zones, cpu=False, idle=idle,
               max_hours=max_hours or MAX_HOURS[gpu][size])


if __name__ == "__main__":
    sys.exit(main())
