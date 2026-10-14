"""Watch the cloud jobs; bring their results home; only then delete their VMs.

    python cloud/gcp/checker.py                       # the two winner jobs, every 5 min
    python cloud/gcp/checker.py rehearse_a rehearse_b --every 1 --dest DIR
    python cloud/gcp/checker.py --once                # one status line per job, then exit
    python cloud/gcp/checker.py qwen4b_all_combo --chain 8b   # one GPU: 4B, then the 8B

Every pass prints, per job: the VM's state, the step it is on, epoch and batch, how far
through training it is, an ETA, and how long since its last heartbeat. A heartbeat
older than STALE minutes on a running VM is flagged; a VM that stopped before its job
finished (maintenance, the max-run-duration guard) is restarted, at most twice, and its
job resumes at the step it was on.

When a job ends -- DONE or FAILED -- the checker:
  1. downloads every output run, and the job's log and status, to --dest;
  2. checks every downloaded file's size and md5 against the bucket's own listing, and
     for a DONE job also against the manifest the VM wrote of its own files -- so a
     file the VM had but the bucket lacks is caught, not just a bad download;
  3. writes a receipt beside the results;
  4. only if every file matched: deletes the VM (DONE), or stops it (FAILED, so its
     disk can be inspected). If anything did not match, the VM is left running and
     the mismatch is reported -- nothing is deleted on a doubt.
It exits once every job has been brought home.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from job import Store, md5  # noqa: E402

PROJECT = os.environ.get("ROAD_GCP_PROJECT", "your-gcp-project")   # placeholder: set your own
BUCKET = f"{PROJECT}-road-htr"
DEST = Path(os.environ.get("ROAD_RUNS", "runs"))            # where finished runs are copied
JOBS = ["qwen4b_all_combo", "qwen8b_all_combo"]
STALE = 15            # minutes without a heartbeat before a running job is flagged
MAX_RESTARTS = 2
GCLOUD = shutil.which("gcloud") or "gcloud"


def parse_time(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s)


def minutes_since(s: str | None) -> float | None:
    if not s:
        return None
    return (dt.datetime.now(dt.timezone.utc) - parse_time(s)).total_seconds() / 60


class Compute:
    """The VMs. Tests pass a fake with the same four methods."""

    def __init__(self, project: str):
        self.project = project

    def _gc(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([GCLOUD, "compute", "instances", *args,
                               f"--project={self.project}"], capture_output=True, text=True)

    def state(self, name: str, zone: str) -> str:
        r = self._gc("describe", name, f"--zone={zone}", "--format=value(status)")
        return r.stdout.strip() if r.returncode == 0 else "GONE"

    def delete(self, name: str, zone: str) -> bool:
        return self._gc("delete", name, f"--zone={zone}", "--quiet").returncode == 0

    def stop(self, name: str, zone: str) -> bool:
        return self._gc("stop", name, f"--zone={zone}").returncode == 0

    def start(self, name: str, zone: str) -> bool:
        return self._gc("start", name, f"--zone={zone}").returncode == 0


class AwsCompute:
    """The same four methods for EC2, in GCE's vocabulary: a stopped instance reports
    TERMINATED (as GCE calls it) and a terminated one GONE."""

    STATES = {"pending": "STAGING", "running": "RUNNING", "stopping": "STOPPING",
              "stopped": "TERMINATED", "shutting-down": "STOPPING", "terminated": "GONE"}

    def __init__(self, region: str, aws: str = "aws"):
        self.region, self.aws = region, aws

    def _ec2(self, verb: str, instance: str, *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run([self.aws, "ec2", verb, "--region", self.region,
                               "--instance-ids", instance, *extra],
                              capture_output=True, text=True)

    def state(self, name: str, zone: str) -> str:
        r = self._ec2("describe-instances", name, "--query",
                      "Reservations[0].Instances[0].State.Name", "--output", "text")
        return self.STATES.get(r.stdout.strip(), "GONE") if r.returncode == 0 else "GONE"

    def delete(self, name: str, zone: str) -> bool:
        return self._ec2("terminate-instances", name).returncode == 0

    def stop(self, name: str, zone: str) -> bool:
        return self._ec2("stop-instances", name).returncode == 0

    def start(self, name: str, zone: str) -> bool:
        return self._ec2("start-instances", name).returncode == 0


def s3_listing(store: Store, prefix: str) -> dict[str, dict]:
    """S3's own listing. Its ETag is the md5 only for a single-part upload; the CLI
    uploads anything over 8 MB in parts, so those are checked by size here and by the
    VM's md5 manifest instead."""
    bucket, key = store.url(prefix)[5:].split("/", 1)
    r = subprocess.run([store.aws, "s3api", "list-objects-v2", "--bucket", bucket,
                        "--prefix", key + "/", "--output", "json"],
                       capture_output=True, text=True)
    if r.returncode or not r.stdout.strip():
        return {}
    out = {}
    for obj in json.loads(r.stdout).get("Contents", []):
        etag = obj.get("ETag", "").strip('"')
        out[obj["Key"]] = {"size": int(obj["Size"]), "md5": None if "-" in etag else etag}
    return out


def listing(store: Store, prefix: str) -> dict[str, dict]:
    """Every object under prefix with its size and md5 hex, from the bucket itself."""
    if store.s3:
        return s3_listing(store, prefix)
    if not store.gcs:
        root = Path(store.url(prefix))
        return {f"{prefix}/{p.relative_to(root).as_posix()}": {"size": p.stat().st_size,
                                                                "md5": md5(p)}
                for p in sorted(root.rglob("*")) if p.is_file()} if root.exists() else {}
    r = subprocess.run([GCLOUD, "storage", "ls", "--recursive", "--json",
                        store.url(prefix) + "/**"], capture_output=True, text=True)
    if r.returncode:
        return {}
    out = {}
    for obj in json.loads(r.stdout or "[]"):
        meta = obj.get("metadata", obj)
        name = meta.get("name")
        if not name or name.endswith("/"):
            continue
        h = meta.get("md5Hash") or meta.get("md5_hash")
        out[name] = {"size": int(meta.get("size", 0)),
                     "md5": base64.b64decode(h).hex() if h else None}
    return out


def verify(local_root: Path, expected: dict[str, dict], strip: str) -> list[str]:
    """Files under local_root that are missing, or differ from expected (rel -> size, md5)."""
    bad = []
    for rel, want in expected.items():
        p = local_root / rel.removeprefix(strip)
        if not p.is_file():
            bad.append(f"missing  {rel}")
        elif p.stat().st_size != want["size"]:
            bad.append(f"size     {rel}: {p.stat().st_size} != {want['size']}")
        elif want.get("md5") and md5(p) != want["md5"]:
            bad.append(f"md5      {rel}")
    return bad


def line(job: str, vm_state: str, status: dict | None, done: bool, failed: bool) -> str:
    if status is None:
        tag = "DONE" if done else "FAILED" if failed else vm_state
        why = "before its first heartbeat" if (done or failed) else \
            "no heartbeat yet (installing, or not started)"
        return f"{job:<18} {tag:<10} {why}"
    s = status
    steps = s.get("steps", [])
    where = f"step {len(s.get('done', [])) + (0 if done else 1)}/{len(steps)} {s.get('step')}"
    bits = [where]
    if s.get("phase"):
        bits.append(s["phase"])
    if s.get("epochs"):
        bits.append(f"epoch {s['epoch']}/{s['epochs']}, batch {s['batch']}/{s['batches']}")
    if "train_fraction" in s:
        f = s["train_fraction"]
        bits.append(f"training {f:.0%}")
        started = minutes_since(s.get("step_started"))
        if s.get("step") == "train" and 0.02 < f < 1 and started:
            bits.append(f"ETA {started * (1 - f) / f:.0f} min")
    age = minutes_since(s.get("updated"))
    stale = " STALE" if age is not None and age > STALE and not (done or failed) else ""
    tag = "DONE" if done else "FAILED" if failed else vm_state
    beat = f"beat {age:.0f}m ago" if age is not None else "no heartbeat time"
    return f"{job:<18} {tag:<10} " + " | ".join(bits) + f" | {beat}{stale}"


class Checker:
    def __init__(self, store: Store, compute: Compute, dest: Path, jobs: list[str]):
        self.store, self.compute, self.dest, self.jobs = store, compute, dest, jobs
        self.restarts = {j: 0 for j in jobs}
        self.home: set[str] = set()
        self.failed: set[str] = set()

    def vm(self, job: str) -> dict | None:
        text = self.store.get_text(f"vms/{job}.json")
        return json.loads(text) if text else None

    def started(self, job: str) -> bool:
        """Its VM exists, or it has already reported -- either way, launched before."""
        vm = self.vm(job)
        live = vm is not None and self.compute.state(vm["instance"], vm["zone"]) != "GONE"
        return live or self.store.exists(f"status/{job}/status.json") or \
            self.store.exists(f"status/{job}/DONE")

    def bring_home(self, job: str, vm: dict | None, done: bool) -> bool:
        """Download, verify, receipt; then delete (DONE) or stop (FAILED). True if home."""
        status_rel = f"status/{job}"
        final = json.loads(self.store.get_text(f"{status_rel}/{'DONE' if done else 'FAILED'}"))
        spec = json.loads(self.store.get_text(f"jobs/{job}.json") or "{}")
        outputs = spec.get("outputs", [])
        side = self.dest / "_gcp" / job
        side.mkdir(parents=True, exist_ok=True)

        problems = []
        for run in outputs:
            remote = listing(self.store, f"runs/{run}")
            if not remote:
                continue           # e.g. an entry a failed job never reached
            self.store.sync_down(f"runs/{run}", self.dest / run)
            problems += verify(self.dest, remote, "runs/")
        self.store.sync_down(status_rel, side)
        problems += verify(side, listing(self.store, status_rel), status_rel + "/")
        if done:   # the VM's own account of what it had: catches a file the sync missed
            manifest = json.loads(self.store.get_text(f"{status_rel}/manifest.json") or "{}")
            problems += [f"VM had, bucket lacks: {p}"
                         for p in verify(self.dest, manifest, "runs/")]

        receipt = {"job": job, "state": "done" if done else "failed", "checked": dt.datetime.now(
            dt.timezone.utc).isoformat(timespec="seconds"), "dest": str(self.dest),
            "outputs": outputs, "problems": problems, "final": final}
        (side / "receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
        if problems:
            print(f"!! {job}: {len(problems)} file(s) did not verify -- VM left as it is:",
                  *problems[:20], sep="\n   ", flush=True)
            return False

        n = sum(1 for _ in self.dest.glob("*")) if self.dest.exists() else 0
        print(f"{job}: results verified in {self.dest} ({len(outputs)} runs; {n} entries "
              f"there), receipt {side / 'receipt.json'}", flush=True)
        if vm and self.compute.state(vm["instance"], vm["zone"]) == "GONE":
            print(f"{job}: VM {vm['instance']} already deleted", flush=True)
        elif vm:
            name, zone = vm["instance"], vm["zone"]
            if done:
                ok = self.compute.delete(name, zone)
                print(f"{job}: VM {name} {'deleted' if ok else 'NOT deleted -- delete it by hand'}",
                      flush=True)
            else:
                ok = self.compute.stop(name, zone)
                print(f"{job}: FAILED at {final.get('failed_step')}; VM {name} "
                      f"{'stopped (disk kept)' if ok else 'NOT stopped'}", flush=True)
        return True

    def tick(self) -> None:
        for job in self.jobs:
            if job in self.home:
                continue
            # Every launched job has its spec in the bucket. If even that cannot be read,
            # the bucket is unreachable (a dropped connection, an expired login): say
            # so and wait, rather than report a running VM as missing.
            if self.store.get_text(f"jobs/{job}.json") is None:
                print(f"{job:<18} UNREACHABLE  cannot read the bucket -- skipping this pass",
                      flush=True)
                continue
            vm = self.vm(job)
            vm_state = self.compute.state(vm["instance"], vm["zone"]) if vm else "NO-VM"
            raw = self.store.get_text(f"status/{job}/status.json")
            status = json.loads(raw) if raw else None
            done = self.store.exists(f"status/{job}/DONE")
            failed = not done and self.store.exists(f"status/{job}/FAILED")
            print(line(job, vm_state, status, done, failed), flush=True)
            if done or failed:
                if self.bring_home(job, vm, done):
                    self.home.add(job)
                    if failed:
                        self.failed.add(job)
            elif vm and vm_state in ("TERMINATED", "STOPPED", "SUSPENDED"):
                if self.restarts[job] < MAX_RESTARTS:
                    self.restarts[job] += 1
                    ok = self.compute.start(vm["instance"], vm["zone"])
                    print(f"{job}: VM had stopped before finishing -- restart "
                          f"{self.restarts[job]}/{MAX_RESTARTS} {'sent' if ok else 'FAILED'}",
                          flush=True)
                else:
                    print(f"!! {job}: VM stopped again after {MAX_RESTARTS} restarts -- "
                          "look at status/{job}/job.log; not restarting", flush=True)

    def run(self, every: float, once: bool, chain: list[str] = (),
            launch_next=None) -> int:
        """chain: jobs to start, in order, each once everything before it is home --
        how two jobs share a one-GPU quota without a person awake to start the second."""
        chain = list(chain)
        while True:
            print(f"--- {dt.datetime.now().strftime('%H:%M:%S')}", flush=True)
            self.tick()
            if len(self.home) == len(self.jobs) and chain and not once and self.failed:
                # the next job would inherit whatever broke this one (on 2026-10-02 a
                # decode out of memory nearly launched the 8B into the same wall)
                print(f"!! not starting {', '.join(chain)}: {', '.join(sorted(self.failed))} "
                      "FAILED -- fix and launch by hand", flush=True)
                return 1
            if len(self.home) == len(self.jobs) and chain and not once:
                nxt = chain.pop(0)
                if self.started(nxt):   # a restarted checker must not launch it twice
                    print(f"{nxt} already started -- watching it", flush=True)
                else:
                    print(f"{', '.join(self.jobs)} home -- starting {nxt}", flush=True)
                    launch_next(nxt)
                self.jobs.append(nxt)
                self.restarts[nxt] = 0
                continue
            elif once or len(self.home) == len(self.jobs):
                return 0
            time.sleep(every * 60)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs", nargs="*", default=JOBS)
    ap.add_argument("--store", default=f"gs://{BUCKET}")
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--every", type=float, default=5.0, help="minutes between passes")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--chain", nargs="*", default=[], choices=["4b", "8b"],
                    help="sizes to launch, in order, once the jobs before them are home")
    ap.add_argument("--gpu", default="l4", help="GPU for chained launches")
    ap.add_argument("--wait-on", default=None,
                    help="passed to the chained 8B: the job whose DONE its ensemble awaits")
    ap.add_argument("--cloud", choices=["gcp", "aws"], default="gcp",
                    help="aws: an s3:// --store, EC2 instances in --region (run where the "
                         "aws CLI is logged in)")
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args(argv)
    if args.cloud == "aws":
        if not args.store.startswith("s3://"):
            ap.error("--cloud aws needs an s3:// --store")
        if args.chain:
            ap.error("--chain launches GCP jobs; it is not available with --cloud aws")
        compute = AwsCompute(args.region)
    else:
        compute = Compute(args.project)
    checker = Checker(Store(args.store, GCLOUD), compute, args.dest, args.jobs)

    def launch_next(job: str) -> None:
        import launch
        launch.start(launch.Gcloud(), [job.removeprefix("qwen").split("_")[0]], args.gpu,
                     wait_on=args.wait_on)

    chain = [f"qwen{size}_all_combo" for size in args.chain]
    return checker.run(args.every, args.once, chain, launch_next)


if __name__ == "__main__":
    sys.exit(main())
