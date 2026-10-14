"""Run one job on a cloud VM so that the VM can vanish at any moment and lose nothing.

A job is a list of steps from a JSON spec in the bucket. Its state lives in the bucket,
not on the VM:

    status/<job>/status.json   heartbeat every minute: step, epoch, batch, last line
    status/<job>/job.log       everything every step printed, uploaded with each sync
    status/<job>/manifest.json size and md5 of every output file, written before DONE
    status/<job>/DONE          written only after the final sync has succeeded
    status/<job>/FAILED        the failing step, its exit code and the end of its log
    runs/<run>/...             each output run, synced every few minutes while it runs

Training writes its best epoch's adapter into the run folder as it goes, so the sync
carries it to the bucket mid-run: a VM lost at epoch 3 still leaves epoch 2 behind.

Steps:
    {"name": "train", "cmd": ["python", "-m", "road.qwenvl", ...]}
    {"name": "wait-4b", "wait_for": "status/x/DONE", "abort_if": "status/x/FAILED",
     "timeout_minutes": 360, "pull": ["runs/x"]}

A wait that times out or sees abort_if ends the job early but cleanly (DONE, with the
skipped steps listed): a later entry that depends on another VM must not cost this
VM's own results. On a reboot the steps already done are skipped.

    python cloud/gcp/job.py --spec job.json --store gs://bucket --repo DIR --runs DIR

Standard library only, so it runs before -- and regardless of -- the training stack.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

WAIT_DEFAULT_MINUTES = 15   # how long a wait step waits unless the spec says otherwise
WAIT_MAX_MINUTES = 30       # the most any wait may take: an idle VM still bills
EPOCHS = re.compile(r"(\d+) epochs x (\d+) batches")
BATCH = re.compile(r"step\s+(\d+)/(\d+)")
EPOCH_DONE = re.compile(r"^\s*epoch\s+(\d+)\s+loss")
DECODE = re.compile(r"(held-out fold|test: \d+ lines decoded|scored under|lines, \d+ candidates)")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class Store:
    """The bucket -- gs:// (Cloud Storage) or s3:// (S3) -- or a plain directory
    standing in for it in tests."""

    def __init__(self, root: str, gcloud: str = "gcloud", aws: str = "aws"):
        self.root = root.rstrip("/")
        self.gcs = self.root.startswith("gs://")
        self.s3 = self.root.startswith("s3://")
        self.gcloud, self.aws = gcloud, aws

    def _gs(self, *args: str) -> subprocess.CompletedProcess:
        if self.s3:
            return self._s3(*args)
        return subprocess.run([self.gcloud, "storage", *args], capture_output=True, text=True)

    def _s3(self, verb: str, *args: str) -> subprocess.CompletedProcess:
        """The gcloud storage verbs this class uses, as their aws s3 equivalents."""
        cmd = {"cp": ["s3", "cp", "--only-show-errors"],
               "rsync": ["s3", "sync", "--only-show-errors"]}[verb]
        args = tuple(a for a in args if a != "--recursive")   # s3 sync always recurses
        return subprocess.run([self.aws, *cmd, *args], capture_output=True, text=True)

    def url(self, rel: str) -> str:
        return f"{self.root}/{rel}"

    def exists(self, rel: str) -> bool:
        if self.s3:   # an exact key: `aws s3 ls` would also match DONE_OLD
            bucket, key = self.url(rel)[5:].split("/", 1)
            return subprocess.run([self.aws, "s3api", "head-object", "--bucket", bucket,
                                   "--key", key], capture_output=True, text=True).returncode == 0
        if self.gcs:
            return self._gs("ls", self.url(rel)).returncode == 0
        return Path(self.url(rel)).exists()

    def get_text(self, rel: str) -> str | None:
        if self.s3:
            r = subprocess.run([self.aws, "s3", "cp", "--only-show-errors", self.url(rel), "-"],
                               capture_output=True, text=True)
            return r.stdout if r.returncode == 0 else None
        if self.gcs:
            r = self._gs("cat", self.url(rel))
            return r.stdout if r.returncode == 0 else None
        p = Path(self.url(rel))
        return p.read_text(encoding="utf-8") if p.exists() else None

    def put_text(self, rel: str, text: str) -> None:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt",
                                         encoding="utf-8") as f:
            f.write(text)
        try:
            self.put_file(Path(f.name), rel)
        finally:
            os.unlink(f.name)

    def put_file(self, local: Path, rel: str) -> None:
        if self.gcs or self.s3:
            r = self._gs("cp", str(local), self.url(rel))
            if r.returncode:
                raise OSError(f"upload {local} -> {rel} failed: {r.stderr.strip()[-500:]}")
        else:
            dest = Path(self.url(rel))
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local, dest)

    def sync_up(self, local: Path, rel: str) -> None:
        """Copy a directory up; never deletes anything already in the bucket."""
        if self.gcs or self.s3:
            r = self._gs("rsync", "--recursive", str(local), self.url(rel))
            if r.returncode:
                raise OSError(f"sync {local} -> {rel} failed: {r.stderr.strip()[-500:]}")
        else:
            shutil.copytree(local, Path(self.url(rel)), dirs_exist_ok=True)

    def sync_down(self, rel: str, local: Path) -> None:
        local.mkdir(parents=True, exist_ok=True)
        if self.gcs or self.s3:
            r = self._gs("rsync", "--recursive", self.url(rel), str(local))
            if r.returncode:
                raise OSError(f"sync {rel} -> {local} failed: {r.stderr.strip()[-500:]}")
        else:
            shutil.copytree(Path(self.url(rel)), local, dirs_exist_ok=True)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def manifest(runs: Path, outputs: list[str]) -> dict[str, dict]:
    """Size and md5 of every file in the output runs, keyed by its path in the bucket."""
    files = {}
    for run in outputs:
        root = runs / run
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file():
                rel = "runs/" + p.relative_to(runs).as_posix()
                files[rel] = {"size": p.stat().st_size, "md5": md5(p)}
    return files


class Progress:
    """What the job is doing, read off the lines its steps print."""

    def __init__(self, job: str, steps: list[dict]):
        self.state = {"job": job, "state": "running", "started": now(), "updated": now(),
                      "steps": [s["name"] for s in steps], "done": [], "step": None,
                      "epochs": None, "epoch": 0, "batches": None, "batch": 0,
                      "phase": None, "last_line": ""}
        self.lock = threading.Lock()

    def begin(self, name: str) -> None:
        with self.lock:
            self.state.update(step=name, epochs=None, epoch=0, batches=None, batch=0,
                              phase=None, step_started=now())

    def finish(self, name: str) -> None:
        with self.lock:
            if name not in self.state["done"]:
                self.state["done"].append(name)

    def line(self, text: str) -> None:
        with self.lock:
            s = self.state
            if m := EPOCHS.search(text):
                s["epochs"], s["batches"], s["phase"] = int(m[1]), int(m[2]), "training"
            elif m := BATCH.search(text):
                s["batch"], s["phase"] = int(m[1]), "training"
            elif m := EPOCH_DONE.search(text):
                s["epoch"], s["batch"] = int(m[1]), 0
            elif DECODE.search(text):
                s["phase"] = "decoding" if "scored" not in text else "scoring"
            if text.strip():
                s["last_line"] = text.strip()[-300:]

    def snapshot(self) -> dict:
        with self.lock:
            s = dict(self.state, updated=now(), done=list(self.state["done"]))
        if s["epochs"] and s["batches"]:
            s["train_fraction"] = round(
                min(1.0, (s["epoch"] + s["batch"] / s["batches"]) / s["epochs"]), 4)
        return s


class Job:
    def __init__(self, spec: dict, store: Store, repo: Path, runs: Path, log: Path,
                 heartbeat: float = 60.0, sync_every: float | None = None):
        self.spec, self.store, self.repo, self.runs, self.log = spec, store, repo, runs, log
        self.name = spec["job"]
        self.outputs = spec.get("outputs", [])
        self.status_rel = f"status/{self.name}"
        self.heartbeat = heartbeat
        self.sync_every = sync_every if sync_every is not None else 60 * spec.get("sync_minutes", 5)
        self.progress = Progress(self.name, spec["steps"])
        self.sync_lock = threading.Lock()
        self.status_lock = threading.Lock()
        self.stop = threading.Event()
        self.beat: threading.Thread | None = None

    # -- the bucket ---------------------------------------------------------------
    def put_status(self) -> None:
        try:
            with self.status_lock:
                self.store.put_text(f"{self.status_rel}/status.json",
                                    json.dumps(self.progress.snapshot(), indent=1))
        except OSError as e:     # a missed heartbeat must never kill the job
            print(f"[job] heartbeat upload failed: {e}", flush=True)

    def halt(self) -> None:
        """Stop the heartbeat and wait for it, so a late beat saying 'running' can never
        land on top of the final status."""
        self.stop.set()
        if self.beat is not None and self.beat is not threading.current_thread():
            self.beat.join(timeout=300)

    def sync(self, attempts: int = 3) -> None:
        """Every output run and the log up to the bucket; retried, then raised."""
        with self.sync_lock:
            for i in range(attempts):
                try:
                    for run in self.outputs:
                        if (self.runs / run).is_dir():
                            self.store.sync_up(self.runs / run, f"runs/{run}")
                    if self.log.exists():
                        self.store.put_file(self.log, f"{self.status_rel}/job.log")
                    return
                except OSError as e:
                    print(f"[job] sync attempt {i + 1} failed: {e}", flush=True)
                    time.sleep(min(30, 5 * (i + 1)))
            raise OSError(f"sync failed {attempts} times")

    def _background(self) -> None:
        last_sync = time.monotonic()
        while not self.stop.wait(self.heartbeat):
            self.put_status()
            if time.monotonic() - last_sync >= self.sync_every:
                try:
                    self.sync(attempts=1)
                except OSError:
                    pass    # the next tick retries; the final sync must succeed
                last_sync = time.monotonic()

    # -- steps --------------------------------------------------------------------
    def env(self) -> dict:
        # expandable segments: the L4's decode OOM had 3.3 GB reserved but unallocated
        return {**os.environ, "PYTHONPATH": str(self.repo / "src"), "ROAD_RUNS": str(self.runs),
                "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    def run_cmd(self, step: dict) -> int:
        cmd = [sys.executable if a == "python" else a for a in step["cmd"]]
        with open(self.log, "a", encoding="utf-8") as log:
            log.write(f"\n===== {step['name']}: {' '.join(cmd)}  [{now()}]\n")
            log.flush()
            proc = subprocess.Popen(cmd, cwd=self.repo, env=self.env(), text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    encoding="utf-8", errors="replace", bufsize=1)
            for line in proc.stdout:
                log.write(line)
                log.flush()
                try:
                    print(line, end="", flush=True)
                except (OSError, ValueError):   # console gone; the log has it all
                    pass
                # tqdm redraws with \r; the newest segment is the current state
                self.progress.line(line.replace("\r", "\n").rstrip().split("\n")[-1])
            return proc.wait()

    def wait(self, step: dict) -> bool:
        """True once wait_for exists; False on abort_if or timeout.

        A waiting VM is an idle VM that is still billing. On 2026-10-03 one waited six
        hours for an upload that had failed, so no wait may exceed WAIT_MAX_MINUTES,
        whatever the spec asks for: the job ends, its VM terminates, and it is
        relaunched once its inputs are really there."""
        minutes = min(step.get("timeout_minutes", WAIT_DEFAULT_MINUTES), WAIT_MAX_MINUTES)
        deadline = time.monotonic() + 60 * minutes
        poll = step.get("poll_seconds", 60)
        while time.monotonic() < deadline:
            if self.store.exists(step["wait_for"]):
                for rel in step.get("pull", []):
                    self.store.sync_down(rel, self.runs / rel.removeprefix("runs/"))
                return True
            if step.get("abort_if") and self.store.exists(step["abort_if"]):
                print(f"[job] {step['abort_if']} exists -- not waiting any longer", flush=True)
                return False
            time.sleep(poll)
        print(f"[job] gave up waiting for {step['wait_for']} after {minutes} min", flush=True)
        return False

    def finish(self, state: str, extra: dict) -> None:
        self.progress.state["state"] = state
        self.put_status()
        body = json.dumps({**self.progress.snapshot(), **extra}, indent=1)
        self.store.put_text(f"{self.status_rel}/{state.upper()}", body)

    def run(self) -> int:
        previous = self.store.get_text(f"{self.status_rel}/status.json")
        if previous:   # a reboot: carry on where it stopped
            self.progress.state["done"] = json.loads(previous).get("done", [])
        if self.store.exists(f"{self.status_rel}/DONE") or \
                self.store.exists(f"{self.status_rel}/FAILED"):
            print(f"[job] {self.name} already finished; nothing to do", flush=True)
            return 0
        self.runs.mkdir(parents=True, exist_ok=True)
        for rel in self.spec.get("pull", []):
            self.store.sync_down(rel, self.runs / rel.removeprefix("runs/"))

        self.beat = threading.Thread(target=self._background, daemon=True)
        self.beat.start()
        skipped: list[str] = []
        try:
            steps = self.spec["steps"]
            for i, step in enumerate(steps):
                if step["name"] in self.progress.state["done"]:
                    continue
                self.progress.begin(step["name"])
                self.put_status()
                if "wait_for" in step:
                    if not self.wait(step):
                        skipped = [s["name"] for s in steps[i:]]
                        break
                else:
                    code = self.run_cmd(step)
                    if code:
                        self.halt()
                        self.sync()
                        tail = self.log.read_text(encoding="utf-8",
                                                  errors="replace").splitlines()[-40:]
                        self.finish("failed", {"failed_step": step["name"], "exit_code": code,
                                               "log_tail": tail})
                        return code
                self.progress.finish(step["name"])
                self.put_status()
            self.halt()
            self.sync()
            files = manifest(self.runs, self.outputs)
            self.store.put_text(f"{self.status_rel}/manifest.json", json.dumps(files, indent=1))
            self.finish("done", {"skipped": skipped, "files": len(files)})
            return 0
        except BaseException as e:
            self.halt()
            try:
                self.sync()
            finally:
                tail = (self.log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
                        if self.log.exists() else [])
                self.finish("failed", {"failed_step": self.progress.state["step"],
                                       "error": f"{type(e).__name__}: {e}", "log_tail": tail})
            raise
        finally:
            self.stop.set()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--store", required=True, help="gs://bucket, or a directory in tests")
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--heartbeat", type=float, default=60.0)
    args = ap.parse_args(argv)
    # The steps' output is echoed here as well as logged. A console that cannot encode a
    # progress bar's block characters must garble them, not end the job.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    log = args.log or args.runs.parent / f"{spec['job']}.log"
    return Job(spec, Store(args.store), args.repo, args.runs, log, args.heartbeat).run()


if __name__ == "__main__":
    sys.exit(main())
