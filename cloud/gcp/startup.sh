#!/bin/bash
# GCE startup script for one ROAD job. Runs as root on EVERY boot, so each stage is
# idempotent: a reboot resumes the job instead of starting it again.
#
#   1. a job already DONE or FAILED in the bucket is never rerun
#   2. install uv, Python 3.13 and the versions .venv-colab tested (torch cu128)
#   3. fetch the code, the three CSVs and the job spec from the bucket
#   4. run cloud/gcp/job.py, which keeps status, logs and results in the bucket
#   5. power off if nothing has deleted the VM IDLE minutes after the job ended --
#      checker.py normally deletes it sooner, once the results are downloaded and
#      verified. Powering off keeps the disk, so even this path loses nothing.
#
# Metadata: road-bucket, road-job, road-idle-minutes (default 2), road-cpu (1 = no GPU)

set -uo pipefail
ROOT=/opt/road
mkdir -p "$ROOT"
exec > >(tee -a "$ROOT/startup.log") 2>&1

META=http://metadata.google.internal/computeMetadata/v1/instance/attributes
md() { curl -sf -H "Metadata-Flavor: Google" "$META/$1"; }
BUCKET=$(md road-bucket)
JOB=$(md road-job)
IDLE=$(md road-idle-minutes || echo 2)
CPU=$(md road-cpu || echo 0)
STORE="gs://$BUCKET"
STATUS="$STORE/status/$JOB"
echo "[startup] $(date -u +%FT%TZ) job=$JOB store=$STORE cpu=$CPU idle=${IDLE}m"

upload_log() { gcloud storage cp "$ROOT/startup.log" "$STATUS/startup.log" >/dev/null 2>&1 || true; }

idle_then_poweroff() {
    upload_log
    echo "[startup] job ended; powering off in ${IDLE}m unless the checker deletes this VM first"
    sleep $((IDLE * 60))
    upload_log
    shutdown -h now
    exit 0
}

fail() {   # a failure before job.py runs must still reach the bucket, or the checker waits forever
    echo "[startup] FAILED: $1"
    python3 - "$1" > "$ROOT/FAILED.json" <<'EOF'
import json, sys, datetime as dt
log = open("/opt/road/startup.log", encoding="utf-8", errors="replace").read().splitlines()
print(json.dumps({"state": "failed", "failed_step": "startup", "error": sys.argv[1],
                  "updated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                  "log_tail": log[-40:]}, indent=1))
EOF
    gcloud storage cp "$ROOT/FAILED.json" "$STATUS/FAILED" || true
    idle_then_poweroff
}

[ -n "$BUCKET" ] && [ -n "$JOB" ] || { echo "[startup] road-bucket/road-job metadata missing"; exit 1; }

if gcloud storage ls "$STATUS/DONE" >/dev/null 2>&1 || gcloud storage ls "$STATUS/FAILED" >/dev/null 2>&1; then
    echo "[startup] $JOB already finished -- not rerunning it"
    idle_then_poweroff
fi

# ---- 2. environment (once per disk) -------------------------------------------------
if [ ! -f "$ROOT/.installed" ]; then
    if [ "$CPU" != "1" ]; then
        for i in $(seq 1 30); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done
        nvidia-smi || fail "no GPU visible to nvidia-smi after 5 minutes"
    fi
    # uv from its GitHub release, checked against a pinned sha256: this image's python3
    # has no pip, and nothing unverified gets to run as root
    UV_TGZ=uv-x86_64-unknown-linux-gnu.tar.gz
    UV_SHA=f97935763c04be3e692460a7aaeaaab8fc3b78fcf8b389da820b38ae7423a638
    curl -fsSL -o "/tmp/$UV_TGZ" "https://github.com/astral-sh/uv/releases/download/0.12.15/$UV_TGZ" \
        || fail "download uv"
    echo "$UV_SHA  /tmp/$UV_TGZ" | sha256sum -c - || fail "uv checksum mismatch"
    tar -xzf "/tmp/$UV_TGZ" -C /tmp && install -m 755 /tmp/uv-x86_64-unknown-linux-gnu/uv /usr/local/bin/uv \
        || fail "install uv"
    export UV_PYTHON_INSTALL_DIR="$ROOT/python"
    uv python install 3.13 || fail "uv python install 3.13"
    uv venv --python 3.13 "$ROOT/venv" || fail "uv venv"
    PY="$ROOT/venv/bin/python"
    # torch family first, from the CUDA 12.8 index: the driver on this image supports it,
    # and these are the versions .venv-colab tests against (there as CPU builds)
    uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 || fail "install torch cu128"
fi
PY="$ROOT/venv/bin/python"

# ---- 3. code, data and the job spec ----------------------------------------------------
if [ ! -d "$ROOT/repo/src" ]; then
    mkdir -p "$ROOT/repo"
    gcloud storage cp "$STORE/code/repo.tar.gz" "$ROOT/repo.tar.gz" || fail "fetch code"
    tar -xzf "$ROOT/repo.tar.gz" -C "$ROOT/repo" || fail "unpack code"
fi
if [ ! -f "$ROOT/.installed" ]; then
    # everything else .venv-colab pins, minus the torch family installed above
    grep -v -E '^(torch|torchvision|torchaudio)==' "$ROOT/repo/requirements-colab-test.txt" \
        > "$ROOT/repo/requirements-gcp.txt"
    (cd "$ROOT/repo" && uv pip install --python "$PY" -r requirements-gcp.txt) \
        || fail "install requirements"
    uv pip uninstall --python "$PY" torchao >/dev/null 2>&1 || true   # peft 0.20 rejects it
    "$PY" - <<EOF || fail "torch check"
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
assert torch.__version__.startswith("2.11.0"), torch.__version__
if "$CPU" != "1":
    assert torch.cuda.is_available(), "torch cannot see the GPU"
    print(torch.cuda.get_device_name(0))
EOF
    touch "$ROOT/.installed"
fi
mkdir -p "$ROOT/csv"
gcloud storage cp "$STORE/csv/*.csv" "$ROOT/csv/" || fail "fetch CSVs"
gcloud storage cp "$STORE/jobs/$JOB.json" "$ROOT/job.json" || fail "fetch job spec"

# ---- 4. the job -------------------------------------------------------------------------
export HF_HOME="$ROOT/hf"
"$PY" "$ROOT/repo/cloud/gcp/job.py" --spec "$ROOT/job.json" --store "$STORE" \
    --repo "$ROOT/repo" --runs "$ROOT/runs" --log "$ROOT/$JOB.log"
echo "[startup] job.py exited with $?"
idle_then_poweroff
