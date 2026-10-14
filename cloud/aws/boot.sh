#!/bin/bash
# EC2 user-data for one ROAD job -- cloud/gcp/startup.sh's stages, on AWS.
#
# cloud-init runs user-data only on the first boot, so this installs the job as a
# systemd service (road.service) and starts it: systemd then runs it on EVERY boot,
# and each stage is idempotent, so a reboot resumes the job instead of repeating it.
#
#   1. a job already DONE or FAILED in the bucket is never rerun
#   2. a hard cap: AWS has no max-run-duration, so every boot schedules a power-off
#      MAX_HOURS ahead (the instance is launched to STOP, never terminate, on power-off)
#   3. install uv (pinned sha256), Python 3.13, torch 2.11 cu128 and .venv-colab's pins
#   4. fetch code, CSVs and the job spec from S3; run cloud/gcp/job.py against s3://
#   5. IDLE minutes (default 2) after the job ends, terminate itself if its DONE or
#      FAILED is in the bucket, else stop with the disk kept -- an ended job never
#      leaves a VM billing, and job.py never waits more than 30 minutes for an input
#
# launch.py fills in the placeholders in MAIN below; a test checks none is left.

mkdir -p /opt/road
cat > /opt/road/main.sh <<'MAIN'
#!/bin/bash
set -uo pipefail
BUCKET="__BUCKET__"
JOB="__JOB__"
IDLE="__IDLE__"
CPU="__CPU__"
MAX_HOURS="__MAX_HOURS__"
export AWS_DEFAULT_REGION="__REGION__"
ROOT=/opt/road
exec > >(tee -a "$ROOT/startup.log") 2>&1
STORE="s3://$BUCKET"
STATUS="$STORE/status/$JOB"
echo "[boot] $(date -u +%FT%TZ) job=$JOB store=$STORE cpu=$CPU idle=${IDLE}m cap=${MAX_HOURS}h"

shutdown -h "+$((MAX_HOURS * 60))" "ROAD hard cap: ${MAX_HOURS}h"

exists() { aws s3api head-object --bucket "$BUCKET" --key "status/$JOB/$1" >/dev/null 2>&1; }
upload_log() { aws s3 cp --only-show-errors "$ROOT/startup.log" "$STATUS/startup.log" || true; }

self_id() {   # IMDSv2: a token first, then the instance id
    local t
    t=$(curl -sf -X PUT http://169.254.169.254/latest/api/token \
        -H "X-aws-ec2-metadata-token-ttl-seconds: 300") || return 1
    curl -sf -H "X-aws-ec2-metadata-token: $t" http://169.254.169.254/latest/meta-data/instance-id
}

idle_then_poweroff() {
    # An ended job's VM must not sit billing. It terminates itself IDLE minutes after the
    # job ends -- but only once the job's DONE or FAILED is in the bucket, which job.py
    # writes after its final sync, so terminating cannot lose results. If that cannot be
    # confirmed it STOPS instead and keeps its disk for recovery.
    upload_log
    echo "[boot] job ended; ending this VM in ${IDLE}m"
    sleep $((IDLE * 60))
    upload_log
    if exists DONE || exists FAILED; then
        id=$(self_id) && aws ec2 terminate-instances --instance-ids "$id" >/dev/null \
            && { echo "[boot] terminating $id"; exit 0; }
        echo "[boot] could not terminate myself -- stopping instead"
    else
        echo "[boot] no DONE or FAILED in the bucket -- stopping, disk kept"
    fi
    upload_log
    shutdown -h now
    exit 0
}

fail() {   # a failure before job.py runs must still reach the bucket
    echo "[boot] FAILED: $1"
    python3 - "$1" > "$ROOT/FAILED.json" <<'EOF'
import json, sys, datetime as dt
log = open("/opt/road/startup.log", encoding="utf-8", errors="replace").read().splitlines()
print(json.dumps({"state": "failed", "failed_step": "startup", "error": sys.argv[1],
                  "updated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                  "log_tail": log[-40:]}, indent=1))
EOF
    aws s3 cp --only-show-errors "$ROOT/FAILED.json" "$STATUS/FAILED" || true
    idle_then_poweroff
}

if exists DONE || exists FAILED; then
    echo "[boot] $JOB already finished -- not rerunning it"
    idle_then_poweroff
fi

if [ ! -f "$ROOT/.installed" ]; then
    if [ "$CPU" != "1" ]; then
        for i in $(seq 1 30); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done
        nvidia-smi || fail "no GPU visible to nvidia-smi after 5 minutes"
    fi
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
    uv pip install --python "$ROOT/venv/bin/python" --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 || fail "install torch cu128"
fi
export UV_PYTHON_INSTALL_DIR="$ROOT/python"
PY="$ROOT/venv/bin/python"

if [ ! -d "$ROOT/repo/src" ]; then
    mkdir -p "$ROOT/repo"
    aws s3 cp --only-show-errors "$STORE/code/repo.tar.gz" "$ROOT/repo.tar.gz" || fail "fetch code"
    tar -xzf "$ROOT/repo.tar.gz" -C "$ROOT/repo" || fail "unpack code"
fi
if [ ! -f "$ROOT/.installed" ]; then
    grep -v -E '^(torch|torchvision|torchaudio)==' "$ROOT/repo/requirements-colab-test.txt" \
        > "$ROOT/repo/requirements-cloud.txt"
    (cd "$ROOT/repo" && uv pip install --python "$PY" -r requirements-cloud.txt) \
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
aws s3 cp --only-show-errors --recursive "$STORE/csv/" "$ROOT/csv/" || fail "fetch CSVs"
aws s3 cp --only-show-errors "$STORE/jobs/$JOB.json" "$ROOT/job.json" || fail "fetch job spec"

export HF_HOME="$ROOT/hf"
"$PY" "$ROOT/repo/cloud/gcp/job.py" --spec "$ROOT/job.json" --store "$STORE" \
    --repo "$ROOT/repo" --runs "$ROOT/runs" --log "$ROOT/$JOB.log"
echo "[boot] job.py exited with $?"
idle_then_poweroff
MAIN
chmod +x /opt/road/main.sh

cat > /etc/systemd/system/road.service <<'UNIT'
[Unit]
Description=ROAD job (resumes on every boot)
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
ExecStart=/opt/road/main.sh
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable road.service
systemctl start --no-block road.service
