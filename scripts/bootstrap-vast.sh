#!/usr/bin/env bash
#
# One-time bootstrap script for a fresh Vast.ai GPU instance.
#
# Usage:
#   sudo HF_TOKEN=hf_xxx VLLM_API_KEY=my-key TP_SIZE=2 ./scripts/bootstrap-vast.sh
#
# Required env vars:
#   HF_TOKEN        – Hugging Face access token with read access to Qwen model
#   VLLM_API_KEY    – API key that clients will use when hitting /v1
#
# Optional overrides:
#   MODEL_REPO      – Defaults to Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
#   MODEL_SUBDIR    – Defaults to qwen3-coder-30b-a3b-fp8
#   HOST_MODELS_DIR – Defaults to /root/models
#   TP_SIZE         – Tensor parallel degree (default 2 for dual GPUs)
#   MAX_MODEL_LEN   – Default 20000
#   MAX_NUM_SEQS    – Default 384
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "[bootstrap] Run this script as root (or with sudo)." >&2
  exit 1
fi

: "${HF_TOKEN:?Set HF_TOKEN env var}"
: "${VLLM_API_KEY:?Set VLLM_API_KEY env var}"

MODEL_REPO=${MODEL_REPO:-Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8}
MODEL_SUBDIR=${MODEL_SUBDIR:-qwen3-coder-30b-a3b-fp8}
HOST_MODELS_DIR=${HOST_MODELS_DIR:-/root/models}
MODEL_DIR="$HOST_MODELS_DIR/$MODEL_SUBDIR"
TP_SIZE=${TP_SIZE:-2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-20000}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-384}

echo "[bootstrap] Updating apt and installing dependencies…"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ca-certificates curl gnupg lsb-release \
  python3-pip git fuse-overlayfs dos2unix \
  docker.io docker-compose-plugin

echo "[bootstrap] Installing Hugging Face CLI…"
pip3 install --upgrade "huggingface_hub"

echo "[bootstrap] Logging into Hugging Face…"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential --non-interactive >/dev/null

echo "[bootstrap] Downloading $MODEL_REPO into $MODEL_DIR …"
mkdir -p "$MODEL_DIR"
HF_HUB_DISABLE_SYMLINKS=1 huggingface-cli download \
  "$MODEL_REPO" \
  --repo-type model \
  --local-dir "$MODEL_DIR" \
  --local-dir-use-symlinks False

echo "[bootstrap] Writing virtualization/vllm/.env.vllm …"
cp virtualization/vllm/.env.example virtualization/vllm/.env.vllm
cat <<EOF > virtualization/vllm/.env.vllm
HOST_MODELS_DIR=$HOST_MODELS_DIR
CONTAINER_MODELS_DIR=/models
MODEL_PATH=/models/$MODEL_SUBDIR
TOKENIZER_PATH=/models/$MODEL_SUBDIR
HF_TOKEN=$HF_TOKEN
VLLM_API_KEY=$VLLM_API_KEY
VLLM_PORT=8000
TENSOR_PARALLEL=$TP_SIZE
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=$MAX_MODEL_LEN
MAX_NUM_SEQS=$MAX_NUM_SEQS
DTYPE=auto
ENFORCE_EAGER=True
VLLM_CONTAINER_NAME=vllm-qwen3-coder-30b
EOF

echo "[bootstrap] Creating helper to start dockerd in restricted environments…"
cat <<'EOF' >/usr/local/bin/start-dockerd-headless.sh
#!/usr/bin/env bash
set -euo pipefail
LOG_FILE=${DOCKERD_LOG:-/var/log/dockerd.log}
mkdir -p "$(dirname "$LOG_FILE")"
exec nohup dockerd \
  --storage-driver=fuse-overlayfs \
  --iptables=false \
  --ip-forward=false \
  --ip-masq=false \
  --bip=none \
  --bridge=none \
  >"$LOG_FILE" 2>&1 &
EOF
chmod +x /usr/local/bin/start-dockerd-headless.sh

echo "[bootstrap] Starting Docker daemon…"
start-dockerd-headless.sh
sleep 4

echo "[bootstrap] Launching vLLM stack (docker compose)…"
bin/local-vllm start

cat <<'EOF'

[bootstrap] Done! Useful commands:
  start-dockerd-headless.sh        # restart daemon if VM reboots
  bin/local-vllm logs -f           # tail vLLM service logs
  curl http://localhost:8000/v1/models -H "Authorization: Bearer $VLLM_API_KEY"

Expose port 8000 via Vast.ai or tunnel: ssh -L 8004:localhost:8000 user@host
EOF
