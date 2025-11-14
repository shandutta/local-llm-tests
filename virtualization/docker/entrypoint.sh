#!/usr/bin/env bash
set -euo pipefail

LLAMA_BIN="${LLAMA_BIN:-/opt/llama/bin/llama-server}"
MODEL_PATH="${MODEL_PATH:-}"
SERVER_PORT="${SERVER_PORT:-8000}"
LLAMA_HOST="${LLAMA_HOST:-0.0.0.0}"
export LD_LIBRARY_PATH="/opt/llama/lib:${LD_LIBRARY_PATH:-}"

if [[ -z "$MODEL_PATH" ]]; then
  echo "[entrypoint] MODEL_PATH is required" >&2
  exit 1
fi

if [[ ! -x "$LLAMA_BIN" ]]; then
  echo "[entrypoint] llama-server not found at $LLAMA_BIN" >&2
  exit 1
fi

readarray -t EXTRA_ARGS < <(python3 - <<'PY'
import json, os, sys
payload = os.environ.get("LLAMA_JSON_ARGS", "[]")
try:
    args = json.loads(payload)
except json.JSONDecodeError as exc:
    sys.stderr.write(f"[entrypoint] failed to decode LLAMA_JSON_ARGS: {exc}\n")
    sys.exit(1)
for item in args:
    print(str(item))
PY
)

CMD=("$LLAMA_BIN" "--model" "$MODEL_PATH" "--host" "$LLAMA_HOST" "--port" "$SERVER_PORT")
CMD+=("${EXTRA_ARGS[@]}")

echo "[entrypoint] launching llama-server with model $MODEL_PATH on port $SERVER_PORT"
exec "${CMD[@]}"
