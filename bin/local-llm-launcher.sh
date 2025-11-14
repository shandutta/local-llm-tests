#!/usr/bin/env bash
#
# Convenience wrapper that starts/stops the entire Local LLM stack:
# - llama-server Docker container (via bin/local-llm)
# - FastAPI control plane (uvicorn on port 8008)
# - Next.js frontend (npm run dev:frontend on port 3000)
#
# The script is designed so it can be wired to a desktop shortcut or run
# manually: `bin/local-llm-launcher.sh start --model gpt-oss-120b`
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLI="$REPO_ROOT/bin/local-llm"
ENV_FILE="$REPO_ROOT/virtualization/docker/.env.runtime"
STATE_DIR="$REPO_ROOT/.runtime"
SERVER_DIR="$REPO_ROOT/server"
WEB_DIR="$REPO_ROOT/web"
SERVER_PID="$STATE_DIR/server.pid"
WEB_PID="$STATE_DIR/web.pid"
SERVER_LOG="$STATE_DIR/server.log"
WEB_LOG="$STATE_DIR/web.log"
FASTAPI_HOST="${FASTAPI_HOST:-0.0.0.0}"
FASTAPI_PORT="${FASTAPI_PORT:-8008}"
DEFAULT_MODEL="${LOCAL_LLM_DEFAULT_MODEL:-gpt-oss-120b}"

mkdir -p "$STATE_DIR"

log() {
  echo "[launcher] $*"
}

is_pid_running() {
  local pid_file=$1
  [[ -f "$pid_file" ]] || return 1
  local pid
  pid="$(cat "$pid_file")"
  if kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  rm -f "$pid_file"
  return 1
}

stop_pid() {
  local pid_file=$1
  local label=$2
  if is_pid_running "$pid_file"; then
    local pid
    pid="$(cat "$pid_file")"
    log "Stopping $label (pid $pid)..."
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
  rm -f "$pid_file"
}

ensure_frontend_dependencies() {
  if [[ ! -d "$WEB_DIR/node_modules" ]]; then
    log "Installing frontend dependencies..."
    (cd "$WEB_DIR" && npm install)
  fi
}

start_fastapi() {
  if is_pid_running "$SERVER_PID"; then
    log "FastAPI server already running (pid $(cat "$SERVER_PID"))."
    return
  fi
  log "Starting FastAPI server on ${FASTAPI_HOST}:${FASTAPI_PORT}..."
  (
    cd "$SERVER_DIR"
    nohup python3 -m uvicorn main:app --host "$FASTAPI_HOST" --port "$FASTAPI_PORT" \
      >>"$SERVER_LOG" 2>&1 &
    echo $! >"$SERVER_PID"
  )
  log "FastAPI server logs: $SERVER_LOG"
}

start_frontend() {
  ensure_frontend_dependencies
  if is_pid_running "$WEB_PID"; then
    log "Frontend already running (pid $(cat "$WEB_PID"))."
    return
  fi
  log "Starting Next.js frontend on http://localhost:3000 ..."
  (
    cd "$WEB_DIR"
    nohup npm run dev:frontend >>"$WEB_LOG" 2>&1 &
    echo $! >"$WEB_PID"
  )
  log "Frontend logs: $WEB_LOG"
}

stop_frontend() {
  stop_pid "$WEB_PID" "frontend"
}

stop_fastapi() {
  stop_pid "$SERVER_PID" "FastAPI server"
}

start_model() {
  local model=$1
  if [[ -f "$ENV_FILE" ]]; then
    # Try to shut down anything else that might be running first.
    log "Stopping any previously running model container..."
    if ! "$CLI" stop >/dev/null 2>&1; then
      log "No prior model container to stop."
    fi
  fi
  log "Starting model container for '$model'..."
  "$CLI" start "$model"
}

stop_model() {
  if [[ -f "$ENV_FILE" ]]; then
    log "Stopping model container..."
    "$CLI" stop
  else
    log "No model container metadata found; skipping stop."
  fi
}

show_status() {
  if [[ -f "$ENV_FILE" ]]; then
    log "Docker status:"
    "$CLI" status || true
  else
    log "No model container currently configured."
  fi
  if is_pid_running "$SERVER_PID"; then
    log "FastAPI server running (pid $(cat "$SERVER_PID")), log: $SERVER_LOG"
  else
    log "FastAPI server not running."
  fi
  if is_pid_running "$WEB_PID"; then
    log "Frontend running (pid $(cat "$WEB_PID")), log: $WEB_LOG"
  else
    log "Frontend not running."
  fi
}

usage() {
  cat <<'EOF'
Usage: local-llm-launcher.sh <start|stop|status> [--model NAME]

Examples:
  # start GPT-OSS (default) plus API + frontend
  bin/local-llm-launcher.sh start

  # start Qwen3 coder instead
  bin/local-llm-launcher.sh start --model qwen3-coder

  # stop everything
  bin/local-llm-launcher.sh stop

  # show state/log locations
  bin/local-llm-launcher.sh status
EOF
}

ACTION="${1:-}"
MODEL="$DEFAULT_MODEL"

if [[ -z "$ACTION" ]]; then
  usage
  exit 1
fi
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|-m)
      MODEL="${2:-}"
      if [[ -z "$MODEL" ]]; then
        echo "--model requires a value" >&2
        exit 1
      fi
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

case "$ACTION" in
  start)
    start_model "$MODEL"
    start_fastapi
    start_frontend
    ;;
  stop)
    stop_frontend
    stop_fastapi
    stop_model
    ;;
  status)
    show_status
    ;;
  *)
    echo "Unknown action '$ACTION'" >&2
    usage
    exit 1
    ;;
esac
