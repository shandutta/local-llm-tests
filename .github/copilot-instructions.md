# Copilot Instructions for Local-LLM-Tests

## Repository Overview

**local-llm-tests** is a comprehensive orchestration platform for running large language models (LLMs) on NVIDIA GPUs via Docker containers with a full-stack web interface.

**Key Facts:**
- **Architecture**: Multi-tier stack with Docker inference backend, FastAPI control plane, and Next.js frontend
- **Languages**: Python (orchestration/backend), TypeScript/TSX (frontend), Bash (shell scripts)
- **Target models**: `gpt-oss-120b` (reasoning), `qwen3-coder-30b` (code generation)
- **GPU Backend**: CUDA 12.6 on RTX 5090 (NVIDIA Container Toolkit required)
- **Service Stack**: llama.cpp (inference) + FastAPI (control) + Next.js (UI) + Docker Compose orchestration
- **Special handling**: GPT-OSS models require prompt formatting via Unsloth Harmony protocol

## Core Architecture & Data Flow

### Three-Tier Stack Design

1. **Backend Orchestration** (`bin/local-llm`, `server/main.py`)
   - Single source of truth: `config/models.yaml` (manifest of available models with llama-server arguments)
   - Python CLI (`bin/local-llm`): Reads manifest → Validates GGUF files → Writes `.env.runtime` → Triggers docker-compose
   - FastAPI Service (`server/main.py`): RESTful wrapper around the CLI with streaming chat support

2. **Docker Inference Layer** (`virtualization/docker/`)
   - **Dockerfile**: Multi-stage build (builder stage compiles llama.cpp with CUDA 12.6, runtime stage runs llama-server)
   - **Entrypoint**: Reads `MODEL_PATH` + `LLAMA_JSON_ARGS` from env → Launches llama-server with those args
   - **docker-compose.yaml**: Binds `HOST_MODELS_DIR` → `CONTAINER_MODELS_DIR:ro`, exposes `SERVER_PORT` to host, requests GPU via `nvidia` device reservation
   - **Key pattern**: All model configuration comes via JSON serialized in env var `LLAMA_JSON_ARGS` (avoids shell escaping issues)

3. **Web Frontend** (`web/`, Next.js 16)
   - Fetches model list and status via `NEXT_PUBLIC_API_BASE` (environment-configurable endpoint)
   - Uses SWR for data-fetching with 5-second refresh interval on status
   - Renders model cards with start/stop buttons → calls `/start`, `/stop` endpoints
   - Chat panel with streaming responses, Harmony reasoning-effort selector for GPT-OSS

### Integration Points

**CLI → Docker**: `bin/local-llm` + `config/models.yaml` → `.env.runtime` → `docker-compose up -d`
- Model selection → Manifest lookup → Argument serialization → Container environment vars

**FastAPI → CLI**: `server/main.py` wraps CLI calls, ensures single-model-at-a-time constraint
- All model state changes go through `/start`, `/stop`, `/restart` endpoints which invoke `bin/local-llm` subprocesses

**FastAPI → llama-server**: Proxies chat requests to running container on `LLAMA_HOST:SERVER_PORT`
- Streams SSE responses, applies Harmony prompt encoding for GPT-OSS models only

**Frontend → FastAPI**: RESTful JSON + Server-Sent Events (SSE) for streaming
- `/models` (GET) → manifest data
- `/status` (GET) → docker container state  
- `/start`, `/stop`, `/restart` (POST) → model lifecycle
- `/chat` (POST) → streaming inference responses

### Data Flow Example: Start Model

```
User clicks "Start GPT-OSS-120B" in web UI
  ↓
POST /start {"model": "gpt-oss-120b"} 
  ↓
FastAPI /start → calls: bin/local-llm stop (noop if nothing running) + bin/local-llm start gpt-oss-120b
  ↓
bin/local-llm loads config/models.yaml, validates gpt-oss-120b entry:
  - Resolves HOST_MODELS_DIR + relative_path to full GGUF path
  - Serializes all --arguments as JSON into LLAMA_JSON_ARGS
  - Writes .env.runtime with: MODEL_PATH, LLAMA_JSON_ARGS, SERVER_PORT, CONTAINER_NAME, etc.
  ↓
docker-compose -f virtualization/docker/docker-compose.yaml --env-file .env.runtime up -d --build
  ↓
Dockerfile builds (or uses cache), spins up llama container
  ↓
entrypoint.sh reads env vars, runs: llama-server -m $MODEL_PATH [args from LLAMA_JSON_ARGS] --listen $LLAMA_HOST:$SERVER_PORT
  ↓
Container listens on LLAMA_HOST:SERVER_PORT, FastAPI proxies requests through
  ↓
Frontend polls /status every 5s, displays "Running on port 8002"
```

## Critical Configuration Patterns

### Model Manifest (`config/models.yaml`)

```yaml
defaults:
  host_models_dir: "/home/shan/models"      # Host path where GGUF files live
  container_models_dir: "/models"           # Where they're mounted inside container
  llama_host: "0.0.0.0"                    # Listen on all interfaces (required for remote access)

models:
  gpt-oss-120b:
    description: "..."
    relative_path: "gpt-oss-120b/gpt-oss-120b-F16.gguf"   # Relative to host_models_dir
    port: 8002                              # Exposed port on host
    arguments: [                            # Passed to llama-server
      "--alias", "gpt-oss-120b",
      "--n-gpu-layers", "99",               # Full VRAM offload (RTX 5090 = 32GB)
      "--n-cpu-moe", "25",                  # MoE routing on CPU for memory efficiency
      "--ctx-size", "131072",               # 128k context window
      "--threads", "14",                    # CPU threads
      "--temp", "1.0",                      # Temperature for sampling
      "--parallel", "4",                    # Concurrent requests
      "--cont-batching",                    # Continuous batching mode
      "--flash-attn", "on"                  # Flash attention optimization
    ]
```

**Critical behaviors:**
- `relative_path` is resolved against `host_models_dir` to get absolute host path
- All `arguments` must be valid llama-server flags (check `llama-server -h` for current version)
- `port` must not conflict with other services (FastAPI is 8008, Next.js is 3000)
- Manifest is YAML; `bin/local-llm` and `server/main.py` both parse it with `yaml.safe_load()`

### Environment Variable Serialization

The critical innovation: Model arguments are serialized as JSON in `LLAMA_JSON_ARGS` environment variable to avoid shell escaping issues.

**In `.env.runtime`** (written by `bin/local-llm`):
```
MODEL_PATH=/models/gpt-oss-120b/gpt-oss-120b-F16.gguf
LLAMA_JSON_ARGS=["--alias","gpt-oss-120b","--n-gpu-layers","99",...]
SERVER_PORT=8002
LLAMA_HOST=0.0.0.0
CONTAINER_NAME=local-llm-gpt-oss-120b
```

**In `entrypoint.sh`** (reads these vars, expands args):
```bash
MODEL_PATH="${MODEL_PATH:?MODEL_PATH not set}"
LLAMA_JSON_ARGS="${LLAMA_JSON_ARGS:?LLAMA_JSON_ARGS not set}"
LLAMA_HOST="${LLAMA_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-8000}"

# Parse JSON array and expand to command-line arguments
ARGS=$(python3 -c "import json, sys; print(' '.join(json.loads('$LLAMA_JSON_ARGS')))")
exec llama-server -m "$MODEL_PATH" $ARGS --listen "$LLAMA_HOST:$SERVER_PORT"
```

## Critical Developer Workflows

### Starting the Full Stack

**Option 1: Launcher script** (recommended for development/desktop use)
```bash
cd /home/shan/local-llm-tests
bin/local-llm-launcher.sh start --model gpt-oss-120b
# or
LOCAL_LLM_DEFAULT_MODEL=qwen3-coder bin/local-llm-launcher.sh start
```
- Orchestrates: Docker model container + FastAPI (port 8008) + Next.js frontend (port 3000)
- Logs to `.runtime/{server,web}.log`
- Idempotent: stop → clean up → start fresh

**Option 2: Manual stack** (for isolated testing)
```bash
# Terminal 1: Start model container
./bin/local-llm start gpt-oss-120b
./bin/local-llm status  # Monitor docker-compose ps output

# Terminal 2: Start FastAPI service
pip install -r server/requirements.txt
uvicorn server.main:app --reload --host 0.0.0.0 --port 8008

# Terminal 3: Start Next.js frontend
cd web && npm install && npm run dev:frontend
```

### Building the Docker Image

The Dockerfile has a two-stage build. Rebuilds happen on `docker-compose up --build`:
```bash
# Full rebuild (happens automatically):
docker-compose -f virtualization/docker/docker-compose.yaml up -d --build

# Check if build cached:
docker images | grep local-llm

# Manual rebuild (force cache invalidation):
docker build --no-cache -f virtualization/docker/Dockerfile .
```

**Build time**: ~10-15 minutes first time (compiling llama.cpp with CUDA), <1 minute on subsequent runs (cached).

### Viewing Logs

**Docker container**:
```bash
# Live logs
docker logs -f local-llm

# Check last 100 lines
docker logs --tail 100 local-llm
```

**FastAPI service** (when run via launcher):
```bash
tail -f .runtime/server.log
```

**Frontend** (when run via launcher):
```bash
tail -f .runtime/web.log
```

### Debugging Model Selection Issues

If a model won't start:
1. Check manifest is valid YAML: `python3 -c "import yaml; print(yaml.safe_load(open('config/models.yaml')))"`
2. Check GGUF file exists: `ls -lh ~/models/gpt-oss-120b/gpt-oss-120b-F16.gguf`
3. Run CLI directly: `bin/local-llm start gpt-oss-120b` (shows stderr)
4. Check docker env: `cat virtualization/docker/.env.runtime`
5. Inspect docker container: `docker inspect local-llm | jq '.[] | {Env, Mounts}'`

## Project-Specific Conventions

### Model-Specific Text Cleaning

**GPT-OSS-120B** uses a special prompt format with channel tags that must be stripped for display:
- Prompt format: `<|start|>assistant<|channel|>final<|message|>[content]<|end|>`
- Response cleanup in **server/main.py**: `encode_harmony_prompt()` uses Unsloth Harmony encoding when available
- Response cleanup in **web/app/page.tsx**: `cleanGptOssText()` extracts final channel, removes metadata tags

**Qwen3-Coder** returns plain text (no special formatting required).

### Single-Model-at-a-Time Constraint

Only one model runs at a time (VRAM exclusivity on RTX 5090). This is enforced by:
- FastAPI `/start` endpoint: Always calls `stop()` before `start()`
- `bin/local-llm` CLI: Verifies only one container is running via docker-compose status

Attempting to start a model while another is running will auto-stop the previous one.

### Frontend Environment Configuration

**NEXT_PUBLIC_API_BASE**: Environment variable controls where frontend fetches API data.
- Default (development): `http://localhost:8008` (hardcoded fallback if env not set)
- Production: Set via `--env-file` or exported in launcher script
- **Required**: Must point to the FastAPI service endpoint, not the docker inference port

## Key Files & Directory Structure

```
local-llm-tests/
├── config/
│   └── models.yaml              ← Source of truth for available models + llama-server args
├── bin/
│   ├── local-llm               ← Python CLI to orchestrate docker-compose
│   └── local-llm-launcher.sh   ← Full stack launcher (model + API + frontend)
├── server/
│   ├── main.py                 ← FastAPI orchestration service
│   └── requirements.txt         ← Python dependencies (fastapi, uvicorn, pyyaml, httpx, unsloth-zoo)
├── web/
│   ├── app/
│   │   ├── page.tsx            ← Main React component (model list, chat, status polling)
│   │   └── layout.tsx          ← Next.js root layout
│   └── package.json            ← Frontend dependencies (next, react, swr, katex, markdown libs)
├── virtualization/
│   └── docker/
│       ├── Dockerfile          ← Multi-stage build: builder (compiles llama.cpp) + runtime
│       ├── docker-compose.yaml ← Defines llama service, GPU binding, port exposure, volume mounts
│       └── entrypoint.sh       ← Parses JSON args env var, launches llama-server
├── gpt-oss-proxy.py            ← Optional Flask proxy to strip GPT-OSS channel tags (legacy)
└── .runtime/                   ← Generated at runtime: .env.runtime, PID files, logs

Related (not in this repo):
- /home/shan/llama.cpp/          ← llama.cpp source code (built into docker image)
- /home/shan/models/             ← GGUF model files (mounted read-only into container)
```

## Integration with Llama.cpp

The Docker image clones and builds llama.cpp from source:
- Default: Master branch (latest features)
- Configurable via `LLAMA_CPP_REF` build arg in docker-compose (e.g., `--build-arg LLAMA_CPP_REF=b2934`)
- CUDA architecture: Hardcoded to SM 89 (`-DCMAKE_CUDA_ARCHITECTURES=89` for RTX 5090)
- To test against a local llama.cpp build: Modify Dockerfile to `COPY /home/shan/llama.cpp /opt/llama.cpp` instead of git clone

## Common Workflows by Role

**Frontend Developer** (web/):
1. Ensure FastAPI + model are running on localhost:8008
2. `npm run dev:frontend` starts hot-reload dev server on 0.0.0.0:3000
3. Edits to `web/app/page.tsx` reflect instantly (React hot reload)
4. Use browser DevTools to inspect API calls; expected endpoints: `/models`, `/status`, `/chat`, `/start`, `/stop`

**Backend Developer** (server/main.py + bin/local-llm):
1. Test manifest changes: `python3 -c "import yaml; yaml.safe_load(open('config/models.yaml'))"`
2. Iterate on CLI: `bin/local-llm list` → `bin/local-llm start <model>` → `bin/local-llm status`
3. Test FastAPI in isolation: `uvicorn server.main:app --reload --host 0.0.0.0 --port 8008` + manual curl/Postman
4. Verify docker env vars are correctly serialized: `cat virtualization/docker/.env.runtime`

**DevOps** (Dockerfile + docker-compose):
1. Test image builds: `docker build -f virtualization/docker/Dockerfile --build-arg LLAMA_CPP_REF=master -t local-llm:test .`
2. Test compose file independently: `docker-compose -f virtualization/docker/docker-compose.yaml config` (validates syntax)
3. Inspect GPU binding: `docker run --rm --gpus all nvidia-smi` (verify nvidia-container-toolkit works)
4. To debug container runtime: Add `stdin_open: true` + `tty: true` to compose, then `docker-compose run llama bash`

## Known Limitations & Gotchas

1. **CUDA Availability**: Docker must have nvidia-container-toolkit installed; falls back to CPU-only if missing (very slow)
2. **VRAM Overflow**: `--n-gpu-layers 99` assumes RTX 5090 (32 GB); lower this for smaller GPUs
3. **Port Conflicts**: Ensure ports 3000, 8001, 8002, 8008 are available; change in launcher script + manifest
4. **LLAMA_HOST Default**: Set to `0.0.0.0` in manifest so models are accessible from other machines; `localhost` breaks remote requests
5. **Model Path Resolution**: Relative paths in manifest are relative to `host_models_dir`; absolute paths are used as-is
6. **Streaming Responses**: The `/chat` endpoint uses Server-Sent Events (SSE); client must handle `event: error` for failures

## Testing & Validation

**Manual integration test**:
```bash
# 1. Verify manifest
python3 -c "import yaml; m=yaml.safe_load(open('config/models.yaml')); print(list(m['models'].keys()))"

# 2. Start a model via CLI
./bin/local-llm start gpt-oss-120b
./bin/local-llm status

# 3. Test via curl (raw llama-server endpoint)
curl -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-oss-120b","messages":[{"role":"user","content":"Hello"}]}'

# 4. Test via FastAPI
curl http://localhost:8008/models

# 5. Test frontend
open http://localhost:3000
```

**To add a new model**:
1. Download GGUF file to `~/models/<model-name>/`
2. Add entry to `config/models.yaml` with correct `relative_path`, `port`, `arguments`
3. Test: `bin/local-llm list` (should appear), `bin/local-llm start <new-model>`
4. Verify: `docker logs local-llm` should show llama-server startup logs
5. Chat test: POST to `/chat` endpoint with model name
