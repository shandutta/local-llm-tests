# Local LLM Tests

Local LLM Tests is a working log and automation hub for my RTX 5090 based llama.cpp deployments. The goal is to consolidate everything that makes it easy to spin up one high-quality model at a time, expose it to the LAN, and prepare for containerized/virtualized hosting when needed.

## Current Environment Snapshot

- **Host**: AMD Ryzen 9950X3D, 96 GB DDR5 (72 GB allocated to WSL2), Ubuntu 22.04 on WSL2.
- **GPU**: NVIDIA RTX 5090 (32 GB VRAM) – CUDA 13 toolchain installed, llama.cpp built with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="89"` (sm_89).
- **Inference stack**: `/home/shan/llama.cpp` build at version `7315 (4d3726278)` running `llama-server` (CUDA 13, host g++-12).
- **Capabilities**: Qwen3 Coder and GPT-OSS-120B both emit OpenAI-style `tool_calls` when started via the current scripts (Jinja + ChatML enabled). Vision/Thinking stay off.
- **Models on disk/configured**:
  - GPT-OSS-120B F16 (`~/models/gpt-oss-120b/gpt-oss-120b-F16.gguf`, ~61 GB) – premium reasoning, tuned with `--n-cpu-moe 25` for ~21.6 tok/s.
  - Qwen3-Coder-30B-A3B-Instruct Q4_K_M (`~/models/Qwen3-Coder-30B-A3B-Instruct/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`, ~18 GB) – fast coding assistant at ~39 tok/s.

Scripts already exist inside llama.cpp for quickly starting each model:

```bash
cd /home/shan/llama.cpp
./start_gpt_oss.sh  # serves port 8002
./start_qwen_coder.sh  # serves port 8001
```

The optional `gpt-oss-proxy.py` (Flask) can strip channel tags and relay port 8003 → 8002.

## Repository Purpose

This repo tracks the higher-level orchestration work:

1. **Documentation** – hardware details, model configs, constraints, and lessons learned.
2. **Virtualization Experiments** – scripts/templates for Docker/VM-based deployment once designed.
3. **Automation Glue** – wrappers that make it trivial to choose one model, start it, and expose it over the LAN with sane defaults.

## Roadmap

- [x] Define requirements for “single-model” virtualization (resource limits, networking, volume mounts).
- [x] Prototype Docker image / compose file that wraps `llama-server` and model selection.
- [x] Add helper CLI to stop currently running model and start a new one safely (handles VRAM exclusivity).
- [ ] Integrate health monitoring (nvidia-smi snapshots, logs) for remote visibility.
- [ ] Build lightweight VM option if Docker access to the RTX 5090 ever becomes problematic.

## Docker Orchestration (MVP)

The first virtualization target is Docker + NVIDIA Container Toolkit. The repo now contains:

- `config/models.yaml` – manifest describing each model, GGUF relative path (relative to a host models directory), and the llama-server flags needed for that profile.
- `bin/local-llm` – helper CLI (`start`, `stop`, `status`, `list`) that reads the manifest, validates GGUF files, writes an `.env` file, and drives Docker Compose.
- `virtualization/docker/Dockerfile` – CUDA 12.6 based image that builds llama.cpp from source with `-DGGML_CUDA=ON`.
- `virtualization/docker/docker-compose.yaml` – single-service compose file that binds the selected model file read-only into the container, exposes the requested port to the LAN, and requests the GPU.
- `server/` – FastAPI orchestration service that exposes REST endpoints to list models, start/stop containers, proxy Harmony-formatted chat streams (via `/chat`), and report Docker status so future UIs can drive the stack without shell access.
- `web/` – Next.js control center that consumes the FastAPI service; includes model cards, start/stop buttons, and a basic chat console with Harmony reasoning-effort selector.

Usage (once Docker Desktop + nvidia-container-toolkit are configured inside WSL2):

```bash
# List configured models (names + ports)
./bin/local-llm list

# Start exactly one model (writes virtualization/docker/.env.runtime and runs docker compose up -d --build)
./bin/local-llm start gpt-oss-120b

# Show status or stop the running container
./bin/local-llm status
./bin/local-llm stop

# Optional: interact through the REST API instead of the CLI
pip install -r server/requirements.txt
uvicorn server.main:app --reload --port 8008
# -> GET http://localhost:8008/models, POST /start {"model":"gpt-oss-120b"}, POST /chat {...}, etc.

# Frontend (Next.js) control panel + Harmony chat console
cd web
npm install
npm run dev:frontend
# -> open http://localhost:3000 (or LAN IP) to manage models + chat
```

## Desktop Launcher / One-Click Start

For a single-click workflow there is now a stack launcher script plus desktop shortcuts:

- `bin/local-llm-launcher.sh` – orchestrates the Docker model container, the FastAPI control plane (`uvicorn` on port 8008), and the Next.js frontend (`npm run dev:frontend`).  
  - `bin/local-llm-launcher.sh start [--model <name>]` – stops any running model, starts the requested one, then boots the API and frontend in the background (logs under `.runtime/`).  
  - `bin/local-llm-launcher.sh stop` – stops the frontend, API, and Docker container.  
  - `bin/local-llm-launcher.sh status` – prints current state and log file locations.  
  - Set `LOCAL_LLM_DEFAULT_MODEL` to change the default model the launcher starts.
- `desktop/local-llm-start.desktop` and `desktop/local-llm-stop.desktop` – sample GNOME/KDE desktop entries wired to the launcher script.
- `bin/local-vllm` + `virtualization/vllm/` – optional vLLM-based stack for Qwen3 Coder 30B (FP8 by default). See `docs/vllm-code-completion.md` for local usage and `docs/vast-ai.md` for spinning it up on a Vast.ai server via `scripts/bootstrap-vast.sh`.

To install the desktop entries:

```bash
mkdir -p ~/.local/share/applications
cp desktop/local-llm-*.desktop ~/.local/share/applications/
chmod +x ~/.local/share/applications/local-llm-*.desktop
```

Most desktop environments will prompt you to “Trust” newly copied `.desktop` files the first time you launch them. Leave the `Terminal=true` setting enabled so you can view the launcher output (and any startup errors) when invoking from the desktop or application launcher.

### Windows desktop shortcuts

- `windows/local-llm-start.bat` and `windows/local-llm-stop.bat` wrap the launcher via `wsl.exe` so you can double-click from the Windows desktop/start menu. They automatically handle being run from a UNC path like `\\wsl.localhost\...` and force WSL to start in `/home/shan` (configurable) so you don’t hit the “Failed to translate Z:\…” error.
- Copy the `.bat` files anywhere on Windows (for example `%USERPROFILE%\Desktop`) and create standard Windows shortcuts pointing to them if you prefer icons with custom names/icons.
- By default they call your default WSL distro. If you need to target a specific distro name, set a user-level environment variable once:  
  `setx LOCAL_LLM_WSL_DISTRO "YourDistroName"`
- To change the WSL starting directory (before the script `cd`s into the repo), set  
  `setx LOCAL_LLM_WSL_HOME "/home/youruser"`
- The batch files pause after execution so you can read any errors; keep them in a visible terminal window to watch startup progress if desired.

### Remote/SSH access & vLLM option

- For the FP8 Qwen3 Coder 30B stack served via vLLM, follow `docs/vllm-code-completion.md` (local) or `docs/vast-ai.md` (cloud). Both serve an OpenAI-compatible endpoint on port 8004 using `bin/local-vllm`.
- Keep the GPU box on, run `bin/local-vllm start`, and SSH-tunnel (`ssh -L 8004:localhost:8004 gpu-box`) to reuse the model from any machine/IDE without exposing it publicly.

Environment variables:
- `LOCAL_LLM_MODELS_DIR` (optional) overrides the host models root if the GGUF files move elsewhere. Defaults to `/home/shan/models` from the manifest.
- `LLAMA_HOST` defaults to `0.0.0.0` so every LAN device can hit the container’s port.

All model-specific llama-server flags remain inside `config/models.yaml`, so swapping/adding new GGUF files is just an edit + commit without touching the docker artifacts.

## Contributing / Next Actions

1. Keep `/home/shan/llama.cpp` up to date (`git pull` and rebuild) to benefit from CUDA/MoE improvements.
2. Capture benchmark data as models/configurations change – update this README regularly.
3. Once Docker/VM design is ready, add the relevant files under a `virtualization/` directory and document usage.
4. Flesh out the new frontend chat console with proper message history, SSE parsing, Harmony reasoning-effort presets, and (eventually) GPT-OSS channel-tag cleanup plus authentication for remote access.

"Local LLM Tests" is meant to become the GitHub project that tracks all of the above so progress can be pushed/shareable. Push instructions are documented below.

## Quick start on this workstation

1. Open a terminal in `/home/shan/local-llm-tests`.
2. Start everything (model + API + web UI) with the launcher:  
   `bin/local-llm-launcher.sh start --model gpt-oss-120b`  
   - Use `--model qwen3-coder` to switch to coding mode.
   - The API listens on `8008`, the web UI on `3000`, and the model on its configured port (8002 for GPT-OSS, 8001 for Qwen).
3. Stop the stack when you are done:  
   `bin/local-llm-launcher.sh stop`

If you only want the model container (no API/UI):
- `bin/local-llm list` – see available models.
- `bin/local-llm start gpt-oss-120b` – start one model via Docker Compose.
- `bin/local-llm stop` – stop it.

If you want to run llama.cpp directly on the host (no Docker):
- `./start_gpt_oss.sh` (serves GPT-OSS on port 8002)
- `./start_qwen_coder.sh` (serves Qwen on port 8001)
  - These scripts default to `~/llama.cpp/build-gcc12/bin/llama-server` (CUDA 13, sm_89). Override with `LLAMA_SERVER_BIN=/path/to/llama-server` if you rebuild elsewhere. A symlink `./build -> ~/llama.cpp/build-gcc12` is present so older paths still resolve.

vLLM option for the FP8 Qwen checkpoint (OpenAI-compatible on port 8004):
- `bin/local-vllm start` to bring up vLLM with the settings in `virtualization/vllm/.env.vllm`.
- `bin/local-vllm stop` to tear it down.

## Use from VS Code “Build with Agent”

The llama.cpp endpoints are OpenAI-compatible, so point VS Code’s agent to the right base URL:

- GPT-OSS (llama.cpp): `http://localhost:8002/v1`, model name `gpt-oss-120b`
- Qwen (llama.cpp): `http://localhost:8001/v1`, model name `qwen3-coder`
- vLLM FP8 Qwen: `http://localhost:8004/v1`, model name `qwen3-coder-30b-a3b-fp8` (or whatever you set)

Most VS Code providers expect an API key; llama.cpp does not enforce one, so enter any placeholder (for example `local-dev`) if the UI requires it.

Steps in VS Code:
1. Start the model stack (launcher or direct script as above).
2. Open VS Code → open the “Build with Agent” prompt → choose a custom/OpenAI-compatible provider.
3. Paste the base URL and model name from the list above, add a dummy API key if prompted.
4. Run a quick message to confirm you see tokens streaming; if you get a connection error, verify the model process is running and the port is reachable.

## Publishing to GitHub

```bash
cd /home/shan/local-llm-tests
git add .
git commit -m "Initial project skeleton"
git remote add origin git@github.com:<your-username>/local-llm-tests.git
# or use https:// url if preferred
git push -u origin main
```

Update the remote URL once the GitHub repo is created. EOF
