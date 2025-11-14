# Virtualization Plan

Objective: provide a single-control-plane experience for spinning up exactly one llama.cpp model at a time, reachable from any device on the LAN via HTTP/Web UI/API. Constraints: 32 GB VRAM → only one large model can be resident; CPU/RAM remain plentiful. This document captures the decision matrix for Docker, lightweight VM, or host-native orchestration.

## Requirements

1. **Single active model** – guard rails prevent two heavy models from launching simultaneously.
2. **Model picker** – operator can choose from `GPT-OSS-120B F16` or `Qwen3-Coder-30B` (extendable list) at launch time.
3. **Network exposure** – container/VM must bind to host `0.0.0.0` on a configurable port so other LAN devices can browse the llama.cpp web UI and API.
4. **Resource visibility** – easy hooks for `nvidia-smi`, logs, and CPU/RAM monitoring.
5. **Fast iteration** – minimal overhead when switching models; ideally reuses pre-downloaded `.gguf` files mounted from host.

## Option A: Docker + docker-compose

- Base image: Ubuntu 22.04 + CUDA runtime 12.6 + llama.cpp compiled with CUDA + FlashAttention.
- Bind mount: `~/models:/models` and `~/llama.cpp:/workspace/llama.cpp` (or bake binary into image).
- Entry point script accepts `MODEL_NAME` env and maps to the GGUF path plus the right `llama-server` flags.
- Compose file exposes port 800X → host, sets `NVIDIA_VISIBLE_DEVICES=all`, and mounts GPU via `--gpus all`.
- Pros: fast to spin down/up; packaging reproducible.
- Cons: needs Docker + nvidia-container-toolkit; WSL2 adds extra setup but manageable.

### Tasks

1. Write Dockerfile that copies a prebuilt llama.cpp binary or builds it during image creation.
2. Create `scripts/docker/start-model.sh` to parse `$MODEL` (gpt-oss or qwen) and exec the appropriate server flags.
3. Provide `docker-compose.yml` with a single service and instructions for `MODEL=gpt-oss docker compose up`.
4. Add helper script to stop running container before launching a different one.

## Option B: Lightweight VM (multipass/WSL distro)

- Provision a dedicated Ubuntu VM with GPU pass-through (via WSL2, Hyper-V, or Proxmox).
- Use cloud-init style scripts to fetch llama.cpp, mount model share via SMB/NFS, and run systemd services per model choice.
- Pros: isolates environment; good if Docker GPU access is problematic.
- Cons: heavier to manage; start/stop times longer.

### Tasks

1. Research GPU exposure method for chosen hypervisor.
2. Automate provisioning with Ansible or cloud-init.
3. Expose VM port (8001/8002) via host-only or bridged network.

## MVP Decision

Start with Docker because llama.cpp already runs smoothly under WSL2 with CUDA. Build the container to mimic the current host setup and rely on environment variables/compose overrides to choose the model. Once that flow works, re-evaluate whether a full VM provides additional value.

### Current Implementation Snapshot

- `config/models.yaml` tracks available models (relative GGUF paths, ports, llama-server arguments).
- `bin/local-llm` CLI orchestrates Docker Compose by writing `virtualization/docker/.env.runtime` and running `docker compose up|down`.
- `virtualization/docker/Dockerfile` builds a CUDA-enabled llama.cpp binary inside the container; compose mounts the host `~/models` directory read-only.
- `server/main.py` exposes REST endpoints (FastAPI) for `/models`, `/start`, `/stop`, `/restart`, and `/status`, so future tooling can steer the same Docker workflow without shelling out.
- Port exposure uses the per-model `port` field so LAN devices can browse either GPT-OSS (8002) or Qwen (8001) depending on what is active.

## Next Steps

1. Add health checks/metrics export (nvidia-smi sampler, llama-server logs) that can be scraped outside the container.
2. Provide optional auth/reverse proxy layer for LAN exposure (e.g., Caddy or Traefik sidecar).
3. Prototype lightweight VM alternative for environments where Docker GPU pass-through is unavailable or undesirable.
4. Harden the new Next.js chat console: stream assistant output via SSE, persist conversations, surface Harmony reasoning effort presets, and clean GPT-OSS channel tags client-side before exposing the UI broadly.
