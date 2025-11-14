# Local LLM Tests

Local LLM Tests is a working log and automation hub for my RTX 5090 based llama.cpp deployments. The goal is to consolidate everything that makes it easy to spin up one high-quality model at a time, expose it to the LAN, and prepare for containerized/virtualized hosting when needed.

## Current Environment Snapshot

- **Host**: AMD Ryzen 9950X3D, 96 GB DDR5 (72 GB allocated to WSL2), Ubuntu 22.04 on WSL2.
- **GPU**: NVIDIA RTX 5090 (32 GB VRAM) – CUDA 12.6 installed, llama.cpp built with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=\"89\"`.
- **Inference stack**: `/home/shan/llama.cpp` build at version `7045 (97d511721)` running `llama-server`.
- **Models on disk**:
  - GPT-OSS-120B F16 (`~/models/gpt-oss-120b/gpt-oss-120b-F16.gguf`, ~61 GB) – premium reasoning, tuned with `--n-cpu-moe 25` for ~21.6 tok/s.
  - Qwen3-Coder-30B-A3B-Instruct Q4_K_M (`~/models/qwen3-coder-32b/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`, ~18 GB) – fast coding assistant at ~39 tok/s.

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

- [ ] Define requirements for “single-model” virtualization (resource limits, networking, volume mounts).
- [ ] Prototype Docker image / compose file that wraps `llama-server` and model selection.
- [ ] Add helper CLI to stop currently running model and start a new one safely (handles VRAM exclusivity).
- [ ] Integrate health monitoring (nvidia-smi snapshots, logs) for remote visibility.

## Contributing / Next Actions

1. Keep `/home/shan/llama.cpp` up to date (`git pull` and rebuild) to benefit from CUDA/MoE improvements.
2. Capture benchmark data as models/configurations change – update this README regularly.
3. Once Docker/VM design is ready, add the relevant files under a `virtualization/` directory and document usage.

"Local LLM Tests" is meant to become the GitHub project that tracks all of the above so progress can be pushed/shareable. Push instructions are documented below.

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
