# vLLM + Qwen3 Coder 30B FP4 Stack

To keep VRAM usage manageable on 32 GB GPUs, this variant uses NVIDIA’s FP4 (W4A4) checkpoint [`NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4`](https://huggingface.co/NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4). FP4 still runs through [vLLM](https://github.com/vllm-project/vllm) and exposes the same OpenAI-compatible endpoint, but frees enough memory to keep longer contexts and larger KV caches compared with FP8.

## 1. Download the FP4 checkpoint

1. Install or update the Hugging Face CLI (`hf`):
   ```bash
   pip install --upgrade "huggingface_hub" --user
   ```
2. Authenticate so the CLI can access gated artifacts:
   ```bash
   hf auth login
   ```
3. Pull the FP4 weights and tokenizer somewhere under your models directory (default `~/models`). Either clone via Git LFS:
   ```bash
   git lfs install
   git clone https://huggingface.co/NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4 \
     ~/models/nvfp4-qwen3-coder-30b-a3b
   ```
   or use the CLI (`HF_HUB_DISABLE_SYMLINKS=1` ensures regular files instead of symlinks):
   ```bash
   HF_HUB_DISABLE_SYMLINKS=1 hf download NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4 \
     --repo-type model \
     --local-dir ~/models/nvfp4-qwen3-coder-30b-a3b
   ```
   If you already cloned the FP8 variant earlier, you can safely delete that folder after copying any notes or custom configs over.

The vLLM image loads the raw `safetensors` and tokenizer JSON instead of GGUF files, so no conversion is required. Any other FP4 or FP8 checkpoint will work as long as `MODEL_PATH` and `TOKENIZER_PATH` point at the directory with `config.json`, tokenizer files, and `model.safetensors`.

## 2. Configure the vLLM compose stack

1. Copy the env template and edit it:
   ```bash
   cp virtualization/vllm/.env.example virtualization/vllm/.env.vllm
   ```
2. Update `.env.vllm`:
   - `HOST_MODELS_DIR` → where you stored the FP4 files (e.g., `~/models/nvfp4-qwen3-coder-30b-a3b`).
   - `MODEL_PATH` / `TOKENIZER_PATH` → container paths (default `/models/...`).
   - `HF_TOKEN` → optional if you need vLLM to pull missing assets.
   - `VLLM_API_KEY` → arbitrary key required by the OpenAI endpoint (use the same key in your IDE).
   - `VLLM_PORT` → host port that forwards to vLLM’s `8000`.
   - Adjust `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, or `GPU_MEMORY_UTILIZATION` if you want to trade latency vs. throughput. On 32 GB GPUs, a good starting point is `MAX_MODEL_LEN=8192`, `MAX_NUM_SEQS=96`, `GPU_MEMORY_UTILIZATION≈0.92`.

## 3. Start/stop the service

### Option A – docker compose

Use the original helper script when you want the model to live alongside the FastAPI + Next.js stack:

```bash
# Start vLLM + expose OpenAI-compatible endpoint on localhost:8004
bin/local-vllm start

# Watch logs or stop later
bin/local-vllm logs
bin/local-vllm stop
```

The container uses NVIDIA Container Toolkit, so Docker must be able to see your GPU (`nvidia-smi` should work inside any other CUDA container first).

### Option B – Docker Model Runner

If you prefer Docker’s experimental `docker model` workflow (nice for bundling with other compose stacks or CI jobs), use the second helper:

```bash
# Start FP4 Qwen3 via docker model run
bin/local-vllm-docker-model start

# Check status / follow logs / stop
bin/local-vllm-docker-model status
bin/local-vllm-docker-model logs
bin/local-vllm-docker-model stop
```

Under the hood, it runs:

```bash
docker model run \
  --name nvfp4-qwen3-coder-30b-a3b \
  --gpus all \
  --port 8004:8000 \
  --env HF_TOKEN=... \
  --mount type=bind,src=$HOME/models,target=/models,readonly \
  vllm/vllm-openai:latest \
  --model /models/nvfp4-qwen3-coder-30b-a3b \
  --tokenizer /models/nvfp4-qwen3-coder-30b-a3b \
  --max-model-len 8192 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 96 \
  --dtype auto \
  --enforce-eager \
  --trust-remote-code \
  --api-key local-vllm-dev
```

## 4. Connect IDEs / code-completion clients

vLLM’s OpenAI server lives at `http://<host>:<VLLM_PORT>/v1` and requires the API key you set in `.env.vllm`.

- **Continue / VS Code** – add a new OpenAI provider that points to `http://<lan-ip>:8004/v1`, set the API key, and pick `nvfp4-qwen3-coder-30b-a3b` (or any alias you prefer). Continue can use this endpoint for both chat and inline completions.
- **Cursor, Zed, Windsurf, etc.** – most IDEs that support custom OpenAI endpoints accept the same URL/key combo.
- **SSH from anywhere** – keep the GPU box on and expose port `8004`. For extra safety, tunnel through SSH instead of opening a firewall hole:
  ```bash
  ssh -L 8004:localhost:8004 gpu-box.example.com
  # now hit http://localhost:8004/v1 from your laptop
  ```

Because vLLM batches and streams requests, you can run many concurrent completion probes (the `MAX_NUM_SEQS` and `max_model_len` flags in `.env.vllm` directly influence concurrency and latency).

## 5. Using alongside the existing llama.cpp stack

Nothing in the original `bin/local-llm` flow changes. You can:

1. Keep GPT‑OSS or GGUF Qwen in `local-llm`.
2. Run `bin/local-vllm start` (or `bin/local-vllm-docker-model start`) for the FP4/vLLM stack when you need IDE completions.
3. Point Continue/VS Code to whichever endpoint you need (Harmony proxy on FastAPI vs. vLLM OpenAI).

This separation also makes it easy to leave the vLLM container running on the GPU workstation, SSH into it from your Mac/WSL machine, and keep the browser-based control center for llama.cpp duties.
