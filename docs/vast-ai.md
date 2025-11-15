# Vast.ai Deployment Guide

These steps reproduce the exact environment we’ve been configuring manually. The instance must allow Docker with CAP_NET_ADMIN (choose a full VM/bare-metal listing on Vast, not a containerized one that blocks iptables).

## 1. Provision the machine

1. Pick a GPU configuration (e.g., 2 × RTX 5090) from the Vast.ai marketplace.
2. Enable SSH and expose the port you want the vLLM API to listen on (default 8000).
3. Upload your SSH public key in the instance “Keys” tab.

## 2. Clone the repo and run the bootstrap script

```bash
ssh root@<public-ip> -p <port>
git clone https://github.com/<your-user>/local-llm-tests.git
cd local-llm-tests
HF_TOKEN=hf_xxx \
VLLM_API_KEY=my-local-key \
TP_SIZE=2 \
./scripts/bootstrap-vast.sh
```

Environment variables:

- `HF_TOKEN` – Hugging Face access token with permission to read `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8`
- `VLLM_API_KEY` – API key your IDE/clients will present to `/v1`
- `TP_SIZE` – tensor parallel degree (2 for a dual-GPU box). Optional defaults to 2.
- Optional overrides: `MODEL_REPO`, `MODEL_SUBDIR`, `HOST_MODELS_DIR`, `MAX_MODEL_LEN`, `MAX_NUM_SEQS`.

The script performs:

1. `apt install` of Docker, fuse-overlayfs, python/pip, Hugging Face CLI dependencies, etc.
2. `python3 -m huggingface_hub.cli login` using your token (non-interactive).
3. Download of the FP8 checkpoint to `/root/models/<subdir>`.
4. Creation of `virtualization/vllm/.env.vllm` with sane defaults.
5. Starting `dockerd` in a “headless” mode (no iptables).
6. Running `bin/local-vllm start` to bring up the compose stack.

## 3. Verifying / operating

After bootstrap completes:

```bash
bin/local-vllm logs -f      # tail vLLM container logs
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer $VLLM_API_KEY"
```

If the VM reboots, restart Docker and the stack:

```bash
start-dockerd-headless.sh
bin/local-vllm start
```

## 4. Exposing the endpoint

- **Direct**: open port 8000 in Vast.ai’s UI, then hit `http://<host>:8000/v1`.
- **SSH tunnel**:
  ```bash
  ssh -L 8004:localhost:8000 root@<public-ip> -p <port>
  ```
  Use `http://localhost:8004/v1` from your IDE.

## 5. IDE configuration

Point Continue/Cursor/Zed/Windsurf to the remote base URL and use the `VLLM_API_KEY` you set above. Select `qwen3-coder-30b-a3b-fp8` (or any alias) as the model name.

## 6. Notes / troubleshooting

- The headless docker launch disables iptables. If your host allows iptables, you can start docker normally (systemd `docker.service`) instead of `start-dockerd-headless.sh`.
- `scripts/bootstrap-vast.sh` assumes a fresh VM. If you re-run it on an existing machine you may want to delete `/root/models/<subdir>` first.
- The FP8 checkpoint is ~40 GB – ensure the instance has enough disk before running the script.
