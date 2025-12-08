#!/bin/bash

MODEL_PATH=~/models/Qwen3-Coder-30B-A3B-Instruct/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
LLAMA_SERVER_BIN=${LLAMA_SERVER_BIN:-"$HOME/llama.cpp/build-gcc12/bin/llama-server"}

"$LLAMA_SERVER_BIN" \
  --model "$MODEL_PATH" \
  --alias "qwen3-coder" \
  --jinja \
  --chat-template "chatml" \
  --n-gpu-layers 999 \
  --ctx-size 32768 \
  --threads 14 \
  --threads-batch 16 \
  --temp 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --repeat-penalty 1.05 \
  --port 8001 \
  --host 0.0.0.0 \
  --parallel 4 \
  --cont-batching \
  --flash-attn on
