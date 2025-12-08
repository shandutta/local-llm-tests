#!/bin/bash

MODEL_PATH=~/models/gpt-oss-120b/gpt-oss-120b-F16.gguf
LLAMA_SERVER_BIN=${LLAMA_SERVER_BIN:-"$HOME/llama.cpp/build-gcc12/bin/llama-server"}

"$LLAMA_SERVER_BIN" \
  --model "$MODEL_PATH" \
  --alias "gpt-oss-120b" \
  --jinja \
  --chat-template "chatml" \
  --n-gpu-layers 99 \
  --n-cpu-moe 25 \
  --ctx-size 131072 \
  --threads 14 \
  --threads-batch 16 \
  --temp 1.0 \
  --top-p 1.0 \
  --top-k 0 \
  --min-p 0.0 \
  --port 8002 \
  --host 0.0.0.0 \
  --parallel 4 \
  --cont-batching \
  --flash-attn on
