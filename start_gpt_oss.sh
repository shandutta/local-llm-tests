#!/bin/bash

MODEL_PATH=~/models/gpt-oss-120b/gpt-oss-120b-F16.gguf

./build/bin/llama-server \
  --model "$MODEL_PATH" \
  --alias "gpt-oss-120b" \
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
