#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:?usage: serve_vllm.sh <model_path> [served_model_name] }"
SERVED_MODEL_NAME="${2:-qwen25-1p5b-apps-full-sft}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --port "${PORT}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}"

