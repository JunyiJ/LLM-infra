#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:?usage: serve_vllm.sh <model_path> [served_model_name] }"
SERVED_MODEL_NAME="${2:-qwen25-1p5b-apps-full-sft}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
DTYPE="${DTYPE:-auto}"
API_KEY="${API_KEY:-local-dev}"
GENERATION_CONFIG="${GENERATION_CONFIG:-vllm}"

vllm serve "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype "${DTYPE}" \
  --api-key "${API_KEY}" \
  --generation-config "${GENERATION_CONFIG}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --served-model-name "${SERVED_MODEL_NAME}"
