#!/usr/bin/env bash
set -euo pipefail

OPENAI_API_BASE="${1:?usage: run_evalplus.sh <openai_api_base> <model_name> [humaneval|mbpp] }"
MODEL_NAME="${2:?usage: run_evalplus.sh <openai_api_base> <model_name> [humaneval|mbpp] }"
BENCHMARK="${3:-humaneval}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY must be set for EvalPlus OpenAI-compatible mode" >&2
  exit 1
fi

if [[ "${BENCHMARK}" == "humaneval" ]]; then
  evalplus.evaluate \
    --dataset humaneval \
    --backend openai \
    --model "${MODEL_NAME}" \
    --base-url "${OPENAI_API_BASE}"
else
  evalplus.evaluate \
    --dataset mbpp \
    --backend openai \
    --model "${MODEL_NAME}" \
    --base-url "${OPENAI_API_BASE}"
fi
