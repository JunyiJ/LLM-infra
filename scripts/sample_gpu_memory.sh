#!/usr/bin/env bash
set -euo pipefail

OUTPUT_PATH="${1:?usage: sample_gpu_memory.sh <output_csv> [interval_sec] }"
INTERVAL_SEC="${2:-1}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")"
echo "timestamp,index,name,memory_used_mb,memory_total_mb,utilization_gpu" > "${OUTPUT_PATH}"

while true; do
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi \
    --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits \
    | awk -v ts="${timestamp}" -F', ' '{print ts "," $1 "," $2 "," $3 "," $4 "," $5}' \
    >> "${OUTPUT_PATH}"
  sleep "${INTERVAL_SEC}"
done
