#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_ROOT="${HF_HOME:-/mnt/hf_cache}/hub"

if [[ ! -d "${CACHE_ROOT}" ]]; then
  echo "Creating cache directory at ${CACHE_ROOT}" >&2
  mkdir -p "${CACHE_ROOT}"
fi

MODELS=("$@")

if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODELS=(
    "meta-llama/Llama-2-7b-chat-hf"
    "meta-llama/Llama-2-13b-chat-hf"
    "meta-llama/Meta-Llama-3-8B-Instruct"
  )
  echo "No models provided; using default set:" >&2
  printf '  - %s\n' "${MODELS[@]}" >&2
fi

cd "${SCRIPT_DIR}"

for MODEL in "${MODELS[@]}"; do
  echo "\n=== Cleaning cached models at ${CACHE_ROOT} ==="
  rm -rf --verbose "${CACHE_ROOT}"/* || true

  echo "=== Running evaluation for ${MODEL} ==="
  python run_eval.py --model-name "${MODEL}"

done
