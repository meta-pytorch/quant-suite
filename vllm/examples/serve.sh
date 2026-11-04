#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

if (( $# < 3 )); then
  echo "usage: $0 MODEL RECIPE {mha|gqa|mla} [vLLM arguments...]" >&2
  exit 2
fi

model=$1
recipe=$2
attention_kind=$3
shift 3

if [[ ! -f "$recipe" || ! -r "$recipe" ]]; then
  echo "recipe must be a readable file: $recipe" >&2
  exit 2
fi
recipe=$(realpath "$recipe")

PYTHON_BIN=${PYTHON_BIN:-python3}
VLLM_BIN=${VLLM_BIN:-vllm}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-quant-suite-validation}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.5}
SERVER_SEED=${SERVER_SEED:-1234}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-0}
ENFORCE_EAGER=${ENFORCE_EAGER:-1}
VLLM_TP_TORCH_FALLBACK=${VLLM_TP_TORCH_FALLBACK:-0}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-CUSTOM}

extra_args=()
if [[ "$TRUST_REMOTE_CODE" == 1 ]]; then
  extra_args+=(--trust-remote-code)
fi
if [[ "$ENFORCE_EAGER" == 1 ]]; then
  extra_args+=(--enforce-eager)
fi
if [[ "$VLLM_TP_TORCH_FALLBACK" == 1 ]]; then
  export VLLM_DISABLE_PYNCCL=1
  export VLLM_ALLREDUCE_USE_SYMM_MEM=0
  export VLLM_ALLREDUCE_USE_FLASHINFER=0
  extra_args+=(--disable-custom-all-reduce)
fi

case "$attention_kind" in
  mha|gqa)
    MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
    unset QUANT_SUITE_ATTN_BACKEND
    ;;
  mla)
    MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
    export QUANT_SUITE_ATTN_BACKEND=CUSTOM_MLA
    ;;
  *)
    echo "attention kind must be mha, gqa, or mla" >&2
    exit 2
    ;;
esac

hf_overrides=$(
  "$PYTHON_BIN" -c \
    'import json, sys; print(json.dumps({"quantization_config_file": sys.argv[1]}))' \
    "$recipe"
)

exec "$VLLM_BIN" serve "$model" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --quantization quant_suite \
  --hf-overrides "$hf_overrides" \
  --attention-backend "$ATTENTION_BACKEND" \
  --dtype bfloat16 \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --seed "$SERVER_SEED" \
  --host "$HOST" \
  --port "$PORT" \
  "${extra_args[@]}" \
  "$@"
