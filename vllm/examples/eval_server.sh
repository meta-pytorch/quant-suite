#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

if (( $# != 3 )); then
  echo "usage: $0 MODEL_LABEL RECIPE_LABEL {pilot|full}" >&2
  exit 2
fi

model_label=$1
recipe_label=$2
run_size=$3

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LM_EVAL_BIN=${LM_EVAL_BIN:-lm-eval}
BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-quant-suite-validation}
NUM_CONCURRENT=${NUM_CONCURRENT:-8}
PILOT_LIMIT=${PILOT_LIMIT:-100}
RESULTS_ROOT=${RESULTS_ROOT:-$script_dir/results/runs}
API_MODE=${API_MODE:-chat}
TOKENIZER=${TOKENIZER:-}
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
output_root="$RESULTS_ROOT/$model_label/$recipe_label/$run_size/$timestamp"
model_output_path="$output_root/$SERVED_MODEL_NAME"

case "$run_size" in
  pilot)
    tasks=gsm8k,mmlu_pro_biology
    limit_args=(--limit "$PILOT_LIMIT")
    ;;
  full)
    tasks=gsm8k,mmlu_pro
    limit_args=()
    ;;
  *)
    echo "run size must be pilot or full" >&2
    exit 2
    ;;
esac

case "$API_MODE" in
  chat)
    model_adapter=local-chat-completions
    endpoint=$BASE_URL/v1/chat/completions
    prompt_args=(--apply_chat_template)
    tokenizer_args=
    ;;
  completions)
    if [[ -z "$TOKENIZER" ]]; then
      echo "TOKENIZER is required when API_MODE=completions" >&2
      exit 2
    fi
    model_adapter=local-completions
    endpoint=$BASE_URL/v1/completions
    prompt_args=()
    tokenizer_args=",tokenizer=$TOKENIZER,tokenizer_backend=huggingface,tokenized_requests=False"
    ;;
  *)
    echo "API_MODE must be chat or completions" >&2
    exit 2
    ;;
esac

mkdir -p "$output_root"

"$LM_EVAL_BIN" run \
  --model "$model_adapter" \
  --model_args "model=$SERVED_MODEL_NAME,base_url=$endpoint,num_concurrent=$NUM_CONCURRENT,max_retries=3,timeout=600,seed=1234$tokenizer_args" \
  --tasks "$tasks" \
  "${prompt_args[@]}" \
  --batch_size 1 \
  --seed 0,1234,1234,1234 \
  --output_path "$output_root" \
  --log_samples \
  "${limit_args[@]}"

echo "Results: $model_output_path"
