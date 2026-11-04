#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

if (( $# < 2 || $# > 3 )); then
  echo "usage: $0 {chat|completions} OUTPUT_DIR [TOKENIZER]" >&2
  exit 2
fi

api_mode=$1
output_dir=$2
tokenizer=${3:-}

LM_EVAL_BIN=${LM_EVAL_BIN:-lm_eval}
BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-quant-suite-validation}
NUM_CONCURRENT=${NUM_CONCURRENT:-8}

case "$api_mode" in
  chat)
    model_adapter=local-chat-completions
    endpoint=$BASE_URL/v1/chat/completions
    model_args="model=$SERVED_MODEL_NAME,base_url=$endpoint,num_concurrent=$NUM_CONCURRENT,max_retries=3,timeout=600,seed=1234"
    prompt_args=(--apply_chat_template)
    ;;
  completions)
    if [[ -z "$tokenizer" ]]; then
      echo "TOKENIZER is required for completions" >&2
      exit 2
    fi
    model_adapter=local-completions
    endpoint=$BASE_URL/v1/completions
    model_args="model=$SERVED_MODEL_NAME,base_url=$endpoint,tokenizer=$tokenizer,tokenizer_backend=huggingface,trust_remote_code=True,tokenized_requests=False,num_concurrent=$NUM_CONCURRENT,max_retries=3,timeout=600,seed=1234,max_gen_toks=256"
    prompt_args=()
    ;;
  *)
    echo "API mode must be chat or completions" >&2
    exit 2
    ;;
esac

mkdir -p "$output_dir"

exec "$LM_EVAL_BIN" run \
  --model "$model_adapter" \
  --model_args "$model_args" \
  --tasks gsm8k \
  "${prompt_args[@]}" \
  --batch_size 1 \
  --seed 0,1234,1234,1234 \
  --output_path "$output_dir" \
  --log_samples
