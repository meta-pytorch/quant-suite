#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
QUANT_SUITE_ROOT=${QUANT_SUITE_ROOT:-$(cd -- "$script_dir/../.." && pwd)}
ENV_DIR=${ENV_DIR:-$QUANT_SUITE_ROOT/.venv-vllm}
PYTHON_BIN=${PYTHON_BIN:-python3}
VLLM_SOURCE=${VLLM_SOURCE:-}
LM_EVAL_SOURCE=${LM_EVAL_SOURCE:-}

if [[ -e "$ENV_DIR" ]]; then
  echo "refusing to replace existing environment: $ENV_DIR" >&2
  exit 2
fi

uv venv --python "$PYTHON_BIN" "$ENV_DIR"
env_python="$ENV_DIR/bin/python"

if [[ -n "$VLLM_SOURCE" ]]; then
  VLLM_USE_PRECOMPILED=1 uv pip install \
    --python "$env_python" \
    --editable "$VLLM_SOURCE" \
    --torch-backend=auto
else
  uv pip install \
    --python "$env_python" \
    'vllm==0.30.0' \
    --torch-backend=auto
fi

if [[ -n "$LM_EVAL_SOURCE" ]]; then
  uv pip install --python "$env_python" --editable "${LM_EVAL_SOURCE}[api]"
else
  uv pip install --python "$env_python" 'lm-eval[api]==0.4.13'
fi

uv pip install --python "$env_python" \
  --editable "$QUANT_SUITE_ROOT/core" \
  --editable "$QUANT_SUITE_ROOT/recipe" \
  --editable "$QUANT_SUITE_ROOT/vllm"

echo "Activate with: source $ENV_DIR/bin/activate"
