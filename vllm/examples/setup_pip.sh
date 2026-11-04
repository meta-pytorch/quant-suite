#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
QUANT_SUITE_ROOT=${QUANT_SUITE_ROOT:-$(cd -- "$script_dir/../.." && pwd)}
ENV_DIR=${ENV_DIR:-$QUANT_SUITE_ROOT/.venv-vllm-pip}
PYTHON_BIN=${PYTHON_BIN:-python3}
VLLM_SOURCE=${VLLM_SOURCE:-}
LM_EVAL_SOURCE=${LM_EVAL_SOURCE:-}

if [[ -e "$ENV_DIR" ]]; then
  echo "refusing to replace existing environment: $ENV_DIR" >&2
  exit 2
fi

"$PYTHON_BIN" -m venv "$ENV_DIR"
env_python="$ENV_DIR/bin/python"
"$env_python" -m pip install --upgrade pip

if [[ -n "$VLLM_SOURCE" ]]; then
  VLLM_USE_PRECOMPILED=1 "$env_python" -m pip install --editable "$VLLM_SOURCE"
else
  "$env_python" -m pip install 'vllm==0.30.0'
fi

if [[ -n "$LM_EVAL_SOURCE" ]]; then
  "$env_python" -m pip install --editable "${LM_EVAL_SOURCE}[api]"
else
  "$env_python" -m pip install 'lm-eval[api]==0.4.13'
fi

"$env_python" -m pip install \
  --editable "$QUANT_SUITE_ROOT/core" \
  --editable "$QUANT_SUITE_ROOT/recipe" \
  --editable "$QUANT_SUITE_ROOT/vllm"

echo "Activate with: source $ENV_DIR/bin/activate"
