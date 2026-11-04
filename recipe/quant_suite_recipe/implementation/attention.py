# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from quant_suite_recipe.api.implementation import QuantizedTensor
from quant_suite_recipe.definition import QuantAttentionRecipe
from quant_suite_recipe.implementation.quant_pipeline import Pipeline


def _torch_bmm(
    left: QuantizedTensor,
    right: QuantizedTensor,
    *,
    transpose_right: bool = False,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    assert left.tensor is not None
    assert right.tensor is not None
    right_tensor = right.tensor.transpose(-1, -2) if transpose_right else right.tensor
    if out_dtype is not None and left.tensor.device.type == "cuda":
        return torch.bmm(left.tensor, right_tensor, out_dtype=out_dtype)
    result = torch.bmm(left.tensor, right_tensor)
    return result.to(out_dtype) if out_dtype is not None else result


class QuantizedAttention(nn.Module):
    """Prepared quantization behavior owned by one Attention module."""

    supports_multi_request_bmm: bool = True
    recipe: QuantAttentionRecipe
    q_pipeline: Pipeline
    k_pipeline: Pipeline
    v_pipeline: Pipeline
    attn_weights_pipeline: Pipeline
    recurrent_state_pipeline: Pipeline
    _bmm: Callable[..., torch.Tensor]

    def __init__(
        self,
        recipe: QuantAttentionRecipe,
        q_pipeline: Pipeline,
        k_pipeline: Pipeline,
        v_pipeline: Pipeline,
        attn_weights_pipeline: Pipeline,
        recurrent_state_pipeline: Pipeline,
    ) -> None:
        super().__init__()
        self.recipe = recipe
        self.q_pipeline = q_pipeline
        self.k_pipeline = k_pipeline
        self.v_pipeline = v_pipeline
        self.attn_weights_pipeline = attn_weights_pipeline
        self.recurrent_state_pipeline = recurrent_state_pipeline
        self._bmm = _torch_bmm

    def quantize_q(self, tensor: torch.Tensor) -> QuantizedTensor:
        return self.q_pipeline(tensor)

    def quantize_k(self, tensor: torch.Tensor) -> QuantizedTensor:
        return self.k_pipeline(tensor)

    def quantize_v(self, tensor: torch.Tensor) -> QuantizedTensor:
        return self.v_pipeline(tensor)

    def quantize_attn_weights(self, tensor: torch.Tensor) -> QuantizedTensor:
        return self.attn_weights_pipeline(tensor)

    def quantize_recurrent_state(self, tensor: torch.Tensor) -> QuantizedTensor:
        return self.recurrent_state_pipeline(tensor)

    def bmm(
        self,
        left: QuantizedTensor,
        right: QuantizedTensor,
        *,
        transpose_right: bool = False,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Multiply prepared tensors using this runtime's attention operation."""
        return self._bmm(
            left,
            right,
            transpose_right=transpose_right,
            out_dtype=out_dtype,
        )


__all__ = ["QuantizedAttention"]
