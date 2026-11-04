# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import torch
import torch.nn as nn
from quant_suite_core.native_presets import PrecisionType
from quant_suite_recipe.api.implementation import QuantizedTensor
from quant_suite_recipe.definition import QuantLinearRecipe
from quant_suite_recipe.implementation.quant_pipeline import Pipeline
from quant_suite_recipe.implementation.quantize_tensor import (
    _check_supported_tensor_dtype,
)


def maybe_quantize_weight(
    weight: torch.Tensor,
    weight_pipeline: nn.Module,
    init_quantized_weight: bool,
) -> QuantizedTensor:
    """Apply a weight pipeline when pre-quantized weight initialization is enabled."""
    if not init_quantized_weight:
        return QuantizedTensor(tensor=weight)

    _check_supported_tensor_dtype(weight)
    result = weight_pipeline(weight)
    assert result.tensor is not None, "Quantized tensor should not be None"
    if result.precision is not None:
        assert result.scale is not None, "Native quantization must have scale"
    return result


def quantized_matmul(
    quantized_x: QuantizedTensor,
    quantized_w: QuantizedTensor,
    quantized_bias: QuantizedTensor | None,
    recipe: QuantLinearRecipe,
) -> torch.Tensor:
    """Multiply tensors for a Linear recipe."""
    xq = quantized_x.tensor
    wq = quantized_w.tensor
    assert xq is not None, "quantized_x.tensor must be provided"
    assert wq is not None, "quantized_w.tensor must be provided"
    if quantized_bias is not None and quantized_bias.precision is not None:
        raise NotImplementedError(
            "Native quantization of a Linear bias is not supported"
        )

    x_precision = quantized_x.precision
    w_precision = quantized_w.precision
    if x_precision is None and w_precision is None:
        biasq = quantized_bias.tensor if quantized_bias is not None else None
        matmul_out = torch.matmul(xq, wq.t())
        if biasq is not None:
            matmul_out = matmul_out + biasq
        return matmul_out

    if x_precision != w_precision:
        raise ValueError(
            "Native quantized matmul requires activation and weight to use the "
            f"same native precision, got {x_precision} and {w_precision}"
        )

    x_scale = quantized_x.scale
    w_scale = quantized_w.scale
    assert x_scale is not None, "x_scale must be provided for native quantization"
    assert w_scale is not None, "w_scale must be provided for native quantization"

    import mslk.gemm  # noqa: F401
    import mslk.quantize  # noqa: F401

    if x_precision == PrecisionType.MX4_32:
        matmul_out = torch.ops.mslk.f4f4bf16(xq, wq, x_scale, w_scale, None, None, 32)
    elif x_precision == PrecisionType.MX4_16:
        matmul_out = torch.ops.mslk.f4f4bf16(xq, wq, x_scale, w_scale, None, None, 16)
    elif x_precision == PrecisionType.NVFP4:
        x_global_scale = quantized_x.global_scale
        w_global_scale = quantized_w.global_scale
        assert x_global_scale is not None, (
            f"x_global_scale must be provided for {x_precision.value}"
        )
        assert w_global_scale is not None, (
            f"w_global_scale must be provided for {x_precision.value}"
        )
        global_scale = 1 / (x_global_scale * w_global_scale)
        matmul_out = torch.ops.mslk.f4f4bf16(
            xq, wq, x_scale, w_scale, None, global_scale
        )
    else:
        raise NotImplementedError(f"Precision {x_precision} is not supported")

    if quantized_bias is not None and quantized_bias.tensor is not None:
        matmul_out = matmul_out + quantized_bias.tensor

    return matmul_out


class QuantizedLinear(nn.Module):
    """Prepared quantization behavior owned by one Linear module."""

    recipe: QuantLinearRecipe
    activation_pipeline: Pipeline
    weight_pipeline: Pipeline
    bias_pipeline: Pipeline

    def __init__(
        self,
        recipe: QuantLinearRecipe,
        activation_pipeline: Pipeline,
        weight_pipeline: Pipeline,
        bias_pipeline: Pipeline,
    ) -> None:
        super().__init__()
        self.recipe = recipe
        self.activation_pipeline = activation_pipeline
        self.weight_pipeline = weight_pipeline
        self.bias_pipeline = bias_pipeline

    def quantize_weight(self, weight: torch.Tensor) -> QuantizedTensor:
        """Prepare a Linear weight according to this recipe."""
        return maybe_quantize_weight(
            weight,
            self.weight_pipeline,
            self.recipe.init_quantized_weight,
        )

    def forward(
        self,
        x: torch.Tensor,
        weight: QuantizedTensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply the prepared pipelines and selected matrix multiplication."""
        quantized_x = self.activation_pipeline(x)
        quantized_bias: QuantizedTensor | None = (
            self.bias_pipeline(bias) if bias is not None else None
        )
        return self.matmul(
            quantized_x,
            weight,
            quantized_bias,
        )

    def matmul(
        self,
        quantized_x: QuantizedTensor,
        quantized_w: QuantizedTensor,
        quantized_bias: QuantizedTensor | None,
    ) -> torch.Tensor:
        return quantized_matmul(
            quantized_x,
            quantized_w,
            quantized_bias,
            self.recipe,
        )


__all__ = [
    "QuantizedLinear",
    "maybe_quantize_weight",
    "quantized_matmul",
]
