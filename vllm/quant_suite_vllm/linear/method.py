# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from typing import Any

import torch
from quant_suite_recipe.api.implementation import QuantizedTensor
from quant_suite_recipe.definition import (
    NativeQuantRecipe,
    PipelineRecipe,
    QuantLinearRecipe,
    QuantRecipe,
)
from quant_suite_recipe.implementation.linear import QuantizedLinear
from quant_suite_recipe.runtime_provider import QUANT_RECIPE_RUNTIME
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.utils import set_weight_attrs


logger: logging.Logger = logging.getLogger(__name__)


def _native_stage(pipeline: PipelineRecipe) -> NativeQuantRecipe | None:
    for stage in pipeline:
        if isinstance(stage, NativeQuantRecipe):
            return stage
    return None


def _validate_linear_recipe(recipe: QuantLinearRecipe) -> None:
    if recipe.bias.has_native_quant:
        raise NotImplementedError(
            "QuantSuite vLLM Linear does not support native bias quantization"
        )
    activation_stage = _native_stage(recipe.activation)
    weight_stage = (
        _native_stage(recipe.weight) if recipe.init_quantized_weight else None
    )
    activation_precision = (
        activation_stage.precision if activation_stage is not None else None
    )
    weight_precision = weight_stage.precision if weight_stage is not None else None
    if activation_precision != weight_precision:
        raise ValueError(
            "Native quantized matmul requires activation and weight to use the "
            f"same native precision, got {activation_precision} and "
            f"{weight_precision} (the weight is quantized only when "
            "init_quantized_weight is true)"
        )


class QuantSuiteLinearMethod(LinearMethodBase):
    """vLLM Linear method backed by a QuantSuite recipe."""

    def __init__(
        self,
        quant_recipe: QuantRecipe,
        layer_name: str,
    ) -> None:
        self.layer_name = layer_name
        self.linear_recipe = quant_recipe.get_linear_recipe(layer_name)
        _validate_linear_recipe(self.linear_recipe)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        weight = torch.nn.Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        quantized_weight = QuantizedTensor.from_registered_parameters(layer, "weight")
        dynamic_layer: Any = layer
        quantized_linear: QuantizedLinear = dynamic_layer.quantized_linear
        return quantized_linear(x, quantized_weight, bias)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        quantized_linear = QUANT_RECIPE_RUNTIME.prepare_linear(self.linear_recipe)
        dynamic_layer: Any = layer
        dynamic_layer.quantized_linear = quantized_linear

        if self.linear_recipe.weight.is_noop():
            logger.info(
                "Skipping Linear weight quantization for layer '%s'",
                self.layer_name,
            )
        else:
            logger.info(
                "Applying weight pipeline for Linear layer '%s'",
                self.layer_name,
            )

        quantized_weight = quantized_linear.quantize_weight(
            layer.get_parameter("weight")
        )
        tensor_attrs = (
            None
            if quantized_weight.tensor is not None
            and hasattr(quantized_weight.tensor, "input_dim")
            else {"input_dim": 1, "output_dim": 0}
        )
        quantized_weight.register_parameters(
            layer,
            "weight",
            tensor_attrs=tensor_attrs,
            set_tensor_attrs=set_weight_attrs,
        )


__all__ = ["QuantSuiteLinearMethod"]
