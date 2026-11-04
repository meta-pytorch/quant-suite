# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, cast

import torch
from quant_suite_recipe.definition import QuantRecipe
from quant_suite_vllm.attention.method import QuantSuiteAttentionMethod
from quant_suite_vllm.linear.method import QuantSuiteLinearMethod
from quant_suite_vllm.moe.method import QuantSuiteMoEMethod
from quant_suite_vllm.plugin import register_attention_backend
from vllm.model_executor.layers import fused_moe as vllm_fused_moe
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)


# Support MoE layers represented by either RoutedExperts or FusedMoE.
_moe_layer_type = getattr(vllm_fused_moe, "RoutedExperts", None)
if _moe_layer_type is None:
    _moe_layer_type = getattr(vllm_fused_moe, "FusedMoE", None)
if _moe_layer_type is None:
    raise ImportError("vLLM must expose RoutedExperts or FusedMoE")
_VLLM_MOE_LAYER_TYPE = cast(type[torch.nn.Module], _moe_layer_type)


def get_quant_method(
    quant_recipe: QuantRecipe,
    quant_config: QuantizationConfig,
    layer: torch.nn.Module,
    prefix: str,
) -> QuantizeMethodBase | None:
    """The QuantSuite method for ``layer``, or None to leave it to vLLM.

    ``prefix`` is the layer's module name, which the method matches against the
    recipe's regex sections. When the recipe has no section for the layer's
    kind, a Linear layer gets vLLM's unquantized method and an MoE or attention
    layer gets None. A layer that no pattern in its section matches gets a
    QuantSuite method with an identity recipe.
    """
    if isinstance(layer, LinearBase):
        if quant_recipe.has_linear_recipes():
            return QuantSuiteLinearMethod(
                quant_recipe,
                prefix,
            )
        return UnquantizedLinearMethod()
    if isinstance(layer, _VLLM_MOE_LAYER_TYPE):
        if quant_recipe.has_moe_recipes():
            dynamic_layer: Any = layer
            return QuantSuiteMoEMethod(
                quant_recipe,
                prefix,
                moe_config=dynamic_layer.moe_config,
            )
        return None
    if isinstance(layer, AttentionLayerBase):
        if quant_recipe.has_attention_recipes():
            return QuantSuiteAttentionMethod(
                quant_recipe,
                prefix,
                quant_config=quant_config,
            )
        return None
    return None


__all__ = [
    "get_quant_method",
    "register_attention_backend",
]
