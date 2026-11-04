# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from typing import Any

import torch
from quant_suite_recipe.definition import QuantRecipe
from quant_suite_recipe.implementation.moe import QuantizedMoE
from quant_suite_recipe.runtime_provider import QUANT_RECIPE_RUNTIME
from quant_suite_vllm.moe.implementation import apply_quantized_fused_experts
from vllm.model_executor.layers.fused_moe import FusedMoEConfig, FusedMoEMethodBase
from vllm.model_executor.utils import set_weight_attrs


logger: logging.Logger = logging.getLogger(__name__)


class QuantSuiteMoEMethod(FusedMoEMethodBase):
    """vLLM fused-MoE method backed by a QuantSuite recipe."""

    def __init__(
        self,
        quant_recipe: QuantRecipe,
        layer_name: str,
        *,
        moe_config: FusedMoEConfig,
    ) -> None:
        super().__init__(moe_config)
        self.layer_name = layer_name
        self.moe_recipe = quant_recipe.get_moe_recipe(layer_name)
        if moe_config.has_bias:
            raise NotImplementedError("QuantSuite vLLM MoE does not support MoE bias")
        if self.moe_recipe.a1.has_native_quant or self.moe_recipe.a2.has_native_quant:
            raise NotImplementedError(
                "QuantSuite vLLM MoE does not support native activation quantization"
            )
        if self.moe_recipe.init_quantized_weight and (
            self.moe_recipe.w13.has_native_quant or self.moe_recipe.w2.has_native_quant
        ):
            raise NotImplementedError(
                "QuantSuite vLLM MoE does not support native weight quantization"
            )

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        dynamic_layer: Any = layer
        dynamic_layer.intermediate_size_per_partition = intermediate_size_per_partition
        dynamic_layer.hidden_size = hidden_size
        dynamic_layer.num_experts = num_experts
        dynamic_layer.orig_dtype = params_dtype

        w13_num_shards = 2 if self.moe.is_act_and_mul else 1
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        quantized_moe = QUANT_RECIPE_RUNTIME.prepare_moe(self.moe_recipe)
        dynamic_layer: Any = layer
        dynamic_layer.quantized_moe = quantized_moe

        if self.moe_recipe.w13.is_noop() and self.moe_recipe.w2.is_noop():
            logger.info(
                "Skipping MoE weight quantization for layer '%s'",
                self.layer_name,
            )
        else:
            logger.info("Applying MoE weight pipelines for layer '%s'", self.layer_name)

        quantized_moe.quantize_w13(
            layer.get_parameter("w13_weight")
        ).register_parameters(
            layer,
            "w13_weight",
        )
        quantized_moe.quantize_w2(layer.get_parameter("w2_weight")).register_parameters(
            layer,
            "w2_weight",
        )

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> None:
        return None

    # Support callers that pass only `shared_experts_input` and callers that
    # also pass `shared_experts`; defaults preserve both call shapes.
    # pyrefly: ignore [bad-override]
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: object | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply routed experts.

        vLLM's runner executes and combines shared experts for methods without
        a modular kernel, so both shared-expert arguments are intentionally left
        to the runner.
        """
        dynamic_layer: Any = layer
        quantized_moe: QuantizedMoE = dynamic_layer.quantized_moe
        return apply_quantized_fused_experts(
            hidden_states=x,
            w13_weight=layer.get_parameter("w13_weight"),
            w2_weight=layer.get_parameter("w2_weight"),
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=dynamic_layer.activation,
            global_num_experts=dynamic_layer.global_num_experts,
            apply_router_weight_on_input=dynamic_layer.apply_router_weight_on_input,
            expert_map=dynamic_layer.expert_map,
            quantized_moe=quantized_moe,
        )


__all__ = ["QuantSuiteMoEMethod"]
