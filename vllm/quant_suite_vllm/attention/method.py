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
from quant_suite_recipe.runtime_provider import QUANT_RECIPE_RUNTIME
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod


logger: logging.Logger = logging.getLogger(__name__)


class QuantSuiteAttentionMethod(BaseKVCacheMethod):
    """vLLM KV-cache method backed by a QuantSuite attention recipe."""

    def __init__(
        self,
        quant_recipe: QuantRecipe,
        layer_name: str,
        *,
        quant_config: QuantizationConfig,
    ) -> None:
        super().__init__(quant_config)
        self.layer_name = layer_name
        self.attention_recipe = quant_recipe.get_attention_recipe(layer_name)
        native_fields = [
            name
            for name in ("q", "k", "v", "attn_weights")
            if getattr(self.attention_recipe, name).has_native_quant
        ]
        if native_fields:
            raise NotImplementedError(
                "QuantSuite vLLM attention does not support native "
                f"quantization for: {', '.join(native_fields)}"
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)

        dynamic_layer: Any = layer
        bind_quantization = getattr(dynamic_layer.impl, "bind_quantization", None)
        if not callable(bind_quantization):
            if not self.attention_recipe.requires_custom_attention_backend():
                return
            raise TypeError(
                "QuantSuiteAttentionMethod requires a QuantSuite attention backend, "
                f"got {type(dynamic_layer.impl).__name__}"
            )

        quantized_attention = QUANT_RECIPE_RUNTIME.prepare_attention(
            self.attention_recipe
        )
        dynamic_layer.quantized_attention = quantized_attention
        bind_quantization(quantized_attention)

        logger.info(
            "Prepared QuantSuite attention for layer '%s'",
            self.layer_name,
        )


__all__ = ["QuantSuiteAttentionMethod"]
