# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch
from quant_suite_recipe.definition import QuantRecipe
from quant_suite_recipe.implementation.attention import QuantizedAttention
from quant_suite_recipe.implementation.quant_pipeline import Pipeline
from quant_suite_recipe.runtime_provider import QUANT_RECIPE_RUNTIME
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase


logger: logging.Logger = logging.getLogger(__name__)


def _prepared_recurrent_state_pipeline(layer: torch.nn.Module) -> Pipeline:
    """Return the recurrent-state pipeline prepared for ``layer``."""
    quantized_attention = getattr(layer, "quantized_attention", None)
    if not isinstance(quantized_attention, QuantizedAttention):
        raise RuntimeError(
            "KDA recurrent-state quantization has not been prepared: the weights "
            "have not been loaded and process_weights_after_loading has not run"
        )
    return quantized_attention.recurrent_state_pipeline


class QuantSuiteKDAMethod(QuantizeMethodBase):
    """Prepare recurrent-state quantization for a KDA layer."""

    def __init__(
        self,
        quant_recipe: QuantRecipe,
        layer_name: str,
    ) -> None:
        self.layer_name = layer_name
        self.attention_recipe = quant_recipe.get_attention_recipe(layer_name)

        unsupported_fields = [
            field_name
            for field_name in ("q", "k", "v", "attn_weights")
            if not getattr(self.attention_recipe, field_name).is_noop()
        ]
        if unsupported_fields:
            raise NotImplementedError(
                "KDA only supports recurrent-state quantization; unsupported "
                f"attention recipe fields: {', '.join(unsupported_fields)}"
            )

        state_recipe = self.attention_recipe.recurrent_state
        self._skip_state_quantization = state_recipe.is_noop()
        if not self._skip_state_quantization and state_recipe.has_native_quant:
            raise NotImplementedError(
                "KDA recurrent-state quantization supports fake quantization only; "
                "native recurrent-state quantization is not supported"
            )

    @property
    def quantizes_recurrent_state(self) -> bool:
        return not self._skip_state_quantization

    def create_weights(self, layer: torch.nn.Module, **kwargs: Any) -> None:
        pass

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "QuantSuiteKDAMethod.apply() should not be called; "
            "KDA layers use _forward() directly"
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self._skip_state_quantization:
            logger.info(
                "Skipping KDA state quantization for layer '%s'",
                self.layer_name,
            )
            return
        if getattr(layer, "num_spec", 0) > 0:
            raise NotImplementedError(
                "KDA recurrent-state quantization does not support speculative decoding"
            )
        if getattr(layer, "use_recoverssm", False):
            raise NotImplementedError(
                "KDA recurrent-state quantization does not support RecoverSSM"
            )

        logger.info(
            "Preparing KDA recurrent-state quantization for layer '%s'",
            self.layer_name,
        )
        dynamic_layer: Any = layer
        dynamic_layer.quantized_attention = QUANT_RECIPE_RUNTIME.prepare_attention(
            self.attention_recipe
        )


def _attach_kda_method(
    layer: torch.nn.Module,
    quant_config: object,
    config_types: tuple[type[Any], ...],
    prefix: str,
    wrap_forward: Callable[[torch.nn.Module], None],
) -> None:
    """Install ``QuantSuiteKDAMethod`` on a KDA ``layer`` built with a QuantSuite
    config that has attention recipes, and wrap the layer's forward with
    recurrent-state quantization when the recipe quantizes that state."""
    if not isinstance(quant_config, config_types):
        return
    dynamic_config: Any = quant_config
    quant_recipe: QuantRecipe = dynamic_config.quant_recipe
    if not quant_recipe.has_attention_recipes():
        return
    quant_method = QuantSuiteKDAMethod(quant_recipe, prefix)
    dynamic_layer: Any = layer
    dynamic_layer.quant_method = quant_method
    if quant_method.quantizes_recurrent_state:
        wrap_forward(layer)


__all__ = ["QuantSuiteKDAMethod"]
