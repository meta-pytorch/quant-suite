# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Attach QuantSuite to KDA layers without a PluggableLayer extension point.

The adapter patches layer construction to install the quantization method and
wraps the layer's forward with recurrent-state quantization.
"""

from __future__ import annotations

import inspect
import types
from functools import wraps
from typing import Any

import torch
from quant_suite_vllm.kda.implementation import quantize_kda_state
from quant_suite_vllm.kda.method import (
    _attach_kda_method,
    _prepared_recurrent_state_pipeline,
)


try:
    from vllm.model_executor.layers.kda import KimiDeltaAttention
except ModuleNotFoundError as error:
    if error.name != "vllm.model_executor.layers.kda":
        raise
    KimiDeltaAttention: Any = None


def _wrap_forward_with_state_quantization(layer: torch.nn.Module) -> None:
    from vllm.forward_context import get_forward_context

    dynamic_layer: Any = layer
    original_forward = dynamic_layer._forward

    def _forward_with_state_quant(
        self: torch.nn.Module,
        q_proj_states: torch.Tensor,
        k_proj_states: torch.Tensor,
        v_proj_states: torch.Tensor,
        g1: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        state_pipeline = _prepared_recurrent_state_pipeline(self)
        original_forward(
            q_proj_states=q_proj_states,
            k_proj_states=k_proj_states,
            v_proj_states=v_proj_states,
            g1=g1,
            beta=beta,
            core_attn_out=core_attn_out,
        )

        dynamic_self: Any = self
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return
        layer_metadata = (
            attn_metadata.get(dynamic_self.prefix)
            if isinstance(attn_metadata, dict)
            else attn_metadata
        )
        if layer_metadata is None:
            return
        state_indices = getattr(layer_metadata, "non_spec_state_indices_tensor", None)
        if state_indices is None:
            raise RuntimeError("KDA metadata does not identify active recurrent states")
        # Negative indices are padding slots.
        active_indices = state_indices[state_indices >= 0].unique()
        if active_indices.numel() == 0:
            return

        recurrent_state = dynamic_self.kv_cache[forward_context.virtual_engine][3]
        # Native KDA has already produced the current output and updated the state.
        # QDQ applies to that new state after prefill and each decode step, so it
        # affects the next token, not the current output.
        quantize_kda_state(recurrent_state, state_pipeline, active_indices)

    dynamic_layer._forward = types.MethodType(_forward_with_state_quant, layer)


def register_constructor_patch_kda_quant_method(
    quant_config_type: type[Any],
) -> bool:
    """Enable QuantSuite method lookup by patching the KDA constructor."""
    if KimiDeltaAttention is None:
        return False

    dynamic_class: Any = KimiDeltaAttention
    registered_config_types = getattr(
        dynamic_class,
        "_quant_suite_kda_config_types",
        (),
    )
    if quant_config_type not in registered_config_types:
        dynamic_class._quant_suite_kda_config_types = (
            *registered_config_types,
            quant_config_type,
        )

    if getattr(dynamic_class, "_quant_suite_kda_init_patched", False):
        return True

    original_init = KimiDeltaAttention.__init__
    init_signature = inspect.signature(original_init)

    @wraps(original_init)
    def patched_init(self: KimiDeltaAttention, *args: Any, **kwargs: Any) -> None:
        arguments = init_signature.bind(self, *args, **kwargs)
        original_init(self, *args, **kwargs)
        dynamic_self: Any = self
        _attach_kda_method(
            dynamic_self,
            arguments.arguments.get("quant_config"),
            dynamic_class._quant_suite_kda_config_types,
            arguments.arguments.get("prefix", ""),
            _wrap_forward_with_state_quantization,
        )

    dynamic_class.__init__ = patched_init
    dynamic_class._quant_suite_kda_init_patched = True
    return True


__all__ = [
    "KimiDeltaAttention",
    "register_constructor_patch_kda_quant_method",
]
