# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""KDA adapters registered through vLLM's PluggableLayer extension point."""

from __future__ import annotations

import types
from importlib import import_module
from typing import Any

import torch
from quant_suite_vllm.kda.implementation import quantize_kda_state
from quant_suite_vllm.kda.method import (
    _attach_kda_method,
    _prepared_recurrent_state_pipeline,
)


_QUANT_SUITE_ADAPTER = "_quant_suite_kda_adapter"
_NULL_BLOCK_ID = 0


def _active_recurrent_state_indices(layer_metadata: Any) -> torch.Tensor:
    if getattr(layer_metadata, "num_spec_decodes", 0) > 0:
        raise NotImplementedError(
            "KDA recurrent-state quantization does not support speculative decoding"
        )

    index_tensors: list[torch.Tensor] = []
    state_indices = getattr(
        layer_metadata,
        "non_spec_state_indices_tensor",
        None,
    )
    if state_indices is not None:
        index_tensors.append(state_indices.reshape(-1))

    checkpoint = getattr(layer_metadata, "checkpoint", None)
    if checkpoint is not None:
        checkpoint_indices = getattr(checkpoint, "state_indices", None)
        if checkpoint_indices is not None:
            index_tensors.append(checkpoint_indices.reshape(-1))

    if not index_tensors:
        raise RuntimeError("KDA metadata does not identify active recurrent states")

    active_indices = torch.cat(index_tensors)
    return active_indices[active_indices > _NULL_BLOCK_ID].unique()


def _wrap_forward_with_state_quantization(layer: torch.nn.Module) -> None:
    get_forward_context = import_module("vllm.forward_context").get_forward_context
    dynamic_layer: Any = layer
    original_forward = dynamic_layer._forward

    def _forward_with_state_quant(
        self: torch.nn.Module,
        mixed_qkv: torch.Tensor,
        g1: torch.Tensor,
        g2: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        state_pipeline = _prepared_recurrent_state_pipeline(self)
        original_forward(
            mixed_qkv=mixed_qkv,
            g1=g1,
            g2=g2,
            beta=beta,
            core_attn_out=core_attn_out,
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return
        if not isinstance(attn_metadata, dict):
            raise RuntimeError("vLLM 0.30 KDA metadata must be keyed by layer name")

        dynamic_self: Any = self
        layer_metadata = attn_metadata.get(dynamic_self.prefix)
        if layer_metadata is None:
            return
        active_indices = _active_recurrent_state_indices(layer_metadata)
        if active_indices.numel() == 0:
            return

        recurrent_state = dynamic_self.kv_cache[1]
        # Native KDA has already produced the current output and updated the state.
        # QDQ applies to that new state after prefill and each decode step, so it
        # affects the next token, not the current output.
        quantize_kda_state(recurrent_state, state_pipeline, active_indices)

    dynamic_layer._forward = types.MethodType(_forward_with_state_quant, layer)


class _QuantSuiteKDAAdapterMixin:
    _quant_suite_kda_config_types: tuple[type[Any], ...]

    def _attach_quant_suite_method(self, vllm_config: Any, prefix: str) -> None:
        dynamic_self: Any = self
        _attach_kda_method(
            dynamic_self,
            vllm_config.quant_config,
            self._quant_suite_kda_config_types,
            prefix,
            _wrap_forward_with_state_quantization,
        )


def _register_adapter(
    pluggable_layer_type: Any,
    oot_registry: dict[str, type[Any]],
    name: str,
    adapter_type: type[Any],
    quant_config_type: type[Any],
) -> None:
    existing = oot_registry.get(name)
    if existing is not None:
        if not getattr(existing, _QUANT_SUITE_ADAPTER, False):
            raise RuntimeError(f"vLLM KDA layer {name!r} already has an OOT adapter")
        registered_types = existing._quant_suite_kda_config_types
        if quant_config_type not in registered_types:
            existing._quant_suite_kda_config_types = (
                *registered_types,
                quant_config_type,
            )
        return

    setattr(adapter_type, _QUANT_SUITE_ADAPTER, True)
    adapter_type._quant_suite_kda_config_types = (quant_config_type,)
    pluggable_layer_type.register_oot(name=name)(adapter_type)


def _register_pluggable_kda_adapters(
    quant_config_type: type[Any],
    pluggable_layer_type: Any,
    oot_registry: dict[str, type[Any]],
    gated_delta_attention_type: Any,
    kimi_k3_delta_attention_type: Any,
) -> None:
    class QuantSuiteKimiGatedDeltaNetAttention(
        _QuantSuiteKDAAdapterMixin,
        gated_delta_attention_type,
    ):
        def __init__(
            self,
            config: Any,
            vllm_config: Any,
            prefix: str = "",
        ) -> None:
            super().__init__(config, vllm_config, prefix)
            self._attach_quant_suite_method(vllm_config, prefix)

    class QuantSuiteKimiK3DeltaAttention(
        _QuantSuiteKDAAdapterMixin,
        kimi_k3_delta_attention_type,
    ):
        def __init__(
            self,
            config: Any,
            vllm_config: Any,
            prefix: str = "",
            aux_stream: torch.cuda.Stream | None = None,
            run_gemm_rs_ar: bool = False,
        ) -> None:
            super().__init__(
                config,
                vllm_config,
                prefix,
                aux_stream,
                run_gemm_rs_ar,
            )
            self._attach_quant_suite_method(vllm_config, prefix)

    _register_adapter(
        pluggable_layer_type,
        oot_registry,
        "KimiGatedDeltaNetAttention",
        QuantSuiteKimiGatedDeltaNetAttention,
        quant_config_type,
    )
    _register_adapter(
        pluggable_layer_type,
        oot_registry,
        "KimiK3DeltaAttention",
        QuantSuiteKimiK3DeltaAttention,
        quant_config_type,
    )


def register_pluggable_kda_quant_method(quant_config_type: type[Any]) -> bool:
    """Register QuantSuite adapters through vLLM's PluggableLayer API."""
    custom_op = import_module("vllm.model_executor.custom_op")
    gated_delta_attention = import_module(
        "vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn"
    ).KimiGatedDeltaNetAttention
    kimi_k3_delta_attention = import_module(
        "vllm.models.kimi_k3.nvidia.kda"
    ).KimiK3DeltaAttention
    _register_pluggable_kda_adapters(
        quant_config_type,
        custom_op.PluggableLayer,
        custom_op.op_registry_oot,
        gated_delta_attention,
        kimi_k3_delta_attention,
    )
    return True


__all__ = ["register_pluggable_kda_quant_method"]
