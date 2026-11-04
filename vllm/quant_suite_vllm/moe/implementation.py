# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# `apply_quantized_fused_experts` is adapted from `fused_experts_impl` in
# vllm/model_executor/layers/fused_moe/fused_moe.py of vLLM
# (https://github.com/vllm-project/vllm), licensed under the Apache License,
# Version 2.0, and modified by Meta Platforms, Inc. and affiliates.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# See THIRD_PARTY_NOTICES.md in the root directory of this package.

"""Quantized fused-experts implementation for the QuantSuite vLLM method."""

from __future__ import annotations

import functools

import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from quant_suite_recipe.implementation.moe import QuantizedMoE
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import (
    apply_moe_activation,
    MoEActivation,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    dispatch_fused_moe_kernel,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.triton_utils import tl


CHUNK_SIZE = 64 * 1024


def apply_quantized_fused_experts(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str | MoEActivation = "silu",
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    *,
    quantized_moe: QuantizedMoE,
) -> torch.Tensor:
    """Apply fused MoE computation with prepared QuantSuite pipelines.

    Activations are quantized through the prepared ``QuantizedMoE`` owned by the
    framework layer. Native activation quantization is unsupported here; callers
    guard against it.

    This follows vLLM's fused-MoE execution and quantizes activations immediately
    before each expert matrix multiplication.
    """
    if isinstance(activation, MoEActivation):
        activation_enum = activation
    else:
        activation_enum = MoEActivation.from_str(activation)

    w1 = w13_weight
    w2 = w2_weight
    w1_scale = None
    w1_zp = None
    w1_bias = None
    w2_scale = None
    w2_zp = None
    w2_bias = None
    assert topk_weights.size() == topk_ids.size(), "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    num_tokens = hidden_states.size(0)
    E, N, _ = w1.size()
    K = w2.size(1)
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.size(1)
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    M = min(num_tokens, CHUNK_SIZE)

    config_dtype = None
    block_shape = None
    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.size(),
        w2.size(),
        top_k_num,
        config_dtype,
        block_shape=block_shape,
    )

    config = get_config_func(M)

    # The two views intentionally share storage. The second kernel may overwrite
    # cache1 only after apply_moe_activation has consumed it.
    cache13 = torch.empty(
        M * top_k_num * max(N, K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = cache13[: M * top_k_num * N].view(M, top_k_num, N)
    intermediate_cache3 = cache13[: M * top_k_num * K].view(M, top_k_num, K)

    # This needs separate memory since it's used concurrently with cache1
    activation_out_dim = mk.FusedMoEExpertsModular.adjust_N_for_activation(
        N, activation_enum
    )
    intermediate_cache2 = torch.empty(
        (M * top_k_num, activation_out_dim),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

    out_hidden_states = torch.empty_like(hidden_states)

    for begin_chunk_idx in range(0, num_tokens, CHUNK_SIZE):
        end_chunk_idx = min(begin_chunk_idx + CHUNK_SIZE, num_tokens)
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.size()

        if tokens_in_chunk < CHUNK_SIZE and begin_chunk_idx > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[
                : tokens_in_chunk * topk_ids.size(1)
            ]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            config = get_config_func(tokens_in_chunk)

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
        qcurr_hidden_states = quantized_moe.quantize_a1(curr_hidden_states).tensor
        assert qcurr_hidden_states is not None
        a1q_scale = None

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            curr_topk_ids, config["BLOCK_SIZE_M"], global_num_experts, expert_map
        )

        dispatch_fused_moe_kernel(
            qcurr_hidden_states,
            w1,
            intermediate_cache1,
            a1q_scale,
            w1_scale,
            w1_zp,
            curr_topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input,
            top_k_num,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=block_shape,
            B_bias=w1_bias,
        )

        apply_moe_activation(
            activation_enum, intermediate_cache2, intermediate_cache1.view(-1, N)
        )

        qintermediate_cache2 = quantized_moe.quantize_a2(intermediate_cache2).tensor
        assert qintermediate_cache2 is not None
        a2q_scale = None

        dispatch_fused_moe_kernel(
            qintermediate_cache2,
            w2,
            intermediate_cache3,
            a2q_scale,
            w2_scale,
            w2_zp,
            curr_topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=block_shape,
            B_bias=w2_bias,
        )

        ops.moe_sum(
            intermediate_cache3,
            out_hidden_states[begin_chunk_idx:end_chunk_idx],
        )

    return out_hidden_states


__all__ = ["apply_quantized_fused_experts"]
