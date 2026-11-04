# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantized KDA (Kimi Delta Attention) recurrent state.

This module provides fake quantization for the recurrent hidden state
in KDA layers. The recurrent state is analogous to the KV cache in
softmax attention — during decode, loading it from HBM is the memory
bandwidth bottleneck. Quantizing it reduces memory footprint and bandwidth.
"""

from __future__ import annotations

import torch
from quant_suite_recipe.implementation.quant_pipeline import Pipeline


def quantize_kda_state(
    recurrent_state: torch.Tensor,
    state_pipeline: Pipeline,
    active_indices: torch.Tensor,
) -> None:
    """Apply fake quantization to KDA recurrent state in-place.

    The state has shape [num_blocks, H, K, V] where H=num_heads, K=key_dim,
    V=value_dim. For Kimi-Linear: [num_blocks, 32, 128, 128] = 1 MB per
    block per layer in bf16.

    Only the blocks at ``active_indices`` are quantized. The underlying buffer
    is shared across multiple layers, so quantizing the entire tensor would
    corrupt other layers' cache data.

    Fake quantization (quantize -> dequantize) simulates the precision
    loss without changing the storage dtype. An empty (identity) pipeline is a
    no-op and returns early.

    Args:
        recurrent_state: The recurrent state tensor, modified in-place.
            Shape: [num_blocks, H, K, V]
        state_pipeline: Quantization pipeline for the state tensor (returns a
            QuantizedTensor); an empty pipeline is the identity.
        active_indices: Non-empty tensor of the block indices that belong to
            the current layer's active requests.

    Raises:
        ValueError: if ``active_indices`` is empty.
    """
    if len(state_pipeline.stages) == 0:
        return

    if active_indices.numel() == 0:
        raise ValueError(
            "quantize_kda_state requires a non-empty active_indices to identify "
            "which cache blocks belong to this layer. The recurrent state buffer "
            "is shared across multiple layers, so quantizing without active "
            "indices would corrupt other layers' data."
        )

    active_states = recurrent_state[active_indices]
    orig_shape = active_states.shape
    flat = active_states.reshape(-1, orig_shape[-1])
    quantized = state_pipeline(flat).tensor
    assert quantized is not None
    recurrent_state[active_indices] = quantized.view(orig_shape)
