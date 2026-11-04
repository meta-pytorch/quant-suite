# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantized MLA (Multi-Latent Attention) computation with dynamic runtime quantization.

The prefill path up-projects compressed KV into full multi-head attention. The
decode path performs multi-query attention in compressed space.

The module contains:
- apply_quantized_mla_prefill(): up-project KV and run full MHA for prefill.
- apply_quantized_mla_decode(): run MQA in compressed space for decode.
- apply_quantized_mla_attention(): shared quantized attention computation.
- up_project_kv(): expand compressed KV into full K and V tensors.
- process_kv_b_proj_weights(): extract the K/V up-projection matrices.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from quant_suite_recipe.implementation.attention import QuantizedAttention
from vllm.model_executor.layers.linear import LinearBase


def _normalize_mla_kv_cache(kv_cache: torch.Tensor) -> torch.Tensor:
    """Return a [blocks, block_size, head_size] cache view."""
    if kv_cache.ndim == 3:
        return kv_cache
    if kv_cache.ndim == 4 and kv_cache.shape[1] == 1:
        return kv_cache.squeeze(1)
    raise ValueError(f"Unsupported MLA KV cache shape {kv_cache.shape}")


def _chunked_context_lengths(
    chunked_context: Any,
    num_prefills: int,
) -> list[int] | None:
    if chunked_context is None:
        return None
    context_lens_list = getattr(chunked_context, "context_lens_list", None)
    if context_lens_list is not None:
        return [int(length) for length in context_lens_list[:num_prefills]]
    context_lens = getattr(chunked_context, "context_lens", None)
    if context_lens is not None:
        return [int(context_lens[index].item()) for index in range(num_prefills)]
    legacy_seq_lens = getattr(chunked_context, "seq_lens", None)
    if legacy_seq_lens is None:
        raise ValueError("Chunked MLA context does not provide context lengths")
    return [
        int(legacy_seq_lens[:, index].sum().item()) for index in range(num_prefills)
    ]


def process_kv_b_proj_weights(
    kv_b_proj: LinearBase,
    act_dtype: torch.dtype,
    num_heads: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract full-precision MLA K/V projection matrices.

    ``kv_b_proj`` stores
    ``[num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]``.
    The returned tensors are:

    - ``W_UK_T``: ``[num_heads, qk_nope_head_dim, kv_lora_rank]``.
    - ``W_UV``: ``[num_heads, kv_lora_rank, v_head_dim]``.
    """
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        get_and_maybe_dequant_weights,
    )

    kv_b_proj_weight = get_and_maybe_dequant_weights(
        kv_b_proj,
        out_dtype=act_dtype,
    ).T
    assert kv_b_proj_weight.shape == (
        kv_lora_rank,
        num_heads * (qk_nope_head_dim + v_head_dim),
    ), (
        f"{kv_b_proj_weight.shape=}, "
        f"{kv_lora_rank=}, "
        f"{num_heads=}, "
        f"{qk_nope_head_dim=}, "
        f"{v_head_dim=}"
    )
    kv_b_proj_weight = kv_b_proj_weight.view(
        kv_lora_rank,
        num_heads,
        qk_nope_head_dim + v_head_dim,
    )
    W_UK, W_UV = kv_b_proj_weight.split(
        [qk_nope_head_dim, v_head_dim],
        dim=-1,
    )

    # [kv_lora_rank, num_heads, qk_nope_head_dim]
    # -> [num_heads, qk_nope_head_dim, kv_lora_rank]
    W_UK_T = W_UK.permute(1, 2, 0).contiguous()

    # [kv_lora_rank, num_heads, v_head_dim]
    # -> [num_heads, kv_lora_rank, v_head_dim]
    W_UV_out = W_UV.transpose(0, 1).contiguous()
    return W_UK_T, W_UV_out


def up_project_kv(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    W_UK_T: torch.Tensor,
    W_UV: torch.Tensor,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Up-project compressed KV to full K and V.

    Args:
        kv_c: [seq_len, kv_lora_rank] compressed KV.
        k_pe: [seq_len, qk_rope_head_dim] positional key embeddings.
        W_UK_T: [num_heads, qk_nope_head_dim, kv_lora_rank] transposed K up-projection.
        W_UV: [num_heads, kv_lora_rank, v_head_dim] V up-projection.
        num_heads: Number of attention heads.

    Returns:
        k_full: [num_heads, seq_len, qk_head_dim] full key tensor.
        v_full: [num_heads, seq_len, v_head_dim] full value tensor.
    """
    # Up-project to K_nope: [num_heads, seq_len, qk_nope_head_dim]
    k_nope = torch.bmm(
        W_UK_T,
        kv_c.T.unsqueeze(0).expand(num_heads, -1, -1),
    ).transpose(1, 2)

    # Up-project to V: [num_heads, seq_len, v_head_dim]
    v_full = torch.bmm(
        kv_c.unsqueeze(0).expand(num_heads, -1, -1),
        W_UV,
    )

    # Expand k_pe for all heads and concat with k_nope
    k_pe_expanded = k_pe.unsqueeze(0).expand(num_heads, -1, -1)
    k_full = torch.cat([k_nope, k_pe_expanded], dim=-1)

    return k_full, v_full


def apply_quantized_mla_prefill(  # noqa: C901
    q: torch.Tensor,
    kv_c_and_k_pe_cache: torch.Tensor,
    attn_metadata: Any,
    output: torch.Tensor,
    *,
    num_heads: int,
    head_size: int,
    kv_lora_rank: int,
    qk_head_dim: int,
    v_head_dim: int,
    scale: float,
    W_UK_T: torch.Tensor,
    W_UV: torch.Tensor,
    quantized_attention: QuantizedAttention,
    micro_batch_size: int | None = None,
) -> None:
    """MHA-style prefill forward pass with quantization.

    Compute-friendly path: up-projects kv_c to full K and V from the paged
    KV cache, then applies quantized attention per request.

    NOTE: KV cache update is NOT done here — the caller
    (MLAAttention.forward) already performed it via do_kv_cache_update().

    NOTE: q is already sliced to only contain prefill tokens
    (q[num_mqa_tokens:]) by MLAAttention.forward_impl().  The attn_metadata
    still refers to ALL tokens, so we offset indices accordingly.

    Args:
        q: [num_prefill_tokens, num_heads, qk_head_dim]
        kv_c_and_k_pe_cache: Paged KV cache in compact
            ``[num_blocks, block_size, head_size]`` or singleton-head
            ``[num_blocks, 1, block_size, head_size]`` layout.
        attn_metadata: MLACommonMetadata (covers ALL tokens).
        output: Pre-allocated output [num_prefill_tokens, num_heads * v_head_dim].
        num_heads: Number of attention heads.
        head_size: kv_lora_rank + qk_rope_head_dim.
        kv_lora_rank: Compressed KV dimension.
        qk_head_dim: qk_nope_head_dim + qk_rope_head_dim.
        v_head_dim: Value dimension per head.
        scale: Attention scale factor.
        W_UK_T: [num_heads, qk_nope_head_dim, kv_lora_rank].
        W_UV: [num_heads, kv_lora_rank, v_head_dim].
        quantized_attention: Prepared attention quantization pipelines.
        micro_batch_size: Number of requests to process together. When None,
            reads QUANT_SUITE_MLA_MHA_MICRO_BATCH_SIZE env var (default 32).
            Always 1 when ``quantized_attention.supports_multi_request_bmm``
            is false.
    """
    if micro_batch_size is None:
        micro_batch_size = int(
            os.environ.get("QUANT_SUITE_MLA_MHA_MICRO_BATCH_SIZE", "32")
        )
    kv_c_and_k_pe_cache = _normalize_mla_kv_cache(kv_c_and_k_pe_cache)
    block_size = kv_c_and_k_pe_cache.shape[1]

    num_prefill_tokens = q.shape[0]
    num_decode_tokens = attn_metadata.num_decode_tokens or 0
    num_decodes = attn_metadata.num_decodes or 0
    num_prefills = attn_metadata.num_prefills or 0
    query_start_loc = attn_metadata.query_start_loc

    # MLACommonMetadata does not have seq_lens directly.
    # Use the prefill metadata's block_table and query_start_loc.
    prefill = attn_metadata.prefill
    block_table = prefill.block_table
    context_lengths = _chunked_context_lengths(
        getattr(prefill, "chunked_context", None),
        num_prefills,
    )
    micro_batch_size = max(micro_batch_size, 1)
    if not quantized_attention.supports_multi_request_bmm:
        micro_batch_size = 1

    # Collect metadata for valid requests (q_len > 0)
    req_q_starts: list[int] = []
    req_q_ends: list[int] = []
    req_q_lens: list[int] = []
    req_seq_lens: list[int] = []
    req_decode_offsets: list[int] = []
    req_prefill_indices: list[int] = []
    req_num_blocks: list[int] = []

    for i in range(num_prefills):
        req_idx = num_decodes + i
        q_start = int(query_start_loc[req_idx].item()) - num_decode_tokens
        if req_idx + 1 < len(query_start_loc):
            q_end = int(query_start_loc[req_idx + 1].item()) - num_decode_tokens
        else:
            q_end = num_prefill_tokens
        q_len = q_end - q_start

        if q_len <= 0:
            continue

        context_len = context_lengths[i] if context_lengths is not None else 0
        seq_len = context_len + q_len

        decode_offset = seq_len - q_len
        num_blocks_occupied = (seq_len + block_size - 1) // block_size

        req_q_starts.append(q_start)
        req_q_ends.append(q_end)
        req_q_lens.append(q_len)
        req_seq_lens.append(seq_len)
        req_decode_offsets.append(decode_offset)
        req_prefill_indices.append(i)
        req_num_blocks.append(num_blocks_occupied)

    num_valid = len(req_q_starts)
    if num_valid == 0:
        return

    device = q.device
    dtype = q.dtype

    # Process in micro-batches
    for chunk_start in range(0, num_valid, micro_batch_size):
        chunk_end = min(chunk_start + micro_batch_size, num_valid)
        mb = chunk_end - chunk_start

        max_q_len = max(req_q_lens[chunk_start:chunk_end])
        max_seq_len = max(req_seq_lens[chunk_start:chunk_end])

        # Pre-allocate zero-padded Q/K/V:
        # Q: [mb * num_heads, max_q_len, qk_head_dim]
        # K: [mb * num_heads, max_seq_len, qk_head_dim]
        # V: [mb * num_heads, max_seq_len, v_head_dim]
        q_batched = torch.zeros(
            mb * num_heads, max_q_len, qk_head_dim, dtype=dtype, device=device
        )
        k_batched = torch.zeros(
            mb * num_heads, max_seq_len, qk_head_dim, dtype=dtype, device=device
        )
        v_batched = torch.zeros(
            mb * num_heads, max_seq_len, v_head_dim, dtype=dtype, device=device
        )

        # Pre-compute index tensors for mask construction
        q_idx = torch.arange(max_q_len, device=device).unsqueeze(1)
        kv_idx = torch.arange(max_seq_len, device=device).unsqueeze(0)
        attn_mask = torch.zeros(
            mb * num_heads, max_q_len, max_seq_len, dtype=torch.bool, device=device
        )

        for j in range(mb):
            ci = chunk_start + j
            prefill_i = req_prefill_indices[ci]
            q_len_j = req_q_lens[ci]
            seq_len_j = req_seq_lens[ci]
            decode_offset_j = req_decode_offsets[ci]
            num_blocks_j = req_num_blocks[ci]

            # Gather paged cache blocks and flatten them to
            # [seq_len_j, head_size].
            prefill_block_row = block_table[prefill_i]
            block_indices_j = prefill_block_row[:num_blocks_j]
            kv_blocks = kv_c_and_k_pe_cache[block_indices_j]
            kv_cached = kv_blocks.reshape(-1, head_size)[:seq_len_j]

            # kv_c_cached: [seq_len_j, kv_lora_rank]
            # k_pe_cached: [seq_len_j, qk_rope_head_dim]
            kv_c_cached = kv_cached[:, :kv_lora_rank]
            k_pe_cached = kv_cached[:, kv_lora_rank:]

            # Up-project to full K and V
            k_full, v_full = up_project_kv(
                kv_c_cached, k_pe_cached, W_UK_T, W_UV, num_heads
            )

            offset = j * num_heads

            # Fill Q: [num_heads, q_len_j, qk_head_dim]
            query_req = q[req_q_starts[ci] : req_q_ends[ci]]
            query_t = query_req.transpose(0, 1)
            q_batched[offset : offset + num_heads, :q_len_j] = query_t

            # Fill K: [num_heads, seq_len_j, qk_head_dim]
            k_batched[offset : offset + num_heads, :seq_len_j] = k_full

            # Fill V: [num_heads, seq_len_j, v_head_dim]
            v_batched[offset : offset + num_heads, :seq_len_j] = v_full

            # Build combined causal + KV-padding mask for this request.
            # Q-padding positions are NOT masked: they get zero Q values
            # which produce small attention scores, yielding valid softmax
            # outputs (no NaN). We discard padded Q outputs later.
            causal = (q_idx + decode_offset_j) >= kv_idx
            kv_valid = kv_idx < seq_len_j
            mask = causal & kv_valid
            attn_mask[offset : offset + num_heads] = mask

        output_chunk, _ = apply_quantized_mla_attention(
            query_t=q_batched,
            key_t=k_batched,
            value_t=v_batched,
            num_tokens=max_q_len,
            cache_len=max_seq_len,
            decode_offset=0,
            scale=scale,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
            quantized_attention=quantized_attention,
            attn_mask=attn_mask,
        )

        # output_chunk: [max_q_len, mb*num_heads, v_head_dim]
        # -> [max_q_len, mb, num_heads, v_head_dim]
        output_chunk = output_chunk.view(max_q_len, mb, num_heads, v_head_dim)

        # Unpad per request and write to output
        for j in range(mb):
            ci = chunk_start + j
            q_len_j = req_q_lens[ci]
            out_req = output_chunk[:q_len_j, j, :, :]
            output[req_q_starts[ci] : req_q_ends[ci]] = out_req.reshape(
                q_len_j, num_heads * v_head_dim
            )


def apply_quantized_mla_decode(  # noqa: C901
    q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    kv_c_and_k_pe_cache: torch.Tensor,
    attn_metadata: Any,
    *,
    num_heads: int,
    head_size: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    scale: float,
    quantized_attention: QuantizedAttention,
    micro_batch_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """MQA-style decode forward pass with quantization.

    Data-movement friendly path: operates in compressed space.
    The caller (MLAAttention.forward_impl) has already absorbed the K
    projection into Q (``ql_nope = q_nope @ W_UK_T``). Q and K use the
    combined ``kv_lora_rank + qk_rope_head_dim`` dimension; V remains in
    ``kv_lora_rank``.
    The V up-projection happens *after* this function returns, in the caller.

    Args:
        q: Either a tuple ``(ql_nope, q_pe)`` or a concatenated tensor.
            - ql_nope: [batch, num_heads, kv_lora_rank] (absorbed Q).
            - q_pe: [batch, num_heads, qk_rope_head_dim].
        kv_c_and_k_pe_cache: Paged KV cache in compact
            ``[num_blocks, block_size, head_size]`` or singleton-head
            ``[num_blocks, 1, block_size, head_size]`` layout.
        attn_metadata: MLACommonMetadata.
        num_heads: Number of attention heads.
        head_size: kv_lora_rank + qk_rope_head_dim.
        kv_lora_rank: Compressed KV dimension.
        qk_rope_head_dim: RoPE key dimension.
        scale: Attention scale factor.
        quantized_attention: Prepared attention quantization pipelines.
        micro_batch_size: Number of requests to process together. When None,
            reads QUANT_SUITE_MLA_MQA_MICRO_BATCH_SIZE env var (default 32).
            Always 1 when ``quantized_attention.supports_multi_request_bmm``
            is false.

    Returns:
        Tuple of (attn_output, lse):
        - attn_output: [batch, num_heads, kv_lora_rank] (compressed space).
        - lse: None (not supported for quantized attention).
    """
    if micro_batch_size is None:
        micro_batch_size = int(
            os.environ.get("QUANT_SUITE_MLA_MQA_MICRO_BATCH_SIZE", "32")
        )
    kv_c_and_k_pe_cache = _normalize_mla_kv_cache(kv_c_and_k_pe_cache)

    # Handle q input format
    if isinstance(q, tuple):
        ql_nope, q_pe = q
        # [batch, num_heads, kv_lora_rank + qk_rope_head_dim]
        q_combined = torch.cat([ql_nope, q_pe], dim=-1)
    else:
        q_combined = q

    batch_size = q_combined.shape[0]

    assert hasattr(attn_metadata, "decode") and attn_metadata.decode is not None
    block_table = attn_metadata.decode.block_table
    seq_lens = attn_metadata.decode.seq_lens
    block_size = kv_c_and_k_pe_cache.shape[1]

    compressed_dim = kv_lora_rank + qk_rope_head_dim

    micro_batch_size = max(micro_batch_size, 1)
    if not quantized_attention.supports_multi_request_bmm:
        micro_batch_size = 1

    device = q_combined.device
    dtype = q_combined.dtype

    outputs = []
    for chunk_start in range(0, batch_size, micro_batch_size):
        chunk_end = min(chunk_start + micro_batch_size, batch_size)
        mb = chunk_end - chunk_start

        # Find max seq_len in this chunk
        chunk_seq_lens = [
            int(seq_lens[i].item()) for i in range(chunk_start, chunk_end)
        ]
        max_seq_len = max(chunk_seq_lens)

        # Q: [mb * num_heads, 1, compressed_dim]. Only K and V need
        # zero-initialized sequence padding.
        q_batched = q_combined[chunk_start:chunk_end].reshape(
            mb * num_heads, 1, compressed_dim
        )
        k_batched = torch.zeros(
            mb * num_heads, max_seq_len, compressed_dim, dtype=dtype, device=device
        )
        v_batched = torch.zeros(
            mb * num_heads, max_seq_len, kv_lora_rank, dtype=dtype, device=device
        )

        for j in range(mb):
            i = chunk_start + j
            seq_len_i = chunk_seq_lens[j]

            # Gather compressed KV cache blocks and flatten them to
            # [seq_len_i, head_size].
            num_blocks_needed = (seq_len_i + block_size - 1) // block_size
            block_indices_i = block_table[i, :num_blocks_needed]
            kv_blocks = kv_c_and_k_pe_cache[block_indices_i]
            kv_cached = kv_blocks.reshape(-1, head_size)[:seq_len_i]

            # kv_c_cached: [seq_len_i, kv_lora_rank]
            # k_pe_cached: [seq_len_i, qk_rope_head_dim]
            kv_c_cached = kv_cached[:, :kv_lora_rank]
            k_pe_cached = kv_cached[:, kv_lora_rank:]

            offset = j * num_heads

            # K: broadcast [seq_len_i, compressed_dim] -> [num_heads, seq_len_i, ...]
            k_row = torch.cat([kv_c_cached, k_pe_cached], dim=-1)
            k_batched[offset : offset + num_heads, :seq_len_i] = k_row

            # V: broadcast [seq_len_i, kv_lora_rank] -> [num_heads, seq_len_i, ...]
            v_batched[offset : offset + num_heads, :seq_len_i] = kv_c_cached

        # Build padding-only mask: [mb*num_heads, 1, max_seq_len]
        # No causal masking needed — decode has num_tokens=1, so
        # (0 + seq_len - 1) >= kv_idx is trivially satisfied for valid positions.
        kv_indices = torch.arange(max_seq_len, device=device)
        attn_mask = torch.zeros(
            mb * num_heads, 1, max_seq_len, dtype=torch.bool, device=device
        )
        for j in range(mb):
            seq_len_j = chunk_seq_lens[j]
            mask_1d = kv_indices < seq_len_j
            offset = j * num_heads
            attn_mask[offset : offset + num_heads, 0] = mask_1d

        output_chunk, _ = apply_quantized_mla_attention(
            query_t=q_batched,
            key_t=k_batched,
            value_t=v_batched,
            num_tokens=1,
            cache_len=max_seq_len,
            decode_offset=0,
            scale=scale,
            qk_head_dim=compressed_dim,
            v_head_dim=kv_lora_rank,
            quantized_attention=quantized_attention,
            attn_mask=attn_mask,
        )

        # output_chunk: [1, mb*num_heads, kv_lora_rank]
        # -> [mb, num_heads, kv_lora_rank]
        output_chunk = output_chunk.squeeze(0).view(mb, num_heads, kv_lora_rank)
        outputs.append(output_chunk)

    attn_output = torch.cat(outputs, dim=0)
    return attn_output, None


def apply_quantized_mla_attention(
    query_t: torch.Tensor,
    key_t: torch.Tensor,
    value_t: torch.Tensor,
    num_tokens: int,
    cache_len: int,
    decode_offset: int,
    scale: float,
    qk_head_dim: int,
    v_head_dim: int,
    quantized_attention: QuantizedAttention,
    return_debug: bool = False,
    attn_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Core quantized attention computation for MLA.

    Handles both the prefill (MHA) and decode (MQA) paths — the caller
    is responsible for presenting Q, K, V in the right coordinate space.

    Args:
        query_t: [num_heads, num_tokens, qk_head_dim]
        key_t: [num_heads, cache_len, qk_head_dim]
        value_t: [num_heads, cache_len, v_head_dim]
        num_tokens: Number of query tokens.
        cache_len: Valid cache length.
        decode_offset: Number of already-computed tokens. Ignored when
            attn_mask is provided.
        scale: Attention scale factor.
        qk_head_dim: Q/K head dimension.
        v_head_dim: V head dimension.
        quantized_attention: Prepared attention quantization pipelines.
        return_debug: Whether to return debug info.
        attn_mask: Optional pre-computed attention mask of shape
            [num_heads, num_tokens, cache_len], boolean, where True = attend
            and False = masked. When provided, the internal causal mask
            construction is skipped and this mask is used directly.

    Returns:
        output: [num_tokens, num_heads, v_head_dim]
        debug_info: Dictionary of intermediate tensors.
    """
    debug_info: dict[str, torch.Tensor] = {}
    num_heads = query_t.shape[0]

    if return_debug:
        debug_info["q"] = query_t
        debug_info["k"] = key_t
        debug_info["v"] = value_t

    # Step 1: Compute attention scores Q @ K^T with quantization
    # Q: [num_heads, num_tokens, qk_head_dim]
    #    -> [num_heads * num_tokens, qk_head_dim]
    # K: [num_heads, cache_len, qk_head_dim]
    #    -> [num_heads * cache_len, qk_head_dim]
    query_2d = query_t.reshape(-1, qk_head_dim)
    key_2d = key_t.reshape(-1, qk_head_dim)

    q_quantized = quantized_attention.quantize_q(query_2d)
    k_quantized = quantized_attention.quantize_k(key_2d)

    q_tensor = q_quantized.tensor
    k_tensor = k_quantized.tensor
    assert q_tensor is not None
    assert k_tensor is not None
    # Restore the batched-head dimensions for Q @ K^T.
    q_quantized.tensor = q_tensor.view(num_heads, num_tokens, qk_head_dim)
    k_quantized.tensor = k_tensor.view(num_heads, cache_len, qk_head_dim)

    # Compute attention scores: Q @ K^T -> [num_heads, num_tokens, cache_len]
    attn_scores = quantized_attention.bmm(
        q_quantized, k_quantized, transpose_right=True
    )

    # Step 2: Scale attention scores
    attn_scores = attn_scores * scale

    # Step 3: Apply mask
    if attn_mask is not None:
        attn_scores = attn_scores.masked_fill(~attn_mask, float("-inf"))
    else:
        q_indices = torch.arange(num_tokens, device=attn_scores.device).unsqueeze(1)
        kv_indices = torch.arange(cache_len, device=attn_scores.device).unsqueeze(0)
        causal_mask = (q_indices + decode_offset) >= kv_indices
        attn_scores = attn_scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

    if return_debug:
        debug_info["attn_scores"] = attn_scores

    # Step 4: Apply softmax
    attn_weights = torch.nn.functional.softmax(
        attn_scores, dim=-1, dtype=torch.float32
    ).to(attn_scores.dtype)

    if return_debug:
        debug_info["attn_weights"] = attn_weights

    # Step 5: Compute attention_weights @ V with quantization
    # Note: V has v_head_dim which may differ from qk_head_dim
    # Weights: [num_heads, num_tokens, cache_len]
    #          -> [num_heads * num_tokens, cache_len]
    # V: [num_heads, cache_len, v_head_dim]
    #    -> [num_heads * cache_len, v_head_dim]
    attn_weights_2d = attn_weights.reshape(-1, cache_len)
    value_2d = value_t.reshape(-1, v_head_dim)

    weights_quantized = quantized_attention.quantize_attn_weights(attn_weights_2d)
    v_quantized = quantized_attention.quantize_v(value_2d)

    weights_tensor = weights_quantized.tensor
    v_tensor = v_quantized.tensor
    assert weights_tensor is not None
    assert v_tensor is not None
    # Restore [num_heads, sequence, feature] dimensions for weights @ V.
    weights_quantized.tensor = weights_tensor.view(num_heads, num_tokens, cache_len)
    v_quantized.tensor = v_tensor.view(num_heads, cache_len, v_head_dim)

    attn_output = quantized_attention.bmm(weights_quantized, v_quantized)

    # Step 6: Reshape output
    # [num_heads, num_tokens, v_head_dim] -> [num_tokens, num_heads, v_head_dim]
    output = attn_output.transpose(0, 1).contiguous()

    return output, debug_info


__all__ = [
    "apply_quantized_mla_attention",
    "apply_quantized_mla_decode",
    "apply_quantized_mla_prefill",
    "process_kv_b_proj_weights",
    "up_project_kv",
]
