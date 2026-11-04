# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""QuantSuite paged attention with runtime quantization.

`apply_quantized_attention` gathers paged K/V cache entries and groups requests
into micro-batches. `_quantized_attention_for_batch` implements the shared
single-request and batched-request attention math.
"""

from __future__ import annotations

import math
import os
from typing import Any, ClassVar

import torch
from quant_suite_recipe.api.implementation import QuantizedTensor
from quant_suite_recipe.implementation.attention import QuantizedAttention
from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.flex_attention import (
    FlexAttentionBackend,
    FlexAttentionMetadata,
    FlexAttentionMetadataBuilder,
)


def _reject_unsupported_features(
    backend_name: str,
    alibi_slopes: list[float] | None,
    sliding_window: int | None,
    logits_soft_cap: float | None,
) -> None:
    unsupported = []
    if alibi_slopes is not None:
        unsupported.append("ALiBi slopes")
    if sliding_window is not None:
        unsupported.append("sliding window")
    if logits_soft_cap:
        unsupported.append("logits soft cap")
    if unsupported:
        raise NotImplementedError(
            f"{backend_name} does not support {', '.join(unsupported)}"
        )


def _split_kv_cache(
    kv_cache: torch.Tensor,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize stacked and LBNHC caches to token-major K/V views."""
    if kv_cache.ndim == 4 and kv_cache.shape[-1] == 2 * head_size:
        key_cache, value_cache = kv_cache.transpose(1, 2).split(head_size, dim=-1)
        return key_cache, value_cache
    if (
        kv_cache.ndim == 5
        and kv_cache.shape[0] == 2
        and kv_cache.shape[-1] == head_size
    ):
        key_cache, value_cache = kv_cache.unbind(0)
        return key_cache, value_cache
    raise ValueError(f"Unsupported KV cache shape {kv_cache.shape}")


def _reshape_quantized(
    quantized: QuantizedTensor,
    shape: tuple[int, ...],
) -> QuantizedTensor:
    assert quantized.tensor is not None
    quantized.tensor = quantized.tensor.view(shape)
    return quantized


def _quantized_attention_for_batch(
    queries: list[torch.Tensor],
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
    query_lengths: list[int],
    key_lengths: list[int],
    decode_offsets: list[int],
    *,
    scale: float,
    quantized_attention: QuantizedAttention,
) -> torch.Tensor:
    """Apply quantized attention to one request chunk.

    Each list entry has already been transposed into head-major layout:

    - queries: ``[num_heads, query_length_i, head_size]``.
    - keys/values: ``[num_heads, key_length_i, head_size]``.

    Requests are padded to the largest lengths in this chunk, flattened to
    ``batch_size * num_heads`` for BMM, then unpadded and concatenated into
    ``[sum(query_lengths), num_heads, head_size]``.
    """
    batch_size = len(queries)
    if batch_size != 1 and not quantized_attention.supports_multi_request_bmm:
        raise NotImplementedError(
            "Selected attention runtime only supports singleton request BMM; "
            f"got {batch_size} requests"
        )
    bmm = quantized_attention.bmm
    num_heads, _, head_size = queries[0].shape
    max_query_length = max(query_lengths)
    max_key_length = max(key_lengths)
    device = queries[0].device
    dtype = queries[0].dtype

    # Zero-pad and stack requests:
    # Q: [batch_size, num_heads, max_query_length, head_size]
    # K/V: [batch_size, num_heads, max_key_length, head_size]
    query = torch.zeros(
        batch_size,
        num_heads,
        max_query_length,
        head_size,
        device=device,
        dtype=dtype,
    )
    key = torch.zeros(
        batch_size,
        num_heads,
        max_key_length,
        head_size,
        device=device,
        dtype=dtype,
    )
    value = torch.zeros_like(key)
    for index in range(batch_size):
        query[index, :, : query_lengths[index], :] = queries[index]
        key[index, :, : key_lengths[index], :] = keys[index]
        value[index, :, : key_lengths[index], :] = values[index]

    # Fold request and head dimensions for batched matrix multiplication:
    # Q: [batch_size * num_heads, max_query_length, head_size]
    # K/V: [batch_size * num_heads, max_key_length, head_size]
    flattened_batch = batch_size * num_heads
    query = query.view(flattened_batch, max_query_length, head_size)
    key = key.view(flattened_batch, max_key_length, head_size)
    value = value.view(flattened_batch, max_key_length, head_size)

    quantized_query = _reshape_quantized(
        quantized_attention.quantize_q(query.reshape(-1, head_size)),
        (flattened_batch, max_query_length, head_size),
    )
    quantized_key = _reshape_quantized(
        quantized_attention.quantize_k(key.reshape(-1, head_size)),
        (flattened_batch, max_key_length, head_size),
    )
    # Q @ K^T -> [batch_size * num_heads, max_query_length, max_key_length].
    # Accumulate scores in FP32 before masking and softmax.
    attention_scores = bmm(
        quantized_query,
        quantized_key,
        transpose_right=True,
        out_dtype=torch.float32,
    )
    attention_scores *= scale

    # Build one [batch_size, max_query_length, max_key_length] mask combining
    # causal positions and per-request Q/K padding.
    query_indices = torch.arange(max_query_length, device=device).view(1, -1, 1)
    key_indices = torch.arange(max_key_length, device=device).view(1, 1, -1)
    query_lengths_tensor = torch.tensor(
        query_lengths, device=device, dtype=torch.long
    ).view(-1, 1, 1)
    key_lengths_tensor = torch.tensor(
        key_lengths, device=device, dtype=torch.long
    ).view(-1, 1, 1)
    decode_offsets_tensor = torch.tensor(
        decode_offsets, device=device, dtype=torch.long
    ).view(-1, 1, 1)
    key_is_valid = key_indices < key_lengths_tensor
    is_causal = query_indices + decode_offsets_tensor >= key_indices
    query_is_padding = query_indices >= query_lengths_tensor
    attention_mask = key_is_valid & (is_causal | query_is_padding)
    attention_mask = (
        attention_mask.unsqueeze(1)
        .expand(-1, num_heads, -1, -1)
        .reshape(flattened_batch, max_query_length, max_key_length)
    )
    attention_scores.masked_fill_(~attention_mask, float("-inf"))

    attention_weights_fp32 = torch.nn.functional.softmax(
        attention_scores,
        dim=-1,
        dtype=torch.float32,
    )
    # Keep softmax in FP32, then downcast at the quantization boundary so the
    # weights and V operands have matching storage dtype for the second BMM.
    attention_weights = attention_weights_fp32.to(dtype)
    quantized_weights = _reshape_quantized(
        quantized_attention.quantize_attn_weights(
            attention_weights.reshape(-1, max_key_length)
        ),
        (flattened_batch, max_query_length, max_key_length),
    )
    quantized_value = _reshape_quantized(
        quantized_attention.quantize_v(value.reshape(-1, head_size)),
        (flattened_batch, max_key_length, head_size),
    )
    # Weights @ V -> [batch_size * num_heads, max_query_length, head_size],
    # then restore [batch_size, num_heads, max_query_length, head_size].
    output = bmm(quantized_weights, quantized_value).view(
        batch_size,
        num_heads,
        max_query_length,
        head_size,
    )
    return torch.cat(
        [
            output[index, :, : query_lengths[index], :].transpose(0, 1).contiguous()
            for index in range(batch_size)
        ],
        dim=0,
    )


def apply_quantized_attention(
    query: torch.Tensor,
    key: torch.Tensor | None,
    value: torch.Tensor | None,
    kv_cache: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    num_kv_heads: int,
    head_size: int,
    num_heads: int,
    attn_metadata: Any,
    scale: float,
    quantization: QuantizedAttention,
    *,
    micro_batch_size: int = 8,
) -> torch.Tensor:
    """Apply causal paged attention with prepared QuantSuite pipelines.

    The operation:

    1. Cache new K/V tensors using vLLM's paged-cache operator.
    2. Gather each request's K/V sequence from its physical block table.
    3. Expand GQA K/V heads to match query heads.
    4. Process requests in micro-batches through quantized Q @ K^T and
       attention-weights @ V.

    Args:
        query: ``[num_tokens, num_heads, head_size]``.
        key/value: Optional new cache entries with shape
            ``[num_tokens, num_kv_heads, head_size]``.
        kv_cache: Either stacked K/V
            ``[2, num_blocks, block_size, num_kv_heads, head_size]`` or LBNHC
            ``[num_blocks, num_kv_heads, block_size, 2 * head_size]``.
        attn_metadata: Request lengths, block tables, slots, and decode offsets.
        quantization: Prepared Q/K/V/attention-weight pipelines.
        micro_batch_size: Maximum number of requests per BMM chunk.

    Returns:
        ``[num_actual_tokens, num_heads, head_size]``.
    """
    num_requests = attn_metadata.num_reqs
    query_start_loc = attn_metadata.query_start_loc
    required_query_offsets = num_requests + 1
    if query_start_loc.ndim != 1 or query_start_loc.shape[0] < required_query_offsets:
        raise ValueError(
            "Attention metadata query_start_loc must be one-dimensional and "
            f"contain at least {required_query_offsets} offsets for "
            f"{num_requests} requests; got shape {tuple(query_start_loc.shape)}"
        )

    key_cache, value_cache = _split_kv_cache(kv_cache, head_size)
    if key is not None and value is not None:
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            attn_metadata.slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

    # GQA expansion factor: each KV head serves this many query heads.
    num_queries_per_kv = num_heads // num_kv_heads
    query_lengths = [
        int(query_start_loc[index + 1].item()) - int(query_start_loc[index].item())
        for index in range(num_requests)
    ]

    outputs: list[torch.Tensor] = []
    effective_batch_size = min(micro_batch_size, num_requests)
    block_table = getattr(
        attn_metadata,
        "block_table",
        getattr(attn_metadata, "block_table_tensor", None),
    )
    if block_table is None:
        raise ValueError("Attention metadata does not provide a block table")
    metadata_decode_offsets = getattr(attn_metadata, "decode_offset", None)
    if metadata_decode_offsets is None:
        metadata_decode_offsets = attn_metadata.num_computed_tokens_cpu.to(
            query.device,
            non_blocking=True,
        )
    block_size = kv_cache.shape[2]
    # Gather request-local Q/K/V tensors, then execute one unified attention
    # helper for both singleton and multi-request chunks.
    for batch_start in range(0, num_requests, effective_batch_size):
        batch_end = min(
            batch_start + effective_batch_size,
            num_requests,
        )
        queries: list[torch.Tensor] = []
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        chunk_query_lengths: list[int] = []
        key_lengths: list[int] = []
        decode_offsets: list[int] = []

        for request_index in range(batch_start, batch_end):
            query_start = int(query_start_loc[request_index].item())
            query_length = query_lengths[request_index]
            query_end = query_start + query_length
            key_length = int(attn_metadata.seq_lens[request_index].item())
            block_count = (key_length + block_size - 1) // block_size
            # Gather paged K/V blocks and flatten to
            # [key_length, num_kv_heads, head_size].
            block_indices = block_table[request_index, :block_count]
            request_key = key_cache[block_indices].reshape(
                -1,
                num_kv_heads,
                head_size,
            )[:key_length]
            request_value = value_cache[block_indices].reshape(
                -1,
                num_kv_heads,
                head_size,
            )[:key_length]
            # [key_length, num_kv_heads, head_size]
            # -> [key_length, num_heads, head_size]
            if num_queries_per_kv > 1:
                request_key = request_key.repeat_interleave(
                    num_queries_per_kv,
                    dim=1,
                )
                request_value = request_value.repeat_interleave(
                    num_queries_per_kv,
                    dim=1,
                )

            # Convert request tensors to head-major layout:
            # [sequence, num_heads, head_size]
            # -> [num_heads, sequence, head_size].
            request_query = query[query_start:query_end].transpose(0, 1)
            request_key = request_key.transpose(0, 1)
            request_value = request_value.transpose(0, 1)
            queries.append(request_query)
            keys.append(request_key)
            values.append(request_value)
            chunk_query_lengths.append(query_length)
            key_lengths.append(key_length)
            decode_offsets.append(int(metadata_decode_offsets[request_index].item()))

        chunk_output = _quantized_attention_for_batch(
            queries,
            keys,
            values,
            chunk_query_lengths,
            key_lengths,
            decode_offsets,
            scale=scale,
            quantized_attention=quantization,
        )
        outputs.append(chunk_output)

    return torch.cat(outputs, dim=0)


class QuantSuiteAttentionMetadataBuilder(FlexAttentionMetadataBuilder):
    """vLLM's FlexAttention metadata builder, with CUDA-graph capture disabled."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER


class QuantSuiteAttentionBackend(AttentionBackend):
    """MHA backend for prepared QuantSuite attention recipes."""

    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto"]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["QuantSuiteAttentionImpl"]:
        return QuantSuiteAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["QuantSuiteAttentionMetadataBuilder"]:
        return QuantSuiteAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        get_shape = getattr(FlexAttentionBackend, "get_kv_cache_shape", None)
        if get_shape is None:
            return (num_blocks, num_kv_heads, block_size, 2 * head_size)
        return get_shape(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            cache_dtype_str,
        )

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[Any, ...] | None:
        from importlib import import_module

        kv_cache_interface = import_module("vllm.v1.kv_cache_interface")
        kv_cache_layout = getattr(kv_cache_interface, "KVCacheLayout", None)
        if kv_cache_layout is None:
            return None
        return (kv_cache_layout.LBNHC,)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size % 8 == 0 and head_size <= 256

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype is None or kv_cache_dtype == "auto"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(9, 0)

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER


class QuantSuiteAttentionImpl(AttentionImpl):
    """vLLM attention implementation bound to one prepared executor."""

    can_return_lse_for_decode: bool = False
    _quantized_attention: QuantizedAttention

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **extra_impl_args: Any,
    ) -> None:
        _reject_unsupported_features(
            "QuantSuite attention", alibi_slopes, sliding_window, logits_soft_cap
        )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "QuantSuite attention only supports decoder attention"
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale) if scale else 1.0 / math.sqrt(head_size)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.micro_batch_size = int(
            os.environ.get("QUANT_SUITE_MHA_MICRO_BATCH_SIZE", "8")
        )

    def bind_quantization(self, quantized_attention: QuantizedAttention) -> None:
        self._quantized_attention = quantized_attention

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlexAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output is None:
            raise NotImplementedError("QuantSuite attention requires an output buffer")
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "QuantSuite attention does not support fused output quantization"
            )
        if attn_metadata is None:
            return output.zero_()

        result = apply_quantized_attention(
            query,
            key,
            value,
            kv_cache,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
            self.num_kv_heads,
            self.head_size,
            self.num_heads,
            attn_metadata,
            self.scale,
            self._quantized_attention,
            micro_batch_size=self.micro_batch_size,
        )
        num_actual_tokens = attn_metadata.num_actual_tokens
        output[:num_actual_tokens].copy_(
            result.view(num_actual_tokens, self.num_heads, self.head_size)
        )
        return output


__all__ = [
    "apply_quantized_attention",
    "QuantSuiteAttentionBackend",
    "QuantSuiteAttentionImpl",
    "QuantSuiteAttentionMetadataBuilder",
]
