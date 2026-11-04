# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""QuantSuite MLA attention backend for vLLM."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from quant_suite_recipe.implementation.attention import QuantizedAttention
from quant_suite_vllm.attention.backend import _reject_unsupported_features
from quant_suite_vllm.attention.mla.implementation import (
    apply_quantized_mla_decode,
    apply_quantized_mla_prefill,
    process_kv_b_proj_weights,
)
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)


class QuantSuiteMLAAttentionBackend(AttentionBackend):
    """MLA backend for prepared QuantSuite attention recipes."""

    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kernel_block_sizes: ClassVar[list[int | MultipleOf]] = [
        MultipleOf(1),
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto"]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["QuantSuiteMLAAttentionImpl"]:
        return QuantSuiteMLAAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[MLACommonMetadataBuilder]:
        return MLACommonMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        return (1, 0, 2, 3) if include_num_layers_dimension else (0, 1, 2)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size > 0

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        return kv_cache_dtype in ["auto"]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(9, 0)

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER


class QuantSuiteMLAAttentionImpl(MLACommonImpl[MLACommonMetadata]):
    """Thin implementation delegating prefill and decode to QuantSuite MLA.

    Inherits from MLACommonImpl (not just MLAAttentionImpl) because
    MLACommonMetadataBuilder asserts isinstance(impl, MLACommonImpl).
    """

    can_return_lse_for_decode: bool = False
    supports_pcp: bool = False
    _quantized_attention: QuantizedAttention | None
    W_UK_T: torch.Tensor | None
    W_UV: torch.Tensor | None

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        **mla_args: Any,
    ) -> None:
        _reject_unsupported_features(
            "QuantSuite MLA", alibi_slopes, sliding_window, logits_soft_cap
        )
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )

        self._quantized_attention = None
        self.W_UK_T = None
        self.W_UV = None

    def bind_quantization(self, quantized_attention: QuantizedAttention) -> None:
        """Bind prepared pipelines during the quant-method post-load pass."""
        self._quantized_attention = quantized_attention

    def _ensure_weights_extracted(self) -> None:
        """Lazily extract W_UK_T and W_UV from kv_b_proj on first use.

        MLAAttention does not call this implementation's post-load hook on all
        supported paths, so forward keeps a lazy fallback.
        """
        if self.W_UK_T is not None:
            return
        assert self.kv_b_proj is not None, "kv_b_proj must be set before forward"
        act_dtype = next(self.kv_b_proj.parameters()).dtype
        self.W_UK_T, self.W_UV = process_kv_b_proj_weights(
            self.kv_b_proj,
            act_dtype,
            self.num_heads,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.v_head_dim,
        )

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        if self.kv_b_proj is None:
            return
        self.W_UK_T, self.W_UV = process_kv_b_proj_weights(
            self.kv_b_proj,
            act_dtype,
            self.num_heads,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.v_head_dim,
        )

    def _prepared_quantization(self) -> QuantizedAttention:
        quantized_attention = self._quantized_attention
        if quantized_attention is None:
            raise RuntimeError("MLA attention quantization has not been prepared")
        return quantized_attention

    # -----------------------------------------------------------------
    # forward_mha (prefill): full-head K/V after up-projection
    # -----------------------------------------------------------------
    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
    ) -> None:
        if output_scale is not None:
            raise NotImplementedError(
                "QuantSuite MLA does not support fused output quantization"
            )
        self._ensure_weights_extracted()
        assert self.W_UK_T is not None and self.W_UV is not None
        assert attn_metadata.prefill is not None
        quantized_attention = self._prepared_quantization()
        apply_quantized_mla_prefill(
            q=q,
            kv_c_and_k_pe_cache=kv_c_and_k_pe_cache,
            attn_metadata=attn_metadata,
            output=output,
            num_heads=self.num_heads,
            head_size=self.head_size,
            kv_lora_rank=self.kv_lora_rank,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            scale=self.scale,
            W_UK_T=self.W_UK_T,
            W_UV=self.W_UV,
            quantized_attention=quantized_attention,
        )

    def fused_output_quant_supported(self, quant_key: Any) -> bool:
        return False

    # -----------------------------------------------------------------
    # forward_mqa (decode): compressed K/V space
    # -----------------------------------------------------------------
    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        quantized_attention = self._prepared_quantization()
        return apply_quantized_mla_decode(
            q=q,
            kv_c_and_k_pe_cache=kv_c_and_k_pe_cache,
            attn_metadata=attn_metadata,
            num_heads=self.num_heads,
            head_size=self.head_size,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            scale=self.scale,
            quantized_attention=quantized_attention,
        )


__all__ = [
    "QuantSuiteMLAAttentionBackend",
    "QuantSuiteMLAAttentionImpl",
]
