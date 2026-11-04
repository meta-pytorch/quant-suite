# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

import quant_suite_vllm.attention.mla.backend as mla_backend
import quant_suite_vllm.plugin as plugin
import torch
from quant_suite_vllm.attention.mla.backend import (
    QuantSuiteMLAAttentionBackend,
    QuantSuiteMLAAttentionImpl,
)
from quant_suite_vllm.plugin import register_attention_backend
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionType


class QuantSuiteMLAAttentionBackendTest(unittest.TestCase):
    def _create_impl(self) -> QuantSuiteMLAAttentionImpl:
        with patch.object(mla_backend.MLACommonImpl, "__init__", return_value=None):
            impl = QuantSuiteMLAAttentionImpl(
                num_heads=4,
                head_size=48,
                scale=0.5,
                num_kv_heads=1,
                alibi_slopes=None,
                sliding_window=None,
                kv_cache_dtype="auto",
                logits_soft_cap=None,
                attn_type=AttentionType.DECODER,
                kv_sharing_target_layer_name=None,
            )

        impl.num_heads = 4
        impl.head_size = 48
        impl.kv_lora_rank = 32
        impl.qk_head_dim = 32
        impl.v_head_dim = 16
        impl.qk_rope_head_dim = 16
        impl.scale = 0.5
        impl._quantized_attention = Mock()
        impl.W_UK_T = torch.randn(4, 16, 32)
        impl.W_UV = torch.randn(4, 32, 16)
        return impl

    def test_declares_supported_configurations(self) -> None:
        self.assertTrue(QuantSuiteMLAAttentionBackend.is_mla())
        self.assertTrue(QuantSuiteMLAAttentionBackend.supports_head_size(576))
        self.assertTrue(QuantSuiteMLAAttentionBackend.supports_kv_cache_dtype("auto"))
        self.assertFalse(
            QuantSuiteMLAAttentionBackend.supports_kv_cache_dtype("fp8_e4m3")
        )
        self.assertTrue(
            QuantSuiteMLAAttentionBackend.supports_compute_capability(
                DeviceCapability(9, 0)
            )
        )
        self.assertFalse(
            QuantSuiteMLAAttentionBackend.supports_compute_capability(
                DeviceCapability(8, 0)
            )
        )
        self.assertTrue(
            QuantSuiteMLAAttentionBackend.supports_attn_type(AttentionType.DECODER)
        )
        self.assertEqual(
            QuantSuiteMLAAttentionBackend.get_kv_cache_shape(3, 16, 1, 576),
            (3, 16, 576),
        )
        self.assertIs(
            QuantSuiteMLAAttentionBackend.get_impl_cls(),
            QuantSuiteMLAAttentionImpl,
        )
        self.assertFalse(QuantSuiteMLAAttentionImpl.supports_pcp)

    def test_rejects_fused_output_quantization(self) -> None:
        impl = self._create_impl()

        self.assertFalse(impl.fused_output_quant_supported(Mock()))
        with self.assertRaisesRegex(
            NotImplementedError,
            "fused output quantization",
        ):
            impl.forward_mha(
                q=torch.empty(0),
                kv_c_normed=torch.empty(0),
                k_pe=torch.empty(0),
                kv_c_and_k_pe_cache=torch.empty(0),
                attn_metadata=Mock(),
                k_scale=torch.tensor(1.0),
                output=torch.empty(0),
                output_scale=torch.tensor(1.0),
            )

    def test_rejects_unsupported_features(self) -> None:
        def build(
            alibi_slopes: list[float] | None = None,
            sliding_window: int | None = None,
            logits_soft_cap: float | None = None,
        ) -> None:
            QuantSuiteMLAAttentionImpl(
                num_heads=4,
                head_size=48,
                scale=0.5,
                num_kv_heads=1,
                alibi_slopes=alibi_slopes,
                sliding_window=sliding_window,
                kv_cache_dtype="auto",
                logits_soft_cap=logits_soft_cap,
                attn_type=AttentionType.DECODER,
                kv_sharing_target_layer_name=None,
            )

        with patch.object(
            mla_backend.MLACommonImpl, "__init__", return_value=None
        ) as base_init:
            with self.assertRaisesRegex(NotImplementedError, "ALiBi slopes"):
                build(alibi_slopes=[1.0])
            with self.assertRaisesRegex(NotImplementedError, "sliding window"):
                build(sliding_window=128)
            with self.assertRaisesRegex(NotImplementedError, "logits soft cap"):
                build(logits_soft_cap=30.0)
            base_init.assert_not_called()

    def test_unprepared_state_fails_fast(self) -> None:
        impl = object.__new__(QuantSuiteMLAAttentionImpl)
        impl._quantized_attention = None

        with self.assertRaisesRegex(RuntimeError, "has not been prepared"):
            impl._prepared_quantization()

    def test_lazy_projection_extraction(self) -> None:
        impl = object.__new__(QuantSuiteMLAAttentionImpl)
        impl.W_UK_T = None
        impl.W_UV = None
        impl.kv_b_proj = Mock()
        impl.num_heads = 2
        impl.kv_lora_rank = 4
        impl.qk_nope_head_dim = 3
        impl.v_head_dim = 5
        weight = torch.nn.Parameter(torch.empty(1, dtype=torch.bfloat16))
        impl.kv_b_proj.parameters.return_value = iter([weight])
        expected_k = torch.randn(2, 3, 4)
        expected_v = torch.randn(2, 4, 5)

        with patch.object(
            mla_backend,
            "process_kv_b_proj_weights",
            return_value=(expected_k, expected_v),
        ) as extract:
            impl._ensure_weights_extracted()

        extract.assert_called_once_with(
            impl.kv_b_proj,
            torch.bfloat16,
            2,
            4,
            3,
            5,
        )
        self.assertIs(impl.W_UK_T, expected_k)
        self.assertIs(impl.W_UV, expected_v)

    def test_registration_selects_custom_mla(self) -> None:
        with (
            patch.dict(os.environ, {"QUANT_SUITE_ATTN_BACKEND": "CUSTOM_MLA"}),
            patch.object(
                plugin,
                "register_backend",
            ) as register_backend,
        ):
            register_attention_backend()

        register_backend.assert_called_once_with(
            plugin.AttentionBackendEnum.CUSTOM,
            "quant_suite_vllm.attention.mla.backend.QuantSuiteMLAAttentionBackend",
        )
