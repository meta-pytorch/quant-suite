# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import torch
from quant_suite_core.native_presets import PrecisionType
from quant_suite_recipe.definition import (
    NativeQuantRecipe,
    PipelineRecipe,
    QuantAttentionRecipe,
    QuantRecipe,
)
from quant_suite_recipe.runtime_provider import QUANT_RECIPE_RUNTIME
from quant_suite_vllm.attention import backend as attention_backend
from quant_suite_vllm.attention.backend import (
    QuantSuiteAttentionBackend,
    QuantSuiteAttentionImpl,
    QuantSuiteAttentionMetadataBuilder,
)
from quant_suite_vllm.attention.method import QuantSuiteAttentionMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType
from vllm.v1.attention.backends.flex_attention import (
    FlexAttentionBackend,
    FlexAttentionMetadata,
    FlexAttentionMetadataBuilder,
)


class QuantSuiteAttentionBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.quant_config = cast(QuantizationConfig, Mock(spec=QuantizationConfig))

    def test_declares_supported_configurations(self) -> None:
        self.assertTrue(
            issubclass(
                QuantSuiteAttentionMetadataBuilder,
                FlexAttentionMetadataBuilder,
            )
        )
        self.assertIs(
            QuantSuiteAttentionMetadataBuilder._cudagraph_support,
            AttentionCGSupport.NEVER,
        )
        self.assertTrue(QuantSuiteAttentionBackend.supports_head_size(128))
        self.assertFalse(QuantSuiteAttentionBackend.supports_head_size(130))
        self.assertTrue(QuantSuiteAttentionBackend.supports_kv_cache_dtype("auto"))
        self.assertFalse(QuantSuiteAttentionBackend.supports_kv_cache_dtype("fp8_e4m3"))
        self.assertTrue(
            QuantSuiteAttentionBackend.supports_compute_capability(
                DeviceCapability(9, 0)
            )
        )
        self.assertFalse(
            QuantSuiteAttentionBackend.supports_compute_capability(
                DeviceCapability(8, 0)
            )
        )
        self.assertTrue(
            QuantSuiteAttentionBackend.supports_attn_type(AttentionType.DECODER)
        )
        self.assertTrue(QuantSuiteAttentionBackend.supports_dtype(torch.bfloat16))
        self.assertTrue(QuantSuiteAttentionBackend.supports_dtype(torch.float32))
        self.assertTrue(QuantSuiteAttentionBackend.supports_dtype(torch.float16))
        get_shape = getattr(FlexAttentionBackend, "get_kv_cache_shape", None)
        expected_cache_shape = (
            (3, 2, 4, 16) if get_shape is None else get_shape(3, 4, 2, 8, "auto")
        )
        self.assertEqual(
            QuantSuiteAttentionBackend.get_kv_cache_shape(3, 4, 2, 8),
            expected_cache_shape,
        )
        supported_layouts = QuantSuiteAttentionBackend.supported_kv_cache_layouts()
        if supported_layouts is not None:
            self.assertEqual(supported_layouts[0].name, "LBNHC")

    def test_binds_prepared_attention_once(self) -> None:
        impl = QuantSuiteAttentionImpl(
            num_heads=2,
            head_size=8,
            scale=0.5,
        )
        quantized_attention = QUANT_RECIPE_RUNTIME.prepare_attention(
            QuantAttentionRecipe()
        )

        impl.bind_quantization(quantized_attention)

        self.assertIs(impl._quantized_attention, quantized_attention)

    def test_method_prepares_and_binds_layer_attention(self) -> None:
        layer = torch.nn.Module()
        dynamic_layer: Any = layer
        dynamic_layer.impl = QuantSuiteAttentionImpl(
            num_heads=2,
            head_size=8,
            scale=0.5,
        )
        recipe = QuantAttentionRecipe()
        quant_recipe = QuantRecipe(
            attention_recipe={"model.layers.0.self_attn": recipe}
        )
        method = QuantSuiteAttentionMethod(
            quant_recipe,
            "model.layers.0.self_attn",
            quant_config=self.quant_config,
        )

        method.process_weights_after_loading(layer)

        self.assertIs(method.attention_recipe, recipe)
        self.assertIs(
            dynamic_layer.impl._quantized_attention,
            dynamic_layer.quantized_attention,
        )

    def test_method_allows_omitted_attention_recipe_with_native_backend(self) -> None:
        layer = torch.nn.Module()
        dynamic_layer: Any = layer
        dynamic_layer.impl = object()
        method = QuantSuiteAttentionMethod(
            QuantRecipe(),
            "model.layers.0.self_attn",
            quant_config=self.quant_config,
        )

        method.process_weights_after_loading(layer)

        self.assertFalse(hasattr(dynamic_layer, "quantized_attention"))

    def test_method_rejects_explicit_no_quant_with_native_backend(self) -> None:
        recipe = QUANT_RECIPE_RUNTIME.deserialize(
            {
                "attention_recipe": {
                    ".*": dict.fromkeys(("q", "k", "v", "attn_weights"), "none")
                }
            }
        )
        layer = torch.nn.Module()
        dynamic_layer: Any = layer
        dynamic_layer.impl = object()
        method = QuantSuiteAttentionMethod(
            recipe,
            "model.layers.0.self_attn",
            quant_config=self.quant_config,
        )

        with self.assertRaisesRegex(
            TypeError,
            "requires a QuantSuite attention backend, got object",
        ):
            method.process_weights_after_loading(layer)

    def test_method_rejects_active_attention_recipe_with_native_backend(self) -> None:
        for field_name in ("q", "k", "v", "attn_weights"):
            with self.subTest(field=field_name):
                recipe = QUANT_RECIPE_RUNTIME.deserialize(
                    {"attention_recipe": {".*": {field_name: "fake_mxfp4"}}}
                )
                layer = torch.nn.Module()
                dynamic_layer: Any = layer
                dynamic_layer.impl = object()
                method = QuantSuiteAttentionMethod(
                    recipe,
                    "model.layers.0.self_attn",
                    quant_config=self.quant_config,
                )

                with self.assertRaisesRegex(
                    TypeError,
                    "requires a QuantSuite attention backend, got object",
                ):
                    method.process_weights_after_loading(layer)

    def test_method_allows_recurrent_state_recipe_with_native_backend(self) -> None:
        recipe = QUANT_RECIPE_RUNTIME.deserialize(
            {"attention_recipe": {".*": {"recurrent_state": "fake_mxfp4"}}}
        )
        layer = torch.nn.Module()
        dynamic_layer: Any = layer
        dynamic_layer.impl = object()
        method = QuantSuiteAttentionMethod(
            recipe,
            "model.layers.0.self_attn",
            quant_config=self.quant_config,
        )

        method.process_weights_after_loading(layer)

        self.assertFalse(hasattr(dynamic_layer, "quantized_attention"))

    def test_method_rejects_native_attention_pipelines(self) -> None:
        native_pipeline = PipelineRecipe([NativeQuantRecipe(PrecisionType.NVFP4)])
        cases: tuple[tuple[str, QuantAttentionRecipe], ...] = (
            ("q", QuantAttentionRecipe(q=native_pipeline)),
            ("k", QuantAttentionRecipe(k=native_pipeline)),
            ("v", QuantAttentionRecipe(v=native_pipeline)),
            (
                "attn_weights",
                QuantAttentionRecipe(attn_weights=native_pipeline),
            ),
        )
        for field_name, recipe in cases:
            with (
                self.subTest(field=field_name),
                self.assertRaisesRegex(NotImplementedError, field_name),
            ):
                QuantSuiteAttentionMethod(
                    QuantRecipe(attention_recipe={".*": recipe}),
                    "model.layers.0.self_attn",
                    quant_config=self.quant_config,
                )

    def test_impl_rejects_unsupported_features(self) -> None:
        def build(
            alibi_slopes: list[float] | None = None,
            sliding_window: int | None = None,
            logits_soft_cap: float | None = None,
        ) -> None:
            QuantSuiteAttentionImpl(
                num_heads=2,
                head_size=8,
                scale=0.5,
                alibi_slopes=alibi_slopes,
                sliding_window=sliding_window,
                logits_soft_cap=logits_soft_cap,
            )

        with self.assertRaisesRegex(NotImplementedError, "ALiBi slopes"):
            build(alibi_slopes=[1.0])
        with self.assertRaisesRegex(NotImplementedError, "sliding window"):
            build(sliding_window=128)
        with self.assertRaisesRegex(NotImplementedError, "logits soft cap"):
            build(logits_soft_cap=30.0)

        impl = QuantSuiteAttentionImpl(
            num_heads=2,
            head_size=8,
            scale=0.5,
            logits_soft_cap=0.0,
            kv_sharing_target_layer_name="model.layers.0.self_attn",
        )
        self.assertEqual(
            impl.kv_sharing_target_layer_name,
            "model.layers.0.self_attn",
        )

        with self.assertRaisesRegex(NotImplementedError, "decoder attention"):
            QuantSuiteAttentionImpl(
                num_heads=2,
                head_size=8,
                scale=0.5,
                attn_type=AttentionType.ENCODER,
            )

    def test_no_quantization_preserves_causal_attention(self) -> None:
        quantized_attention = QUANT_RECIPE_RUNTIME.prepare_attention(
            QuantAttentionRecipe()
        )
        query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        key = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        value = torch.tensor([[[2.0, 3.0], [5.0, 7.0]]])

        result = attention_backend._quantized_attention_for_batch(
            [query],
            [key],
            [value],
            [2],
            [2],
            [0],
            scale=1.0,
            quantized_attention=quantized_attention,
        )

        scores = torch.bmm(query, key.transpose(-1, -2))
        causal_mask = torch.tensor([[True, False], [True, True]])
        weights = torch.nn.functional.softmax(
            scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf")),
            dim=-1,
        )
        expected = torch.bmm(weights, value).transpose(0, 1).contiguous()
        torch.testing.assert_close(result, expected)

    def test_full_attention_supports_single_request_gqa(self) -> None:
        query = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ]
        )
        key_cache = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
        value_cache = torch.tensor([[[[2.0, 3.0]], [[5.0, 7.0]]]])
        metadata = self._metadata(
            query_start_loc=[0, 2],
            seq_lens=[2],
            block_table=[[0]],
            decode_offsets=[0],
        )
        quantized_attention = QUANT_RECIPE_RUNTIME.prepare_attention(
            QuantAttentionRecipe()
        )
        repeated_key = key_cache[0].repeat_interleave(2, dim=1).transpose(0, 1)
        repeated_value = value_cache[0].repeat_interleave(2, dim=1).transpose(0, 1)
        expected = self._causal_attention(
            query.transpose(0, 1),
            repeated_key,
            repeated_value,
            decode_offset=0,
        )
        caches = {
            "legacy": torch.stack((key_cache, value_cache)),
            "logical_lbnhc": torch.cat((key_cache, value_cache), dim=-1).transpose(
                1, 2
            ),
        }
        for layout, kv_cache in caches.items():
            with self.subTest(layout=layout):
                bmm = Mock(wraps=quantized_attention.bmm)
                with (
                    patch.object(quantized_attention, "bmm", bmm),
                    patch(
                        "quant_suite_vllm.attention.backend.torch.ops._C_cache_ops.reshape_and_cache_flash"
                    ) as cache_update,
                ):
                    result = attention_backend.apply_quantized_attention(
                        query,
                        torch.empty(2, 1, 2),
                        torch.empty(2, 1, 2),
                        kv_cache,
                        "auto",
                        torch.tensor(1.0),
                        torch.tensor(1.0),
                        1,
                        2,
                        2,
                        metadata,
                        1.0,
                        quantized_attention,
                        micro_batch_size=1,
                    )

                torch.testing.assert_close(result, expected)
                cache_update.assert_called_once()
                self.assertEqual(bmm.call_count, 2)
                self.assertTrue(bmm.call_args_list[0].kwargs["transpose_right"])
                self.assertEqual(
                    bmm.call_args_list[0].kwargs["out_dtype"], torch.float32
                )

    def test_full_attention_batches_requests_with_padded_offsets(self) -> None:
        query = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
        key_cache = torch.tensor(
            [
                [[[1.0, 0.0]], [[0.0, 0.0]]],
                [[[1.0, 0.0]], [[0.0, 1.0]]],
            ]
        )
        value_cache = torch.tensor(
            [
                [[[2.0, 3.0]], [[0.0, 0.0]]],
                [[[5.0, 7.0]], [[11.0, 13.0]]],
            ]
        )
        metadata = self._metadata(
            query_start_loc=[0, 1, 2, 2, 2],
            seq_lens=[1, 2, 0, 0],
            block_table=[[0], [1], [0], [0]],
            decode_offsets=[0, 1],
            num_reqs=2,
        )
        quantized_attention = QUANT_RECIPE_RUNTIME.prepare_attention(
            QuantAttentionRecipe()
        )
        runtime_bmm = Mock(wraps=quantized_attention.bmm)

        with (
            patch.object(quantized_attention, "bmm", runtime_bmm),
            patch(
                "quant_suite_vllm.attention.backend.torch.ops._C_cache_ops.reshape_and_cache_flash"
            ),
        ):
            result = attention_backend.apply_quantized_attention(
                query,
                torch.empty(2, 1, 2),
                torch.empty(2, 1, 2),
                torch.stack((key_cache, value_cache)),
                "auto",
                torch.tensor(1.0),
                torch.tensor(1.0),
                1,
                2,
                1,
                metadata,
                1.0,
                quantized_attention,
                micro_batch_size=2,
            )

        expected = torch.cat(
            [
                self._causal_attention(
                    query[0:1].transpose(0, 1),
                    key_cache[0, :1].transpose(0, 1),
                    value_cache[0, :1].transpose(0, 1),
                    decode_offset=0,
                ),
                self._causal_attention(
                    query[1:2].transpose(0, 1),
                    key_cache[1].transpose(0, 1),
                    value_cache[1].transpose(0, 1),
                    decode_offset=1,
                ),
            ],
            dim=0,
        )
        torch.testing.assert_close(result, expected)
        self.assertEqual(runtime_bmm.call_count, 2)

    def test_full_attention_rejects_missing_query_offset(self) -> None:
        metadata = self._metadata(
            query_start_loc=[0, 1],
            seq_lens=[1, 1],
            block_table=[[0], [0]],
            decode_offsets=[0, 0],
            num_reqs=2,
        )

        with self.assertRaisesRegex(
            ValueError,
            "query_start_loc must be one-dimensional and contain at least 3 offsets",
        ):
            attention_backend.apply_quantized_attention(
                torch.empty(1, 1, 2),
                None,
                None,
                torch.empty(2, 1, 1, 1, 2),
                "auto",
                torch.tensor(1.0),
                torch.tensor(1.0),
                1,
                2,
                1,
                metadata,
                1.0,
                QUANT_RECIPE_RUNTIME.prepare_attention(QuantAttentionRecipe()),
            )

    def test_full_attention_preserves_common_metadata_fallbacks(self) -> None:
        query = torch.tensor([[[1.0, 0.0]]])
        key_cache = torch.tensor([[[[1.0, 0.0]]]])
        value_cache = torch.tensor([[[[2.0, 3.0]]]])
        metadata = SimpleNamespace(
            query_start_loc=torch.tensor([0, 1]),
            seq_lens=torch.tensor([1]),
            block_table_tensor=torch.tensor([[0]]),
            num_computed_tokens_cpu=torch.tensor([0]),
            slot_mapping=torch.tensor([0]),
            num_reqs=1,
            num_actual_tokens=1,
        )

        with patch(
            "quant_suite_vllm.attention.backend.torch.ops._C_cache_ops.reshape_and_cache_flash"
        ) as cache_update:
            result = attention_backend.apply_quantized_attention(
                query,
                None,
                None,
                torch.stack((key_cache, value_cache)),
                "auto",
                torch.tensor(1.0),
                torch.tensor(1.0),
                1,
                2,
                1,
                metadata,
                1.0,
                QUANT_RECIPE_RUNTIME.prepare_attention(QuantAttentionRecipe()),
                micro_batch_size=1,
            )

        expected = self._causal_attention(
            query.transpose(0, 1),
            key_cache[0].transpose(0, 1),
            value_cache[0].transpose(0, 1),
            decode_offset=0,
        )
        torch.testing.assert_close(result, expected)
        cache_update.assert_not_called()

    def test_forward_writes_preallocated_output(self) -> None:
        impl = QuantSuiteAttentionImpl(2, 8, 0.5)
        impl.bind_quantization(
            QUANT_RECIPE_RUNTIME.prepare_attention(QuantAttentionRecipe())
        )
        expected = torch.arange(32, dtype=torch.float32).view(2, 2, 8)
        output = torch.full((3, 2, 8), -1.0)
        layer = SimpleNamespace(
            _k_scale=torch.tensor(1.0),
            _v_scale=torch.tensor(1.0),
        )
        metadata = cast(
            FlexAttentionMetadata,
            SimpleNamespace(num_actual_tokens=2),
        )

        with patch(
            "quant_suite_vllm.attention.backend.apply_quantized_attention",
            return_value=expected,
        ):
            result = impl.forward(
                cast(Any, layer),
                torch.empty(2, 2, 8),
                torch.empty(2, 2, 8),
                torch.empty(2, 2, 8),
                torch.empty(2, 1, 1, 2, 8),
                metadata,
                output,
            )

        self.assertIs(result, output)
        torch.testing.assert_close(output[:2], expected)
        torch.testing.assert_close(output[2], torch.full((2, 8), -1.0))

    @staticmethod
    def _metadata(
        *,
        query_start_loc: list[int],
        seq_lens: list[int],
        block_table: list[list[int]],
        decode_offsets: list[int],
        num_reqs: int | None = None,
    ) -> FlexAttentionMetadata:
        return cast(
            FlexAttentionMetadata,
            SimpleNamespace(
                query_start_loc=torch.tensor(query_start_loc),
                seq_lens=torch.tensor(seq_lens),
                block_table=torch.tensor(block_table),
                decode_offset=torch.tensor(decode_offsets),
                slot_mapping=torch.arange(query_start_loc[-1]),
                block_size=2,
                num_reqs=len(seq_lens) if num_reqs is None else num_reqs,
                num_actual_tokens=query_start_loc[-1],
            ),
        )

    @staticmethod
    def _causal_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        decode_offset: int,
    ) -> torch.Tensor:
        scores = torch.bmm(query, key.transpose(-1, -2))
        query_indices = torch.arange(query.shape[1]).unsqueeze(1)
        key_indices = torch.arange(key.shape[1]).unsqueeze(0)
        mask = query_indices + decode_offset >= key_indices
        weights = torch.nn.functional.softmax(
            scores.masked_fill(~mask.unsqueeze(0), float("-inf")),
            dim=-1,
        )
        return torch.bmm(weights, value).transpose(0, 1).contiguous()
