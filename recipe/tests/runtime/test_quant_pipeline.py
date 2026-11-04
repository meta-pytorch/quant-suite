# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import unittest
from unittest import mock

import torch
import torch.nn as nn
from quant_suite_core.native_presets import PrecisionType
from quant_suite_core.rht_generator import HadamardMode, RHTGenerator
from quant_suite_recipe.api.implementation import QuantizedTensor
from quant_suite_recipe.definition import (
    FakeQuantRecipe,
    NativeQuantRecipe,
    NoQuantRecipe,
    PipelineRecipe,
    QuantAttentionRecipe,
    QuantMoERecipe,
    RHTRecipe,
)
from quant_suite_recipe.implementation.quant_pipeline import (
    Pipeline,
    PipelineSlice,
    QuantizeStage,
    RHTStage,
    split_pipeline,
)
from quant_suite_recipe.implementation.quantize_tensor import fake_quantize_tensor
from quant_suite_recipe.runtime import QuantRecipeRuntime


_BUILTIN_RUNTIME = QuantRecipeRuntime()


class QuantPipelineTest(unittest.TestCase):
    def test_create_pipeline_accepts_extracted_stage_recipe(self) -> None:
        recipe = _BUILTIN_RUNTIME.deserialize(
            {
                "linear_recipe": {
                    ".*": {
                        "weight": {
                            "type": "fake",
                            "quant_config": "nvfp4",
                            "use_triton": False,
                        }
                    }
                }
            }
        )
        quantize_stage = recipe.get_linear_recipe("layer").weight[0]
        pipeline = _BUILTIN_RUNTIME.create_pipeline(quantize_stage)

        result = _BUILTIN_RUNTIME.apply_pipeline(
            pipeline,
            torch.randn(2, 16, dtype=torch.bfloat16),
        )

        assert result.tensor is not None
        self.assertEqual((2, 16), result.tensor.shape)
        self.assertEqual(torch.bfloat16, result.tensor.dtype)

    def test_pipeline_stores_concrete_handler(self) -> None:
        recipe = _BUILTIN_RUNTIME.deserialize(
            {
                "linear_recipe": {
                    ".*": {
                        "weight": {
                            "type": "fake",
                            "quant_config": "nvfp4",
                            "use_triton": False,
                        }
                    }
                }
            }
        )
        pipeline = _BUILTIN_RUNTIME.create_pipeline(
            recipe.get_linear_recipe("layer").weight
        )

        stage = pipeline.stages[0]

        self.assertIsInstance(stage, QuantizeStage)
        assert isinstance(stage, QuantizeStage)
        self.assertIs(stage._quantize_tensor, fake_quantize_tensor)
        self.assertEqual("", stage.tensor_name)

    def test_quantize_stage_exposes_recipe_block_size(self) -> None:
        recipe = FakeQuantRecipe.from_dict(
            {"quant_config": "nvfp4", "use_triton": False}
        )
        pipeline = _BUILTIN_RUNTIME.create_pipeline(recipe)

        self.assertEqual(len(pipeline.stages), 1)
        stage = pipeline.stages[0]
        self.assertIsInstance(stage, QuantizeStage)
        assert isinstance(stage, QuantizeStage)
        self.assertEqual(stage.block_size, (1, 16))
        self.assertEqual(stage.block_size, recipe.block_size)

    def test_prepare_linear_sets_tensor_name_on_quantize_stage(self) -> None:
        recipe = _BUILTIN_RUNTIME.deserialize(
            {
                "linear_recipe": {
                    ".*": {
                        "weight": {
                            "type": "fake",
                            "quant_config": "nvfp4",
                            "use_triton": False,
                        }
                    }
                }
            }
        )

        quantized_linear = _BUILTIN_RUNTIME.prepare_linear(
            recipe.get_linear_recipe("layer")
        )

        pipeline = quantized_linear.weight_pipeline
        stage = pipeline.stages[0]
        self.assertIsInstance(stage, QuantizeStage)
        assert isinstance(stage, QuantizeStage)
        self.assertEqual("weight", stage.tensor_name)

    def test_prepare_moe_owns_named_pipelines(self) -> None:
        quantized_moe = _BUILTIN_RUNTIME.prepare_moe(QuantMoERecipe())
        tensor = torch.randn(2, 4)

        self.assertEqual(
            ["a1_pipeline", "a2_pipeline", "w13_pipeline", "w2_pipeline"],
            list(dict(quantized_moe.named_children())),
        )
        for pipeline in quantized_moe.children():
            self.assertIsInstance(pipeline, Pipeline)
            assert isinstance(pipeline, Pipeline)
            self.assertTrue(pipeline.is_noop())
        for quantize in (quantized_moe.quantize_a1, quantized_moe.quantize_a2):
            self.assertIs(quantize(tensor).tensor, tensor)

    def test_prepare_attention_owns_named_pipelines(self) -> None:
        quantized_attention = _BUILTIN_RUNTIME.prepare_attention(QuantAttentionRecipe())
        tensor = torch.randn(2, 4)

        self.assertEqual(
            [
                "q_pipeline",
                "k_pipeline",
                "v_pipeline",
                "attn_weights_pipeline",
                "recurrent_state_pipeline",
            ],
            list(dict(quantized_attention.named_children())),
        )
        for pipeline in quantized_attention.children():
            self.assertIsInstance(pipeline, Pipeline)
            assert isinstance(pipeline, Pipeline)
            self.assertTrue(pipeline.is_noop())
        for quantize in (
            quantized_attention.quantize_q,
            quantized_attention.quantize_k,
            quantized_attention.quantize_v,
            quantized_attention.quantize_attn_weights,
            quantized_attention.quantize_recurrent_state,
        ):
            self.assertIs(quantize(tensor).tensor, tensor)

    def test_attention_bmm_supports_transpose_and_output_dtype(self) -> None:
        quantized_attention = _BUILTIN_RUNTIME.prepare_attention(QuantAttentionRecipe())
        left = QuantizedTensor(tensor=torch.tensor([[[1.0, 2.0]]]))
        right = QuantizedTensor(tensor=torch.tensor([[[3.0, 4.0]]]))

        result = quantized_attention.bmm(
            left,
            right,
            transpose_right=True,
            out_dtype=torch.float64,
        )

        torch.testing.assert_close(
            result, torch.tensor([[[11.0]]], dtype=torch.float64)
        )

    def test_prepare_attention_allows_native_fields_for_non_bmm_consumers(self) -> None:
        native_pipeline = PipelineRecipe([NativeQuantRecipe(PrecisionType.NVFP4)])

        quantized_attention = _BUILTIN_RUNTIME.prepare_attention(
            QuantAttentionRecipe(q=native_pipeline)
        )

        self.assertIs(quantized_attention.recipe.q, native_pipeline)

    def test_create_pipeline_covers_attention_fields(self) -> None:
        recipe = _BUILTIN_RUNTIME.deserialize(
            {
                "attention_recipe": {
                    ".*": {
                        "q": "none",
                        "k": "none",
                        "v": "none",
                        "attn_weights": "none",
                        "recurrent_state": "none",
                    }
                }
            }
        )
        attention_recipe = recipe.get_attention_recipe("attention")
        tensor = torch.randn(2, 4)

        for field_name in QuantAttentionRecipe.pipeline_fields():
            with self.subTest(field_name=field_name):
                pipeline = _BUILTIN_RUNTIME.create_pipeline(
                    getattr(attention_recipe, field_name)
                )
                result = _BUILTIN_RUNTIME.apply_pipeline(pipeline, tensor)

                self.assertIs(result.tensor, tensor)

    def test_rht_round_trip_is_preserved(self) -> None:
        recipe = _BUILTIN_RUNTIME.deserialize(
            {
                "linear_recipe": {
                    ".*": {
                        "weight": [
                            {
                                "rht": {
                                    "type": "right",
                                    "block_size": 4,
                                    "seed": 7,
                                }
                            },
                            {
                                "rht": {
                                    "type": "right_inv",
                                    "block_size": 4,
                                    "seed": 7,
                                }
                            },
                        ]
                    }
                }
            }
        )
        pipeline = _BUILTIN_RUNTIME.create_pipeline(
            recipe.get_linear_recipe("layer").weight
        )
        tensor = torch.randn(2, 4)

        result = _BUILTIN_RUNTIME.apply_pipeline(pipeline, tensor)

        assert result.tensor is not None
        torch.testing.assert_close(result.tensor, tensor)

    def test_runtime_rejects_extension_recipe_payload(self) -> None:
        with self.assertRaisesRegex(ValueError, "unregistered_test_type"):
            _BUILTIN_RUNTIME.deserialize(
                {
                    "linear_recipe": {
                        ".*": {
                            "weight": {
                                "type": "unregistered_test_type",
                            }
                        }
                    }
                }
            )


class PipelineSliceTest(unittest.TestCase):
    def _reconstruct(
        self,
        tensor: torch.Tensor,
        pipeline_slice: PipelineSlice,
    ) -> QuantizedTensor:
        result = pipeline_slice.head.stages(QuantizedTensor(tensor=tensor))
        if pipeline_slice.qdq is not None:
            result = pipeline_slice.qdq(result)
        return pipeline_slice.tail.stages(result)

    def test_split_and_reconstruct_without_qdq(self) -> None:
        tensor = torch.randn(2, 16)
        stage = RHTStage(HadamardMode.RIGHT, block_size=16, seed=3)
        pipeline = Pipeline(stage)

        pipeline_slice = split_pipeline(pipeline)
        expected = pipeline(tensor)
        actual = self._reconstruct(tensor, pipeline_slice)

        self.assertEqual(tuple(pipeline_slice.head.stages), (stage,))
        self.assertIsNone(pipeline_slice.qdq)
        self.assertTrue(pipeline_slice.tail.is_noop())
        assert expected.tensor is not None and actual.tensor is not None
        self.assertTrue(torch.equal(expected.tensor, actual.tensor))

    def test_split_and_reconstruct_single_qdq(self) -> None:
        tensor = torch.randn(2, 32)
        pipeline = _BUILTIN_RUNTIME.create_pipeline(
            FakeQuantRecipe.from_dict({"quant_config": "mxfp8", "use_triton": False})
        )
        qdq = pipeline.stages[0]
        self.assertIsInstance(qdq, QuantizeStage)
        assert isinstance(qdq, QuantizeStage)

        pipeline_slice = split_pipeline(pipeline)
        expected = pipeline(tensor)
        actual = self._reconstruct(tensor, pipeline_slice)

        self.assertTrue(pipeline_slice.head.is_noop())
        self.assertIs(pipeline_slice.qdq, qdq)
        self.assertTrue(pipeline_slice.tail.is_noop())
        assert expected.tensor is not None and actual.tensor is not None
        self.assertTrue(torch.equal(expected.tensor, actual.tensor))

    def test_split_selects_final_qdq_with_tail_and_reconstructs(self) -> None:
        tensor = torch.randn(2, 32)
        recipe = FakeQuantRecipe.from_dict(
            {"quant_config": "mxfp8", "use_triton": False}
        )
        first_qdq = _BUILTIN_RUNTIME.create_pipeline(recipe).stages[0]
        final_qdq = _BUILTIN_RUNTIME.create_pipeline(recipe).stages[0]
        assert isinstance(first_qdq, QuantizeStage)
        assert isinstance(final_qdq, QuantizeStage)
        rht = RHTStage(HadamardMode.RIGHT, block_size=16, seed=5)
        tail = RHTStage(HadamardMode.RIGHT_INV, block_size=16, seed=5)
        pipeline = Pipeline(first_qdq, rht, final_qdq, tail)

        pipeline_slice = split_pipeline(pipeline)
        expected = pipeline(tensor)
        actual = self._reconstruct(tensor, pipeline_slice)

        self.assertEqual(tuple(pipeline_slice.head.stages), (first_qdq, rht))
        self.assertIs(pipeline_slice.qdq, final_qdq)
        self.assertEqual(tuple(pipeline_slice.tail.stages), (tail,))
        assert expected.tensor is not None and actual.tensor is not None
        self.assertTrue(torch.equal(expected.tensor, actual.tensor))


class RHTStageTest(unittest.TestCase):
    def setUp(self) -> None:
        self._device_patcher = mock.patch(
            "quant_suite_core.rht_generator._default_device",
            return_value=torch.device("cpu"),
        )
        self._device_patcher.start()
        self.addCleanup(self._device_patcher.stop)
        self.addCleanup(RHTGenerator.clear_cache)
        RHTGenerator.clear_cache()

    def _rht_stage(
        self, mode: HadamardMode, block_size: int = 16, seed: int = 7
    ) -> RHTStage:
        return RHTStage(mode=mode, block_size=block_size, seed=seed)

    def test_build_pipeline_skips_noops_and_keeps_rht(self) -> None:
        pipeline = _BUILTIN_RUNTIME.create_pipeline(
            PipelineRecipe(
                [
                    RHTRecipe(mode=HadamardMode.RIGHT, block_size=16, seed=1),
                    NoQuantRecipe(),
                    RHTRecipe(mode=HadamardMode.NONE),
                    RHTRecipe(
                        mode=HadamardMode.RIGHT_INV,
                        block_size=16,
                        seed=1,
                    ),
                ]
            )
        )

        self.assertEqual(len(pipeline.stages), 2)
        stages = list(pipeline.stages)
        for stage in stages:
            self.assertIsInstance(stage, RHTStage)
        self.assertEqual(
            [stage.mode for stage in stages if isinstance(stage, RHTStage)],
            [HadamardMode.RIGHT, HadamardMode.RIGHT_INV],
        )

    def test_rht_stage_maps_mode_to_transform_direction(self) -> None:
        tensor = torch.randn(16, 16, dtype=torch.float32)
        cases = (
            (HadamardMode.RIGHT, False, False),
            (HadamardMode.RIGHT_INV, False, True),
            (HadamardMode.LEFT, True, False),
            (HadamardMode.LEFT_INV, True, True),
        )

        for mode, is_left, inverse in cases:
            with (
                self.subTest(mode=mode),
                mock.patch(
                    "quant_suite_recipe.implementation.quant_pipeline.apply_block_hadamard",
                    return_value=tensor,
                ) as transform,
            ):
                result = self._rht_stage(mode)(QuantizedTensor(tensor=tensor))

                self.assertIs(result.tensor, tensor)
                transform.assert_called_once_with(
                    tensor,
                    mock.ANY,
                    is_left=is_left,
                    inverse=inverse,
                )

    def test_rht_stage_buffer_is_non_trainable_singleton(self) -> None:
        stage = self._rht_stage(HadamardMode.RIGHT)
        inverse = self._rht_stage(HadamardMode.RIGHT_INV)

        self.assertEqual(list(stage.parameters()), [])
        buffers = list(stage.buffers())
        self.assertEqual(len(buffers), 1)
        self.assertFalse(buffers[0].requires_grad)
        self.assertIs(stage._generator, inverse._generator)
        self.assertIs(stage._generator.get_matrix(), inverse._generator.get_matrix())

    def test_rht_left_round_trip_is_identity(self) -> None:
        tensor = torch.randn(16, 4, dtype=torch.float32)
        pipeline = Pipeline(
            self._rht_stage(HadamardMode.LEFT),
            self._rht_stage(HadamardMode.LEFT_INV),
        )

        result = pipeline(tensor)

        assert result.tensor is not None
        self.assertTrue(torch.allclose(result.tensor, tensor, atol=1e-3))

    def test_rht_single_stage_transforms_tensor(self) -> None:
        tensor = torch.randn(2, 16, dtype=torch.float32)
        result = self._rht_stage(HadamardMode.RIGHT)(QuantizedTensor(tensor=tensor))

        assert result.tensor is not None
        self.assertEqual(result.tensor.shape, tensor.shape)
        self.assertFalse(torch.allclose(result.tensor, tensor, atol=1e-3))

    def test_rht_stage_is_dynamo_traceable(self) -> None:
        class _RHTModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.stage = RHTStage(
                    mode=HadamardMode.RIGHT,
                    block_size=16,
                    seed=7,
                )

            def forward(self, tensor: torch.Tensor) -> torch.Tensor:
                result = self.stage(QuantizedTensor(tensor=tensor))
                assert result.tensor is not None
                return result.tensor

        model = _RHTModel()
        tensor = torch.randn(4, 16, dtype=torch.float32)

        eager = model(tensor)
        compiled = torch.compile(model, fullgraph=True, backend="eager")(tensor)

        self.assertTrue(torch.equal(compiled, eager))

    def test_rht_recipe_serialization_round_trip(self) -> None:
        config = RHTRecipe(mode=HadamardMode.RIGHT_INV, block_size=128, seed=2026)

        serialized = config.to_dict()
        serialized_rht = serialized["rht"]
        assert isinstance(serialized_rht, dict)
        restored = RHTRecipe.from_dict(serialized_rht)

        self.assertEqual(
            serialized,
            {"rht": {"type": "right_inv", "block_size": 128, "seed": 2026}},
        )
        self.assertEqual(restored, config)

    def test_rht_recipe_from_dict_is_case_insensitive(self) -> None:
        restored = RHTRecipe.from_dict({"type": "RIGHT_INV", "block_size": 64})

        self.assertEqual(restored.mode, HadamardMode.RIGHT_INV)
        self.assertEqual(restored.block_size, 64)
        self.assertIsNone(restored.seed)

    def test_rht_recipe_rejects_abbreviated_mode(self) -> None:
        for short in ("r", "l", "r_inv", "l_inv"):
            with self.assertRaises(ValueError):
                RHTRecipe.from_dict({"type": short, "block_size": 64})

    def test_rht_recipe_rejects_degenerate_block_size(self) -> None:
        for block_size in (0, 1):
            with self.assertRaisesRegex(ValueError, "block_size must be >= 2"):
                RHTRecipe(mode=HadamardMode.RIGHT, block_size=block_size)
        self.assertTrue(RHTRecipe(mode=HadamardMode.NONE).is_noop())
