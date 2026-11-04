# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for KDA (Kimi Delta Attention) recurrent state quantization.

Tests verify that:
1. quantize_kda_state with an empty pipeline leaves the state unchanged (CPU)
2. quantize_kda_state with fake mxfp4 modifies the state (GPU)
3. quantize_kda_state only modifies the active blocks (GPU)
4. quantize_kda_state raises when active_indices is empty (GPU)
5. The wrapped KDA forward correctly applies state
   quantization after _forward() completes (GPU, uses mock KDA layer)
"""

from __future__ import annotations

import inspect
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import quant_suite_vllm.kda.constructor_patch as kda_constructor_patch_module
import torch
from quant_suite_core import (
    BlockSize,
    ElemDtype,
    ElemRoundMode,
    GlobalScaleGranularity,
    QuantConfig,
    QuantMode,
    ScaleDtype,
    ScaleNormMode,
    ScaleRoundMode,
)
from quant_suite_core.native_presets import PrecisionType
from quant_suite_recipe.definition import (
    FakeQuantRecipe,
    NativeQuantRecipe,
    PipelineRecipe,
    QuantAttentionRecipe,
    QuantRecipe,
)
from quant_suite_recipe.implementation.attention import QuantizedAttention
from quant_suite_recipe.implementation.quant_pipeline import Pipeline
from quant_suite_recipe.runtime import QuantRecipeRuntime
from quant_suite_recipe.runtime_provider import QUANT_RECIPE_RUNTIME
from quant_suite_vllm.kda.constructor_patch import (
    _wrap_forward_with_state_quantization,
    register_constructor_patch_kda_quant_method,
)
from quant_suite_vllm.kda.implementation import quantize_kda_state
from quant_suite_vllm.kda.method import _attach_kda_method, QuantSuiteKDAMethod


def _create_mxfp4_quant_config() -> QuantConfig:
    """Create an MXFP4 fake quantization config for testing.

    Returns:
        QuantConfig for MXFP4 fake quantization (E2M1 with E8M0 scale)
    """
    return QuantConfig(
        quant_mode=QuantMode.SYMM,
        scale_dtype=ScaleDtype.E8M0,
        scale_norm_mode=ScaleNormMode.POW2MAX,
        scale_round_mode=ScaleRoundMode.OAS,
        global_scale_granularity=GlobalScaleGranularity.NONE,
        elem_dtype=ElemDtype.E2M1,
        elem_round_mode=ElemRoundMode.NEAREST,
        block_size=BlockSize(1, 16),
    )


def _empty_quantized_attention() -> QuantizedAttention:
    return QuantizedAttention(
        QuantAttentionRecipe(),
        Pipeline(),
        Pipeline(),
        Pipeline(),
        Pipeline(),
        Pipeline(),
    )


class QuantKDAStateCPUTest(unittest.TestCase):
    """CPU tests for quantize_kda_state with an empty pipeline.

    An empty pipeline is a pure Python no-op (no Triton kernels), so it can
    run on CPU.
    """

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.num_heads = 8
        self.head_dim = 32
        self.num_blocks = 4

    def test_no_quant_leaves_state_unchanged(self) -> None:
        """With NoQuantRecipe, quantize_kda_state should be a no-op."""
        state = torch.randn(
            self.num_blocks,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            dtype=torch.bfloat16,
        )
        state_before = state.clone()

        quantize_kda_state(state, Pipeline(), torch.arange(self.num_blocks))

        self.assertTrue(
            torch.equal(state, state_before),
            "State should be unchanged with an empty pipeline",
        )

    def test_no_quant_single_block(self) -> None:
        """NoQuant with num_blocks=1 (common decode case)."""
        state = torch.randn(
            1,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            dtype=torch.bfloat16,
        )
        state_before = state.clone()

        quantize_kda_state(state, Pipeline(), torch.tensor([0]))

        self.assertTrue(torch.equal(state, state_before))

    def test_active_indices_is_required(self) -> None:
        parameter = inspect.signature(quantize_kda_state).parameters["active_indices"]
        self.assertIs(parameter.default, inspect.Parameter.empty)


class QuantSuiteKDAMethodTest(unittest.TestCase):
    def test_registration_attaches_method_once_during_layer_init(self) -> None:
        class _QuantConfig:
            def __init__(self) -> None:
                self.quant_recipe = QuantRecipe(
                    attention_recipe={".*": QuantAttentionRecipe()}
                )

        class _KDALayer(torch.nn.Module):
            init_calls = 0

            def __init__(
                self,
                *,
                quant_config: object | None = None,
                prefix: str = "",
            ) -> None:
                super().__init__()
                type(self).init_calls += 1
                self.prefix = prefix

        config = _QuantConfig()
        with patch.object(
            kda_constructor_patch_module,
            "KimiDeltaAttention",
            _KDALayer,
        ):
            register_constructor_patch_kda_quant_method(_QuantConfig)
            register_constructor_patch_kda_quant_method(_QuantConfig)
            layer = _KDALayer(quant_config=config, prefix="model.kda")

        self.assertEqual(_KDALayer.init_calls, 1)
        self.assertIsInstance(layer.quant_method, QuantSuiteKDAMethod)
        self.assertIs(
            layer.quant_method.attention_recipe,
            config.quant_recipe.get_attention_recipe("model.kda"),
        )

    def test_registration_is_skipped_without_a_compatible_kda_layer(self) -> None:
        class _QuantConfig:
            pass

        with patch.object(
            kda_constructor_patch_module,
            "KimiDeltaAttention",
            None,
        ):
            registered = register_constructor_patch_kda_quant_method(_QuantConfig)

        self.assertFalse(registered)

    def test_registration_accepts_an_additional_config_type(self) -> None:
        class _FirstQuantConfig:
            pass

        class _SecondQuantConfig:
            def __init__(self) -> None:
                self.quant_recipe = QuantRecipe(
                    attention_recipe={".*": QuantAttentionRecipe()}
                )

        class _KDALayer(torch.nn.Module):
            def __init__(
                self,
                *,
                quant_config: object | None = None,
                prefix: str = "",
            ) -> None:
                super().__init__()

        config = _SecondQuantConfig()
        with patch.object(
            kda_constructor_patch_module,
            "KimiDeltaAttention",
            _KDALayer,
        ):
            register_constructor_patch_kda_quant_method(_FirstQuantConfig)
            register_constructor_patch_kda_quant_method(_SecondQuantConfig)
            layer = _KDALayer(quant_config=config, prefix="model.kda")

        self.assertIsInstance(layer.quant_method, QuantSuiteKDAMethod)
        self.assertIs(
            layer.quant_method.attention_recipe,
            config.quant_recipe.get_attention_recipe("model.kda"),
        )

    def test_registration_leaves_other_quantization_configs_unchanged(self) -> None:
        class _QuantConfig:
            def __init__(self) -> None:
                self.quant_recipe = QuantRecipe()

        class _OtherQuantConfig:
            pass

        class _KDALayer(torch.nn.Module):
            init_calls = 0

            def __init__(
                self,
                *,
                quant_config: object | None = None,
                prefix: str = "",
            ) -> None:
                super().__init__()
                type(self).init_calls += 1

        with patch.object(
            kda_constructor_patch_module,
            "KimiDeltaAttention",
            _KDALayer,
        ):
            register_constructor_patch_kda_quant_method(_QuantConfig)
            layer = _KDALayer(
                quant_config=_OtherQuantConfig(),
                prefix="model.kda",
            )

        self.assertEqual(_KDALayer.init_calls, 1)
        self.assertFalse(hasattr(layer, "quant_method"))

    def test_rejects_native_recurrent_state_at_construction(self) -> None:
        native_pipeline = PipelineRecipe([NativeQuantRecipe(PrecisionType.NVFP4)])
        recipe = QuantRecipe(
            attention_recipe={
                ".*": QuantAttentionRecipe(recurrent_state=native_pipeline)
            }
        )

        with self.assertRaisesRegex(
            NotImplementedError,
            "supports fake quantization only",
        ):
            QuantSuiteKDAMethod(recipe, "model.kda")

    def test_rejects_non_recurrent_attention_recipe_fields_at_construction(
        self,
    ) -> None:
        fake_pipeline = PipelineRecipe(
            [FakeQuantRecipe(quant_config=_create_mxfp4_quant_config())]
        )
        recipe = QuantRecipe(
            attention_recipe={
                ".*": QuantAttentionRecipe(
                    q=fake_pipeline,
                    recurrent_state=fake_pipeline,
                )
            }
        )

        with self.assertRaisesRegex(NotImplementedError, "unsupported.*q"):
            QuantSuiteKDAMethod(recipe, "model.kda")

    def test_rejects_speculative_recurrent_state_quantization(self) -> None:
        fake_pipeline = PipelineRecipe(
            [FakeQuantRecipe(quant_config=_create_mxfp4_quant_config())]
        )
        method = QuantSuiteKDAMethod(
            QuantRecipe(
                attention_recipe={
                    ".*": QuantAttentionRecipe(recurrent_state=fake_pipeline)
                }
            ),
            "model.kda",
        )
        layer = torch.nn.Module()
        object.__setattr__(layer, "num_spec", 1)

        with self.assertRaisesRegex(NotImplementedError, "speculative decoding"):
            method.process_weights_after_loading(layer)

    def test_rejects_recoverssm_recurrent_state_quantization(self) -> None:
        fake_pipeline = PipelineRecipe(
            [FakeQuantRecipe(quant_config=_create_mxfp4_quant_config())]
        )
        method = QuantSuiteKDAMethod(
            QuantRecipe(
                attention_recipe={
                    ".*": QuantAttentionRecipe(recurrent_state=fake_pipeline)
                }
            ),
            "model.kda",
        )
        layer = torch.nn.Module()
        object.__setattr__(layer, "use_recoverssm", True)

        with self.assertRaisesRegex(NotImplementedError, "RecoverSSM"):
            method.process_weights_after_loading(layer)

    def test_prepares_state_without_wrapping_forward(self) -> None:
        attention_recipe = QuantAttentionRecipe(
            recurrent_state=PipelineRecipe(
                [FakeQuantRecipe(quant_config=_create_mxfp4_quant_config())]
            )
        )
        runtime = Mock(spec=QuantRecipeRuntime)
        quantized_attention = QUANT_RECIPE_RUNTIME.prepare_attention(attention_recipe)
        runtime.prepare_attention.return_value = quantized_attention
        method = QuantSuiteKDAMethod(
            QuantRecipe(attention_recipe={".*": attention_recipe}),
            "model.kda",
        )
        layer: Any = torch.nn.Module()
        original_forward = Mock()
        layer._forward = original_forward

        with patch("quant_suite_vllm.kda.method.QUANT_RECIPE_RUNTIME", runtime):
            method.process_weights_after_loading(layer)

        runtime.prepare_attention.assert_called_once_with(attention_recipe)
        self.assertIs(layer.quantized_attention, quantized_attention)
        self.assertIs(layer._forward, original_forward)

    def test_repeated_process_weights_after_loading_wraps_forward_once(
        self,
    ) -> None:
        layer = self._build_quantized_kda_layer()
        wrapped_forward = layer._forward
        first = _empty_quantized_attention()
        second = _empty_quantized_attention()
        runtime = Mock(spec=QuantRecipeRuntime)
        runtime.prepare_attention.side_effect = [first, second]

        with patch("quant_suite_vllm.kda.method.QUANT_RECIPE_RUNTIME", runtime):
            layer.quant_method.process_weights_after_loading(layer)
            layer.quant_method.process_weights_after_loading(layer)

        self.assertIs(layer._forward, wrapped_forward)
        self.assertIs(layer.quantized_attention, second)
        with patch(
            "quant_suite_vllm.kda.constructor_patch.quantize_kda_state"
        ) as quantize_state:
            self._call_forward(layer)

        self.assertEqual(runtime.prepare_attention.call_count, 2)
        self.assertEqual(layer.forward_calls, 1)
        quantize_state.assert_called_once()
        self.assertIs(quantize_state.call_args.args[0], layer.kv_cache[0][3])
        self.assertIs(
            quantize_state.call_args.args[1],
            second.recurrent_state_pipeline,
        )

    def test_wrapped_forward_before_process_weights_after_loading_raises(
        self,
    ) -> None:
        layer = self._build_quantized_kda_layer()

        with self.assertRaisesRegex(
            RuntimeError,
            "process_weights_after_loading has not run",
        ):
            self._call_forward(layer)

        self.assertEqual(layer.forward_calls, 0)

    def _build_quantized_kda_layer(self) -> Any:
        state_pipeline = PipelineRecipe(
            [FakeQuantRecipe(quant_config=_create_mxfp4_quant_config())]
        )

        class _QuantConfig:
            def __init__(self) -> None:
                self.quant_recipe = QuantRecipe(
                    attention_recipe={
                        ".*": QuantAttentionRecipe(recurrent_state=state_pipeline)
                    }
                )

        class _KDALayer(torch.nn.Module):
            def __init__(
                self,
                *,
                quant_config: object | None = None,
                prefix: str = "",
            ) -> None:
                super().__init__()
                self.prefix = prefix
                self.forward_calls = 0
                conv_state = torch.empty(0)
                self.kv_cache = [
                    (conv_state, conv_state, conv_state, torch.zeros(2, 1, 1, 1))
                ]

            def _forward(
                self,
                q_proj_states: torch.Tensor,
                k_proj_states: torch.Tensor,
                v_proj_states: torch.Tensor,
                g1: torch.Tensor,
                beta: torch.Tensor,
                core_attn_out: torch.Tensor,
            ) -> None:
                self.forward_calls += 1

        layer_metadata = Mock(non_spec_state_indices_tensor=torch.tensor([1]))
        forward_context = Mock(
            attn_metadata={"model.kda": layer_metadata},
            virtual_engine=0,
        )
        with (
            patch.object(
                kda_constructor_patch_module,
                "KimiDeltaAttention",
                _KDALayer,
            ),
            patch(
                "vllm.forward_context.get_forward_context",
                return_value=forward_context,
            ),
        ):
            register_constructor_patch_kda_quant_method(_QuantConfig)
            return _KDALayer(quant_config=_QuantConfig(), prefix="model.kda")

    def _call_forward(self, layer: Any) -> None:
        tensor = torch.empty(0)
        layer._forward(
            q_proj_states=tensor,
            k_proj_states=tensor,
            v_proj_states=tensor,
            g1=tensor,
            beta=tensor,
            core_attn_out=tensor,
        )


class AttachKDAMethodTest(unittest.TestCase):
    @dataclass(frozen=True)
    class _Config:
        quant_recipe: QuantRecipe

    def _attach(self, quant_config: object) -> tuple[torch.nn.Module, Mock]:
        layer = torch.nn.Module()
        wrap_forward = Mock()
        _attach_kda_method(
            layer, quant_config, (self._Config,), "model.kda", wrap_forward
        )
        return layer, wrap_forward

    def test_ignores_other_configs_and_recipes_without_attention(self) -> None:
        for quant_config in (
            object(),
            self._Config(QuantRecipe()),
        ):
            with self.subTest(quant_config=quant_config):
                layer, wrap_forward = self._attach(quant_config)
                self.assertFalse(hasattr(layer, "quant_method"))
                wrap_forward.assert_not_called()

    def test_wraps_forward_only_when_the_state_is_quantized(self) -> None:
        state = PipelineRecipe(
            [FakeQuantRecipe(quant_config=_create_mxfp4_quant_config())]
        )
        for recurrent_state, wrapped in ((PipelineRecipe(), False), (state, True)):
            with self.subTest(wrapped=wrapped):
                layer, wrap_forward = self._attach(
                    self._Config(
                        QuantRecipe(
                            attention_recipe={
                                ".*": QuantAttentionRecipe(
                                    recurrent_state=recurrent_state
                                )
                            }
                        )
                    )
                )
                self.assertIsInstance(layer.quant_method, QuantSuiteKDAMethod)
                if wrapped:
                    wrap_forward.assert_called_once_with(layer)
                else:
                    wrap_forward.assert_not_called()


class ConstructorPatchActiveIndicesTest(unittest.TestCase):
    """The constructor-patch wrapper skips batches with no active state."""

    def _run(self, attn_metadata: object) -> Mock:
        layer: Any = torch.nn.Module()
        empty = torch.empty(0)
        layer.kv_cache = [(empty, empty, empty, torch.zeros(4, 1, 1, 1))]
        layer.prefix = "model.kda"
        layer._forward = Mock()
        layer.quantized_attention = _empty_quantized_attention()
        with (
            patch(
                "vllm.forward_context.get_forward_context",
                return_value=SimpleNamespace(
                    attn_metadata=attn_metadata, virtual_engine=0
                ),
            ),
            patch(
                "quant_suite_vllm.kda.constructor_patch.quantize_kda_state"
            ) as quantize_state,
        ):
            _wrap_forward_with_state_quantization(layer)
            layer._forward(
                q_proj_states=empty,
                k_proj_states=empty,
                v_proj_states=empty,
                g1=empty,
                beta=empty,
                core_attn_out=empty,
            )
        return quantize_state

    def _metadata(self, indices: list[int]) -> dict[str, SimpleNamespace]:
        return {
            "model.kda": SimpleNamespace(
                non_spec_state_indices_tensor=torch.tensor(indices)
            )
        }

    def test_returns_early_without_active_states(self) -> None:
        for name, attn_metadata in (
            ("empty batch", self._metadata([])),
            ("padding-only batch", self._metadata([-1, -1])),
            ("no metadata for this layer", {"model.other": SimpleNamespace()}),
        ):
            with self.subTest(name):
                self._run(attn_metadata).assert_not_called()

    def test_quantizes_only_non_padding_states(self) -> None:
        quantize_state = self._run(self._metadata([2, -1, 2, 1]))

        quantize_state.assert_called_once()
        torch.testing.assert_close(
            quantize_state.call_args.args[2], torch.tensor([1, 2])
        )

    def test_rejects_metadata_without_state_indices(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "does not identify active"):
            self._run({"model.kda": SimpleNamespace()})


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Skip when cuda is not available",
)
class QuantKDAStateGPUTest(unittest.TestCase):
    """GPU tests for quantize_kda_state with fake mxfp4.

    mxfp4 quantization uses Triton kernels which require GPU.
    """

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.device = "cuda"
        self.num_heads = 8
        self.head_dim = 32
        self.num_blocks = 4

    def _create_state(self, num_blocks: int | None = None) -> torch.Tensor:
        return torch.randn(
            num_blocks or self.num_blocks,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )

    def test_mxfp4_modifies_active_blocks(self) -> None:
        """With fake mxfp4, quantize_kda_state should modify only active blocks."""
        state = self._create_state()
        state_before = state.clone()
        active_indices = torch.tensor([0, 1], device=self.device)

        mxfp4_recipe = QUANT_RECIPE_RUNTIME.create_pipeline(
            FakeQuantRecipe(quant_config=_create_mxfp4_quant_config())
        )
        quantize_kda_state(state, mxfp4_recipe, active_indices)

        # Active blocks should be modified
        self.assertFalse(
            torch.equal(state[0], state_before[0]),
            "Active block 0 should be modified with mxfp4 fake quantization",
        )
        self.assertFalse(
            torch.equal(state[1], state_before[1]),
            "Active block 1 should be modified with mxfp4 fake quantization",
        )
        # Inactive blocks should be unchanged
        self.assertTrue(
            torch.equal(state[2], state_before[2]),
            "Inactive block 2 should be unchanged",
        )
        self.assertTrue(
            torch.equal(state[3], state_before[3]),
            "Inactive block 3 should be unchanged",
        )

    def test_mxfp4_preserves_shape_and_dtype(self) -> None:
        """Fake quantization should preserve the tensor shape and dtype."""
        state = self._create_state()
        expected_shape = state.shape
        expected_dtype = state.dtype
        active_indices = torch.tensor([0], device=self.device)

        mxfp4_recipe = QUANT_RECIPE_RUNTIME.create_pipeline(
            FakeQuantRecipe(quant_config=_create_mxfp4_quant_config())
        )
        quantize_kda_state(state, mxfp4_recipe, active_indices)

        self.assertEqual(state.shape, expected_shape)
        self.assertEqual(state.dtype, expected_dtype)

    def test_mxfp4_is_idempotent(self) -> None:
        """Applying fake quantization twice should give the same result as once."""
        state = self._create_state()
        active_indices = torch.tensor([0, 1], device=self.device)

        mxfp4_recipe = QUANT_RECIPE_RUNTIME.create_pipeline(
            FakeQuantRecipe(quant_config=_create_mxfp4_quant_config())
        )

        quantize_kda_state(state, mxfp4_recipe, active_indices)
        state_after_first = state.clone()

        quantize_kda_state(state, mxfp4_recipe, active_indices)

        self.assertTrue(
            torch.equal(state, state_after_first),
            "Fake quantization should be idempotent",
        )

    def test_in_place_modification(self) -> None:
        """quantize_kda_state should modify the tensor in-place (same data_ptr)."""
        state = self._create_state()
        original_data_ptr = state.data_ptr()
        active_indices = torch.tensor([0], device=self.device)

        mxfp4_recipe = QUANT_RECIPE_RUNTIME.create_pipeline(
            FakeQuantRecipe(quant_config=_create_mxfp4_quant_config())
        )
        quantize_kda_state(state, mxfp4_recipe, active_indices)

        self.assertEqual(
            state.data_ptr(),
            original_data_ptr,
            "quantize_kda_state should modify the tensor in-place",
        )

    def test_single_active_block(self) -> None:
        """Test mxfp4 with a single active block."""
        state = self._create_state()
        state_before = state.clone()
        active_indices = torch.tensor([2], device=self.device)

        mxfp4_recipe = QUANT_RECIPE_RUNTIME.create_pipeline(
            FakeQuantRecipe(quant_config=_create_mxfp4_quant_config())
        )
        quantize_kda_state(state, mxfp4_recipe, active_indices)

        # Only block 2 should be modified
        self.assertFalse(torch.equal(state[2], state_before[2]))
        self.assertTrue(torch.equal(state[0], state_before[0]))
        self.assertTrue(torch.equal(state[1], state_before[1]))
        self.assertTrue(torch.equal(state[3], state_before[3]))

    def test_raises_with_empty_active_indices(self) -> None:
        """quantize_kda_state should raise when active_indices is empty."""
        state = self._create_state()
        mxfp4_recipe = QUANT_RECIPE_RUNTIME.create_pipeline(
            FakeQuantRecipe(quant_config=_create_mxfp4_quant_config())
        )
        empty_indices = torch.tensor([], dtype=torch.long, device=self.device)

        with self.assertRaisesRegex(ValueError, "non-empty active_indices"):
            quantize_kda_state(state, mxfp4_recipe, empty_indices)

    def test_ignores_negative_padding_indices(self) -> None:
        state = self._create_state()
        state_before = state.clone()
        raw_indices = torch.tensor([1, -1, -1, -1], device=self.device)
        active_indices = raw_indices[raw_indices >= 0].unique()

        mxfp4_recipe = QUANT_RECIPE_RUNTIME.create_pipeline(
            FakeQuantRecipe(quant_config=_create_mxfp4_quant_config())
        )
        quantize_kda_state(state, mxfp4_recipe, active_indices)

        self.assertFalse(torch.equal(state[1], state_before[1]))
        self.assertTrue(torch.equal(state[0], state_before[0]))

    def test_all_negative_padding_indices_raise_after_filtering(self) -> None:
        state = self._create_state()
        raw_indices = torch.tensor([-1, -1], device=self.device)
        active_indices = raw_indices[raw_indices >= 0].unique()
        mxfp4_recipe = QUANT_RECIPE_RUNTIME.create_pipeline(
            FakeQuantRecipe(quant_config=_create_mxfp4_quant_config())
        )

        with self.assertRaisesRegex(ValueError, "non-empty active_indices"):
            quantize_kda_state(state, mxfp4_recipe, active_indices)


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Skip when cuda is not available",
)
class QuantKDAStateWrapperTest(unittest.TestCase):
    """GPU tests for bound KDA recurrent-state quantization.

    Uses a mock KDA layer to test that the wrapper correctly applies
    state quantization after _forward() completes, without needing
    a full KimiDeltaAttention instantiation.
    """

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.device = "cuda"
        self.num_heads = 8
        self.head_dim = 32
        self.num_blocks = 4

    def _create_mock_attn_metadata(
        self,
        prefix: str,
        active_block_ids: list[int],
    ) -> dict[str, SimpleNamespace]:
        """Create mock attn_metadata dict with non_spec_state_indices_tensor."""
        layer_meta = SimpleNamespace(
            non_spec_state_indices_tensor=torch.tensor(
                active_block_ids, device=self.device
            ),
            num_prefills=0,
            num_decodes=len(active_block_ids),
            num_actual_tokens=len(active_block_ids),
        )
        return {prefix: layer_meta}

    def _create_mock_kda_layer(
        self, recurrent_state: torch.Tensor, prefix: str = "model.layers.12.self_attn"
    ) -> Any:
        """Create a mock KDA layer with a _forward that updates recurrent_state."""
        layer: Any = torch.nn.Module()

        # Simulate kv_cache structure: (conv_q, conv_k, conv_v, recurrent_state)
        dummy_conv = torch.zeros(1, device=self.device)
        layer.kv_cache = [(dummy_conv, dummy_conv, dummy_conv, recurrent_state)]

        # Set prefix for attn_metadata lookup
        layer.prefix = prefix

        # A simple _forward that adds some value to the active block
        def _forward(
            q_proj_states: torch.Tensor,
            k_proj_states: torch.Tensor,
            v_proj_states: torch.Tensor,
            g1: torch.Tensor,
            beta: torch.Tensor,
            core_attn_out: torch.Tensor,
        ) -> None:
            recurrent_state[1].add_(0.1)  # Only modify block 1 (active)

        layer._forward = _forward
        return layer

    def test_wrapper_with_no_quant_leaves_state_unchanged(self) -> None:
        """With NoQuantRecipe, the wrapper should not modify the state beyond _forward."""
        recurrent_state = torch.randn(
            self.num_blocks,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        expected = recurrent_state.clone()
        expected[1].add_(0.1)  # What _forward does

        prefix = "model.layers.12.self_attn"
        layer = self._create_mock_kda_layer(recurrent_state, prefix)
        layer.quantized_attention = QUANT_RECIPE_RUNTIME.prepare_attention(
            QuantAttentionRecipe()
        )

        dummy = torch.zeros(1, device=self.device)
        attn_metadata = self._create_mock_attn_metadata(prefix, [1])

        with patch(
            "vllm.forward_context.get_forward_context",
            return_value=SimpleNamespace(
                attn_metadata=attn_metadata,
                virtual_engine=0,
            ),
        ):
            _wrap_forward_with_state_quantization(layer)
            layer._forward(
                q_proj_states=dummy,
                k_proj_states=dummy,
                v_proj_states=dummy,
                g1=dummy,
                beta=dummy,
                core_attn_out=dummy,
            )

        self.assertTrue(
            torch.equal(recurrent_state, expected),
            "NoQuant wrapper should not modify state beyond _forward()",
        )

    def test_wrapper_with_mxfp4_modifies_state(self) -> None:
        """With fake mxfp4, the wrapper should quantize the state after _forward."""
        recurrent_state = torch.randn(
            self.num_blocks,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        expected_no_quant = recurrent_state.clone()
        expected_no_quant[1].add_(0.1)

        prefix = "model.layers.12.self_attn"
        layer = self._create_mock_kda_layer(recurrent_state, prefix)

        quantized_attention = QUANT_RECIPE_RUNTIME.prepare_attention(
            QuantAttentionRecipe(
                recurrent_state=PipelineRecipe(
                    [FakeQuantRecipe(quant_config=_create_mxfp4_quant_config())]
                )
            )
        )
        layer.quantized_attention = quantized_attention

        dummy = torch.zeros(1, device=self.device)
        attn_metadata = self._create_mock_attn_metadata(prefix, [1])

        with patch(
            "vllm.forward_context.get_forward_context",
            return_value=SimpleNamespace(
                attn_metadata=attn_metadata,
                virtual_engine=0,
            ),
        ):
            _wrap_forward_with_state_quantization(layer)
            layer._forward(
                q_proj_states=dummy,
                k_proj_states=dummy,
                v_proj_states=dummy,
                g1=dummy,
                beta=dummy,
                core_attn_out=dummy,
            )

        # Active block should differ from unquantized expected
        self.assertFalse(
            torch.equal(recurrent_state[1], expected_no_quant[1]),
            "mxfp4 wrapper should quantize active block after _forward()",
        )
        # Inactive blocks should be unchanged
        self.assertTrue(
            torch.equal(recurrent_state[0], expected_no_quant[0]),
            "Inactive block 0 should be unchanged by wrapper",
        )
        self.assertTrue(
            torch.equal(recurrent_state[2], expected_no_quant[2]),
            "Inactive block 2 should be unchanged by wrapper",
        )

    def test_wrapper_skips_during_profile_run(self) -> None:
        """When attn_metadata is None (profile run), wrapper should skip quantization."""
        recurrent_state = torch.randn(
            self.num_blocks,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        expected_no_quant = recurrent_state.clone()
        expected_no_quant[1].add_(0.1)

        prefix = "model.layers.12.self_attn"
        layer = self._create_mock_kda_layer(recurrent_state, prefix)

        quantized_attention = QUANT_RECIPE_RUNTIME.prepare_attention(
            QuantAttentionRecipe(
                recurrent_state=PipelineRecipe(
                    [FakeQuantRecipe(quant_config=_create_mxfp4_quant_config())]
                )
            )
        )
        layer.quantized_attention = quantized_attention

        dummy = torch.zeros(1, device=self.device)
        with patch(
            "vllm.forward_context.get_forward_context",
            return_value=SimpleNamespace(
                attn_metadata=None,
                virtual_engine=0,
            ),
        ):
            _wrap_forward_with_state_quantization(layer)
            layer._forward(
                q_proj_states=dummy,
                k_proj_states=dummy,
                v_proj_states=dummy,
                g1=dummy,
                beta=dummy,
                core_attn_out=dummy,
            )

        self.assertTrue(
            torch.equal(recurrent_state, expected_no_quant),
            "Wrapper should skip quantization when attn_metadata is None",
        )

    def test_wrapper_only_quantizes_active_blocks(self) -> None:
        """Wrapper should only quantize blocks in non_spec_state_indices_tensor."""
        recurrent_state = torch.randn(
            self.num_blocks,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        block_2_before = recurrent_state[2].clone()
        block_3_before = recurrent_state[3].clone()

        prefix = "model.layers.12.self_attn"
        layer = self._create_mock_kda_layer(recurrent_state, prefix)

        quantized_attention = QUANT_RECIPE_RUNTIME.prepare_attention(
            QuantAttentionRecipe(
                recurrent_state=PipelineRecipe(
                    [FakeQuantRecipe(quant_config=_create_mxfp4_quant_config())]
                )
            )
        )
        layer.quantized_attention = quantized_attention

        dummy = torch.zeros(1, device=self.device)
        attn_metadata = self._create_mock_attn_metadata(prefix, [1])

        with patch(
            "vllm.forward_context.get_forward_context",
            return_value=SimpleNamespace(
                attn_metadata=attn_metadata,
                virtual_engine=0,
            ),
        ):
            _wrap_forward_with_state_quantization(layer)
            layer._forward(
                q_proj_states=dummy,
                k_proj_states=dummy,
                v_proj_states=dummy,
                g1=dummy,
                beta=dummy,
                core_attn_out=dummy,
            )

        # Blocks 2, 3 (other layers' data) must be untouched
        self.assertTrue(
            torch.equal(recurrent_state[2], block_2_before),
            "Block 2 (other layer's data) should be untouched by wrapper",
        )
        self.assertTrue(
            torch.equal(recurrent_state[3], block_3_before),
            "Block 3 (other layer's data) should be untouched by wrapper",
        )

    def test_wrapper_selects_virtual_engine_cache(self) -> None:
        inactive_state = torch.randn(
            self.num_blocks,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        recurrent_state = torch.randn_like(inactive_state)
        expected_without_quant = recurrent_state.clone()
        expected_without_quant[1].add_(0.1)
        inactive_before = inactive_state.clone()
        layer = self._create_mock_kda_layer(recurrent_state)
        dummy_conv = torch.zeros(1, device=self.device)
        active_cache = layer.kv_cache[0]
        layer.kv_cache = [
            (dummy_conv, dummy_conv, dummy_conv, inactive_state),
            active_cache,
        ]
        quantized_attention = QUANT_RECIPE_RUNTIME.prepare_attention(
            QuantAttentionRecipe(
                recurrent_state=PipelineRecipe(
                    [FakeQuantRecipe(quant_config=_create_mxfp4_quant_config())]
                )
            )
        )
        layer.quantized_attention = quantized_attention
        dummy = torch.zeros(1, device=self.device)

        with patch(
            "vllm.forward_context.get_forward_context",
            return_value=SimpleNamespace(
                attn_metadata=self._create_mock_attn_metadata(layer.prefix, [1]),
                virtual_engine=1,
            ),
        ):
            _wrap_forward_with_state_quantization(layer)
            layer._forward(
                q_proj_states=dummy,
                k_proj_states=dummy,
                v_proj_states=dummy,
                g1=dummy,
                beta=dummy,
                core_attn_out=dummy,
            )

        self.assertTrue(torch.equal(inactive_state, inactive_before))
        self.assertFalse(torch.equal(recurrent_state[1], expected_without_quant[1]))
