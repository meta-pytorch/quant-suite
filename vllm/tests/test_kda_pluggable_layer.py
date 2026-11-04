# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import torch
from quant_suite_recipe.api.implementation import QuantizedTensor
from quant_suite_recipe.definition import QuantAttentionRecipe, QuantRecipe
from quant_suite_recipe.implementation.attention import QuantizedAttention
from quant_suite_recipe.implementation.quant_pipeline import Pipeline, PipelineStage
from quant_suite_vllm.kda import register_kda_quant_method
from quant_suite_vllm.kda.method import QuantSuiteKDAMethod
from quant_suite_vllm.kda.pluggable_layer import (
    _wrap_forward_with_state_quantization,
    register_pluggable_kda_quant_method,
)


class _AddOneStage(PipelineStage):
    def forward(self, quantized: QuantizedTensor) -> QuantizedTensor:
        assert quantized.tensor is not None
        return QuantizedTensor(tensor=quantized.tensor + 1)


class _PluggableLayer:
    registry: dict[str, type[Any]] = {}

    @classmethod
    def register_oot(cls, name: str) -> Any:
        def register(layer_type: type[Any]) -> type[Any]:
            cls.registry[name] = layer_type
            return layer_type

        return register


class _GatedDeltaAttention(torch.nn.Module):
    def __init__(self, config: Any, vllm_config: Any, prefix: str = "") -> None:
        super().__init__()
        self.init_arguments = (config, vllm_config, prefix)


class _KimiK3DeltaAttention(torch.nn.Module):
    def __init__(
        self,
        config: Any,
        vllm_config: Any,
        prefix: str = "",
        aux_stream: torch.cuda.Stream | None = None,
        run_gemm_rs_ar: bool = False,
    ) -> None:
        super().__init__()
        self.init_arguments = (
            config,
            vllm_config,
            prefix,
            aux_stream,
            run_gemm_rs_ar,
        )


class _QuantConfig:
    def __init__(self) -> None:
        self.quant_recipe = QuantRecipe(attention_recipe={".*": QuantAttentionRecipe()})


def _quantized_attention(state_pipeline: Pipeline) -> QuantizedAttention:
    return QuantizedAttention(
        QuantAttentionRecipe(),
        Pipeline(),
        Pipeline(),
        Pipeline(),
        Pipeline(),
        state_pipeline,
    )


class KDARegistrationDispatchTest(unittest.TestCase):
    def test_available_kimi_k3_module_uses_pluggable_adapter(self) -> None:
        registration = Mock(return_value=True)

        with (
            patch(
                "quant_suite_vllm.kda.find_spec",
                return_value=Mock(),
            ) as find_spec_mock,
            patch(
                "quant_suite_vllm.kda.import_module",
                return_value=SimpleNamespace(
                    register_pluggable_kda_quant_method=registration,
                ),
            ) as import_module_mock,
        ):
            self.assertTrue(register_kda_quant_method(_QuantConfig))

        self.assertEqual(
            [call.args[0] for call in find_spec_mock.call_args_list],
            [
                "vllm",
                "vllm.models",
                "vllm.models.kimi_k3",
                "vllm.models.kimi_k3.nvidia",
                "vllm.models.kimi_k3.nvidia.kda",
            ],
        )
        import_module_mock.assert_called_once_with(
            "quant_suite_vllm.kda.pluggable_layer"
        )
        registration.assert_called_once_with(_QuantConfig)

    def test_missing_kimi_k3_module_uses_constructor_patch_adapter(self) -> None:
        constructor_patch_registration = Mock(return_value=True)

        with (
            patch(
                "quant_suite_vllm.kda.find_spec",
                side_effect=[Mock(), Mock(), Mock(), Mock(), None],
            ),
            patch(
                "quant_suite_vllm.kda.import_module",
                return_value=SimpleNamespace(
                    register_constructor_patch_kda_quant_method=(
                        constructor_patch_registration
                    ),
                ),
            ),
        ):
            self.assertTrue(register_kda_quant_method(_QuantConfig))

        constructor_patch_registration.assert_called_once_with(_QuantConfig)

    def test_missing_kimi_k3_ancestor_uses_constructor_patch_adapter(self) -> None:
        constructor_patch_registration = Mock(return_value=True)

        with (
            patch(
                "quant_suite_vllm.kda.find_spec",
                side_effect=[Mock(), Mock(), None],
            ),
            patch(
                "quant_suite_vllm.kda.import_module",
                return_value=SimpleNamespace(
                    register_constructor_patch_kda_quant_method=(
                        constructor_patch_registration
                    ),
                ),
            ),
        ):
            self.assertTrue(register_kda_quant_method(_QuantConfig))

        constructor_patch_registration.assert_called_once_with(_QuantConfig)

    def test_transitive_import_failure_is_not_masked(self) -> None:
        transitive_failure = ModuleNotFoundError(name="vllm.models")

        with (
            patch(
                "quant_suite_vllm.kda.find_spec",
                side_effect=[Mock(), transitive_failure],
            ),
            patch("quant_suite_vllm.kda.import_module") as import_module_mock,
            self.assertRaises(ModuleNotFoundError) as raised,
        ):
            register_kda_quant_method(_QuantConfig)

        self.assertIs(raised.exception, transitive_failure)
        import_module_mock.assert_not_called()


class PluggableLayerKDARegistrationTest(unittest.TestCase):
    def setUp(self) -> None:
        _PluggableLayer.registry = {}
        self.modules = {
            "vllm.model_executor.custom_op": SimpleNamespace(
                PluggableLayer=_PluggableLayer,
                op_registry_oot=_PluggableLayer.registry,
            ),
            "vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn": (
                SimpleNamespace(KimiGatedDeltaNetAttention=_GatedDeltaAttention)
            ),
            "vllm.models.kimi_k3.nvidia.kda": SimpleNamespace(
                KimiK3DeltaAttention=_KimiK3DeltaAttention
            ),
        }

    def _import_module(self, name: str) -> Any:
        return self.modules[name]

    def test_registers_idempotent_adapters_for_both_upstream_layers(self) -> None:
        with patch(
            "quant_suite_vllm.kda.pluggable_layer.import_module",
            side_effect=self._import_module,
        ):
            self.assertTrue(register_pluggable_kda_quant_method(_QuantConfig))
            first_registration = dict(_PluggableLayer.registry)
            self.assertTrue(register_pluggable_kda_quant_method(_QuantConfig))

        self.assertEqual(
            set(_PluggableLayer.registry),
            {"KimiGatedDeltaNetAttention", "KimiK3DeltaAttention"},
        )
        self.assertEqual(first_registration, _PluggableLayer.registry)

    def test_adapters_preserve_upstream_constructors_and_attach_quant_method(
        self,
    ) -> None:
        with patch(
            "quant_suite_vllm.kda.pluggable_layer.import_module",
            side_effect=self._import_module,
        ):
            register_pluggable_kda_quant_method(_QuantConfig)

        config = object()
        quant_config = _QuantConfig()
        vllm_config = SimpleNamespace(quant_config=quant_config)
        gated_layer = _PluggableLayer.registry["KimiGatedDeltaNetAttention"](
            config,
            vllm_config,
            "model.gated",
        )
        k3_layer = _PluggableLayer.registry["KimiK3DeltaAttention"](
            config,
            vllm_config,
            "model.k3",
            None,
            True,
        )

        self.assertEqual(
            gated_layer.init_arguments,
            (config, vllm_config, "model.gated"),
        )
        self.assertEqual(
            k3_layer.init_arguments,
            (config, vllm_config, "model.k3", None, True),
        )
        self.assertIsInstance(gated_layer.quant_method, QuantSuiteKDAMethod)
        self.assertIsInstance(k3_layer.quant_method, QuantSuiteKDAMethod)

        unquantized_config = SimpleNamespace(quant_config=object())
        unquantized_layer = _PluggableLayer.registry["KimiGatedDeltaNetAttention"](
            config, unquantized_config
        )
        self.assertFalse(hasattr(unquantized_layer, "quant_method"))


class PluggableLayerKDAStateBindingTest(unittest.TestCase):
    def test_quantizes_active_and_checkpoint_states_but_not_null_block(self) -> None:
        recurrent_state = torch.zeros(6, 1, 1, 1)
        layer: Any = torch.nn.Module()
        layer.prefix = "model.kda"
        layer.kv_cache = (torch.empty(0), recurrent_state)

        def _forward(**kwargs: torch.Tensor) -> None:
            recurrent_state[2].fill_(2)
            recurrent_state[4].fill_(4)

        layer._forward = _forward
        metadata = SimpleNamespace(
            num_spec_decodes=0,
            non_spec_state_indices_tensor=torch.tensor([0, 2, 2]),
            checkpoint=SimpleNamespace(state_indices=torch.tensor([0, 4])),
        )
        forward_context = SimpleNamespace(
            attn_metadata={"model.kda": metadata},
        )
        with patch(
            "quant_suite_vllm.kda.pluggable_layer.import_module",
            return_value=SimpleNamespace(
                get_forward_context=lambda: forward_context,
            ),
        ):
            _wrap_forward_with_state_quantization(layer)
        layer.quantized_attention = _quantized_attention(Pipeline(_AddOneStage()))

        tensor = torch.empty(0)
        layer._forward(
            mixed_qkv=tensor,
            g1=tensor,
            g2=tensor,
            beta=tensor,
            core_attn_out=tensor,
        )

        self.assertEqual(recurrent_state[0].item(), 0)
        self.assertEqual(recurrent_state[2].item(), 3)
        self.assertEqual(recurrent_state[4].item(), 5)

    def test_rejects_speculative_metadata(self) -> None:
        recurrent_state = torch.zeros(2, 1, 1, 1)
        layer: Any = torch.nn.Module()
        layer.prefix = "model.kda"
        layer.kv_cache = (torch.empty(0), recurrent_state)
        layer._forward = lambda **kwargs: None
        forward_context = SimpleNamespace(
            attn_metadata={
                "model.kda": SimpleNamespace(
                    num_spec_decodes=1,
                    non_spec_state_indices_tensor=torch.tensor([1]),
                    checkpoint=None,
                )
            }
        )
        with patch(
            "quant_suite_vllm.kda.pluggable_layer.import_module",
            return_value=SimpleNamespace(
                get_forward_context=lambda: forward_context,
            ),
        ):
            _wrap_forward_with_state_quantization(layer)
        layer.quantized_attention = _quantized_attention(Pipeline(_AddOneStage()))

        tensor = torch.empty(0)
        with self.assertRaisesRegex(NotImplementedError, "speculative decoding"):
            layer._forward(
                mixed_qkv=tensor,
                g1=tensor,
                g2=tensor,
                beta=tensor,
                core_attn_out=tensor,
            )
