# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import unittest
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast, ClassVar, Self

import torch
import torch.nn as nn
from quant_suite_core.extension import QuantConfigExtension
from quant_suite_core.fake_presets import FAKE_PRESETS
from quant_suite_core.native_presets import PrecisionType
from quant_suite_core.quant_config import DEFAULT_QUANT_CONFIG_BUILDER
from quant_suite_recipe.api import (
    QuantizedTensor,
    QuantTensorRecipe,
    QuantTensorRecipeRegistration,
)
from quant_suite_recipe.definition import (
    FakeQuantRecipe,
    NativeQuantRecipe,
    NoQuantRecipe,
    PipelineRecipe,
    QuantAttentionRecipe,
    QuantRecipe,
)
from quant_suite_recipe.implementation.attention import QuantizedAttention
from quant_suite_recipe.implementation.quant_pipeline import Pipeline
from quant_suite_recipe.runtime import QUANT_RECIPE_REGISTRATIONS, QuantRecipeRuntime


_BUILTIN_RUNTIME = QuantRecipeRuntime()


@dataclass
class _CustomRecipe(QuantTensorRecipe):
    discriminator: ClassVar[str] = "custom"

    amount: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return {self.stage_key: {"type": "custom", "amount": self.amount}}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(amount=float(data.get("amount", 1.0)))


@dataclass
class _ConflictingRecipe(QuantTensorRecipe):
    discriminator: ClassVar[str] = "custom"

    def to_dict(self) -> dict[str, object]:
        return {self.stage_key: {"type": "custom"}}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls()


@dataclass
class _ShadowFakeRecipe(_CustomRecipe):
    discriminator: ClassVar[str] = "fake"


@dataclass
class _UnnamedRecipe(_CustomRecipe):
    discriminator: ClassVar[str] = ""


class _CustomQuantizedAttention(QuantizedAttention):
    pass


def _quantize_custom(
    tensor: torch.Tensor,
    tensor_name: str,
    recipe: QuantTensorRecipe,
    transpose: bool,
    return_quant_info: bool,
) -> QuantizedTensor:
    assert isinstance(recipe, _CustomRecipe)
    return QuantizedTensor(
        tensor=tensor + recipe.amount,
        metadata={
            "tensor_name": tensor_name,
            "transpose": transpose,
            "return_quant_info": return_quant_info,
        },
    )


@dataclass(frozen=True)
class _ToyBits(QuantConfigExtension):
    name: ClassVar[str] = "toy_bits"
    bits: int = 2


_TOY_BITS_SPEC: dict[str, Any] = {
    **FAKE_PRESETS["mxfp4"],
    "extensions": {"toy_bits": {"bits": 3}},
}
_CUSTOM_REGISTRATION: QuantTensorRecipeRegistration = (
    _CustomRecipe,
    _quantize_custom,
)


class QuantizedTensorTest(unittest.TestCase):
    def test_registered_parameters_round_trip(self) -> None:
        module = nn.Module()
        quantized_tensor = QuantizedTensor(
            tensor=torch.ones(2, 4, dtype=torch.bfloat16),
            scale=torch.ones(2, 1, dtype=torch.float32),
            global_scale=torch.ones((), dtype=torch.float32),
            precision=PrecisionType.NVFP4,
        )

        registered = quantized_tensor.register_parameters(module, "weight")
        round_tripped = QuantizedTensor.from_registered_parameters(module, "weight")
        weight = module._parameters["weight"]
        scale = module._parameters["weight_scale"]
        global_scale = module._parameters["weight_global_scale"]

        self.assertIs(weight, registered)
        self.assertIs(round_tripped.tensor, weight)
        self.assertIs(round_tripped.scale, scale)
        self.assertIs(round_tripped.global_scale, global_scale)
        self.assertIs(module.__dict__["weight_precision"], PrecisionType.NVFP4)
        self.assertIs(round_tripped.precision, PrecisionType.NVFP4)


class QuantRecipeRuntimeTest(unittest.TestCase):
    def test_selects_attention_implementation_type(self) -> None:
        runtime = QuantRecipeRuntime(
            (),
            quantized_attention_type=_CustomQuantizedAttention,
        )

        quantized_attention = runtime.prepare_attention(QuantAttentionRecipe())

        self.assertIsInstance(quantized_attention, _CustomQuantizedAttention)

    def test_registration_updates_deserialization_and_execution_atomically(
        self,
    ) -> None:
        runtime = QuantRecipeRuntime(
            (_CUSTOM_REGISTRATION,),
        )
        recipe = runtime.deserialize(
            {"linear_recipe": {".*": {"weight": {"type": "custom", "amount": 2}}}}
        )
        pipeline = runtime.create_pipeline(
            recipe.linear_recipe[".*"].weight,
            tensor_name="custom_weight",
        )

        result = runtime.apply_pipeline(pipeline, torch.zeros(1))

        torch.testing.assert_close(result.tensor, torch.tensor([2.0]))
        self.assertEqual("custom_weight", result.metadata["tensor_name"])

    def test_quantize_tensor_dispatches_registered_handler(self) -> None:
        runtime = QuantRecipeRuntime(
            (_CUSTOM_REGISTRATION,),
        )

        result = runtime.quantize_tensor(
            torch.zeros(1),
            "custom_weight",
            _CustomRecipe(amount=2),
            transpose=True,
            return_quant_info=True,
        )

        torch.testing.assert_close(result.tensor, torch.tensor([2.0]))
        self.assertEqual(
            {
                "tensor_name": "custom_weight",
                "transpose": True,
                "return_quant_info": True,
            },
            result.metadata,
        )

    def test_pipeline_construction_validates_registered_recipe_type(self) -> None:
        runtime = QuantRecipeRuntime(
            (_CUSTOM_REGISTRATION,),
        )

        with self.assertRaisesRegex(TypeError, "is bound to _CustomRecipe"):
            runtime.create_pipeline(_ConflictingRecipe())

    def test_deserialize_returns_model_recipe_from_complete_payload(self) -> None:
        recipe = _BUILTIN_RUNTIME.deserialize(
            {
                "linear_recipe": {
                    "encoder\\..*": {
                        "activation": "none",
                        "weight": "fake_nvfp4",
                        "bias": "native_nvfp4",
                    }
                }
            }
        )
        linear_recipe = recipe.get_linear_recipe("encoder.layer0")

        self.assertIsInstance(recipe, QuantRecipe)
        self.assertIsInstance(linear_recipe.weight, PipelineRecipe)
        self.assertIsInstance(linear_recipe.activation[0], NoQuantRecipe)
        self.assertIsInstance(linear_recipe.weight[0], FakeQuantRecipe)
        self.assertIsInstance(linear_recipe.bias[0], NativeQuantRecipe)
        self.assertEqual(PrecisionType.NVFP4, linear_recipe.bias[0].precision)

    def test_deserialize_accepts_flattened_complete_payload(self) -> None:
        recipe = _BUILTIN_RUNTIME.deserialize(
            {
                "linear_recipe::decoder\\..*::weight": {
                    "type": "fake",
                    "quant_config": "nvfp4",
                    "use_triton": False,
                },
                "attention_recipe::decoder\\..*\\.self_attn::q": "none",
            }
        )

        linear_recipe = recipe.get_linear_recipe("decoder.layer0")
        attention_recipe = recipe.get_attention_recipe("decoder.layer0.self_attn")

        self.assertIsInstance(recipe, QuantRecipe)
        self.assertIsInstance(linear_recipe.weight[0], FakeQuantRecipe)
        self.assertIsInstance(attention_recipe.q[0], NoQuantRecipe)

    def test_deserialize_accepts_empty_payload_as_inactive_recipe(self) -> None:
        recipe = _BUILTIN_RUNTIME.deserialize({})

        self.assertEqual({}, recipe.linear_recipe)
        self.assertEqual({}, recipe.moe_recipe)
        self.assertEqual({}, recipe.attention_recipe)

    def test_deserialize_rejects_non_model_payloads(self) -> None:
        invalid_payloads: tuple[Any, ...] = (
            "none",
            {"type": "fake", "quant_config": "nvfp4"},
            {"stages": [{"quantize": "fake_nvfp4"}]},
            {"linear_reicpe": {".*": {"weight": "fake_nvfp4"}}},
            {"linear_recipe::.*": {"weight": "fake_nvfp4"}},
            {"linear_recipe::.*::weigth": "fake_nvfp4"},
            {"linear_recipe": {".*": {"weight": "none"}}, "unknown": True},
        )
        for data in invalid_payloads:
            with self.subTest(data=data):
                with self.assertRaisesRegex(ValueError, "complete model-level"):
                    _BUILTIN_RUNTIME.deserialize(data)

    def test_deserialize_rejects_unknown_nested_recipe_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, r"Unknown key\(s\) \['weigth'\]"):
            _BUILTIN_RUNTIME.deserialize(
                {"linear_recipe": {".*": {"weigth": "fake_nvfp4"}}}
            )

    def test_deserialize_rejects_mixed_model_recipe_representations(self) -> None:
        payloads = (
            {
                "linear_recipe": {".*": {"weight": "none"}},
                "attention_recipe::.*::q": "none",
            },
            {
                "attention_recipe::.*::q": "none",
                "linear_recipe": {".*": {"weight": "none"}},
            },
        )
        for data in payloads:
            with self.subTest(data=data):
                with self.assertRaisesRegex(ValueError, "cannot mix"):
                    _BUILTIN_RUNTIME.deserialize(data)

    def test_create_and_apply_pipeline_from_extracted_recipe(self) -> None:
        recipe = _BUILTIN_RUNTIME.deserialize(
            {"linear_recipe": {".*": {"weight": "none"}}}
        )
        pipeline_recipe = recipe.get_linear_recipe("layer").weight
        pipeline = _BUILTIN_RUNTIME.create_pipeline(pipeline_recipe)
        tensor = torch.randn(2, 4)

        result = _BUILTIN_RUNTIME.apply_pipeline(pipeline, tensor)

        self.assertIs(result.tensor, tensor)

    def test_prepare_linear_builds_named_pipelines(self) -> None:
        recipe = _BUILTIN_RUNTIME.deserialize(
            {"linear_recipe": {".*": {"weight": "none"}}}
        )

        quantized_linear = _BUILTIN_RUNTIME.prepare_linear(
            recipe.get_linear_recipe("layer")
        )

        for pipeline in (
            quantized_linear.activation_pipeline,
            quantized_linear.weight_pipeline,
            quantized_linear.bias_pipeline,
        ):
            self.assertIsInstance(pipeline, Pipeline)
            self.assertTrue(pipeline.is_noop())

    def test_runtime_rejects_unregistered_extension_type(self) -> None:
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


class CustomRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = QuantRecipeRuntime(
            (*QUANT_RECIPE_REGISTRATIONS, _CUSTOM_REGISTRATION),
            config_builder=DEFAULT_QUANT_CONFIG_BUILDER.extend(
                (_ToyBits,), {"mxfp4_toy_bits": _TOY_BITS_SPEC}
            ),
            quantized_attention_type=_CustomQuantizedAttention,
        )

    def test_default_arguments_are_the_built_ins(self) -> None:
        payload = {"linear_recipe": {"layer": {"weight": "fake_nvfp4"}}}

        self.assertEqual(
            QuantRecipeRuntime().deserialize(payload).to_dict(),
            QuantRecipeRuntime(
                QUANT_RECIPE_REGISTRATIONS,
                config_builder=DEFAULT_QUANT_CONFIG_BUILDER,
            )
            .deserialize(payload)
            .to_dict(),
        )

    def test_config_builder_resolves_every_config_form(self) -> None:
        recipe = self.runtime.deserialize(
            {
                "attention_recipe": {
                    "attn": {
                        "q": "fake_mxfp4_toy_bits",
                        "k": {"type": "fake", "quant_config": "mxfp4_toy_bits"},
                        "v": {"type": "fake", "quant_config": _TOY_BITS_SPEC},
                    }
                }
            }
        )
        attention = recipe.get_attention_recipe("attn")
        configs = []
        for name in ("q", "k", "v"):
            (stage,) = getattr(attention, name)
            assert isinstance(stage, FakeQuantRecipe)
            configs.append(stage.quant_config)

        self.assertEqual(configs[0].extensions, (_ToyBits(bits=3),))
        self.assertEqual(configs[1], configs[0])
        self.assertEqual(configs[2], configs[0])
        self.assertEqual(
            self.runtime.deserialize(recipe.to_dict()).to_dict(), recipe.to_dict()
        )
        self.assertIsInstance(
            self.runtime.prepare_attention(attention), _CustomQuantizedAttention
        )

    def test_built_in_and_added_recipe_types_both_run(self) -> None:
        recipe = self.runtime.deserialize(
            {
                "linear_recipe": {
                    ".*": {
                        "weight": {"type": "custom", "amount": 2},
                        "activation": "fake_mxfp4_toy_bits",
                    }
                }
            }
        )
        linear = recipe.get_linear_recipe("layer")
        pipeline = self.runtime.create_pipeline(linear.weight)

        torch.testing.assert_close(
            self.runtime.apply_pipeline(pipeline, torch.zeros(1)).tensor,
            torch.tensor([2.0]),
        )
        self.assertIsInstance(linear.activation[0], FakeQuantRecipe)

    def test_built_in_runtime_rejects_the_additions(self) -> None:
        with self.assertRaisesRegex(ValueError, "fake_mxfp4_toy_bits"):
            _BUILTIN_RUNTIME.deserialize(
                {"linear_recipe": {"layer": {"weight": "fake_mxfp4_toy_bits"}}}
            )
        with self.assertRaisesRegex(ValueError, "toy_bits"):
            _BUILTIN_RUNTIME.deserialize(
                {
                    "linear_recipe": {
                        "layer": {
                            "weight": {"type": "fake", "quant_config": _TOY_BITS_SPEC}
                        }
                    }
                }
            )

    def test_rejects_a_registration_that_repeats_a_built_in(self) -> None:
        with self.assertRaisesRegex(ValueError, "discriminator 'fake'"):
            QuantRecipeRuntime(
                (
                    *QUANT_RECIPE_REGISTRATIONS,
                    (_ShadowFakeRecipe, _quantize_custom),
                )
            )

    def test_rejects_an_invalid_recipe_type(self) -> None:
        with self.assertRaisesRegex(TypeError, "must extend QuantTensorRecipe"):
            QuantRecipeRuntime(((cast(Any, QuantRecipe), _quantize_custom),))
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            QuantRecipeRuntime(((_UnnamedRecipe, _quantize_custom),))
