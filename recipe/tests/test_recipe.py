# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import re
import tempfile
import unittest
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, ClassVar, Dict

from quant_suite_core.native_presets import NativeRoundingMode, PrecisionType
from quant_suite_recipe.api import QuantTensorRecipe
from quant_suite_recipe.definition import (
    FakeQuantRecipe,
    NativeQuantRecipe,
    NoQuantRecipe,
    PipelineRecipe,
    QuantAttentionRecipe,
    QuantLinearRecipe,
    QuantMoERecipe,
    QuantRecipe,
    RHTRecipe,
)
from quant_suite_recipe.runtime import QUANT_RECIPE_REGISTRATIONS, QuantRecipeRuntime


_BUILTIN_RUNTIME = QuantRecipeRuntime()


@dataclass
class _PrivateRecipe(QuantTensorRecipe):
    discriminator: ClassVar[str] = "private"

    def to_dict(self) -> Dict[str, object]:
        return {self.stage_key: {"type": self.discriminator}}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "_PrivateRecipe":
        return cls()


def _deserialize_tensor_recipe(serialized: Dict[str, object]) -> QuantTensorRecipe:
    recipe = _BUILTIN_RUNTIME.deserialize(
        {"linear_recipe": {".*": {"weight": serialized}}}
    )
    (stage,) = recipe.linear_recipe[".*"].weight
    assert isinstance(stage, QuantTensorRecipe)
    return stage


def _deserialize_linear_recipe(serialized: Mapping[str, Any]) -> QuantLinearRecipe:
    recipe = _BUILTIN_RUNTIME.deserialize({"linear_recipe": {".*": serialized}})
    return recipe.linear_recipe[".*"]


class RecipeTest(unittest.TestCase):
    def test_fake_quant_config_accepts_preset_name_and_dict(self) -> None:
        by_name = FakeQuantRecipe.from_dict({"quant_config": "nvfp4"})
        by_dict = FakeQuantRecipe.from_dict(
            {"quant_config": by_name.quant_config.to_dict()}
        )

        self.assertEqual(by_dict, by_name)

    def test_fake_quant_config_rejects_other_values(self) -> None:
        for raw in (
            None,
            "not_a_preset",
            "symm,e8m0,pow2max,oas,none,e2m1,nearest,1x32,none",
            3,
        ):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(
                    ValueError, "preset name .* or a dict of QuantConfig fields"
                ):
                    FakeQuantRecipe.from_dict({"quant_config": raw})

    def test_quant_tensor_block_sizes_round_trip_without_new_state(self) -> None:
        fake = FakeQuantRecipe.from_dict(
            {
                "quant_config": {
                    "elem_dtype": "e2m1",
                    "elem_round_mode": "nearest",
                    "block_size": "1x16",
                }
            }
        )
        cases: tuple[tuple[str, QuantTensorRecipe, tuple[int, ...] | None], ...] = (
            ("none", NoQuantRecipe(), None),
            ("fake", fake, (1, 16)),
            ("native_nvfp4", NativeQuantRecipe(PrecisionType.NVFP4), (1, 16)),
            ("native_mx4_16", NativeQuantRecipe(PrecisionType.MX4_16), (1, 16)),
            ("native_mx4_32", NativeQuantRecipe(PrecisionType.MX4_32), (1, 32)),
        )

        for name, recipe, expected in cases:
            with self.subTest(name=name):
                serialized = recipe.to_dict()
                restored = _deserialize_tensor_recipe(serialized)

                self.assertEqual(recipe.block_size, expected)
                self.assertEqual(restored.block_size, expected)
                self.assertEqual(restored.to_dict(), serialized)

    def test_fake_to_dict_emits_quant_config(self) -> None:
        recipe = FakeQuantRecipe.from_dict(
            {"quant_config": "nvfp4", "use_triton": False}
        )

        payload = recipe.to_dict()["quantize"]

        self.assertIsInstance(payload, dict)
        assert isinstance(payload, dict)
        self.assertEqual(set(payload), {"type", "quant_config", "use_triton"})

    def test_built_in_registrations_cover_only_supported_types(self) -> None:
        self.assertEqual(
            {
                recipe_type.discriminator
                for recipe_type, _ in QUANT_RECIPE_REGISTRATIONS
            },
            {"none", "fake", "native"},
        )

    def test_runtime_deserialize_accepts_supported_types(self) -> None:
        recipe = _BUILTIN_RUNTIME.deserialize(
            {
                "linear_recipe": {
                    ".*": {
                        "weight": {
                            "type": "native",
                            "precision": "mx4-32",
                            "scale_round_mode": "even",
                        }
                    }
                }
            }
        )
        self.assertIsInstance(recipe.linear_recipe[".*"].weight[0], NativeQuantRecipe)
        recipe = _BUILTIN_RUNTIME.deserialize(
            {
                "linear_recipe": {
                    ".*": {"weight": {"type": "native", "precision": "nvfp4"}}
                }
            }
        )
        self.assertIsInstance(
            recipe.linear_recipe[".*"].weight[0],
            NativeQuantRecipe,
        )

    def test_concrete_from_dict_accepts_data_keyword(self) -> None:
        recipe = NativeQuantRecipe.from_dict(
            data={"precision": "nvfp4", "scale_round_mode": "nearest"}
        )
        self.assertEqual(recipe.precision.value, "nvfp4")

    def test_native_nvfp4_rejects_stochastic_scale_rounding(self) -> None:
        regex = "nvfp4 quantization does not support scale_round_mode='stochastic'"
        with self.assertRaisesRegex(ValueError, regex):
            NativeQuantRecipe(
                precision=PrecisionType.NVFP4,
                scale_round_mode=NativeRoundingMode.STOCHASTIC,
            )
        with self.assertRaisesRegex(ValueError, regex):
            _BUILTIN_RUNTIME.deserialize(
                {
                    "linear_recipe": {
                        ".*": {
                            "weight": {
                                "type": "native",
                                "precision": "nvfp4",
                                "scale_round_mode": "stochastic",
                            }
                        }
                    }
                }
            )
        for mode in NativeRoundingMode:
            if mode is not NativeRoundingMode.STOCHASTIC:
                with self.subTest(mode=mode):
                    recipe = NativeQuantRecipe(
                        precision=PrecisionType.NVFP4, scale_round_mode=mode
                    )
                    self.assertIs(recipe.scale_round_mode, mode)
        recipe = NativeQuantRecipe(
            precision=PrecisionType.MX4_32,
            scale_round_mode=NativeRoundingMode.STOCHASTIC,
        )
        self.assertIs(recipe.scale_round_mode, NativeRoundingMode.STOCHASTIC)

    def test_default_dispatch_rejects_unregistered_discriminator(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid config type"):
            _BUILTIN_RUNTIME.deserialize(
                {
                    "linear_recipe": {
                        ".*": {"weight": {"type": "unregistered_test_type"}}
                    }
                }
            )

    def test_dispatch_rejects_malformed_discriminator(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be a string"):
            _BUILTIN_RUNTIME.deserialize(
                {"linear_recipe": {".*": {"weight": {"type": []}}}}
            )

    def test_dispatch_rejects_mapping_without_type(self) -> None:
        for payload in (
            {"quant_config": "nvfp4"},
            {"quantize": {"precision": "nvfp4"}},
            {"quantize": {}},
        ):
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(
                    ValueError, r"'quantize' stage config .* has no 'type'"
                ):
                    _BUILTIN_RUNTIME.deserialize(
                        {"linear_recipe": {".*": {"weight": payload}}}
                    )

    def test_dispatch_accepts_shorthands_and_explicit_none(self) -> None:
        recipe = _deserialize_linear_recipe(
            {
                "activation": "fake_nvfp4",
                "weight": "native_nvfp4",
                "bias": "none",
            }
        )
        self.assertIsInstance(recipe.activation[0], FakeQuantRecipe)
        self.assertIsInstance(recipe.weight[0], NativeQuantRecipe)
        self.assertIsInstance(recipe.bias[0], NoQuantRecipe)
        recipe = _deserialize_linear_recipe({"weight": {"type": "none"}})
        self.assertIsInstance(recipe.weight[0], NoQuantRecipe)

    def test_default_dispatch_validates_preconstructed_stages(self) -> None:
        private = _PrivateRecipe()
        with self.assertRaisesRegex(ValueError, "Invalid config type"):
            _BUILTIN_RUNTIME.deserialize(
                {"linear_recipe": {".*": {"weight": PipelineRecipe([private])}}}
            )

    def test_default_deserializer_recurses_through_full_recipe(self) -> None:
        recipe = _BUILTIN_RUNTIME.deserialize(
            {
                "linear_recipe": {
                    ".*": {"weight": {"type": "native", "precision": "nvfp4"}}
                }
            }
        )
        self.assertIsInstance(
            recipe.get_linear_recipe("layer").weight[0], NativeQuantRecipe
        )

    def test_attention_only_nested_recipe_keeps_other_sections_empty(self) -> None:
        recipe = _BUILTIN_RUNTIME.deserialize(
            {
                "attention_recipe": {
                    "^model\\.attention$": {
                        "q": {"type": "none"},
                        "k": {"type": "none"},
                        "v": {"type": "none"},
                        "attn_weights": {"type": "none"},
                    }
                }
            }
        )

        self.assertEqual(recipe.linear_recipe, {})
        self.assertEqual(recipe.moe_recipe, {})
        self.assertTrue(
            all(key.startswith("attention_recipe::") for key in recipe.to_dict())
        )
        restored = _BUILTIN_RUNTIME.deserialize(recipe.to_dict())
        self.assertEqual(restored.linear_recipe, {})
        self.assertEqual(restored.moe_recipe, {})

    def test_invalid_layer_pattern_raises(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"Invalid regex '\[' in linear_recipe: "
        ):
            _BUILTIN_RUNTIME.deserialize({"linear_recipe": {"[": {"weight": "none"}}})
        with self.assertRaisesRegex(
            ValueError, r"Invalid regex '\(' in attention_recipe: "
        ):
            QuantRecipe(attention_recipe={"(": QuantAttentionRecipe()})

        recipe = QuantRecipe()
        recipe.moe_recipe = {"[": QuantMoERecipe()}
        with self.assertRaisesRegex(ValueError, r"Invalid regex '\[' in moe_recipe: "):
            recipe.get_moe_recipe("layers.0.moe")

    def test_py_typed_is_packaged(self) -> None:
        self.assertTrue(
            resources.files("quant_suite_recipe").joinpath("py.typed").is_file()
        )

    def test_bare_recipe_instance_remains_invalid_pipeline_input(self) -> None:
        recipe = NativeQuantRecipe.from_dict(data={"precision": "nvfp4"})
        with self.assertRaisesRegex(ValueError, "bare QuantStageRecipe"):
            _deserialize_linear_recipe({"weight": recipe})

    def test_deserialize_and_yaml_use_default_dispatch(self) -> None:
        recipe = _deserialize_linear_recipe(
            {"weight": {"type": "native", "precision": "nvfp4"}}
        )
        self.assertIsInstance(recipe.weight[0], NativeQuantRecipe)

        with tempfile.TemporaryDirectory() as directory:
            yaml_path = Path(directory) / "recipe.yaml"
            yaml_path.write_text(
                "linear_recipe:\n  '.*':\n    weight:\n      type: native\n"
                "      precision: nvfp4\n"
            )
            loaded = _BUILTIN_RUNTIME.load_quant_recipe_from_yaml(str(yaml_path))
        self.assertIsInstance(
            loaded.get_linear_recipe("layer").weight[0],
            NativeQuantRecipe,
        )

        with self.assertRaisesRegex(ValueError, "Invalid config type"):
            _deserialize_linear_recipe({"weight": {"type": "unregistered_test_type"}})

        with tempfile.TemporaryDirectory() as directory:
            yaml_path = Path(directory) / "extension_recipe.yaml"
            yaml_path.write_text(
                "linear_recipe:\n  '.*':\n    weight:\n      type: unregistered_test_type\n"
            )
            with self.assertRaisesRegex(ValueError, "Invalid config type"):
                _BUILTIN_RUNTIME.load_quant_recipe_from_yaml(str(yaml_path))


class UnknownKeyTest(unittest.TestCase):
    def test_fake_rejects_misspelled_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown key.*'use_trition'"):
            FakeQuantRecipe.from_dict({"quant_config": "nvfp4", "use_trition": True})
        with self.assertRaisesRegex(ValueError, "Unknown key.*'use_trition'"):
            _deserialize_tensor_recipe(
                {"type": "fake", "quant_config": "nvfp4", "use_trition": True}
            )

    def test_native_rejects_misspelled_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown key.*'scale_rounding'"):
            NativeQuantRecipe.from_dict(
                {"type": "native", "precision": "nvfp4", "scale_rounding": "floor"}
            )

    def test_rht_rejects_generator_key(self) -> None:
        message = (
            "Unknown key(s) ['generator'] in 'rht' stage config; "
            "allowed keys: ['block_size', 'seed', 'type']"
        )
        with self.assertRaisesRegex(ValueError, re.escape(message)):
            RHTRecipe.from_dict({"type": "right", "block_size": 16, "generator": 3})
        with self.assertRaisesRegex(ValueError, re.escape(message)):
            _deserialize_linear_recipe(
                {
                    "weight": [
                        {"rht": {"type": "right", "block_size": 16, "generator": 3}}
                    ]
                }
            )

    def test_rht_requires_type(self) -> None:
        with self.assertRaisesRegex(ValueError, r"has no 'type'.*'right_inv'"):
            RHTRecipe.from_dict({"block_size": 16})

    def test_none_rejects_extra_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown key.*'quant_config'"):
            NoQuantRecipe.from_dict({"type": "none", "quant_config": "nvfp4"})

    def test_module_stage_and_pipeline_reject_misspelled_keys(self) -> None:
        cases: tuple[tuple[dict[str, Any], str], ...] = (
            ({"moe_recipe": {".*": {"w31": "fake_nvfp4"}}}, "w31"),
            (
                {
                    "linear_recipe": {
                        ".*": {"weight": [{"quantize": "none", "note": 1}]}
                    }
                },
                "note",
            ),
            (
                {"linear_recipe": {".*": {"weight": {"quantize": "none", "note": 1}}}},
                "note",
            ),
            (
                {"linear_recipe": {".*": {"weight": {"stages": [], "stage": []}}}},
                "stage",
            ),
        )
        for data, key in cases:
            with self.subTest(data=data):
                with self.assertRaisesRegex(ValueError, f"Unknown key.*'{key}'"):
                    _BUILTIN_RUNTIME.deserialize(data)

    def test_yaml_rejects_unknown_top_level_key(self) -> None:
        cases = (
            ("linear_recipe:\n  '.*':\n    weight: none\nnote: x\n", "note"),
            ("lienar_recipe:\n  '.*':\n    weight: none\n", "lienar_recipe"),
        )
        for body, key in cases:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                yaml_path = Path(directory) / "recipe.yaml"
                yaml_path.write_text(body)
                with self.assertRaisesRegex(ValueError, f"Unknown key.*'{key}'"):
                    _BUILTIN_RUNTIME.load_quant_recipe_from_yaml(str(yaml_path))

    def test_non_mapping_section_or_entry_rejected(self) -> None:
        cases: tuple[tuple[dict[str, Any], str], ...] = (
            (
                {"linear_recipe": [{"weight": "none"}]},
                "'linear_recipe' to be a mapping",
            ),
            ({"moe_recipe": "fake_nvfp4"}, "'moe_recipe' to be a mapping"),
            ({"attention_recipe": {".*": ["q"]}}, "'attention_recipe' entry to map"),
        )
        for data, message in cases:
            with self.subTest(data=data):
                with self.assertRaisesRegex(ValueError, message):
                    _BUILTIN_RUNTIME.deserialize(data)
        with tempfile.TemporaryDirectory() as directory:
            yaml_path = Path(directory) / "recipe.yaml"
            yaml_path.write_text("linear_recipe:\n  - weight: none\n")
            with self.assertRaisesRegex(ValueError, "'linear_recipe' to be a mapping"):
                _BUILTIN_RUNTIME.load_quant_recipe_from_yaml(str(yaml_path))


class FieldTypeTest(unittest.TestCase):
    def test_use_triton_must_be_bool(self) -> None:
        for value in ("false", 0, 1, "true"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "use_triton must be a bool"):
                    FakeQuantRecipe(
                        quant_config=_BUILTIN_RUNTIME.config_builder.build_preset(
                            "mxfp4"
                        ),
                        use_triton=value,  # pyrefly: ignore[bad-argument-type]
                    )
                with self.assertRaisesRegex(TypeError, "use_triton must be a bool"):
                    _deserialize_tensor_recipe(
                        {"type": "fake", "quant_config": "mxfp4", "use_triton": value}
                    )
        for value in (None, True, False):
            with self.subTest(value=value):
                stage = _deserialize_tensor_recipe(
                    {"type": "fake", "quant_config": "mxfp4", "use_triton": value}
                )
                assert isinstance(stage, FakeQuantRecipe)
                self.assertIs(stage.use_triton, value)

    def test_rht_block_size_must_be_int(self) -> None:
        for value in (3.7, 128.0, True, "128", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "block_size must be an int"):
                    RHTRecipe.from_dict({"type": "right", "block_size": value})
        self.assertEqual(
            RHTRecipe.from_dict({"type": "right", "block_size": 128}).block_size, 128
        )
        self.assertEqual(RHTRecipe.from_dict({"type": "none"}).block_size, 0)

    def test_rht_seed_must_be_int_or_none(self) -> None:
        for value in ("1", 1.5, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "seed must be an int"):
                    RHTRecipe.from_dict(
                        {"type": "right", "block_size": 16, "seed": value}
                    )
        for value in (None, 0, 2026):
            with self.subTest(value=value):
                recipe = RHTRecipe.from_dict(
                    {"type": "right", "block_size": 16, "seed": value}
                )
                self.assertEqual(recipe.seed, value)

    def test_init_quantized_weight_must_be_bool(self) -> None:
        for value in ("false", 0, None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    TypeError, "init_quantized_weight must be a bool"
                ):
                    _deserialize_linear_recipe({"init_quantized_weight": value})
                with self.assertRaisesRegex(
                    TypeError, "QuantMoERecipe.init_quantized_weight must be a bool"
                ):
                    _BUILTIN_RUNTIME.deserialize(
                        {"moe_recipe": {".*": {"init_quantized_weight": value}}}
                    )
        self.assertFalse(
            _deserialize_linear_recipe(
                {"init_quantized_weight": False}
            ).init_quantized_weight
        )
