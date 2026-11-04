# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``MxPlus`` in a recipe runtime: every YAML config form parses. CPU only."""

import unittest
from pathlib import Path

from mx_plus.config import MxPlus
from mx_plus.presets import MX_PLUS_PRESETS
from quant_suite_core.quant_config import DEFAULT_QUANT_CONFIG_BUILDER
from quant_suite_recipe.definition import FakeQuantRecipe
from quant_suite_recipe.runtime import QuantRecipeRuntime

_ATTN: str = "model.layers.0.self_attn.attn"
_ATTN_PATTERN: str = r"^model\.layers\.\d+\.self_attn\.attn$"

_DICT_FORM: dict[str, object] = {
    "type": "fake",
    "quant_config": {
        "elem_dtype": "e2m1",
        "block_size": "1x32",
        "scale_dtype": "e8m0",
        "scale_norm_mode": "pow2max",
        "scale_round_mode": "floor",
        "extensions": {"mx_plus": {"extra_mantissa_bits": 2}},
    },
}

# The three config forms, on a linear layer and an attention tensor.
_RECIPE: dict[str, object] = {
    "linear_recipe": {
        ".*": {
            "weight": "fake_mxfp4_plus2",
            "activation": {"type": "fake", "quant_config": "mxfp4_plus2"},
        }
    },
    "attention_recipe": {
        _ATTN_PATTERN: {
            "k": _DICT_FORM,
        }
    },
}

# examples/<model>/mxfp4_plus2_core.yaml: MX+ on the linear layers only.
_EXAMPLE_RECIPE: dict[str, object] = {
    "linear_recipe": {
        ".*": {
            "weight": "fake_mxfp4_plus2",
            "activation": {"type": "fake", "quant_config": "mxfp4_plus2"},
            "bias": "none",
        }
    },
}

_EXAMPLES_DIR: Path = Path(__file__).resolve().parents[1] / "examples"
_EXAMPLES: tuple[Path, ...] = (
    _EXAMPLES_DIR / "llama_3_1_8b_instruct" / "mxfp4_plus2_core.yaml",
    _EXAMPLES_DIR / "qwen3_8b" / "mxfp4_plus2_core.yaml",
)


def _stage_configs(runtime: QuantRecipeRuntime, data: object) -> list[object]:
    recipe = runtime.deserialize(data)
    linear = recipe.get_linear_recipe("layer")
    attention = recipe.get_attention_recipe(_ATTN)
    stages = [*linear.weight, *linear.activation, *attention.k]
    configs = []
    for stage in stages:
        assert isinstance(stage, FakeQuantRecipe)
        configs.append(stage.quant_config)
    return configs


class TestRecipeForms(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = QuantRecipeRuntime(
            config_builder=DEFAULT_QUANT_CONFIG_BUILDER.extend(
                (MxPlus,), MX_PLUS_PRESETS
            )
        )

    def test_three_forms_parse_to_one_config(self) -> None:
        configs = _stage_configs(self.runtime, _RECIPE)
        self.assertEqual(len(configs), 3)
        for config in configs:
            self.assertEqual(config.extensions, (MxPlus(extra_mantissa_bits=2),))
            self.assertEqual(config, configs[0])

    def test_to_dict_round_trip(self) -> None:
        recipe = self.runtime.deserialize(_RECIPE)
        self.assertEqual(
            self.runtime.deserialize(recipe.to_dict()).to_dict(), recipe.to_dict()
        )

    @unittest.skipUnless(_EXAMPLES_DIR.is_dir(), "examples not shipped with this build")
    def test_example_yaml_parses(self) -> None:
        expected = self.runtime.deserialize(_EXAMPLE_RECIPE).to_dict()
        for example in _EXAMPLES:
            with self.subTest(example=example.relative_to(_EXAMPLES_DIR)):
                self.assertTrue(example.is_file(), f"missing {example}")
                recipe = self.runtime.load_quant_recipe_from_yaml(str(example))
                self.assertEqual(recipe.to_dict(), expected)

    def test_runtime_without_the_extension_rejects_every_form(self) -> None:
        public = QuantRecipeRuntime()
        forms = {
            "shorthand": "fake_mxfp4_plus2",
            "preset": {"type": "fake", "quant_config": "mxfp4_plus2"},
            "dict": _DICT_FORM,
        }
        for name, stage in forms.items():
            with self.subTest(form=name):
                with self.assertRaises(ValueError):
                    public.deserialize(
                        {"attention_recipe": {_ATTN_PATTERN: {"k": stage}}}
                    )


if __name__ == "__main__":
    unittest.main()
