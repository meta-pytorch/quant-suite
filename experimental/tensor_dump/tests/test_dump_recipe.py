# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The ``dump`` recipe type: parsing, validation and the handler. CPU only."""

import unittest
from pathlib import Path

import torch
from dump_test_utils import dump_session
from quant_suite_recipe.definition import (
    NoQuantRecipe,
    PipelineRecipe,
    QuantAttentionRecipe,
    QuantLinearRecipe,
)
from quant_suite_recipe.runtime import QuantRecipeRuntime
from tensor_dump.recipe import dump_tensor, DumpRecipe
from tensor_dump.runtime import RUNTIME

_EXAMPLE: Path = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "llama_3_1_8b_instruct"
    / "dump_bf16.yaml"
)

_EXAMPLE_RECIPE: dict[str, object] = {
    "linear_recipe": {
        r"^model\.layers\.0\.": {
            "activation": {"type": "dump", "tensors": ["input", "weight", "output"]}
        },
        ".*": {"activation": {"type": "dump", "tensors": ["input", "output"]}},
    },
    "attention_recipe": {
        ".*": {"q": {"type": "dump", "tensors": ["q", "k", "v", "output"]}},
    },
}


def _linear(stage: object) -> dict[str, object]:
    return {"linear_recipe": {".*": {"activation": stage}}}


def _activation_stage(runtime: QuantRecipeRuntime, stage: object) -> object:
    (parsed,) = runtime.deserialize(_linear(stage)).get_linear_recipe("x").activation
    return parsed


class DumpRecipeTest(unittest.TestCase):
    def test_forms_parse(self) -> None:
        self.assertEqual(
            _activation_stage(RUNTIME, {"type": "dump"}), DumpRecipe(tensors=None)
        )
        self.assertEqual(
            _activation_stage(
                RUNTIME, {"type": "dump", "tensors": ["output", "input"]}
            ),
            DumpRecipe(tensors=("input", "output")),
        )

    def test_invalid_payloads_raise(self) -> None:
        payloads = {
            "unknown key": {"type": "dump", "tensors": ["input"], "every": 2},
            "unknown tensor": {"type": "dump", "tensors": ["scores"]},
            "duplicate": {"type": "dump", "tensors": ["input", "input"]},
            "empty list": {"type": "dump", "tensors": []},
            "bare string": {"type": "dump", "tensors": "input"},
        }
        for name, payload in payloads.items():
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    RUNTIME.deserialize(_linear(payload))

    def test_to_dict_round_trips(self) -> None:
        recipe = RUNTIME.deserialize(_EXAMPLE_RECIPE)
        self.assertEqual(
            RUNTIME.deserialize(recipe.to_dict()).to_dict(), recipe.to_dict()
        )
        self.assertEqual(
            DumpRecipe(tensors=("q", "k")).to_dict(),
            {"quantize": {"type": "dump", "tensors": ["k", "q"]}},
        )

    @unittest.skipUnless(
        _EXAMPLE.parent.is_dir(), "examples not shipped with this build"
    )
    def test_example_yaml_parses(self) -> None:
        recipe = RUNTIME.load_quant_recipe_from_yaml(str(_EXAMPLE))
        self.assertEqual(
            recipe.to_dict(), RUNTIME.deserialize(_EXAMPLE_RECIPE).to_dict()
        )

    def test_builtin_runtime_rejects_dump(self) -> None:
        with self.assertRaises(ValueError):
            QuantRecipeRuntime().deserialize(_linear({"type": "dump"}))

    def test_prepare_rejects_tensors_of_the_other_kind(self) -> None:
        with dump_session(None):
            with self.assertRaisesRegex(ValueError, "not valid for linear layers"):
                RUNTIME.prepare_linear(
                    QuantLinearRecipe(
                        activation=PipelineRecipe([DumpRecipe(tensors=("q",))])
                    )
                )
            with self.assertRaisesRegex(ValueError, "not valid for attention layers"):
                RUNTIME.prepare_attention(
                    QuantAttentionRecipe(
                        q=PipelineRecipe([DumpRecipe(tensors=("input",))])
                    )
                )

    def test_prepare_rejects_two_dump_stages(self) -> None:
        recipe = QuantLinearRecipe(
            activation=PipelineRecipe([DumpRecipe()]),
            weight=PipelineRecipe([DumpRecipe()]),
        )
        with dump_session(None):
            with self.assertRaisesRegex(ValueError, "at most one is allowed"):
                RUNTIME.prepare_linear(recipe)

    def test_handler_returns_its_input(self) -> None:
        x = torch.randn(3, 4)
        self.assertIs(
            dump_tensor(x, "activation", DumpRecipe(), False, False).tensor, x
        )
        with self.assertRaises(TypeError):
            dump_tensor(x, "activation", NoQuantRecipe(), False, False)


if __name__ == "__main__":
    unittest.main()
