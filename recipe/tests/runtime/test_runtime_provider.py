# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import tempfile
import unittest

import torch
import yaml
from quant_suite_recipe.api.implementation import QuantizedTensor
from quant_suite_recipe.definition import FakeQuantRecipe, QuantLinearRecipe
from quant_suite_recipe.runtime_provider import QUANT_RECIPE_RUNTIME  # @manual


class RuntimeProviderTest(unittest.TestCase):
    def test_exposes_default_runtime(self) -> None:
        recipe = QUANT_RECIPE_RUNTIME.deserialize(
            {"linear_recipe": {"layer": {"weight": "fake_nvfp4"}}}
        )

        self.assertIsInstance(
            recipe.get_linear_recipe("layer").weight[0], FakeQuantRecipe
        )

    def test_selected_linear_uses_default_behavior(self) -> None:
        recipe = QuantLinearRecipe()
        quantized_linear = QUANT_RECIPE_RUNTIME.prepare_linear(recipe)
        x = torch.randn(2, 4)
        weight = torch.randn(3, 4)

        result = quantized_linear(
            x,
            QuantizedTensor(tensor=weight),
            None,
        )

        torch.testing.assert_close(result, torch.matmul(x, weight.t()))

    def test_rejects_unregistered_recipe_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "unregistered_test_type"):
            QUANT_RECIPE_RUNTIME.deserialize(
                {
                    "linear_recipe": {
                        "layer": {"weight": {"type": "unregistered_test_type", "k": 4}}
                    }
                }
            )

    def test_selected_yaml_loader_uses_default_runtime(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as yaml_file:
            yaml.safe_dump(
                {"linear_recipe": {"layer": {"weight": "fake_nvfp4"}}},
                yaml_file,
            )
            yaml_file.flush()
            recipe = QUANT_RECIPE_RUNTIME.load_quant_recipe_from_yaml(yaml_file.name)

        self.assertIsInstance(
            recipe.get_linear_recipe("layer").weight[0], FakeQuantRecipe
        )
