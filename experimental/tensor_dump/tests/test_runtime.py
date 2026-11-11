# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``tensor_dump.runtime.RUNTIME``, the published recipe runtime. CPU only."""

import os
import unittest
from importlib import metadata
from unittest.mock import patch

from dump_test_utils import dump_session
from quant_suite_recipe import _runtime_selection
from quant_suite_recipe.definition import (
    FakeQuantRecipe,
    QuantAttentionRecipe,
    QuantLinearRecipe,
)
from quant_suite_recipe.runtime import QuantRecipeRuntime
from tensor_dump.executors import DumpAttention, DumpLinear
from tensor_dump.runtime import RUNTIME

_ENTRY_POINT = metadata.EntryPoint(
    name="tensor_dump",
    value="tensor_dump.runtime:RUNTIME",
    group="quant_suite.recipe_runtimes",
)


class RuntimeTest(unittest.TestCase):
    def test_installed_entry_point_becomes_the_environment_runtime(self) -> None:
        with (
            patch.dict(os.environ),
            patch.object(
                _runtime_selection.metadata,
                "entry_points",
                return_value=(_ENTRY_POINT,),
            ),
        ):
            os.environ.pop("QUANT_SUITE_RECIPE_RUNTIME", None)
            runtime = _runtime_selection.resolve_runtime(QuantRecipeRuntime())

        self.assertIs(runtime, RUNTIME)

    def test_prepare_builds_dump_executors(self) -> None:
        with dump_session(None):
            self.assertIsInstance(
                RUNTIME.prepare_linear(QuantLinearRecipe()), DumpLinear
            )
            self.assertIsInstance(
                RUNTIME.prepare_attention(QuantAttentionRecipe()), DumpAttention
            )

    def test_builtin_preset_still_parses(self) -> None:
        recipe = RUNTIME.deserialize(
            {"linear_recipe": {".*": {"weight": "fake_nvfp4"}}}
        )
        (stage,) = recipe.get_linear_recipe("layer").weight

        self.assertIsInstance(stage, FakeQuantRecipe)


if __name__ == "__main__":
    unittest.main()
