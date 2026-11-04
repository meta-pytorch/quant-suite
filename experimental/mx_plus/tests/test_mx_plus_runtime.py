# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``mx_plus.runtime.RUNTIME``, the published recipe runtime. CPU only."""

import os
import unittest
from importlib import metadata
from unittest.mock import patch

from mx_plus.config import MxPlus
from mx_plus.runtime import RUNTIME
from quant_suite_recipe import _runtime_selection
from quant_suite_recipe.definition import FakeQuantRecipe
from quant_suite_recipe.runtime import QuantRecipeRuntime

_ENTRY_POINT = metadata.EntryPoint(
    name="mx_plus",
    value="mx_plus.runtime:RUNTIME",
    group="quant_suite.recipe_runtimes",
)


class RuntimeTest(unittest.TestCase):
    def test_resolves_the_preset_shorthand(self) -> None:
        recipe = RUNTIME.deserialize(
            {"linear_recipe": {".*": {"weight": "fake_mxfp4_plus2"}}}
        )
        (stage,) = recipe.get_linear_recipe("layer").weight

        assert isinstance(stage, FakeQuantRecipe)
        self.assertEqual(
            stage.quant_config.extensions, (MxPlus(extra_mantissa_bits=2),)
        )

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


if __name__ == "__main__":
    unittest.main()
