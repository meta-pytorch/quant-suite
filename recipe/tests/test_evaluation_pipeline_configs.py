# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import replace

from evaluation_pipeline_configs import evaluation_cases, RHT_PRESETS
from quant_suite_core.fake_presets import FAKE_PRESETS
from quant_suite_core.rht_generator import HadamardMode
from quant_suite_recipe.definition import FakeQuantRecipe, PipelineRecipe
from quant_suite_recipe.runtime import QuantRecipeRuntime


_BUILTIN_RUNTIME = QuantRecipeRuntime()


class EvaluationPipelineConfigsTest(unittest.TestCase):
    def test_evaluation_cases_are_unique_and_stably_named(self) -> None:
        cases = list(evaluation_cases())
        names = [name for name, _, _ in cases]

        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(names), len(FAKE_PRESETS) * len(RHT_PRESETS))
        self.assertEqual(
            [
                (name, rht_recipe.mode)
                for name, fake_preset, rht_recipe in cases
                if fake_preset == "mxfp4"
            ],
            [
                ("mxfp4", HadamardMode.NONE),
                ("mxfp4_l_had16", HadamardMode.LEFT),
                ("mxfp4_r_had16", HadamardMode.RIGHT),
            ],
        )

    def test_pipeline_construction_prunes_trivial_rht(self) -> None:
        for name, fake_preset, rht_recipe in evaluation_cases():
            recipe = PipelineRecipe(
                [
                    replace(rht_recipe, seed=17),
                    FakeQuantRecipe.from_dict(
                        {"quant_config": fake_preset, "use_triton": False}
                    ),
                ]
            )
            pipeline = _BUILTIN_RUNTIME.create_pipeline(recipe, tensor_name=name)

            self.assertEqual(
                len(pipeline.stages),
                1 if rht_recipe.mode == HadamardMode.NONE else 2,
            )


if __name__ == "__main__":
    unittest.main()
