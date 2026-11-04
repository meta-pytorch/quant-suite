# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import torch
from quant_suite_core import QuantConfig
from quant_suite_core.native_presets import PrecisionType
from quant_suite_recipe.definition import (
    FakeQuantRecipe,
    NativeQuantRecipe,
    PipelineRecipe,
    QuantLinearRecipe,
    QuantRecipe,
)
from quant_suite_vllm.linear.method import QuantSuiteLinearMethod
from quant_suite_vllm.quant_config import QuantSuiteConfig
from vllm.model_executor.layers.linear import ReplicatedLinear


def _method(linear_recipe: QuantLinearRecipe) -> QuantSuiteLinearMethod:
    return QuantSuiteLinearMethod(
        QuantRecipe(linear_recipe={".*": linear_recipe}),
        "model.layers.0.mlp.down_proj",
    )


def _native(precision: PrecisionType) -> PipelineRecipe:
    return PipelineRecipe([NativeQuantRecipe(precision)])


class QuantSuiteLinearMethodTest(unittest.TestCase):
    def setUp(self) -> None:
        self.layer = ReplicatedLinear(
            input_size=3,
            output_size=2,
            bias=True,
            params_dtype=torch.bfloat16,
            quant_config=QuantSuiteConfig(
                QuantRecipe(linear_recipe={".*": QuantLinearRecipe()})
            ),
            prefix="model.layers.0.mlp.down_proj",
            return_bias=False,
            disable_tp=True,
        )

        with torch.no_grad():
            self.layer.get_parameter("weight").copy_(
                torch.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    dtype=torch.bfloat16,
                )
            )
            self.layer.get_parameter("bias").copy_(
                torch.tensor([0.5, -0.5], dtype=torch.bfloat16)
            )

        quant_method = self.layer.quant_method
        self.assertIsInstance(quant_method, QuantSuiteLinearMethod)
        quant_method.process_weights_after_loading(self.layer)

    def test_applies_linear_recipe(self) -> None:
        result = self.layer(torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.bfloat16))

        self.assertIsInstance(self.layer.quant_method, QuantSuiteLinearMethod)
        self.assertIsInstance(result, torch.Tensor)
        torch.testing.assert_close(
            result,
            torch.tensor([[14.5, 31.5]], dtype=torch.bfloat16),
        )


class QuantSuiteLinearMethodValidationTest(unittest.TestCase):
    def test_rejects_native_bias(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "native bias"):
            _method(QuantLinearRecipe(bias=_native(PrecisionType.NVFP4)))

    def test_rejects_native_precision_mismatch(self) -> None:
        fake = PipelineRecipe([FakeQuantRecipe(QuantConfig.from_dict({}))])
        cases = {
            "native activation, fake weight": QuantLinearRecipe(
                activation=_native(PrecisionType.NVFP4), weight=fake
            ),
            "fake activation, native weight": QuantLinearRecipe(
                activation=fake, weight=_native(PrecisionType.MX4_32)
            ),
            "different precisions": QuantLinearRecipe(
                activation=_native(PrecisionType.MX4_32),
                weight=_native(PrecisionType.NVFP4),
            ),
            "weight not initialized": QuantLinearRecipe(
                activation=_native(PrecisionType.NVFP4),
                weight=_native(PrecisionType.NVFP4),
                init_quantized_weight=False,
            ),
        }
        for name, recipe in cases.items():
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, "same native precision"):
                    _method(recipe)

    def test_accepts_matching_native_precisions(self) -> None:
        for precision in PrecisionType:
            with self.subTest(precision=precision):
                recipe = QuantLinearRecipe(
                    activation=_native(precision), weight=_native(precision)
                )
                self.assertEqual(_method(recipe).linear_recipe, recipe)

    def test_accepts_uninitialized_native_weight_with_unquantized_activation(
        self,
    ) -> None:
        recipe = QuantLinearRecipe(
            weight=_native(PrecisionType.NVFP4), init_quantized_weight=False
        )
        self.assertEqual(_method(recipe).linear_recipe, recipe)
