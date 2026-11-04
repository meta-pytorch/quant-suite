# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
import torch.nn as nn
from quant_suite_recipe.api.implementation import QuantizedTensor
from quant_suite_recipe.definition import QuantMoERecipe
from quant_suite_recipe.implementation.linear import maybe_quantize_weight
from quant_suite_recipe.implementation.quant_pipeline import Pipeline


class QuantizedMoE(nn.Module):
    """Prepared quantization behavior owned by one MoE module."""

    recipe: QuantMoERecipe
    a1_pipeline: Pipeline
    a2_pipeline: Pipeline
    w13_pipeline: Pipeline
    w2_pipeline: Pipeline

    def __init__(
        self,
        recipe: QuantMoERecipe,
        a1_pipeline: Pipeline,
        a2_pipeline: Pipeline,
        w13_pipeline: Pipeline,
        w2_pipeline: Pipeline,
    ) -> None:
        super().__init__()
        self.recipe = recipe
        self.a1_pipeline = a1_pipeline
        self.a2_pipeline = a2_pipeline
        self.w13_pipeline = w13_pipeline
        self.w2_pipeline = w2_pipeline

    def quantize_a1(self, tensor: torch.Tensor) -> QuantizedTensor:
        return self.a1_pipeline(tensor)

    def quantize_a2(self, tensor: torch.Tensor) -> QuantizedTensor:
        return self.a2_pipeline(tensor)

    def quantize_w13(self, weight: torch.Tensor) -> QuantizedTensor:
        return maybe_quantize_weight(
            weight,
            self.w13_pipeline,
            self.recipe.init_quantized_weight,
        )

    def quantize_w2(self, weight: torch.Tensor) -> QuantizedTensor:
        return maybe_quantize_weight(
            weight,
            self.w2_pipeline,
            self.recipe.init_quantized_weight,
        )


__all__ = ["QuantizedMoE"]
