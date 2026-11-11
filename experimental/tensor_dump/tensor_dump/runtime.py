# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``RUNTIME``: the built-in recipe runtime plus the ``dump`` recipe type and
the ``DumpLinear`` executor.

Published in the ``quant_suite.recipe_runtimes`` entry-point group, so
``quant_suite_recipe.runtime_provider.QUANT_RECIPE_RUNTIME`` resolves to it once
this package is installed. Its executors compute what the built-in ones compute
and write nothing unless ``QUANT_SUITE_TENSOR_DUMP_DIR`` is set.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from quant_suite_recipe.definition import QuantAttentionRecipe, QuantRecipe
from quant_suite_recipe.implementation.attention import QuantizedAttention
from quant_suite_recipe.runtime import QUANT_RECIPE_REGISTRATIONS, QuantRecipeRuntime
from tensor_dump._options import _ATTENTION_ENV, _ATTENTION_TENSORS_ENV
from tensor_dump.executors import DumpLinear
from tensor_dump.recipe import dump_tensor, DumpRecipe


def _reject_attention_dump(recipe: QuantAttentionRecipe, where: str) -> None:
    """Raise if an attention recipe holds a ``dump`` stage."""
    for name in recipe.pipeline_fields():
        if any(isinstance(stage, DumpRecipe) for stage in getattr(recipe, name)):
            raise ValueError(
                f"{where} has a dump stage in {name!r}; attention layers are "
                f"selected with {_ATTENTION_ENV} "
                f"and {_ATTENTION_TENSORS_ENV}, not with recipe stages"
            )


class _DumpRuntime(QuantRecipeRuntime):
    """A ``QuantRecipeRuntime`` that rejects ``dump`` stages in attention
    recipes."""

    def deserialize(self, data: Mapping[str, Any]) -> QuantRecipe:
        recipe = super().deserialize(data)
        for pattern, attention in recipe.attention_recipe.items():
            _reject_attention_dump(attention, f"attention_recipe {pattern!r}")
        return recipe

    def prepare_attention(self, recipe: QuantAttentionRecipe) -> QuantizedAttention:
        _reject_attention_dump(recipe, "the attention recipe")
        return super().prepare_attention(recipe)


RUNTIME: QuantRecipeRuntime = _DumpRuntime(
    (*QUANT_RECIPE_REGISTRATIONS, (DumpRecipe, dump_tensor)),
    quantized_linear_type=DumpLinear,
)

__all__ = ["RUNTIME"]
