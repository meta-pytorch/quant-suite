# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``RUNTIME``: the built-in recipe runtime plus the ``dump`` recipe type and
the dump executors.

Published in the ``quant_suite.recipe_runtimes`` entry-point group, so
``quant_suite_recipe.runtime_provider.QUANT_RECIPE_RUNTIME`` resolves to it once
this package is installed. Its executors compute what the built-in ones compute
and write nothing unless ``QUANT_SUITE_TENSOR_DUMP_DIR`` is set.
"""

from quant_suite_recipe.runtime import QUANT_RECIPE_REGISTRATIONS, QuantRecipeRuntime
from tensor_dump.executors import DumpAttention, DumpLinear
from tensor_dump.recipe import dump_tensor, DumpRecipe

RUNTIME: QuantRecipeRuntime = QuantRecipeRuntime(
    (*QUANT_RECIPE_REGISTRATIONS, (DumpRecipe, dump_tensor)),
    quantized_linear_type=DumpLinear,
    quantized_attention_type=DumpAttention,
)

__all__ = ["RUNTIME"]
