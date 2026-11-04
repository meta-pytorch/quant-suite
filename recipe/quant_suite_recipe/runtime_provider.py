# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""The environment's quantization recipe runtime.

Framework integrations import ``QUANT_RECIPE_RUNTIME`` from this module. It is
resolved once, at import, from the built-in runtime ``QuantRecipeRuntime()`` and
any runtime published in the ``quant_suite.recipe_runtimes`` entry-point group.
"""

from quant_suite_recipe._runtime_selection import resolve_runtime
from quant_suite_recipe.runtime import QuantRecipeRuntime


QUANT_RECIPE_RUNTIME: QuantRecipeRuntime = resolve_runtime(QuantRecipeRuntime())


__all__ = ["QUANT_RECIPE_RUNTIME"]
