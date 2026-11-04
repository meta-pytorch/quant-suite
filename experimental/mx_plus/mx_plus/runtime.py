# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``RUNTIME``: the built-in recipe runtime plus ``MxPlus`` and its presets.

Published in the ``quant_suite.recipe_runtimes`` entry-point group, so
``quant_suite_recipe.runtime_provider.QUANT_RECIPE_RUNTIME`` resolves to it once
this package is installed.
"""

from mx_plus.config import MxPlus
from mx_plus.presets import MX_PLUS_PRESETS
from quant_suite_core.quant_config import DEFAULT_QUANT_CONFIG_BUILDER
from quant_suite_recipe.runtime import QuantRecipeRuntime

RUNTIME: QuantRecipeRuntime = QuantRecipeRuntime(
    config_builder=DEFAULT_QUANT_CONFIG_BUILDER.extend((MxPlus,), MX_PLUS_PRESETS)
)
