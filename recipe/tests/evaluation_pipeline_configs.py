# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fake-quant x RHT preset cases recorded by golden_vectors.py."""

from collections.abc import Iterator, Mapping
from types import MappingProxyType

from quant_suite_core.fake_presets import FAKE_PRESETS
from quant_suite_core.rht_generator import HadamardMode
from quant_suite_recipe.definition.quant_recipe import RHTRecipe


# Named like FAKE_PRESETS entries so an evaluation case is simply
# ``fake_preset`` or ``f"{fake_preset}_{rht_preset}"``. The NONE entry lets
# callers use one nested loop for rotated and unrotated cases. Runtime pipeline
# construction drops it because RHTRecipe(mode=NONE).is_noop() is true.
RHT_PRESETS: Mapping[str, RHTRecipe] = MappingProxyType(
    {
        "none": RHTRecipe(mode=HadamardMode.NONE),
        "l_had16": RHTRecipe(mode=HadamardMode.LEFT, block_size=16),
        "r_had16": RHTRecipe(mode=HadamardMode.RIGHT, block_size=16),
    }
)


def evaluation_cases() -> Iterator[tuple[str, str, RHTRecipe]]:
    """Cross every fake-quant preset with each RHT evaluation variant."""
    for fake_preset in FAKE_PRESETS:
        for rht_preset, rht_recipe in RHT_PRESETS.items():
            name = (
                fake_preset if rht_preset == "none" else f"{fake_preset}_{rht_preset}"
            )
            yield name, fake_preset, rht_recipe
