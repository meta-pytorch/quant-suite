# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MX+ for QuantSuite: extra mantissa bits on each block's largest element.

- ``mx_plus.config.MxPlus``: the ``QuantConfigExtension``.
- ``mx_plus.presets.MX_PLUS_PRESETS``: the ``mxfp4_plus2`` preset.
- ``mx_plus.runtime.RUNTIME``: the built-in recipe runtime extended with both,
  published in the ``quant_suite.recipe_runtimes`` entry-point group.
"""

__version__ = "0.1.0"
