# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native quantization enums and named presets.

``PrecisionType`` names the packed native format and ``NativeRoundingMode``
the scale rounding mode of the native MXFP4 and NVFP4 kernels.

``NATIVE_PRESETS`` values are ``quant_suite_recipe`` ``NativeQuantRecipe``
fields, not ``QuantConfig`` fields: ``precision`` is a ``PrecisionType`` value
(``nvfp4``, ``mx4-16``, ``mx4-32``, where the suffix is the block size) and
``scale_round_mode`` is a ``NativeRoundingMode`` value. The recipe package
resolves the ``native_<name>`` shorthand through this table. They cannot be
passed to ``QuantConfig.from_dict``.

MXFP4 preset names carry the block size and the scale rounding:
``mxfp4_ocp`` is the OCP MX layout (1x32, ``floor``), ``mxfp4_16_floor`` is
1x16 with ``floor``, and ``mxfp4_32_oas`` / ``mxfp4_16_oas`` are 1x32 / 1x16
with ``even``. ``FAKE_PRESETS["mxfp4"]`` is 1x16; there is no native preset
named ``mxfp4``.

This module has no dependencies, so it imports without torch or triton.
"""

from __future__ import annotations

from enum import Enum


class NativeRoundingMode(Enum):
    """Scale rounding mode of the native MXFP4 and NVFP4 quantization kernels."""

    NEAREST = "nearest"
    FLOOR = "floor"
    EVEN = "even"
    STOCHASTIC = "stochastic"
    CEIL = "ceil"


class PrecisionType(Enum):
    """Precision format for native quantization

    Supported formats:
    - NVFP4: NVIDIA FP4 format with 1x16 block size
    - MX4_16: MX FP4 format with 1x16 block size
    - MX4_32: MX FP4 format with 1x32 block size (default OCP MX format)

    Note: Rounding mode (floor, even, nearest, etc.) is configured separately
    via the scale_round_mode field in NativeQuantRecipe.
    """

    NVFP4 = "nvfp4"
    MX4_16 = "mx4-16"
    MX4_32 = "mx4-32"


NATIVE_PRESETS: dict[str, dict[str, str]] = {
    "nvfp4": {"precision": "nvfp4", "scale_round_mode": "nearest"},
    "mxfp4_ocp": {"precision": "mx4-32", "scale_round_mode": "floor"},
    "mxfp4_16_floor": {"precision": "mx4-16", "scale_round_mode": "floor"},
    "mxfp4_32_oas": {"precision": "mx4-32", "scale_round_mode": "even"},
    "mxfp4_16_oas": {"precision": "mx4-16", "scale_round_mode": "even"},
}
