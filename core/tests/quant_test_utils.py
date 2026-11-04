# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
import unittest
from fractions import Fraction

import torch
from quant_suite_core import (
    BlockSize,
    ElemDtype,
    ElemRoundMode,
    GlobalScaleGranularity,
    QuantConfig,
    QuantMode,
    ScaleDtype,
    ScaleNormMode,
    ScaleRoundMode,
)
from quant_suite_core.quant_config import MBSBlockSize

# Reusable decorator: skip a TestCase when CUDA is unavailable.
skip_if_no_cuda = unittest.skipIf(
    not torch.cuda.is_available(),
    "Skip when cuda is not available",
)

# Shared MX/NV element-format parametrize rows:
# (name, use_triton, elem_dtype, scale_dtype, scale_norm_mode, scale_round_mode, global_scale_granularity)
MX_NV_ELEM_CONFIGS = [
    (
        "MX_E2M1_triton",
        True,
        ElemDtype.E2M1,
        ScaleDtype.E8M0,
        ScaleNormMode.POW2MAX,
        ScaleRoundMode.OAS,
        GlobalScaleGranularity.NONE,
    ),
    (
        "MX_E2M1_non_triton",
        False,
        ElemDtype.E2M1,
        ScaleDtype.E8M0,
        ScaleNormMode.POW2MAX,
        ScaleRoundMode.OAS,
        GlobalScaleGranularity.NONE,
    ),
    (
        "NV_E2M1_triton",
        True,
        ElemDtype.E2M1,
        ScaleDtype.FP8,
        ScaleNormMode.AMAX,
        ScaleRoundMode.NONE,
        GlobalScaleGranularity.TENSORWISE,
    ),
    (
        "NV_E2M1_non_triton",
        False,
        ElemDtype.E2M1,
        ScaleDtype.FP8,
        ScaleNormMode.AMAX,
        ScaleRoundMode.NONE,
        GlobalScaleGranularity.TENSORWISE,
    ),
    (
        "MX_E2M3_triton",
        True,
        ElemDtype.E2M3,
        ScaleDtype.E8M0,
        ScaleNormMode.POW2MAX,
        ScaleRoundMode.OAS,
        GlobalScaleGranularity.NONE,
    ),
    (
        "MX_E2M3_non_triton",
        False,
        ElemDtype.E2M3,
        ScaleDtype.E8M0,
        ScaleNormMode.POW2MAX,
        ScaleRoundMode.OAS,
        GlobalScaleGranularity.NONE,
    ),
    (
        "MX_E4M3_triton",
        True,
        ElemDtype.E4M3,
        ScaleDtype.E8M0,
        ScaleNormMode.POW2MAX,
        ScaleRoundMode.OAS,
        GlobalScaleGranularity.NONE,
    ),
    (
        "MX_E4M3_non_triton",
        False,
        ElemDtype.E4M3,
        ScaleDtype.E8M0,
        ScaleNormMode.POW2MAX,
        ScaleRoundMode.OAS,
        GlobalScaleGranularity.NONE,
    ),
]


def make_quant_config(
    quant_mode: QuantMode = QuantMode.SYMM,
    scale_dtype: ScaleDtype = ScaleDtype.E8M0,
    scale_norm_mode: ScaleNormMode = ScaleNormMode.AMAX,
    scale_round_mode: ScaleRoundMode = ScaleRoundMode.OAS,
    global_scale_granularity: GlobalScaleGranularity = GlobalScaleGranularity.NONE,
    elem_dtype: ElemDtype = ElemDtype.E2M1,
    elem_round_mode: ElemRoundMode = ElemRoundMode.NEAREST,
    block_size: BlockSize | None = None,
    global_scale_dtype: ScaleDtype = ScaleDtype.NONE,
    mbs_block_size: MBSBlockSize | None = None,
    **extra: object,
) -> QuantConfig:
    """Canonical QuantConfig factory with SYMM defaults.

    Callers override only the
    fields they care about. ``extra`` passes any other ``QuantConfig`` field
    through, such as ``extensions``.
    """
    if block_size is None:
        block_size = BlockSize(1, 32)
    return QuantConfig(
        quant_mode=quant_mode,
        scale_dtype=scale_dtype,
        scale_norm_mode=scale_norm_mode,
        scale_round_mode=scale_round_mode,
        global_scale_granularity=global_scale_granularity,
        elem_dtype=elem_dtype,
        elem_round_mode=elem_round_mode,
        block_size=block_size,
        global_scale_dtype=global_scale_dtype,
        mbs_block_size=mbs_block_size,
        **extra,
    )


def e8m0_scale_byte(local_unscale: torch.Tensor) -> torch.Tensor:
    """Extract the E8M0 exponent byte from a power-of-two fp32 unscale.

    For MX, local_unscale is an exact power of two (or +Inf, whose fp32
    exponent field is all ones == E8M0 NaN byte 255).
    """
    bits = local_unscale.to(torch.float32).contiguous().view(torch.int32)
    return ((bits & 0x7FFFFFFF) >> 23) & 0xFF


def nearest_fp32(q: Fraction) -> float:
    """The fp32 value nearest the non-negative rational ``q``, ties to even,
    +Inf at and above the overflow threshold, by exact integer arithmetic."""
    if q == 0:
        return 0.0
    # 2^e <= q < 2^(e + 1); fp32's grid there is 2^(e - 23), and 2^-149 below
    # the normal range.
    e = q.numerator.bit_length() - q.denominator.bit_length()
    if Fraction(2) ** e > q:
        e -= 1
    ulp = Fraction(2) ** (max(e, -126) - 23)
    steps, rest = divmod(q / ulp, 1)
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and steps % 2 == 1):
        steps += 1
    value = steps * ulp
    return math.inf if value >= 2**128 else float(value)


def reference_nvfp4_global_scale(amax: float, max_pos: float) -> float:
    """NVFP4's global scale as the backends compute it: ``max_pos / amax``
    (``max_pos`` taken as an fp32 value) rounded to fp32, times 448 in fp64,
    by exact arithmetic; FLT_MAX for a positive amax where that exceeds it."""
    if math.isnan(amax):
        return math.nan
    if amax == 0:
        return math.inf
    if math.isinf(amax):
        return 0.0
    max_pos32 = torch.tensor(max_pos, dtype=torch.float32).item()
    scale = 448.0 * nearest_fp32(Fraction(max_pos32) / Fraction(amax))
    return min(scale, torch.finfo(torch.float32).max)
