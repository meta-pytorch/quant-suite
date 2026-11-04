# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# `_safe_lshift`, `_safe_rshift` and the default (non-`ranking_only`) path of
# `_fp32_to_fpx_helper` are Triton ports of code adapted from mx/elemwise_ops.py
# in Microsoft microxcaling (https://github.com/microsoft/microxcaling), licensed
# under the MIT License.
# Copyright (c) Microsoft Corporation.
# See THIRD_PARTY_NOTICES.md in the root directory of this package.

"""Triton kernels, device helpers and host-side launch helpers behind the Triton
QDQ and native backends.

Private to the package. Format names reach the kernels as ``tl.constexpr``
strings, which must match the ``QuantConfig`` enum values.
"""

from typing import Optional, Tuple

import torch
import triton  # @manual
import triton.language as tl  # @manual
from quant_suite_core.quant_config import (
    ELEM_DTYPE_BITS_MAP,
    ElemRoundMode,
    QuantConfig,
)

# `_n_ones_host` is the host-side `_n_ones`; this module's own is a `@jit` function.
from quant_suite_core.quant_primitives import (
    _n_ones as _n_ones_host,
    _TRITON_MAX_BLOCK_NUMEL_PER_QUANT_BLOCK,
)
from triton import jit  # @manual
from triton.language.extra import libdevice  # @manual

# *****************************************************************************
# * Constants and Maps                                                        *
# *****************************************************************************
SBITS = tl.constexpr(1)
EBITS_F32 = tl.constexpr(8)
MBITS_F32 = tl.constexpr(23)
E8M0_EXPONENT_BIAS = tl.constexpr(127)
F32_EXP_BIAS = tl.constexpr(127)
F32_MIN_NORMAL = tl.constexpr(2 ** (-F32_EXP_BIAS + 1))
# Largest finite fp32 value (FLT_MAX = (2 - 2**-23) * 2**127). Used to saturate
# ±Inf to ±FLT_MAX before MX scale selection.
F32_MAX = tl.constexpr((2 - 2**-23) * 2**127)
# MBS: a quant block whose finite peak is at least this gets the overflow
# headroom (see `_mbs_headroom` in qdq_pytorch.py).
MBS_HEADROOM_MIN_PEAK = tl.constexpr(2.0**126)
F16_EXP_BIAS = tl.constexpr(15)
F16_MIN_NORMAL = tl.constexpr(2 ** (-F16_EXP_BIAS + 1))
# The smallest positive fp32 and fp16 values, the least FP32 and FP16 block
# scale of a nonzero block (quant_config.py).
F32_MIN_SUBNORMAL = tl.constexpr(2.0**-149)
F16_MIN_SUBNORMAL = tl.constexpr(2.0**-24)


# *****************************************************************************
# * Helper Functions                                                          *
# *****************************************************************************
@jit
def _safe_lshift(x, bits, exp=0):
    return x / tl.exp2(exp) * (2**bits)


@jit
def _safe_rshift(x, bits, exp=0):
    return x / (2**bits) * tl.exp2(exp)


@jit
def _n_ones(n: int) -> int:
    return (1 << n) - 1


@jit
def _round_int(x):
    # round-half-to-even (matches torch.round exactly, incl. sign of zero)
    return libdevice.rint(x)


# *****************************************************************************
# * Triton Kernels                                                            *
# *****************************************************************************


@jit
def _fp32_to_e8m1_kernel(
    exponent_A,
    mantissa_A,
):
    mask_0 = mantissa_A <= (32 << 16)  # <= 100000, mantissa_A is 0
    mask_1 = (~mask_0) & (mantissa_A <= (95 << 16))  # <= 1011111, mantissa_A is 1
    mask_2 = (~mask_0) & (~mask_1)  # exp need round up, mantissa_A is 0
    mantissa_A = tl.where(mask_0 | mask_2, 0, mantissa_A)
    mantissa_A = tl.where(mask_1, (1 << 22), mantissa_A)
    exponent_A = tl.where(mask_2, exponent_A + 1, exponent_A)
    mask_overflow = (exponent_A == 255) & mask_2
    exponent_A = tl.where(mask_overflow, exponent_A - 1, exponent_A)
    mantissa_A = tl.where(mask_overflow, (1 << 22), mantissa_A)

    return exponent_A, mantissa_A


@jit
def _fp32_to_e8m2_kernel(
    exponent_A,
    mantissa_A,
):
    mask_0 = mantissa_A <= (16 << 16)  # <= 010000, mantissa_A is 0
    mask_1 = (~mask_0) & (mantissa_A <= (47 << 16))  # <= 101111, mantissa_A is 1
    mask_2 = (
        (~mask_0) & (~mask_1) & (mantissa_A <= (80 << 16))
    )  # <= 1010000, mantissa_A is 2
    mask_3 = (
        (~mask_0) & (~mask_1) & (~mask_2) & (mantissa_A <= (111 << 16))
    )  # <= 1101111, mantissa_A is 3
    mask_4 = (
        (~mask_0) & (~mask_1) & (~mask_2) & (~mask_3)
    )  # exp need round up, mantissa_A is 0
    mantissa_A = tl.where(mask_0 | mask_4, 0, mantissa_A)
    mantissa_A = tl.where(mask_1, (1 << 21), mantissa_A)
    mantissa_A = tl.where(mask_2, (2 << 21), mantissa_A)
    mantissa_A = tl.where(mask_3, (3 << 21), mantissa_A)
    exponent_A = tl.where(mask_4, exponent_A + 1, exponent_A)
    mask_overflow = (exponent_A == 255) & mask_4
    exponent_A = tl.where(mask_overflow, exponent_A - 1, exponent_A)
    mantissa_A = tl.where(mask_overflow, (3 << 21), mantissa_A)

    return exponent_A, mantissa_A


@jit
def _fp32_to_e8m3_kernel(
    exponent_A,
    mantissa_A,
):
    mask_0 = mantissa_A <= (8 << 16)  # <= 001000, mantissa_A is 0
    mask_1 = (~mask_0) & (mantissa_A <= (23 << 16))  # <= 0010111, mantissa_A is 1
    mask_2 = (
        (~mask_0) & (~mask_1) & (mantissa_A <= (40 << 16))
    )  # <= 0101000, mantissa_A is 2
    mask_3 = (
        (~mask_0) & (~mask_1) & (~mask_2) & (mantissa_A <= (55 << 16))
    )  # <= 0110111, mantissa_A is 3
    mask_4 = ((~mask_0) & (~mask_1) & (~mask_2) & (~mask_3)) & (
        mantissa_A <= (72 << 16)
    )  # <= 0111000, mantissa_A is 4
    mask_5 = ((~mask_0) & (~mask_1) & (~mask_2) & (~mask_3) & (~mask_4)) & (
        mantissa_A <= (87 << 16)
    )  # <= 1010111, mantissa_A is 5
    mask_6 = ((~mask_0) & (~mask_1) & (~mask_2) & (~mask_3) & (~mask_4) & (~mask_5)) & (
        mantissa_A <= (104 << 16)
    )  # <= 1101000, mantissa_A is 6
    mask_7 = (
        (~mask_0)
        & (~mask_1)
        & (~mask_2)
        & (~mask_3)
        & (~mask_4)
        & (~mask_5)
        & (~mask_6)
    ) & (mantissa_A <= (119 << 16))  # <= 1110111, mantissa_A is 7
    mask_8 = (
        (~mask_0)
        & (~mask_1)
        & (~mask_2)
        & (~mask_3)
        & (~mask_4)
        & (~mask_5)
        & (~mask_6)
        & (~mask_7)
    )  # exp need round up, mantissa_A is 0
    mantissa_A = tl.where(mask_0 | mask_8, 0, mantissa_A)
    mantissa_A = tl.where(mask_1, (1 << 20), mantissa_A)
    mantissa_A = tl.where(mask_2, (2 << 20), mantissa_A)
    mantissa_A = tl.where(mask_3, (3 << 20), mantissa_A)
    mantissa_A = tl.where(mask_4, (4 << 20), mantissa_A)
    mantissa_A = tl.where(mask_5, (5 << 20), mantissa_A)
    mantissa_A = tl.where(mask_6, (6 << 20), mantissa_A)
    mantissa_A = tl.where(mask_7, (7 << 20), mantissa_A)
    exponent_A = tl.where(mask_8, exponent_A + 1, exponent_A)
    mask_overflow = (exponent_A == 255) & mask_8
    exponent_A = tl.where(mask_overflow, exponent_A - 1, exponent_A)
    mantissa_A = tl.where(mask_overflow, (7 << 20), mantissa_A)

    return exponent_A, mantissa_A


@jit
def _fp32_floor(
    x,
    ebits: tl.constexpr,
    mbits: tl.constexpr,
    max_normal: tl.constexpr,
    min_normal: tl.constexpr,
):
    x = tl.cast(x, tl.uint32, bitcast=True)
    SIGN_MASK: tl.uint32 = 0x80000000
    sign = x & SIGN_MASK
    x = x ^ sign
    x = tl.cast(x, tl.float32, bitcast=True)

    saturate_mask = x >= max_normal
    denormal_mask = (~(saturate_mask)) & (x < min_normal)

    # generate a shifting tensor for subnormal values
    shifting_tensor = tl.where(denormal_mask, min_normal, 0.0)

    # shift the subnormal values to normal range
    x += shifting_tensor

    # view x as int32
    x = tl.cast(x, tl.uint32, bitcast=True)

    # zero out the last (23 - mbits) bits, only keep the leading (9 + mbits) bits
    # e.g. 0xFFF00000 for mbits=3
    ZERO_OUT_MASK = tl.full(
        (),
        (((1 << (9 + mbits)) - 1) << (32 - (9 + mbits))),
        tl.uint32,
    )
    x = x & ZERO_OUT_MASK

    x = tl.cast(x, tl.float32, bitcast=True)

    # shift the subnormal values back
    x -= shifting_tensor

    # add sign back
    x = tl.cast(x, tl.uint32, bitcast=True)
    x = x | sign
    x = tl.cast(x, tl.float32, bitcast=True)

    return x


@jit
def _stochastic_rounding_helper(
    x,
    rand_values,
    ebits: tl.constexpr,
    mbits: tl.constexpr,
    max_normal: tl.constexpr,
    min_normal: tl.constexpr,
) -> tl.tensor:
    x = tl.cast(x, tl.uint32, bitcast=True)
    SIGN_MASK: tl.uint32 = 0x80000000
    sign = x & SIGN_MASK
    x = x ^ sign
    x = tl.cast(x, tl.float32, bitcast=True)

    saturate_mask = x >= max_normal
    denormal_mask = (~(saturate_mask)) & (x < min_normal)

    # generate a shifting tensor for subnormal values
    shifting_tensor = tl.where(denormal_mask, min_normal, 0.0)

    # shift the subnormal values to normal range
    x += shifting_tensor

    # view x as int32
    x = tl.cast(x, tl.int32, bitcast=True)

    # zero out the leading (1 + 8 + mbits) bits
    # use 23 instead of MBITS_F32 to avoid syntax error of triton...
    rand_mask = (1 << (23 - mbits)) - 1
    # add the random bits
    x += rand_values & rand_mask

    # take the leading (1 + 8 + mbits) bits
    x = x & (~rand_mask)
    x = tl.cast(x, tl.float32, bitcast=True)

    # shift the subnormal values back
    x -= shifting_tensor

    # add sign back
    x = tl.cast(x, tl.uint32, bitcast=True)
    x = x | sign
    x = tl.cast(x, tl.float32, bitcast=True)

    return x


@jit
def _fp32_to_fpx_helper(
    A,
    mbits: tl.constexpr,
    min_exp: tl.constexpr,
    max_pos: tl.constexpr,
    # Cheaper equivalents for callers that only compare results against each other.
    # Only safe for |A| < 2^127: the 2^exp construction below builds the exponent
    # field directly, and a larger exponent makes `127 - exp` fall out of the normal
    # range and produce garbage rather than saturating. The MBS search satisfies this
    # because it passes block-scaled data, bounded by roughly `max_pow2`; a caller
    # passing raw tensor values does not automatically.
    ranking_only: tl.constexpr = False,
) -> tl.tensor:
    # private_exp = floor(log2(|A|)), which for a normal A is its float32
    # exponent field: read it instead of calling a transcendental. A subnormal
    # or zero A reads as -127 and is clamped to min_exp below (a zero scales to
    # the same signed zero with any exponent). Where the log2 of qdq() rounds up
    # to the next exponent, just below a power of two, A rounds to that power of
    # two with either exponent.
    exp_bits = (tl.cast(A, tl.int32, bitcast=True) >> 23) & 0xFF
    private_exp = (exp_bits - 127).to(tl.float32)

    # Clamp private_exp to min_exp
    private_exp = tl.maximum(private_exp, min_exp)

    # Scale up so appropriate number of bits are in the integer portion of the number
    if ranking_only:
        # private_exp is integer-valued, so 2^private_exp can be built from the
        # exponent field instead of calling exp2 twice per element per candidate.
        # Both shifts then become multiplies.
        exp_i = private_exp.to(tl.int32)
        pow2_pos = ((exp_i + 127) << 23).to(tl.float32, bitcast=True)
        pow2_neg = ((127 - exp_i) << 23).to(tl.float32, bitcast=True)
        out = A * pow2_neg * (2**mbits)
    else:
        out = _safe_lshift(A, mbits, private_exp)

    # Round mantissa bits to round to nearest even
    absA = tl.abs(out)
    # Whether absA - 0.5 is an even integer, (absA - 0.5) % 2 == 0, tested
    # without the division of a floating remainder: the halving, the floor and
    # the doubling are exact.
    tie = absA - 0.5
    maskA = (tl.floor(tie * 0.5) * 2.0 == tie).to(tl.float32)
    # Sign from the bit pattern (not out < 0) so -0.0 stays -0.0: out < 0 is
    # False for -0.0 and would flush it to +0.0, but -0 is a distinct element
    # code. Mirrors the copysign in the
    # pytorch fp32_to_fpx.
    out_sign = tl.where(
        (tl.cast(out, tl.uint32, bitcast=True) & 0x80000000) != 0, -1.0, 1.0
    )
    # Always mirror the pure fp32_to_fpx fp64 upcast for the round-to-nearest
    # step. In fp32, an absA ~1 ULP below a tie (e.g. 0.49999997) has
    # (absA + 0.5) round up to 1.0, so floor pushes a below-midpoint element up a
    # whole grid step. The pure qdq reference does this in fp64, so we do the
    # round in fp64 unconditionally to match it bit-for-bit.
    if ranking_only:
        # fp64 is heavily throttled on B200 and this runs once per element per
        # candidate. A 1-ULP difference in an error estimate essentially never
        # changes the argmin, and the winning factor is re-quantized through the
        # exact path afterwards, so emitted values are unaffected.
        rounded = tl.floor(absA + 0.5)
    else:
        rounded = tl.floor(absA.to(tl.float64) + 0.5).to(tl.float32)
    out = out_sign * (rounded - maskA)

    # Undo scaling
    if ranking_only:
        out = out * (1.0 / (2**mbits)) * pow2_pos
    else:
        out = _safe_rshift(out, mbits, private_exp)

    # Handle saturation
    out = tl.minimum(out, max_pos)
    out = tl.maximum(out, -max_pos)

    return out


@jit
def _fp32_to_e8mx_helper(
    A,
    fp8mx_kernel_func,
) -> tl.tensor:
    int32_A = tl.cast(A, tl.int32, bitcast=True)
    sign_A = (int32_A >> 31) & 0x1
    exponent_A = (int32_A >> 23) & 0xFF
    mantissa_A = int32_A & 0x7FFFFF

    exponent_A, mantissa_A = fp8mx_kernel_func(exponent_A, mantissa_A)
    # Combine sign, exponent, and mantissa
    bits = (sign_A << 31) | (exponent_A << 23) | mantissa_A
    out = tl.cast(bits, tl.float32, bitcast=True)

    return out


@jit
def _mul_rn(a, b):
    """``a * b`` in fp32, rounded once and never fused into an FMA, keeping a
    subnormal product (libdevice's ``mul_rn`` flushes it to zero)."""
    return tl.inline_asm_elementwise(
        asm="mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@jit
def _cal_fp32_scale_unscale(
    unscale,
    pre,
):
    unscale = unscale.to(tl.float32)
    # Correctly rounded, as qdq()'s reciprocal; `1.0 / unscale` is not. `pre`
    # (1 or 2^64) keeps the reciprocal of a tiny scale finite.
    scale = tl.math.div_rn(1.0, unscale * pre)

    return scale, unscale


@jit
def _cal_fp16_scale_unscale(
    unscale,
):
    unscale = unscale.to(tl.float16)
    # The fp32 reciprocal of the fp16 scale, correctly rounded as in qdq().
    scale = tl.math.div_rn(1.0, unscale.to(tl.float32))

    return scale, unscale


@jit
def _cal_fp8_scale_unscale_for_nvfp4(
    global_amax,  # scalar
    local_amax,  # 1d tensor
    two_level_scale,  # scalar
    max_pos: tl.constexpr,
    F32_MIN_NORMAL: tl.constexpr,
    scale_round_mode: tl.constexpr,
    use_local_amax: tl.constexpr = False,
):
    """
    Tensorwise global amax
    """
    # Check for invalid values
    local_amax_invalid = (
        (local_amax == 0)
        | (local_amax == float("inf"))
        | (local_amax == float("-inf"))
        | (local_amax != local_amax)
    )
    global_amax_invalid = (
        (global_amax == 0)
        | (global_amax == float("inf"))
        | (global_amax == float("-inf"))
        | (global_amax != global_amax)
    )

    if use_local_amax:
        # A rule supplied ``local_amax`` with its per-block magnitude already
        # folded in, so there is nothing left to place it on.
        elem_format_max = 1.0
    else:
        elem_format_max = max_pos

    # Calculate local unscale
    local_unscale = (local_amax / elem_format_max).to(tl.float32)
    # two_level_scale is computed on the host (see `calculate_fp8_scale_for_nvfp4`)
    # so the kernel and eager paths share one fp64 value. A global amax of 0
    # gets 1.0 instead, so its block scales are 0.
    two_level_scale = tl.where(global_amax == 0, 1.0, two_level_scale)

    local_unscale_before_fp8 = (local_unscale * two_level_scale).to(tl.float32)
    # Ceil rounding adds to the raw bits, which turns the 0x7FFFFFFF NaN into -0.
    scale_is_nan = local_unscale_before_fp8 != local_unscale_before_fp8

    if scale_round_mode == "floor":
        # FP8_E4M3
        exp_bias = 7
        max_normal = 448.0
        min_normal = tl.exp2(1.0 - exp_bias)
        local_unscale_before_fp8 = _fp32_floor(
            local_unscale_before_fp8, 4, 3, max_normal, min_normal
        )
    elif scale_round_mode == "ceil":
        round_up_vals: tl.uint32 = 0x000FFFFF
        # FP8_E4M3
        exp_bias = 7
        max_normal = 448.0
        min_normal = tl.exp2(1.0 - exp_bias)
        # using the similar stochastic_rounding logic, but set the rand_values and 0x000FFFFF
        local_unscale_before_fp8 = _stochastic_rounding_helper(
            local_unscale_before_fp8, round_up_vals, 4, 3, max_normal, min_normal
        )
    # Saturate finite overflow and keep NaN, without PropagateNan before the FP8 cast.
    local_unscale_before_fp8 = tl.where(
        scale_is_nan,
        float("nan"),
        tl.clamp(local_unscale_before_fp8, -448.0, 448.0),
    )

    local_unscale_fp8 = local_unscale_before_fp8.to(tl.float8e4nv)
    local_unscale_q = local_unscale_fp8.to(tl.float32).to(tl.float64) / two_level_scale

    amax_invalid = local_amax_invalid | global_amax_invalid

    local_unscale_q = tl.where(amax_invalid, 1.0, local_unscale_q)

    scale = (1.0 / (local_unscale_q + F32_MIN_NORMAL)).to(tl.float32)
    unscale = (local_unscale_q).to(tl.float32)

    return scale, unscale, local_unscale_fp8


@jit
def _cal_fp32_to_e8m0_scale_unscale(
    unscale,
    mbits: tl.constexpr,
    scale_round_mode: tl.constexpr,
    norm_exp: tl.constexpr = 0,
):
    # When norm_exp != 0 (POW2MAX), `unscale` is the block amax (a normal fp32),
    # not amax/max_pow2. We round amax to a power of two and fold /max_pow2 =
    # /2**norm_exp in as an integer exponent shift below, so a subnormal
    # amax/max_pow2 cannot slip the oas boundary (losing the implicit 1). Mirrors
    # the pytorch fp32_to_e8m0.
    sign = tl.where(unscale < 0, -1.0, 1.0)
    abs_tensor = tl.abs(unscale)

    # MBITS_F32 = 23
    if scale_round_mode == "oas":
        val_to_add = (1 << (MBITS_F32 - mbits - 1)) - 1
    elif scale_round_mode == "oas_s":
        val_to_add = (3 << (MBITS_F32 - mbits - 1)) - 1
    elif scale_round_mode == "ceil":
        val_to_add = (1 << MBITS_F32) - 1
    else:
        val_to_add = 0

    mask_exponent = ((1 << (EBITS_F32 + SBITS)) - 1) << MBITS_F32
    mask_mantissa = (1 << MBITS_F32) - 1

    fp32_bits = tl.cast(abs_tensor, tl.int32, bitcast=True)
    fp32_bits_exp = (fp32_bits + val_to_add) & mask_exponent
    exponent = (fp32_bits_exp >> MBITS_F32) & 0xFF

    if scale_round_mode == "nv_round":
        mantissa = fp32_bits & mask_mantissa
        is_denormal = (exponent == 0) & (mantissa != 0)
        is_normal = ~is_denormal
        condition1 = is_normal & (exponent < 254) & (mantissa > 0)
        condition2 = is_denormal & (mantissa / (2**MBITS_F32) > 0.5)

        exponent = tl.where(condition1 | condition2, exponent + 1, exponent)

    if norm_exp != 0:
        # POW2MAX: amax was rounded to a power of two above; divide by
        # max_pow2 = 2**norm_exp via an integer exponent shift. maximum(.,1)
        # pins a subnormal amax's exponent to min-normal (1 - bias);
        # maximum(.,0) is the E8M0 floor.
        exponent = tl.maximum(tl.maximum(exponent, 1) - norm_exp, 0)

    exponent = exponent.to(tl.float32)
    e8m0_values = sign * tl.exp2(exponent - E8M0_EXPONENT_BIAS)

    unscale = e8m0_values
    # In case unscale=0 (scale will be inf), or unscale=inf or nan, we set the scale to 1.0
    unscale_invalid_mask = (
        (e8m0_values == 0)
        | (e8m0_values == float("inf"))
        | (e8m0_values == float("-inf"))
        | (e8m0_values != e8m0_values)
    )
    if norm_exp != 0:
        # `unscale` here is amax; the exponent shift can map an inf amax to a
        # finite e8m0, so guard inf/nan amax on the input explicitly.
        unscale_invalid_mask = (
            unscale_invalid_mask
            | (abs_tensor == float("inf"))
            | (abs_tensor != abs_tensor)
        )
    unscale = tl.where(unscale_invalid_mask, 1.0, unscale)

    scale = 1.0 / unscale

    return scale, unscale


@jit
def _mbs_static_factor(
    mbs_amax,
    fp_max_mbs: tl.constexpr,
    global_scale_dtype_mbs: tl.constexpr,
):
    # The static MBS factor of each macro group: fp_max / amax, as an E0M8
    # mantissa in [1, 2) or, with an FP32 global scale, the quotient itself.
    max_abs_safe = tl.where(mbs_amax == 0, 1.0, mbs_amax)
    sf = (
        tl.math.div_rn(tl.full(max_abs_safe.shape, 1.0, dtype=tl.float32), max_abs_safe)
        * fp_max_mbs
    )
    if global_scale_dtype_mbs == "e0m8":
        sf_bits = sf.to(tl.int32, bitcast=True)
        sf_bits = (sf_bits & 0x007FFFFF) | (127 << 23)
        mbs_val = sf_bits.to(tl.float32, bitcast=True)
    else:
        mbs_val = sf
    return mbs_val


@jit
def _mbs_dynamic_search(
    data_hp,
    per_row_amax,
    headroom,
    sse_zero_rows,
    max_pos: tl.constexpr,
    max_pow2: tl.constexpr,
    e8m0_norm_exp: tl.constexpr,
    min_exp: tl.constexpr,
    mbits: tl.constexpr,
    scale_norm_mode: tl.constexpr,
    scale_round_mode: tl.constexpr,
    mbs_dynamic_slots: tl.constexpr,
    num_mbs_groups: tl.constexpr,
    mbs_span: tl.constexpr,
):
    """The dynamic MBS search: the best factor of each macro group.

    Two things keep the loop cheap:
      * the sub-block amax scales linearly with the candidate, so
        `per_row_amax * gg` avoids re-reducing per candidate;
      * only the winning factor is carried between iterations (one scalar per
        macro group), not a reconstructed tile, so register pressure does not
        grow with the slot count.
    ``headroom`` (4 or 1 per quant block) is the headroom of qdq(): the E8M0
    scale of a top block is taken from its amax divided by 4 and multiplied
    back (both exact), so `amax * gg` cannot overflow.

    Limitation, as in qdq(): if every candidate's SSE overflows fp32 (peak
    >= 2^57), the group keeps 1.0 (plain MXFP4), where an input above
    1.75 * 2^127 (about 2.98e38) can still dequantize to +-Inf.
    """
    per_row_amax = per_row_amax / headroom
    best_sse = tl.full((num_mbs_groups,), float("inf"), tl.float32)
    best_g = tl.full((num_mbs_groups,), 1.0, tl.float32)
    for slot in tl.static_range(mbs_dynamic_slots):
        gg = 1.0 + slot / mbs_dynamic_slots
        delta_g = per_row_amax * gg
        # Correctly rounded, as in qdq(); see the kernel's unscale.
        if scale_norm_mode == "pow2max" and e8m0_norm_exp != 0:
            unscale_g = delta_g / max_pow2
        elif scale_norm_mode == "pow2max":
            unscale_g = tl.math.div_rn(delta_g, max_pow2)
        else:
            unscale_g = tl.math.div_rn(delta_g, max_pos)
        _, unscale_g = _cal_fp32_to_e8m0_scale_unscale(
            delta_g if e8m0_norm_exp != 0 else unscale_g,
            mbits,
            scale_round_mode,
            e8m0_norm_exp,
        )
        safe_g = tl.where(unscale_g == 0, 1.0, unscale_g * headroom)
        # Reciprocals hoisted out of the element-wise work: `safe_g` is one
        # value per quant block ([BLOCK_M]) and `gg` is a constant in this
        # unrolled iteration, so the tile sees multiplies rather than two
        # full-tile divisions per candidate.
        inv_g = tl.math.div_rn(tl.full(safe_g.shape, 1.0, dtype=tl.float32), safe_g)
        scale_back = safe_g * (1.0 / gg)
        q = _fp32_to_fpx_helper(
            data_hp * (gg * inv_g[:, None]),
            mbits,
            min_exp,
            max_pos,
            ranking_only=True,
        )
        diff = q * scale_back[:, None] - data_hp
        sse_row = tl.sum(diff * diff, axis=1)
        # A NaN poisons its quant block, which qdq() scores as zero error rather
        # than letting it make every candidate's SSE NaN.
        sse_row = tl.where(sse_zero_rows, 0.0, sse_row)
        sse_grp = tl.sum(tl.reshape(sse_row, (num_mbs_groups, mbs_span)), axis=1)
        take = sse_grp < best_sse
        best_sse = tl.where(take, sse_grp, best_sse)
        best_g = tl.where(take, gg, best_g)
    return best_g


@jit
def _ptx_fp32_to_packed_e2m1_i32(
    data_lp,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Convert fp32 data to packed int32 E2M1 using Blackwell PTX.

    Packs 8 fp32 values into 1 int32 (4 bytes, 2 FP4 per byte) using
    cvt.rn.satfinite.e2m1x2.f32 + mov.b32.

    Args:
        data_lp: [BLOCK_M, BLOCK_N] fp32 tensor (scaled, not yet rounded)

    Returns:
        [BLOCK_M, BLOCK_N // 8] int32 tensor with packed E2M1 values
    """
    NUMEL: tl.constexpr = BLOCK_M * BLOCK_N

    # Extract 8 individual element streams via successive splits.
    # tl.split on [N, 2] separates even/odd indexed elements.
    lo, hi = tl.split(tl.reshape(data_lp, [NUMEL // 2, 2]))
    lo_a, lo_b = tl.split(tl.reshape(lo, [NUMEL // 4, 2]))
    hi_a, hi_b = tl.split(tl.reshape(hi, [NUMEL // 4, 2]))
    e0, e4 = tl.split(tl.reshape(lo_a, [NUMEL // 8, 2]))
    e2, e6 = tl.split(tl.reshape(lo_b, [NUMEL // 8, 2]))
    e1, e5 = tl.split(tl.reshape(hi_a, [NUMEL // 8, 2]))
    e3, e7 = tl.split(tl.reshape(hi_b, [NUMEL // 8, 2]))

    # Pack 8 fp32 → 4 bytes → 1 int32
    # byte_k = cvt(elem[2k+1], elem[2k]) for k=0..3
    packed_i32 = tl.inline_asm_elementwise(
        asm="""
        {
            .reg .b8 byte0;
            .reg .b8 byte1;
            .reg .b8 byte2;
            .reg .b8 byte3;
            cvt.rn.satfinite.e2m1x2.f32  byte0, $2, $1;
            cvt.rn.satfinite.e2m1x2.f32  byte1, $4, $3;
            cvt.rn.satfinite.e2m1x2.f32  byte2, $6, $5;
            cvt.rn.satfinite.e2m1x2.f32  byte3, $8, $7;
            mov.b32 $0, {byte0, byte1, byte2, byte3};
        }
        """,
        constraints="=r,f,f,f,f,f,f,f,f",
        args=[e0, e1, e2, e3, e4, e5, e6, e7],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )

    return tl.reshape(packed_i32, [BLOCK_M, BLOCK_N // 8])


@jit
def _mega_fake_quant_triton_kernel(
    in_ptr,
    out_ptr,
    global_amax_ptr,  # for nv-style scaling factor
    two_level_scale_ptr,  # for nv-style scaling factor
    local_unscale_ptr,
    data_lp_ptr,
    local_amax_ptr,
    seed,  # stochastic rounding only: an int32 scalar, or a pointer to an int64[1]
    M,
    N,
    new_M,  # M / m *  N / n
    new_N,  # m * n
    stride_in0,
    stride_in1,
    stride_out0,
    stride_out1,
    stride_lp0,
    stride_lp1,
    m: tl.constexpr,  # the quant block size m
    n: tl.constexpr,  # the quant block size n
    min_pos: tl.constexpr,
    max_pos: tl.constexpr,
    max_pow2: tl.constexpr,
    e8m0_norm_exp: tl.constexpr,  # log2(max_pow2) for POW2MAX e8m0, else 0
    min_exp: tl.constexpr,
    ebits: tl.constexpr,
    mbits: tl.constexpr,
    max_normal: tl.constexpr,
    min_normal: tl.constexpr,
    scale_dtype: tl.constexpr,
    scale_norm_mode: tl.constexpr,
    scale_round_mode: tl.constexpr,
    quant_mode: tl.constexpr,
    elem_dtype: tl.constexpr,
    elem_round_mode: tl.constexpr,
    global_scale_granularity: tl.constexpr,
    use_mbs: tl.constexpr,  # Whether MBS is enabled
    return_quant_info: tl.constexpr,
    native_quant: tl.constexpr,  # Output packed native format (uint8/fp8) instead of dequant
    MBS_BLOCK_SIZE: tl.constexpr,  # 0 when MBS not used
    fp_max_mbs: tl.constexpr,  # max representable value for elem_dtype, 0 when MBS not used
    global_scale_dtype_mbs: tl.constexpr,  # "e0m8" or "fp32" with MBS, else "none"
    mbs_dynamic: tl.constexpr,  # search the macro factor instead of deriving it from amax
    mbs_dynamic_slots: tl.constexpr,  # candidate count for that search, 0 when unused
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,  # = new_N
    F32_MIN_NORMAL: tl.constexpr,
    # Whether the per-block amax is read from `local_amax_ptr` -- already
    # divided by the rule's per-block magnitude -- instead of reduced here and
    # placed on the scalar `max_pos`.
    use_local_amax: tl.constexpr = False,
    # Valid-row bound (see `qdq_triton_mega`'s `num_valid_rows`): the input
    # holds data only in its first `num_valid_rows_ptr[0]` rows.
    num_valid_rows_ptr=None,
    HAS_NUM_VALID_ROWS: tl.constexpr = False,
    # The unpadded rows and columns of the input, which is zero-padded to whole
    # quant blocks. Read only under ASYMM, which leaves the padding out of each
    # block's min and max.
    unpadded_M=None,
    unpadded_N=None,
):
    if HAS_NUM_VALID_ROWS:
        # Loaded here, not on the host, so the bound needs no sync and a CUDA
        # graph replays it with a new value.
        valid_rows = tl.load(num_valid_rows_ptr).to(tl.int64)
        valid_new_M = ((valid_rows + m - 1) // m) * (N // n)
        if tl.program_id(0).to(tl.int64) * BLOCK_M >= valid_new_M:
            # The bound only skips programs whose whole tile is past it (rounded
            # up to whole quant-block rows). Programs that run use the same
            # load/store masks as without a bound, so rows below it are
            # bit-identical; at most one program straddles the bound and also
            # writes rows past it, whose output is unspecified.
            return
    # reshape (M, N) to (M / m *  N / n, m * n)
    # [BLOCK_M, BLOCK_N]
    row_offset = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offset = tl.arange(0, BLOCK_N)

    n_offset = tl.arange(0, BLOCK_M * BLOCK_N)
    n_offset = tl.reshape(n_offset, (BLOCK_M, BLOCK_N))

    mask = (row_offset[:, None] < new_M) & (col_offset[None, :] < new_N)

    tiles_per_row = N // n
    # which block in the grid
    tile_row = row_offset // tiles_per_row
    tile_col = row_offset % tiles_per_row

    # within block coordinates
    intra_row = col_offset // n
    intra_col = col_offset % n

    # map to original row_idx
    orig_row_idx = tile_row[:, None] * m + intra_row[None, :]
    orig_col_idx = tile_col[:, None] * n + intra_col[None, :]

    in_offset = orig_row_idx * stride_in0 + orig_col_idx * stride_in1
    out_offset = orig_row_idx * stride_out0 + orig_col_idx * stride_out1
    lp_offset = orig_row_idx * stride_lp0 + orig_col_idx * stride_lp1

    data_hp = tl.load(in_ptr + in_offset, mask=mask, other=0.0).to(tl.float32)

    if quant_mode == "asymm":
        # Zero is not neutral for a min or a max, so ASYMM leaves the zero
        # padding of a partial quant block out of them.
        excluded = (orig_row_idx >= unpadded_M) | (orig_col_idx >= unpadded_N)

    # MX (E8M0) FP formats: a NaN
    # poisons the whole block (E8M0 NaN scale byte 255 -> all-NaN dequant) and
    # ±Inf saturates to ±FLT_MAX before the amax/scale path. Mirrors the
    # pure-PyTorch qdq edge-case handling. Other formats keep verbatim Inf/NaN
    # passthrough, except ASYMM_MEAN_SHIFT, which has its own rule (below).
    is_mx_fp: tl.constexpr = (
        (scale_dtype == "e8m0")
        and ((elem_dtype == "e2m1") or (elem_dtype == "e2m3") or (elem_dtype == "e4m3"))
        and (quant_mode != "asymm_mean_shift")
    )
    # A quant block whose finite peak is 2^126 or more can overflow fp32 once
    # multiplied by an MBS factor in [1, 2): it is divided by 4 / factor instead,
    # as in qdq(). An FP32/FP16 static factor is fp_max / amax and cannot.
    use_headroom: tl.constexpr = use_mbs and (
        mbs_dynamic or (global_scale_dtype_mbs == "e0m8")
    )
    if use_mbs and not mbs_dynamic:
        # The static factor is derived from the input before the +-FLT_MAX clamp
        # below, as in qdq(): an Inf amax has a zero reciprocal, so its MBS block
        # gets factor 1.0 rather than the factor of FLT_MAX.
        static_row_amax = tl.max(tl.abs(data_hp), axis=1)

    if is_mx_fp:
        mx_nan_block = tl.max((data_hp != data_hp).to(tl.int32), 1) > 0
        data_hp = tl.clamp(data_hp, -F32_MAX, F32_MAX)

    if use_mbs:
        # Compute MBS inline from loaded data
        # mbs_span: number of quant blocks per MBS block
        mbs_span: tl.constexpr = MBS_BLOCK_SIZE // n
        num_mbs_groups: tl.constexpr = BLOCK_M // mbs_span

        row_mask = row_offset < new_M
        if mbs_dynamic:
            per_row_amax = tl.max(tl.abs(data_hp), axis=1)
        else:
            per_row_amax = static_row_amax
        per_row_amax = tl.where(row_mask, per_row_amax, 0.0)

        if mbs_span == 1:
            mbs_amax = per_row_amax
        else:
            group_amaxes = tl.reshape(per_row_amax, (num_mbs_groups, mbs_span))
            mbs_amax = tl.max(group_amaxes, axis=1)

        if mbs_dynamic:
            # Search the macro factor instead of deriving it from the amax.
            #
            # Fully fused: the candidates are evaluated on `data_hp` while it is
            # still in registers, so this costs neither extra HBM traffic nor extra
            # kernel launches -- which is the whole reason the out-of-kernel version
            # is dramatically slower than static, far beyond the ~slot-count factor
            # the extra quantization passes alone would predict.
            if is_mx_fp:
                sse_zero_rows = mx_nan_block
            else:
                sse_zero_rows = row_offset < 0
            # For MX FP formats `per_row_amax` is post-clamp, never Inf, so it needs
            # no `<= F32_MAX` bound (unlike `_mbs_headroom`). qdq() searches unclamped
            # data, so an Inf in a dynamic macro block can still pick another factor.
            row_top = per_row_amax >= MBS_HEADROOM_MIN_PEAK
            headroom = tl.where(row_top, 4.0, 1.0)
            mbs_val = _mbs_dynamic_search(
                data_hp,
                per_row_amax,
                headroom,
                sse_zero_rows,
                max_pos,
                max_pow2,
                e8m0_norm_exp,
                min_exp,
                mbits,
                scale_norm_mode,
                scale_round_mode,
                mbs_dynamic_slots,
                num_mbs_groups,
                mbs_span,
            )
        else:
            mbs_val = _mbs_static_factor(mbs_amax, fp_max_mbs, global_scale_dtype_mbs)
            if use_headroom:
                row_top = (static_row_amax >= MBS_HEADROOM_MIN_PEAK) & (
                    static_row_amax <= F32_MAX
                )

        mbs_val = tl.where(mbs_amax == 0, 1.0, mbs_val)
        mbs_reciprocal_per_group = tl.math.div_rn(
            tl.full(mbs_val.shape, 1.0, dtype=tl.float32), mbs_val
        )

        if mbs_span == 1:
            mbs_reciprocal = mbs_reciprocal_per_group
        else:
            mbs_reciprocal = tl.reshape(
                tl.broadcast_to(
                    mbs_reciprocal_per_group[:, None], (num_mbs_groups, mbs_span)
                ),
                (BLOCK_M,),
            )
        if use_headroom:
            mbs_reciprocal = tl.where(row_top, mbs_reciprocal * 4.0, mbs_reciprocal)
        data_hp = tl.math.div_rn(data_hp, mbs_reciprocal[:, None])

    """
    Calculate the local_amax of m*n block
    """

    if quant_mode == "asymm":
        # The padding is left out, and a block of padding only is a zero
        # block.
        max_value = tl.max(tl.where(excluded, float("-inf"), data_hp), 1)
        min_value = tl.min(tl.where(excluded, float("inf"), data_hp), 1)
        empty = max_value < min_value
        max_value = tl.where(empty, 0.0, max_value)
        min_value = tl.where(empty, 0.0, min_value)
        zero_point = (max_value + min_value) / 2 if min_pos < 0 else min_value
        quant_zero_point = zero_point
        # we assume the offsets use fp16 dtype
        zero_point = zero_point.to(tl.float16)
        # tl.store(zero_point_ptr + row_offset, zero_point, mask=row_mask)

        delta = max_value - min_value
        # Float formats are signed and centered on the midpoint, so they span
        # +-delta/2; int formats are unsigned from the min, so they span delta.
        if not (elem_dtype == "int3" or elem_dtype == "int4" or elem_dtype == "int8"):
            delta = delta / 2.0

    elif quant_mode == "asymm_mean_shift":
        # Compute the mean shift through fixed-point ints so Triton reduction
        # order does not change the shift bitwise. As in qdq(): tl.max skips a
        # NaN, a NaN (and Inf * 0) converts to int 0, and an Inf gives
        # int_scale = 0, so the mean of its block is 0 / 0 = NaN.
        max_abs = tl.maximum(tl.max(tl.abs(data_hp), 1), 1.0e-12)
        fixed_point_one: tl.constexpr = 2097152
        int_scale = fixed_point_one / max_abs
        out_int = (data_hp * int_scale[:, None]).to(tl.int32)
        int_sum = tl.sum(out_int, 1)
        quant_zero_point = int_sum.to(tl.float32) / (int_scale * new_N)
        zero_point = quant_zero_point.to(tl.float16)

        centered_vals = data_hp - quant_zero_point[:, None]
        delta = tl.max(tl.abs(centered_vals), 1)
        # All NaN (a block with an Inf): a zero block, as in qdq().
        delta = tl.where(delta == delta, delta, 0.0)

    else:  # "symm"
        # local amax
        abs_vals = tl.abs(data_hp)
        delta = tl.max(abs_vals, 1)

    # Triton's `/` by a constant multiplies by its rounded reciprocal. Divide
    # correctly rounded instead, as qdq() does, unless the divisor is a power
    # of two (exact either way) or under ASYMM_MEAN_SHIFT, which keeps `/`
    # (not `div_rn`) in qdq() and here.
    if scale_norm_mode == "pow2max":
        if e8m0_norm_exp != 0 or quant_mode == "asymm_mean_shift":
            unscale = delta / max_pow2
        else:
            unscale = tl.math.div_rn(delta, max_pow2)
    elif quant_mode == "asymm_mean_shift":
        unscale = delta / max_pos
    else:  # amax
        unscale = tl.math.div_rn(delta, max_pos)
    quant_unscale = unscale

    """
    Cast scaling factor
    """
    if scale_dtype == "fp32":
        if quant_mode == "asymm_mean_shift":
            unscale = tl.maximum(unscale, 1.0e-12)
            fp32_pre = 1.0
        else:
            # As in qdq(): a zero block gets FLT_MIN, any other block at least
            # the smallest positive scale, and a scale below FLT_MIN is
            # inverted with the elements times 2^64 (`_scale_elements`).
            eps = F32_MIN_NORMAL * (delta == 0).to(tl.float32)
            unscale = unscale + eps
            unscale = tl.where(unscale < F32_MIN_SUBNORMAL, F32_MIN_SUBNORMAL, unscale)
            fp32_pre = tl.where(unscale < F32_MIN_NORMAL, 2.0**64, 1.0)
        quant_unscale = unscale
        scale, unscale = _cal_fp32_scale_unscale(unscale, fp32_pre)
    elif scale_dtype == "fp16":
        if quant_mode == "asymm_mean_shift":
            unscale = tl.maximum(unscale, 1.0e-12)
        else:
            eps = F16_MIN_NORMAL * (delta == 0).to(tl.float32)
            unscale = unscale + eps
            unscale = tl.where(unscale < F16_MIN_SUBNORMAL, F16_MIN_SUBNORMAL, unscale)
        quant_unscale = unscale
        scale, unscale = _cal_fp16_scale_unscale(
            unscale,
        )
    elif scale_dtype == "fp8":
        if global_scale_granularity == "rowwise":
            global_amax = tl.load(
                global_amax_ptr + tile_row,
                mask=row_offset < new_M,
                other=1.0,
            )
            two_level_scale = tl.load(
                two_level_scale_ptr + tile_row,
                mask=row_offset < new_M,
                other=1.0,
            )
        elif global_scale_granularity == "tensorwise":
            global_amax = tl.load(global_amax_ptr)
            two_level_scale = tl.load(two_level_scale_ptr)
        else:
            # A single-level E4M3 block scale: a unit global scale.
            global_amax = 1.0
            two_level_scale = 1.0
        if use_local_amax:
            local_amax = tl.load(
                local_amax_ptr + row_offset,
                mask=row_offset < new_M,
                other=0.0,
            )
        else:
            local_amax = delta
        scale, unscale, local_unscale_fp8 = _cal_fp8_scale_unscale_for_nvfp4(
            global_amax,
            local_amax,
            two_level_scale,
            max_pos,
            F32_MIN_NORMAL=F32_MIN_NORMAL,
            scale_round_mode=scale_round_mode,
            use_local_amax=use_local_amax,
        )
    elif scale_dtype == "e8m0":
        if scale_norm_mode == "pow2max":
            # Round amax (delta) then fold /max_pow2 as an integer exponent
            # shift, instead of rounding the (possibly subnormal) delta/max_pow2.
            # For non-pow2 max_pow2 (E8Mx) e8m0_norm_exp == 0 and we fall back to
            # the pre-divided `unscale` (= delta / max_pow2).
            scale, unscale = _cal_fp32_to_e8m0_scale_unscale(
                delta if e8m0_norm_exp != 0 else unscale,
                mbits,
                scale_round_mode,
                e8m0_norm_exp,
            )
        else:
            scale, unscale = _cal_fp32_to_e8m0_scale_unscale(
                unscale,
                mbits,
                scale_round_mode,
            )

    if is_mx_fp:
        # NaN block -> E8M0 NaN scale byte 255 (unscale = +Inf, fp32 exponent
        # field all ones). The whole block dequantizes to NaN below.
        unscale = tl.where(mx_nan_block, float("inf"), unscale)

    if native_quant:
        # Store scales in packed hardware format directly
        if scale_dtype == "fp8":
            tl.store(
                local_unscale_ptr + row_offset,
                local_unscale_fp8.to(tl.float8e4nv).to(tl.uint8, bitcast=True),
                mask=row_offset < new_M,
            )
        else:  # e8m0: extract biased exponent as uint8
            unscale_bits = unscale.to(tl.int32, bitcast=True)
            abs_bits = unscale_bits & 0x7FFFFFFF
            biased_exp = (abs_bits >> 23) & 0xFF
            biased_exp = tl.where(abs_bits == 0, 0, biased_exp)
            tl.store(
                local_unscale_ptr + row_offset,
                biased_exp.to(tl.uint8),
                mask=row_offset < new_M,
            )
    elif return_quant_info:
        if scale_dtype == "fp8":
            tl.store(
                local_unscale_ptr + row_offset,
                local_unscale_fp8.to(tl.float32),
                mask=row_offset < new_M,
            )
        elif use_headroom:
            # Report the block scale of the factor-scaled data: the 4 a top
            # quant block was divided by belongs to its scale.
            tl.store(
                local_unscale_ptr + row_offset,
                tl.where(row_top, unscale * 4.0, unscale),
                mask=row_offset < new_M,
            )
        else:
            tl.store(local_unscale_ptr + row_offset, unscale, mask=row_offset < new_M)

    """
    Cast data_hp to data_lp
    data_lp = cast(data_hp * scale)
    dequant_data = data_lp * unscale
    """

    if quant_mode == "asymm_mean_shift":
        data_hp = data_hp - quant_zero_point[:, None]
    elif quant_mode == "asymm":
        data_hp = data_hp - zero_point[:, None]
    if (
        scale_dtype == "fp8"
        and quant_mode == "symm"
        and (
            global_scale_granularity == "tensorwise"
            or global_scale_granularity == "rowwise"
        )
    ):
        # NVFP4 precise: use (data_hp * gs) / lus instead of data_hp * scale
        # to preserve exact E2M1 midpoints. Avoids ~1 ULP drift from the
        # fp64->fp32 cast on the combined factor. ASYMM and ASYMM_MEAN_SHIFT
        # quantize with the combined factor instead, as qdq() does.
        # local_unscale_fp8 (per-row fp8 byte) and two_level_scale (fp64)
        # are loaded above in the scale_dtype == "fp8" branch.
        gs_fp32 = two_level_scale.to(tl.float32)
        lus_fp32 = local_unscale_fp8.to(tl.float32)
        gs_invalid = (
            (gs_fp32 == 0)
            | (gs_fp32 == float("inf"))
            | (gs_fp32 == float("-inf"))
            | (gs_fp32 != gs_fp32)
        )
        lus_invalid = (
            (lus_fp32 == 0)
            | (lus_fp32 == float("inf"))
            | (lus_fp32 == float("-inf"))
            | (lus_fp32 != lus_fp32)
        )
        gs_safe = tl.where(gs_invalid, 1.0, gs_fp32)
        lus_safe = tl.where(lus_invalid, 1.0, lus_fp32)
        if global_scale_granularity == "rowwise":
            data_lp = tl.math.div_rn(data_hp * gs_safe[:, None], lus_safe[:, None])
        else:
            data_lp = tl.math.div_rn(data_hp * gs_safe, lus_safe[:, None])
    elif quant_mode == "asymm_mean_shift" and scale_dtype == "fp16":
        # Quantize with the fp32 scale; the fp16 scale is stored for dequant.
        data_lp = data_hp / quant_unscale[:, None]
    elif scale_dtype == "fp32" and quant_mode != "asymm_mean_shift":
        data_lp = (data_hp * fp32_pre[:, None]) * scale[:, None]
    else:
        data_lp = data_hp * scale[:, None]

    if elem_round_mode == "stochastic":
        if seed.dtype.is_ptr():
            tile_seed = tl.load(seed) + tl.program_id(0)
        else:
            tile_seed = seed + tl.program_id(0)
        rand_val = tl.randint(tile_seed, n_offset)
        rand_val = rand_val.to(tl.uint32)
        data_lp = _stochastic_rounding_helper(
            data_lp, rand_val, ebits, mbits, max_normal, min_normal
        )

    if elem_dtype == "int3" or elem_dtype == "int4" or elem_dtype == "int8":
        # clamp to [min_pos, max_pos]
        data_lp = tl.clamp(_round_int(data_lp), min_pos, max_pos)
    elif (elem_dtype == "e3m0" or elem_dtype == "e2m1") or (elem_dtype == "e2m3"):
        if is_mx_fp:
            # MX NaN-block element code for E2M1/E2M3 is canonical all-ones
            # (max-magnitude finite, sign dropped). Apply before native E2M1
            # packing so packed output gets the same element code.
            data_lp = tl.where(mx_nan_block[:, None], max_pos, data_lp)
        if native_quant and elem_dtype == "e2m1":
            packed_e2m1 = _ptx_fp32_to_packed_e2m1_i32(data_lp, BLOCK_M, BLOCK_N)
        else:
            # _fp32_to_fpx_helper rounds in fp64 unconditionally to match
            # the pure fp32_to_fpx reference. A scaled value ~1 ULP below a grid
            # midpoint (e.g. 0.49999997) rounds up to 1.0 when (absA + 0.5) is
            # evaluated in fp32, pushing a below-midpoint element up a grid step;
            # fp64 rounds it correctly. A rule-supplied per-block `local_amax`
            # produces exactly such values for bf16 inputs, so without the fp64
            # step the fused kernel would diverge from qdq() at the
            # zero/first-step boundary.
            data_lp = _fp32_to_fpx_helper(data_lp, mbits, min_exp, max_pos)
    elif elem_dtype == "e4m3":
        if native_quant or not is_mx_fp or (use_mbs and not use_headroom):
            data_lp = tl.clamp(data_lp, min_pos, max_pos)
        # Otherwise the conversion below saturates to [min_pos, max_pos] itself
        # (satfinite): an MX block is finite here (+-Inf was clamped to
        # +-FLT_MAX above), and its NaN is set just below. The clamp also maps a
        # NaN element to a code: outside MX, and under a static FP32 MBS factor,
        # which overflows to Inf on a macro block with an amax below
        # fp_max / FLT_MAX and turns its zeros into NaN.
        if is_mx_fp:
            # Apply before casting to tl.float8e4nv. Triton cannot materialize
            # an fp8 scalar NaN constant inside tl.where after the cast.
            data_lp = tl.where(mx_nan_block[:, None], float("nan"), data_lp)
        data_lp = data_lp.to(tl.float8e4nv)
        if not native_quant:
            data_lp = data_lp.to(tl.float32)
    elif elem_dtype == "e8m1":
        data_lp = _fp32_to_e8mx_helper(data_lp, _fp32_to_e8m1_kernel)
    elif elem_dtype == "e8m2":
        data_lp = _fp32_to_e8mx_helper(data_lp, _fp32_to_e8m2_kernel)
    elif elem_dtype == "e8m3":
        data_lp = _fp32_to_e8mx_helper(data_lp, _fp32_to_e8m3_kernel)

    if native_quant and elem_dtype == "e2m1":
        pack_col = tl.arange(0, BLOCK_N // 8)
        pack_orig_col = tile_col[:, None] * (n // 8) + pack_col[None, :]
        lp_offset = tile_row[:, None] * stride_lp0 + pack_orig_col * stride_lp1
        mask = row_offset[:, None] < new_M
        tl.store(data_lp_ptr + lp_offset, packed_e2m1, mask=mask)
        return
    elif native_quant:
        tl.store(data_lp_ptr + lp_offset, data_lp, mask=mask)
        return

    if quant_mode == "asymm_mean_shift":
        # As in qdq(): a NaN input, and every element of a block with an Inf,
        # get a NaN code and dequantize to NaN, with no passthrough.
        data_lp = tl.where(data_hp != data_hp, float("nan"), data_lp)

    if return_quant_info:
        tl.store(data_lp_ptr + lp_offset, data_lp, mask=mask)

    # dequantize
    if (
        quant_mode == "asymm_mean_shift"
        and scale_dtype == "fp16"
        and elem_dtype == "e4m3"
    ):
        # Dequant FP8 -> FP16, then FP16 scale/shift multiply-add.
        dequant_data = (
            (data_lp.to(tl.float16) * unscale[:, None]) + zero_point[:, None]
        ).to(tl.float32)
    elif quant_mode == "asymm" or quant_mode == "asymm_mean_shift":
        # Round the product before adding the zero point, as qdq() does: a
        # contracted multiply-add rounds once and can land 1 ULP away when the
        # unscale is not a power of two. An fp16 unscale is widened first, as
        # qdq()'s type promotion does.
        dequant_data = (
            _mul_rn(data_lp, unscale[:, None].to(tl.float32)) + zero_point[:, None]
        )
    else:
        dequant_data = data_lp * unscale[:, None]

    if use_mbs:
        dequant_data = dequant_data * mbs_reciprocal[:, None]

    # handle edge cases for input data
    if is_mx_fp:
        # NaN poisons the whole MX block; ±Inf was already saturated pre-amax.
        dequant_data = tl.where(mx_nan_block[:, None], float("nan"), dequant_data)
    elif quant_mode != "asymm_mean_shift":
        data_hp_inf_mask = (data_hp == float("inf")) | (data_hp == float("-inf"))
        data_hp_nan_mask = data_hp != data_hp
        inf_nan_mask = data_hp_inf_mask | data_hp_nan_mask
        dequant_data = tl.where(inf_nan_mask, data_hp, dequant_data)

    """
    Save the dequant_data back to the original tensor shape
    """

    tl.store(out_ptr + out_offset, dequant_data, mask=mask)


@jit
def _group_counts_to_offsets_and_zero_kernel(
    group_counts_ptr,
    group_offsets_ptr,
    group_amax_ptr,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    e = tl.arange(0, BLOCK_E)
    e_mask = e < E
    counts = tl.load(group_counts_ptr + e, mask=e_mask, other=0).to(tl.int32)
    offsets = tl.cumsum(counts, axis=0)
    tl.store(group_offsets_ptr + e, offsets, mask=e_mask)
    tl.store(group_amax_ptr + e, tl.zeros((BLOCK_E,), dtype=tl.float32), mask=e_mask)


@jit
def _group_ids_from_offsets(
    row_ids,
    row_mask,
    group_offsets_ptr,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    e = tl.arange(0, BLOCK_E)
    e_mask = e < E
    offsets = tl.load(group_offsets_ptr + e, mask=e_mask, other=2147483647).to(tl.int64)
    past_group = row_ids[:, None] >= offsets[None, :]
    group_ids = tl.sum((past_group & row_mask[:, None]).to(tl.int32), axis=1)
    return tl.minimum(group_ids, E - 1)


@jit
def _grouped_nvfp4_amax_triton_kernel(
    in_ptr,
    group_offsets_ptr,
    group_amax_ptr,
    M,
    N,
    stride_in0,
    stride_in1,
    E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    total_rows = tl.load(group_offsets_ptr + E - 1).to(tl.int64)
    row_mask = (rows < M) & (rows < total_rows)
    col_mask = cols < N

    offsets = rows[:, None] * stride_in0 + cols[None, :] * stride_in1
    data = tl.load(
        in_ptr + offsets,
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    )
    row_amax = tl.max(tl.abs(data.to(tl.float32)), axis=1)
    group_ids = _group_ids_from_offsets(
        rows.to(tl.int64),
        row_mask,
        group_offsets_ptr,
        E,
        BLOCK_E,
    )
    tl.atomic_max(group_amax_ptr + group_ids, row_amax, mask=row_mask)


@jit
def _grouped_nvfp4_fake_quant_triton_kernel(
    in_ptr,
    out_ptr,
    group_offsets_ptr,
    group_amax_ptr,
    group_two_level_scale_ptr,
    seed,  # stochastic rounding only: an int32 scalar, or a pointer to an int64[1]
    M,
    N,
    new_M,
    new_N,
    stride_in0,
    stride_in1,
    stride_out0,
    stride_out1,
    n: tl.constexpr,
    min_exp: tl.constexpr,
    ebits: tl.constexpr,
    mbits: tl.constexpr,
    max_normal: tl.constexpr,
    min_normal: tl.constexpr,
    max_pos: tl.constexpr,
    scale_round_mode: tl.constexpr,
    elem_round_mode: tl.constexpr,
    E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_E: tl.constexpr,
    F32_MIN_NORMAL: tl.constexpr,
    use_static_global_scale: tl.constexpr,
):
    row_offset = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offset = tl.arange(0, BLOCK_N)
    n_offset = tl.arange(0, BLOCK_M * BLOCK_N)
    n_offset = tl.reshape(n_offset, (BLOCK_M, BLOCK_N))

    mask = (row_offset[:, None] < new_M) & (col_offset[None, :] < new_N)
    tiles_per_row = N // n
    tile_row = row_offset // tiles_per_row
    tile_col = row_offset % tiles_per_row
    orig_col_idx = tile_col[:, None] * n + col_offset[None, :]

    total_rows = tl.load(group_offsets_ptr + E - 1).to(tl.int64)
    quant_row_mask = (tile_row < M) & (tile_row < total_rows)
    group_ids = _group_ids_from_offsets(
        tile_row.to(tl.int64),
        quant_row_mask,
        group_offsets_ptr,
        E,
        BLOCK_E,
    )

    in_offset = tile_row[:, None] * stride_in0 + orig_col_idx * stride_in1
    out_offset = tile_row[:, None] * stride_out0 + orig_col_idx * stride_out1
    data_hp = tl.load(in_ptr + in_offset, mask=mask, other=0.0).to(tl.float32)

    local_amax = tl.max(tl.abs(data_hp), axis=1)
    if use_static_global_scale:
        global_amax = tl.full((BLOCK_M,), 1.0, tl.float32)
    else:
        global_amax = tl.load(
            group_amax_ptr + group_ids, mask=quant_row_mask, other=1.0
        )
    two_level_scale = tl.load(
        group_two_level_scale_ptr + group_ids, mask=quant_row_mask, other=1.0
    )
    _, unscale, local_unscale_fp8 = _cal_fp8_scale_unscale_for_nvfp4(
        global_amax,
        local_amax,
        two_level_scale,
        max_pos,
        F32_MIN_NORMAL=F32_MIN_NORMAL,
        scale_round_mode=scale_round_mode,
    )

    gs_fp32 = two_level_scale.to(tl.float32)
    lus_fp32 = local_unscale_fp8.to(tl.float32)
    gs_invalid = (
        (gs_fp32 == 0)
        | (gs_fp32 == float("inf"))
        | (gs_fp32 == float("-inf"))
        | (gs_fp32 != gs_fp32)
    )
    lus_invalid = (
        (lus_fp32 == 0)
        | (lus_fp32 == float("inf"))
        | (lus_fp32 == float("-inf"))
        | (lus_fp32 != lus_fp32)
    )
    gs_safe = tl.where(gs_invalid, 1.0, gs_fp32)
    lus_safe = tl.where(lus_invalid, 1.0, lus_fp32)
    data_lp = tl.math.div_rn(data_hp * gs_safe[:, None], lus_safe[:, None])

    if elem_round_mode == "stochastic":
        if seed.dtype.is_ptr():
            tile_seed = tl.load(seed) + tl.program_id(0)
        else:
            tile_seed = seed + tl.program_id(0)
        rand_val = tl.randint(tile_seed, n_offset)
        rand_val = rand_val.to(tl.uint32)
        data_lp = _stochastic_rounding_helper(
            data_lp, rand_val, ebits, mbits, max_normal, min_normal
        )

    data_lp = _fp32_to_fpx_helper(data_lp, mbits, min_exp, max_pos)
    dequant_data = data_lp * unscale[:, None]

    data_hp_inf_mask = (data_hp == float("inf")) | (data_hp == float("-inf"))
    data_hp_nan_mask = data_hp != data_hp
    inf_nan_mask = data_hp_inf_mask | data_hp_nan_mask
    dequant_data = tl.where(inf_nan_mask, data_hp, dequant_data)
    dequant_data = tl.where(quant_row_mask[:, None], dequant_data, data_hp)

    tl.store(out_ptr + out_offset, dequant_data, mask=mask)


# *****************************************************************************
# * Host-side launch helpers                                                  *
# *****************************************************************************
_U64_MASK = (1 << 64) - 1


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _U64_MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _U64_MASK
    return x ^ (x >> 31)


def _triton_sr_seed(device: torch.device) -> "torch.Tensor | int":
    """A fresh stochastic-rounding seed from ``device``'s CUDA generator.

    Eager: a host int below 2**30 derived from the generator's seed and philox
    offset, which is then advanced; no device work, no sync, and the kernel takes
    it as an int32 scalar argument (2**30 headroom for seed + program_id). Under
    CUDA graph capture: a one-element int64 tensor below 2**62 drawn on the
    device, so each replay gets a new seed.
    """
    if torch.cuda.is_current_stream_capturing():
        return torch.randint(2**62, (1,), device=device, dtype=torch.int64)
    index = device.index if device.index is not None else torch.cuda.current_device()
    gen = torch.cuda.default_generators[index]
    offset = gen.get_offset()
    gen.set_offset(offset + 4)
    seed = _splitmix64(gen.initial_seed() ^ _splitmix64(offset)) >> 34
    # Triton compiles an int argument equal to 1 as a constexpr, which has no
    # dtype for the kernel's scalar-vs-pointer check.
    return 2 if seed == 1 else seed


def _stochastic_rounding_params(
    config: QuantConfig, device: torch.device, max_pos: float
) -> Tuple["Optional[torch.Tensor | int]", float, float]:
    """``(seed, max_normal, min_normal)`` for a Triton QDQ kernel launch.

    With ``ElemRoundMode.STOCHASTIC`` the seed comes from ``_triton_sr_seed`` and
    the normal range is the element format's, as in
    ``quant_primitives.stochastic_rounding``. An eager call gets a fresh int seed
    drawn from the device's CUDA generator, which the draw advances, so repeated
    calls round with fresh random bits and ``torch.manual_seed`` reproduces them.
    Under CUDA graph capture the seed is a one-element int64 tensor drawn on the
    device, so every replay of the graph gets a new seed. Otherwise the seed is
    ``None``, which the kernel never reads, and the range is ``(max_pos, 0)``.
    """
    if config.elem_round_mode != ElemRoundMode.STOCHASTIC:
        return None, max_pos, 0
    seed = _triton_sr_seed(device)
    ebits, mbits = ELEM_DTYPE_BITS_MAP[config.elem_dtype]
    exp_bias = _n_ones_host(ebits - 1)
    max_normal = 2 ** (_n_ones_host(ebits) - exp_bias) * (
        _n_ones_host(mbits + 1) / (2**mbits)
    )
    min_normal = 2 ** (1 - exp_bias)
    return seed, max_normal, min_normal


def _exceeds_triton_block_numel(config: QuantConfig) -> bool:
    """Whether this block size overflows Triton's per-tensor numel cap.

    A Triton launch tiles up to TRITON_MEGA_BLOCK_M quant blocks of m * n elements
    each, so the ceiling is m * n <= 8192. It is a property of the config alone: no
    input shape makes an oversized block compile.
    """
    return (
        config.block_size.m * config.block_size.n
        > _TRITON_MAX_BLOCK_NUMEL_PER_QUANT_BLOCK
    )
