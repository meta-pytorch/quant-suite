# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file, including `_safe_lshift`, `_safe_rshift` and the
# exponent, scaling and rounding steps in `fp32_to_fpx`, are adapted from
# mx/elemwise_ops.py in Microsoft microxcaling
# (https://github.com/microsoft/microxcaling), licensed under the MIT License.
# Copyright (c) Microsoft Corporation.
# See THIRD_PARTY_NOTICES.md in the root directory of this package.

"""Numeric and reshape primitives shared by both QDQ backends.

Element-format and scale conversions, block reshaping, and the Triton tile limits.
Depends only on the configuration layer, so it sits below the backends and the
dispatch layer and can be imported on its own.
"""

import math
from typing import Optional, Tuple

import torch
import torch.fx
from quant_suite_core.mx_packing import _e8m0_value
from quant_suite_core.quant_config import (
    BlockSize,
    EBITS_F32,
    ELEM_DTYPE_BITS_MAP,
    ElemDtype,
    F32_MAX,
    F32_MIN_NORMAL,
    GlobalScaleGranularity,
    MBITS_F32,
    SBITS,
    ScaleRoundMode,
)


# Upper bound on the quant blocks per Triton program instance in the mega kernel. Its
# tile is BLOCK_M x (m * n) elements with BLOCK_M <= TRITON_MEGA_BLOCK_M -- a
# compile-time constexpr, independent of the input shape -- so Triton's per-tensor
# numel cap puts the ceiling at
# m * n <= TRITON_MAX_BLOCK_NUMEL / TRITON_MEGA_BLOCK_M = 8192.
TRITON_MEGA_BLOCK_M: int = 128
# The mega-kernel tile size BLOCK_M aims for: 16 elements per thread at the default
# 4 warps. Registers grow with elements per thread, and past 16 the kernel crosses
# occupancy cliffs (1x32) or spills (16x16).
TRITON_MEGA_TILE_NUMEL: int = 2048
TRITON_MAX_BLOCK_NUMEL: int = 2**20

# The public contract: BLOCK_M is our tuning knob and TRITON_MAX_BLOCK_NUMEL is the
# compiler's cap, but a caller only needs their quotient -- the per-quant-block
# ceiling m * n must satisfy. Exporting the quotient instead of the two factors lets
# BLOCK_M be retuned without that being a public API break.
_TRITON_MAX_BLOCK_NUMEL_PER_QUANT_BLOCK: int = (
    TRITON_MAX_BLOCK_NUMEL // TRITON_MEGA_BLOCK_M
)  # == 8192


# *****************************************************************************
# * Helper Functions                                                          *
# *****************************************************************************
def _n_ones(n: int) -> int:
    return (1 << n) - 1


def _safe_lshift(x, bits, exp):
    if exp is None:
        return x * (2**bits)
    else:
        return x / (2**exp) * (2**bits)


def _safe_rshift(x, bits, exp):
    if exp is None:
        return x / (2**bits)
    else:
        return x / (2**bits) * (2**exp)


def _is_fx_proxy(x) -> bool:
    """Whether ``x`` is a torch.fx Proxy (symbolic tracing).

    Branching on a Proxy's shape or dtype raises a TraceError, so the shape and
    dtype checks below take a trace-safe form only for a Proxy. Eager tensors,
    and the FakeTensors of torch.export, keep the original code path.
    """
    return isinstance(x, torch.fx.Proxy)


def _bitcast(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``x.view(dtype)``, in a form graph tracers keep as a bitcast.

    Some op-normalizing tracers rewrite every traced ``Tensor.view(...)`` into a
    reshape, which then fails shape propagation. The ``aten.view.dtype``
    overload is the same op and is left alone.
    """
    if _is_fx_proxy(x):
        return torch.ops.aten.view.dtype(x, dtype)
    return x.view(dtype)


# ******************************************************************************
# * Input Reshaping Functions                                                  *
# ******************************************************************************
def _reshape_input(x, m, n):
    """Reshape 2D tensor (H, W) into blocks of size (m, n), returning shape (-1, m*n).

    When m == 1 (1D block quantization), uses a simple reshape for efficiency.
    When m > 1 (2D blocks), uses unfold to gather non-contiguous rows.
    """
    if m == 1:
        return x.reshape(-1, n)
    blocks = x.unfold(0, m, m).unfold(1, n, n)
    return blocks.permute(0, 1, 2, 3).contiguous().view(-1, m * n)


def _reshape_2d_tensor(tensor: torch.Tensor):
    """Flatten tensor to 2D by merging all dims except the last into rows.

    []->[1,1], [N]->[1,N], [M,N]->[M,N], [*,N]->[-1,N]

    Raises ValueError for a zero-size last dim, whatever the rank.
    """
    if _is_fx_proxy(tensor):
        # Rank is unknown while tracing; this gives the same shape for every rank.
        # A zero-size last dim cannot be checked on a Proxy; reshape(-1, 0) raises
        # for it when the graph runs, which is intended.
        tensor = torch.atleast_1d(tensor)
        return tensor.reshape(-1, tensor.shape[-1])
    if tensor.ndim >= 1 and tensor.shape[-1] == 0:
        raise ValueError(
            f"cannot quantize a tensor with a zero-size last dim, got shape "
            f"{tuple(tensor.shape)}"
        )
    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    elif tensor.ndim == 1:
        return tensor.unsqueeze(0)
    elif tensor.ndim == 2:
        return tensor
    else:
        return tensor.reshape(-1, tensor.shape[-1])


def _reshape_input_back(x, m, n, org_shape):
    """Inverse of _reshape_input: restore (-1, m*n) block layout back to (H, W).

    When m == 1, this is a simple reshape. When m > 1, blocks are interleaved
    across rows and must be un-permuted.
    """
    if m == 1:
        return x.reshape(org_shape)
    H, W = org_shape
    blocks = x.view(H // m, W // n, m, n)
    return blocks.permute(0, 2, 1, 3).contiguous().view(H, W)


def _prepare_2d_padded(tensor, m, n, transpose):
    """Flatten to 2D, optionally transpose, and zero-pad to block-aligned shape.

    Inverse: _restore_2d_padded(x, M, N, transpose, orig_shape)
    """
    orig_shape = tensor.shape
    orig_dtype = tensor.dtype
    tensor = _reshape_2d_tensor(tensor)
    if transpose:
        tensor = tensor.T.contiguous()
    M, N = tensor.shape
    if _is_fx_proxy(tensor):
        # Divisibility is unknown while tracing, so always pad (by 0 if aligned).
        # Kept out of eager, where it would copy an already aligned tensor.
        padded = torch.nn.functional.pad(tensor, (0, (-N) % n, 0, (-M) % m))
    elif M % m != 0 or N % n != 0:
        padded = torch.zeros(
            ((M + m - 1) // m) * m,
            ((N + n - 1) // n) * n,
            device=tensor.device,
            dtype=orig_dtype,
        )
        padded[:M, :N] = tensor
    else:
        padded = tensor
    return padded, M, N, orig_shape, orig_dtype


def _restore_2d_padded(
    x: torch.Tensor, M: int, N: int, transpose: bool, orig_shape: torch.Size
) -> torch.Tensor:
    """Crop padded 2D tensor, transpose back if needed, and reshape to original shape."""
    x = x[:M, :N]
    if transpose:
        x = x.T.contiguous()
    return x.reshape(orig_shape)


def _block_padding_mask(padded, M, N, m, n) -> Optional[torch.Tensor]:
    """Which elements of ``_reshape_input(padded, m, n)`` are padding: those at
    row >= M or column >= N of ``padded``, the zero-padded 2-D tensor whose
    unpadded shape is (M, N).

    None when nothing is padded. While tracing the shape is unknown, so a mask
    is always built (all False when nothing is padded).
    """
    if not _is_fx_proxy(padded) and padded.shape[0] == M and padded.shape[1] == N:
        return None
    is_pad = torch.nn.functional.pad(
        torch.zeros_like(padded[:M, :N], dtype=torch.bool),
        (0, padded.shape[1] - N, 0, padded.shape[0] - M),
        value=True,
    )
    return _reshape_input(is_pad, m, n)


def _div_rn(x: torch.Tensor, divisor: float) -> torch.Tensor:
    """``x / divisor`` rounded once, on CPU and CUDA alike.

    CUDA divides a tensor by a Python number as ``x * (1 / divisor)``, which
    rounds twice; dividing by a tensor holding ``divisor`` does not. A power of
    two divides exactly either way.
    """
    if math.frexp(divisor)[0] == 0.5:
        return x / divisor
    # new_full is a device-side fill, legal under CUDA-graph capture.
    return x / x.new_full((), divisor)


def _chunked_abs_amax(tensor, block_size: BlockSize):
    """
    Get abs.amax for a tensor every block_size.m rows.

    Args:
        tensor: Input tensor of shape (H, W)
        block_size: Block size

    Returns:
        Tensor with abs.amax values for each chunk
    """

    _, W = tensor.shape
    reshaped = _reshape_input(tensor, block_size.m, W)

    # keepdim spelled out: op-normalizing tracers need it to normalize a traced
    # Tensor.amax call.
    return reshaped.abs().amax(dim=-1, keepdim=False)  # Shape: (num_chunks,)


def _mean_shift_stats(blocks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ASYMM_MEAN_SHIFT's per-block (fp32 zero point, amax of the centred
    values) of fp32 ``blocks`` (num_blocks, block width), as the fused kernel
    computes them.

    The mean goes through fixed-point ints so that reduction order cannot change
    it. Inf and NaN, as in the kernel: both maxima skip a NaN, a NaN counts as
    zero in the fixed-point sum (the divisor stays the block width), and an Inf
    gives int_scale = 0, so the mean is 0 / 0 = NaN, the centred amax 0, and
    the whole block dequantizes to NaN.
    """
    # new_full, not torch.tensor(scalar, device=...): the latter builds the
    # value on the host and H2D-copies it, which raises "Cannot copy between
    # CPU and CUDA tensors during CUDA graph capture" when a caller captures
    # this QDQ in a CUDA graph. new_full is a device-side fill with the same
    # dtype and value.
    max_abs = torch.maximum(
        torch.amax(
            torch.nan_to_num(torch.abs(blocks), nan=0.0, posinf=float("inf")), 1
        ),
        blocks.new_full((), 1.0e-12, dtype=torch.float32),
    )
    int_scale = 2097152.0 / max_abs
    # NaN, and Inf * 0, truncate to 0 as in the kernel's float-to-int
    # conversion; torch's conversion of a NaN is platform-dependent.
    data_int = torch.nan_to_num(blocks * int_scale.unsqueeze(1), nan=0.0).to(
        torch.int32
    )
    zero_point = data_int.sum(dim=1).to(torch.float32) / (int_scale * blocks.shape[1])
    delta = torch.amax(
        torch.nan_to_num(
            torch.abs(blocks - zero_point.unsqueeze(1)), nan=0.0, posinf=float("inf")
        ),
        1,
    )
    return zero_point, delta


# ******************************************************************************
# * Scale Quantization Functions                                               *
# ******************************************************************************
def fp32_to_e8m0(
    tensor: torch.Tensor,
    elem_dtype: ElemDtype,
    scale_round_mode: ScaleRoundMode,
    norm_exp: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Round per-block fp32 scale values to E8M0 (powers of two).

    Args:
        tensor: Per-block values to convert, typically `amax / max_pos`. When
            `norm_exp != 0` it is the raw block amax instead (see `norm_exp`).
        elem_dtype: Element format of the quantized data. Its mantissa width sets
            the round-up threshold of the `OAS` / `OAS_S` modes.
        scale_round_mode: How the fp32 value is rounded to a power of two
            (`OAS`, `OAS_S`, `CEIL`, `FLOOR` or `NV_ROUND`).
        norm_exp: For the `pow2max` scale-norm mode, `log2(max_pow2)` as returned by
            `pow2max_norm_exp`. `tensor` is then rounded as amax and divided by
            `2**norm_exp` as an integer exponent shift, which keeps the rounding
            decision on a normalized mantissa even when `amax / max_pow2` would be
            fp32-subnormal. 0 (the default) applies no shift.

    Returns:
        `(scale, unscale)`, both shaped like `tensor`. `unscale` is the E8M0 value
        itself, carrying the sign of `tensor` (zero counts as positive); dequantize
        by multiplying by it. `scale` is `1 / unscale`; quantize by multiplying by
        it. Blocks whose `unscale` would be zero, inf or nan (or, with
        `norm_exp != 0`, whose input is inf or nan) get `scale = unscale = 1.0`.
    """
    _, mbits = ELEM_DTYPE_BITS_MAP[elem_dtype]

    # Treat +0/-0 as positive (amax is always >= 0). torch.sign(0) == 0 would
    # make e8m0_values 0 for a zero block, tripping the invalid-scale fallback
    # to 1.0 (byte 127); a zero block instead emits byte 0 (2^-127), matching
    # the Triton path's tl.where. A zero block dequants to 0 regardless, so this
    # only aligns the stored scale byte.
    sign = torch.where(tensor < 0, -1.0, 1.0)
    abs_tensor = torch.abs(tensor)

    if scale_round_mode == ScaleRoundMode.OAS:
        val_to_add = (1 << (MBITS_F32 - mbits - 1)) - 1
    elif scale_round_mode == ScaleRoundMode.OAS_S:
        val_to_add = (3 << (MBITS_F32 - mbits - 1)) - 1
    elif scale_round_mode == ScaleRoundMode.CEIL:
        val_to_add = (1 << MBITS_F32) - 1
    else:
        assert scale_round_mode in (ScaleRoundMode.FLOOR, ScaleRoundMode.NV_ROUND)
        val_to_add = 0

    mask_exponent = ((1 << (EBITS_F32 + SBITS)) - 1) << MBITS_F32
    mask_mantissa = (1 << MBITS_F32) - 1

    fp32_bits = _bitcast(abs_tensor, torch.int32)
    fp32_bits_exp = (fp32_bits + val_to_add) & mask_exponent
    exponent = (fp32_bits_exp >> MBITS_F32) & 0xFF

    if scale_round_mode == ScaleRoundMode.NV_ROUND:
        mantissa = fp32_bits & mask_mantissa
        is_denormal = (exponent == 0) & (mantissa != 0)
        is_normal = ~is_denormal
        condition1 = is_normal & (exponent < 254) & (mantissa > 0)
        condition2 = is_denormal & (mantissa / (2**MBITS_F32) > 0.5)

        exponent = torch.where(condition1 | condition2, exponent + 1, exponent)

    if norm_exp != 0:
        # POW2MAX: amax was rounded to a power of two above; now divide by
        # max_pow2 = 2**norm_exp via an integer exponent shift. clamp(min=1)
        # pins a subnormal amax's exponent to the min-normal value (1 - bias);
        # clamp(min=0) is the E8M0 floor.
        exponent = torch.clamp(torch.clamp(exponent, min=1) - norm_exp, min=0)

    e8m0_values = sign * _e8m0_value(exponent)
    unscale = e8m0_values
    # In case unscale=0 (scale will be inf), or unscale=inf or nan, we set the scale to 1.0
    unscale_invalid_mask = (unscale == 0) | torch.isinf(unscale) | torch.isnan(unscale)
    if norm_exp != 0:
        # `tensor` is amax (not amax/max_pow2); the exponent shift can map an inf
        # amax to a finite e8m0, so guard inf/nan amax on the input explicitly.
        unscale_invalid_mask = (
            unscale_invalid_mask | torch.isinf(tensor) | torch.isnan(tensor)
        )
    unscale = torch.where(unscale_invalid_mask, 1.0, unscale)
    scale = 1.0 / unscale

    return scale, unscale


def fp32_floor_fp8(tensor: torch.Tensor) -> torch.Tensor:
    """Round each fp32 value's magnitude down to the E4M3 grid, keeping its sign.

    Subnormal magnitudes round on the E4M3 subnormal grid. Values are not
    clamped to the E4M3 maximum; the caller saturates.
    """
    ebits, mbits = 4, 3

    x = tensor
    x = x.view(torch.int32)
    sign = x & 0x80000000
    x = x ^ sign
    x = x.view(torch.float)

    exp_bias = _n_ones(ebits - 1)
    max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2**mbits))
    min_normal = 2 ** (1 - exp_bias)

    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), x < min_normal)

    # generate a shifting tensor for subnormal values
    shifting_tensor = torch.zeros_like(x, device=x.device)
    shifting_tensor = torch.where(denormal_mask, min_normal, shifting_tensor)

    # shift the subnormal values to normal range
    x += shifting_tensor

    # view x as int32
    x = x.view(torch.int32)

    # zero out the last (23 - mbits) bits
    ZERO_OUT_MASK = 0xFFF00000
    x = x & ZERO_OUT_MASK

    x = x.view(torch.float)

    # shift the subnormal values back
    x -= shifting_tensor

    # add sign back
    x = x.view(torch.int32)
    x = x | sign
    x = x.view(torch.float)

    return x


def fp32_ceil_fp8(tensor: torch.Tensor) -> torch.Tensor:
    """Round each fp32 value's magnitude up to the E4M3 grid, keeping its sign.

    Subnormal magnitudes round on the E4M3 subnormal grid. Values are not
    clamped to the E4M3 maximum; the caller saturates.
    """
    ebits, mbits = 4, 3

    x = tensor
    x = x.view(torch.int32)
    sign = x & 0x80000000
    x = x ^ sign
    x = x.view(torch.float)

    exp_bias = _n_ones(ebits - 1)
    max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2**mbits))
    min_normal = 2 ** (1 - exp_bias)

    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), x < min_normal)

    # generate a shifting tensor for subnormal values
    shifting_tensor = torch.zeros_like(x, device=x.device)
    shifting_tensor = torch.where(denormal_mask, min_normal, shifting_tensor)

    # shift the subnormal values to normal range
    x += shifting_tensor

    # view x as int32
    x = x.view(torch.int32)

    # zero out the leading (1 + 8 + mbits) bits
    mask = (1 << (MBITS_F32 - mbits)) - 1
    # add the (0x000FFFFF) bits for round up, last 20 bits are 1s
    x += 0x000FFFFF

    # take the leading (1 + 8 + mbits) bits
    x = x & (~mask)
    x = x.view(torch.float)

    # shift the subnormal values back
    x -= shifting_tensor

    # add sign back
    x = x.view(torch.int32)
    x = x | sign
    x = x.view(torch.float)

    return x


def nvfp4_global_scale(global_amax: torch.Tensor, max_pos: float) -> torch.Tensor:
    """NVFP4's global (second-level) scale ``448 * max_pos / global_amax``, in
    fp64.

    448 is the largest E4M3 value, so the scale maps the global amax onto the
    product of the two formats' maxima (2688 for E2M1). ``max_pos / global_amax``
    is divided in fp32 and the quotient multiplied by 448 in fp64.

    A positive amax whose scale exceeds FLT_MAX (below about
    ``448 * max_pos / FLT_MAX``) gets FLT_MAX, the largest scale an fp32 global
    scale can hold. A zero amax gives +Inf, an infinite amax 0 and a NaN amax
    NaN; what such a scale means is up to the caller.

    Args:
        global_amax: The tensorwise or rowwise amax.
        max_pos: The element format's largest value.

    Returns:
        The fp64 scale, shape ``(1,)`` for a 0-d ``global_amax`` and the shape
        of ``global_amax`` otherwise.
    """
    # new_full is a device-side fill, legal under CUDA-graph capture, and keeps the
    # exact f32/f32 division. A python-scalar numerator
    # (max_pos / global_amax) would round differently.
    scale = 448.0 * (
        global_amax.new_full((1,), max_pos, dtype=torch.float32) / global_amax
    ).to(torch.float64)
    return torch.where(global_amax > 0, scale.clamp(max=F32_MAX), scale)


def _nvfp4_global_scale_or_one(
    global_amax: torch.Tensor, two_level_scale: torch.Tensor
) -> torch.Tensor:
    """``two_level_scale``, or 1.0 where ``global_amax`` is zero.

    A zero amax belongs to an all-zero tensor (or row), which has no global
    scale; 1.0 makes its block scales 0. ``qdq`` and the fused kernel report,
    and build their block scales from, this value.
    """
    return torch.where(global_amax == 0, 1.0, two_level_scale)


def calculate_fp8_scale_for_nvfp4(
    global_amax: torch.Tensor,
    local_amax: torch.Tensor,
    max_pos: float,
    global_scale_granularity: GlobalScaleGranularity,
    num_blocks_row: int,
    num_blocks_col: int,
    scale_round_mode: ScaleRoundMode,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive NVFP4's two-level scale: FP8 E4M3 block scales under an fp64 global.

    Args:
        global_amax: The global amax, shape ``(1,)`` for ``TENSORWISE`` or
            ``(num_blocks_row,)`` for ``ROWWISE``.
        local_amax: The per-block amax, shape ``(num_blocks,)``.
        max_pos: The element format's largest value.
        global_scale_granularity: ``TENSORWISE`` or ``ROWWISE``.
        num_blocks_row: Block rows, used to lay out ``ROWWISE`` blocks.
        num_blocks_col: Block columns, used to lay out ``ROWWISE`` blocks.
        scale_round_mode: ``NONE`` (round to nearest), ``FLOOR`` or ``CEIL``,
            applied when the block scale is cast to E4M3.

    Returns:
        ``(scale, unscale, local_unscale_fp8, two_level_scale)``.
        ``two_level_scale`` is `nvfp4_global_scale` (fp64), or 1.0 where
        ``global_amax`` is zero (as `_nvfp4_global_scale_or_one`), one per
        tensor or per row. ``local_unscale_fp8`` is the E4M3 block scale,
        ``local_amax / max_pos * two_level_scale``. ``unscale`` is the fp32
        dequantization factor ``local_unscale_fp8 / two_level_scale`` and
        ``scale`` its reciprocal, both shaped like ``local_amax``. A block whose
        local or global amax is zero, inf or NaN gets ``unscale = 1.0``.
    """
    # check https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-for-fp8-and-fp4-data-types
    local_amax_shape = local_amax.shape
    local_amax_invalid = (
        (local_amax == 0) | torch.isinf(local_amax) | torch.isnan(local_amax)
    )
    global_amax_invalid = (
        (global_amax == 0) | torch.isinf(global_amax) | torch.isnan(global_amax)
    )

    elem_format_max = max_pos

    # shape=(H*W//(block_size.m*block_size.n),)
    local_unscale = (local_amax / elem_format_max).to(torch.float32)
    # shape=(H//block_size.m,)
    two_level_scale = _nvfp4_global_scale_or_one(
        global_amax, nvfp4_global_scale(global_amax, elem_format_max)
    )

    if global_scale_granularity == GlobalScaleGranularity.ROWWISE:
        # shape=(H//block_size.m, N//block_size.n)
        local_unscale = local_unscale.view(num_blocks_row, num_blocks_col)
        two_level_scale = two_level_scale.view(-1, 1)
        global_amax_invalid = global_amax_invalid.view(-1, 1)

    local_unscale_before_fp8 = (local_unscale * two_level_scale).to(torch.float32)
    if scale_round_mode == ScaleRoundMode.FLOOR:
        local_unscale_before_fp8 = fp32_floor_fp8(local_unscale_before_fp8)
    elif scale_round_mode == ScaleRoundMode.CEIL:
        # round up
        local_unscale_before_fp8 = fp32_ceil_fp8(local_unscale_before_fp8)
        local_unscale_before_fp8 = torch.clamp(
            local_unscale_before_fp8, min=-448.0, max=448.0
        )

    local_unscale_fp8 = local_unscale_before_fp8.to(torch.float8_e4m3fn)

    local_unscale_q = local_unscale_fp8.to(torch.float64) / two_level_scale

    if global_scale_granularity == GlobalScaleGranularity.ROWWISE:
        local_unscale_q = local_unscale_q.view(local_amax_shape)
        amax_invalid = (
            global_amax_invalid
            | local_amax_invalid.view(num_blocks_row, num_blocks_col)
        ).view(local_amax_shape)
        amax_invalid = amax_invalid.view(local_amax_shape)
    else:
        amax_invalid = local_amax_invalid | global_amax_invalid

    local_unscale_q = torch.where(amax_invalid, 1.0, local_unscale_q)

    scale = (1.0 / (local_unscale_q + F32_MIN_NORMAL)).to(torch.float32)
    unscale = (local_unscale_q).to(torch.float32)

    return scale, unscale, local_unscale_fp8, two_level_scale


# ******************************************************************************
# * Data Element Quantization Function                                        *
# ******************************************************************************
def fp32_to_fpx(
    tensor: torch.Tensor, elem_dtype: ElemDtype, max_pos: float
) -> torch.Tensor:
    """Round scaled fp32 values onto ``elem_dtype``'s grid.

    Rounds to nearest, ties to even, with ``elem_dtype``'s exponent and mantissa
    widths and its subnormal range, then saturates to ``[-max_pos, max_pos]``.
    The input must be fp32 and already divided by the block scale. The result
    stays in fp32.
    """
    assert _is_fx_proxy(tensor) or tensor.dtype == torch.float32

    A = tensor
    out = A

    ebits, mbits = ELEM_DTYPE_BITS_MAP[elem_dtype]

    private_exp = torch.floor(torch.log2(torch.abs(A) + (A == 0).to(torch.float32)))

    # Minimum normal exponent for `ebits` exponent bits.
    min_exp = -(2 ** (ebits - 1)) + 2
    # clamp, not its alias clip: op-normalizing tracers handle a traced clip
    # only when both min and max are given.
    private_exp = private_exp.clamp(min=min_exp)

    # Scale up so appropriate number of bits are in the integer portion of the number
    out = _safe_lshift(out, mbits, private_exp)

    # Round mantissa bits to round to nearest even
    absA = torch.abs(out)
    # find 0.5, 2.5, 4.5 ...
    maskA = ((absA - 0.5) % 2 == torch.zeros_like(out)).to(torch.float32)
    # copysign (not torch.sign) so -0.0 stays -0.0: torch.sign(-0.0) == 0 would
    # flush negative zero to +0.0, but -0 is a distinct element code. copysign matches torch.sign for every nonzero value.
    # NOTE: compute (absA + 0.5) in float64. In float32 a value ~1 ULP below a
    # tie (e.g. absA = 0.49999997) has absA + 0.5 round up to 1.0, so floor
    # returns 1 and a below-midpoint element is pushed up to the next grid step
    # instead of rounding to nearest.
    out = torch.copysign(torch.ones_like(out), out) * (
        torch.floor(absA.to(torch.float64) + 0.5).to(torch.float32) - maskA
    )

    # Undo scaling
    out = _safe_rshift(out, mbits, private_exp)

    # Handle saturation
    out = torch.clamp(out, min=-max_pos, max=max_pos)

    return out


def fp32_to_e8mx(tensor: torch.Tensor, elem_dtype: ElemDtype) -> torch.Tensor:
    """Round fp32 values to E8M1, E8M2 or E8M3, keeping fp32's 8-bit exponent.

    The mantissa rounds to nearest, ties to even, to 1, 2 or 3 bits; a carry
    into the exponent at the largest finite exponent saturates instead. The
    input must be fp32 and the result stays in fp32.
    """
    assert _is_fx_proxy(tensor) or tensor.dtype == torch.float32

    A = tensor

    ebits, mbits = ELEM_DTYPE_BITS_MAP[elem_dtype]
    assert ebits == 8

    # because shift of uint32 is not supported, we use int32
    int32_A = _bitcast(A, torch.int32)
    sign_A = (int32_A >> 31) & 0x1
    exponent_A = (int32_A >> 23) & 0xFF
    mantissa_A = int32_A & 0x7FFFFF

    # round-to-nearest-even
    if mbits == 1:
        # E8M1
        mask_0 = mantissa_A <= (32 << 16)  # <= 100000, mantissa_A is 0
        mask_1 = (~mask_0) & (mantissa_A <= (95 << 16))  # <= 1011111, mantissa_A is 1
        mask_2 = (~mask_0) & (~mask_1)  # exp need round up, mantissa_A is 0
        mantissa_A = torch.where(mask_0 | mask_2, 0, mantissa_A)
        mantissa_A = torch.where(mask_1, (1 << 22), mantissa_A)
        exponent_A = torch.where(mask_2, exponent_A + 1, exponent_A)
        mask_overflow = (exponent_A == 255) & mask_2
        exponent_A = torch.where(mask_overflow, exponent_A - 1, exponent_A)
        mantissa_A = torch.where(mask_overflow, (1 << 22), mantissa_A)

    elif mbits == 2:
        # E8M2
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
        mantissa_A = torch.where(mask_0 | mask_4, 0, mantissa_A)
        mantissa_A = torch.where(mask_1, (1 << 21), mantissa_A)
        mantissa_A = torch.where(mask_2, (2 << 21), mantissa_A)
        mantissa_A = torch.where(mask_3, (3 << 21), mantissa_A)
        exponent_A = torch.where(mask_4, exponent_A + 1, exponent_A)
        mask_overflow = (exponent_A == 255) & mask_4
        exponent_A = torch.where(mask_overflow, exponent_A - 1, exponent_A)
        mantissa_A = torch.where(mask_overflow, (3 << 21), mantissa_A)

    elif mbits == 3:
        # E8M3
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
        mask_6 = (
            (~mask_0)
            & (~mask_1)
            & (~mask_2)
            & (~mask_3)
            & (~mask_4)
            & (~mask_5)
            & (mantissa_A <= (104 << 16))
        )  # <= 1101000, mantissa_A is 6
        mask_7 = (
            (~mask_0)
            & (~mask_1)
            & (~mask_2)
            & (~mask_3)
            & (~mask_4)
            & (~mask_5)
            & (~mask_6)
            & (mantissa_A <= (119 << 16))
        )  # <= 1110111, mantissa_A is 7
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
        mantissa_A = torch.where(mask_0 | mask_8, 0, mantissa_A)
        mantissa_A = torch.where(mask_1, (1 << 20), mantissa_A)
        mantissa_A = torch.where(mask_2, (2 << 20), mantissa_A)
        mantissa_A = torch.where(mask_3, (3 << 20), mantissa_A)
        mantissa_A = torch.where(mask_4, (4 << 20), mantissa_A)
        mantissa_A = torch.where(mask_5, (5 << 20), mantissa_A)
        mantissa_A = torch.where(mask_6, (6 << 20), mantissa_A)
        mantissa_A = torch.where(mask_7, (7 << 20), mantissa_A)
        exponent_A = torch.where(mask_8, exponent_A + 1, exponent_A)
        mask_overflow = (exponent_A == 255) & mask_8
        exponent_A = torch.where(mask_overflow, exponent_A - 1, exponent_A)
        mantissa_A = torch.where(mask_overflow, (7 << 20), mantissa_A)
    else:
        raise ValueError(f"Unsupported elem dtype {elem_dtype}")

    # torch.bitwise_or, not `|`: some torch.fx tracers map a traced `|` to
    # logical_or.
    bits = torch.bitwise_or(
        torch.bitwise_or(sign_A << 31, exponent_A << 23), mantissa_A
    )
    out = _bitcast(bits, torch.float32)

    return out


def stochastic_rounding(tensor: torch.Tensor, elem_dtype: ElemDtype) -> torch.Tensor:
    """Stochastically round fp32 values to ``elem_dtype``'s mantissa width.

    Adds uniform random bits below the kept mantissa bits, drawn from the global
    torch RNG, and truncates. Subnormal magnitudes round on the format's
    subnormal grid. Values are not saturated; the caller clamps.
    """
    ebits, mbits = ELEM_DTYPE_BITS_MAP[elem_dtype]

    x = tensor
    x = x.view(torch.int32)
    sign = x & 0x80000000
    x = x ^ sign
    x = x.view(torch.float)

    exp_bias = _n_ones(ebits - 1)
    max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2**mbits))
    min_normal = 2 ** (1 - exp_bias)

    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), x < min_normal)

    # generate a shifting tensor for subnormal values
    shifting_tensor = torch.zeros_like(x, device=x.device)
    shifting_tensor = torch.where(denormal_mask, min_normal, shifting_tensor)

    # shift the subnormal values to normal range
    x += shifting_tensor

    # view x as int32
    x = x.view(torch.int32)

    # generate a 32-bit integer
    rand_values = torch.randint(low=0, high=2**32 - 1, size=x.size(), device=x.device)

    # zero out the leading (1 + 8 + mbits) bits
    rand_mask = (1 << (MBITS_F32 - mbits)) - 1
    # add the random bits
    x += rand_values & rand_mask

    # take the leading (1 + 8 + mbits) bits
    x = x & (~rand_mask)
    x = x.view(torch.float)

    # shift the subnormal values back
    x -= shifting_tensor

    # add sign back
    x = x.view(torch.int32)
    x = x | sign
    x = x.view(torch.float)
    return x
