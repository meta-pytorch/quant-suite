# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The eager QDQ backend: ``qdq`` plus the scale-derivation and MBS rules that
only it uses.

``qdq`` is the numerics reference the Triton backend is validated against.

It runs in four phases -- pre-process, scaling factor, quantize, post-process --
marked by the banners in the body. The ends of the second and the fourth are
extension points: ``QuantConfig._derive_scale`` and ``QuantConfig._post_process``,
reached through the config object so an enabled ``QuantConfigExtension`` can
bind its own steps without a call-site change. ``_EagerScalePhase`` /
``_EagerPostProcess`` are the state those hooks receive.
"""

from dataclasses import dataclass, replace as dataclasses_replace
from typing import Literal, NamedTuple, Optional, overload, Tuple, Union

import torch
from quant_suite_core.extension import NVFP4AmaxRule
from quant_suite_core.quant_config import (
    ELEM_DTYPE_BITS_MAP,
    elem_dtype_max_pow2,
    ELEM_DTYPE_RANGE_MAP,
    ElemDtype,
    ElemRoundMode,
    F16_MIN_NORMAL,
    F16_MIN_SUBNORMAL,
    F32_MAX,
    F32_MIN_NORMAL,
    F32_MIN_SUBNORMAL,
    GlobalScaleGranularity,
    MBS_DYNAMIC_SLOTS,
    pow2max_norm_exp,
    QuantConfig,
    QuantInfo,
    QuantMode,
    ScaleDtype,
    ScaleNormMode,
    ScaleRoundMode,
)
from quant_suite_core.quant_primitives import (
    _bitcast,
    _block_padding_mask,
    _chunked_abs_amax,
    _div_rn,
    _is_fx_proxy,
    _mean_shift_stats,
    _prepare_2d_padded,
    _reshape_input,
    _reshape_input_back,
    _restore_2d_padded,
    calculate_fp8_scale_for_nvfp4,
    fp32_ceil_fp8,
    fp32_floor_fp8,
    fp32_to_e8m0,
    fp32_to_e8mx,
    fp32_to_fpx,
    stochastic_rounding,
)


_ScaleQuadruple = Tuple[
    torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
]


@dataclass
class _ScaleContext:
    """What a scale-derivation rule needs besides the normalized block scale.

    Attributes:
        config: The quantization configuration.
        delta: The per-block amax. POW2MAX forwards this undivided to E8M0.
        norm_exp: log2(max_pow2) under POW2MAX, 0 otherwise.
        fp8_local_amax: The per-block amax the scale is built from.
        fp8_max_pos: The magnitude that amax is placed on -- the element-format
            max, or 1.0 once a per-block magnitude has been folded into
            ``fp8_local_amax``.
        global_amax: The tensorwise or rowwise amax; only set for FP8 scales.
        padded_tensor_shape: Shape of the padded 2-D tensor, in elements.
    """

    config: QuantConfig
    delta: torch.Tensor
    norm_exp: int
    fp8_local_amax: torch.Tensor
    fp8_max_pos: float
    global_amax: Optional[torch.Tensor]
    padded_tensor_shape: torch.Size


def _derive_scale_fp32(unscale: torch.Tensor, ctx: _ScaleContext) -> _ScaleQuadruple:
    return (
        1.0 / unscale.to(torch.float32),
        unscale.to(torch.float32),
        None,
        None,
    )


def _derive_scale_fp16(unscale: torch.Tensor, ctx: _ScaleContext) -> _ScaleQuadruple:
    # The reciprocal in fp32: an fp16 one is coarser, and overflows below 2^-16.
    unscale_fp16 = unscale.to(torch.float16)
    return (
        1.0 / unscale_fp16.to(torch.float32),
        unscale_fp16,
        None,
        None,
    )


def _derive_scale_fp8(unscale: torch.Tensor, ctx: _ScaleContext) -> _ScaleQuadruple:
    if ctx.global_amax is None:
        # A single-level E4M3 block scale: NVFP4's block scale under a unit
        # global scale, which the fused kernel derives the same way.
        if ctx.config.scale_round_mode == ScaleRoundMode.FLOOR:
            unscale = fp32_floor_fp8(unscale)
        elif ctx.config.scale_round_mode == ScaleRoundMode.CEIL:
            unscale = fp32_ceil_fp8(unscale)
        unscale_fp8 = torch.clamp(unscale, min=-448.0, max=448.0).to(
            torch.float8_e4m3fn
        )
        local_amax = ctx.fp8_local_amax
        unscale_q = torch.where(
            (local_amax == 0) | ~torch.isfinite(local_amax),
            1.0,
            unscale_fp8.to(torch.float64),
        )
        return (
            (1.0 / (unscale_q + F32_MIN_NORMAL)).to(torch.float32),
            unscale_q.to(torch.float32),
            unscale_fp8,
            None,
        )
    return calculate_fp8_scale_for_nvfp4(
        global_amax=ctx.global_amax,
        local_amax=ctx.fp8_local_amax,
        max_pos=ctx.fp8_max_pos,
        global_scale_granularity=ctx.config.global_scale_granularity,
        num_blocks_row=ctx.padded_tensor_shape[0] // ctx.config.block_size.m,
        num_blocks_col=ctx.padded_tensor_shape[1] // ctx.config.block_size.n,
        scale_round_mode=ctx.config.scale_round_mode,
    )


def _derive_scale_e8m0(unscale: torch.Tensor, ctx: _ScaleContext) -> _ScaleQuadruple:
    return (
        *fp32_to_e8m0(
            ctx.delta if ctx.norm_exp != 0 else unscale,
            elem_dtype=ctx.config.elem_dtype,
            scale_round_mode=ctx.config.scale_round_mode,
            norm_exp=ctx.norm_exp,
        ),
        None,
        None,
    )


# The seam for scale derivation: scale dtype -> the rule turning the normalized
# block scale into (scale, unscale, local_unscale, global_scale).
_SCALE_DTYPE_QUANT_MAP = {
    ScaleDtype.FP32: _derive_scale_fp32,
    ScaleDtype.FP16: _derive_scale_fp16,
    ScaleDtype.FP8: _derive_scale_fp8,
    ScaleDtype.E8M0: _derive_scale_e8m0,
}


def _mbs_quant_block_peaks(rows: torch.Tensor, n: int) -> torch.Tensor:
    """The |x| peak of each 1 x ``n`` quant block along the rows of a 2-D
    tensor, NaN skipped (an Inf counts); shape (rows, width / n).

    max_pool1d rather than amax: a reduction over 16 contiguous elements runs
    at about a third of its bandwidth.
    """
    abs_finite_or_inf = torch.nan_to_num(torch.abs(rows), nan=0.0, posinf=float("inf"))
    return torch.nn.functional.max_pool1d(
        abs_finite_or_inf.unsqueeze(1), kernel_size=n, stride=n
    ).squeeze(1)


def _mbs_factor_from_amax(
    max_abs_per_mb: torch.Tensor, config: QuantConfig, fp_max: float
) -> torch.Tensor:
    """The static MBS factor of each macro block from its amax; see
    `_mbs_factor_static`."""
    max_abs_safe = torch.where(
        max_abs_per_mb == 0, torch.ones_like(max_abs_per_mb), max_abs_per_mb
    )
    sf = fp_max / max_abs_safe

    if config.global_scale_dtype == ScaleDtype.E0M8:
        # bits &= 0x007FFFFF (keep mantissa) | (127 << 23) (exponent 2^0) -> [1, 2)
        # torch.bitwise_or, not `|`: some torch.fx tracers map a traced `|` to
        # logical_or.
        sf_bits = _bitcast(sf, torch.int32)
        sf_bits = torch.bitwise_or(sf_bits & 0x007FFFFF, 127 << 23)
        mbs = _bitcast(sf_bits, torch.float32)
    elif config.global_scale_dtype == ScaleDtype.FP32:
        mbs = sf.to(torch.float32)
    else:
        raise NotImplementedError(
            f"Unsupported MBS global scale dtype {config.global_scale_dtype}; "
            "supported: e0m8, fp32"
        )

    # zero blocks get no scaling
    return torch.where(max_abs_per_mb == 0, torch.ones_like(mbs), mbs)


def _mbs_factor_static(
    mbs_reshaped: torch.Tensor, config: QuantConfig, fp_max: float
) -> torch.Tensor:
    """Derive the MBS macro factor from the macro-block amax (the default rule).

    ``sf = fp_max / amax``, then the mantissa is extracted so the factor lands in
    [1, 2) -- i.e. the block peak is placed on fp_max. Decided entirely by the
    single largest element of the block. A NaN is skipped: it poisons only its
    own quant block, and the rest of the macro block keeps the factor it would
    have without it. An Inf still sets the amax, which gives factor 1.0.

    Returns one factor per macro block, shape (M, num_mbs_per_row).
    """
    # posinf=inf: an Inf still sets the amax.
    abs_finite_or_inf = torch.nan_to_num(
        torch.abs(mbs_reshaped), nan=0.0, posinf=float("inf")
    )
    max_abs_per_mb = torch.amax(abs_finite_or_inf, dim=2, keepdim=False)
    return _mbs_factor_from_amax(max_abs_per_mb, config, fp_max)


def _mbs_headroom(quant_block_peaks: torch.Tensor) -> torch.Tensor:
    """The extra divisor of each quant block of an MBS macro block, from its
    peak (`_mbs_quant_block_peaks`).

    Scaled by a macro factor in [1, 2), a value of 2^126 or more can overflow
    fp32, either itself or through the grid point it rounds to (1.92 * 2^127
    rounds to the e2m1 code 4 * 2^126 = 2^128). A quant block whose finite peak
    is at least 2^126 is divided by ``4 / factor`` instead of ``1 / factor``:
    its scaled data stay finite, its E8M0 block scale comes out exactly 4x
    smaller, and dequant multiplies the 4 back. Where nothing overflowed the
    result is bitwise the same: the shift is exact, and an element whose quarter
    is subnormal sits more than 2^250 below its block peak and quantizes to zero
    either way. The cut is per quant block, not per macro block: a quant block
    with a tiny peak sits at the E8M0 floor, where a 4x smaller input would get
    a coarser grid. An Inf (clamped to FLT_MAX) is left alone.

    Returns 4 or 1 per quant block, the shape of ``quant_block_peaks``.
    """
    top = (quant_block_peaks >= 2.0**126) & (quant_block_peaks <= F32_MAX)
    ones = torch.ones_like(quant_block_peaks)
    return torch.where(top, ones * 4.0, ones)


def _mbs_factor_dynamic(
    mbs_reshaped: torch.Tensor,
    tensor_name: str,
    config: QuantConfig,
    quant_block_peaks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Choose the MBS macro factor per block by SSE search rather than from the amax.

    The static rule is largely redundant when the per-block scale is already an
    amax-aligned E8M0: both are pinned by the same outlier. This tries
    MBS_DYNAMIC_SLOTS factors spread over [1, 2) -- the same octave the static rule
    lands in, and equivalent up to a power of two because the E8M0 per-block scale
    absorbs any factor of 2 -- and keeps whichever minimises reconstruction SSE over
    the whole macro block.

    A new MBS rule should be added here, as a sibling of `_mbs_factor_static`, not
    as another `qdq` variant: everything around this step (padding, reshape, scale,
    quantize, unscale) is rule-independent and already shared.

    NOT FUSED: evaluating a candidate requires quantizing, so this re-enters ``qdq``
    once per slot with MBS disabled. It calls ``qdq`` directly because ``qdq`` is its
    only caller -- ``qdq_triton_mega`` runs the equivalent search inside the kernel
    and never reaches here. Cost is ~MBS_DYNAMIC_SLOTS x a plain QDQ, since the
    launches and HBM round-trips are per candidate.

    ``quant_block_peaks`` is `_mbs_quant_block_peaks` of the quant blocks,
    shape (M, num_mbs_per_row, MBS block // n), computed here if not given.

    Returns one factor per macro block, shape (M, num_mbs_per_row).
    """
    # Validate the global-scale dtype exactly as `_mbs_factor_static` does. Without
    # this the dynamic rule silently accepts dtypes the static rule rejects, and --
    # worse -- emits factors the declared format cannot represent.
    if config.global_scale_dtype not in (ScaleDtype.E0M8, ScaleDtype.FP32):
        raise NotImplementedError(
            f"Unsupported global scale dtype {config.global_scale_dtype} for "
            "mbs_dynamic; supported: e0m8, fp32"
        )
    # `qdq` is re-entered on `inner`, never on `config`: MBS off means the recursive
    # call cannot reach this search again.
    inner = dataclasses_replace(
        config,
        global_scale_granularity=GlobalScaleGranularity.NONE,
        mbs_block_size=None,
        global_scale_dtype=ScaleDtype.NONE,
        mbs_dynamic=False,
    )
    m, num_mbs, block = mbs_reshaped.shape
    # Each macro block is scored by re-entering the quantizer on its own rows, which
    # only models the real quantization when a quant block fits inside one macro
    # block. Otherwise a quant block straddles two macro blocks with different
    # factors and the SSE minimised here is not the error actually incurred.
    # `qdq_triton_mega` asserts the same precondition.
    if not (_is_fx_proxy(block) or block % config.block_size.n == 0):
        raise NotImplementedError(
            f"mbs_dynamic needs the MBS block ({block}) to be a multiple of the "
            f"quant block n ({config.block_size.n})"
        )
    ref = mbs_reshaped.reshape(m * num_mbs, block)
    n = config.block_size.n
    ref_blocks = ref.reshape(m * num_mbs, block // n, n)
    if quant_block_peaks is None:
        quant_block_peaks = _mbs_quant_block_peaks(ref, n)
    quant_block_peaks = quant_block_peaks.reshape(m * num_mbs, block // n)
    # Pre-divide by the headroom of `qdq` (exact), so no candidate overflows
    # fp32 on a huge finite peak; the 4 is multiplied back with the factor.
    headroom = _mbs_headroom(quant_block_peaks)
    ref_scaled = (ref_blocks / headroom.unsqueeze(2)).reshape(m * num_mbs, block)
    # A NaN or +-Inf input element would make every candidate's error
    # non-finite, so it is scored as zero: `qdq` masks/clamps it separately,
    # after the factor is chosen. As NaN it is zeroed with the NaN-poisoned
    # quant blocks, while a finite element whose reconstruction overflows
    # keeps its infinite error.
    ref_err = torch.where(torch.isinf(ref), float("nan"), ref)

    best_err: Optional[torch.Tensor] = None
    best_mbs: Optional[torch.Tensor] = None
    for i in range(MBS_DYNAMIC_SLOTS):
        cand = 1.0 + i / MBS_DYNAMIC_SLOTS
        # Apply the factor the same way `qdq` does -- divide by its reciprocal, then
        # multiply back -- so the objective measured here is the error actually
        # incurred. `x / (1/c)` and `x * c` differ in floating point.
        inv = 1.0 / cand
        rec = qdq(ref_scaled / inv, tensor_name, inner).to(torch.float32)
        rec = (
            rec.reshape(m * num_mbs, block // n, n) * (headroom * inv).unsqueeze(2)
        ).reshape(m * num_mbs, block)
        err = torch.nan_to_num(
            rec - ref_err, nan=0.0, posinf=float("inf"), neginf=float("-inf")
        )
        err = err.pow(2).sum(dim=1)
        if best_err is None:
            best_err = err
            best_mbs = torch.full_like(err, cand)
        else:
            take = err < best_err
            best_err = torch.where(take, err, best_err)
            best_mbs = torch.where(take, torch.full_like(err, cand), best_mbs)
        del rec, err
    assert best_mbs is not None
    # Limitation: if every candidate's SSE overflows fp32 (macro peak >= 2^57),
    # the block keeps c = 1.0 (plain MXFP4), where an input above 1.75 * 2^127
    # (about 2.98e38) can still dequantize to +-Inf.
    # zero blocks get no scaling, matching the static rule
    zero = torch.amax(torch.abs(ref), dim=1) == 0
    best_mbs = torch.where(zero, torch.ones_like(best_mbs), best_mbs)
    return best_mbs.reshape(m, num_mbs)


def _scale_elements(
    data: torch.Tensor, scale: torch.Tensor, unscale: torch.Tensor, config: QuantConfig
) -> torch.Tensor:
    """``data * scale`` for each block, ``scale`` the reciprocal of ``unscale``.

    The fp32 reciprocal of an FP32 scale below 2^-128 overflows, so outside
    ASYMM_MEAN_SHIFT a scale below FLT_MIN is inverted as ``2^64 * unscale``
    and the elements are multiplied by 2^64 too. Both are exact, so the codes
    are those of a reciprocal with an unbounded exponent; other blocks are
    unchanged.
    """
    if (
        config.scale_dtype != ScaleDtype.FP32
        or config.quant_mode == QuantMode.ASYMM_MEAN_SHIFT
    ):
        return data * scale.unsqueeze(1)
    pre = torch.where(unscale < F32_MIN_NORMAL, 2.0**64, 1.0)
    return (data * pre.unsqueeze(1)) * (1.0 / (unscale * pre)).unsqueeze(1)


class _EagerScale(NamedTuple):
    """What ``qdq`` takes out of the scaling-factor phase.

    ``quant_unscale`` is the unscale before the scale-dtype conversion; the
    ASYMM_MEAN_SHIFT FP16 path divides by it rather than by ``scale``.
    """

    quant_unscale: torch.Tensor
    scale: torch.Tensor
    unscale: torch.Tensor
    local_unscale: Optional[torch.Tensor]
    global_scale: Optional[torch.Tensor]


@dataclass
class _EagerScalePhase:
    """``qdq``'s scaling-factor phase, for ``QuantConfig._derive_scale``.

    The fields below the marker are what an amax rule substitutes; everything
    ``derive`` does with them is fixed.
    """

    config: QuantConfig
    data_hp: torch.Tensor
    delta: torch.Tensor
    padded_tensor: torch.Tensor
    padded_tensor_shape: torch.Size
    max_pos: float
    # --- substitutable by an amax rule ---
    # ``local_amax`` is the per-block amax the scale is built from and
    # ``local_max_pos`` the magnitude it is placed on. Substituting the pair
    # reaches both levels of the scale, not just the per-block one: a tensorwise
    # NVFP4 global amax is reduced from ``local_amax`` below, so it follows.
    # Unset means "not substituted" -- ``derive`` falls back to ``delta`` and
    # ``max_pos``, so the call site passes neither.
    local_amax: Optional[torch.Tensor] = None
    local_max_pos: Optional[float] = None
    # Set only by a rule. ``derive`` computes it when this is None, which is the
    # default path -- so a config without a rule is untouched by this field.
    global_amax: Optional[torch.Tensor] = None

    def with_amax_rule(self, rule: NVFP4AmaxRule) -> "_EagerScalePhase":
        """Take all three NVFP4 amax quantities from the rule."""
        local_amax, local_max_pos, global_amax = rule.amax(self.data_hp, self.max_pos)
        return dataclasses_replace(
            self,
            local_amax=local_amax,
            local_max_pos=local_max_pos,
            global_amax=global_amax,
        )

    def derive(self) -> _EagerScale:
        config = self.config
        delta = self.delta
        local_amax = self.local_amax if self.local_amax is not None else self.delta
        local_max_pos = (
            self.local_max_pos if self.local_max_pos is not None else self.max_pos
        )

        # Initialize global_amax (only used for NVFP4 tensorwise or rowwise scales)
        global_amax: Optional[torch.Tensor] = None
        if config.global_scale_granularity in (
            GlobalScaleGranularity.TENSORWISE,
            GlobalScaleGranularity.ROWWISE,
        ):
            global_amax = self.global_amax
            if global_amax is None:
                if config.global_scale_granularity == GlobalScaleGranularity.TENSORWISE:
                    global_amax = torch.amax(local_amax)
                else:
                    global_amax = _chunked_abs_amax(
                        self.padded_tensor, config.block_size
                    )

        # compute the scale
        # norm_exp != 0 only for POW2MAX: round amax to a power of two in fp32_to_e8m0
        # and fold the /max_pow2 in as an integer exponent shift, rather than
        # pre-dividing into a (possibly fp32-subnormal) amax/max_pow2.
        norm_exp = 0
        # Correctly rounded on every device, but under ASYMM_MEAN_SHIFT, which
        # uses plain `/` (not `_div_rn`): a multiply by the rounded reciprocal,
        # as torch on CUDA and Triton compute it.
        mean_shift = config.quant_mode == QuantMode.ASYMM_MEAN_SHIFT
        if config.scale_norm_mode == ScaleNormMode.POW2MAX:
            max_pow2 = elem_dtype_max_pow2(config.quant_mode, config.elem_dtype)
            norm_exp = pow2max_norm_exp(max_pow2)
            # Keep delta / max_pow2 here for the FP32/FP16/FP8 rules; only the E8M0
            # rule forwards delta + norm_exp (round-then-shift). For non-pow2
            # max_pow2 (E8Mx) norm_exp == 0 and the scale is the pre-divided value.
            unscale = delta / max_pow2 if mean_shift else _div_rn(delta, max_pow2)
        elif config.scale_norm_mode == ScaleNormMode.AMAX:
            unscale = (
                local_amax / local_max_pos
                if mean_shift
                else _div_rn(local_amax, local_max_pos)
            )
        else:
            raise ValueError(f"Unsupported scale norm mode {config.scale_norm_mode}")

        scale_ctx = _ScaleContext(
            config=config,
            delta=delta,
            norm_exp=norm_exp,
            fp8_local_amax=local_amax,
            fp8_max_pos=local_max_pos,
            global_amax=global_amax,
            padded_tensor_shape=self.padded_tensor_shape,
        )
        if config.scale_dtype == ScaleDtype.FP32:
            if config.quant_mode == QuantMode.ASYMM_MEAN_SHIFT:
                unscale = torch.maximum(
                    unscale,
                    # Device-side fill; see the ASYMM_MEAN_SHIFT note in `qdq` for
                    # why torch.tensor(scalar, device=...) breaks graph capture.
                    unscale.new_full((), 1.0e-12),
                )
            else:
                absmax_zero_mask = delta == 0
                eps = F32_MIN_NORMAL * (absmax_zero_mask).type(delta.dtype)
                unscale += eps
                # A nonzero block whose scale underflows to zero gets the
                # smallest positive one (see `_scale_elements` for its
                # reciprocal).
                unscale = torch.clamp(unscale, min=F32_MIN_SUBNORMAL)
        elif config.scale_dtype == ScaleDtype.FP16:
            if config.quant_mode == QuantMode.ASYMM_MEAN_SHIFT:
                unscale = torch.maximum(
                    unscale,
                    # Device-side fill; see the ASYMM_MEAN_SHIFT note in `qdq` for
                    # why torch.tensor(scalar, device=...) breaks graph capture.
                    unscale.new_full((), 1.0e-12),
                )
            else:
                absmax_zero_mask = delta == 0
                eps = F16_MIN_NORMAL * (absmax_zero_mask).type(delta.dtype)
                unscale += eps
                # Likewise for a scale that would round to an fp16 zero.
                unscale = torch.clamp(unscale, min=F16_MIN_SUBNORMAL)
        quant_unscale = unscale
        # convert the scale to the desired dtype
        scale, unscale, local_unscale, global_scale = _SCALE_DTYPE_QUANT_MAP[
            config.scale_dtype
        ](unscale, scale_ctx)
        return _EagerScale(
            quant_unscale=quant_unscale,
            scale=scale,
            unscale=unscale,
            local_unscale=local_unscale,
            global_scale=global_scale,
        )


@dataclass
class _EagerPostProcess:
    """``qdq``'s state at the end of post-process, for ``QuantConfig._post_process``.

    ``data`` is the quantized block view, still in the element domain, so a step
    that writes here picks up the shared unscale, MBS unscale and NaN-poisoning
    tail rather than applying them itself.
    """

    data: torch.Tensor
    data_hp: torch.Tensor
    scale: torch.Tensor
    min_pos: float
    max_pos: float

    def block_view(self) -> torch.Tensor:
        # The +-FLT_MAX-clamped block view when `is_mx_fp` (see `qdq`).
        return self.data_hp

    def block_scale(self) -> torch.Tensor:
        # The scale the block already quantized with; this backend has it.
        return self.scale

    def scatter_block_elements(
        self, index: torch.Tensor, value_lp: torch.Tensor
    ) -> torch.Tensor:
        data_lp = self.data
        num_blocks, block_size = self.data_hp.shape

        # Create a mask for the replaced positions and update data_lp
        mask = torch.zeros_like(data_lp, dtype=torch.bool)
        mask[torch.arange(num_blocks, device=data_lp.device), index] = True

        # The element domain is what this backend unwinds from, so the values go
        # in as they are and pick up the shared unscale/NaN-poisoning below.
        return torch.where(mask, value_lp.unsqueeze(1).expand(-1, block_size), data_lp)


@overload
def qdq(
    tensor: torch.Tensor,
    tensor_name: str,
    config: QuantConfig,
    transpose: bool = ...,
    return_quant_info: Literal[False] = ...,
) -> torch.Tensor: ...


@overload
def qdq(
    tensor: torch.Tensor,
    tensor_name: str,
    config: QuantConfig,
    transpose: bool,
    return_quant_info: Literal[True],
) -> QuantInfo: ...


@overload
def qdq(
    tensor: torch.Tensor,
    tensor_name: str,
    config: QuantConfig,
    transpose: bool = ...,
    *,
    return_quant_info: Literal[True],
) -> QuantInfo: ...


@overload
def qdq(
    tensor: torch.Tensor,
    tensor_name: str,
    config: QuantConfig,
    transpose: bool = ...,
    return_quant_info: bool = ...,
) -> Union[torch.Tensor, QuantInfo]: ...


def qdq(  # noqa: C901
    tensor: torch.Tensor,
    tensor_name: str,
    config: QuantConfig,
    transpose: bool = False,
    return_quant_info: bool = False,
) -> Union[torch.Tensor, QuantInfo]:
    """Quantize and dequantize a tensor with the eager reference backend.

    Plain PyTorch ops, so it runs on CPU as well as CUDA, and is the numerics
    reference that ``qdq_triton_mega`` is validated against.

    Args:
        tensor (Tensor): The input tensor to be quantized.
        tensor_name (str): Name of the tensor (for debugging/logging).
        config (QuantConfig): The quantization configuration.
        transpose (bool): Whether to transpose the block dimensions.
        return_quant_info (bool): If True, returns QuantInfo with quantized values
            and scales instead of just the dequantized tensor.

    Returns:
        Union[torch.Tensor, QuantInfo]: The dequantized tensor, or QuantInfo
            containing dequant_data, quantized_val, local_unscale, and global_scale.
    """
    # ── Phase 1: Pre-process -- pad, reshape, MBS ─────────────────────
    m = config.block_size.m
    n = config.block_size.n
    padded_tensor, M, N, orig_shape, orig_dtype = _prepare_2d_padded(
        tensor, m, n, transpose
    )
    padded_tensor_shape = padded_tensor.shape
    # reshape the input tensor
    data_hp = _reshape_input(padded_tensor, m, n)
    data_hp = data_hp.to(torch.float32)
    # Zero is not neutral for a min or a max, so ASYMM leaves the zero padding
    # of a partial quant block out of them.
    excluded: Optional[torch.Tensor] = None
    if config.quant_mode == QuantMode.ASYMM:
        excluded = _block_padding_mask(padded_tensor, M, N, m, n)

    # This clamps ±Inf to ±FLT_MAX before selecting the block's E8M0 scale,
    # so an Inf-containing block gets a real large scale byte and the Inf
    # element saturates to the max-power-of-two code (sign preserved). A
    # NaN anywhere in the block instead poisons the whole block (E8M0 NaN
    # scale byte 255), handled after dequant below. ASYMM_MEAN_SHIFT has its
    # own Inf/NaN rule (below), whatever its scale dtype.
    is_mx_fp = (
        config.scale_dtype == ScaleDtype.E8M0
        and config.elem_dtype in (ElemDtype.E2M1, ElemDtype.E2M3, ElemDtype.E4M3)
        and config.quant_mode != QuantMode.ASYMM_MEAN_SHIFT
    )
    mx_nan_mask: Optional[torch.Tensor] = None
    if is_mx_fp:
        if (
            config.global_scale_granularity.is_mbs()
            and not config.mbs_dynamic
            and config.global_scale_dtype != ScaleDtype.E0M8
        ):
            # A static FP32 MBS factor, fp_max / amax, overflows to Inf on a
            # macro block whose amax is below fp_max / FLT_MAX, and the zeros of
            # its quant blocks become 0 / 0 = NaN below: here a NaN amax does
            # not mean a NaN input, so the NaN blocks come from the input.
            mx_nan_mask = torch.isnan(data_hp)
        data_hp = torch.clamp(data_hp, min=-F32_MAX, max=F32_MAX)

    # MBS preserves mantissa bits of the largest value in every N elements.
    # Flow (when MBS enabled):
    # 1. Calculate MBS for every N elements on the 2D tensor
    # 2. Rescale all values with MBS (multiply by MBS to scale up)
    # 3. Quantize the scaled data
    # 4. Unscale with 1/MBS (multiply by 1/mbs to restore original scale)

    use_mbs = config.global_scale_granularity.is_mbs()
    mbs_headroom: Optional[torch.Tensor] = None
    if use_mbs:
        padded_M, padded_N = padded_tensor_shape
        MBS_BLOCK_SIZE = config.get_mbs_block_size()

        _, fp_max = ELEM_DTYPE_RANGE_MAP[config.quant_mode][config.elem_dtype]

        # Convert to float32 for MBS calculation
        padded_tensor_fp32 = padded_tensor.to(torch.float32)
        num_mbs_per_row = (padded_N + MBS_BLOCK_SIZE - 1) // MBS_BLOCK_SIZE

        # Pad the tensor width to be a multiple of MBS_BLOCK_SIZE if needed
        if _is_fx_proxy(padded_tensor_fp32):
            # Divisibility is unknown while tracing, so always pad (by 0 if aligned).
            mbs_padded_N = num_mbs_per_row * MBS_BLOCK_SIZE
            mbs_padded_tensor = torch.nn.functional.pad(
                padded_tensor_fp32, (0, mbs_padded_N - padded_N)
            )
        elif padded_N % MBS_BLOCK_SIZE != 0:
            mbs_padded_N = num_mbs_per_row * MBS_BLOCK_SIZE
            mbs_padded_tensor = torch.zeros(
                padded_M, mbs_padded_N, device=padded_tensor.device, dtype=torch.float32
            )
            mbs_padded_tensor[:, :padded_N] = padded_tensor_fp32
        else:
            mbs_padded_N = padded_N
            mbs_padded_tensor = padded_tensor_fp32

        mbs_reshaped = mbs_padded_tensor.view(padded_M, num_mbs_per_row, MBS_BLOCK_SIZE)

        # Derive the macro factor. This is the ONLY step that differs between MBS
        # rules; the padding/reshape above and the expand/scale below are shared.
        # Add a new rule as a sibling of these two, not as a new qdq variant.
        quant_blocks_per_mb = MBS_BLOCK_SIZE // n
        quant_block_peaks = _mbs_quant_block_peaks(mbs_padded_tensor, n).reshape(
            padded_M, num_mbs_per_row, quant_blocks_per_mb
        )
        if config.mbs_dynamic:
            mbs = _mbs_factor_dynamic(
                mbs_reshaped, tensor_name, config, quant_block_peaks
            )
        else:
            mbs = _mbs_factor_from_amax(
                torch.amax(quant_block_peaks, dim=2, keepdim=False), config, fp_max
            )

        def per_quant_block(per_chunk: torch.Tensor, blocks: int) -> torch.Tensor:
            # (M, chunks) with one value per `blocks` quant blocks -> one value
            # per quant block, (num_blocks, 1); MBS needs m == 1, so a quant
            # block is n consecutive elements of a row.
            expanded = per_chunk.unsqueeze(2).expand(-1, -1, blocks)
            per_block = expanded.reshape(padded_M, mbs_padded_N // n)
            return per_block[:, : padded_N // n].reshape(-1, 1)

        mbs_divisor = 1.0 / mbs
        divisor_blocks = quant_blocks_per_mb
        # An FP32/FP16 static factor is fp_max / amax and cannot overflow.
        if config.mbs_dynamic or config.global_scale_dtype == ScaleDtype.E0M8:
            headroom = _mbs_headroom(quant_block_peaks)
            mbs_divisor = (mbs_divisor.unsqueeze(2) * headroom).reshape(
                padded_M, num_mbs_per_row * quant_blocks_per_mb
            )
            mbs_headroom = per_quant_block(
                headroom.reshape(padded_M, num_mbs_per_row * quant_blocks_per_mb), 1
            )
            divisor_blocks = 1
        mbs_for_unscale = per_quant_block(mbs_divisor, divisor_blocks)

        data_hp = data_hp / mbs_for_unscale
    else:
        mbs_for_unscale = None

    # ── Phase 2: Scaling factor -- block statistics, amax rule, scale dtype ──
    # find the max and min value for each block
    if config.quant_mode == QuantMode.ASYMM:
        if excluded is None:
            max_value = torch.amax(data_hp, 1)
            min_value = torch.amin(data_hp, 1)
        else:
            max_value = torch.amax(data_hp.masked_fill(excluded, float("-inf")), 1)
            min_value = torch.amin(data_hp.masked_fill(excluded, float("inf")), 1)
            # A block of padding only is a zero block.
            empty = excluded.all(dim=1)
            max_value = max_value.masked_fill(empty, 0.0)
            min_value = min_value.masked_fill(empty, 0.0)
        delta = max_value - min_value
        delta = (
            delta / 2.0
            if config.elem_dtype not in [ElemDtype.INT3, ElemDtype.INT4, ElemDtype.INT8]
            else delta
        )
    elif config.quant_mode == QuantMode.ASYMM_MEAN_SHIFT:
        quant_zero_point, delta = _mean_shift_stats(data_hp)
        zero_point = quant_zero_point.to(torch.float16)
    else:
        max_abs_value = torch.amax(torch.abs(data_hp), 1)
        delta = max_abs_value

    # get the min and max values for the elem dtype
    min_pos, max_pos = ELEM_DTYPE_RANGE_MAP[config.quant_mode][config.elem_dtype]

    derived = config._derive_scale(
        _EagerScalePhase(
            config=config,
            data_hp=data_hp,
            delta=delta,
            padded_tensor=padded_tensor,
            padded_tensor_shape=padded_tensor_shape,
            max_pos=max_pos,
        )
    )
    quant_unscale = derived.quant_unscale
    scale = derived.scale
    unscale = derived.unscale
    local_unscale = derived.local_unscale
    global_scale = derived.global_scale

    # ── Phase 3: Quantize -- scale the data and cast to the element format ──
    # compute the low precision data
    if config.quant_mode in (QuantMode.ASYMM, QuantMode.ASYMM_MEAN_SHIFT):
        if config.quant_mode == QuantMode.ASYMM:
            # when min_pos is 0, the range is not centered at 0, so use the min_abs_value as the zero point instead
            zero_point = (max_value + min_value) / 2 if min_pos < 0 else min_value
            # we assume the offsets use fp16 dtype
            zero_point = zero_point.to(torch.float16)
            # get the low precision data by subtracting the zero point and dividing by the scale
            data_lp = _scale_elements(
                data_hp - zero_point.unsqueeze(1), scale, unscale, config
            )
        elif config.scale_dtype == ScaleDtype.FP16:
            data_lp = (data_hp - quant_zero_point.unsqueeze(1)) / (
                quant_unscale.unsqueeze(1)
            )
        else:
            data_lp = (data_hp - quant_zero_point.unsqueeze(1)) * scale.unsqueeze(1)
    else:
        zero_point: torch.Tensor = None
        if global_scale is not None:
            # NVFP4: divide by per-block FP8 scale and apply global_scale
            # separately. A single combined factor (`data_hp * scale`, where
            # scale = 1/(fp8_byte/global_scale_fp64) cast to fp32) loses ~1 ULP
            # at exact E2M1 midpoints because the fp64->fp32 cast on the
            # combined factor isn't bit-equivalent to two clean fp32 ops.
            per_block_scale_fp32 = local_unscale.to(torch.float32)
            global_scale_fp32 = global_scale.to(torch.float32)
            # Mask invalid amax cells to 1.0 so data_lp = data_hp passes
            # through inf/nan/zero unchanged. Matches the masked-safe
            # local_unscale_q = 1.0 behavior in
            # calculate_fp8_scale_for_nvfp4.
            per_block_scale_fp32 = torch.where(
                (per_block_scale_fp32 == 0)
                | torch.isinf(per_block_scale_fp32)
                | torch.isnan(per_block_scale_fp32),
                1.0,
                per_block_scale_fp32,
            )
            global_scale_fp32 = torch.where(
                (global_scale_fp32 == 0)
                | torch.isinf(global_scale_fp32)
                | torch.isnan(global_scale_fp32),
                1.0,
                global_scale_fp32,
            )
            if config.global_scale_granularity == GlobalScaleGranularity.ROWWISE:
                num_blocks_col = padded_tensor_shape[1] // n
                global_scale_fp32 = (
                    global_scale_fp32.expand(-1, num_blocks_col)
                    .reshape(-1)
                    .unsqueeze(1)
                )
            # data_hp is [num_blocks, block_size], so per-block scale must be
            # [num_blocks, 1] for both tensorwise and rowwise global scales.
            per_block_scale_fp32 = per_block_scale_fp32.reshape(-1, 1)
            data_lp = (data_hp * global_scale_fp32) / per_block_scale_fp32
        else:
            data_lp = _scale_elements(data_hp, scale, unscale, config)

    if config.elem_round_mode == ElemRoundMode.STOCHASTIC:
        data_lp = stochastic_rounding(data_lp, config.elem_dtype)

    ebits, mbits = ELEM_DTYPE_BITS_MAP[config.elem_dtype]
    if config.elem_dtype in (ElemDtype.INT3, ElemDtype.INT4, ElemDtype.INT8):
        # Clamp before rounding: with integer bounds the codes are those of
        # rounding first, but a value just below ASYMM's bound 0 becomes +0.0,
        # not -0.0, as in the kernel.
        data_lp = torch.round(torch.clamp(data_lp, min=min_pos, max=max_pos))
    elif config.elem_dtype in (
        ElemDtype.E3M0,
        ElemDtype.E2M1,
        ElemDtype.E2M3,
        ElemDtype.E4M3,
    ):
        if (
            config.quant_mode == QuantMode.ASYMM_MEAN_SHIFT
            and config.elem_dtype == ElemDtype.E4M3
        ):
            data_lp = torch.clamp(data_lp, min=min_pos, max=max_pos)
            data_lp = data_lp.to(torch.float8_e4m3fn).to(torch.float32)
        else:
            data_lp = fp32_to_fpx(
                data_lp, elem_dtype=config.elem_dtype, max_pos=max_pos
            )
    elif config.elem_dtype in (
        ElemDtype.E8M1,
        ElemDtype.E8M2,
        ElemDtype.E8M3,
    ):
        data_lp = fp32_to_e8mx(data_lp, elem_dtype=config.elem_dtype)
    else:
        raise ValueError(f"Unsupported elem dtype {config.elem_dtype}")

    # ── Phase 4: Post-process -- per-element corrections, then unwind ──
    data_lp = config._post_process(
        _EagerPostProcess(
            data=data_lp,
            data_hp=data_hp,
            scale=scale,
            min_pos=min_pos,
            max_pos=max_pos,
        )
    )

    if config.quant_mode == QuantMode.ASYMM_MEAN_SHIFT:
        # No passthrough: a NaN input, and every element of a block with an
        # Inf (whose mean is NaN), get a NaN code and dequantize to NaN. Set it
        # here, as an element conversion may saturate a NaN to a code.
        data_lp = data_lp.masked_fill(
            torch.isnan(data_hp - quant_zero_point.unsqueeze(1)), float("nan")
        )

    # A NaN anywhere in an MX block maps to the E8M0 NaN scale byte (255), so the
    # whole block dequantizes to NaN; such a block, and only such a block, has
    # a NaN amax here (+-Inf was clamped to +-FLT_MAX above), but for the static
    # FP32 MBS factor above.
    mx_nan_block: Optional[torch.Tensor] = None
    dequant_unscale = unscale
    if is_mx_fp:
        mx_nan_block = (
            torch.isnan(delta) if mx_nan_mask is None else mx_nan_mask.any(dim=1)
        )
        dequant_unscale = unscale.masked_fill(mx_nan_block, float("nan"))

    # dequantize the tensor and reshape it back to the original shape
    if (
        config.quant_mode == QuantMode.ASYMM_MEAN_SHIFT
        and config.scale_dtype == ScaleDtype.FP16
        and config.elem_dtype == ElemDtype.E4M3
    ):
        # Triton rounds the fused FP16 multiply-add once. The FP32 operations
        # are exact for FP16 operands; cast only the final sum back to FP16.
        dequant_data = (
            (
                data_lp.to(torch.float16).to(torch.float32)
                * unscale.unsqueeze(1).to(torch.float32)
                + zero_point.unsqueeze(1).to(torch.float32)
            )
            .to(torch.float16)
            .to(torch.float32)
        )
    else:
        dequant_data = data_lp * dequant_unscale.unsqueeze(1)
        if config.quant_mode in (QuantMode.ASYMM, QuantMode.ASYMM_MEAN_SHIFT):
            dequant_data += zero_point.unsqueeze(1)

    if mbs_for_unscale is not None:
        dequant_data = dequant_data * mbs_for_unscale
    # handle edge cases for input data
    if mx_nan_block is not None:
        # (±Inf was already saturated to ±FLT_MAX before scaling, so it flows
        # through the normal path and lands on the signed max-power-of-two
        # code.)
        if return_quant_info:
            # E8M0 scale byte 255 (unscale = +Inf, whose fp32 exponent field is all ones)
            # and the canonical all-ones element code (max-magnitude finite for
            # E2M1/E2M3, NaN for E4M3; sign dropped).
            unscale = unscale.masked_fill(mx_nan_block, float("inf"))
            nan_code_val = (
                float("nan") if config.elem_dtype == ElemDtype.E4M3 else max_pos
            )
            data_lp = data_lp.masked_fill(mx_nan_block.unsqueeze(1), nan_code_val)
    elif config.quant_mode != QuantMode.ASYMM_MEAN_SHIFT:
        original_data_hp = (
            data_hp if mbs_for_unscale is None else data_hp * mbs_for_unscale
        )
        data_hp_inf_mask = torch.isinf(original_data_hp)
        data_hp_nan_mask = torch.isnan(original_data_hp)
        mask = data_hp_inf_mask | data_hp_nan_mask
        dequant_data = torch.where(mask, original_data_hp, dequant_data)

    # reshape back
    dequant_data = _reshape_input_back(dequant_data, m, n, padded_tensor_shape)
    dequant_data = _restore_2d_padded(dequant_data, M, N, transpose, orig_shape)
    dequant_data = dequant_data.to(orig_dtype)

    if return_quant_info:
        quantized_val = _reshape_input_back(data_lp, m, n, padded_tensor_shape)
        quantized_val = _restore_2d_padded(quantized_val, M, N, transpose, orig_shape)
        if mbs_headroom is not None:
            # The scale of the block as the factor alone scales it: the 4 of an
            # overflow-guarded quant block moves from the divisor to the scale.
            unscale = unscale * mbs_headroom.reshape(-1)
        local_unscale = (
            local_unscale if config.scale_dtype == ScaleDtype.FP8 else unscale
        )
        # global_scale is only non-None for FP8 scale (NVFP4 two-level scaling)
        return QuantInfo(
            dequant_data=dequant_data,
            quantized_val=quantized_val,
            local_unscale=local_unscale.to(torch.float32),
            global_scale=(
                global_scale.to(config.global_scale_dtype.to_torch_dtype())
                if global_scale is not None
                else None
            ),
        )
    return dequant_data
