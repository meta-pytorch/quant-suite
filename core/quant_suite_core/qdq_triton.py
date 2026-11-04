# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The Triton QDQ backend: ``qdq_triton_mega`` plus the block-size guard that only
it uses.

One fused mega kernel does the reshape, scale selection, element cast and
reshape-back.

The numbered phases in the body are the same four this backend shares with
``qdq``, at a finer grain: pre-process through Phase 2, the host half of the
scaling factor in Phase 4, quantize inside the kernel (Phase 6), post-process in
Phase 7.
The ends of the scaling factor and of post-process are the
``QuantConfig._derive_scale`` / ``QuantConfig._post_process`` extension points,
whose state is ``_FusedScalePhase`` / ``_FusedPostProcess``.
"""

from dataclasses import dataclass, replace as dataclasses_replace
from typing import NamedTuple, Optional, Tuple, Union

import torch
from quant_suite_core.extension import NVFP4AmaxRule
from quant_suite_core.quant_config import (
    _global_scale_granularity_token,
    ELEM_DTYPE_BITS_MAP,
    elem_dtype_max_pow2,
    ELEM_DTYPE_MAX_POW2_MAP,
    ELEM_DTYPE_RANGE_MAP,
    ElemDtype,
    F32_MIN_NORMAL,
    GlobalScaleGranularity,
    MBS_DYNAMIC_SLOTS,
    pow2max_norm_exp,
    QuantConfig,
    QuantInfo,
    QuantMode,
    ScaleDtype,
    ScaleNormMode,
)
from quant_suite_core.quant_primitives import (
    _block_padding_mask,
    _div_rn,
    _mean_shift_stats,
    _nvfp4_global_scale_or_one,
    _prepare_2d_padded,
    _reshape_input,
    _restore_2d_padded,
    _TRITON_MAX_BLOCK_NUMEL_PER_QUANT_BLOCK,
    fp32_to_e8m0,
    nvfp4_global_scale,
    TRITON_MAX_BLOCK_NUMEL,
    TRITON_MEGA_BLOCK_M,
    TRITON_MEGA_TILE_NUMEL,
)
from quant_suite_core.triton_utils import (
    _exceeds_triton_block_numel,
    _mega_fake_quant_triton_kernel,
    _stochastic_rounding_params,
)
from triton import cdiv  # @manual


def _abs_amax(x: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """The largest |x|, over ``dim`` or the whole tensor, in ``x``'s dtype.

    ``vector_norm`` with ``ord=inf`` is that maximum, exactly (a NaN gives NaN,
    as in ``amax``), as one reduction with no |x| temporary.
    """
    return torch.linalg.vector_norm(x, ord=float("inf"), dim=dim)


class _FusedScale(NamedTuple):
    """What ``qdq_triton_mega`` takes out of the scaling-factor phase.

    Normally only the tensor-global half, because the kernel reduces the
    per-block amax itself. Under a rule it also carries ``local_amax``, which
    the kernel loads in place of that reduction -- the rule has already computed
    it, so recomputing would be both wasted work and a second definition.
    """

    two_level_scale: Optional[torch.Tensor]
    local_amax: Optional[torch.Tensor]
    use_local_amax: bool
    # None unless the scale dtype is FP8, the only case the kernel reads it.
    global_amax: Optional[torch.Tensor]


@dataclass
class _FusedScalePhase:
    """``qdq_triton_mega``'s scaling-factor phase, for ``QuantConfig._derive_scale``.

    The fields below the marker are what an amax rule substitutes; everything
    ``derive`` does with them is fixed.
    """

    config: QuantConfig
    tensor: torch.Tensor
    padded_tensor: torch.Tensor
    # The unpadded 2-D shape of ``padded_tensor``.
    M: int
    N: int
    m: int
    n: int
    new_M: int
    # --- substitutable by an amax rule ---
    # The three NVFP4 amax quantities. Unset here means the default derivation
    # runs: the kernel reduces the per-block amax and ``derive`` the global one.
    local_max_pos: float
    local_amax: Optional[torch.Tensor] = None
    global_amax: Optional[torch.Tensor] = None

    def with_amax_rule(self, rule: NVFP4AmaxRule) -> "_FusedScalePhase":
        """Take all three NVFP4 amax quantities from the rule.

        ``local_amax`` goes to the kernel as a tensor it loads instead of
        reducing, which is the same value it would have reduced -- the rule
        computed the per-block amax already in order to make its choice.
        """
        data_blocks = _reshape_input(
            self.padded_tensor.to(torch.float32), self.m, self.n
        )
        local_amax, local_max_pos, global_amax = rule.amax(
            data_blocks, self.local_max_pos
        )
        return dataclasses_replace(
            self,
            local_amax=local_amax,
            local_max_pos=local_max_pos,
            global_amax=global_amax,
        )

    def derive(self) -> _FusedScale:
        config = self.config
        if config.global_scale_granularity not in (
            GlobalScaleGranularity.TENSORWISE,
            GlobalScaleGranularity.ROWWISE,
        ):
            return _FusedScale(
                two_level_scale=None,
                local_amax=None,
                use_local_amax=False,
                global_amax=None,
            )

        global_amax = self.global_amax
        if global_amax is None:
            if config.global_scale_granularity == GlobalScaleGranularity.ROWWISE:
                # One amax per row of quant blocks.
                rows = _reshape_input(
                    self.padded_tensor, config.block_size.m, self.padded_tensor.shape[1]
                )
                global_amax = _abs_amax(rows, dim=-1)
            elif config.quant_mode == QuantMode.ASYMM:
                # The largest per-block delta, halved for float elements exactly
                # as the kernel halves each block's own delta. The zero padding
                # is left out of the block min and max, which are exact in the
                # input dtype, so only they are widened to fp32.
                blocks = _reshape_input(self.padded_tensor, self.m, self.n)
                padding = _block_padding_mask(
                    self.padded_tensor, self.M, self.N, self.m, self.n
                )
                if padding is None:
                    lo, hi = torch.aminmax(blocks, dim=1)
                else:
                    hi = torch.amax(blocks.masked_fill(padding, float("-inf")), 1)
                    lo = torch.amin(blocks.masked_fill(padding, float("inf")), 1)
                    # A block of padding only is a zero block.
                    empty = padding.all(dim=1)
                    hi = hi.masked_fill(empty, 0.0)
                    lo = lo.masked_fill(empty, 0.0)
                delta = hi.to(torch.float32) - lo.to(torch.float32)
                if config.elem_dtype not in (
                    ElemDtype.INT3,
                    ElemDtype.INT4,
                    ElemDtype.INT8,
                ):
                    delta = delta / 2.0
                global_amax = torch.amax(delta)
            elif config.quant_mode == QuantMode.ASYMM_MEAN_SHIFT:
                # The largest per-block amax of the mean-centred values, as in
                # qdq(); the kernel centres each block on the same mean.
                _, delta = _mean_shift_stats(
                    _reshape_input(self.padded_tensor.to(torch.float32), self.m, self.n)
                )
                global_amax = torch.amax(delta)
            else:
                global_amax = _abs_amax(self.tensor).to(torch.float32)

        # Capture-safe, and the value calculate_fp8_scale_for_nvfp4 uses.
        two_level_scale = nvfp4_global_scale(global_amax, self.local_max_pos)
        return _FusedScale(
            two_level_scale=two_level_scale,
            local_amax=self.local_amax,
            use_local_amax=self.local_amax is not None,
            global_amax=global_amax,
        )


@dataclass
class _FusedPostProcess:
    """The fused backend's state at the end of post-process, for
    ``QuantConfig._post_process``.

    ``data`` is the dequantized padded 2-D output the kernel just wrote, so a
    step here works in the dequantized domain and applies its own unscale.
    """

    config: QuantConfig
    data: torch.Tensor
    padded_tensor: torch.Tensor
    m: int
    n: int
    padded_N: int
    min_pos: float
    max_pos: float
    max_pow2: float
    # Derived on demand, then cached: a post-process step needs both the block
    # view and the scale, and reconstructing the scale is the expensive half.
    _blocks: Optional[torch.Tensor] = None
    _scale: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def block_view(self) -> torch.Tensor:
        if self._blocks is None:
            self._blocks = _reshape_input(
                self.padded_tensor.to(torch.float32), self.m, self.n
            )
        return self._blocks

    def block_scale(self) -> torch.Tensor:
        return self._scale_pair()[0]

    def _scale_pair(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The per-block (scale, unscale) the kernel used, recomputed.

        The mega kernel does not expose the scale it baked into the dequantized
        data, so this reproduces its derivation -- including the POW2MAX
        round-then-shift, so the two agree on subnormal blocks.

        Known divergences from the eager backend's equivalent:
         1. No +-FLT_MAX clamp on the blocks: an Inf element gives an Inf amax,
            whose e8m0 unscale is rejected as invalid and replaced by 1.0, so
            `_safe_lshift` evaluates Inf/Inf = NaN. `qdq` clamps first and lands
            on max_pos * unscale.
         2. `delta` is always the symmetric amax, so an ASYMM / ASYMM_MEAN_SHIFT
            config gets a different scale here from the one `qdq` uses.
         3. NaN-block poisoning already happened inside the kernel, so a write
            through `scatter_block_elements` lands on top of it; in `qdq` the
            post-pass runs first and the poisoning covers it.
        """
        if self._scale is not None:
            return self._scale
        config = self.config
        norm_exp_mx = 0
        delta = torch.amax(torch.abs(self.block_view()), dim=1)
        # Divided correctly rounded, as the kernel does.
        if config.scale_norm_mode == ScaleNormMode.AMAX:
            unscale_mx = _div_rn(delta, self.max_pos)
        else:  # POW2MAX
            norm_exp_mx = pow2max_norm_exp(self.max_pow2)
            # round-then-shift via norm_exp; pre-divide only for non-pow2
            # max_pow2 (E8Mx, where norm_exp_mx == 0)
            unscale_mx = delta if norm_exp_mx != 0 else _div_rn(delta, self.max_pow2)

        # Apply E8M0 scale quantization
        self._scale = fp32_to_e8m0(
            unscale_mx,
            elem_dtype=config.elem_dtype,
            scale_round_mode=config.scale_round_mode,
            norm_exp=norm_exp_mx,
        )
        return self._scale

    def scatter_block_elements(
        self, index: torch.Tensor, value_lp: torch.Tensor
    ) -> torch.Tensor:
        m, n = self.m, self.n
        dequant_data = self.data
        num_blocks = self.block_view().shape[0]

        # This backend writes into the dequantized domain, so the unscale is
        # applied here rather than by a shared tail.
        value_dequant = value_lp * self._scale_pair()[1]

        # dequant_data has shape (padded_M, padded_N), so map block indices back.
        # Block (i, j) in the reshaped view covers rows [i*m:(i+1)*m] and cols
        # [j*n:(j+1)*n].
        num_blocks_col = self.padded_N // n
        block_index = torch.arange(num_blocks, device=dequant_data.device)
        block_row_idx = block_index // num_blocks_col
        block_col_idx = block_index % num_blocks_col

        # `index` is the position within the flattened block (0 to m*n-1);
        # split it into a row and a column offset inside the block.
        global_row = block_row_idx * m + index // n
        global_col = block_col_idx * n + index % n

        dequant_data[global_row, global_col] = value_dequant.to(dequant_data.dtype)
        return dequant_data


def qdq_triton_mega(  # noqa: C901
    tensor: torch.Tensor,
    tensor_name: str,
    config: QuantConfig,
    transpose: bool = False,
    return_quant_info: bool = False,
    *,
    num_valid_rows: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, QuantInfo]:
    """
    Perform quantization and dequantization using a Triton mega kernel.

    One fused Triton kernel does the reshape, scale calculation, element cast
    and reshape back, for every config ``is_triton_mega_supported`` accepts
    (MX, NVFP4, MBS and integer formats). Needs a CUDA device.

    With ``return_quant_info=True`` the kernel also writes the quantized
    elements and the per-block unscale, which are returned as
    ``QuantInfo.quantized_val`` and ``QuantInfo.local_unscale``.

    With stochastic rounding, an eager call draws a fresh seed from the CUDA
    generator and advances it: repeated calls round with fresh random bits, and
    ``torch.manual_seed`` reproduces a call. A captured CUDA graph draws its seed
    on the device, so each replay gets a new one.

    Args:
        tensor (Tensor): The input tensor to be quantized.
        tensor_name (str): Name of the tensor (for debugging/logging).
        config (QuantConfig): The quantization configuration.
        transpose (bool): Whether to transpose the block dimensions.
        return_quant_info (bool): If True, returns QuantInfo with dequant_data,
            quantized_val, local_unscale and global_scale.
        num_valid_rows (Tensor, optional): 0-d int32/int64 tensor on the
            input's device holding the number of leading rows (of the 2-D view,
            leading dimensions flattened) that hold data, for example in a
            preallocated buffer filled to a variable length. The kernel reads
            it and quantizes only those rows, rounded up to whole quant blocks.
            The host never reads the value, so the call is CUDA-graph safe
            with a value that changes between replays. Output rows below the
            bound are bit-identical to a call without ``num_valid_rows``; the
            rest of the output (and of ``quantized_val`` / ``local_unscale``)
            is unspecified. Scale values computed outside the kernel (a
            tensorwise or rowwise global amax) still cover the whole tensor.
            Not supported with ``transpose=True`` or config extensions.

    Returns:
        Union[torch.Tensor, QuantInfo]: The dequantized tensor, or QuantInfo.
    """
    # ── Phase 1: Extract config values and prepare padded 2D tensor ──────
    _reject_oversized_triton_block(config, "qdq_triton_mega")
    if num_valid_rows is not None:
        _check_num_valid_rows(tensor, config, transpose, num_valid_rows)
    m = config.block_size.m
    n = config.block_size.n
    BLOCK_M = _mega_block_m(config)
    padded_tensor, M, N, orig_shape, orig_dtype = _prepare_2d_padded(
        tensor, m, n, transpose
    )

    quant_mode = config.quant_mode.value
    scale_dtype = config.scale_dtype.value
    scale_round_mode = config.scale_round_mode.value
    scale_norm_mode = config.scale_norm_mode.value
    elem_round_mode = config.elem_round_mode.value
    elem_dtype = config.elem_dtype.value
    # The packed token, not the bare granularity: it is a `tl.constexpr` and the
    # kernel branches on the packed spelling.
    global_scale_granularity = _global_scale_granularity_token(config)

    ebits, mbits = ELEM_DTYPE_BITS_MAP[config.elem_dtype]
    min_exp = -(2 ** (ebits - 1)) + 2
    min_pos, max_pos = ELEM_DTYPE_RANGE_MAP[config.quant_mode][config.elem_dtype]
    max_pow2 = (
        elem_dtype_max_pow2(config.quant_mode, config.elem_dtype)
        if config.elem_dtype in ELEM_DTYPE_MAX_POW2_MAP
        else 1.0
    )
    if config.elem_dtype not in ELEM_DTYPE_MAX_POW2_MAP:
        assert config.scale_norm_mode != ScaleNormMode.POW2MAX, (
            f"{config.elem_dtype} does not support POW2MAX"
        )
    padded_M, padded_N = padded_tensor.shape
    new_N = m * n
    new_M = padded_M * padded_N // new_N

    if config.mbs_dynamic and config.scale_dtype != ScaleDtype.E8M0:
        # The in-kernel search derives each candidate's scale through the e8m0 path
        # only, so another scale dtype would be ranked against the wrong quantizer
        # and return a plausible-looking result at roughly the right SQNR.
        raise NotImplementedError(
            f"qdq_triton_mega implements mbs_dynamic for scale_dtype e8m0 only, got "
            f"{config.scale_dtype.value}; use fake_quantize(use_triton=False), "
            f"whose eager qdq path implements the rule for every scale dtype."
        )

    # ── Phase 2: MBS setup ────────────────────────────────────────────
    # The width qdq() works on; the MBS padding below widens the kernel's.
    quant_padded_N = padded_N
    use_mbs = config.global_scale_granularity.is_mbs()
    MBS_BLOCK_SIZE = 0
    fp_max_mbs = 0.0
    global_scale_dtype_mbs = "none"
    if use_mbs:
        _, fp_max_mbs = ELEM_DTYPE_RANGE_MAP[config.quant_mode][config.elem_dtype]
        MBS_BLOCK_SIZE = config.get_mbs_block_size()
        if config.global_scale_dtype not in (ScaleDtype.E0M8, ScaleDtype.FP32):
            raise NotImplementedError(
                f"qdq_triton_mega implements the MBS global scale dtypes e0m8 and "
                f"fp32 only, got {config.global_scale_dtype.value}"
            )
        global_scale_dtype_mbs = config.global_scale_dtype.value
        if MBS_BLOCK_SIZE % n != 0:
            raise NotImplementedError(
                f"MBS_BLOCK_SIZE ({MBS_BLOCK_SIZE}) must be divisible by quant block size n ({n})"
            )
        mbs_span = MBS_BLOCK_SIZE // n
        if BLOCK_M % mbs_span != 0:
            raise NotImplementedError(
                f"BLOCK_M ({BLOCK_M}) must be divisible by mbs_span ({mbs_span} = MBS_BLOCK_SIZE {MBS_BLOCK_SIZE} / n {n})"
            )
        # Pad padded_N to be a multiple of MBS_BLOCK_SIZE for fused kernel
        mbs_padded_N = (
            (padded_N + MBS_BLOCK_SIZE - 1) // MBS_BLOCK_SIZE
        ) * MBS_BLOCK_SIZE
        if mbs_padded_N != padded_N:
            new_padded = torch.zeros(
                padded_M, mbs_padded_N, device=tensor.device, dtype=orig_dtype
            )
            new_padded[:, :padded_N] = padded_tensor
            padded_tensor = new_padded
            padded_N = mbs_padded_N
        # Recompute new_M after potential re-padding
        new_M = padded_M * padded_N // new_N

    # ── Phase 3: Allocate output tensors ──────────────────────────────
    dequant_data = torch.empty(
        padded_tensor.shape, device=tensor.device, dtype=orig_dtype
    )
    # The kernel writes these only with return_quant_info.
    local_unscale: Optional[torch.Tensor] = None
    data_lp: Optional[torch.Tensor] = None
    if return_quant_info:
        local_unscale = torch.empty(new_M, device=tensor.device, dtype=torch.float32)
        data_lp = torch.empty(
            (padded_M, padded_N), device=tensor.device, dtype=torch.float32
        )

    # ── Phase 4: FP8 two-level scale (NVFP4) -- closes the scaling factor ──
    two_level_scale, local_amax_override, use_local_amax, global_amax = (
        config._derive_scale(
            _FusedScalePhase(
                config=config,
                tensor=tensor,
                padded_tensor=padded_tensor,
                M=M,
                N=N,
                m=m,
                n=n,
                new_M=new_M,
                local_max_pos=max_pos,
            )
        )
    )

    # ── Phase 5: Stochastic rounding params ───────────────────────────
    seed, max_normal, min_normal = _stochastic_rounding_params(
        config, tensor.device, max_pos
    )

    # ── Phase 6: Launch kernel ────────────────────────────────────────
    def grid_0(META):
        return (cdiv(new_M, META["BLOCK_M"]),)

    _mega_fake_quant_triton_kernel[grid_0](
        in_ptr=padded_tensor,
        out_ptr=dequant_data,
        global_amax_ptr=global_amax,
        two_level_scale_ptr=two_level_scale,
        local_unscale_ptr=local_unscale,
        data_lp_ptr=data_lp,
        local_amax_ptr=local_amax_override,
        seed=seed,
        M=padded_M,
        N=padded_N,
        new_M=new_M,
        new_N=new_N,
        stride_in0=padded_tensor.stride(0),
        stride_in1=padded_tensor.stride(1),
        stride_out0=dequant_data.stride(0),
        stride_out1=dequant_data.stride(1),
        stride_lp0=0 if data_lp is None else data_lp.stride(0),
        stride_lp1=0 if data_lp is None else data_lp.stride(1),
        m=m,
        n=n,
        min_pos=min_pos,
        max_pos=max_pos,
        max_pow2=max_pow2,
        e8m0_norm_exp=(
            pow2max_norm_exp(max_pow2) if scale_norm_mode == "pow2max" else 0
        ),
        min_exp=min_exp,
        ebits=ebits,
        mbits=mbits,
        max_normal=max_normal,
        min_normal=min_normal,
        scale_dtype=scale_dtype,
        scale_norm_mode=scale_norm_mode,
        scale_round_mode=scale_round_mode,
        quant_mode=quant_mode,
        elem_dtype=elem_dtype,
        elem_round_mode=elem_round_mode,
        global_scale_granularity=global_scale_granularity,
        use_mbs=use_mbs,
        return_quant_info=return_quant_info,
        native_quant=False,
        MBS_BLOCK_SIZE=MBS_BLOCK_SIZE,
        fp_max_mbs=fp_max_mbs,
        global_scale_dtype_mbs=global_scale_dtype_mbs,
        mbs_dynamic=config.mbs_dynamic,
        mbs_dynamic_slots=MBS_DYNAMIC_SLOTS if config.mbs_dynamic else 0,
        BLOCK_M=BLOCK_M,
        BLOCK_N=new_N,
        F32_MIN_NORMAL=F32_MIN_NORMAL,
        use_local_amax=use_local_amax,
        # torch.compile does not fill in a user kernel's default arguments, so
        # every parameter is passed explicitly.
        num_valid_rows_ptr=num_valid_rows,
        HAS_NUM_VALID_ROWS=num_valid_rows is not None,
        unpadded_M=M if config.quant_mode == QuantMode.ASYMM else None,
        unpadded_N=N if config.quant_mode == QuantMode.ASYMM else None,
    )

    # ── Phase 7: Post-process extension point ────────────────────────
    dequant_data = config._post_process(
        _FusedPostProcess(
            config=config,
            data=dequant_data,
            padded_tensor=padded_tensor,
            m=m,
            n=n,
            padded_N=padded_N,
            min_pos=min_pos,
            max_pos=max_pos,
            max_pow2=max_pow2,
        )
    )

    # ── Phase 8: Restore shape & return ──────────────────────────────
    dequant_data = _restore_2d_padded(dequant_data, M, N, transpose, orig_shape)

    if return_quant_info:
        assert data_lp is not None and local_unscale is not None
        data_lp = _restore_2d_padded(data_lp, M, N, transpose, orig_shape)
        rowwise = config.global_scale_granularity == GlobalScaleGranularity.ROWWISE
        # The shape qdq() returns: no blocks of the MBS padding (MBS needs
        # m == 1, so a row of quant blocks is a row of the input), and one
        # row of blocks per input row for a rowwise global scale. The widths
        # are explicit so an empty input keeps its shape.
        num_blocks_row = padded_M // m
        if padded_N != quant_padded_N:
            local_unscale = local_unscale.view(num_blocks_row, padded_N // n)[
                :, : quant_padded_N // n
            ].reshape(-1)
        if rowwise:
            local_unscale = local_unscale.view(num_blocks_row, quant_padded_N // n)
        global_scale = None
        # global_scale is only non-None for FP8 scale (NVFP4 two-level scaling)
        if two_level_scale is not None:
            # The kernel builds the block scales from this value too.
            global_scale = _nvfp4_global_scale_or_one(global_amax, two_level_scale).to(
                config.global_scale_dtype.to_torch_dtype()
            )
            if rowwise:
                global_scale = global_scale.view(-1, 1)
        return QuantInfo(
            dequant_data=dequant_data,
            quantized_val=data_lp,
            local_unscale=local_unscale,
            global_scale=global_scale,
        )
    return dequant_data


def _mega_block_m(config: QuantConfig) -> int:
    """Quant blocks per mega-kernel program instance.

    The largest power of two whose tile stays within TRITON_MEGA_TILE_NUMEL
    elements, capped at TRITON_MEGA_BLOCK_M and raised to one MBS macro block,
    which a tile must hold whole.

    Two families' outputs depend on BLOCK_M. Stochastic rounding keys its Philox
    stream by tile (seed = the call's seed + ``program_id``, counter = the index
    within the tile), and ``mbs_dynamic`` ranks its candidates by an fp32 sum whose
    reduction order follows the tile layout. Every such preset is 1x16, where the
    rule lands exactly on the cap of 128; giving one of them a smaller tile changes
    its output.
    """
    block_numel = config.block_size.m * config.block_size.n
    mbs_span = (
        config.get_mbs_block_size() // config.block_size.n
        if config.global_scale_granularity.is_mbs()
        else 1
    )
    fit = max(1, TRITON_MEGA_TILE_NUMEL // block_numel)
    return min(TRITON_MEGA_BLOCK_M, max(mbs_span, 1 << (fit.bit_length() - 1)))


def _check_num_valid_rows(
    tensor: torch.Tensor,
    config: QuantConfig,
    transpose: bool,
    num_valid_rows: torch.Tensor,
) -> None:
    """Host-side checks of ``qdq_triton_mega``'s ``num_valid_rows``.

    Reads only metadata: the value stays on the device.
    """
    if transpose:
        raise NotImplementedError(
            "num_valid_rows bounds rows; it needs transpose=False"
        )
    if config.extensions:
        # An extension's post-process may read rows past the bound, which the
        # kernel leaves uninitialized.
        raise NotImplementedError("num_valid_rows does not support config extensions")
    if (
        num_valid_rows.ndim != 0
        or num_valid_rows.dtype not in (torch.int32, torch.int64)
        or num_valid_rows.device != tensor.device
    ):
        raise ValueError(
            "num_valid_rows must be a 0-d int32/int64 tensor on "
            f"{tensor.device}; got shape {tuple(num_valid_rows.shape)}, "
            f"dtype {num_valid_rows.dtype}, device {num_valid_rows.device}"
        )


def _reject_oversized_triton_block(config: QuantConfig, fn_name: str) -> None:
    """Raise for a block size no Triton launcher can compile.

    Without this the caller gets Triton's own `numel (...) exceeds triton maximum
    tensor numel (...)` from deep inside the compiler, naming neither the config nor
    the way out.
    """
    if not _exceeds_triton_block_numel(config):
        return
    block_numel = config.block_size.m * config.block_size.n
    raise NotImplementedError(
        f"{fn_name} cannot serve block_size {config.block_size}: it tiles up to "
        f"BLOCK_M={TRITON_MEGA_BLOCK_M} quant blocks x BLOCK_N=m*n={block_numel} "
        f"elements = {TRITON_MEGA_BLOCK_M * block_numel}, over Triton's "
        f"{TRITON_MAX_BLOCK_NUMEL} tensor numel cap, so the ceiling is "
        f"m*n <= {_TRITON_MAX_BLOCK_NUMEL_PER_QUANT_BLOCK}; quantize with "
        f"use_triton=False, which runs eager qdq."
    )
