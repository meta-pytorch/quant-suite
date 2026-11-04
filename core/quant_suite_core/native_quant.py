# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native MXFP4, MXFP8 and NVFP4 quantization to packed hardware layouts.

Provides native_quantize() and native_dequantize(). Data is packed uint8 FP4
(two values per byte) or float8_e4m3fn; scales are uint8 E8M0 for MX formats,
or float8_e4m3fn block scales plus a tensorwise or rowwise global scale for
NVFP4. The MX layouts match flashinfer.fp4_quantize / flashinfer.mxfp8_quantize.
Only SYMM configs with E2M1 or E4M3 elements, an E8M0 or FP8 scale, 1xN
blocks and no MBS global scale are accepted. Extension hooks are not run.

Unlike qdq() which quantizes then immediately dequantizes (wasting the
dequantize step), native_quantize() only performs quantization and packing.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from quant_suite_core.mx_packing import (
    e8m0_uint8_to_float,
    float_e2m1_to_packed_uint8,
    float_to_e8m0_uint8,
    packed_uint8_to_float_e2m1,
)
from quant_suite_core.quant_config import (
    _global_scale_granularity_token,
    ELEM_DTYPE_BITS_MAP,
    elem_dtype_max_pow2,
    ELEM_DTYPE_MAX_POW2_MAP,
    ELEM_DTYPE_RANGE_MAP,
    ElemDtype,
    GlobalScaleGranularity,
    pow2max_norm_exp,
    QuantConfig,
    QuantMode,
    ScaleDtype,
    ScaleNormMode,
)
from quant_suite_core.quant_primitives import (
    _chunked_abs_amax,
    _div_rn,
    _reshape_2d_tensor,
    _reshape_input,
    _reshape_input_back,
    calculate_fp8_scale_for_nvfp4,
    fp32_to_e8m0,
    fp32_to_fpx,
    nvfp4_global_scale,
    stochastic_rounding,
)
from quant_suite_core.triton_utils import (
    _mega_fake_quant_triton_kernel,
    _stochastic_rounding_params,
)


@dataclass
class NativeQuantResult:
    """Result of native MX quantization.

    Attributes:
        data: Packed low-precision data.
            - MXFP4 (E2M1): uint8 [M, K//2], two FP4 values packed per byte
            - MXFP8 (E4M3): float8_e4m3fn [M, K]
        scale: Block scales.
            - MXFP4/MXFP8 (E8M0): uint8 [M, K//block_size]
            - NVFP4 (FP8): float8_e4m3fn [M, K//block_size]
        config: QuantConfig used for quantization.
        global_scale: Optional global scale for NVFP4-style two-level scaling.
            Scalar for tensorwise, [M] for rowwise granularity.
        orig_shape: Original tensor shape before quantization.
    """

    data: torch.Tensor
    scale: torch.Tensor
    config: QuantConfig
    global_scale: Optional[torch.Tensor] = None
    orig_shape: Optional[Tuple[int, ...]] = None


def _swizzle_scale(sf: torch.Tensor) -> torch.Tensor:
    """Swizzle scale factors from linear [M, K_sf] to 128x4 blocked layout.

    This matches the Tensor Core scale factor layout used by flashinfer
    (is_sf_swizzled_layout=True).

    The permutation tiles the [M, K_sf] matrix into 128-row x 4-column blocks,
    where each 128x4 tile is decomposed as [32_outerM, 4_innerM, 4_innerK].

    Args:
        sf: Scale factor tensor of shape [M, K_sf].

    Returns:
        Flattened uint8 tensor with swizzled layout.
    """
    rows, cols = sf.shape
    n_row_blocks = (rows + 127) // 128
    n_col_blocks = (cols + 3) // 4
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols), device=sf.device, dtype=sf.dtype
        )
        padded[:rows, :cols] = sf
    else:
        padded = sf

    # [M/128, 4_innerM, 32_outerM, K_sf/4, 4_innerK]
    # -> [M/128, K_sf/4, 32_outerM, 4_innerM, 4_innerK]
    return (
        padded.view(n_row_blocks, 4, 32, n_col_blocks, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .flatten()
    )


def _unswizzle_scale(sf: torch.Tensor, orig_rows: int, orig_cols: int) -> torch.Tensor:
    """Reverse the 128x4 blocked layout back to linear [rows, cols].

    Args:
        sf: Flattened swizzled scale factor tensor.
        orig_rows: Original number of rows before padding.
        orig_cols: Original number of scale columns before padding.

    Returns:
        Scale factor tensor of shape [orig_rows, orig_cols].
    """
    n_row_blocks = (orig_rows + 127) // 128
    n_col_blocks = (orig_cols + 3) // 4
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    # [M/128, K_sf/4, 32, 4, 4] -> [M/128, 4, 32, K_sf/4, 4]
    padded = (
        sf.view(n_row_blocks, n_col_blocks, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(padded_rows, padded_cols)
    )
    return padded[:orig_rows, :orig_cols].contiguous()


def _mx_quantize_fused_triton(
    tensor: torch.Tensor,
    config: QuantConfig,
    block_m: int = 128,
) -> NativeQuantResult:
    """Fused native quantization using the mega Triton kernel.

    Calls _mega_fake_quant_triton_kernel with native_quant=True to do
    reshape + amax + scale + element rounding + packing in a single kernel.
    """
    from triton import cdiv  # @manual

    orig_shape = tensor.shape
    orig_dtype = tensor.dtype
    m = config.block_size.m  # always 1 for native quant
    n = config.block_size.n

    # Reshape to 2D and pad
    tensor_2d = _reshape_2d_tensor(tensor)
    M, N = tensor_2d.shape
    padded_M = ((M + m - 1) // m) * m
    padded_N = ((N + n - 1) // n) * n
    if (M, N) != (padded_M, padded_N):
        padded_tensor = torch.zeros(
            padded_M, padded_N, device=tensor.device, dtype=orig_dtype
        )
        padded_tensor[:M, :N] = tensor_2d
    else:
        padded_tensor = tensor_2d

    new_N = m * n
    new_M = padded_M * padded_N // new_N

    # Config values
    quant_mode = config.quant_mode.value
    scale_dtype = config.scale_dtype.value
    scale_round_mode = config.scale_round_mode.value
    scale_norm_mode = config.scale_norm_mode.value
    elem_round_mode = config.elem_round_mode.value
    elem_dtype = config.elem_dtype.value
    global_scale_granularity = _global_scale_granularity_token(config)

    min_pos, max_pos = ELEM_DTYPE_RANGE_MAP[config.quant_mode][config.elem_dtype]
    max_pow2 = (
        elem_dtype_max_pow2(config.quant_mode, config.elem_dtype)
        if config.elem_dtype in ELEM_DTYPE_MAX_POW2_MAP
        else 1.0
    )
    ebits, mbits = ELEM_DTYPE_BITS_MAP[config.elem_dtype]
    min_exp = -(2 ** (ebits - 1)) + 2

    # Allocate native output tensors.
    if config.elem_dtype == ElemDtype.E2M1:
        # PTX packs 8 FP4 values into 1 int32 → shape [M, N//8]
        data_lp = torch.empty(
            (padded_M, padded_N // 8), device=tensor.device, dtype=torch.int32
        )
    else:
        # E4M3
        data_lp = torch.empty(
            (padded_M, padded_N), device=tensor.device, dtype=torch.float8_e4m3fn
        )

    # Scale output: uint8 [new_M] (kernel stores packed e8m0/fp8 directly)
    local_unscale = torch.empty(new_M, device=tensor.device, dtype=torch.uint8)

    # Dummy output (not used with native_quant=True)
    dequant_data = torch.empty(0, device=tensor.device, dtype=orig_dtype)

    two_level_scale = None
    if config.scale_dtype == ScaleDtype.FP8:
        if config.global_scale_granularity == GlobalScaleGranularity.ROWWISE:
            global_amax = _chunked_abs_amax(padded_tensor, config.block_size).to(
                torch.float32
            )
        else:
            global_amax = torch.amax(torch.abs(padded_tensor)).reshape(1)
        global_amax_invalid = (
            (global_amax == 0) | torch.isinf(global_amax) | torch.isnan(global_amax)
        )
        two_level_scale = nvfp4_global_scale(global_amax, max_pos)
        two_level_scale = torch.where(global_amax_invalid, 1.0, two_level_scale)
    else:
        global_amax = torch.ones(1, device=tensor.device, dtype=torch.float32)

    seed, max_normal, min_normal = _stochastic_rounding_params(
        config, tensor.device, max_pos
    )

    def grid_0(META):
        return (cdiv(new_M, META["BLOCK_M"]),)

    _mega_fake_quant_triton_kernel[grid_0](
        in_ptr=padded_tensor,
        out_ptr=dequant_data,
        global_amax_ptr=global_amax,
        two_level_scale_ptr=two_level_scale,
        local_unscale_ptr=local_unscale,
        data_lp_ptr=data_lp,
        local_amax_ptr=None,
        seed=seed,
        M=padded_M,
        N=padded_N,
        new_M=new_M,
        new_N=new_N,
        stride_in0=padded_tensor.stride(0),
        stride_in1=padded_tensor.stride(1),
        stride_out0=0,
        stride_out1=0,
        stride_lp0=data_lp.stride(0),
        stride_lp1=data_lp.stride(1),
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
        use_mbs=False,
        return_quant_info=False,
        native_quant=True,
        MBS_BLOCK_SIZE=0,
        fp_max_mbs=0,
        global_scale_dtype_mbs="none",
        mbs_dynamic=False,
        mbs_dynamic_slots=0,
        BLOCK_M=block_m,
        BLOCK_N=new_N,
        F32_MIN_NORMAL=2 ** (-126),
        use_local_amax=False,
        num_valid_rows_ptr=None,
        HAS_NUM_VALID_ROWS=False,
        unpadded_M=None,
        unpadded_N=None,
    )

    # View int32 as uint8 for E2M1: [M, N//8] int32 → [M, N//2] uint8
    if config.elem_dtype == ElemDtype.E2M1:
        data_lp = data_lp.view(torch.uint8).reshape(padded_M, padded_N // 2)

    # Trim padding (both rows and columns)
    if config.elem_dtype == ElemDtype.E2M1:
        packed_data = data_lp[:M, : N // 2]
    else:
        packed_data = data_lp[:M, :N]
    num_scale_cols = padded_N // n
    if config.scale_dtype == ScaleDtype.FP8:
        scale_2d = local_unscale.view(torch.float8_e4m3fn).reshape(
            padded_M, num_scale_cols
        )[:M, :]
    else:
        scale_2d = local_unscale.reshape(padded_M, num_scale_cols)[:M, :]

    if (
        config.scale_dtype == ScaleDtype.FP8
        and config.global_scale_granularity == GlobalScaleGranularity.ROWWISE
    ):
        assert two_level_scale is not None
        two_level_scale = two_level_scale[:M]

    # Restore original batch dimensions
    if len(orig_shape) > 2:
        packed_data = packed_data.reshape(*orig_shape[:-1], packed_data.shape[-1])
        scale_2d = scale_2d.reshape(*orig_shape[:-1], num_scale_cols)
        if (
            config.scale_dtype == ScaleDtype.FP8
            and config.global_scale_granularity == GlobalScaleGranularity.ROWWISE
        ):
            assert two_level_scale is not None
            two_level_scale = two_level_scale.reshape(*orig_shape[:-1])
    elif len(orig_shape) == 1:
        packed_data = packed_data.squeeze(0)
        scale_2d = scale_2d.squeeze(0)

    return NativeQuantResult(
        data=packed_data,
        scale=scale_2d,
        config=config,
        global_scale=two_level_scale,
        orig_shape=orig_shape,
    )


def _validate_native_quant_config(config: QuantConfig) -> None:
    """Validate that config is supported for native quantization."""
    if config.quant_mode != QuantMode.SYMM:
        raise NotImplementedError(
            f"Native quantization only supports SYMM mode, got {config.quant_mode}"
        )
    if config.elem_dtype not in (ElemDtype.E2M1, ElemDtype.E4M3):
        # TODO: Add MXFP6/E2M3 support once native packing and a reference
        # format are defined for native_quantize()/native_dequantize().
        raise NotImplementedError(
            f"Native quantization supports E2M1 (FP4) and E4M3 (FP8), got {config.elem_dtype}"
        )
    if config.scale_dtype == ScaleDtype.FP8:
        if config.elem_dtype != ElemDtype.E2M1:
            raise NotImplementedError(
                f"NVFP4 only supports E2M1 element dtype, got {config.elem_dtype}"
            )
        if config.global_scale_granularity not in (
            GlobalScaleGranularity.TENSORWISE,
            GlobalScaleGranularity.ROWWISE,
        ):
            raise NotImplementedError(
                "NVFP4 requires TENSORWISE or ROWWISE global_scale_granularity, "
                f"got {config.global_scale_granularity}"
            )
    elif config.scale_dtype != ScaleDtype.E8M0:
        raise NotImplementedError(
            f"Native quantization only supports E8M0 or FP8 scale dtype, "
            f"got {config.scale_dtype}"
        )
    if config.block_size.m != 1:
        raise NotImplementedError(
            f"Native quantization only supports 1xN block sizes, got {config.block_size}"
        )
    if config.global_scale_granularity.is_mbs():
        raise NotImplementedError(
            "Native quantization does not support MBS global_scale_granularity, got "
            f"mbs_block_size={config.get_mbs_block_size()} and "
            f"mbs_dynamic={config.mbs_dynamic}"
        )


def _mx_quantize_pure_pytorch(
    tensor: torch.Tensor,
    config: QuantConfig,
) -> NativeQuantResult:
    """Pure PyTorch native quantization (no Triton kernels).

    Performs reshape → amax → scale → element rounding → pack using
    standard PyTorch ops.
    """
    orig_shape = tensor.shape
    orig_dtype = tensor.dtype

    m = config.block_size.m  # always 1 for native quant
    n = config.block_size.n  # 16 or 32

    # Step 1: Reshape to 2D and pad
    tensor_2d = _reshape_2d_tensor(tensor)
    M, N = tensor_2d.shape

    if M % m != 0 or N % n != 0:
        padded_M = ((M + m - 1) // m) * m
        padded_N = ((N + n - 1) // n) * n
        padded_tensor = torch.zeros(
            padded_M, padded_N, device=tensor.device, dtype=orig_dtype
        )
        padded_tensor[:M, :N] = tensor_2d
    else:
        padded_M, padded_N = M, N
        padded_tensor = tensor_2d

    padded_shape = (padded_M, padded_N)

    # Step 2: Reshape into blocks — (num_blocks, block_size)
    data_hp = _reshape_input(padded_tensor, m, n)
    data_hp = data_hp.to(torch.float32)
    is_mx_fp = config.scale_dtype == ScaleDtype.E8M0 and config.elem_dtype in (
        ElemDtype.E2M1,
        ElemDtype.E4M3,
    )
    if is_mx_fp:
        mx_nan_block = torch.isnan(data_hp).any(dim=1)
        data_hp = torch.clamp(
            data_hp,
            min=-torch.finfo(torch.float32).max,
            max=torch.finfo(torch.float32).max,
        )
    # NVFP4 (FP8 scale) has no NaN sentinel encoding, so NaN inputs are not
    # clamped here — NaN propagates naturally through the arithmetic and
    # fp32_to_fpx. The scale side is handled by calculate_fp8_scale_for_nvfp4's
    # amax_invalid masking.  The equivalent fake-quant path preserves NaN in
    # dequant_data via torch.where at the output stage.

    # Step 3: Compute per-block max
    max_abs_value = torch.amax(torch.abs(data_hp), 1)  # (num_blocks,)

    # Step 4-5: Compute scales
    _, max_pos = ELEM_DTYPE_RANGE_MAP[QuantMode.SYMM][config.elem_dtype]

    norm_exp = 0
    global_scale = None
    if config.scale_dtype == ScaleDtype.FP8:
        num_blocks_row = padded_M // m
        num_blocks_col = padded_N // n
        if config.global_scale_granularity == GlobalScaleGranularity.TENSORWISE:
            global_amax = torch.amax(max_abs_value)
        else:
            global_amax = _chunked_abs_amax(padded_tensor, config.block_size)

        scale, _, local_unscale_fp8, two_level_scale = calculate_fp8_scale_for_nvfp4(
            global_amax=global_amax,
            local_amax=max_abs_value,
            max_pos=max_pos,
            global_scale_granularity=config.global_scale_granularity,
            num_blocks_row=num_blocks_row,
            num_blocks_col=num_blocks_col,
            scale_round_mode=config.scale_round_mode,
        )
        global_scale = two_level_scale
    else:
        # Correctly rounded, as in qdq() and the kernel.
        if config.scale_norm_mode == ScaleNormMode.AMAX:
            unscale = _div_rn(max_abs_value, max_pos)
        elif config.scale_norm_mode == ScaleNormMode.POW2MAX:
            max_pow2 = elem_dtype_max_pow2(config.quant_mode, config.elem_dtype)
            norm_exp = pow2max_norm_exp(max_pow2)
            unscale = (
                max_abs_value if norm_exp != 0 else _div_rn(max_abs_value, max_pow2)
            )
        else:
            raise ValueError(f"Unsupported scale norm mode {config.scale_norm_mode}")

        scale, unscale_float = fp32_to_e8m0(
            unscale,
            elem_dtype=config.elem_dtype,
            scale_round_mode=config.scale_round_mode,
            norm_exp=norm_exp,
        )
        if is_mx_fp:
            unscale_float = unscale_float.masked_fill(mx_nan_block, float("inf"))

    # Step 6: Scale data and round elements
    if config.scale_dtype == ScaleDtype.FP8:
        local_unscale_fp32 = local_unscale_fp8.to(torch.float32)
        global_scale_fp32 = global_scale.to(torch.float32)
        if config.global_scale_granularity == GlobalScaleGranularity.ROWWISE:
            local_unscale_fp32 = local_unscale_fp32.reshape(-1)
            global_scale_fp32 = global_scale_fp32.expand(-1, num_blocks_col).reshape(-1)
        local_unscale_fp32 = torch.where(
            (local_unscale_fp32 == 0)
            | torch.isinf(local_unscale_fp32)
            | torch.isnan(local_unscale_fp32),
            1.0,
            local_unscale_fp32,
        )
        global_scale_fp32 = torch.where(
            (global_scale_fp32 == 0)
            | torch.isinf(global_scale_fp32)
            | torch.isnan(global_scale_fp32),
            1.0,
            global_scale_fp32,
        )
        data_lp = (data_hp * global_scale_fp32.reshape(-1, 1)) / (
            local_unscale_fp32.reshape(-1, 1)
        )
    else:
        data_lp = data_hp * scale.unsqueeze(1)

    if config.elem_round_mode.value == "stochastic":
        data_lp = stochastic_rounding(data_lp, config.elem_dtype)

    data_lp = fp32_to_fpx(data_lp, elem_dtype=config.elem_dtype, max_pos=max_pos)
    if is_mx_fp:
        nan_code_val = float("nan") if config.elem_dtype == ElemDtype.E4M3 else max_pos
        data_lp = data_lp.masked_fill(mx_nan_block.unsqueeze(1), nan_code_val)

    # Step 7: Reshape quantized data back to 2D
    data_lp_2d = _reshape_input_back(data_lp, m, n, padded_shape)
    data_lp_2d = data_lp_2d[:M, :N]

    # Step 8: Reshape scales back to 2D — (num_blocks,) → (padded_M//m, padded_N//n)
    num_blocks_h = padded_M // m
    num_blocks_w = padded_N // n
    if config.scale_dtype == ScaleDtype.FP8:
        scale_2d = local_unscale_fp8.reshape(num_blocks_h, num_blocks_w)
    else:
        scale_2d = unscale_float.reshape(num_blocks_h, num_blocks_w)
    scale_2d = scale_2d[:M, :]  # trim padding rows (m=1 so no block row grouping)
    if (
        config.scale_dtype == ScaleDtype.FP8
        and config.global_scale_granularity == GlobalScaleGranularity.ROWWISE
    ):
        assert global_scale is not None
        if global_scale.dim() > 1 and global_scale.shape[-1] == 1:
            global_scale = global_scale.squeeze(-1)
        global_scale = global_scale[:M]

    # Restore original batch dimensions for data and scales
    if len(orig_shape) > 2:
        data_shape = orig_shape[:-1] + (N,)
        data_lp_2d = data_lp_2d.reshape(data_shape)
        scale_shape = orig_shape[:-1] + (num_blocks_w,)
        scale_2d = scale_2d.reshape(scale_shape)
        if (
            config.scale_dtype == ScaleDtype.FP8
            and config.global_scale_granularity == GlobalScaleGranularity.ROWWISE
        ):
            assert global_scale is not None
            global_scale = global_scale.reshape(*orig_shape[:-1])
    elif len(orig_shape) == 1:
        data_lp_2d = data_lp_2d.squeeze(0)
        scale_2d = scale_2d.squeeze(0)

    # Step 9: Pack to hardware format
    if config.elem_dtype == ElemDtype.E2M1:
        packed_data = float_e2m1_to_packed_uint8(data_lp_2d)
    elif config.elem_dtype == ElemDtype.E4M3:
        packed_data = data_lp_2d.to(torch.float8_e4m3fn)

    if config.scale_dtype == ScaleDtype.FP8:
        packed_scale = scale_2d
    else:
        packed_scale = float_to_e8m0_uint8(scale_2d)

    return NativeQuantResult(
        data=packed_data,
        scale=packed_scale,
        config=config,
        global_scale=global_scale,
        orig_shape=orig_shape,
    )


_NATIVE_E2M1_MIN_CAPABILITY: Tuple[int, int] = (10, 0)


def _check_native_e2m1_capability(
    tensor: torch.Tensor, tensor_name: str, config: QuantConfig
) -> None:
    """Raise if the fused Triton E2M1 packing cannot run on ``tensor``'s device.

    The packing uses the ``cvt.rn.satfinite.e2m1x2.f32`` PTX instruction, which
    requires SM100 or newer.
    """
    if config.elem_dtype != ElemDtype.E2M1 or not tensor.is_cuda:
        return
    capability = torch.cuda.get_device_capability(tensor.device)
    if capability < _NATIVE_E2M1_MIN_CAPABILITY:
        required = "SM{}{}".format(*_NATIVE_E2M1_MIN_CAPABILITY)
        found = "SM{}{}".format(*capability)
        raise RuntimeError(
            f"Native E2M1 packing with use_triton=True requires {required} or "
            f"newer, but tensor {tensor_name!r} is on {tensor.device} ({found}). "
            "Use use_triton=False on this device."
        )


def native_quantize(
    tensor: torch.Tensor,
    tensor_name: str,
    config: QuantConfig,
    is_sf_swizzled_layout: bool = False,
    use_triton: bool = True,
) -> NativeQuantResult:
    """Quantize tensor to native low-precision format using QuantConfig.

    Supports MXFP4/MXFP8 with E8M0 single-level scaling and NVFP4 with
    FP8 E4M3 two-level scaling.

    Args:
        tensor: Input tensor (bf16 or fp32), any shape. Last dim is quantized.
        tensor_name: Name of the tensor (for debugging/logging).
        config: QuantConfig specifying scale rounding mode, element dtype, etc.
        is_sf_swizzled_layout: If True, output scales use the 128x4 blocked
            (swizzled) layout matching the flashinfer Tensor Core format.
            This is only applied to MX formats with E8M0 scales. NVFP4 scales
            remain in row-major FP8 E4M3 layout.
        use_triton: If True, use a single fused Triton mega kernel for quant+pack.
            With E2M1 elements this requires an SM100 or newer CUDA device.

    Returns:
        NativeQuantResult with packed data and native scales.

    Raises:
        NotImplementedError: If ``config`` is not a supported native format,
            including any config with an MBS ``global_scale_granularity``, or
            if ``is_sf_swizzled_layout`` is True for a result that is not 2D.
        RuntimeError: If ``use_triton`` is True, the elements are E2M1 and
            ``tensor`` is on a CUDA device below SM100.
    """
    # Validate supported configurations (shared by both paths)
    _validate_native_quant_config(config)

    # Dispatch to implementation
    if use_triton:
        _check_native_e2m1_capability(tensor, tensor_name, config)
        result = _mx_quantize_fused_triton(tensor, config)
    else:
        result = _mx_quantize_pure_pytorch(tensor, config)

    # Optionally swizzle MX E8M0 scales to 128x4 blocked layout.
    if is_sf_swizzled_layout and result.config.scale_dtype != ScaleDtype.FP8:
        if result.scale.ndim != 2:
            raise NotImplementedError(
                "Swizzled scale layout is only supported for 2D native quant "
                f"results, got scale shape {result.scale.shape} for tensor "
                f"{tensor_name!r}"
            )
        result = NativeQuantResult(
            data=result.data,
            scale=_swizzle_scale(result.scale),
            config=result.config,
            global_scale=result.global_scale,
            orig_shape=result.orig_shape,
        )

    return result


def native_dequantize(
    result: NativeQuantResult,
    output_dtype: torch.dtype = torch.bfloat16,
    is_sf_swizzled_layout: bool = False,
) -> torch.Tensor:
    """Dequantize a NativeQuantResult back to float.

    Performs ``output = unpack(data) * e8m0_to_float(scale)`` for MX formats and
    ``output = unpack(data) * fp8_to_float(scale) / global_scale`` for NVFP4.

    Args:
        result: Output from native_quantize().
        output_dtype: Desired output dtype (default: bfloat16).
        is_sf_swizzled_layout: If True, result.scale is interpreted as the
            flattened 128x4 blocked layout produced by native_quantize(...,
            is_sf_swizzled_layout=True), and is unswizzled before use. This
            layout is currently supported only for 2D native quant results.

    Returns:
        Dequantized tensor in the original shape.
    """
    # Unpack data to float
    if result.config.elem_dtype == ElemDtype.E2M1:
        data_float = packed_uint8_to_float_e2m1(result.data)
    elif result.config.elem_dtype == ElemDtype.E4M3:
        data_float = result.data.to(torch.float32)
    else:
        raise NotImplementedError(f"Unsupported elem dtype {result.config.elem_dtype}")

    # Apply scales: each scale covers block_size_n elements
    block_size = result.config.block_size.n
    data_shape = data_float.shape
    K = data_float.shape[-1]
    num_groups = K // block_size

    scale = result.scale
    if is_sf_swizzled_layout:
        if result.config.scale_dtype == ScaleDtype.FP8:
            raise ValueError(
                "is_sf_swizzled_layout=True is not supported for FP8 scale dtype "
                "(NVFP4). native_quantize skips swizzling for FP8 scales, so there is "
                "nothing to unswizzle."
            )
        if len(data_shape) != 2:
            raise ValueError(
                "Swizzled scale layout is only supported for 2D native quant "
                f"results, got data shape {data_shape}"
            )
        if scale.ndim != 1:
            raise ValueError(
                "Expected flattened 1D scale when is_sf_swizzled_layout=True, "
                f"got scale shape {scale.shape}"
            )

        scale_rows = data_shape[0]
        scale = _unswizzle_scale(scale, scale_rows, num_groups)

    # Convert scales to float. MX formats use E8M0; NVFP4 uses FP8 E4M3
    # block scales plus a tensorwise/rowwise global quant scale.
    if result.config.scale_dtype == ScaleDtype.FP8:
        if result.global_scale is None:
            raise ValueError("NVFP4 dequantization requires global_scale")
        scale_float = scale.to(torch.float64)
        global_scale = result.global_scale.to(
            device=scale_float.device,
        )
        if global_scale.numel() == 1:
            scale_float = (scale_float / global_scale.reshape(())).to(torch.float32)
        else:
            while global_scale.dim() < scale_float.dim():
                global_scale = global_scale.unsqueeze(-1)
            scale_float = (scale_float / global_scale).to(torch.float32)
    else:
        scale_float = e8m0_uint8_to_float(scale)

    # Reshape for broadcasting: data [..., num_groups, block_size] * scale [..., num_groups, 1]
    data_blocked = data_float.reshape(*data_shape[:-1], num_groups, block_size)
    scale_expanded = scale_float.unsqueeze(-1)  # [..., num_groups, 1]

    dequantized = (data_blocked * scale_expanded).reshape(data_shape)

    return dequantized.to(output_dtype)


def native_data_shape(
    param_shape: torch.Size,
    elem_dtype: ElemDtype,
) -> torch.Size:
    """Expected checkpoint data shape for a native MX weight with the given param shape."""
    if elem_dtype == ElemDtype.E2M1:
        packed_shape = list(param_shape)
        packed_shape[-1] = (packed_shape[-1] + 1) // 2
        return torch.Size(packed_shape)
    if elem_dtype == ElemDtype.E4M3:
        return param_shape
    raise NotImplementedError(f"Unsupported native MX elem dtype {elem_dtype}")


def native_dtype_matches(tensor: torch.Tensor, elem_dtype: ElemDtype) -> bool:
    """Check whether a tensor's dtype matches the expected native MX element dtype."""
    if elem_dtype == ElemDtype.E2M1:
        return tensor.dtype == torch.uint8
    if elem_dtype == ElemDtype.E4M3:
        return tensor.dtype == torch.float8_e4m3fn
    raise NotImplementedError(f"Unsupported native MX elem dtype {elem_dtype}")
