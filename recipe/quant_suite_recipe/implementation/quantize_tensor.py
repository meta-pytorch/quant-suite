# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import math
from typing import Any, cast

import torch
from quant_suite_core import fake_quantize, QuantInfo
from quant_suite_core.native_presets import NativeRoundingMode, PrecisionType
from quant_suite_core.quant_primitives import _nvfp4_global_scale_or_one
from quant_suite_recipe.api.definition import QuantTensorRecipe
from quant_suite_recipe.api.implementation import QuantizedTensor
from quant_suite_recipe.definition.quant_recipe import (
    FakeQuantRecipe,
    NativeQuantRecipe,
    NoQuantRecipe,
)


_SUPPORTED_TENSOR_DTYPES: tuple[torch.dtype, ...] = (
    torch.bfloat16,
    torch.float16,
    torch.float32,
)
_NATIVE_MX4_TENSOR_DTYPES: tuple[torch.dtype, ...] = (torch.bfloat16, torch.float16)


def _check_supported_tensor_dtype(tensor: torch.Tensor) -> None:
    if tensor.dtype not in _SUPPORTED_TENSOR_DTYPES:
        raise ValueError(
            "Only bfloat16, float16 and float32 tensors are supported, "
            f"got {tensor.dtype}"
        )


def _from_blocked(
    tensor: torch.Tensor,
    shape: tuple[int, int],
) -> torch.Tensor:
    from mslk.quantize.triton.fp4_quantize import _from_blocked as mslk_from_blocked

    return cast(torch.Tensor, mslk_from_blocked(tensor, shape))


def triton_quantize_mx4_unpack(
    tensor: torch.Tensor,
    *,
    group_size: int,
    rounding_mode: NativeRoundingMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    from mslk.quantize.triton.fp4_primitives.constants import (
        RoundingMode as MslkRoundingMode,
    )
    from mslk.quantize.triton.fp4_quantize import (
        triton_quantize_mx4_unpack as mslk_quantize_mx4_unpack,
    )

    return cast(
        tuple[torch.Tensor, torch.Tensor],
        mslk_quantize_mx4_unpack(
            tensor,
            group_size=group_size,
            rounding_mode=MslkRoundingMode[rounding_mode.value],
        ),
    )


def triton_quantize_nvfp4(
    tensor: torch.Tensor,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from mslk.quantize.triton.fp4_quantize import (
        triton_quantize_nvfp4 as mslk_quantize_nvfp4,
    )

    return cast(
        tuple[torch.Tensor, torch.Tensor],
        mslk_quantize_nvfp4(
            tensor,
            global_scale,
        ),
    )


# -----------------------------------------------------------------------------
# Helper functions for quantization and dequantization
# -----------------------------------------------------------------------------


def fp4_to_float(x: torch.Tensor) -> torch.Tensor:
    # Handle Float4_e2m1fn_x2 dtype by viewing as uint8 for bitwise ops.
    if x.dtype not in (torch.uint8, torch.int8):
        x = x.view(torch.uint8)
    # Start by unpacking the FP4 values into separate integers.
    low_mx4 = torch.bitwise_and(x, 0xF)
    high_mx4 = torch.bitwise_and(x >> 4, 0xF)
    comb_shape = x.shape[:-1] + (x.shape[-1] * 2,)
    x_comb = torch.stack([low_mx4, high_mx4], dim=-1).flatten(-2).to(torch.int32)
    # Map to float with a lookup table.
    E2M1_LUT = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
        device=x.device,
    )
    return torch.index_select(E2M1_LUT, 0, x_comb.view(-1)).view(comb_shape)


def scale_mx4(x: torch.Tensor, exp: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    # Input should be dequantized to float, just need to apply shared exponent.
    FP32_EXP_BIAS = 127
    scale = torch.exp2(exp.view(torch.uint8).to(torch.int32) - FP32_EXP_BIAS)
    # View input as chunks of group size.
    orig_shape = x.shape
    num_groups = orig_shape[-1] // group_size
    scaled_x = (
        x.view(-1, num_groups, group_size)
        * scale.view(x.shape[0], -1)[:, :num_groups, None]
    )
    return scaled_x.view(orig_shape)


def scale_nvfp4(
    x: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
    group_size: int = 16,
) -> torch.Tensor:
    # NVFP4 uses a trick where global scales are folded into the scales
    # but not x itself. Those scales are normally removed in the epilogue.
    # Here, we manually get the scaling for x by removing global components.
    true_scale = scale.view(torch.float8_e4m3fn).to(torch.float) / global_scale
    # Now we can reverse scaling of x.
    num_groups = x.shape[-1] // group_size
    scaled_x = (
        x.view(-1, num_groups, group_size)
        * true_scale.view(x.shape[0], -1)[:, :num_groups, None]
    )
    return scaled_x.view(x.shape)


# -----------------------------------------------------------------------------
# Quantization and dequantization functions
# -----------------------------------------------------------------------------


def no_quantize_tensor(
    tensor: torch.Tensor,
    tensor_name: str,
    recipe: QuantTensorRecipe,
    transpose: bool,
    return_quant_info: bool,
) -> QuantizedTensor:
    _check_supported_tensor_dtype(tensor)
    assert isinstance(recipe, NoQuantRecipe)
    return QuantizedTensor(tensor=tensor)


def fake_quantize_tensor(
    tensor: torch.Tensor,
    tensor_name: str,
    recipe: QuantTensorRecipe,
    transpose: bool,
    return_quant_info: bool,
) -> QuantizedTensor:
    _check_supported_tensor_dtype(tensor)
    assert isinstance(recipe, FakeQuantRecipe)
    use_triton = recipe.use_triton
    if use_triton is None:
        use_triton = True
    result = fake_quantize(
        tensor=tensor,
        tensor_name=tensor_name,
        config=recipe.quant_config,
        transpose=transpose,
        use_triton=use_triton,
        return_quant_info=return_quant_info,
    )
    if return_quant_info:
        assert isinstance(result, QuantInfo)
        return QuantizedTensor(
            tensor=result.dequant_data,
            scale=result.local_unscale,
            global_scale=result.global_scale,
            quantized_val=result.quantized_val,
        )
    assert isinstance(result, torch.Tensor)
    return QuantizedTensor(tensor=result)


def native_quantize_tensor(
    tensor: torch.Tensor,
    tensor_name: str,
    recipe: QuantTensorRecipe,
    transpose: bool,
    return_quant_info: bool,
) -> QuantizedTensor:
    """Quantize ``tensor`` with a native kernel along its last dimension.

    The result always holds the packed quantized tensor with its ``scale`` and,
    for NVFP4, its ``global_scale``, so ``return_quant_info`` adds nothing.
    ``transpose=True`` raises ``NotImplementedError``.
    """
    _check_supported_tensor_dtype(tensor)
    assert isinstance(recipe, NativeQuantRecipe)
    if transpose:
        raise NotImplementedError("Native quantization does not support transpose=True")
    precision = recipe.precision

    if precision == PrecisionType.NVFP4:
        global_amax = torch.max(torch.abs(tensor)).float()
        global_scale: torch.Tensor = (448.0 * 6.0) / global_amax
        # A positive amax whose scale overflows fp32 gets the largest finite one.
        global_scale = torch.where(
            global_amax > 0,
            global_scale.clamp(max=torch.finfo(torch.float32).max),
            global_scale,
        )
        # An all-zero tensor gets 1.0.
        global_scale = _nvfp4_global_scale_or_one(global_amax, global_scale)
        xq, x_scale = triton_quantize_nvfp4(tensor, global_scale)
        return QuantizedTensor(
            tensor=xq,
            scale=x_scale,
            global_scale=global_scale,
            precision=precision,
        )

    if precision in (PrecisionType.MX4_32, PrecisionType.MX4_16):
        if tensor.dtype not in _NATIVE_MX4_TENSOR_DTYPES:
            raise NotImplementedError(
                f"Native {precision.value} quantization requires a bfloat16 or "
                f"float16 tensor, got {tensor.dtype}"
            )
        group_size = 32 if precision == PrecisionType.MX4_32 else 16
        xq, x_scale = triton_quantize_mx4_unpack(
            tensor,
            group_size=group_size,
            rounding_mode=recipe.scale_round_mode,
        )
        return QuantizedTensor(tensor=xq, scale=x_scale, precision=precision)

    raise NotImplementedError(f"Precision {precision} is not supported")


def dequantize_tensor(
    quantized_tensor: QuantizedTensor,
    orig_shape: tuple[int, int] | None = None,
    orig_dtype: torch.dtype | None = None,
) -> Any:
    tensor = quantized_tensor.tensor
    precision = quantized_tensor.precision
    if precision is None or tensor is None:
        return tensor

    else:
        scale = quantized_tensor.scale
        assert scale is not None, "scale must be provided for Native quantization"
        assert orig_shape is not None, (
            "orig_shape must be provided for Native quantization"
        )
        assert orig_dtype is not None, (
            "orig_dtype must be provided for Native quantization"
        )
        M, N = orig_shape
        if precision in (PrecisionType.MX4_32, PrecisionType.MX4_16):
            group_size = 32 if precision == PrecisionType.MX4_32 else 16
            num_groups = math.ceil(N / group_size)
            # Convert blocked scale format back to (M, num_groups) layout.
            scale = _from_blocked(scale, (M, num_groups))
            xq_float = fp4_to_float(tensor)
            xq_dequant = scale_mx4(xq_float, scale, group_size).to(orig_dtype)
        elif precision == PrecisionType.NVFP4:
            global_scale = quantized_tensor.global_scale
            assert global_scale is not None, (
                f"global_scale must be provided for {precision.name.lower()}"
            )
            group_size = 16
            num_groups = math.ceil(N / group_size)
            # Convert blocked scale format back to (M, num_groups) layout.
            scale = _from_blocked(scale, (M, num_groups))
            xq_float = fp4_to_float(tensor)
            xq_dequant = scale_nvfp4(xq_float, scale, global_scale, group_size).to(
                orig_dtype
            )
        else:
            raise NotImplementedError(f"Precision {precision} is not supported")

        return xq_dequant
