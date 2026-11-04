# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fake-quant dispatch: the ``fake_quantize`` entry point and the grouped-NVFP4 MoE
path.

Supports a range of quantization strategies, not only OCP MX -- the element
format, scale dtype, quant mode, block shape and global-scale granularity are all
independently configurable.
"""

from dataclasses import dataclass
from typing import Literal, Optional, overload, Union

import torch
from quant_suite_core.qdq_pytorch import qdq
from quant_suite_core.qdq_triton import qdq_triton_mega
from quant_suite_core.quant_config import (
    ELEM_DTYPE_BITS_MAP,
    ELEM_DTYPE_RANGE_MAP,
    ElemDtype,
    F32_MIN_NORMAL,
    GlobalScaleGranularity,
    QuantConfig,
    QuantInfo,
    QuantMode,
    ScaleDtype,
    ScaleNormMode,
)
from quant_suite_core.quant_primitives import (
    _is_fx_proxy,
    _prepare_2d_padded,
    _restore_2d_padded,
    nvfp4_global_scale,
)
from quant_suite_core.triton_utils import (
    _exceeds_triton_block_numel,
    _group_counts_to_offsets_and_zero_kernel,
    _grouped_nvfp4_amax_triton_kernel,
    _grouped_nvfp4_fake_quant_triton_kernel,
    _stochastic_rounding_params,
)
from triton import cdiv  # @manual


def is_grouped_nvfp4_triton_supported(config: QuantConfig) -> bool:
    """Return whether grouped fake QDQ can use the MoE NVFP4 Triton fast path."""
    return (
        config.quant_mode == QuantMode.SYMM
        and config.scale_dtype == ScaleDtype.FP8
        and config.scale_norm_mode == ScaleNormMode.AMAX
        # The grouped kernel derives the scale itself and calls none of the
        # config's hooks, so a config with any extension would be quantized as
        # if it had none.
        and not config.extensions
        and config.global_scale_granularity == GlobalScaleGranularity.TENSORWISE
        and config.elem_dtype == ElemDtype.E2M1
        and config.block_size.m == 1
    )


def _grouped_nvfp4_counts(
    group_counts: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    counts = torch.as_tensor(group_counts, device=device, dtype=torch.int32)
    if counts.ndim != 1:
        raise ValueError(f"group_counts must be 1D, got shape {counts.shape}")
    if counts.numel() <= 0:
        raise ValueError("group_counts must contain at least one group")
    return counts.contiguous()


def fake_quantize_grouped_nvfp4(
    tensor: torch.Tensor,
    tensor_name: str,
    config: QuantConfig,
    group_counts: torch.Tensor,
    out: torch.Tensor | None = None,
    global_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fake-quantize a 2D tensor with tensorwise NVFP4 scales per row group.

    Rows are assumed to be laid out contiguously by group. ``group_counts`` gives
    the row count for each group and is consumed on device, so the runtime path
    avoids synchronizing routed MoE token counts back to the CPU. By default,
    each group derives its global scale dynamically. ``global_scale`` may supply
    one calibrated scale for all groups or one scale per group instead.

    Args:
        tensor (torch.Tensor): The 2D CUDA tensor to fake-quantize.
        tensor_name (str): Name of the tensor (for error messages).
        config (QuantConfig): A tensorwise NVFP4 configuration.
        group_counts (torch.Tensor): 1D row count per group.
        out (torch.Tensor | None): Optional output buffer of ``tensor``'s shape.
        global_scale (torch.Tensor | None): Optional calibrated global scale,
            scalar or one per group.

    Returns:
        torch.Tensor: The fake-quantized tensor (``out`` when given).
    """
    if not is_grouped_nvfp4_triton_supported(config):
        raise NotImplementedError(f"unsupported grouped NVFP4 config: {config}")
    if tensor.ndim != 2:
        raise ValueError(
            f"grouped NVFP4 QDQ expects a 2D tensor, got {tensor.shape} "
            f"for tensor {tensor_name!r}"
        )
    if not tensor.is_cuda:
        raise NotImplementedError(
            f"grouped NVFP4 QDQ requires a CUDA tensor, got {tensor.device} "
            f"for tensor {tensor_name!r}"
        )
    if out is not None and out.shape != tensor.shape:
        raise ValueError(
            f"out shape {out.shape} must match tensor shape {tensor.shape} "
            f"for tensor {tensor_name!r}"
        )
    if tensor.numel() == 0:
        if out is None:
            return tensor.clone()
        out.copy_(tensor)
        return out

    m = config.block_size.m
    n = config.block_size.n
    padded_tensor, M, N, orig_shape, orig_dtype = _prepare_2d_padded(
        tensor, m, n, transpose=False
    )
    padded_M, padded_N = padded_tensor.shape
    new_N = m * n
    new_M = padded_M * padded_N // new_N

    counts = _grouped_nvfp4_counts(group_counts, tensor.device)
    E = counts.numel()
    BLOCK_E = 1 << (E - 1).bit_length()

    group_offsets = torch.empty((E,), device=tensor.device, dtype=torch.int32)
    group_amax = torch.empty((E,), device=tensor.device, dtype=torch.float32)
    _group_counts_to_offsets_and_zero_kernel[(1,)](
        counts,
        group_offsets,
        group_amax,
        E=E,
        BLOCK_E=BLOCK_E,
    )

    group_two_level_scale, max_pos, use_static_global_scale = (
        _grouped_nvfp4_global_scale(
            padded_tensor,
            group_offsets,
            group_amax,
            config,
            E,
            BLOCK_E,
            global_scale,
        )
    )

    needs_restore = padded_M != M or padded_N != N
    can_write_direct = out is not None and not needs_restore
    dequant_data = (
        out
        if can_write_direct
        else torch.empty(padded_tensor.shape, device=tensor.device, dtype=orig_dtype)
    )

    ebits, mbits = ELEM_DTYPE_BITS_MAP[config.elem_dtype]
    min_exp = -(2 ** (ebits - 1)) + 2
    seed, max_normal, min_normal = _stochastic_rounding_params(
        config, tensor.device, max_pos
    )

    def qdq_grid(META):
        return (cdiv(new_M, META["BLOCK_M"]),)

    _grouped_nvfp4_fake_quant_triton_kernel[qdq_grid](
        padded_tensor,
        dequant_data,
        group_offsets,
        group_amax,
        group_two_level_scale,
        seed,
        M=padded_M,
        N=padded_N,
        new_M=new_M,
        new_N=new_N,
        stride_in0=padded_tensor.stride(0),
        stride_in1=padded_tensor.stride(1),
        stride_out0=dequant_data.stride(0),
        stride_out1=dequant_data.stride(1),
        n=n,
        min_exp=min_exp,
        ebits=ebits,
        mbits=mbits,
        max_normal=max_normal,
        min_normal=min_normal,
        max_pos=max_pos,
        scale_round_mode=config.scale_round_mode.value,
        elem_round_mode=config.elem_round_mode.value,
        E=E,
        BLOCK_M=128,
        BLOCK_N=new_N,
        BLOCK_E=BLOCK_E,
        F32_MIN_NORMAL=F32_MIN_NORMAL,
        use_static_global_scale=use_static_global_scale,
    )

    if can_write_direct:
        return out

    dequant_data = _restore_2d_padded(dequant_data, M, N, False, orig_shape)
    if out is None:
        return dequant_data
    out.copy_(dequant_data)
    return out


def _grouped_nvfp4_global_scale(
    padded_tensor: torch.Tensor,
    group_offsets: torch.Tensor,
    group_amax: torch.Tensor,
    config: QuantConfig,
    num_groups: int,
    block_groups: int,
    global_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, float, bool]:
    _, max_pos = ELEM_DTYPE_RANGE_MAP[config.quant_mode][config.elem_dtype]
    if global_scale is not None:
        scales = torch.as_tensor(
            global_scale,
            device=padded_tensor.device,
            dtype=config.global_scale_dtype.to_torch_dtype(),
        )
        if scales.numel() == 1:
            scales = scales.reshape(1).expand(num_groups)
        elif scales.shape != (num_groups,):
            raise ValueError(
                "global_scale must be scalar or have one value per group, got "
                f"shape {scales.shape} for {num_groups} groups"
            )
        return scales.contiguous(), max_pos, True

    def amax_grid(META):
        return (
            cdiv(padded_tensor.shape[0], META["BLOCK_M"]),
            cdiv(padded_tensor.shape[1], META["BLOCK_N"]),
        )

    _grouped_nvfp4_amax_triton_kernel[amax_grid](
        padded_tensor,
        group_offsets,
        group_amax,
        M=padded_tensor.shape[0],
        N=padded_tensor.shape[1],
        stride_in0=padded_tensor.stride(0),
        stride_in1=padded_tensor.stride(1),
        E=num_groups,
        BLOCK_M=16,
        BLOCK_N=1024,
        BLOCK_E=block_groups,
    )
    scales = nvfp4_global_scale(group_amax, max_pos)
    return scales, max_pos, False


def is_triton_mega_supported(config: QuantConfig) -> bool:
    """Whether ``qdq_triton_mega`` implements this config.

    ``fake_quantize`` does not consult this -- the mega kernel is the only
    Triton backend, and it raises on a config it cannot serve. Callers that need
    to know *before* launching ask here.
    """
    if _exceeds_triton_block_numel(config):
        # The block size overflows the Triton tile; `qdq_triton_mega` raises on a
        # direct call.
        return False
    # The in-kernel candidate search reproduces the e8m0 scale derivation only, so
    # any other scale dtype would be ranked against the wrong quantizer.
    return not (config.mbs_dynamic and config.scale_dtype != ScaleDtype.E8M0)


@dataclass
class _DispatchResult:
    """``fake_quantize``'s state for ``QuantConfig._finalize``.

    Hides whether the call is returning a bare dequant tensor or a ``QuantInfo``
    wrapping one, so a step reads and writes ``dequantized`` either way.
    """

    source: torch.Tensor
    result: Union[torch.Tensor, QuantInfo]

    @property
    def dequantized(self) -> torch.Tensor:
        if isinstance(self.result, QuantInfo):
            return self.result.dequant_data
        return self.result

    def with_dequantized(
        self, dequantized: torch.Tensor
    ) -> Union[torch.Tensor, QuantInfo]:
        if isinstance(self.result, QuantInfo):
            self.result.dequant_data = dequantized
            return self.result
        return dequantized


@overload
def fake_quantize(
    tensor: torch.Tensor,
    tensor_name: str,
    config: QuantConfig,
    transpose: bool = ...,
    use_triton: bool = ...,
    return_quant_info: Literal[False] = ...,
    *,
    num_valid_rows: Optional[torch.Tensor] = ...,
) -> torch.Tensor: ...


@overload
def fake_quantize(
    tensor: torch.Tensor,
    tensor_name: str,
    config: QuantConfig,
    transpose: bool,
    use_triton: bool,
    return_quant_info: Literal[True],
    *,
    num_valid_rows: Optional[torch.Tensor] = ...,
) -> QuantInfo: ...


@overload
def fake_quantize(
    tensor: torch.Tensor,
    tensor_name: str,
    config: QuantConfig,
    transpose: bool = ...,
    use_triton: bool = ...,
    *,
    return_quant_info: Literal[True],
    num_valid_rows: Optional[torch.Tensor] = ...,
) -> QuantInfo: ...


@overload
def fake_quantize(
    tensor: torch.Tensor,
    tensor_name: str,
    config: QuantConfig,
    transpose: bool = ...,
    use_triton: bool = ...,
    return_quant_info: bool = ...,
    *,
    num_valid_rows: Optional[torch.Tensor] = ...,
) -> Union[torch.Tensor, QuantInfo]: ...


def fake_quantize(
    tensor: torch.Tensor,
    tensor_name: str,
    config: QuantConfig,
    transpose: bool = False,
    use_triton: bool = True,
    return_quant_info: bool = False,
    *,
    num_valid_rows: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, QuantInfo]:
    """Quantize and dequantize a tensor according to ``config``.

    The output has the input's shape and dtype and holds the values the
    quantized format can represent.

    Args:
        tensor (Tensor): The input tensor (bf16, fp16 or fp32) to be quantized.
        tensor_name (str): Name of the tensor (for debugging/logging).
        config (QuantConfig): The quantization configuration.
        transpose (bool): Whether to transpose the block dimensions.
        use_triton (bool): If True (the default), run ``qdq_triton_mega``, which
            needs a CUDA device and Triton. If False, run the eager ``qdq``
            reference backend, which also runs on CPU.
        return_quant_info (bool): If True, returns QuantInfo with dequant_data,
            quantized_val, local_unscale, and global_scale.
        num_valid_rows (Tensor, optional): 0-d device tensor with the number
            of leading rows that hold data; see ``qdq_triton_mega``. Output
            rows below the bound are bit-identical to a call without it, and
            the rest are unspecified. The eager backend ignores it and
            quantizes every row, which meets the same contract.

    Returns:
        Tensor: The quantized tensor (or QuantInfo if return_quant_info=True).
    """
    if not (
        _is_fx_proxy(tensor)
        or tensor.dtype in (torch.bfloat16, torch.float16, torch.float)
    ):
        raise ValueError(
            f"tensor dtype {tensor.dtype} is not supported; expected bfloat16, "
            "float16 or float32"
        )

    if use_triton:
        q_tensor = qdq_triton_mega(
            tensor,
            tensor_name,
            config,
            transpose,
            return_quant_info,
            num_valid_rows=num_valid_rows,
        )
    else:
        q_tensor = qdq(
            tensor,
            tensor_name,
            config,
            transpose,
            return_quant_info,
        )

    # `_finalize`, not `_post_process`: that hook fires inside the backend, before
    # it unwinds padding and the transpose, and once per backend call -- `qdq` is
    # re-entrant (`_mbs_factor_dynamic`) and both backends are also public entry
    # points that can be called directly. A dispatch-level step needs the
    # restored shape and exactly one firing.
    return config._finalize(_DispatchResult(source=tensor, result=q_tensor))
