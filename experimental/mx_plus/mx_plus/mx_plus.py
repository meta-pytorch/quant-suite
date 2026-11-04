# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# `_safe_lshift`, `_safe_rshift` and the rounding in `_mx_plus_amax_lp` are
# adapted from mx/elemwise_ops.py in Microsoft microxcaling
# (https://github.com/microsoft/microxcaling), licensed under the MIT License.
# Copyright (c) Microsoft Corporation.
# See THIRD_PARTY_NOTICES.md in the root directory of this package.

"""MX+: keep extra mantissa bits on the amax element of each block.

A post-process rule, so it reaches both QDQ backends through
``MxPlus.post_process`` and the ``PostProcessPhase`` a backend hands it.
One implementation serves both: the phase owns the block view, the per-block
scale and the scatter, which is where the two backends differ, and this module
owns the rule, which is where they do not.
"""

from typing import Tuple

import torch
from quant_suite_core.extension import PostProcessPhase
from quant_suite_core.quant_config import ELEM_DTYPE_BITS_MAP, QuantConfig

# The largest ``extra_mantissa_bits`` accepted, and the range the rule is
# validated over.
MAX_EXTRA_MANTISSA_BITS: int = 8


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


def apply_mx_plus(
    config: QuantConfig, phase: PostProcessPhase, extra_mantissa_bits: int
) -> torch.Tensor:
    """Re-quantize each block's amax element at extra precision and write it back."""
    data_hp = phase.block_view()
    amax_indices, amax_lp = _mx_plus_amax_lp(
        data_hp,
        phase.block_scale(),
        config,
        extra_mantissa_bits,
        phase.min_pos,
        phase.max_pos,
    )
    return phase.scatter_block_elements(amax_indices, amax_lp)


def _mx_plus_amax_lp(
    data_hp: torch.Tensor,
    scale: torch.Tensor,
    config: QuantConfig,
    extra_mantissa_bits: int,
    min_pos: float,
    max_pos: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Re-quantize each block's amax element with extra mantissa bits (MX+).

    `fp32_to_fpx` rounding carried out at `base_mbits + extra_mantissa_bits` rather than
    `base_mbits`, on the single largest-magnitude element of each block.

    Neither Triton kernel has a branch for MX+, so both backends run it as
    this Python post-pass. Only the arithmetic here is shared -- the two do NOT
    agree on the inputs their phase supplies. `qdq` passes blocks already clamped
    to +-FLT_MAX and the scale its main path computed; `qdq_triton_mega` passes
    raw blocks and a scale it re-derives, because the kernel does not expose one.
    See the DIVERGENCE note on that phase.

    Args:
        data_hp: High-precision blocks, shape (num_blocks, block_numel).
        scale: Per-block scale, shape (num_blocks,).
        config: The quantization configuration.
        extra_mantissa_bits: Mantissa bits kept beyond the element format's.
        min_pos: Element-format minimum, for saturation.
        max_pos: Element-format maximum, for saturation.

    Returns:
        (amax_indices, amax_lp): each block's amax position within the flattened
        block, and that element's value in the quantized domain. The phase writes
        it into whichever domain it owns.
    """
    ebits, base_mbits = ELEM_DTYPE_BITS_MAP[config.elem_dtype]
    total_mbits = base_mbits + extra_mantissa_bits  # Total mantissa bits for amax

    # Find the index of the amax element in each block
    abs_data_hp = torch.abs(data_hp)
    amax_indices = torch.argmax(abs_data_hp, dim=1)  # (num_blocks,)
    num_blocks = data_hp.shape[0]

    amax_values_hp = data_hp[
        torch.arange(num_blocks, device=data_hp.device), amax_indices
    ]

    # Scale the original values to the quantized domain
    amax_scaled = amax_values_hp * scale

    # Compute private exponent
    private_exp = torch.floor(
        torch.log2(torch.abs(amax_scaled) + (amax_scaled == 0).type(amax_scaled.dtype))
    )
    min_exp = -(2 ** (ebits - 1)) + 2
    private_exp = private_exp.clip(min=min_exp)

    # Scale up so appropriate number of bits are in the integer portion
    amax_scaled_up = _safe_lshift(amax_scaled, total_mbits, private_exp)

    # Round to nearest even (same as fp32_to_fpx)
    absA = torch.abs(amax_scaled_up)
    # find 0.5, 2.5, 4.5 ... for round-to-nearest-even
    maskA = ((absA - 0.5) % 2 == torch.zeros_like(amax_scaled_up)).type(
        amax_scaled_up.dtype
    )
    # copysign keeps -0.0 as -0.0, where torch.sign would return +0.0. absA + 0.5
    # is taken in float64: in float32 a value 1 ULP below a tie (e.g. 0.49999997)
    # rounds up to 1.0 on the add, so floor would move it up a grid step.
    amax_rounded = torch.copysign(torch.ones_like(amax_scaled_up), amax_scaled_up) * (
        torch.floor(absA.to(torch.float64) + 0.5).to(amax_scaled_up.dtype) - maskA
    )

    # Undo scaling
    amax_lp = _safe_rshift(amax_rounded, total_mbits, private_exp)

    # Clamp to valid range
    return amax_indices, torch.clamp(amax_lp, min=min_pos, max=max_pos)
