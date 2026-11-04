# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Packing/unpacking utilities for MX format tensors.

Converts between float-domain quantized values and hardware-packed formats:
- FP4 E2M1: float ↔ packed uint8 (2 values per byte)
- E8M0 scales: float (power-of-2) ↔ uint8 biased exponent
"""

import torch

# E2M1 lookup table: index → float value
# Matches the encoding used by flashinfer
# Low nibble (bits 3:0) = first value, high nibble (bits 7:4) = second value
E2M1_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)

# E8M0 exponent bias
E8M0_EXPONENT_BIAS = 127


def float_e2m1_to_packed_uint8(tensor: torch.Tensor) -> torch.Tensor:
    """Convert float E2M1 values to packed uint8 (2 values per byte).

    Args:
        tensor: float tensor [..., K] with values from E2M1 set
                {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}

    Returns:
        uint8 tensor [..., K//2] with two FP4 values packed per byte.
        Packing: low nibble = even index, high nibble = odd index.
        This matches the flashinfer fp4_quantize output format.
    """
    assert tensor.shape[-1] % 2 == 0, f"Last dim must be even, got {tensor.shape[-1]}"

    # Build reverse LUT on the correct device (float → 4-bit index)
    lut = E2M1_LUT.to(device=tensor.device)

    # Vectorized lookup: find the index in LUT for each float value by a
    # bit-exact int32 comparison against all 16 entries, which also tells
    # -0.0 from 0.0.
    flat = tensor.float().reshape(-1)
    flat_bits = flat.view(torch.int32)
    lut_bits = lut.view(torch.int32)

    matches = flat_bits.unsqueeze(1) == lut_bits.unsqueeze(0)  # [N, 16]
    indices = matches.int().argmax(dim=1).to(torch.uint8)  # [N]

    # Reshape back to original shape
    indices = indices.reshape(tensor.shape)

    # Pack pairs: low nibble = even index, high nibble = odd index
    even = indices[..., 0::2]  # low nibble
    odd = indices[..., 1::2]  # high nibble
    packed = even | (odd << 4)

    return packed


def packed_uint8_to_float_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 packed FP4 E2M1 to float tensor.

    Args:
        packed: uint8 tensor [..., K//2] with two FP4 values per byte.

    Returns:
        float32 tensor [..., K] with E2M1 float values.
    """
    lut = E2M1_LUT.to(device=packed.device)

    low = packed & 0x0F  # even indices (low nibble)
    high = (packed >> 4) & 0x0F  # odd indices (high nibble)

    # Interleave: [low0, high0, low1, high1, ...]
    out_shape = packed.shape[:-1] + (packed.shape[-1] * 2,)
    interleaved = torch.stack([low, high], dim=-1).reshape(out_shape)

    # LUT lookup
    return torch.index_select(lut, 0, interleaved.reshape(-1).to(torch.int64)).reshape(
        out_shape
    )


def float_to_e8m0_uint8(unscale: torch.Tensor) -> torch.Tensor:
    """Convert float E8M0 unscale values to uint8 biased exponents.

    E8M0 format: value = 2^(biased_exp - 127), stored as uint8.

    Args:
        unscale: float tensor of power-of-2 scale values (output of fp32_to_e8m0).

    Returns:
        uint8 tensor of biased exponents.
    """
    abs_unscale = torch.abs(unscale).float()

    # Extract biased exponent from the float bit representation
    # For power-of-2 values, this is exact
    fp32_bits = abs_unscale.view(torch.int32)
    biased_exp = (fp32_bits >> 23) & 0xFF

    # Handle special cases
    # Zero → biased_exp 0 (2^-127, the smallest E8M0 value)
    biased_exp = torch.where(abs_unscale == 0, torch.zeros_like(biased_exp), biased_exp)
    # NaN → 0xFF (E8M0 NaN encoding)
    biased_exp = torch.where(
        torch.isnan(unscale), torch.full_like(biased_exp, 0xFF), biased_exp
    )

    return biased_exp.to(torch.uint8)


def _e8m0_value(exponent: torch.Tensor) -> torch.Tensor:
    """``2**(exponent - 127)`` for biased E8M0 exponents in [0, 255], as fp32.

    Built from the fp32 bits, so it is exact on every backend; ``torch.exp2`` and
    ``torch.pow`` are not (CUDA ``exp2(-127.)`` is ``0x003FFFFF``, one subnormal
    ULP short, and ``pow(2.0, k)`` misses some ``k`` on other GPUs). Exponent 0
    is the fp32 subnormal ``2**-127`` and 255 is ``+Inf``.
    """
    bits = torch.where(exponent == 0, 0x00400000, exponent * (1 << 23))
    # aten.view.dtype, not Tensor.view: some graph tracers rewrite a traced
    # Tensor.view into a reshape.
    return torch.ops.aten.view.dtype(bits.to(torch.int32), torch.float32)


def e8m0_uint8_to_float(scale: torch.Tensor) -> torch.Tensor:
    """Convert uint8 E8M0 biased exponents to float scale values.

    Args:
        scale: uint8 tensor of biased exponents.

    Returns:
        float32 tensor of power-of-2 scale values: 2^(biased_exp - 127).
    """
    biased_exp = scale.to(torch.int32)
    # Handle NaN (0xFF)
    is_nan = biased_exp == 0xFF
    float_val = _e8m0_value(biased_exp)
    float_val = torch.where(
        is_nan, torch.tensor(float("nan"), device=scale.device), float_val
    )
    return float_val
