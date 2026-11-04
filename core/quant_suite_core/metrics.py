# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantization error metrics."""

import torch


def cal_sqnr(
    x: torch.Tensor,
    x_ref: torch.Tensor,
    qsnr_inf_val: float = 1000.0,
    allow_negative: bool = False,
) -> float:
    """Compute SQNR (signal-to-quantization-noise ratio) in dB.

    Args:
        x: The dequantized tensor.
        x_ref: The reference tensor.
        qsnr_inf_val: Value returned when x and x_ref are bit-identical
            (SQNR = +Inf).
        allow_negative: If True, allow a negative (or -inf) SQNR instead of
            raising ``ValueError``.

    Returns:
        SQNR in dB.

    Raises:
        ValueError: If the SQNR is negative and ``allow_negative`` is False.
    """
    x_ref = x_ref.to(torch.double)
    x = x.to(torch.double)

    error = x_ref - x
    signal = x_ref * x_ref
    error_power = error * error
    # Bit-identical tensors: return a fixed large value as a placeholder for +Inf
    # (default 1000.0) to keep downstream logging/aggregation/plotting stable.
    if error_power.sum() == 0.0:
        return qsnr_inf_val
    if signal.sum() == 0.0:
        if allow_negative:
            return float("-inf")
        raise ValueError(
            "SQNR is -inf: x_ref is all zeros but x is not. "
            "Pass allow_negative=True to accept negative SQNR."
        )
    ratio = signal.mean() / error_power.mean()
    sqnr_db = 10.0 * torch.log10(ratio).item()
    # A negative SQNR usually means a broken dequant, so keep it as a sanity check
    # by default. It is legitimate in a few cases, so those callers opt out rather
    # than the check being dropped everywhere.
    if not allow_negative and sqnr_db < 0.0:
        raise ValueError(f"sqnr should be positive. Got: {sqnr_db}")
    return sqnr_db
