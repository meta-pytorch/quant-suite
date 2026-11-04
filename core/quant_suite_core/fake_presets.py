# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fake quantization presets.

Each preset maps a short name (e.g., "mxfp4", "nvfp4", "mxfp6") to a dict
of QuantConfig fields. Covers MXFP4/6/8, NVFP4, MBS, and stochastic rounding
variants. The MX presets default to 1x16 blocks; the ``_32`` and ``_ocp``
variants use the 1x32 OCP block size. Tensor transforms such as RHT belong in
recipe pipelines, not in quantization presets.

This module has no dependencies, so it imports without torch or triton.
"""

from typing import Any

# Common field sets to reduce repetition
_NVFP4_BASE = {
    "elem_dtype": "e2m1",
    "elem_round_mode": "nearest",
    "block_size": "1x16",
    "scale_dtype": "fp8",
    "scale_norm_mode": "amax",
    "scale_round_mode": "none",
    "global_scale_granularity": "tensorwise",
    "global_scale_dtype": "fp32",
}

_MXFP4_BASE = {
    "elem_dtype": "e2m1",
    "elem_round_mode": "nearest",
    "block_size": "1x16",
    "scale_dtype": "e8m0",
    "scale_norm_mode": "pow2max",
    "scale_round_mode": "oas",
}

_MXFP6_BASE = {
    "elem_dtype": "e2m3",
    "elem_round_mode": "nearest",
    "block_size": "1x16",
    "scale_dtype": "e8m0",
    "scale_norm_mode": "pow2max",
    "scale_round_mode": "oas",
}

_MXFP8_BASE = {
    "elem_dtype": "e4m3",
    "elem_round_mode": "nearest",
    "block_size": "1x16",
    "scale_dtype": "e8m0",
    "scale_norm_mode": "pow2max",
    "scale_round_mode": "oas_s",
}

# Identical to _MXFP6_BASE except the element dtype is int8 (integer 8-bit).
_MXINT8_BASE = {
    "elem_dtype": "int8",
    "elem_round_mode": "nearest",
    "block_size": "1x16",
    "scale_dtype": "e8m0",
    "scale_norm_mode": "pow2max",
    "scale_round_mode": "oas",
}

_MBS128 = {"global_scale_granularity": "mbs128", "global_scale_dtype": "e0m8"}
_MBS256 = {"global_scale_granularity": "mbs256", "global_scale_dtype": "e0m8"}
_MBS16 = {"global_scale_granularity": "mbs16", "global_scale_dtype": "e0m8"}
_MBS_DYN = {"mbs_dynamic": True}
_SR = {"elem_round_mode": "stochastic"}
_16x16 = {"block_size": "16x16"}


def _preset(base: dict, *extras: dict) -> dict[str, Any]:
    result = {**base}
    for e in extras:
        result.update(e)
    return result


FAKE_PRESETS: dict[str, dict[str, Any]] = {
    # =========================================================================
    # NVFP4 — E2M1 with FP8 two-level scaling
    # =========================================================================
    "nvfp4": _preset(_NVFP4_BASE),
    "rowwise_nvfp4": _preset(_NVFP4_BASE, {"global_scale_granularity": "rowwise"}),
    "nvfp4_sr": _preset(_NVFP4_BASE, _SR),
    "nvfp4_16x16": _preset(_NVFP4_BASE, _16x16),
    # =========================================================================
    # MXFP4 — E2M1 with E8M0 scaling
    # =========================================================================
    "mxfp4": _preset(_MXFP4_BASE),
    "mxfp4_32": _preset(_MXFP4_BASE, {"block_size": "1x32"}),
    "mxfp4_sr": _preset(_MXFP4_BASE, _SR),
    "mxfp4_16x16": _preset(_MXFP4_BASE, _16x16),
    # MXFP4 + MBS128
    "mxfp4_mbs128": _preset(_MXFP4_BASE, _MBS128),
    "mxfp4_mbs128_sr": _preset(_MXFP4_BASE, _MBS128, _SR),
    # MXFP4 + MBS16
    "mxfp4_mbs16": _preset(_MXFP4_BASE, _MBS16),
    "mxfp4_mbs16_sr": _preset(_MXFP4_BASE, _MBS16, _SR),
    # MXFP4 + MBS256, with the static and the dynamic (searched) macro factor.
    "mxfp4_mbs256": _preset(_MXFP4_BASE, _MBS256),
    "mxfp4_mbs256_dyn": _preset(_MXFP4_BASE, _MBS256, _MBS_DYN),
    # =========================================================================
    # MXFP4 OCP — 1x32 block size
    # =========================================================================
    "mxfp4_ocp": _preset(
        _MXFP4_BASE, {"block_size": "1x32", "scale_round_mode": "floor"}
    ),
    "mxfp4_ocp_ceil": _preset(
        _MXFP4_BASE, {"block_size": "1x32", "scale_round_mode": "ceil"}
    ),
    # =========================================================================
    # MXFP6 — E2M3 with E8M0 scaling
    # =========================================================================
    "mxfp6": _preset(_MXFP6_BASE),
    "mxfp6_sr": _preset(_MXFP6_BASE, _SR),
    # =========================================================================
    # MXFP8 — E4M3 with E8M0 scaling
    # =========================================================================
    "mxfp8": _preset(_MXFP8_BASE),
    "mxfp8_32": _preset(_MXFP8_BASE, {"block_size": "1x32"}),
    "mxfp8_sr": _preset(_MXFP8_BASE, _SR),
    "mxfp8_ocp": _preset(
        _MXFP8_BASE, {"block_size": "1x32", "scale_round_mode": "floor"}
    ),
    "mxfp8_ocp_ceil": _preset(
        _MXFP8_BASE, {"block_size": "1x32", "scale_round_mode": "ceil"}
    ),
    # =========================================================================
    # MXINT8 — INT8 elements with E8M0 scaling
    # =========================================================================
    "mxint8": _preset(_MXINT8_BASE),
}
