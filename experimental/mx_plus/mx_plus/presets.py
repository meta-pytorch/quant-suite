# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fake-quantization presets that use ``MxPlus``.

Plain specs with no imports, so a preset table can be read without torch.
"""

from typing import Any, Dict

MX_PLUS_PRESETS: Dict[str, Dict[str, Any]] = {
    # MXFP4 + MX+ (2 extra mantissa bits for the amax element)
    "mxfp4_plus2": {
        "quant_mode": "symm",
        "elem_dtype": "e2m1",
        "elem_round_mode": "nearest",
        "block_size": "1x32",
        "scale_dtype": "e8m0",
        "scale_norm_mode": "pow2max",
        "scale_round_mode": "floor",
        "global_scale_granularity": "none",
        "global_scale_dtype": "none",
        "extensions": {"mx_plus": {"extra_mantissa_bits": 2}},
    },
}
