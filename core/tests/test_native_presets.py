# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

from quant_suite_core.native_presets import NATIVE_PRESETS


class NativePresetsTest(unittest.TestCase):
    def test_presets(self) -> None:
        self.assertEqual(
            NATIVE_PRESETS,
            {
                "nvfp4": {"precision": "nvfp4", "scale_round_mode": "nearest"},
                "mxfp4_ocp": {"precision": "mx4-32", "scale_round_mode": "floor"},
                "mxfp4_16_floor": {
                    "precision": "mx4-16",
                    "scale_round_mode": "floor",
                },
                "mxfp4_32_oas": {
                    "precision": "mx4-32",
                    "scale_round_mode": "even",
                },
                "mxfp4_16_oas": {
                    "precision": "mx4-16",
                    "scale_round_mode": "even",
                },
            },
        )
