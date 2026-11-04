# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

from quant_suite_core import ElemRoundMode, QuantConfig
from quant_suite_core.fake_presets import FAKE_PRESETS
from quant_suite_core.qdq_triton import _mega_block_m
from quant_suite_core.quant_primitives import (
    _TRITON_MAX_BLOCK_NUMEL_PER_QUANT_BLOCK,
    TRITON_MAX_BLOCK_NUMEL,
    TRITON_MEGA_BLOCK_M,
)


def _config(preset: str, **overrides: str) -> QuantConfig:
    return QuantConfig.from_dict({**FAKE_PRESETS[preset], **overrides})


class MegaBlockMTest(unittest.TestCase):
    def test_block_shapes(self) -> None:
        for block_size, expected in (("1x16", 128), ("1x32", 64), ("16x16", 8)):
            with self.subTest(block_size=block_size):
                self.assertEqual(
                    _mega_block_m(_config("mxfp4", block_size=block_size)), expected
                )

    def test_tile_dependent_presets_keep_the_cap(self) -> None:
        # Stochastic rounding and mbs_dynamic outputs depend on BLOCK_M.
        names = [
            name
            for name in FAKE_PRESETS
            if _config(name).elem_round_mode == ElemRoundMode.STOCHASTIC
            or _config(name).mbs_dynamic
        ]
        self.assertTrue(names)
        for name in names:
            with self.subTest(preset=name):
                self.assertEqual(_mega_block_m(_config(name)), TRITON_MEGA_BLOCK_M)

    def test_mbs_never_below_span(self) -> None:
        configs = [
            _config(name)
            for name in FAKE_PRESETS
            if _config(name).global_scale_granularity.is_mbs()
        ]
        self.assertTrue(configs)
        # MBS requires block_size.m == 1.
        for n in (16, 32, 64, 128, 256, 512):
            for mbs in (16, 32, 64, 128, 256, 512):
                if mbs % n == 0:
                    configs.append(
                        _config(
                            "mxfp4_mbs128",
                            block_size=f"1x{n}",
                            global_scale_granularity=f"mbs{mbs}",
                        )
                    )
        for config in configs:
            span = config.get_mbs_block_size() // config.block_size.n
            with self.subTest(config=config):
                block_m = _mega_block_m(config)
                self.assertGreaterEqual(block_m, span)
                self.assertEqual(block_m % span, 0)

    def test_tile_within_numel_cap(self) -> None:
        sizes = [1 << i for i in range(14)]
        for m in sizes:
            for n in sizes:
                if m * n > _TRITON_MAX_BLOCK_NUMEL_PER_QUANT_BLOCK:
                    continue
                with self.subTest(block_size=f"{m}x{n}"):
                    block_m = _mega_block_m(_config("mxfp4", block_size=f"{m}x{n}"))
                    self.assertEqual(block_m & (block_m - 1), 0)
                    self.assertLessEqual(block_m, TRITON_MEGA_BLOCK_M)
                    self.assertLessEqual(block_m * m * n, TRITON_MAX_BLOCK_NUMEL)


if __name__ == "__main__":
    unittest.main()
