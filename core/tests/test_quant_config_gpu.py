# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from quant_suite_core import QuantConfig
from quant_suite_core.fake_presets import FAKE_PRESETS
from quant_suite_core.fake_quant import is_triton_mega_supported
from quant_suite_core.qdq_pytorch import qdq
from quant_suite_core.qdq_triton import qdq_triton_mega


class TestMegaDynamicMbsSearch(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU for Triton")
    def test_mega_search_never_worse_than_eager(self) -> None:
        """The fused search must not regress the objective on any macro block.

        Deliberately not a bitwise check. The kernel reduces SSE in a different order
        than the eager path, so when two candidates are near-tied the argmin can land
        on either -- on a real weight that happens for roughly one macro block in
        70k. What must hold is the thing the search optimises: per-block SSE is never
        higher than the eager search's beyond fp32 reduction noise. A genuinely wrong
        candidate evaluation would show up here as a tail of worse blocks, which
        aggregate SQNR would average away.
        """
        cfg = QuantConfig.from_dict(
            {
                **FAKE_PRESETS["mxfp4"],
                "global_scale_granularity": "mbs256",
                "global_scale_dtype": "e0m8",
                "mbs_dynamic": "mbs_dynamic",
            }
        )
        self.assertTrue(is_triton_mega_supported(cfg))
        torch.manual_seed(0)
        x = (torch.randn(512, 512, dtype=torch.bfloat16) * 0.05).cuda()
        ref = x.float()

        def block_sse(got: torch.Tensor) -> torch.Tensor:
            return (got.float() - ref).pow(2).reshape(-1, 256).sum(dim=1)

        eager = block_sse(qdq(x, "w", cfg))
        mega = block_sse(qdq_triton_mega(x, "w", cfg))
        rel = (mega - eager) / eager.clamp_min(1e-30)
        self.assertLess(float(rel.max()), 1e-5)


if __name__ == "__main__":
    unittest.main()
