# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""qdq()'s trace-safe branches agree bitwise with its eager branches, on CUDA.

The CUDA counterpart of the CPU sweep in test_qdq_trace_branch_equivalence,
plus the qdq_triton_mega cases. Requires a CUDA GPU; skipped otherwise.

Set QUANT_SUITE_BRANCH_FUZZ_SCALE=<k> to draw k times as many random shapes.
"""

from __future__ import annotations

import unittest

import torch
from qdq_branch_equivalence_utils import (
    BranchEquivalenceTestCase,
    cases,
    EAGER_CALLS,
    MEGA_SHAPES,
)
from quant_suite_core.qdq_triton import qdq_triton_mega


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class QdqTraceBranchEquivalenceGpuTest(BranchEquivalenceTestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda", torch.cuda.current_device())

    def test_qdq_and_fake_quantize_branches_agree_bitwise(self) -> None:
        self.sweep(cases(("qdq", "fake_quantize")), self.device, "qdq+fake_quantize")

    def test_qdq_triton_mega_branches_agree_bitwise(self) -> None:
        # The forced branch only reaches the shared padding helpers here, and
        # every new shape costs Triton compiles, so a few shapes suffice.
        mega_cases = (
            c._replace(entry="qdq_triton_mega") for c in cases(("qdq",), MEGA_SHAPES)
        )
        self.sweep(mega_cases, self.device, "qdq_triton_mega")

    def test_zero_width_last_dim_raises_on_both_branches(self) -> None:
        calls = dict(EAGER_CALLS)
        calls["qdq_triton_mega"] = lambda x, cfg: qdq_triton_mega(x, "activation", cfg)
        self.check_zero_width_raises(calls, self.device)
