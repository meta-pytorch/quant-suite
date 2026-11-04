# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""qdq()'s trace-safe branches agree bitwise with its eager branches, on CPU.

The harness and its rationale live in qdq_branch_equivalence_utils. This module
covers the qdq and fake_quantize entry points on CPU only and needs no GPU. The
CUDA and qdq_triton_mega cases are in test_qdq_trace_branch_equivalence_gpu.

Set QUANT_SUITE_BRANCH_FUZZ_SCALE=<k> to draw k times as many random shapes.
"""

from __future__ import annotations

import quant_suite_core.quant_primitives as quant_primitives
import torch
from qdq_branch_equivalence_utils import (
    BranchEquivalenceTestCase,
    cases,
    EAGER_CALLS,
    force_trace_branch,
    importers,
)

_CPU = torch.device("cpu")


class QdqTraceBranchEquivalenceCpuTest(BranchEquivalenceTestCase):
    def test_patch_reaches_every_importer(self) -> None:
        names = {m.__name__ for m in importers()}
        for expected in ("quant_primitives", "qdq_pytorch", "fake_quant"):
            self.assertIn(f"quant_suite_core.{expected}", names)
        # An aligned eager tensor is returned as is; the forced branch always pads.
        x = torch.randn(4, 32)
        self.assertIs(quant_primitives._prepare_2d_padded(x, 1, 32, False)[0], x)
        with force_trace_branch():
            self.assertIsNot(quant_primitives._prepare_2d_padded(x, 1, 32, False)[0], x)

    def test_qdq_and_fake_quantize_branches_agree_bitwise(self) -> None:
        self.sweep(cases(("qdq", "fake_quantize")), _CPU, "qdq+fake_quantize")

    def test_zero_width_last_dim_raises_on_both_branches(self) -> None:
        self.check_zero_width_raises(EAGER_CALLS, _CPU)
