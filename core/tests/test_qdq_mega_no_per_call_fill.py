# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``qdq_triton_mega`` launches one kernel and nothing else for non-FP8 scales.

The kernel reads ``global_amax`` only for FP8 (NVFP4) scales, and writes the
quantized elements and per-block unscales only with ``return_quant_info``.
Otherwise the launcher passes no tensor for them. A call allocates only its
output and launches only the QDQ kernel; no extra fill nodes are recorded in a
CUDA graph.
"""

import unittest
from functools import partial
from unittest.mock import patch

import torch
from quant_suite_core import QuantConfig
from quant_suite_core.fake_presets import FAKE_PRESETS
from quant_suite_core.qdq_triton import _FusedScalePhase, qdq_triton_mega
from torch.profiler import profile, ProfilerActivity

NON_FP8_PRESETS = ("mxfp4", "mxfp6", "mxfp8", "mxfp4_mbs128", "mxint8")


def _config(preset: str) -> QuantConfig:
    return QuantConfig.from_dict(FAKE_PRESETS[preset])


def _cuda_kernels(fn) -> list[str]:
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]


@unittest.skipIf(not torch.cuda.is_available(), "Skip when cuda is not available")
class NoPerCallFillTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.x = torch.randn(512, 3072, device="cuda", dtype=torch.bfloat16) * 3

    def test_non_fp8_launch_is_one_kernel_and_one_allocation(self) -> None:
        for preset in NON_FP8_PRESETS:
            with self.subTest(preset=preset):
                config = _config(preset)
                qdq_triton_mega(self.x, "a", config)  # compile and warm the allocator
                kernels = _cuda_kernels(partial(qdq_triton_mega, self.x, "a", config))
                self.assertEqual(len(kernels), 1, kernels)
                self.assertIn("_mega_fake_quant_triton_kernel", kernels[0])
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                out = qdq_triton_mega(self.x, "a", config)
                after = torch.cuda.memory_stats()["allocation.all.allocated"]
                self.assertEqual(after - before, 1)  # the output only
                del out

    def test_output_is_invariant_to_unit_global_amax_for_non_fp8_scales(self) -> None:
        # The kernel does not read global_amax for these scale dtypes, so
        # substituting torch.ones(1) must not change the output.
        derive = _FusedScalePhase.derive

        def derive_with_unit_amax(phase: _FusedScalePhase):
            scale = derive(phase)
            if scale.global_amax is not None:
                return scale
            return scale._replace(global_amax=torch.ones(1, device=phase.tensor.device))

        for preset in NON_FP8_PRESETS:
            with self.subTest(preset=preset):
                config = _config(preset)
                result = qdq_triton_mega(self.x, "a", config)
                with patch.object(_FusedScalePhase, "derive", derive_with_unit_amax):
                    reference = qdq_triton_mega(self.x, "a", config)
                self.assertTrue(
                    torch.equal(result.view(torch.int16), reference.view(torch.int16))
                )

    def test_return_quant_info_allocates_its_buffers(self) -> None:
        config = _config("mxfp4")
        plain = qdq_triton_mega(self.x, "a", config)
        info = qdq_triton_mega(self.x, "a", config, return_quant_info=True)
        self.assertTrue(
            torch.equal(info.dequant_data.view(torch.int16), plain.view(torch.int16))
        )
        self.assertEqual(info.quantized_val.shape, self.x.shape)
        self.assertEqual(
            info.local_unscale.numel(), self.x.numel() // config.block_size.n
        )

    def test_fp8_scales_pass_global_amax(self) -> None:
        for preset in ("nvfp4", "rowwise_nvfp4"):
            with self.subTest(preset=preset):
                config = _config(preset)
                scale = config._derive_scale(
                    _FusedScalePhase(
                        config=config,
                        tensor=self.x,
                        padded_tensor=self.x,
                        M=self.x.shape[0],
                        N=self.x.shape[1],
                        m=config.block_size.m,
                        n=config.block_size.n,
                        new_M=self.x.numel() // config.block_size.n,
                        local_max_pos=6.0,
                    )
                )
                self.assertIsNotNone(scale.global_amax)

    def test_cuda_graph_replay_matches_eager(self) -> None:
        config = _config("mxfp4")
        reference = qdq_triton_mega(self.x, "a", config)
        static_in = self.x.clone()
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            qdq_triton_mega(static_in, "a", config)
        torch.cuda.current_stream().wait_stream(warmup)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = qdq_triton_mega(static_in, "a", config)
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out.view(torch.int16), reference.view(torch.int16)))


if __name__ == "__main__":
    unittest.main()
