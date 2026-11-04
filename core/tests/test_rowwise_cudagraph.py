# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""CUDA-graph capture of the NVFP4 `two_level_scale`.

The scale numerator must be built on the device (`global_amax.new_full((1,),
...)`), not materialized from host data with `torch.tensor([elem_format_max],
...)`, which is illegal during CUDA-graph capture. Both forms give the same
(1,) shape/dtype/value, so the f32/f32 division is bit-identical.

Two functions compute this scale independently, so both are covered:
- `calculate_fp8_scale_for_nvfp4` (used by `qdq`) --
  `RowwiseCudaGraphTest`, `Nvfp4ScaleParityTest`.
- `qdq_triton_mega` (the kernel `fake_quantize(use_triton=True)` dispatches to
  for a 1x16 rowwise NVFP4 config) -- graph replay must be bit-identical to
  eager -- `MegaKernelCudaGraphTest`.
"""

import unittest

import torch
from quant_suite_core import fake_quantize, QuantConfig
from quant_suite_core.qdq_pytorch import qdq

# A 1x16 rowwise NVFP4 config that flows through the ROWWISE
# NVFP4 scale path.
ROWWISE_NVFP4 = {
    "elem_dtype": "e2m1",
    "elem_round_mode": "nearest",
    "block_size": "1x16",
    "scale_dtype": "fp8",
    "scale_norm_mode": "amax",
    "scale_round_mode": "none",
    "global_scale_granularity": "rowwise",
    "global_scale_dtype": "fp32",
}

# Same config with a tensorwise global scale (0-dim global_amax), which shares
# the same scale expression.
TENSORWISE_NVFP4 = {**ROWWISE_NVFP4, "global_scale_granularity": "tensorwise"}


def _capture_and_replay(fn, x: torch.Tensor) -> torch.Tensor:
    """Capture fn(static_in) into a CUDA graph, replay on x, return the output."""
    static_in = x.clone()
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            fn(static_in)
    torch.cuda.current_stream().wait_stream(warmup)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = fn(static_in)

    static_in.copy_(x)
    g.replay()
    torch.cuda.synchronize()
    return static_out


@unittest.skipIf(not torch.cuda.is_available(), "Skip when cuda is not available")
class RowwiseCudaGraphTest(unittest.TestCase):
    """calculate_fp8_scale_for_nvfp4 via qdq (use_triton=False)."""

    def setUp(self) -> None:
        self.device = "cuda"
        torch.manual_seed(0)

    def test_rowwise_qdq_cudagraph_capture_and_parity(self) -> None:
        cfg = QuantConfig.from_dict(ROWWISE_NVFP4)
        x = torch.randn(128, 512, device=self.device, dtype=torch.bfloat16)
        y_ref = qdq(x, "k", cfg).clone()
        y_graph = _capture_and_replay(lambda t: qdq(t, "k", cfg), x)
        self.assertTrue(
            torch.equal(y_graph.float(), y_ref.float()),
            "graph replay diverged from eager rowwise QDQ",
        )

    def test_host_tensor_numerator_breaks_capture(self) -> None:
        # Materializing a device tensor from host data during capture is illegal
        # and raises.
        global_amax = torch.rand(8, device=self.device).float() + 0.1
        g = torch.cuda.CUDAGraph()
        with self.assertRaises(RuntimeError):
            with torch.cuda.graph(g):
                _ = 448.0 * (
                    torch.tensor([6.0], dtype=torch.float32, device=global_amax.device)
                    / global_amax
                ).to(torch.float64)


@unittest.skipIf(not torch.cuda.is_available(), "Skip when cuda is not available")
class MegaKernelCudaGraphTest(unittest.TestCase):
    """qdq_triton_mega -- the kernel a 1x16 rowwise NVFP4 config dispatches to
    (fake_quantize(use_triton=True), is_triton_mega_supported=True). Verify
    it captures and that graph replay is bit-identical to eager."""

    def setUp(self) -> None:
        self.device = "cuda"
        torch.manual_seed(0)

    def _run(self, config: dict) -> None:
        cfg = QuantConfig.from_dict(config)
        x = torch.randn(128, 512, device=self.device, dtype=torch.bfloat16)

        def mega(t: torch.Tensor) -> torch.Tensor:
            # block_size 1x16 -> is_triton_mega_supported -> qdq_triton_mega
            return fake_quantize(t, "k", cfg, use_triton=True)

        eager = mega(x).clone()
        graphed = _capture_and_replay(mega, x)
        self.assertTrue(
            torch.equal(graphed.float(), eager.float()),
            "mega-kernel graph replay diverged from eager (capture != eager)",
        )
        self.assertTrue(torch.isfinite(eager.float()).all())

    def test_rowwise_mega_cudagraph_parity(self) -> None:
        self._run(ROWWISE_NVFP4)

    def test_tensorwise_mega_cudagraph_parity(self) -> None:
        self._run(TENSORWISE_NVFP4)


# e2m1 max_pos and the fp8 scale-format max, as used in the two_level_scale exprs.
_ELEM_FORMAT_MAX = 6.0
_SCALE_FORMAT_MAX = 448.0


def _host_tensor_two_level_scale(global_amax: torch.Tensor) -> torch.Tensor:
    # Numerator materialized from a host scalar; not capture-safe.
    return _SCALE_FORMAT_MAX * (
        torch.tensor([_ELEM_FORMAT_MAX], dtype=torch.float32, device=global_amax.device)
        / global_amax
    ).to(torch.float64)


def _device_fill_two_level_scale(global_amax: torch.Tensor) -> torch.Tensor:
    # MUST mirror `calculate_fp8_scale_for_nvfp4` in quant_primitives.py: new_full
    # has the same (1,) shape/dtype/value as torch.tensor([...]), so the f32/f32
    # division is bit-identical to _host_tensor_two_level_scale. (A python-scalar
    # numerator would round ~1 ULP differently and is NOT used.)
    return _SCALE_FORMAT_MAX * (
        global_amax.new_full((1,), _ELEM_FORMAT_MAX, dtype=torch.float32) / global_amax
    ).to(torch.float64)


@unittest.skipIf(not torch.cuda.is_available(), "Skip when cuda is not available")
class Nvfp4ScaleParityTest(unittest.TestCase):
    """The device-fill numerator gives the same two_level_scale value as the
    host-tensor one, for tensorwise (0-dim global_amax) and rowwise."""

    def setUp(self) -> None:
        self.device = "cuda"
        torch.manual_seed(0)

    def test_two_level_scale_expr_matches_host_tensor(self) -> None:
        cases = {
            "tensorwise_fp32": torch.rand((), device=self.device) + 0.05,
            "tensorwise_bf16": (torch.rand((), device=self.device) + 0.05).to(
                torch.bfloat16
            ),
            "rowwise_fp32": torch.rand(37, device=self.device) + 0.05,
            "rowwise_bf16": (torch.rand(37, device=self.device) + 0.05).to(
                torch.bfloat16
            ),
        }
        for name, global_amax in cases.items():
            host = _host_tensor_two_level_scale(global_amax)
            device = _device_fill_two_level_scale(global_amax)
            self.assertTrue(
                torch.equal(host.flatten(), device.flatten()),
                f"{name}: device-fill two_level_scale differs from host-tensor",
            )

    def test_tensorwise_qdq_end_to_end(self) -> None:
        # The tensorwise fake-QDQ path runs and is deterministic; a fresh call reproduces bit-identical output.
        cfg = QuantConfig.from_dict(TENSORWISE_NVFP4)
        x = torch.randn(128, 512, device=self.device, dtype=torch.bfloat16)
        y1 = qdq(x, "k", cfg)
        y2 = qdq(x, "k", cfg)
        self.assertTrue(torch.equal(y1.float(), y2.float()))
        self.assertTrue(torch.isfinite(y1.float()).all())
        # It really quantized (lossy) rather than passing the input through.
        self.assertFalse(torch.equal(y1.float(), x.float()))
