# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``qdq_triton_mega(num_valid_rows=...)``: QDQ only the leading data rows.

An input may hold data only in its leading ``num_valid_rows`` rows. Rows below
that bound must be bit-identical to the full-tensor call, in eager mode and in
a CUDA graph replayed with a new value.
"""

import unittest

import torch
from quant_suite_core import fake_quantize, QuantConfig
from quant_suite_core.fake_presets import FAKE_PRESETS
from quant_suite_core.qdq_triton import qdq_triton_mega

# Rows of width 3072; capacities span small, medium and large leading-row
# buffers.
WIDTH = 3072
CAPACITIES = (4096, 65536, 262144)


def _config(preset: str) -> QuantConfig:
    return QuantConfig.from_dict(FAKE_PRESETS[preset])


def _row_cases(capacity: int) -> list[int]:
    """No rows, a few rows, partial (aligned and unaligned) and full buffers."""
    return [0, 1, 4, 384, capacity // 3, capacity // 2, capacity]


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    # Bitwise comparison, NaN payloads included.
    return tensor.view(torch.int16)


def _input(rows: int, width: int = WIDTH) -> torch.Tensor:
    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16) * 3
    # Outliers and non-finite values stay inside their own rows.
    x[::97, ::31] = 1e4
    x[5::211, 7] = float("nan")
    x[3::401, 11] = float("inf")
    return x


def _rows(value: int, dtype: torch.dtype = torch.int32) -> torch.Tensor:
    return torch.tensor(value, device="cuda", dtype=dtype)


@unittest.skipIf(not torch.cuda.is_available(), "Skip when cuda is not available")
class ValidRowsTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def _check(
        self,
        x: torch.Tensor,
        config: QuantConfig,
        rows: int,
        dtype: torch.dtype = torch.int32,
    ) -> None:
        # Both calls start from one RNG state, so stochastic rounding draws the
        # same seed.
        with torch.random.fork_rng(devices=[x.device]):
            reference = qdq_triton_mega(x, "x", config)
        with torch.random.fork_rng(devices=[x.device]):
            result = qdq_triton_mega(x, "x", config, num_valid_rows=_rows(rows, dtype))
        bound = min(rows, x.shape[0])
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, x.dtype)
        self.assertTrue(torch.equal(_bits(result[:bound]), _bits(reference[:bound])))

    def test_mxfp4_shapes_and_row_counts(self) -> None:
        config = _config("mxfp4")
        for capacity in CAPACITIES:
            x = _input(capacity)
            for rows in _row_cases(capacity):
                with self.subTest(capacity=capacity, rows=rows):
                    self._check(x, config, rows)

    def test_other_row_local_presets(self) -> None:
        # rowwise_nvfp4 derives a per-row global scale outside the kernel;
        # mxfp4_mbs128 and mxfp4_sr depend on the tile; mxfp4_16x16 rounds the
        # bound up to whole 16-row quant blocks.
        x = _input(4096)
        for preset in (
            "mxfp8",
            "rowwise_nvfp4",
            "mxfp4_mbs128",
            "mxfp4_sr",
            "mxfp4_16x16",
        ):
            config = _config(preset)
            for rows in (0, 23, 1701, 4096):
                with self.subTest(preset=preset, rows=rows):
                    self._check(x, config, rows)

    def test_int64_and_bounds_past_the_tensor(self) -> None:
        config = _config("mxfp4")
        x = _input(4096)
        self._check(x, config, 14, dtype=torch.int64)
        # A bound past the tensor quantizes every row.
        self._check(x, config, 8192)

    def test_fake_quantize_threads_the_bound(self) -> None:
        config = _config("mxfp4")
        x = _input(4096)
        rows = _rows(14)
        reference = fake_quantize(x, "x", config)
        result = fake_quantize(x, "x", config, num_valid_rows=rows)
        self.assertTrue(torch.equal(_bits(result[:14]), _bits(reference[:14])))
        # The eager backend ignores the bound and quantizes every row.
        eager = fake_quantize(x, "x", config, use_triton=False)
        eager_bounded = fake_quantize(
            x, "x", config, use_triton=False, num_valid_rows=rows
        )
        self.assertTrue(torch.equal(_bits(eager), _bits(eager_bounded)))

    def test_cuda_graph_replays_with_changing_row_counts(self) -> None:
        config = _config("mxfp4")
        for capacity in (4096, 65536):
            x = _input(capacity)
            reference = qdq_triton_mega(x, "x", config)
            num_valid_rows = _rows(0)
            warmup = torch.cuda.Stream()
            warmup.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup):
                for _ in range(2):
                    qdq_triton_mega(x, "x", config, num_valid_rows=num_valid_rows)
            torch.cuda.current_stream().wait_stream(warmup)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = qdq_triton_mega(x, "x", config, num_valid_rows=num_valid_rows)
            for rows in _row_cases(capacity) + [3]:
                with self.subTest(capacity=capacity, rows=rows):
                    num_valid_rows.fill_(rows)
                    out.zero_()
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assertTrue(
                        torch.equal(_bits(out[:rows]), _bits(reference[:rows]))
                    )
                    # Programs past the bound do not run: with a bound in the
                    # first half, the zeroed second half stays zero.
                    if rows <= capacity // 2:
                        self.assertFalse(bool(out[capacity // 2 :].any()))

    def test_rejects_invalid_arguments(self) -> None:
        config = _config("mxfp4")
        x = _input(256, 64)
        good = _rows(4)
        with self.assertRaises(NotImplementedError):
            qdq_triton_mega(x, "x", config, num_valid_rows=good, transpose=True)
        for kwargs in (
            {"num_valid_rows": good.float()},
            {"num_valid_rows": good.cpu()},
            {"num_valid_rows": good.reshape(1)},
            {"num_valid_rows": torch.ones(4, device="cuda", dtype=torch.int32)},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    qdq_triton_mega(x, "x", config, **kwargs)


if __name__ == "__main__":
    unittest.main()
