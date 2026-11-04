# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Bitwise `qdq` vs `qdq_triton_mega` preset-equivalence framework.

This test locks the invariant that, for every deterministic fake-quant
preset in ``FAKE_PRESETS``, the pure-PyTorch reference path ``qdq()`` and the
fused Triton mega kernel ``qdq_triton_mega()`` produce *bit-identical*
dequantized output. Any single-grid-step drift (or a masked signed-zero
difference) between the two implementations fails the corresponding case.

The comparison framework (:func:`_bit_view` plus
:meth:`QdqMegaEquivalenceTest.assert_qdq_matches_mega`) is parametrized per
preset -- the preset values are fed one by one into independent unit tests --
and each preset test sweeps every ``(dtype, shape)`` combination through the
reusable helper, which makes a divergence report point at the exact
preset/dtype/shape that broke. (Parametrizing per preset rather than per
``(preset, dtype, shape)`` amortizes the heavy per-process CUDA/Triton
initialization across the dtype/shape sweep.)

One preset family is legitimately skipped (this is NOT a masked divergence):
stochastic-rounding presets (``elem_round_mode == STOCHASTIC``). The pure path
uses ``torch.randint`` while the mega kernel uses its own in-kernel RNG seeded
from the CUDA RNG state. The two RNG streams are unrelated, so bit-identical
output is not well-defined even under a shared seed.

RHT coverage lives in the explicit recipe-pipeline golden tests. This test is
intentionally limited to the quant-only ``qdq()`` and ``qdq_triton_mega()``
backends.
"""

from __future__ import annotations

import dataclasses
import unittest
from collections.abc import Callable, Mapping

import torch
from parameterized import parameterized
from quant_suite_core import (
    ElemRoundMode,
    GlobalScaleGranularity,
    QuantConfig,
    ScaleDtype,
)
from quant_suite_core.fake_presets import FAKE_PRESETS
from quant_suite_core.metrics import cal_sqnr
from quant_suite_core.qdq_pytorch import qdq
from quant_suite_core.qdq_triton import _mega_block_m, qdq_triton_mega
from quant_suite_core.quant_config import ELEM_DTYPE_RANGE_MAP
from quant_suite_core.quant_primitives import TRITON_MEGA_BLOCK_M

# Each preset test sweeps this dtype x shape grid through the helper. Widths are
# divisible by 128 so every block size (16/32) and every MBS macro block
# (16/128) tiles cleanly with multiple macro blocks per row.
_DTYPES: tuple[torch.dtype, ...] = (torch.float32, torch.bfloat16)
_SHAPES: tuple[tuple[int, int], ...] = ((128, 256), (64, 512))
# For the presets tiled below TRITON_MEGA_BLOCK_M (1x32 at 64, 16x16 at 8): quant
# block counts that leave the last tile partial, with the input zero-padded to whole
# quant blocks (40 x 408) or already whole (24 x 4128 at 1x32).
_PARTIAL_TILE_SHAPES: tuple[tuple[int, int], ...] = ((40, 408), (24, 4128))
_SMALL_TILE_PRESETS: list[str] = sorted(
    name
    for name, preset in FAKE_PRESETS.items()
    if _mega_block_m(QuantConfig.from_dict(preset)) < TRITON_MEGA_BLOCK_M
)
# No preset uses ASYMM, so these cover its float (signed range, zero point at the
# midpoint) and int (unsigned range, zero point at the min) element formats.
_ASYMM_PRESETS: dict[str, dict[str, str]] = {
    f"asymm_mx{elem_dtype}": {
        "quant_mode": "asymm",
        "elem_dtype": elem_dtype,
        "elem_round_mode": "nearest",
        "block_size": "1x32",
        "scale_dtype": "e8m0",
        "scale_norm_mode": "pow2max",
        "scale_round_mode": "oas",
    }
    for elem_dtype in ("e2m1", "e4m3", "int3", "int4", "int8")
}
# The two-level (fp8) scale, whose tensorwise global amax is reduced from the
# per-block deltas.
_ASYMM_PRESETS["asymm_nvfp4"] = {
    "quant_mode": "asymm",
    "elem_dtype": "e2m1",
    "elem_round_mode": "nearest",
    "block_size": "1x16",
    "scale_dtype": "fp8",
    "scale_norm_mode": "amax",
    "scale_round_mode": "none",
    "global_scale_granularity": "tensorwise",
    "global_scale_dtype": "fp32",
}
_PRESET_SPECS: dict[str, Mapping[str, object]] = {**FAKE_PRESETS, **_ASYMM_PRESETS}
_STATIC_MBS_PRESETS: list[str] = ["mxfp4_mbs16", "mxfp4_mbs128", "mxfp4_mbs256"]
_MBS_OVERFLOW_PRESETS: list[str] = _STATIC_MBS_PRESETS + ["mxfp4_mbs256_dyn"]
_MBS_NAN_PRESETS: list[str] = [
    "mxfp4_mbs16",
    "mxfp4_mbs16_sr",
    "mxfp4_mbs128",
    "mxfp4_mbs128_sr",
    "mxfp4_mbs256",
    "mxfp4_mbs256_dyn",
]


def _mbs_nan_input(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Random data with NaNs inside MBS macro blocks: one under a 5.0 peak, one
    in a row of its own, and a whole quant block of NaNs."""
    torch.manual_seed(0)
    x = torch.randn(128, 256)
    x[0] = x[0].clamp(-4.0, 4.0)
    x[0, 3] = 5.0
    x[0, 40] = float("nan")
    x[7, 200] = float("nan")
    x[9, 16:32] = float("nan")
    return x.to(dtype).to(device)


def _static_mbs_inf_input(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Random data with +-Inf elements, one alone in its macro block and two
    sharing one, plus finite values above 2^126 that the factor must not
    overflow."""
    torch.manual_seed(0)
    x = torch.randn(128, 256, device=device)
    x[0, 7] = float("inf")
    x[5, 130] = float("-inf")
    x[9, 20] = float("inf")
    x[9, 100] = float("-inf")
    x[20, 40] = 1.2 * 2.0**126
    x[30, 250] = 2.0**127
    return x.to(dtype)


_F32_MAX: float = torch.finfo(torch.float32).max
_BF16_MAX: float = torch.finfo(torch.bfloat16).max


def _mbs_overflow_input(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Random data with finite peaks that overflow fp32 once multiplied by their
    macro factor, each over 1e38 neighbours in its quant block: 3.0e38, FLT_MAX
    (the bf16 max in bf16), the bf16 max, and just above 1.5 * 2^127. Row 4 holds
    tiny values under a -3.0e38 peak. In row 5, a factor just below 2 takes a
    1.6e38 quant-block peak to 1.88 * 2^127, whose grid point is 2^128."""
    torch.manual_seed(0)
    x = torch.randn(128, 256)
    x[0, 0:16] = 1e38
    x[0, 5] = 3.0e38
    x[1, 16:20] = -1e38
    x[1, 20] = _F32_MAX
    x[2, 1:16] = 1e38
    x[2, 0] = _BF16_MAX
    x[3, 101] = -1e38
    x[3, 100] = 1.5 * 2.0**127 * (1 + 2.0**-20)
    x[4, 0] = -3.0e38
    x[4, 1:16] = torch.tensor([1e-38, -1e-40, 1e-45, 2.0**-126, -(2.0**-149)] * 3)
    x[5, 0] = 1.5 * 2.0**127 * (1 + 2.0**-20)
    x[5, 16] = 1.6e38
    if dtype == torch.bfloat16:
        x = x.clamp(-_BF16_MAX, _BF16_MAX)
    return x.to(dtype).to(device)


def _bit_view(t: torch.Tensor) -> torch.Tensor:
    """Reinterpret float bits as a same-width integer for an exact compare
    (distinguishes -0.0 from 0.0 and catches any single-grid-step drift)."""
    if t.element_size() == 4:
        return t.view(torch.int32)
    if t.element_size() == 2:
        return t.view(torch.int16)
    return t


# For the ASYMM padding tests: each element family on 1-D and 2-D blocks, with an
# FP32 block scale and with the two-level FP8 scale.
_ASYMM_PADDING_SPECS: dict[str, dict[str, str]] = {
    f"{elem_dtype}_{block_size}_{scale_dtype}": {
        "quant_mode": "asymm",
        "elem_dtype": elem_dtype,
        "elem_round_mode": "nearest",
        "block_size": block_size,
        "scale_dtype": scale_dtype,
        "scale_norm_mode": "amax",
        "scale_round_mode": "none",
        **(
            {"global_scale_granularity": "tensorwise", "global_scale_dtype": "fp32"}
            if scale_dtype == "fp8"
            else {}
        ),
    }
    for elem_dtype in ("e2m1", "e4m3", "int8")
    for block_size in ("1x16", "16x16")
    for scale_dtype in ("fp32", "fp8")
}
# The last 1x16 block of each row holds 8 elements; the last row of 16x16 blocks
# holds 5 rows and the last column of them 8 columns.
_PADDED_SHAPE: tuple[int, int] = (21, 40)


def _far_from_zero_input(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """100 and 101 in a checkerboard, so every block, whole or partial, spans
    [100, 101]."""
    rows = torch.arange(_PADDED_SHAPE[0], device=device).unsqueeze(1)
    cols = torch.arange(_PADDED_SHAPE[1], device=device).unsqueeze(0)
    return (100 + (rows + cols) % 2).to(dtype)


def _replicate_to_whole_blocks(x: torch.Tensor, m: int, n: int) -> torch.Tensor:
    """``x`` extended to whole m x n blocks with copies of its last row and
    column, so each block holds only values it held before and keeps its min and
    max."""
    rows = torch.arange(-(-x.shape[0] // m) * m, device=x.device)
    cols = torch.arange(-(-x.shape[1] // n) * n, device=x.device)
    return x[rows.clamp(max=x.shape[0] - 1)][:, cols.clamp(max=x.shape[1] - 1)]


def _amax_scale_config(
    quant_mode: str,
    elem_dtype: str,
    scale_dtype: str,
    scale_norm_mode: str = "amax",
    scale_round_mode: str = "none",
) -> QuantConfig:
    """A 1x16 config, by default with an amax-normalized block scale (no preset
    uses one with an FP32 or FP16 scale)."""
    return QuantConfig.from_dict(
        {
            "quant_mode": quant_mode,
            "elem_dtype": elem_dtype,
            "elem_round_mode": "nearest",
            "block_size": "1x16",
            "scale_dtype": scale_dtype,
            "scale_norm_mode": scale_norm_mode,
            "scale_round_mode": scale_round_mode,
        }
    )


def _near_midpoint_input(rows: int) -> torch.Tensor:
    """Random 1x16 blocks whose elements sit a few ULPs from an int8 rounding
    midpoint of their block (amax / 127 times k + 1/2), where a 1-ULP error in
    the reciprocal of the block scale moves the code."""
    generator = torch.Generator().manual_seed(7)
    amax = torch.empty(rows).uniform_(0.5, 8.0, generator=generator)
    k = torch.randint(0, 120, (rows, 15), generator=generator).float()
    ulps = torch.randint(-3, 4, (rows, 15), generator=generator, dtype=torch.int32)
    x = ((k + 0.5) * (amax / 127.0)[:, None]).float()
    x = (x.view(torch.int32) + ulps).view(torch.float32)
    return torch.cat([amax[:, None], x], dim=1)


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Skip when cuda is not available",
)
class QdqMegaEquivalenceTest(unittest.TestCase):
    """Bitwise-equivalence of ``qdq()`` vs ``qdq_triton_mega()`` per preset."""

    def setUp(self) -> None:
        self.device = torch.device("cuda")

    def test_asymm_mean_shift_paths_match_bitwise(self) -> None:
        """Mean-shift eager and mega paths agree bit-for-bit."""
        config = QuantConfig.from_dict(
            {
                "quant_mode": "asymm_mean_shift",
                "elem_dtype": "e4m3",
                "scale_dtype": "fp16",
                "scale_norm_mode": "amax",
                "scale_round_mode": "none",
                "block_size": "1x64",
            }
        )
        base = torch.linspace(-2.0, 3.0, steps=64, device=self.device)

        for dtype in _DTYPES:
            x = (
                base.unsqueeze(0)
                + torch.arange(8, device=self.device).unsqueeze(1) * 0.1
            ).to(dtype)
            eager = qdq(x, "asymm_mean_shift", config)
            mega = qdq_triton_mega(x, "asymm_mean_shift", config)

            eager_bits = _bit_view(eager)
            mega_bits = _bit_view(mega)
            mismatch = eager_bits != mega_bits
            self.assertTrue(
                torch.equal(eager_bits, mega_bits),
                f"qdq vs qdq_triton_mega differs for dtype={dtype}: "
                f"mismatches={mismatch.sum().item()}/{eager.numel()}, "
                f"maxdiff={(eager.float() - mega.float()).abs().max().item()}",
            )

    @parameterized.expand(
        [
            ("e2m1_tensorwise", "e2m1", "tensorwise"),
            ("e2m1_rowwise", "e2m1", "rowwise"),
            ("e4m3_tensorwise", "e4m3", "tensorwise"),
            ("int4_tensorwise", "int4", "tensorwise"),
        ]
    )
    def test_asymm_fp8_scale_paths_match_bitwise(
        self, _name: str, elem_dtype: str, global_scale_granularity: str
    ) -> None:
        """ASYMM with a two-level FP8 scale agrees bit-for-bit on both paths."""
        config = QuantConfig.from_dict(
            {
                "quant_mode": "asymm",
                "elem_dtype": elem_dtype,
                "scale_dtype": "fp8",
                "scale_norm_mode": "amax",
                "scale_round_mode": "none",
                "global_scale_granularity": global_scale_granularity,
                "global_scale_dtype": "fp32",
                "block_size": "1x16",
            }
        )
        torch.manual_seed(0)
        # Per-row offsets make every block's range sit off zero.
        offsets = torch.linspace(-3.0, 5.0, steps=64, device=self.device)
        base = torch.randn(64, 256, device=self.device) + offsets.unsqueeze(1)

        for dtype in _DTYPES:
            x = base.to(dtype)
            eager = qdq(x, "asymm_fp8", config)
            mega = qdq_triton_mega(x, "asymm_fp8", config)

            eager_bits = _bit_view(eager)
            mega_bits = _bit_view(mega)
            mismatch = eager_bits != mega_bits
            self.assertTrue(
                torch.equal(eager_bits, mega_bits),
                f"qdq vs qdq_triton_mega differs for dtype={dtype}: "
                f"mismatches={mismatch.sum().item()}/{eager.numel()}, "
                f"maxdiff={(eager.float() - mega.float()).abs().max().item()}",
            )

    def assert_qdq_matches_mega(
        self,
        preset_name: str,
        dtype: torch.dtype,
        shape: tuple[int, int],
        x: torch.Tensor | None = None,
    ) -> None:
        """Assert ``qdq`` and ``qdq_triton_mega`` agree bit-for-bit.

        Encapsulates the stochastic-rounding skip and the bit-identity check for
        a single ``(preset, dtype, shape)``. The input is seeded random data of
        that shape unless ``x`` is given.
        """
        config = QuantConfig.from_dict(_PRESET_SPECS[preset_name])

        if config.elem_round_mode == ElemRoundMode.STOCHASTIC:
            self.skipTest(
                f"{preset_name}: nondeterministic preset (stochastic rounding); "
                "bitwise equivalence not well-defined -- pure-PyTorch "
                "torch.randint and the Triton in-kernel RNG are independent "
                "streams"
            )

        if x is None:
            torch.manual_seed(0)
            x = torch.randn(*shape, dtype=dtype, device=self.device)
        ref = qdq(x, preset_name, config)
        mega = qdq_triton_mega(x, preset_name, config)

        self.assertEqual(ref.shape, mega.shape)
        self.assertEqual(ref.dtype, mega.dtype)

        ref_bits = _bit_view(ref)
        mega_bits = _bit_view(mega)
        if torch.equal(ref_bits, mega_bits):
            return

        mismatch = ref_bits != mega_bits
        n_mismatch = int(mismatch.sum().item())
        maxdiff = (ref.float() - mega.float()).abs().max().item()
        sqnr = cal_sqnr(mega.float(), ref.float())
        sqnr_val = sqnr.item() if torch.is_tensor(sqnr) else float(sqnr)
        self.fail(
            f"{preset_name}: qdq vs qdq_triton_mega NOT bitwise-equal "
            f"[{config}] block={config.block_size} "
            f"dtype={dtype} shape={tuple(shape)}: "
            f"mismatches={n_mismatch}/{ref.numel()} "
            f"maxdiff={maxdiff:.6g} sqnr={sqnr_val:.2f}dB"
        )

    @parameterized.expand(sorted(FAKE_PRESETS.keys()))
    def test_qdq_matches_mega(self, preset_name: str) -> None:
        # Pass each preset in one by one, sweeping the dtype x shape grid
        # through the reusable helper within a single test process.
        for dtype in _DTYPES:
            for shape in _SHAPES:
                self.assert_qdq_matches_mega(preset_name, dtype, shape)

    @parameterized.expand(_SMALL_TILE_PRESETS)
    def test_qdq_matches_mega_partial_tiles(self, preset_name: str) -> None:
        for dtype in _DTYPES:
            for shape in _PARTIAL_TILE_SHAPES:
                self.assert_qdq_matches_mega(preset_name, dtype, shape)

    @parameterized.expand(sorted(_ASYMM_PRESETS.keys()))
    def test_qdq_matches_mega_asymm(self, preset_name: str) -> None:
        for dtype in _DTYPES:
            for shape in _SHAPES + _PARTIAL_TILE_SHAPES:
                self.assert_qdq_matches_mega(preset_name, dtype, shape)

    def _assert_bitwise(self, got: torch.Tensor, want: torch.Tensor, what: str) -> None:
        mismatch = _bit_view(got) != _bit_view(want)
        self.assertFalse(
            bool(mismatch.any()),
            f"{what}: mismatches={int(mismatch.sum())}/{got.numel()}, "
            f"maxdiff={(got.float() - want.float()).abs().max().item()}",
        )

    @parameterized.expand(sorted(_ASYMM_PADDING_SPECS.keys()))
    def test_asymm_partial_block_ignores_padding(self, name: str) -> None:
        # The zero padding of a partial block is left out of its min and max, so
        # a partial block spanning [100, 101] is quantized as a whole block of
        # the same values would be.
        config = QuantConfig.from_dict(_ASYMM_PADDING_SPECS[name])
        m, n = config.block_size.m, config.block_size.n
        for dtype in _DTYPES:
            x = _far_from_zero_input(dtype, self.device)
            for transpose in (False, True):
                what = f"{name} {dtype} transpose={transpose}"
                eager = qdq(x, name, config, transpose)
                mega = qdq_triton_mega(x, name, config, transpose)
                self._assert_bitwise(mega, eager, f"{what}, qdq_triton_mega vs qdq")
                err = (eager.float() - x.float()).abs().max().item()
                self.assertLess(err, 0.05, what)
            whole = qdq(_replicate_to_whole_blocks(x, m, n), name, config)
            self._assert_bitwise(
                qdq(x, name, config),
                whole[: x.shape[0], : x.shape[1]],
                f"{name} {dtype}, partial vs whole blocks",
            )

    @parameterized.expand(sorted(_ASYMM_PADDING_SPECS.keys()))
    def test_asymm_partial_block_counts_real_zero(self, name: str) -> None:
        # Only padding is left out: a real 0.0 in a partial block sets the
        # block's min, so it dequantizes to about 0 rather than to about 100.
        config = QuantConfig.from_dict(_ASYMM_PADDING_SPECS[name])
        for dtype in _DTYPES:
            x = _far_from_zero_input(dtype, self.device)
            x[-1, -1] = 0.0
            eager = qdq(x, name, config)
            mega = qdq_triton_mega(x, name, config)
            what = f"{name} {dtype}"
            self._assert_bitwise(mega, eager, f"{what}, qdq_triton_mega vs qdq")
            self.assertLess(abs(eager[-1, -1].item()), 1.0, what)

    @parameterized.expand(_STATIC_MBS_PRESETS)
    def test_qdq_matches_mega_static_mbs_inf(self, preset_name: str) -> None:
        # The macro factor comes from the unclamped input: an Inf amax gives a
        # zero reciprocal, so its macro block is not scaled.
        for dtype in _DTYPES:
            x = _static_mbs_inf_input(dtype, self.device)
            self.assert_qdq_matches_mega(preset_name, dtype, tuple(x.shape), x=x)

    def assert_same_bits_and_nans(
        self, ref: torch.Tensor, out: torch.Tensor, what: str
    ) -> None:
        """Bitwise equality except the NaN payload; NaN positions must match."""
        ref_nan = torch.isnan(ref)
        out_nan = torch.isnan(out)
        self.assertTrue(
            torch.equal(ref_nan, out_nan),
            f"{what}: NaN positions differ "
            f"({int(ref_nan.sum())} vs {int(out_nan.sum())} NaNs)",
        )
        mismatch = _bit_view(ref.masked_fill(ref_nan, 0)) != _bit_view(
            out.masked_fill(out_nan, 0)
        )
        n_mismatch = int(mismatch.sum().item())
        self.assertEqual(
            n_mismatch,
            0,
            f"{what}: mismatches={n_mismatch}/{ref.numel()} "
            f"maxdiff={(ref.float() - out.float()).abs().nan_to_num().max().item():.6g}",
        )

    @parameterized.expand(_MBS_NAN_PRESETS)
    def test_qdq_matches_mega_mbs_nan(self, preset_name: str) -> None:
        # A NaN poisons its own quant block only: the macro factor is taken
        # over the other elements, so it is the same on CPU, on CUDA and in the
        # kernel, and the neighbours quantize as if the NaN were zero.
        config = QuantConfig.from_dict(_PRESET_SPECS[preset_name])
        stochastic = config.elem_round_mode == ElemRoundMode.STOCHASTIC
        n = config.block_size.n

        def run(fn: Callable[..., torch.Tensor], x: torch.Tensor) -> torch.Tensor:
            # Stochastic rounding draws its bits per position, so reseeding makes
            # two calls on same-shaped inputs use the same bits.
            torch.manual_seed(0)
            return fn(x, preset_name, config).cpu()

        for dtype in _DTYPES:
            what = f"{preset_name} dtype={dtype}"
            x = _mbs_nan_input(dtype, self.device)
            nan_block = torch.isnan(x.cpu()).reshape(x.shape[0], -1, n).any(dim=2)
            expected_nan = nan_block.repeat_interleave(n, dim=1)
            # The same input with each NaN replaced by zero.
            x_zero = x.nan_to_num(nan=0.0)
            outs = {
                "cpu": run(qdq, x.cpu()),
                "cuda": run(qdq, x),
                "mega": run(qdq_triton_mega, x),
            }
            for backend, out in outs.items():
                self.assertTrue(
                    torch.equal(torch.isnan(out), expected_nan),
                    f"{what} {backend}: NaN spread beyond its quant block",
                )
            if not stochastic:
                self.assert_same_bits_and_nans(
                    outs["cpu"], outs["cuda"], f"{what} cpu vs cuda"
                )
                self.assert_same_bits_and_nans(
                    outs["cuda"], outs["mega"], f"{what} qdq vs mega"
                )
            if config.mbs_dynamic:
                # The search scores a poisoned block as zero error, so its
                # zero-filled twin can pick a different factor.
                continue
            zeros = {
                "cpu": run(qdq, x_zero.cpu()),
                "cuda": run(qdq, x_zero),
                "mega": run(qdq_triton_mega, x_zero),
            }
            for backend, out in outs.items():
                self.assert_same_bits_and_nans(
                    out.masked_fill(expected_nan, float("nan")),
                    zeros[backend].masked_fill(expected_nan, float("nan")),
                    f"{what} {backend}: NaN vs zero-filled input",
                )

    @parameterized.expand(_MBS_OVERFLOW_PRESETS)
    def test_qdq_matches_mega_mbs_overflow(self, preset_name: str) -> None:
        # Where `x * factor` could overflow, the quant block is divided by 4
        # more and the 4 moves into its block scale, so finite inputs return on
        # the grid of the scaled block instead of turning into NaN.
        config = QuantConfig.from_dict(_PRESET_SPECS[preset_name])
        for dtype in _DTYPES:
            what = f"{preset_name} dtype={dtype}"
            x = _mbs_overflow_input(dtype, self.device)
            cpu = qdq(x.cpu(), preset_name, config)
            cuda = qdq(x, preset_name, config).cpu()
            mega = qdq_triton_mega(x, preset_name, config).cpu()
            self.assert_same_bits_and_nans(cpu, cuda, f"{what} cpu vs cuda")
            self.assert_same_bits_and_nans(cuda, mega, f"{what} qdq vs mega")
            self.assertFalse(bool(torch.isnan(cuda).any()), f"{what}: NaN output")
            info = qdq(x, preset_name, config, return_quant_info=True)
            info_mega = qdq_triton_mega(x, preset_name, config, return_quant_info=True)
            self.assert_same_bits_and_nans(
                info.local_unscale.cpu(), info_mega.local_unscale.cpu(), what
            )
            self.assert_same_bits_and_nans(
                info.quantized_val.cpu(), info_mega.quantized_val.cpu(), what
            )
            out = cuda.float()
            if config.mbs_dynamic:
                # In rows 0 to 5 the SSE of every candidate overflows, so the
                # search keeps c = 1.0 there and the checks below do not apply.
                continue
            ref = x.cpu().float()
            peaks = [(0, 5), (2, 0), (3, 100), (5, 0)]
            if dtype == torch.bfloat16:
                peaks.append((1, 20))
            else:
                # The nearest grid point of FLT_MAX's block, 6 * scale / factor,
                # is above FLT_MAX and rounds to +Inf, as plain MXFP4 does.
                self.assertEqual(out[1, 20].item(), float("inf"), what)
            for r, c in peaks:
                torch.testing.assert_close(
                    out[r, c], ref[r, c], rtol=1e-2, atol=0.0, msg=f"{what} [{r}, {c}]"
                )
            # One e2m1 grid step around the neighbours.
            for r, c in ((0, 0), (0, 6), (1, 16), (2, 1), (3, 101), (5, 16)):
                torch.testing.assert_close(
                    out[r, c], ref[r, c], rtol=0.25, atol=0.0, msg=f"{what} [{r}, {c}]"
                )
            self.assertTrue(bool((out[4, 1:16] == 0).all()), f"{what}: tiny values")
            inf = torch.isinf(out)
            if dtype == torch.float32:
                inf[1, 20] = False
            self.assertFalse(bool(inf.any()), f"{what}: Inf output")
            # quant_info reports the block scale of the factor-scaled data:
            # 3.0e38 * factor = 6 * 2^126.
            self.assertEqual(info.local_unscale[0].item(), 2.0**126, what)

    @parameterized.expand(["mxfp4_mbs128", "mxfp4_mbs256"])
    def test_mbs_headroom_keeps_tiny_quant_blocks(self, preset_name: str) -> None:
        # The overflow headroom is per quant block: a quant block of tiny values
        # next to a peak above 2^127 sits at the E8M0 scale floor and keeps the
        # result of applying the factor directly, which does not overflow there.
        # Eager only: the kernel flushes such blocks (it computes the scale
        # 2^-127 as 0).
        config = QuantConfig.from_dict(_PRESET_SPECS[preset_name])
        inner = dataclasses.replace(
            config,
            global_scale_granularity=GlobalScaleGranularity.NONE,
            mbs_block_size=None,
            global_scale_dtype=ScaleDtype.NONE,
        )
        width = config.get_mbs_block_size()
        torch.manual_seed(0)
        x = torch.randn(4, 256) * 1e-38
        x[:, 0] = torch.tensor([1.0, 1.2, 1.49, 1.5 * (1 + 2.0**-20)]) * 2.0**127
        x[:, 1:16] = torch.randn(4, 15)
        # The static factor: the mantissa of 6 / macro peak.
        peak = x.abs().reshape(4, -1, width).amax(dim=2)
        bits = (6.0 / peak).view(torch.int32)
        factor = ((bits & 0x007FFFFF) | (127 << 23)).view(torch.float32)
        inv = (1.0 / factor).repeat_interleave(width, dim=1)
        for device in (torch.device("cpu"), self.device):
            xd = x.to(device)
            want = (qdq(xd / inv.to(device), preset_name, inner) * inv.to(device)).cpu()
            got = qdq(xd, preset_name, config).cpu()
            # Row 3 overflows in the direct order; compare its tiny blocks only.
            self.assert_same_bits_and_nans(
                want[:3], got[:3], f"{preset_name} {device.type}"
            )
            self.assert_same_bits_and_nans(
                want[3, 16:], got[3, 16:], f"{preset_name} {device.type} row 3"
            )

    @parameterized.expand(
        [
            ("asymm", "int3", 4),
            ("asymm", "int4", 8),
            ("asymm", "int8", 128),
            ("asymm_mean_shift", "int3", 2),
            ("asymm_mean_shift", "int4", 4),
            ("asymm_mean_shift", "int8", 64),
        ]
    )
    def test_asymm_int_pow2max_spans_full_range(
        self, quant_mode: str, elem_dtype: str, max_pow2: int
    ) -> None:
        # The unsigned integer range of ASYMM reaches twice the power of two of
        # SYMM: it maps the span 8 of a 1x32 block on [1, 9] onto codes
        # 0..max_pow2, a step of 8 / max_pow2. The signed range of the mean
        # shift maps the largest distance from the mean, 4, onto -max_pow2 and
        # +max_pow2, so the values below the mean keep their codes.
        config = QuantConfig.from_dict(
            {
                "quant_mode": quant_mode,
                "elem_dtype": elem_dtype,
                "elem_round_mode": "nearest",
                "block_size": "1x32",
                "scale_dtype": "e8m0",
                "scale_norm_mode": "pow2max",
                "scale_round_mode": "oas",
            }
        )
        torch.manual_seed(0)
        x = torch.randn(8, 64)
        x[:2] = torch.linspace(1.0, 9.0, 32).repeat(2, 2)
        step = (8.0 if quant_mode == "asymm" else 4.0) / max_pow2
        what = f"{quant_mode} {elem_dtype}"
        cpu = qdq(x, what, config, return_quant_info=True)
        cuda = qdq(x.to(self.device), what, config, return_quant_info=True)
        mega = qdq_triton_mega(x.to(self.device), what, config, return_quant_info=True)
        for field in ("dequant_data", "local_unscale"):
            self.assert_same_bits_and_nans(
                getattr(cpu, field), getattr(cuda, field).cpu(), f"{what} {field}"
            )
            self.assert_same_bits_and_nans(
                getattr(cuda, field).cpu(),
                getattr(mega, field).cpu(),
                f"{what} {field}",
            )
        # By value: a signed code that rounds to -0.0 is -0.0 in `qdq` (CPU and
        # CUDA) and +0.0 in the kernel.
        self.assertTrue(torch.equal(cpu.quantized_val, cuda.quantized_val.cpu()), what)
        self.assertTrue(torch.equal(cpu.quantized_val, mega.quantized_val.cpu()), what)
        codes = cpu.quantized_val[:2]
        self.assertEqual(codes.max().item(), max_pow2, what)
        self.assertEqual(cpu.local_unscale[0].item(), step, what)
        self.assertEqual(
            codes.min().item(), 0.0 if quant_mode == "asymm" else -max_pow2, what
        )
        err = (cpu.dequant_data[:2] - x[:2]).abs().max().item()
        self.assertLessEqual(err, step / 2, what)

    @parameterized.expand(
        [
            ("e4m3_fp16", "e4m3", "fp16", "amax"),
            ("e4m3_e8m0", "e4m3", "e8m0", "pow2max"),
            ("int3_fp16", "int3", "fp16", "amax"),
            ("int4_fp16", "int4", "fp16", "amax"),
            ("int8_fp16", "int8", "fp16", "amax"),
            ("int3_e8m0", "int3", "e8m0", "pow2max"),
            ("int4_e8m0", "int4", "e8m0", "pow2max"),
            ("int8_e8m0", "int8", "e8m0", "pow2max"),
        ]
    )
    def test_mean_shift_inf_nan(
        self, name: str, elem_dtype: str, scale_dtype: str, scale_norm_mode: str
    ) -> None:
        # In ASYMM_MEAN_SHIFT an Inf makes its whole block NaN (its fixed-point
        # mean is 0 / 0), and a NaN counts as zero in the mean, sets no scale
        # and dequantizes to NaN, with the rest of its block finite; the same
        # in `qdq` on CPU and CUDA and in the kernel, codes and scales too.
        config = QuantConfig.from_dict(
            {
                "quant_mode": "asymm_mean_shift",
                "elem_dtype": elem_dtype,
                "elem_round_mode": "nearest",
                "block_size": "1x64",
                "scale_dtype": scale_dtype,
                "scale_norm_mode": scale_norm_mode,
                "scale_round_mode": "oas" if scale_dtype == "e8m0" else "none",
            }
        )
        nan, inf = float("nan"), float("inf")
        torch.manual_seed(0)
        x = torch.randn(10, 64) + torch.arange(10).unsqueeze(1) * 0.25
        x[0, 5] = inf
        x[1, 5] = -inf
        x[2, [5, 9]] = torch.tensor([inf, nan])
        x[3] = inf
        x[4, 5] = nan
        x[5, [5, 40]] = nan
        x[6, x[6].abs().argmax()] = nan
        x[7] = nan
        whole_block_nan = torch.tensor([1, 1, 1, 1, 0, 0, 0, 1, 0, 0], dtype=torch.bool)
        for dtype in _DTYPES:
            xd = x.to(dtype)
            what = f"{name} {dtype}"
            cpu = qdq(xd, what, config, return_quant_info=True)
            cuda = qdq(xd.to(self.device), what, config, return_quant_info=True)
            mega = qdq_triton_mega(
                xd.to(self.device), what, config, return_quant_info=True
            )
            # A block whose deviation is 0 or NaN is a zero block: under an
            # E8M0 scale `qdq` reports 2^-127 and the kernel 1.0.
            scale_rows = ~whole_block_nan if scale_dtype == "e8m0" else slice(None)
            for out, label in ((cuda, "cuda"), (mega, "mega")):
                self.assert_same_bits_and_nans(
                    cpu.dequant_data, out.dequant_data.cpu(), f"{what} cpu vs {label}"
                )
                self.assert_same_bits_and_nans(
                    cpu.local_unscale.reshape(10)[scale_rows],
                    out.local_unscale.cpu().reshape(10)[scale_rows],
                    f"{what} cpu vs {label} local_unscale",
                )
                # By value: a code that rounds to -0.0 is +0.0 in the kernel.
                want, got = cpu.quantized_val, out.quantized_val.cpu()
                self.assertTrue(
                    torch.equal(torch.isnan(want), torch.isnan(got))
                    and torch.equal(want.nan_to_num(), got.nan_to_num()),
                    f"{what} cpu vs {label} quantized_val",
                )
            out_nan = torch.isnan(cpu.dequant_data)
            self.assertTrue(bool(out_nan[whole_block_nan].all()), what)
            self.assertTrue(
                torch.equal(
                    out_nan[~whole_block_nan], torch.isnan(xd)[~whole_block_nan]
                ),
                what,
            )
            self.assertTrue(torch.equal(torch.isnan(cpu.quantized_val), out_nan), what)

    @parameterized.expand(["int3", "int4", "int8"])
    def test_asymm_int_codes_have_no_negative_zero(self, elem_dtype: str) -> None:
        # ASYMM's unsigned int range puts the block min on code 0. When the fp16
        # zero point rounds up past the min (1.0009 -> 1.0009766), the min's
        # scaled value is just below 0 and rounds to -0.0, which every backend
        # must return as +0.0.
        config = QuantConfig.from_dict(
            {
                "quant_mode": "asymm",
                "elem_dtype": elem_dtype,
                "elem_round_mode": "nearest",
                "block_size": "1x32",
                "scale_dtype": "e8m0",
                "scale_norm_mode": "pow2max",
                "scale_round_mode": "oas",
            }
        )
        x = torch.linspace(1.0, 9.0, 32).repeat(4, 2)
        x[:, [0, 32]] = 1.0009
        cpu = qdq(x, elem_dtype, config, return_quant_info=True)
        cuda = qdq(x.to(self.device), elem_dtype, config, return_quant_info=True)
        mega = qdq_triton_mega(
            x.to(self.device), elem_dtype, config, return_quant_info=True
        )
        codes = cpu.quantized_val
        self.assertTrue(bool((codes[:, [0, 32]] == 0).all()), elem_dtype)
        self.assertFalse(bool(torch.signbit(codes).any()), f"{elem_dtype}: -0.0")
        for out, label in ((cuda, "cuda"), (mega, "mega")):
            for field in ("quantized_val", "dequant_data", "local_unscale"):
                self.assert_same_bits_and_nans(
                    getattr(cpu, field),
                    getattr(out, field).cpu(),
                    f"{elem_dtype} cpu vs {label} {field}",
                )

    @parameterized.expand(
        [
            ("e4m3_tensorwise", "e4m3", "tensorwise"),
            ("int8_tensorwise", "int8", "tensorwise"),
            ("int4_tensorwise", "int4", "tensorwise"),
            ("e4m3_rowwise", "e4m3", "rowwise"),
            ("int8_rowwise", "int8", "rowwise"),
        ]
    )
    def test_mean_shift_fp8_scale_matches_qdq(
        self, name: str, elem_dtype: str, granularity: str
    ) -> None:
        # ASYMM_MEAN_SHIFT under the two-level (fp8) scale: the tensorwise
        # global amax is the largest amax of the mean-centred blocks, and the
        # centred data is quantized with the combined scale, as in `qdq`.
        config = QuantConfig.from_dict(
            {
                "quant_mode": "asymm_mean_shift",
                "elem_dtype": elem_dtype,
                "elem_round_mode": "nearest",
                "block_size": "1x16",
                "scale_dtype": "fp8",
                "scale_norm_mode": "amax",
                "scale_round_mode": "none",
                "global_scale_granularity": granularity,
                "global_scale_dtype": "fp32",
            }
        )
        torch.manual_seed(0)
        # Blocks far from zero: their centred amax is much smaller than |x|.
        x = torch.randn(32, 128) + torch.randint(-200, 200, (32, 8)).repeat_interleave(
            16, dim=1
        )
        # Each block's |x| peak is a power of two, so the fixed-point mean has
        # no rounded division: the kernel's fp32 division is not correctly
        # rounded, and the means of the two backends can differ elsewhere.
        blocks = x.view(32, 8, 16)
        peak = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(dim=2))))
        blocks[:, :, 0] = torch.copysign(peak, blocks[:, :, 0])
        for dtype in _DTYPES:
            xd = x.to(dtype)
            what = f"{name} {dtype}"
            cpu = qdq(xd, what, config, return_quant_info=True)
            cuda = qdq(xd.to(self.device), what, config, return_quant_info=True)
            mega = qdq_triton_mega(
                xd.to(self.device), what, config, return_quant_info=True
            )
            for out, label in ((cuda, "cuda"), (mega, "mega")):
                for field in ("dequant_data", "local_unscale", "global_scale"):
                    self.assert_same_bits_and_nans(
                        getattr(cpu, field).float(),
                        getattr(out, field).cpu().float(),
                        f"{what} cpu vs {label} {field}",
                    )
                # By value: a signed code that rounds to -0.0 is +0.0 in the
                # kernel.
                self.assertTrue(
                    torch.equal(cpu.quantized_val, out.quantized_val.cpu()),
                    f"{what} cpu vs {label} quantized_val",
                )

    @parameterized.expand(
        [
            "mxfp4_mbs128",
            "mxfp4_mbs256_dyn",
            "mxfp8",
            "mxfp4_16x16",
            "nvfp4",
            "rowwise_nvfp4",
        ]
    )
    def test_quant_info_matches_qdq(self, preset_name: str) -> None:
        # Every QuantInfo field has qdq()'s shape and dtype and the same values,
        # on shapes that need padding to whole quant blocks and MBS macro blocks.
        config = QuantConfig.from_dict(_PRESET_SPECS[preset_name])
        for shape in ((33, 100), (40, 408), (3, 5, 40)):
            torch.manual_seed(0)
            x = torch.randn(*shape, dtype=torch.bfloat16, device=self.device)
            ref = qdq(x, preset_name, config, return_quant_info=True)
            out = qdq_triton_mega(x, preset_name, config, return_quant_info=True)
            for field in (f.name for f in dataclasses.fields(ref)):
                want, got = getattr(ref, field), getattr(out, field)
                what = f"{preset_name} {shape} {field}"
                if want is None:
                    self.assertIsNone(got, what)
                    continue
                self.assertIsNotNone(got, what)
                self.assertEqual(tuple(got.shape), tuple(want.shape), what)
                self.assertEqual(got.dtype, want.dtype, what)
                self.assert_same_bits_and_nans(want.cpu(), got.cpu(), what)

    @parameterized.expand(["mxfp4", "mxfp8", "mxint8"])
    def test_mega_flushes_all_subnormal_blocks(self, preset_name: str) -> None:
        # Accepted deviation, pinned so it cannot change silently: a quant block
        # whose values are all below about 1.2e-38 gets the E8M0 scale 2^-127.
        # qdq() builds it exactly (CPU and CUDA agree); the kernel computes it
        # with tl.exp2, which returns 0 there, so it falls back to scale 1.0
        # and flushes the block to zero. Matching it in the kernel would cost
        # more than 3%.
        config = QuantConfig.from_dict(_PRESET_SPECS[preset_name])
        torch.manual_seed(0)
        x = torch.randn(4, 64)
        x[0] = (x[0] * 3e-39).clamp(-1.1e-38, 1.1e-38)
        x = x.to(self.device)
        cpu = qdq(x.cpu(), preset_name, config)
        cuda = qdq(x, preset_name, config).cpu()
        mega = qdq_triton_mega(x, preset_name, config).cpu()
        self.assert_same_bits_and_nans(cpu, cuda, f"{preset_name} cpu vs cuda")
        self.assertTrue(bool((cuda[0] != 0).any()), f"{preset_name}: qdq flushed")
        self.assertTrue(bool((mega[0] == 0).all()), f"{preset_name}: mega kept")
        self.assert_same_bits_and_nans(
            cuda[1:], mega[1:], f"{preset_name} qdq vs mega, normal rows"
        )

    def test_block_scale_reciprocal_is_correctly_rounded(self) -> None:
        # With an FP32 or FP16 block scale, elements are multiplied by the
        # correctly rounded fp32 reciprocal of the stored scale, in qdq() (CPU
        # and CUDA) and in the kernel. An fp16 reciprocal would have 2^-11
        # relative error and overflow to Inf for a scale below 2^-16, and
        # Triton's `1.0 / scale` is not correctly rounded.
        torch.manual_seed(0)
        random = torch.randn(1024, 64)
        near_midpoint = _near_midpoint_input(4096)
        for quant_mode in ("symm", "asymm"):
            for elem_dtype in ("int8", "int4", "e2m1", "e4m3"):
                for scale_dtype in ("fp32", "fp16"):
                    config = _amax_scale_config(quant_mode, elem_dtype, scale_dtype)
                    for x in (random, near_midpoint):
                        x = x.to(self.device)
                        ref = qdq(x, "x", config, return_quant_info=True)
                        out = qdq_triton_mega(x, "x", config, return_quant_info=True)
                        for field in ("quantized_val", "dequant_data", "local_unscale"):
                            self.assert_same_bits_and_nans(
                                getattr(ref, field).cpu(),
                                getattr(out, field).cpu(),
                                f"{quant_mode} {elem_dtype} {scale_dtype} "
                                f"{tuple(x.shape)} {field}",
                            )
        # The codes are those of the fp32 reciprocal of the reported scale. In
        # the first block 0.3760359287 / 0.0103023536 = 36.500002 takes code
        # 37; a reciprocal 1 ULP low would give 36.
        for scale_dtype in ("fp32", "fp16"):
            config = _amax_scale_config("symm", "int8", scale_dtype)
            for device in ("cpu", self.device):
                info = qdq(
                    near_midpoint.to(device), "x", config, return_quant_info=True
                )
                unscale = info.local_unscale.cpu().reshape(-1, 1)
                scaled = near_midpoint * (1.0 / unscale)
                want = torch.round(torch.clamp(scaled, -127.0, 127.0))
                self.assert_same_bits_and_nans(
                    want, info.quantized_val.cpu(), f"{scale_dtype} {device} codes"
                )
        # An FP16 scale below 2^-16 has a finite fp32 reciprocal: the block
        # [1e-3, 5e-4, 0, ...] quantizes to within half a step.
        config = _amax_scale_config("symm", "int8", "fp16")
        x = torch.zeros(1, 16)
        x[0, :2] = torch.tensor([1e-3, 5e-4])
        for device in ("cpu", self.device):
            info = qdq(x.to(device), "x", config, return_quant_info=True)
            unscale = info.local_unscale.item()
            self.assertLess(unscale, 2.0**-16)
            out = info.dequant_data.cpu()
            self.assertTrue(bool(torch.isfinite(out).all()), f"{device}: {out}")
            self.assertLessEqual((out - x).abs().max().item(), unscale / 2, device)

    def test_unscale_is_correctly_rounded_on_every_device(self) -> None:
        # The block scale amax / max_pos (and an E8Mx amax / max_pow2) is
        # divided correctly rounded by qdq() on CPU and CUDA and by the kernel.
        # A multiply by the rounded reciprocal (torch on CUDA dividing by a
        # Python number, Triton's `/` by a constant) can differ: for the first
        # block of this input under ASYMM e2m1 with an FP32 scale,
        # 1.6482992 / 6 = 0.274716536 rounds to 0.27471653, and the reciprocal
        # multiply gives 0.27471656.
        torch.manual_seed(1)
        x = torch.randn(1024, 64)
        configs = [
            (quant_mode, elem_dtype, scale_dtype, "amax", "none")
            for quant_mode in ("symm", "asymm")
            for elem_dtype in ("int8", "e2m1", "e4m3", "e8m1")
            for scale_dtype in ("fp32", "fp16")
        ]
        configs += [
            ("symm", "e2m1", "e8m0", "amax", "floor"),
            ("symm", "e8m1", "e8m0", "pow2max", "oas"),
        ]
        for spec in configs:
            config = _amax_scale_config(*spec)
            cpu = qdq(x, "x", config, return_quant_info=True)
            for label, out in (
                ("cuda", qdq(x.to(self.device), "x", config, return_quant_info=True)),
                (
                    "mega",
                    qdq_triton_mega(
                        x.to(self.device), "x", config, return_quant_info=True
                    ),
                ),
            ):
                for field in ("quantized_val", "dequant_data", "local_unscale"):
                    self.assert_same_bits_and_nans(
                        getattr(cpu, field),
                        getattr(out, field).cpu(),
                        f"{spec} cpu vs {label} {field}",
                    )
            if spec[2] != "fp32":
                continue
            # The FP32 scale is the exact quotient rounded once: an fp64
            # quotient of these operands rounds to fp32 as the exact one does.
            blocks = x.reshape(-1, 16).double()
            if spec[0] == "symm":
                delta = blocks.abs().amax(dim=1)
            else:
                delta = blocks.amax(dim=1) - blocks.amin(dim=1)
                if spec[1] != "int8":
                    delta = delta / 2
            _, max_pos = ELEM_DTYPE_RANGE_MAP[config.quant_mode][config.elem_dtype]
            want = (delta.float().double() / max_pos).float()
            self.assert_same_bits_and_nans(
                want, cpu.local_unscale.reshape(-1), f"{spec} exact"
            )

    def test_tiny_blocks_quantize_to_their_values(self) -> None:
        # A nonzero block whose FP32 or FP16 scale underflows takes the
        # smallest positive scale (2^-149, 2^-24), and an FP32 scale below
        # FLT_MIN is inverted with a 2^64 prescale, so tiny blocks quantize like
        # any other, in qdq() (CPU, CUDA) and the kernel alike. Without them,
        # these blocks would get a zero scale or a reciprocal that overflows,
        # and an infinite reciprocal times zero is NaN.
        tiny = 2.0**-148
        f32_min_subnormal = 2.0**-149
        cases = [
            # (quant_mode, elem_dtype, scale_dtype, leading values, expected output)
            ("asymm", "int8", "fp32", [tiny], [tiny]),
            ("asymm", "e2m1", "fp32", [tiny], [tiny]),
            ("symm", "e2m1", "fp32", [tiny], [tiny]),
            ("symm", "int8", "fp32", [-tiny, tiny / 2], [-tiny, tiny / 2]),
            ("symm", "e2m1", "fp32", [1e-39], None),
            ("symm", "int8", "fp16", [1e-6, 5e-7], [17 * 2.0**-24, 8 * 2.0**-24]),
            ("asymm", "int8", "fp16", [1e-10], [0.0]),
            ("asymm", "int8", "fp32", [3.0] * 16, [3.0] * 16),
        ]
        torch.manual_seed(0)
        random_tiny = torch.randn(4, 64) * 1e-40
        for quant_mode, elem_dtype, scale_dtype, values, want in cases:
            config = _amax_scale_config(quant_mode, elem_dtype, scale_dtype)
            x = torch.zeros(1, 16)
            x[0, : len(values)] = torch.tensor(values)
            what = f"{quant_mode} {elem_dtype} {scale_dtype} {values[:2]}"
            outs = [
                qdq(x, "x", config, return_quant_info=True),
                qdq(x.to(self.device), "x", config, return_quant_info=True),
                qdq_triton_mega(x.to(self.device), "x", config, return_quant_info=True),
            ]
            for out, label in zip(outs[1:], ("cuda", "mega")):
                for field in ("quantized_val", "dequant_data", "local_unscale"):
                    self.assert_same_bits_and_nans(
                        getattr(outs[0], field),
                        getattr(out, field).cpu(),
                        f"{what} cpu vs {label} {field}",
                    )
            got = outs[0].dequant_data
            self.assertTrue(bool(torch.isfinite(got).all()), f"{what}: {got}")
            if want is None:
                # The scale 1e-39 / 6 is used as it is: the block takes code 6.
                unscale = outs[0].local_unscale.item()
                self.assertEqual(got[0, 0].item(), 6 * unscale, what)
                self.assertTrue(bool((got[0, 1:] == 0).all()), what)
            else:
                self.assertEqual(got[0, : len(want)].tolist(), want, what)
        for quant_mode in ("symm", "asymm"):
            config = _amax_scale_config(quant_mode, "int8", "fp32")
            outs = [
                qdq(random_tiny, "x", config, return_quant_info=True),
                qdq(random_tiny.to(self.device), "x", config, return_quant_info=True),
                qdq_triton_mega(
                    random_tiny.to(self.device), "x", config, return_quant_info=True
                ),
            ]
            for out in outs[1:]:
                for field in ("quantized_val", "dequant_data", "local_unscale"):
                    self.assert_same_bits_and_nans(
                        getattr(outs[0], field),
                        getattr(out, field).cpu(),
                        f"{quant_mode} randn * 1e-40 {field}",
                    )
            self.assertTrue(bool(torch.isfinite(outs[0].dequant_data).all()))
            if quant_mode == "asymm":
                # Its fp16 zero point (the block min) underflows to zero here.
                continue
            # Within half a step of the scale (plus the subnormal rounding).
            step = outs[0].local_unscale.reshape(-1, 1)
            err = (outs[0].dequant_data - random_tiny).abs().reshape(-1, 16)
            self.assertTrue(
                bool((err <= step / 2 + f32_min_subnormal).all()),
                f"{quant_mode}: max error {err.max().item()}",
            )


if __name__ == "__main__":
    unittest.main()
