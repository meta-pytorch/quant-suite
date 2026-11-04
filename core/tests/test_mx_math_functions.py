# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from quant_suite_core import (
    ElemDtype,
    GlobalScaleGranularity,
    QuantMode,
    ScaleRoundMode,
)
from quant_suite_core.quant_config import ELEM_DTYPE_RANGE_MAP
from quant_suite_core.quant_primitives import (
    calculate_fp8_scale_for_nvfp4,
    fp32_ceil_fp8,
    fp32_floor_fp8,
    fp32_to_e8m0,
    fp32_to_e8mx,
    fp32_to_fpx,
    nvfp4_global_scale,
)
from quant_test_utils import reference_nvfp4_global_scale, skip_if_no_cuda


# ******************************************************************************
# * Standalone Math Function Tests (GPU required)                     *
# ******************************************************************************
@skip_if_no_cuda
class TestElementQuantizationAndScalingFactor(unittest.TestCase):
    """Tests for standalone math/quantization functions."""

    def setUp(self) -> None:
        self.device = torch.device("cuda")

    def test_fp32_to_e8mx_e8m1_known_values(self) -> None:
        """E8M1 quantizes to {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}."""
        x = torch.tensor([0.0, 0.7, 1.2, 1.8, 3.5, 5.5], device=self.device)
        result = fp32_to_e8mx(x, ElemDtype.E8M1)
        # E8M1: round-to-nearest-even with 1 mantissa bit
        # 0.7 -> 0.5 or 0.75 -> nearest representable
        # Values should be representable in E8M1 format
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, torch.float32)
        # result[0] = 0.0 (exact)
        self.assertEqual(result[0].item(), 0.0)

    def test_fp32_to_e8mx_e8m2_known_values(self) -> None:
        """E8M2 has 2 mantissa bits, so representable: {0, 0.25, 0.5, 0.75, 1.0, 1.25, ...}."""
        x = torch.tensor([0.0, 1.0, 2.0, 3.5, 5.0], device=self.device)
        result = fp32_to_e8mx(x, ElemDtype.E8M2)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result[0].item(), 0.0)
        self.assertEqual(result[1].item(), 1.0)
        self.assertEqual(result[2].item(), 2.0)

    def test_fp32_to_e8mx_e8m3_known_values(self) -> None:
        """E8M3 has 3 mantissa bits."""
        x = torch.tensor([0.0, 1.0, 1.125, 2.0, 7.5], device=self.device)
        result = fp32_to_e8mx(x, ElemDtype.E8M3)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result[0].item(), 0.0)
        self.assertEqual(result[1].item(), 1.0)
        self.assertEqual(result[2].item(), 1.125)

    def test_fp32_to_e8mx_preserves_sign(self) -> None:
        x = torch.tensor([-1.0, -2.0, 1.0, 2.0], device=self.device)
        result = fp32_to_e8mx(x, ElemDtype.E8M1)
        self.assertTrue(result[0].item() < 0)
        self.assertTrue(result[1].item() < 0)
        self.assertTrue(result[2].item() > 0)
        self.assertTrue(result[3].item() > 0)

    def test_fp32_to_e8mx_invalid_mbits_raises(self) -> None:
        x = torch.tensor([1.0], device=self.device)
        with self.assertRaises((ValueError, AssertionError)):
            fp32_to_e8mx(x, ElemDtype.E2M1)  # ebits=2, not 8
        self.assertTrue(True, "Expected exception was raised")

    def test_fp32_floor_fp8_known_values(self) -> None:
        """Floor rounding to FP8 E4M3 grid."""
        # 1.0 is exactly representable in FP8
        x = torch.tensor([1.0], device=self.device)
        result = fp32_floor_fp8(x)
        self.assertEqual(result[0].item(), 1.0)

    def test_fp32_floor_fp8_rounds_down(self) -> None:
        """Values between grid points should round down."""
        # 1.0 and 1.125 are both representable in FP8 E4M3
        # 1.1 should floor to 1.0
        x = torch.tensor([1.1], device=self.device)
        result = fp32_floor_fp8(x)
        self.assertLessEqual(result[0].item(), 1.1)
        self.assertGreaterEqual(result[0].item(), 1.0)

    def test_fp32_floor_fp8_negative(self) -> None:
        x = torch.tensor([-1.1], device=self.device)
        result = fp32_floor_fp8(x)
        # For negative values, floor means towards -inf
        self.assertTrue(result[0].item() < 0)

    def test_fp32_ceil_fp8_known_values(self) -> None:
        """Ceil rounding to FP8 E4M3 grid."""
        x = torch.tensor([1.0], device=self.device)
        result = fp32_ceil_fp8(x)
        self.assertEqual(result[0].item(), 1.0)

    def test_fp32_ceil_fp8_rounds_up(self) -> None:
        """Values between grid points should round up."""
        x = torch.tensor([1.1], device=self.device)
        result = fp32_ceil_fp8(x)
        self.assertGreaterEqual(result[0].item(), 1.1)

    def test_fp32_floor_ceil_fp8_consistency(self) -> None:
        """For any value, floor <= value <= ceil."""
        x = torch.tensor([0.3, 1.5, 3.7, 100.0, 0.001], device=self.device)
        floored = fp32_floor_fp8(x)
        ceiled = fp32_ceil_fp8(x)
        for i in range(len(x)):
            self.assertLessEqual(
                floored[i].item(),
                x[i].item() + 1e-6,
                f"floor({x[i].item()})={floored[i].item()} should be <= {x[i].item()}",
            )
            self.assertGreaterEqual(
                ceiled[i].item(),
                x[i].item() - 1e-6,
                f"ceil({x[i].item()})={ceiled[i].item()} should be >= {x[i].item()}",
            )

    def test_calculate_fp8_scale_for_nvfp4_tensorwise(self) -> None:
        """Test 2-level FP8 scale calculation with tensorwise granularity."""
        local_amax = torch.tensor([1.0, 2.0, 3.0, 4.0], device=self.device)
        global_amax = torch.tensor([4.0], device=self.device)
        max_pos = 6.0
        scale, unscale, local_unscale_fp8, two_level_scale = (
            calculate_fp8_scale_for_nvfp4(
                global_amax=global_amax,
                local_amax=local_amax,
                max_pos=max_pos,
                global_scale_granularity=GlobalScaleGranularity.TENSORWISE,
                num_blocks_row=2,
                num_blocks_col=2,
                scale_round_mode=ScaleRoundMode.NONE,
            )
        )
        self.assertEqual(scale.shape, local_amax.shape)
        self.assertEqual(unscale.shape, local_amax.shape)
        # Scale * unscale should be approximately 1
        product = scale * unscale
        for val in product.tolist():
            self.assertAlmostEqual(val, 1.0, places=3)

    def test_calculate_fp8_scale_for_nvfp4_rowwise(self) -> None:
        """Test with rowwise granularity."""
        local_amax = torch.tensor([1.0, 2.0, 3.0, 4.0], device=self.device)
        global_amax = torch.tensor([3.0, 4.0], device=self.device)
        max_pos = 6.0
        scale, unscale, _, _ = calculate_fp8_scale_for_nvfp4(
            global_amax=global_amax,
            local_amax=local_amax,
            max_pos=max_pos,
            global_scale_granularity=GlobalScaleGranularity.ROWWISE,
            num_blocks_row=2,
            num_blocks_col=2,
            scale_round_mode=ScaleRoundMode.NONE,
        )
        self.assertEqual(scale.shape, local_amax.shape)

    def test_calculate_fp8_scale_for_nvfp4_floor_round(self) -> None:
        """Test with FLOOR scale rounding."""
        local_amax = torch.tensor([1.0, 2.0, 3.0, 4.0], device=self.device)
        global_amax = torch.tensor([4.0], device=self.device)
        max_pos = 6.0
        scale, unscale, _, _ = calculate_fp8_scale_for_nvfp4(
            global_amax=global_amax,
            local_amax=local_amax,
            max_pos=max_pos,
            global_scale_granularity=GlobalScaleGranularity.TENSORWISE,
            num_blocks_row=2,
            num_blocks_col=2,
            scale_round_mode=ScaleRoundMode.FLOOR,
        )
        self.assertEqual(scale.shape, local_amax.shape)

    def test_calculate_fp8_scale_for_nvfp4_ceil_round(self) -> None:
        """Test with CEIL scale rounding."""
        local_amax = torch.tensor([1.0, 2.0, 3.0, 4.0], device=self.device)
        global_amax = torch.tensor([4.0], device=self.device)
        max_pos = 6.0
        scale, unscale, _, _ = calculate_fp8_scale_for_nvfp4(
            global_amax=global_amax,
            local_amax=local_amax,
            max_pos=max_pos,
            global_scale_granularity=GlobalScaleGranularity.TENSORWISE,
            num_blocks_row=2,
            num_blocks_col=2,
            scale_round_mode=ScaleRoundMode.CEIL,
        )
        self.assertEqual(scale.shape, local_amax.shape)

    def test_calculate_fp8_scale_for_nvfp4_zero_amax(self) -> None:
        """Zero in local_amax should not cause NaN/Inf."""
        local_amax = torch.tensor([0.0, 1.0, 2.0, 0.0], device=self.device)
        global_amax = torch.tensor([2.0], device=self.device)
        max_pos = 6.0
        scale, unscale, _, _ = calculate_fp8_scale_for_nvfp4(
            global_amax=global_amax,
            local_amax=local_amax,
            max_pos=max_pos,
            global_scale_granularity=GlobalScaleGranularity.TENSORWISE,
            num_blocks_row=2,
            num_blocks_col=2,
            scale_round_mode=ScaleRoundMode.NONE,
        )
        # Should not contain NaN or Inf
        self.assertFalse(torch.isnan(scale).any().item())
        self.assertFalse(torch.isinf(scale).any().item())
        self.assertFalse(torch.isnan(unscale).any().item())

    # -----------------------------------------------------------------
    # fp32_to_fpx direct known-value tests. Grid points are exactly
    # representable, so quantize-dequantize is the identity on them.
    # -----------------------------------------------------------------
    def _assert_fpx_identity(self, elem_dtype: ElemDtype, grid: list[float]) -> None:
        max_pos = ELEM_DTYPE_RANGE_MAP[QuantMode.SYMM][elem_dtype][1]
        x = torch.tensor(grid, dtype=torch.float32, device=self.device)
        result = fp32_to_fpx(x, elem_dtype, max_pos)
        torch.testing.assert_close(result, x, rtol=0, atol=0)

    def test_fp32_to_fpx_e2m1_grid(self) -> None:
        # E2M1 positive grid: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
        grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        self._assert_fpx_identity(ElemDtype.E2M1, grid)
        # sign is preserved for negatives
        self._assert_fpx_identity(
            ElemDtype.E2M1, [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5]
        )

    def test_fp32_to_fpx_e2m3_grid(self) -> None:
        # E2M3 has 3 mantissa bits; a few representative points incl. max 7.5.
        self._assert_fpx_identity(ElemDtype.E2M3, [0.0, 0.5, 1.0, 1.125, 7.5])
        self._assert_fpx_identity(ElemDtype.E2M3, [-7.5, -1.125, -1.0, -0.5])

    def test_fp32_to_fpx_e3m0_grid(self) -> None:
        # E3M0 has 0 mantissa bits -> only powers of two (min_exp = -2).
        self._assert_fpx_identity(
            ElemDtype.E3M0, [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        )

    def test_fp32_to_fpx_saturates_above_max(self) -> None:
        # Inputs above the element max saturate (clamp) to +/- max_pos.
        for elem_dtype in (ElemDtype.E2M1, ElemDtype.E2M3, ElemDtype.E3M0):
            with self.subTest(elem_dtype=elem_dtype):
                max_pos = ELEM_DTYPE_RANGE_MAP[QuantMode.SYMM][elem_dtype][1]
                x = torch.tensor(
                    [100.0, -100.0], dtype=torch.float32, device=self.device
                )
                result = fp32_to_fpx(x, elem_dtype, max_pos)
                expected = torch.tensor(
                    [max_pos, -max_pos], dtype=torch.float32, device=self.device
                )
                torch.testing.assert_close(result, expected)

    # -----------------------------------------------------------------
    # fp32_floor_fp8 / fp32_ceil_fp8 saturation + negative-subnormal
    # branches.
    # -----------------------------------------------------------------
    def test_fp32_floor_ceil_fp8_saturation(self) -> None:
        # E4M3 max_normal == 480; values >= it take the saturate branch (no
        # subnormal shift) and are just mantissa-truncated (no clamp to 448).
        x = torch.tensor([1000.0], device=self.device)
        self.assertEqual(fp32_floor_fp8(x).item(), 960.0)
        self.assertEqual(fp32_ceil_fp8(x).item(), 1024.0)

    def test_fp32_floor_ceil_fp8_negative_subnormal(self) -> None:
        # |x| < min_normal (E4M3 min_normal == 2**-6) exercises the denormal
        # shift branch with a negative sign.
        x = torch.tensor([-0.001], device=self.device)
        floored = fp32_floor_fp8(x)
        ceiled = fp32_ceil_fp8(x)
        self.assertFalse(torch.isnan(floored).any().item())
        self.assertFalse(torch.isnan(ceiled).any().item())
        # floor truncates magnitude toward zero -> -0.0
        self.assertEqual(floored.item(), 0.0)
        # ceil rounds magnitude away from zero to the next subnormal step
        self.assertAlmostEqual(ceiled.item(), -0.001953125, places=9)
        self.assertLessEqual(ceiled.item(), 0.0)


class TestFp32ToE8m0ExactPowersOfTwo(unittest.TestCase):
    def setUp(self) -> None:
        self.devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    def test_unscale_is_exact_power_of_two_for_every_exponent(self) -> None:
        # All 255 finite E8M0 values, 2**-127 (the fp32 subnormal 0x00400000)
        # through 2**127. CUDA exp2(-127.) is one subnormal ULP short.
        exponents = torch.arange(-127, 128)
        # FLOOR maps 1.5 * 2**k to 2**k (2**127 itself for k = 127).
        x = torch.ldexp(
            torch.full((exponents.numel(),), 1.5, dtype=torch.float64), exponents
        )
        x[-1] = 2.0**127
        expected = torch.ldexp(torch.ones_like(x), exponents).to(torch.float32)
        self.assertEqual(expected[0].view(torch.int32).item(), 0x00400000)
        for device in self.devices:
            with self.subTest(device=device):
                _, unscale = fp32_to_e8m0(
                    x.to(torch.float32).to(device),
                    elem_dtype=ElemDtype.E4M3,
                    scale_round_mode=ScaleRoundMode.FLOOR,
                )
                self.assertTrue(
                    torch.equal(
                        unscale.cpu().view(torch.int32), expected.view(torch.int32)
                    )
                )

    def test_scale_below_the_e8m0_range_clamps_to_its_smallest_value(self) -> None:
        # A scale below 2**-127 is not representable; it clamps to 2**-127, for
        # the amax norm (input amax / max_pos) and the pow2max norm (input amax,
        # exponent shifted by norm_exp) alike. A zero block gets it too.
        tiny = torch.tensor(
            [0.0, 2.0**-149, 1e-42, 2.0**-130, 2.0**-128, 1e-38], dtype=torch.float32
        )
        smallest = torch.full_like(tiny, 2.0**-127)
        for device in self.devices:
            for norm_exp in (0, 2, 8):
                for mode in (ScaleRoundMode.FLOOR, ScaleRoundMode.OAS):
                    with self.subTest(device=device, norm_exp=norm_exp, mode=mode):
                        _, unscale = fp32_to_e8m0(
                            tiny.to(device),
                            elem_dtype=ElemDtype.E2M1,
                            scale_round_mode=mode,
                            norm_exp=norm_exp,
                        )
                        got = unscale.cpu()
                        if norm_exp == 0:
                            # OAS rounds 1e-38 up to 2**-126 unless it is shifted.
                            got, want = got[:-1], smallest[:-1]
                        else:
                            want = smallest
                        self.assertTrue(
                            torch.equal(got.view(torch.int32), want.view(torch.int32))
                        )


class TestNvfp4GlobalScale(unittest.TestCase):
    def setUp(self) -> None:
        self.devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    @staticmethod
    def _amaxes() -> torch.Tensor:
        # Every positive finite bf16 value (subnormals included), random fp32
        # bit patterns over the whole positive finite range, and edge values:
        # the fp32 subnormal and normal limits, FLT_MAX, and the amaxes around
        # 6 / FLT_MAX (where the fp32 quotient overflows) and around
        # 2688 / FLT_MAX (below which the scale saturates).
        bf16 = torch.arange(1, 0x7F80, dtype=torch.int32).to(torch.int16)
        bf16 = bf16.view(torch.bfloat16).to(torch.float32)
        generator = torch.Generator().manual_seed(0)
        random = torch.randint(1, 0x7F800000, (20000,), generator=generator)
        random = random.to(torch.int32).view(torch.float32)
        f32 = torch.finfo(torch.float32)
        edge = torch.tensor(
            [2.0**-149, f32.smallest_normal * (1 - 2.0**-23), f32.smallest_normal]
            + [f32.max, 1.0, 2688.0, 5.0, 10.0, 11.0, 4.5, 2.75, 29.0, 31.0, 37.0],
            dtype=torch.float32,
        )
        steps = torch.arange(-4, 5, dtype=torch.int32)
        near = []
        for c in (6.0, 2688.0):
            bits = torch.tensor(c / f32.max, dtype=torch.float32).view(torch.int32)
            near.append((bits + steps).view(torch.float32))
        return torch.cat([bf16, random, edge, *near])

    def test_matches_the_reference(self) -> None:
        amaxes = self._amaxes()
        # E2M1, E2M3, E4M3, and a value that is not an fp32 number.
        for max_pos in (6.0, 7.5, 448.0, 5.4):
            expected = torch.tensor(
                [reference_nvfp4_global_scale(a, max_pos) for a in amaxes.tolist()],
                dtype=torch.float64,
            )
            for device in self.devices:
                with self.subTest(max_pos=max_pos, device=device):
                    got = nvfp4_global_scale(amaxes.to(device), max_pos).cpu()
                    self.assertEqual(got.dtype, torch.float64)
                    mismatch = got.view(torch.int64) != expected.view(torch.int64)
                    self.assertEqual(
                        int(mismatch.sum()),
                        0,
                        f"first amaxes off: {amaxes[mismatch][:4].tolist()}",
                    )
            if max_pos == 6.0:
                # Below 2688 / FLT_MAX the scale saturates to FLT_MAX, above it
                # is the quotient.
                f32_max = torch.finfo(torch.float32).max
                self.assertTrue(bool((expected[-18:] <= f32_max).all()))
                self.assertTrue(bool((expected[-9:-5] == f32_max).all()))
                self.assertTrue(bool((expected[-4:] < f32_max).all()))

    def test_matches_the_unsaturated_expression(self) -> None:
        # Wherever it does not saturate, the scale is bitwise the expression
        # `448 * fp64(fp32(max_pos) / amax)`, for fp32, bf16 and fp16 amaxes.
        amaxes = self._amaxes()
        f32_max = torch.finfo(torch.float32).max
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            amax = amaxes.to(dtype)
            amax = amax[torch.isfinite(amax) & (amax > 0)]
            for device in self.devices:
                with self.subTest(dtype=dtype, device=device):
                    amax_d = amax.to(device)
                    got = nvfp4_global_scale(amax_d, 6.0)
                    amax32 = amax_d.to(torch.float32)
                    want = 448.0 * (torch.full_like(amax32, 6.0) / amax32).to(
                        torch.float64
                    )
                    want = want.clamp(max=f32_max)
                    self.assertTrue(
                        torch.equal(got.view(torch.int64), want.view(torch.int64))
                    )

    def test_invalid_amax_and_shapes(self) -> None:
        amax = torch.tensor([0.0, float("inf"), float("nan")])
        got = nvfp4_global_scale(amax, 6.0)
        self.assertEqual(got[0].item(), float("inf"))
        self.assertEqual(got[1].item(), 0.0)
        self.assertTrue(bool(torch.isnan(got[2])))
        self.assertEqual(nvfp4_global_scale(torch.tensor(5.0), 6.0).shape, (1,))
        self.assertEqual(nvfp4_global_scale(torch.ones(3, 1), 6.0).shape, (3, 1))


if __name__ == "__main__":
    unittest.main()
