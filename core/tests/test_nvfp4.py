# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from dataclasses import replace

import torch
from parameterized import parameterized
from quant_suite_core import (
    BlockSize,
    ElemDtype,
    ElemRoundMode,
    fake_quantize,
    GlobalScaleGranularity,
    QuantConfig,
    QuantInfo,
    QuantMode,
    ScaleDtype,
    ScaleNormMode,
    ScaleRoundMode,
)
from quant_suite_core.fake_presets import FAKE_PRESETS
from quant_suite_core.fake_quant import fake_quantize_grouped_nvfp4
from quant_suite_core.native_quant import native_dequantize, native_quantize
from quant_suite_core.qdq_pytorch import qdq
from quant_suite_core.qdq_triton import qdq_triton_mega
from quant_suite_core.quant_primitives import calculate_fp8_scale_for_nvfp4
from quant_test_utils import make_quant_config, skip_if_no_cuda


@skip_if_no_cuda
class TestGroupedNVFP4QDQ(unittest.TestCase):
    def test_matches_independent_per_group_qdq(self) -> None:
        torch.manual_seed(0)
        config = QuantConfig.from_dict(FAKE_PRESETS["nvfp4"])
        group_counts = torch.tensor([2, 0, 3, 1], device="cuda", dtype=torch.int32)
        groups = [
            torch.randn((2, 64), device="cuda", dtype=torch.bfloat16) * 0.125,
            torch.randn((3, 64), device="cuda", dtype=torch.bfloat16) * 8.0,
            torch.randn((1, 64), device="cuda", dtype=torch.bfloat16) * 128.0,
        ]
        tensor = torch.cat(groups)

        actual = fake_quantize_grouped_nvfp4(tensor, "tensor", config, group_counts)
        expected = torch.cat(
            [qdq(group, "grouped_nvfp4_reference", config) for group in groups]
        )

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_static_group_scales_match_dynamic_group_scales(self) -> None:
        torch.manual_seed(1)
        config = QuantConfig.from_dict(FAKE_PRESETS["nvfp4"])
        group_counts = torch.tensor([2, 0, 3], device="cuda", dtype=torch.int32)
        groups = [
            torch.randn((2, 64), device="cuda", dtype=torch.bfloat16) * 0.25,
            torch.randn((3, 64), device="cuda", dtype=torch.bfloat16) * 16.0,
        ]
        tensor = torch.cat(groups)
        group_amax = torch.stack(
            (
                groups[0].abs().amax(),
                torch.tensor(2688.0, device="cuda"),
                groups[1].abs().amax(),
            )
        )
        global_scales = 448.0 * (torch.full_like(group_amax, 6.0) / group_amax).to(
            torch.float64
        )

        actual = fake_quantize_grouped_nvfp4(
            tensor,
            "tensor",
            config,
            group_counts,
            global_scale=global_scales,
        )
        expected = fake_quantize_grouped_nvfp4(tensor, "tensor", config, group_counts)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_scalar_static_scale_is_shared_by_all_groups(self) -> None:
        torch.manual_seed(2)
        config = QuantConfig.from_dict(FAKE_PRESETS["nvfp4"])
        group_counts = torch.tensor([2, 3], device="cuda", dtype=torch.int32)
        tensor = torch.randn((5, 64), device="cuda", dtype=torch.bfloat16)
        global_scale = torch.tensor(672.0, device="cuda", dtype=torch.float32)

        shared = fake_quantize_grouped_nvfp4(
            tensor,
            "tensor",
            config,
            group_counts,
            global_scale=global_scale,
        )
        repeated = fake_quantize_grouped_nvfp4(
            tensor,
            "tensor",
            config,
            group_counts,
            global_scale=global_scale.repeat(2),
        )

        torch.testing.assert_close(shared, repeated, rtol=0, atol=0)


@skip_if_no_cuda
class TestNVFP4TinyGlobalAmax(unittest.TestCase):
    """A tensor (or row) whose amax is below 2688 / FLT_MAX (about 7.9e-36)
    gets the global scale FLT_MAX and quantizes like any other, in every
    backend. Its exact scale exceeds FLT_MAX; without saturation the tensor
    would quantize to zeros."""

    def test_every_backend(self) -> None:
        f32_max = torch.finfo(torch.float32).max
        for preset in ("nvfp4", "rowwise_nvfp4"):
            config = QuantConfig.from_dict(FAKE_PRESETS[preset])
            for magnitude in (1e-37, 1e-40, 1e-44):
                torch.manual_seed(0)
                x = (torch.randn(8, 64) * magnitude).cuda()
                what = f"{preset} {magnitude}"
                info = qdq(x.cpu(), "x", config, return_quant_info=True)
                assert info.global_scale is not None
                self.assertTrue(bool((info.global_scale == f32_max).all()), what)
                ref = info.dequant_data
                self.assertTrue(bool(torch.isfinite(ref).all()), what)
                outs = {
                    "qdq cuda": qdq(x, "x", config),
                    "mega": qdq_triton_mega(x, "x", config),
                }
                for use_triton in (True, False):
                    result = native_quantize(x, "x", config, use_triton=use_triton)
                    assert result.global_scale is not None
                    self.assertTrue(bool((result.global_scale == f32_max).all()), what)
                    outs[f"native use_triton={use_triton}"] = native_dequantize(
                        result, output_dtype=torch.float32
                    )
                if preset == "nvfp4":
                    counts = torch.tensor([8], device="cuda", dtype=torch.int32)
                    outs["grouped"] = fake_quantize_grouped_nvfp4(
                        x, "x", config, counts
                    )
                for label, out in outs.items():
                    self.assertTrue(
                        torch.equal(out.cpu().view(torch.int32), ref.view(torch.int32)),
                        f"{what}: {label} != qdq cpu",
                    )
                if magnitude == 1e-37:
                    # As accurate as a tensor of normal magnitude.
                    err = (ref - x.cpu()).abs().max() / x.abs().max().cpu()
                    self.assertLess(err.item(), 0.2, what)


@skip_if_no_cuda
class TestNVFP4AllZeroQuantInfo(unittest.TestCase):
    """An all-zero tensor (or row) has no global scale: every backend reports
    1.0 for it, block scales 0, and zeros."""

    def test_every_backend(self) -> None:
        zeros = torch.zeros(32, 64)
        one_row = zeros.clone()
        one_row[1] = torch.randn(64, generator=torch.Generator().manual_seed(0))
        for preset, x in (
            ("nvfp4", zeros),
            ("nvfp4_16x16", zeros),
            ("rowwise_nvfp4", zeros),
            ("rowwise_nvfp4", one_row),
        ):
            config = QuantConfig.from_dict(FAKE_PRESETS[preset])
            zero_rows = x.abs().amax(dim=1) == 0
            for dtype in (torch.float32, torch.bfloat16):
                xd = x.to(dtype)
                what = f"{preset} {dtype} nonzero rows={int((~zero_rows).sum())}"
                infos = {
                    "qdq cpu": qdq(xd, "x", config, return_quant_info=True),
                    "qdq cuda": qdq(xd.cuda(), "x", config, return_quant_info=True),
                    "mega": qdq_triton_mega(
                        xd.cuda(), "x", config, return_quant_info=True
                    ),
                }
                ref = infos["qdq cpu"]
                assert ref.global_scale is not None
                gs = ref.global_scale.reshape(-1)
                if preset == "rowwise_nvfp4":
                    self.assertTrue(bool((gs[zero_rows] == 1.0).all()), what)
                    scales = ref.local_unscale.reshape(len(x), -1)[zero_rows]
                else:
                    self.assertEqual(gs.tolist(), [1.0], what)
                    scales = ref.local_unscale
                self.assertTrue(bool((scales == 0).all()), what)
                for label, info in infos.items():
                    assert info.global_scale is not None
                    zeros_out = info.dequant_data.cpu()[zero_rows]
                    self.assertTrue(
                        torch.equal(zeros_out, xd[zero_rows]), f"{what}: {label}"
                    )
                    fields = ("dequant_data", "quantized_val", "local_unscale")
                    for field in ("global_scale", *fields):
                        got = getattr(info, field).cpu().float()
                        want = getattr(ref, field).float()
                        self.assertTrue(
                            torch.equal(got.view(torch.int32), want.view(torch.int32)),
                            f"{what}: {label} {field} {got.reshape(-1)[:4].tolist()}",
                        )
                if preset == "nvfp4_16x16":
                    continue
                for use_triton in (True, False):
                    result = native_quantize(
                        xd.cuda(), "x", config, use_triton=use_triton
                    )
                    assert result.global_scale is not None
                    # The same value as qdq() reports, in its dtype.
                    native_gs = result.global_scale.reshape(-1).cpu()
                    self.assertTrue(
                        torch.equal(
                            native_gs.to(ref.global_scale.dtype),
                            ref.global_scale.reshape(-1),
                        ),
                        f"{what}: native {use_triton=}",
                    )


@skip_if_no_cuda
class TestNVFP4QuantScaleRoundMode(unittest.TestCase):
    """Test scale_round_mode for nvfp4 quantization with manual examples.

    Manual example with a 1x64 tensor (4 blocks of 16 elements each):
    - Block 1: amax = 1.3, Block 2: amax = 448.0, Block 3: amax = 1.4 * 2^-9, Block 4: amax = 1.6 * 2^-9
    - Global amax = 448.0
    - global_scale = 448.0 * (6.0 / 448.0) = 6.0

    Block 1 (amax = 1.3):
    - local_unscale_before_fp8 = (1.3 / 6.0) * 6.0 = 1.3
    - With scale_round_mode='none', local_unscale_fp8 = 1.25
    - With scale_round_mode='floor', local_unscale_fp8 = 1.25
    - With scale_round_mode='ceil', local_unscale_fp8 = 1.375

    Block 2 (amax = 448.0):
    - local_unscale_before_fp8 = (448.0 / 6.0) * 6.0 = 448.0
    - With all scale_round_modes, local_unscale_fp8 = 448.0 (exact fp8 representation)

    Block 3 (amax = 1.4 * 2^-9):
    - Test subnormal number, min subnormal number of fp8_e4m3 = 2^-9
    - local_unscale_before_fp8 = (1.4 * 2^-9 / 6.0) * 6.0 = 1.4 * 2^-9
    - With scale_round_mode='none', local_unscale_fp8 = 2^-9
    - With scale_round_mode='floor', local_unscale_fp8 = 2^-9
    - With scale_round_mode='ceil', local_unscale_fp8 = 2^-8

    Block4 (amax = 1.6 * 2^-9):
    - Test subnormal number, min subnormal number of fp8_e4m3 = 2^-9
    - local_unscale_before_fp8 = (1.6 * 2^-9 / 6.0)
    - With scale_round_mode='none', local_unscale_fp8 = 2^-8
    - With scale_round_mode='floor', local_unscale_fp8 = 2^-9
    - With scale_round_mode='ceil', local_unscale_fp8 = 2^-8
    """

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.device = "cuda"

    def _create_nvfp4_config(self, scale_round_mode: str) -> QuantConfig:
        """Create a nvfp4 quantization config with the specified scale_round_mode."""
        return make_quant_config(
            scale_dtype=ScaleDtype.FP8,
            scale_norm_mode=ScaleNormMode.AMAX,
            scale_round_mode=ScaleRoundMode(scale_round_mode),
            global_scale_granularity=GlobalScaleGranularity.TENSORWISE,
            elem_dtype=ElemDtype.E2M1,
            elem_round_mode=ElemRoundMode.NEAREST,
            block_size=BlockSize(1, 16),
            global_scale_dtype=ScaleDtype.FP32,
        )

    def _create_test_tensor(self) -> torch.Tensor:
        """Create a 1x64 test tensor with 4 blocks of 16 elements each.

        Block 1: amax = 1.3
        Block 2: amax = 448.0
        Block 3: amax = 1.4 * 2^-9
        Block 4: amax = 1.6 * 2^-9
        """
        return torch.tensor(
            [
                [
                    # Block 1: amax = 1.3
                    1.3,
                    1.0,
                    0.5,
                    0.0,
                    -0.5,
                    -1.0,
                    -1.3,
                    0.25,
                    0.125,
                    0.0625,
                    -0.125,
                    -0.25,
                    0.75,
                    -0.75,
                    1.25,
                    -1.25,
                    # Block 2: amax = 448.0
                    448.0,
                    2.0,
                    1.0,
                    0.0,
                    -1.0,
                    -2.0,
                    -448.0,
                    0.5,
                    0.25,
                    0.125,
                    -0.25,
                    -0.5,
                    1.5,
                    -1.5,
                    2.5,
                    -2.5,
                    # Block 3: amax = 1.4 * 2^-9
                    1.4 * 2**-9,
                    1.0 * 2**-9,
                    0.5 * 2**-9,
                    0,
                    -0.5 * 2**-9,
                    -1.0 * 2**-9,
                    -1.4 * 2**-9,
                    0.25 * 2**-9,
                    0.125 * 2**-9,
                    0.0625 * 2**-9,
                    -0.125 * 2**-9,
                    -0.25 * 2**-9,
                    0.75 * 2**-9,
                    -0.75 * 2**-9,
                    0,
                    0.1 * 2**-9,
                    # Block 4: amax = 1.6 * 2^-9
                    1.6 * 2**-9,
                    1.0 * 2**-9,
                    0.5 * 2**-9,
                    0,
                    -0.5 * 2**-9,
                    -1.0 * 2**-9,
                    -1.6 * 2**-9,
                    0.25 * 2**-9,
                    0.125 * 2**-9,
                    0.0625 * 2**-9,
                    -0.125 * 2**-9,
                    -0.25 * 2**-9,
                    0.75 * 2**-9,
                    -0.75 * 2**-9,
                    0,
                    0.1 * 2**-9,
                ]
            ],
            dtype=torch.bfloat16,
            device=self.device,
        )

    def test_scale_round_mode_none_nvfp4(self) -> None:
        """Test scale_round_mode='none' uses default fp8 rounding (round to nearest even)."""
        config = self._create_nvfp4_config("none")
        tensor = self._create_test_tensor()

        result_non_triton = fake_quantize(
            tensor,
            "test",
            config,
            transpose=False,
            use_triton=False,
            return_quant_info=True,
        )
        result_triton = fake_quantize(
            tensor,
            "test",
            config,
            transpose=False,
            use_triton=True,
            return_quant_info=True,
        )

        # Make sure both non-triton and triton results are the same

        # global_scale = global_amax * (elem_max / scale_max) = 448.0 * (6.0 / 448.0) = 6.0
        self.assertAlmostEqual(result_non_triton.global_scale.item(), 6.0, places=4)
        self.assertAlmostEqual(result_triton.global_scale.item(), 6.0, places=4)

        # Block 1: local_unscale = 1.3 -> fp8 round to nearest even = 1.25
        # Block 2: local_unscale = 448.0 -> fp8 exact representation = 448.0
        expected_local_unscale = torch.tensor(
            [1.25, 448.0, 2**-9, 2**-8], dtype=torch.float32, device=self.device
        )
        torch.testing.assert_close(
            result_non_triton.local_unscale, expected_local_unscale
        )
        torch.testing.assert_close(result_triton.local_unscale, expected_local_unscale)

    def test_scale_round_mode_floor_nvfp4(self) -> None:
        """Test scale_round_mode='floor' truncates mantissa (rounds towards zero)."""
        config = self._create_nvfp4_config("floor")
        tensor = self._create_test_tensor()

        result_non_triton = fake_quantize(
            tensor,
            "test",
            config,
            transpose=False,
            use_triton=False,
            return_quant_info=True,
        )
        result_triton = fake_quantize(
            tensor,
            "test",
            config,
            transpose=False,
            use_triton=True,
            return_quant_info=True,
        )

        # Make sure both non-triton and triton results are the same

        # global_scale = global_amax * (elem_max / scale_max) = 448.0 * (6.0 / 448.0) = 6.0
        self.assertAlmostEqual(result_non_triton.global_scale.item(), 6.0, places=4)
        self.assertAlmostEqual(result_triton.global_scale.item(), 6.0, places=4)

        # Block 1: local_unscale = 1.3 -> fp8 floor = 1.25
        # Block 2: local_unscale = 448.0 -> fp8 exact representation = 448.0
        expected_local_unscale = torch.tensor(
            [1.25, 448.0, 2**-9, 2**-9], dtype=torch.float32, device=self.device
        )
        torch.testing.assert_close(
            result_non_triton.local_unscale, expected_local_unscale
        )
        torch.testing.assert_close(result_triton.local_unscale, expected_local_unscale)

    def test_scale_round_mode_ceil_nvfp4(self) -> None:
        """Test scale_round_mode='ceil' rounds up the scale."""
        config = self._create_nvfp4_config("ceil")
        tensor = self._create_test_tensor()

        result_non_triton = fake_quantize(
            tensor,
            "test",
            config,
            transpose=False,
            use_triton=False,
            return_quant_info=True,
        )
        result_triton = fake_quantize(
            tensor,
            "test",
            config,
            transpose=False,
            use_triton=True,
            return_quant_info=True,
        )

        # Make sure both non-triton and triton results are the same

        # global_scale = global_amax * (elem_max / scale_max) = 448.0 * (6.0 / 448.0) = 6.0
        self.assertAlmostEqual(result_non_triton.global_scale.item(), 6.0, places=4)
        self.assertAlmostEqual(result_triton.global_scale.item(), 6.0, places=4)

        # Block 1: local_unscale = 1.3 -> fp8 ceil = 1.375
        # Block 2: local_unscale = 448.0 -> fp8 exact representation = 448.0
        expected_local_unscale = torch.tensor(
            [1.375, 448.0, 2**-8, 2**-8], dtype=torch.float32, device=self.device
        )
        torch.testing.assert_close(
            result_non_triton.local_unscale, expected_local_unscale
        )
        torch.testing.assert_close(result_triton.local_unscale, expected_local_unscale)


class TestNVFP4MidpointRounding(unittest.TestCase):
    """Test that NVFP4 fake_quantize rounds exact E2M1 midpoints with RNE.

    A single-factor `data_lp = data_hp * scale.unsqueeze(1)` (scale = inverse
    of per_block_scale_fp8 / global_scale, cast to fp32) drifts ~1 ULP off
    algebraic E2M1 midpoints. `qdq` computes
    `data_lp = (data_hp * global_scale_fp32) / per_block_scale_fp32`
    directly, which preserves exact midpoints.

    The drift only manifests when `(per_block_scale_fp8 / global_scale)`
    is not bit-exact in fp32 — i.e., when the per-block FP8 byte is not
    the FP8 max (448). This test fixture forces a non-max per-block FP8
    byte (= 0x64 = 48) by pairing the test block with a padding block
    whose amax is much larger than the test block's amax.
    """

    def _build_nvfp4_config(self) -> QuantConfig:
        return QuantConfig(
            quant_mode=QuantMode.SYMM,
            scale_dtype=ScaleDtype.FP8,
            scale_norm_mode=ScaleNormMode.AMAX,
            scale_round_mode=ScaleRoundMode.NONE,
            global_scale_granularity=GlobalScaleGranularity.TENSORWISE,
            elem_dtype=ElemDtype.E2M1,
            elem_round_mode=ElemRoundMode.NEAREST,
            block_size=BlockSize(1, 16),
            global_scale_dtype=ScaleDtype.FP32,
        )

    def _build_midpoint_tensor(self) -> torch.Tensor:
        # Block 0: bf16 raw values chosen so that, after multiplying by
        # the resulting global_scale = 33363.78125 and dividing by the
        # per-block FP8 byte (= 0x64 = 48), element 15 lands EXACTLY on
        # the E2M1 midpoint 3.5 (between codes 5 and 6). RNE on the E2M1
        # mantissa LSB requires picking the even-LSB neighbor: 4.0
        # (code 6), NOT 3.0 (code 5).
        block0 = [
            0.00738525390625,
            0.0029296875,
            -0.00421142578125,
            -0.0025787353515625,
            -0.002685546875,
            0.0009002685546875,
            -0.0087890625,
            -0.006683349609375,
            0.0084228515625,
            -7.009506225585938e-05,
            0.0042724609375,
            0.00183868408203125,
            -0.004913330078125,
            -0.0042724609375,
            0.000732421875,
            0.005035400390625,  # ← element 15: scaled = 168.0, midpoint 3.5
        ]
        # Block 1: padding so the tensor's global_amax = 0.08056640625,
        # giving global_scale = (448*6) / 0.08056640625 = 33363.78125.
        block1 = [0.08056640625] + [0.0] * 15
        return torch.tensor([block0, block1], dtype=torch.bfloat16)

    @parameterized.expand(
        [
            ("eager", False),
            ("triton", True),
        ]
    )
    def test_e2m1_midpoint_with_non_max_per_block_scale(
        self, _name: str, use_triton: bool
    ) -> None:
        tensor = self._build_midpoint_tensor()
        config = self._build_nvfp4_config()
        if use_triton:
            if not torch.cuda.is_available():
                self.skipTest("Skip when cuda is not available")
            tensor = tensor.cuda()

        # Sanity: the test fixture must reproduce the expected scaling.
        self.assertEqual(float(tensor.abs().max()), 0.08056640625)

        qt = fake_quantize(
            tensor=tensor,
            tensor_name="midpoint_test",
            config=config,
            transpose=False,
            use_triton=use_triton,
            return_quant_info=True,
        )
        self.assertIsInstance(qt, QuantInfo)
        self.assertIsNotNone(qt.global_scale)

        # Fixture preconditions: global_scale and per-block FP8 byte must
        # be exactly these values for the midpoint case to actually fire.
        self.assertEqual(
            float(qt.global_scale.reshape(-1)[0].item()),
            33363.78125,
            "global_scale must equal (448*6)/global_amax = 33363.78125",
        )
        per_block_scale_fp8_block0 = qt.local_unscale.reshape(-1)[0]
        self.assertEqual(
            per_block_scale_fp8_block0.to(torch.float32).item(),
            48.0,
            "Block 0's per-block FP8 byte must be 0x64 = 48 for the "
            "midpoint at element 15 to hit",
        )

        # Element 15 of block 0 must quantize to E2M1 grid value 4.0 (RNE on
        # midpoint 3.5), NOT 3.0 which a single-factor `data_hp * scale` gives.
        quantized_val = qt.quantized_val.reshape(-1, 16)
        self.assertEqual(
            float(quantized_val[0, 15].item()),
            4.0,
            "Midpoint 3.5 (= 168/48) must round to 4.0 via RNE on the "
            "E2M1 mantissa LSB. Got 3.0 means a single-factor path is in "
            "use: `data_hp * scale` drifts ~1 ULP below 3.5 due to the fp32 "
            "cast on (per_block_scale_fp8 / global_scale).",
        )


@skip_if_no_cuda
class TestGroupedNVFP4Errors(unittest.TestCase):
    """Validation / error paths of fake_quantize_grouped_nvfp4."""

    @staticmethod
    def _valid_config() -> QuantConfig:
        return QuantConfig.from_dict(FAKE_PRESETS["nvfp4"])

    def test_unsupported_config_raises(self) -> None:
        config = QuantConfig.from_dict(FAKE_PRESETS["mxfp4"])
        x = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
        group_counts = torch.tensor([2], device="cuda", dtype=torch.int32)
        with self.assertRaisesRegex(
            NotImplementedError, "unsupported grouped NVFP4 config"
        ):
            fake_quantize_grouped_nvfp4(x, "x", config, group_counts)

    def test_non_2d_tensor_raises(self) -> None:
        x = torch.randn(2, 3, 64, device="cuda", dtype=torch.bfloat16)
        group_counts = torch.tensor([2], device="cuda", dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "expects a 2D tensor"):
            fake_quantize_grouped_nvfp4(x, "x", self._valid_config(), group_counts)

    def test_non_cuda_tensor_raises(self) -> None:
        x = torch.randn(2, 64, dtype=torch.bfloat16)  # CPU tensor
        group_counts = torch.tensor([2], dtype=torch.int32)
        with self.assertRaisesRegex(NotImplementedError, "requires a CUDA tensor"):
            fake_quantize_grouped_nvfp4(x, "x", self._valid_config(), group_counts)

    def test_out_shape_mismatch_raises(self) -> None:
        x = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
        group_counts = torch.tensor([2], device="cuda", dtype=torch.int32)
        out = torch.empty(2, 32, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "out shape"):
            fake_quantize_grouped_nvfp4(
                x, "x", self._valid_config(), group_counts, out=out
            )

    def test_empty_tensor_out_none_returns_clone(self) -> None:
        x = torch.empty(0, 64, device="cuda", dtype=torch.bfloat16)
        group_counts = torch.tensor([0], device="cuda", dtype=torch.int32)
        result = fake_quantize_grouped_nvfp4(x, "x", self._valid_config(), group_counts)
        self.assertEqual(result.shape, x.shape)
        self.assertIsNot(result, x)

    def test_empty_tensor_with_out_returns_out(self) -> None:
        x = torch.empty(0, 64, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(0, 64, device="cuda", dtype=torch.bfloat16)
        group_counts = torch.tensor([0], device="cuda", dtype=torch.int32)
        result = fake_quantize_grouped_nvfp4(
            x, "x", self._valid_config(), group_counts, out=out
        )
        self.assertIs(result, out)

    def test_group_counts_not_1d_raises(self) -> None:
        x = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
        group_counts = torch.tensor([[2]], device="cuda", dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "group_counts must be 1D"):
            fake_quantize_grouped_nvfp4(x, "x", self._valid_config(), group_counts)

    def test_group_counts_empty_raises(self) -> None:
        x = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
        group_counts = torch.tensor([], device="cuda", dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "at least one group"):
            fake_quantize_grouped_nvfp4(x, "x", self._valid_config(), group_counts)

    def test_global_scale_wrong_shape_raises(self) -> None:
        x = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
        group_counts = torch.tensor([1, 1], device="cuda", dtype=torch.int32)
        global_scale = torch.ones(3, device="cuda", dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "global_scale must be scalar"):
            fake_quantize_grouped_nvfp4(
                x,
                "x",
                self._valid_config(),
                group_counts,
                global_scale=global_scale,
            )


def _without_global_scale(config: QuantConfig) -> QuantConfig:
    return replace(
        config,
        global_scale_granularity=GlobalScaleGranularity.NONE,
        global_scale_dtype=ScaleDtype.NONE,
    )


def _e2m1_reference(scaled: torch.Tensor) -> torch.Tensor:
    """Round to the E2M1 grid, ties to the even code, saturating at +-6."""
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)
    distance = (scaled.double().abs().clamp(max=6.0).unsqueeze(-1) - grid).abs()
    # Exact midpoints tie; the odd-code penalty is far below an fp32 ULP here.
    distance = distance + (torch.arange(8) % 2) * 1e-12
    return (grid[distance.argmin(-1)] * scaled.double().sign()).float()


def _heavy_tailed_weights(dtype: torch.dtype) -> torch.Tensor:
    """Student-t and Laplace rows spanning four decades, an outlier and a zero row."""
    generator = torch.Generator().manual_seed(0)
    rows, columns = 48, 200
    normal = torch.randn(rows, columns, generator=generator)
    chi = torch.randn(rows, columns, 3, generator=generator).square().mean(-1)
    student_t = normal / chi.sqrt()
    laplace = torch.empty(rows, columns).exponential_(generator=generator) * (
        torch.randint(0, 2, (rows, columns), generator=generator) * 2 - 1
    )
    weights = torch.where(torch.arange(rows)[:, None] % 2 == 0, student_t, laplace)
    weights *= torch.logspace(-2, 2, rows)[:, None]
    weights[5, 17] = 4 * weights.abs().max()
    weights[7] = 0
    return weights.to(dtype)


class TestNVFP4WithoutGlobalScale(unittest.TestCase):
    def test_matches_independent_block_reference(self) -> None:
        config = _without_global_scale(QuantConfig.from_dict(FAKE_PRESETS["nvfp4"]))
        source = torch.randn(4, 64, generator=torch.Generator().manual_seed(1)) * 300
        source[0, :16] = 0
        # Its E4M3 block scale rounds to zero, so the block dequantizes to zero.
        source[1, 16:32] = 1e-5
        info = qdq(source, "w", config, return_quant_info=True)
        assert isinstance(info, QuantInfo)

        blocks = source.reshape(-1, 16)
        block_scale = (
            (blocks.abs().amax(dim=1, keepdim=True) / 6.0)
            .to(torch.float8_e4m3fn)
            .float()
        )
        # Single-level scales quantize with the correctly rounded reciprocal.
        reciprocal = (1.0 / block_scale.double()).float()
        expected = torch.where(
            block_scale == 0,
            0.0,
            _e2m1_reference(blocks * reciprocal) * block_scale,
        ).reshape(4, 64)

        self.assertIsNone(info.global_scale)
        torch.testing.assert_close(info.dequant_data, expected, rtol=0, atol=0)

    def test_block_scales_saturate_finite(self) -> None:
        config = _without_global_scale(QuantConfig.from_dict(FAKE_PRESETS["nvfp4"]))
        # Block amax / 6 in (448, 480), at least 480, and far beyond E4M3.
        amax = torch.tensor([6 * 460.0, 6 * 1000.0, 6e30])
        source = amax[:, None] * torch.linspace(-1, 1, 16)
        for use_triton in (False, True) if torch.cuda.is_available() else (False,):
            with self.subTest(use_triton=use_triton):
                info = fake_quantize(
                    source.cuda() if use_triton else source,
                    "w",
                    config,
                    use_triton=use_triton,
                    return_quant_info=True,
                )
                assert isinstance(info, QuantInfo)
                self.assertTrue(bool(torch.all(info.local_unscale == 448.0)))
                self.assertTrue(bool(torch.isfinite(info.dequant_data).all()))
                torch.testing.assert_close(
                    info.dequant_data.abs().amax(dim=1).cpu(),
                    torch.full((3,), 6 * 448.0),
                    rtol=0,
                    atol=0,
                )

    @skip_if_no_cuda
    def test_triton_matches_eager(self) -> None:
        source = _heavy_tailed_weights(torch.float32) * 50
        for preset in ("nvfp4", "nvfp4_16x16"):
            with self.subTest(preset=preset):
                config = _without_global_scale(
                    QuantConfig.from_dict(FAKE_PRESETS[preset])
                )
                eager = fake_quantize(source, "w", config, use_triton=False)
                triton = fake_quantize(source.cuda(), "w", config, use_triton=True)
                torch.testing.assert_close(triton.cpu(), eager, rtol=0, atol=0)


class TestNVFP4GlobalScaleSaturation(unittest.TestCase):
    @parameterized.expand(
        [
            ("none", ScaleRoundMode.NONE),
            ("floor", ScaleRoundMode.FLOOR),
            ("ceil", ScaleRoundMode.CEIL),
        ]
    )
    def test_block_scale_saturates_finite(
        self, _name: str, scale_round_mode: ScaleRoundMode
    ) -> None:
        # Pre-cast block scales are 448 * local / global: 448, ~461, 1792, ~4e22.
        local_amax = torch.tensor([1.0, 1.03, 4.0, 1e20])
        for granularity, global_amax in (
            (GlobalScaleGranularity.TENSORWISE, torch.tensor([1.0])),
            (GlobalScaleGranularity.ROWWISE, torch.tensor([1.0, 1.0])),
        ):
            with self.subTest(granularity=granularity):
                _, unscale, local_unscale_fp8, _ = calculate_fp8_scale_for_nvfp4(
                    global_amax=global_amax,
                    local_amax=local_amax,
                    max_pos=6.0,
                    global_scale_granularity=granularity,
                    num_blocks_row=2,
                    num_blocks_col=2,
                    scale_round_mode=scale_round_mode,
                )
                self.assertTrue(bool(torch.all(local_unscale_fp8.float() == 448.0)))
                torch.testing.assert_close(
                    unscale, torch.full((4,), 448.0 / 2688.0), rtol=0, atol=0
                )


class TestNVFP4BlockScaleWithMBS(unittest.TestCase):
    """FP8 block scales under an MBS macro scale, with no NVFP4 global scale."""

    def _config(self) -> QuantConfig:
        return QuantConfig.from_dict(
            {
                **FAKE_PRESETS["nvfp4"],
                "global_scale_granularity": "mbs128",
                "global_scale_dtype": "e0m8",
            }
        )

    def _unit_factor_weights(self) -> torch.Tensor:
        # Each 128-wide macro block peaks at 6 * 2^k, so its E0M8 factor is 1.
        weights = torch.randn(8, 256, generator=torch.Generator().manual_seed(6))
        macro = weights.reshape(8, 2, 128)
        exponents = torch.arange(16, dtype=torch.float32).reshape(8, 2) - 8
        peaks = 6.0 * torch.exp2(exponents)
        macro = macro / macro.abs().amax(dim=2, keepdim=True) * peaks[..., None]
        return macro.reshape(8, 256)

    def test_unit_macro_factor_matches_no_global_scale(self) -> None:
        weights = self._unit_factor_weights()
        info = qdq(weights, "w", self._config(), return_quant_info=True)
        assert isinstance(info, QuantInfo)
        baseline = qdq(
            weights,
            "w",
            _without_global_scale(QuantConfig.from_dict(FAKE_PRESETS["nvfp4"])),
        )

        self.assertIsNone(info.global_scale)
        torch.testing.assert_close(info.dequant_data, baseline, rtol=0, atol=0)

    @skip_if_no_cuda
    def test_triton_matches_eager(self) -> None:
        config = self._config()
        for weights in (
            self._unit_factor_weights(),
            _heavy_tailed_weights(torch.float32),
        ):
            eager = fake_quantize(weights, "w", config, use_triton=False)
            triton = fake_quantize(weights.cuda(), "w", config, use_triton=True)
            torch.testing.assert_close(triton.cpu(), eager, rtol=0, atol=0)


class TestNVFP4NaNBlockScale(unittest.TestCase):
    @parameterized.expand(
        [
            (f"{granularity}_{mode}", granularity, mode)
            for granularity in ("rowwise", "none")
            for mode in ("none", "floor", "ceil")
        ]
    )
    def test_nan_block_scale_survives_the_e4m3_clamp(
        self, _name: str, granularity: str, mode: str
    ) -> None:
        preset = "rowwise_nvfp4" if granularity == "rowwise" else "nvfp4"
        config = QuantConfig.from_dict(
            {**FAKE_PRESETS[preset], "scale_round_mode": mode}
        )
        if granularity == "none":
            config = _without_global_scale(config)
        # One 16-element block per row; row 2 is a block of NaN only.
        source = torch.randn(4, 16, generator=torch.Generator().manual_seed(8))
        source[2] = float("nan")

        eager = qdq(source, "w", config, return_quant_info=True)
        assert isinstance(eager, QuantInfo)
        eager_scales = eager.local_unscale.reshape(-1)
        self.assertTrue(
            torch.equal(
                torch.isnan(eager_scales), torch.tensor([False, False, True, False])
            )
        )
        if not torch.cuda.is_available():
            return
        triton = fake_quantize(
            source.cuda(), "w", config, use_triton=True, return_quant_info=True
        )
        assert isinstance(triton, QuantInfo)
        torch.testing.assert_close(
            triton.local_unscale.reshape(-1).cpu(),
            eager_scales,
            rtol=0,
            atol=0,
            equal_nan=True,
        )
        torch.testing.assert_close(
            triton.dequant_data.cpu(),
            eager.dequant_data,
            rtol=0,
            atol=0,
            equal_nan=True,
        )


if __name__ == "__main__":
    unittest.main()
