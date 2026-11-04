# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

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
from quant_suite_core.fake_quant import is_triton_mega_supported
from quant_suite_core.metrics import cal_sqnr
from quant_suite_core.qdq_pytorch import qdq
from quant_suite_core.qdq_triton import qdq_triton_mega
from quant_suite_core.quant_config import MBSBlockSize
from quant_suite_core.triton_utils import _exceeds_triton_block_numel
from quant_test_utils import make_quant_config, skip_if_no_cuda


# ******************************************************************************
# * qdq() Pure PyTorch Path Tests (GPU required)                              *
# ******************************************************************************
@skip_if_no_cuda
class TestQDQPurePyTorch(unittest.TestCase):
    """Tests for qdq() pure PyTorch quantize-dequantize path."""

    def setUp(self) -> None:
        self.device = torch.device("cuda")

    def test_qdq_e2m1_e8m0_basic(self) -> None:
        """Standard MX4: E2M1 + E8M0/OAS."""
        config = make_quant_config()
        x = torch.randn(32, 64, device=self.device)
        result = qdq(x, "test_tensor", config)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, x.dtype)
        self.assertFalse(torch.isnan(result).any().item())
        self.assertFalse(torch.isinf(result).any().item())

    def test_qdq_e4m3_e8m0_oas_s(self) -> None:
        """E4M3 + E8M0/OAS_S (recommended for FP8 E4M3)."""
        config = make_quant_config(
            elem_dtype=ElemDtype.E4M3,
            scale_round_mode=ScaleRoundMode.OAS_S,
        )
        x = torch.randn(32, 64, device=self.device)
        result = qdq(x, "test_tensor", config)
        self.assertEqual(result.shape, x.shape)
        self.assertFalse(torch.isnan(result).any().item())

    def test_qdq_fp32_scale_dtype(self) -> None:
        """FP32 scale with NONE round mode (covers eps handling for zero delta)."""
        config = make_quant_config(
            scale_dtype=ScaleDtype.FP32,
            scale_round_mode=ScaleRoundMode.NONE,
        )
        x = torch.randn(32, 64, device=self.device)
        result = qdq(x, "test_tensor", config)
        self.assertEqual(result.shape, x.shape)
        self.assertFalse(torch.isnan(result).any().item())

    def test_qdq_fp16_scale_dtype(self) -> None:
        """FP16 scale path (covers FP16 eps handling)."""
        config = make_quant_config(
            scale_dtype=ScaleDtype.FP16,
            scale_round_mode=ScaleRoundMode.NONE,
        )
        x = torch.randn(32, 64, device=self.device)
        result = qdq(x, "test_tensor", config)
        self.assertEqual(result.shape, x.shape)
        self.assertFalse(torch.isnan(result).any().item())

    def test_qdq_e8m1_elem_dtype(self) -> None:
        """E8M1 element dtype dispatches to fp32_to_e8mx."""
        config = make_quant_config(elem_dtype=ElemDtype.E8M1)
        x = torch.randn(32, 64, device=self.device)
        result = qdq(x, "test_tensor", config)
        self.assertEqual(result.shape, x.shape)
        self.assertFalse(torch.isnan(result).any().item())

    def test_qdq_e8m2_elem_dtype(self) -> None:
        """E8M2 element dtype dispatches to fp32_to_e8mx."""
        config = make_quant_config(elem_dtype=ElemDtype.E8M2)
        x = torch.randn(32, 64, device=self.device)
        result = qdq(x, "test_tensor", config)
        self.assertEqual(result.shape, x.shape)
        self.assertFalse(torch.isnan(result).any().item())

    def test_qdq_e8m3_elem_dtype(self) -> None:
        """E8M3 element dtype dispatches to fp32_to_e8mx."""
        config = make_quant_config(elem_dtype=ElemDtype.E8M3)
        x = torch.randn(32, 64, device=self.device)
        result = qdq(x, "test_tensor", config)
        self.assertEqual(result.shape, x.shape)
        self.assertFalse(torch.isnan(result).any().item())

    def test_qdq_fp8_scale_tensorwise(self) -> None:
        """FP8 scale with TENSORWISE global scale granularity."""
        config = make_quant_config(
            scale_dtype=ScaleDtype.FP8,
            scale_round_mode=ScaleRoundMode.NONE,
            global_scale_granularity=GlobalScaleGranularity.TENSORWISE,
            global_scale_dtype=ScaleDtype.FP32,
        )
        x = torch.randn(32, 64, device=self.device)
        result = qdq(x, "test_tensor", config)
        self.assertEqual(result.shape, x.shape)
        self.assertFalse(torch.isnan(result).any().item())

    def test_qdq_with_mbs_granularity(self) -> None:
        """MBS32 macro block scale granularity."""
        config = make_quant_config(
            global_scale_granularity=GlobalScaleGranularity.MBS,
            mbs_block_size=MBSBlockSize.MBS32,
            global_scale_dtype=ScaleDtype.FP32,
        )
        x = torch.randn(32, 64, device=self.device)
        result = qdq(x, "test_tensor", config)
        self.assertEqual(result.shape, x.shape)
        self.assertFalse(torch.isnan(result).any().item())

    def test_qdq_return_quant_info(self) -> None:
        """return_quant_info=True returns QuantInfo with all fields."""
        config = make_quant_config()
        x = torch.randn(32, 64, device=self.device)
        result = qdq(x, "test_tensor", config, return_quant_info=True)
        self.assertIsInstance(result, QuantInfo)
        self.assertEqual(result.dequant_data.shape, x.shape)
        self.assertEqual(result.quantized_val.shape, x.shape)
        # E8M0 scale has no global_scale
        self.assertIsNone(result.global_scale)

    def test_qdq_return_quant_info_fp8_scale(self) -> None:
        """return_quant_info with FP8 scale includes global_scale."""
        config = make_quant_config(
            scale_dtype=ScaleDtype.FP8,
            scale_round_mode=ScaleRoundMode.NONE,
            global_scale_granularity=GlobalScaleGranularity.TENSORWISE,
            global_scale_dtype=ScaleDtype.FP32,
        )
        x = torch.randn(32, 64, device=self.device)
        result = qdq(x, "test_tensor", config, return_quant_info=True)
        self.assertIsInstance(result, QuantInfo)
        self.assertEqual(result.dequant_data.shape, x.shape)
        self.assertIsNotNone(result.global_scale)

    def test_qdq_transpose(self) -> None:
        """transpose=True swaps block dimensions."""
        config = make_quant_config(block_size=BlockSize(1, 32))
        x = torch.randn(32, 64, device=self.device)
        result = qdq(x, "test_tensor", config, transpose=True)
        self.assertEqual(result.shape, x.shape)
        self.assertFalse(torch.isnan(result).any().item())

    def test_qdq_non_divisible_shape(self) -> None:
        """Tensor shape not divisible by block size triggers padding."""
        config = make_quant_config(block_size=BlockSize(1, 32))
        # 50 is not divisible by 32
        x = torch.randn(17, 50, device=self.device)
        result = qdq(x, "test_tensor", config)
        self.assertEqual(result.shape, x.shape)
        self.assertFalse(torch.isnan(result).any().item())

    def test_qdq_3d_input(self) -> None:
        """3D input is reshaped to 2D internally, then back."""
        config = make_quant_config()
        x = torch.randn(2, 32, 64, device=self.device)
        result = qdq(x, "test_tensor", config)
        self.assertEqual(result.shape, x.shape)
        self.assertFalse(torch.isnan(result).any().item())

    def test_qdq_bfloat16_input(self) -> None:
        """bfloat16 input dtype is preserved in output."""
        config = make_quant_config()
        x = torch.randn(32, 64, device=self.device, dtype=torch.bfloat16)
        result = qdq(x, "test_tensor", config)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertFalse(torch.isnan(result).any().item())

    def test_qdq_float16_input(self) -> None:
        """float16 input dtype is preserved in output."""
        config = make_quant_config()
        x = torch.randn(32, 64, device=self.device, dtype=torch.float16)
        result = qdq(x, "test_tensor", config)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, torch.float16)
        self.assertFalse(torch.isnan(result).any().item())

    def test_qdq_quantization_reduces_unique_values(self) -> None:
        """Quantized output should have fewer unique values than input."""
        config = make_quant_config()
        x = torch.randn(32, 64, device=self.device)
        result = qdq(x, "test_tensor", config)
        unique_input = torch.unique(x).numel()
        unique_output = torch.unique(result).numel()
        self.assertLess(unique_output, unique_input)

    def test_qdq_pow2max_scale_norm(self) -> None:
        """POW2MAX scale normalization uses power-of-2 max."""
        config = make_quant_config(scale_norm_mode=ScaleNormMode.POW2MAX)
        x = torch.randn(32, 64, device=self.device)
        result = qdq(x, "test_tensor", config)
        self.assertEqual(result.shape, x.shape)
        self.assertFalse(torch.isnan(result).any().item())

    # -----------------------------------------------------------------
    # INT4 / INT8 element dtypes: direct qdq on the integer grid.
    # -----------------------------------------------------------------
    def _int_config(self, elem_dtype: ElemDtype) -> QuantConfig:
        return make_quant_config(
            quant_mode=QuantMode.SYMM,
            scale_dtype=ScaleDtype.E8M0,
            scale_norm_mode=ScaleNormMode.POW2MAX,
            scale_round_mode=ScaleRoundMode.OAS,
            elem_dtype=elem_dtype,
            block_size=BlockSize(1, 32),
        )

    def test_qdq_int8_symm_integer_grid(self) -> None:
        """INT8 SYMM: quantized values are integers within [-127, 127]."""
        config = self._int_config(ElemDtype.INT8)
        x = torch.randn(32, 64, device=self.device)
        result = qdq(x, "test_tensor", config, return_quant_info=True)
        self.assertIsInstance(result, QuantInfo)
        qv = result.quantized_val
        self.assertFalse(torch.isnan(result.dequant_data).any().item())
        self.assertTrue(torch.equal(qv, qv.round()))
        self.assertTrue((qv.abs() <= 127).all().item())

    def test_qdq_int4_symm_integer_grid(self) -> None:
        """INT4 SYMM: quantized values are integers within [-7, 7]."""
        config = self._int_config(ElemDtype.INT4)
        x = torch.randn(32, 64, device=self.device)
        result = qdq(x, "test_tensor", config, return_quant_info=True)
        self.assertIsInstance(result, QuantInfo)
        qv = result.quantized_val
        self.assertFalse(torch.isnan(result.dequant_data).any().item())
        self.assertTrue(torch.equal(qv, qv.round()))
        self.assertTrue((qv.abs() <= 7).all().item())

    # -----------------------------------------------------------------
    # E3M0 element dtype end-to-end: magnitudes are pure powers of 2.
    # -----------------------------------------------------------------
    def test_qdq_e3m0_pow2max(self) -> None:
        config = make_quant_config(
            elem_dtype=ElemDtype.E3M0,
            scale_dtype=ScaleDtype.E8M0,
            scale_norm_mode=ScaleNormMode.POW2MAX,
            scale_round_mode=ScaleRoundMode.OAS,
            block_size=BlockSize(1, 32),
        )
        x = torch.randn(32, 64, device=self.device)
        result = qdq(x, "test_tensor", config, return_quant_info=True)
        self.assertIsInstance(result, QuantInfo)
        self.assertEqual(result.dequant_data.shape, x.shape)
        self.assertFalse(torch.isnan(result.dequant_data).any().item())
        qv = result.quantized_val
        # E3M0 has 0 mantissa bits, so every non-zero code is ±2**k, clamped to 16.
        self.assertTrue((qv.abs() <= 16.0).all().item())
        nonzero = qv[qv != 0].abs()
        log2 = nonzero.log2()
        self.assertTrue(torch.allclose(log2, log2.round()))

    # -----------------------------------------------------------------
    # _reshape_2d_tensor non-2D branches via qdq: 1D / 0D / 4D.
    # -----------------------------------------------------------------
    def test_qdq_1d_input(self) -> None:
        config = make_quant_config(block_size=BlockSize(1, 32))
        x = torch.randn(64, device=self.device)
        result = qdq(x, "test_tensor", config)
        self.assertEqual(result.shape, x.shape)
        self.assertFalse(torch.isnan(result).any().item())

    def test_qdq_scalar_input(self) -> None:
        config = make_quant_config(block_size=BlockSize(1, 32))
        x = torch.tensor(3.0, device=self.device)
        result = qdq(x, "test_tensor", config)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.ndim, 0)
        self.assertFalse(torch.isnan(result).any().item())

    def test_qdq_4d_input(self) -> None:
        config = make_quant_config(block_size=BlockSize(1, 32))
        x = torch.randn(2, 2, 4, 32, device=self.device)
        result = qdq(x, "test_tensor", config)
        self.assertEqual(result.shape, x.shape)
        self.assertFalse(torch.isnan(result).any().item())


# ******************************************************************************
# * Triton Quantization Path Tests (GPU required)                             *
# ******************************************************************************
@skip_if_no_cuda
class TestTritonQuantization(unittest.TestCase):
    """Tests for qdq_triton_mega() and related functions."""

    def setUp(self) -> None:
        self.device = torch.device("cuda")

    def test_is_triton_mega_supported_small_block(self) -> None:
        """A block well under the m*n <= 8192 Triton tile cap is valid for the mega kernel."""
        config = make_quant_config(block_size=BlockSize(1, 32))
        self.assertTrue(is_triton_mega_supported(config))

    def test_is_triton_mega_supported_large_block(self) -> None:
        """Block size constrains mega-kernel validity only past the Triton tile cap."""
        config = make_quant_config(block_size=BlockSize(1, 128))
        self.assertTrue(is_triton_mega_supported(config))

    def test_exceeds_triton_block_numel(self) -> None:
        """Only m*n > 8192 overflows the BLOCK_M x m*n Triton tile."""
        for m, n, expected in ((128, 128, True), (16, 16, False), (1, 16, False)):
            config = make_quant_config(block_size=BlockSize(m, n))
            self.assertEqual(_exceeds_triton_block_numel(config), expected, f"{m}x{n}")
            self.assertEqual(is_triton_mega_supported(config), not expected)

    def test_fake_quant_rejects_oversized_block(self) -> None:
        """An unservable block size must name the ceiling, not Triton's raw numel."""
        config = make_quant_config(block_size=BlockSize(128, 128))
        x = torch.randn(256, 256, device=self.device)
        with self.assertRaisesRegex(NotImplementedError, r"m\*n <= 8192"):
            fake_quantize(x, "test_tensor", config, use_triton=True)

    def test_is_triton_mega_supported_fp8_rowwise(self) -> None:
        """ROWWISE FP8 granularity should be valid for mega kernel."""
        config = QuantConfig(
            quant_mode=QuantMode.SYMM,
            scale_dtype=ScaleDtype.FP8,
            scale_norm_mode=ScaleNormMode.AMAX,
            scale_round_mode=ScaleRoundMode.NONE,
            global_scale_granularity=GlobalScaleGranularity.ROWWISE,
            global_scale_dtype=ScaleDtype.FP32,
            elem_dtype=ElemDtype.E2M1,
            elem_round_mode=ElemRoundMode.NEAREST,
            block_size=BlockSize(1, 32),
        )
        self.assertTrue(is_triton_mega_supported(config))

    def test_qdq_triton_mega_e2m1(self) -> None:
        """qdq_triton_mega with E2M1 + E8M0/OAS."""
        config = make_quant_config()
        x = torch.randn(32, 64, device=self.device)
        result = qdq_triton_mega(x, "test_tensor", config)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, x.dtype)
        self.assertFalse(torch.isnan(result).any().item())

    def test_qdq_triton_mega_e4m3(self) -> None:
        """qdq_triton_mega with E4M3 + E8M0/OAS_S."""
        config = make_quant_config(
            elem_dtype=ElemDtype.E4M3,
            scale_round_mode=ScaleRoundMode.OAS_S,
        )
        x = torch.randn(32, 64, device=self.device)
        result = qdq_triton_mega(x, "test_tensor", config)
        self.assertEqual(result.shape, x.shape)
        self.assertFalse(torch.isnan(result).any().item())

    def test_qdq_triton_mega_large_block(self) -> None:
        """A 1x128 block still fits the mega kernel's m*n <= 8192 numel ceiling."""
        config = make_quant_config(block_size=BlockSize(1, 128))
        x = torch.randn(128, 256, device=self.device)
        result = qdq_triton_mega(x, "test_tensor", config)
        self.assertEqual(result.shape, x.shape)
        self.assertFalse(torch.isnan(result).any().item())

    def test_qdq_triton_mega_fp8_scale_tensorwise(self) -> None:
        """qdq_triton_mega with FP8 scale and TENSORWISE granularity."""
        config = make_quant_config(
            scale_dtype=ScaleDtype.FP8,
            scale_round_mode=ScaleRoundMode.NONE,
            global_scale_granularity=GlobalScaleGranularity.TENSORWISE,
            global_scale_dtype=ScaleDtype.FP32,
        )
        x = torch.randn(32, 64, device=self.device)
        result = qdq_triton_mega(x, "test_tensor", config)
        self.assertEqual(result.shape, x.shape)
        self.assertFalse(torch.isnan(result).any().item())

    def test_qdq_triton_mega_fp8_scale_rowwise_matches_pure(self) -> None:
        """qdq_triton_mega with FP8 scale and ROWWISE granularity."""
        config = make_quant_config(
            scale_dtype=ScaleDtype.FP8,
            scale_round_mode=ScaleRoundMode.NONE,
            global_scale_granularity=GlobalScaleGranularity.ROWWISE,
            global_scale_dtype=ScaleDtype.FP32,
        )
        x = torch.randn(128, 128, device=self.device)
        result_pytorch = qdq(x, "test_tensor", config)
        result_triton = qdq_triton_mega(x, "test_tensor", config)
        pytorch_bits = result_pytorch.view(torch.int32)
        triton_bits = result_triton.view(torch.int32)
        self.assertTrue(
            torch.equal(pytorch_bits, triton_bits),
            (
                "qdq_triton_mega ROWWISE FP8 output must match qdq() bitwise; "
                f"mismatches={(pytorch_bits != triton_bits).sum().item()}, "
                f"max_abs_diff={(result_pytorch - result_triton).abs().max().item()}"
            ),
        )

    def test_qdq_triton_mega_fp8_scale_rowwise_improves_sqnr(self) -> None:
        """ROWWISE NVFP4 should improve SQNR over TENSORWISE for uneven row scales."""
        tensorwise_config = make_quant_config(
            scale_dtype=ScaleDtype.FP8,
            scale_round_mode=ScaleRoundMode.NONE,
            global_scale_granularity=GlobalScaleGranularity.TENSORWISE,
            global_scale_dtype=ScaleDtype.FP32,
        )
        rowwise_config = make_quant_config(
            scale_dtype=ScaleDtype.FP8,
            scale_round_mode=ScaleRoundMode.NONE,
            global_scale_granularity=GlobalScaleGranularity.ROWWISE,
            global_scale_dtype=ScaleDtype.FP32,
        )
        base = torch.linspace(-1.0, 1.0, steps=64, device=self.device).reshape(1, 64)
        row_scales = torch.tensor(
            [1.0, 2.0**-8, 2.0**-12, 2.0**-16],
            device=self.device,
        ).unsqueeze(1)
        x = (base * row_scales).repeat_interleave(8, dim=0)

        tensorwise_result = qdq_triton_mega(x, "test_tensor", tensorwise_config)
        rowwise_result = qdq_triton_mega(x, "test_tensor", rowwise_config)
        tensorwise_sqnr = cal_sqnr(tensorwise_result, x)
        rowwise_sqnr = cal_sqnr(rowwise_result, x)

        self.assertGreater(rowwise_sqnr, tensorwise_sqnr)

    def test_qdq_pytorch_vs_triton_consistency(self) -> None:
        """qdq() and qdq_triton_mega() should produce similar results."""
        config = make_quant_config()
        x = torch.randn(32, 64, device=self.device)
        result_pytorch = qdq(x, "test_tensor", config)
        result_triton = qdq_triton_mega(x, "test_tensor", config)
        torch.testing.assert_close(result_pytorch, result_triton, atol=1e-5, rtol=1e-5)

    def test_qdq_asymm_mean_shift_differs_from_midpoint_asymm(self) -> None:
        """Mean-shift and midpoint asymm should differ for skewed blocks."""
        common_kwargs = {
            "elem_dtype": ElemDtype.E4M3,
            "scale_dtype": ScaleDtype.FP16,
            "scale_round_mode": ScaleRoundMode.NONE,
            "scale_norm_mode": ScaleNormMode.AMAX,
            "block_size": BlockSize(1, 64),
        }
        mean_shift_config = make_quant_config(
            quant_mode=QuantMode.ASYMM_MEAN_SHIFT,
            **common_kwargs,
        )
        midpoint_config = make_quant_config(
            quant_mode=QuantMode.ASYMM,
            **common_kwargs,
        )
        x = torch.cat(
            [
                torch.zeros(1, 63, device=self.device),
                torch.full((1, 1), 100.0, device=self.device),
            ],
            dim=1,
        )

        mean_shift = qdq(x, "test_tensor", mean_shift_config)
        midpoint = qdq(x, "test_tensor", midpoint_config)
        max_diff = torch.max(torch.abs(mean_shift - midpoint)).item()
        self.assertGreater(max_diff, 1e-3)

    def test_fake_quant_dispatches_to_qdq(self) -> None:
        """fake_quantize with use_triton=False dispatches to qdq."""
        config = make_quant_config()
        x = torch.randn(32, 64, device=self.device)
        result = fake_quantize(x, "test_tensor", config, use_triton=False)
        expected = qdq(x, "test_tensor", config)
        torch.testing.assert_close(result, expected)

    def test_fake_quant_dispatches_to_triton(self) -> None:
        """fake_quantize with use_triton=True dispatches to the Triton path."""
        config = make_quant_config()
        x = torch.randn(32, 64, device=self.device)
        result = fake_quantize(x, "test_tensor", config, use_triton=True)
        self.assertEqual(result.shape, x.shape)
        self.assertFalse(torch.isnan(result).any().item())

    def test_qdq_triton_mega_non_divisible_shape(self) -> None:
        """Triton path handles non-divisible tensor shapes via padding."""
        config = make_quant_config(block_size=BlockSize(1, 32))
        x = torch.randn(17, 50, device=self.device)
        result = qdq_triton_mega(x, "test_tensor", config)
        self.assertEqual(result.shape, x.shape)
        self.assertFalse(torch.isnan(result).any().item())

    def test_qdq_triton_mega_bfloat16(self) -> None:
        """bfloat16 dtype is preserved through triton mega kernel."""
        config = make_quant_config()
        x = torch.randn(32, 64, device=self.device, dtype=torch.bfloat16)
        result = qdq_triton_mega(x, "test_tensor", config)
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertFalse(torch.isnan(result).any().item())

    def test_qdq_triton_mega_float16(self) -> None:
        """float16 dtype is preserved through triton mega kernel."""
        config = make_quant_config()
        x = torch.randn(32, 64, device=self.device, dtype=torch.float16)
        result = qdq_triton_mega(x, "test_tensor", config)
        self.assertEqual(result.dtype, torch.float16)
        self.assertFalse(torch.isnan(result).any().item())

    def test_qdq_triton_mega_with_mbs_fp32(self) -> None:
        """Mega kernel with MBS32 pre-scaling (FP32 global scale dtype).

        Exercises the MBS path of `_mega_fake_quant_triton_kernel`: the host
        only pads each row to whole macro blocks; the kernel computes the
        per-macro-block amax and the FP32 factor, pre-scales, and unscales.
        """
        config = make_quant_config(
            global_scale_granularity=GlobalScaleGranularity.MBS,
            mbs_block_size=MBSBlockSize.MBS32,
            global_scale_dtype=ScaleDtype.FP32,
            block_size=BlockSize(1, 32),
        )
        x = torch.randn(32, 64, device=self.device)
        result = qdq_triton_mega(x, "test_tensor", config)
        self.assertEqual(result.shape, x.shape)
        self.assertFalse(torch.isnan(result).any().item())

    def test_qdq_triton_mega_with_mbs_e0m8(self) -> None:
        """Mega kernel with MBS32 pre-scaling (E0M8 global scale dtype).

        Covers the static-MBS E0M8 branch of `_mega_fake_quant_triton_kernel`:
        extract mantissa bits via int32 view, mask with 0x007FFFFF, set exponent
        to 2^0.
        """
        config = make_quant_config(
            global_scale_granularity=GlobalScaleGranularity.MBS,
            mbs_block_size=MBSBlockSize.MBS32,
            global_scale_dtype=ScaleDtype.E0M8,
            block_size=BlockSize(1, 32),
        )
        x = torch.randn(32, 64, device=self.device)
        result = qdq_triton_mega(x, "test_tensor", config)
        self.assertEqual(result.shape, x.shape)
        self.assertFalse(torch.isnan(result).any().item())

    def test_qdq_triton_mega_mbs_matches_eager(self) -> None:
        """Mega-kernel MBS32 vs eager `qdq` MBS32 cross-validation.

        The mega kernel derives the macro factor, pre-scales and unscales inside
        the kernel; `qdq` does the same steps eagerly. Agreement is what shows the two
        halves of the mega path are consistent -- neither smoke test alone does.
        """
        config_mbs = make_quant_config(
            global_scale_granularity=GlobalScaleGranularity.MBS,
            mbs_block_size=MBSBlockSize.MBS32,
            global_scale_dtype=ScaleDtype.FP32,
            block_size=BlockSize(1, 32),
        )
        x = torch.randn(32, 64, device=self.device)
        result_mega = qdq_triton_mega(x, "test_tensor", config_mbs)
        result_eager = qdq(x, "test_tensor", config_mbs)
        self.assertEqual(result_mega.shape, x.shape)
        self.assertFalse(torch.isnan(result_mega).any().item())
        torch.testing.assert_close(result_mega, result_eager, atol=1e-5, rtol=1e-5)

    # -----------------------------------------------------------------
    # fake_quantize dtype check: only bf16/fp16/fp32 allowed.
    # -----------------------------------------------------------------
    def test_fake_quant_rejects_int_dtype(self) -> None:
        config = make_quant_config()
        x = torch.randint(0, 10, (32, 64), device=self.device, dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "is not supported"):
            fake_quantize(x, "test_tensor", config, use_triton=False)

    def test_fake_quant_rejects_float64_dtype(self) -> None:
        config = make_quant_config()
        x = torch.randn(32, 64, device=self.device, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "is not supported"):
            fake_quantize(x, "test_tensor", config, use_triton=False)


# ******************************************************************************
# * ASYMM Quantization Path Tests (GPU required)                              *
# ******************************************************************************
@skip_if_no_cuda
class TestAsymmQuantization(unittest.TestCase):
    """qdq()/qdq_triton_mega() with QuantMode.ASYMM (zero_point path).

    Every other test uses SYMM; these exercise the asymmetric min/max range,
    the ``min_pos < 0`` vs ``min_pos == 0`` zero-point branch, and the INT
    ``delta`` (no ``/2``) path.
    """

    def setUp(self) -> None:
        self.device = torch.device("cuda")
        torch.manual_seed(0)

    def _asymm_config(self, elem_dtype: ElemDtype) -> QuantConfig:
        return make_quant_config(
            quant_mode=QuantMode.ASYMM,
            scale_dtype=ScaleDtype.E8M0,
            scale_norm_mode=ScaleNormMode.POW2MAX,
            scale_round_mode=ScaleRoundMode.OAS,
            elem_dtype=elem_dtype,
            block_size=BlockSize(1, 32),
        )

    def _check_output(self, result: torch.Tensor, x: torch.Tensor) -> None:
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, x.dtype)
        self.assertFalse(torch.isnan(result).any().item())
        self.assertFalse(torch.isinf(result).any().item())
        # Round-trip should stay in a sane range (loose sanity, not a bit pin).
        err = (result.float() - x.float()).abs().mean().item()
        self.assertLess(err, x.float().abs().mean().item() + 1.0)

    @parameterized.expand(
        [
            ("int8", ElemDtype.INT8),
            ("int4", ElemDtype.INT4),
            ("e2m1", ElemDtype.E2M1),
        ]
    )
    def test_qdq_asymm_eager(self, _name: str, elem_dtype: ElemDtype) -> None:
        config = self._asymm_config(elem_dtype)
        x = torch.randn(32, 64, device=self.device)
        result = qdq(x, "test_tensor", config)
        self._check_output(result, x)

    @parameterized.expand(
        [
            ("int8", ElemDtype.INT8),
            ("int4", ElemDtype.INT4),
            ("e2m1", ElemDtype.E2M1),
        ]
    )
    def test_qdq_triton_mega_asymm(self, _name: str, elem_dtype: ElemDtype) -> None:
        config = self._asymm_config(elem_dtype)
        x = torch.randn(32, 64, device=self.device)
        result = qdq_triton_mega(x, "test_tensor", config)
        self._check_output(result, x)

    def test_qdq_asymm_int8_accurate(self) -> None:
        """INT8 ASYMM has enough levels to reconstruct randn closely."""
        config = self._asymm_config(ElemDtype.INT8)
        x = torch.randn(32, 64, device=self.device)
        result = qdq(x, "test_tensor", config)
        err = (result.float() - x.float()).abs().mean().item()
        self.assertLess(err, 0.1)


# ******************************************************************************
# * High-Level API Tests (GPU required)                                       *
# ******************************************************************************
@skip_if_no_cuda
class TestHighLevelAPI(unittest.TestCase):
    """Tests for the fake_quantize() entry point."""

    def setUp(self) -> None:
        self.device = torch.device("cuda")
        self.config = QuantConfig(
            quant_mode=QuantMode.SYMM,
            scale_dtype=ScaleDtype.E8M0,
            scale_norm_mode=ScaleNormMode.AMAX,
            scale_round_mode=ScaleRoundMode.OAS,
            global_scale_granularity=GlobalScaleGranularity.NONE,
            elem_dtype=ElemDtype.E2M1,
            elem_round_mode=ElemRoundMode.NEAREST,
            block_size=BlockSize(1, 32),
        )

    def test_fake_quantize_eager_backend(self) -> None:
        """fake_quantize with use_triton=False dispatches through qdq."""
        x = torch.randn(32, 64, device=self.device)
        result = fake_quantize(
            x,
            tensor_name="test",
            config=self.config,
            use_triton=False,
        )
        self.assertEqual(result.shape, x.shape)
        self.assertFalse(torch.isnan(result).any().item())
        self.assertFalse(torch.equal(result, x))

    def test_fake_quant_float16_all_paths(self) -> None:
        """float16 dtype is preserved through fake_quantize for both triton and non-triton paths."""
        x = torch.randn(32, 64, device=self.device, dtype=torch.float16)
        # Non-triton path
        result_no_triton = fake_quantize(
            x,
            tensor_name="test",
            config=self.config,
            use_triton=False,
        )
        self.assertEqual(result_no_triton.shape, x.shape)
        self.assertEqual(result_no_triton.dtype, torch.float16)
        self.assertFalse(torch.isnan(result_no_triton).any().item())
        # Triton path
        result_triton = fake_quantize(
            x,
            tensor_name="test",
            config=self.config,
            use_triton=True,
        )
        self.assertEqual(result_triton.shape, x.shape)
        self.assertEqual(result_triton.dtype, torch.float16)
        self.assertFalse(torch.isnan(result_triton).any().item())

    def test_fake_quant_return_quant_info(self) -> None:
        """fake_quantize with return_quant_info=True returns QuantInfo."""
        x = torch.randn(32, 64, device=self.device)
        result = fake_quantize(
            x,
            tensor_name="test",
            config=self.config,
            use_triton=False,
            return_quant_info=True,
        )
        self.assertIsInstance(result, QuantInfo)
        self.assertEqual(result.dequant_data.shape, x.shape)


@skip_if_no_cuda
class TestMXINT3Quant(unittest.TestCase):
    """int3 fake quant via fake_quantize: lossy, SQNR ordered below int4/int8, 3-bit grid."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.device = "cuda"

    def _int_cfg(self, elem: ElemDtype) -> QuantConfig:
        # Reuse the mxint8 preset (symm, e8m0, pow2max, oas, 1x16) and swap only
        # the integer width, so int3/int4/int8 differ solely in elem_dtype.
        base = dict(FAKE_PRESETS["mxint8"])
        base["elem_dtype"] = elem.value
        return QuantConfig.from_dict(base)

    def test_int3_lossy_and_sqnr_orders_below_int4_int8(self) -> None:
        x = torch.randn(4096, 128, device=self.device, dtype=torch.bfloat16)
        sqnr = {}
        for elem in (ElemDtype.INT3, ElemDtype.INT4, ElemDtype.INT8):
            dq = fake_quantize(x, "t", self._int_cfg(elem), use_triton=False)
            self.assertEqual(dq.shape, x.shape)
            self.assertFalse(torch.equal(dq, x))  # lossy
            sqnr[elem] = cal_sqnr(dq, x)
        self.assertLess(sqnr[ElemDtype.INT3], sqnr[ElemDtype.INT4])
        self.assertLess(sqnr[ElemDtype.INT4], sqnr[ElemDtype.INT8])

    def test_int3_quantizes_to_at_most_7_levels_per_block(self) -> None:
        # A single 1x16 block has one scale, so the int3 dequant lands on at most
        # 7 symmetric levels (-3..3).
        x = torch.randn(1, 16, device=self.device, dtype=torch.bfloat16)
        dq3 = fake_quantize(x, "t", self._int_cfg(ElemDtype.INT3), use_triton=False)
        self.assertLessEqual(torch.unique(dq3).numel(), 7)


if __name__ == "__main__":
    unittest.main()
