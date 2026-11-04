# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any
from unittest import mock

import torch
from quant_suite_core import (
    ElemDtype,
    native_quant as native_quant_module,
    QuantConfig,
    QuantInfo,
)
from quant_suite_core.fake_presets import FAKE_PRESETS
from quant_suite_core.mx_packing import (
    e8m0_uint8_to_float,
    float_e2m1_to_packed_uint8,
    float_to_e8m0_uint8,
    packed_uint8_to_float_e2m1,
)
from quant_suite_core.native_quant import (
    _swizzle_scale,
    _unswizzle_scale,
    native_data_shape,
    native_dequantize,
    native_dtype_matches,
    native_quantize,
    NativeQuantResult,
)
from quant_suite_core.qdq_pytorch import qdq
from quant_suite_core.qdq_triton import qdq_triton_mega


def _config(preset_name: str, **overrides: Any) -> QuantConfig:
    preset = {**FAKE_PRESETS[preset_name], **overrides}
    return QuantConfig.from_dict(preset)


def _mxfp4_config(scale_round_mode: str, block_size: int = 32) -> QuantConfig:
    return _config(
        "mxfp4",
        block_size=f"1x{block_size}",
        scale_round_mode=scale_round_mode,
    )


def _mxfp8_config(scale_round_mode: str, block_size: int = 32) -> QuantConfig:
    return _config(
        "mxfp8",
        block_size=f"1x{block_size}",
        scale_round_mode=scale_round_mode,
    )


def _requires_sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] >= 10


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Skip when cuda is not available",
)
class TestMxPacking(unittest.TestCase):
    def test_float_e2m1_pack_unpack_roundtrip(self) -> None:
        values = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=torch.float32,
            device="cuda",
        )

        packed = float_e2m1_to_packed_uint8(values)
        self.assertEqual(packed.shape, (8,))
        self.assertEqual(packed.dtype, torch.uint8)

        unpacked = packed_uint8_to_float_e2m1(packed)
        self.assertTrue(
            torch.equal(values.view(torch.int32), unpacked.view(torch.int32))
        )

    def test_float_e2m1_pack_2d(self) -> None:
        values = torch.tensor(
            [[0.0, 0.5, 1.0, 1.5], [2.0, 3.0, 4.0, 6.0]],
            dtype=torch.float32,
            device="cuda",
        )

        packed = float_e2m1_to_packed_uint8(values)
        self.assertEqual(packed.shape, (2, 2))
        torch.testing.assert_close(
            packed_uint8_to_float_e2m1(packed),
            values,
            rtol=0,
            atol=0,
        )

    def test_e8m0_pack_unpack_roundtrip(self) -> None:
        values = torch.tensor(
            [0.25, 0.5, 1.0, 2.0, 4.0, 16.0, 256.0],
            dtype=torch.float32,
            device="cuda",
        )

        packed = float_to_e8m0_uint8(values)
        self.assertEqual(packed.dtype, torch.uint8)
        torch.testing.assert_close(
            e8m0_uint8_to_float(packed),
            values,
            rtol=0,
            atol=0,
        )

    def test_e8m0_uint8_to_float_is_exact_for_every_byte(self) -> None:
        # Bytes 0..254 are 2**-127 (the fp32 subnormal) .. 2**127; 255 is NaN.
        exponents = torch.arange(-127, 128)
        expected = torch.ldexp(
            torch.ones(exponents.numel(), dtype=torch.float64), exponents
        ).to(torch.float32)
        for device in ("cpu", "cuda"):
            with self.subTest(device=device):
                got = e8m0_uint8_to_float(
                    torch.arange(256, device=device).to(torch.uint8)
                ).cpu()
                self.assertTrue(
                    torch.equal(got[:255].view(torch.int32), expected.view(torch.int32))
                )
                self.assertTrue(bool(torch.isnan(got[255])))


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Skip when cuda is not available",
)
class TestMxNativeQuantize(unittest.TestCase):
    def _compare_quantize_vs_qdq(
        self,
        config: QuantConfig,
        x: torch.Tensor,
    ) -> None:
        result = native_quantize(x, "weight", config, use_triton=False)
        quant_info = qdq(x.float(), "weight", config, return_quant_info=True)
        self.assertIsInstance(quant_info, QuantInfo)

        if config.elem_dtype == ElemDtype.E2M1:
            result_data = packed_uint8_to_float_e2m1(result.data)
        else:
            result_data = result.data.to(torch.float32)

        expected_data = quant_info.quantized_val.reshape(result_data.shape)
        self.assertTrue(
            torch.equal(result_data.view(torch.int32), expected_data.view(torch.int32)),
            f"Quantized data mismatch for {config}",
        )

        result_scale = e8m0_uint8_to_float(result.scale)
        expected_scale = quant_info.local_unscale.reshape(result_scale.shape)
        torch.testing.assert_close(result_scale, expected_scale, rtol=0, atol=0)

    def test_quantize_matches_qdq_reference(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(16, 128, dtype=torch.bfloat16, device="cuda")
        configs = [
            _mxfp4_config("oas"),
            _mxfp4_config("ceil"),
            _mxfp4_config("floor"),
            _mxfp4_config("nv_round"),
            _mxfp4_config("oas", block_size=16),
            _mxfp8_config("oas_s"),
            _mxfp8_config("ceil"),
            _mxfp8_config("floor"),
        ]

        for config in configs:
            with self.subTest(config=config):
                self._compare_quantize_vs_qdq(config, x)

    def test_quantize_matches_qdq_reference_for_3d_input(self) -> None:
        torch.manual_seed(1)
        x = torch.randn(2, 8, 128, dtype=torch.bfloat16, device="cuda")

        self._compare_quantize_vs_qdq(_mxfp4_config("oas"), x)

    def test_quantize_3d_non_divisible_last_dim(self) -> None:
        torch.manual_seed(2)
        x = torch.randn(2, 8, 100, dtype=torch.bfloat16, device="cuda")

        result = native_quantize(x, "weight", _mxfp4_config("ceil"), use_triton=False)

        self.assertEqual(result.data.shape, (2, 8, 50))
        self.assertEqual(result.scale.shape, (2, 8, 4))
        self._compare_quantize_vs_qdq(_mxfp4_config("ceil"), x)

    def test_output_format(self) -> None:
        torch.manual_seed(3)
        x = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")

        mxfp4_result = native_quantize(
            x, "weight", _mxfp4_config("oas"), use_triton=False
        )
        self.assertEqual(mxfp4_result.data.dtype, torch.uint8)
        self.assertEqual(mxfp4_result.data.shape, (8, 64))
        self.assertEqual(mxfp4_result.scale.dtype, torch.uint8)
        self.assertEqual(mxfp4_result.scale.shape, (8, 4))

        mxfp8_result = native_quantize(
            x, "weight", _mxfp8_config("ceil"), use_triton=False
        )
        self.assertEqual(mxfp8_result.data.dtype, torch.float8_e4m3fn)
        self.assertEqual(mxfp8_result.data.shape, x.shape)
        self.assertEqual(mxfp8_result.scale.dtype, torch.uint8)
        self.assertEqual(mxfp8_result.scale.shape, (8, 4))

    def test_quantize_matches_qdq_reference_subnormal_blocks(self) -> None:
        # Bottom-of-range / subnormal blocks: amax / max_pow2 is fp32-subnormal,
        # driving the E8M0 scale to the floor (byte 0). The pure-pytorch native
        # path (`_mx_quantize_pure_pytorch`) must produce the same floor scale
        # byte as `qdq`; pre-dividing `amax / max_pow2` would diverge on these
        # blocks.
        patterns = [0x0000, 0x8000, 0x0080, 0x0100, 0x0141, 0x0300]
        signed = [p - 0x10000 if p >= 0x8000 else p for p in patterns]
        vals = (
            torch.tensor(signed, dtype=torch.int16)
            .view(torch.bfloat16)
            .to(torch.float32)
        )
        # One 1x32 block per pattern (uniform block, amax == the value).
        x = (
            vals.unsqueeze(1)
            .expand(len(patterns), 32)
            .to(torch.bfloat16)
            .contiguous()
            .cuda()
        )

        def _qdq_scale_bytes(qi: QuantInfo) -> list[int]:
            u = qi.local_unscale.reshape(-1).float().view(torch.int32)
            return [((int(b) & 0x7FFFFFFF) >> 23) & 0xFF for b in u]

        for config in (_mxfp4_config("oas"), _mxfp8_config("oas_s")):
            with self.subTest(config=config):
                result = native_quantize(x, "weight", config, use_triton=False)
                native_bytes = [int(b) for b in result.scale.reshape(-1)]
                qi = qdq(x.float(), "weight", config, return_quant_info=True)
                qdq_bytes = _qdq_scale_bytes(qi)
                self.assertEqual(
                    native_bytes,
                    qdq_bytes,
                    f"scale byte mismatch {config}: "
                    f"patterns={[hex(p) for p in patterns]} "
                    f"native={native_bytes} qdq={qdq_bytes}",
                )


@unittest.skipIf(
    not torch.cuda.is_available() or not _requires_sm100(),
    "Skip fused MXFP8 native quant test unless running on B200+ (sm100)",
)
class TestMxNativeQuantizeFusedTriton(unittest.TestCase):
    def _assert_equal_bytes(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        message: str,
    ) -> None:
        actual_bytes = actual.contiguous().view(torch.uint8).flatten()
        expected_bytes = expected.contiguous().view(torch.uint8).flatten()
        self.assertTrue(
            torch.equal(actual_bytes, expected_bytes),
            message,
        )

    def _compare_vs_non_triton(
        self,
        x: torch.Tensor,
        config: QuantConfig,
    ) -> None:
        expected = native_quantize(x, "weight", config, use_triton=False)
        actual = native_quantize(x, "weight", config, use_triton=True)

        self.assertEqual(actual.data.dtype, expected.data.dtype)
        self.assertEqual(actual.data.shape, expected.data.shape)
        self.assertEqual(actual.scale.dtype, expected.scale.dtype)
        self.assertEqual(actual.scale.shape, expected.scale.shape)
        self._assert_equal_bytes(
            actual.data,
            expected.data,
            f"Fused native data mismatch for {config}",
        )
        self._assert_equal_bytes(
            actual.scale,
            expected.scale,
            f"Fused native scale mismatch for {config}",
        )

    def test_fused_native_quant_matches_non_triton_for_finite_inputs(self) -> None:
        torch.manual_seed(4)
        x = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")
        configs = [
            _mxfp4_config("oas", block_size=16),
            _mxfp8_config("oas_s", block_size=16),
            _mxfp8_config("ceil", block_size=32),
        ]

        for config in configs:
            with self.subTest(config=config):
                self._compare_vs_non_triton(x, config)

    def test_fused_native_quant_matches_non_triton_with_nan_inputs(self) -> None:
        torch.manual_seed(4)
        x = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")
        # Insert NaN values at various positions
        x[0, 0] = float("nan")
        x[2, 15] = float("nan")
        x[5, 64] = float("nan")
        x[7, 127] = float("nan")
        configs = [
            _mxfp4_config("oas", block_size=16),
            _mxfp8_config("oas_s", block_size=16),
            _mxfp8_config("ceil", block_size=32),
        ]

        for config in configs:
            with self.subTest(config=config):
                self._compare_vs_non_triton(x, config)


@unittest.skipIf(
    not torch.cuda.is_available() or not _requires_sm100(),
    "triton_quantize_mx4_unpack requires B200+ (sm100)",
)
class TestMxNativeQuantizeVsMslk(unittest.TestCase):
    def _compare_quantize_vs_mslk(
        self,
        scale_round_mode: str,
        triton_rounding_mode: Any,
        block_size: int,
    ) -> None:
        try:
            from mslk.quantize.triton.fp4_quantize import triton_quantize_mx4_unpack
        except ImportError:
            self.skipTest("mslk is not available")

        torch.manual_seed(4)
        x = torch.randn(128, 1024, dtype=torch.bfloat16, device="cuda")

        expected_data, expected_scale = triton_quantize_mx4_unpack(
            x,
            group_size=block_size,
            rounding_mode=triton_rounding_mode,
        )
        result = native_quantize(
            x,
            "weight",
            _mxfp4_config(scale_round_mode, block_size=block_size),
            is_sf_swizzled_layout=True,
            use_triton=True,
        )

        self.assertTrue(
            torch.equal(result.data.view(torch.uint8), expected_data.view(torch.uint8)),
            f"Packed data mismatch for {scale_round_mode=} {block_size=}",
        )
        self.assertTrue(
            torch.equal(result.scale.view(torch.int8), expected_scale.view(torch.int8)),
            f"Swizzled scale mismatch for {scale_round_mode=} {block_size=}",
        )

    def test_quantize_matches_mslk_triton_quantize_mx4_unpack(self) -> None:
        try:
            from mslk.quantize.triton.fp4_quantize import RoundingMode
        except ImportError:
            self.skipTest("mslk is not available")

        cases = [
            ("oas", RoundingMode.even, 32),
            ("ceil", RoundingMode.ceil, 32),
            ("floor", RoundingMode.floor, 32),
            ("oas", RoundingMode.even, 16),
            ("ceil", RoundingMode.ceil, 16),
            ("floor", RoundingMode.floor, 16),
        ]
        for scale_round_mode, triton_rounding_mode, block_size in cases:
            with self.subTest(
                scale_round_mode=scale_round_mode,
                block_size=block_size,
            ):
                self._compare_quantize_vs_mslk(
                    scale_round_mode,
                    triton_rounding_mode,
                    block_size,
                )


@unittest.skipIf(
    not torch.cuda.is_available() or not _requires_sm100(),
    "flashinfer quantize requires B200+ (sm100)",
)
class TestMxNativeQuantizeVsFlashInfer(unittest.TestCase):
    def _assert_equal_uint8(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        message: str,
    ) -> None:
        actual_bytes = actual.contiguous().view(torch.uint8).flatten()
        expected_bytes = expected.contiguous().view(torch.uint8).flatten()
        self.assertEqual(actual_bytes.numel(), expected_bytes.numel())
        if torch.equal(actual_bytes, expected_bytes):
            return

        mismatch = actual_bytes != expected_bytes
        mismatch_indices = torch.nonzero(mismatch).flatten()[:16]
        self.fail(
            f"{message}: {int(mismatch.sum().item())}/{actual_bytes.numel()} bytes differ; "
            f"first_indices={mismatch_indices.cpu().tolist()}; "
            f"actual={actual_bytes[mismatch_indices].cpu().tolist()}; "
            f"expected={expected_bytes[mismatch_indices].cpu().tolist()}"
        )

    def test_mxfp4_matches_flashinfer(self) -> None:
        try:
            from flashinfer import fp4_quantize
        except ImportError:
            self.skipTest("flashinfer is not available")

        torch.manual_seed(5)
        x = torch.randn(64, 1024, dtype=torch.bfloat16, device="cuda")

        expected_data, expected_scale = fp4_quantize(
            x,
            global_scale=None,
            sf_vec_size=32,
            sf_use_ue8m0=True,
            is_sf_swizzled_layout=False,
        )
        result = native_quantize(
            x,
            "weight",
            _config(
                "mxfp4",
                block_size="1x32",
                scale_norm_mode="amax",
                scale_round_mode="ceil",
            ),
            use_triton=True,
        )

        self._assert_equal_uint8(
            result.data,
            expected_data,
            "Packed data mismatch vs flashinfer.fp4_quantize",
        )
        self._assert_equal_uint8(
            result.scale,
            expected_scale.reshape(result.scale.shape),
            "Scale mismatch vs flashinfer.fp4_quantize",
        )

    def test_mxfp8_matches_flashinfer(self) -> None:
        try:
            from flashinfer import mxfp8_quantize
        except ImportError:
            self.skipTest("flashinfer is not available")

        torch.manual_seed(6)
        x = torch.randn(64, 1024, dtype=torch.bfloat16, device="cuda")

        expected_data, expected_scale = mxfp8_quantize(
            x,
            is_sf_swizzled_layout=False,
        )
        result = native_quantize(
            x,
            "weight",
            _config(
                "mxfp8",
                block_size="1x32",
                scale_norm_mode="amax",
                scale_round_mode="ceil",
            ),
            use_triton=True,
        )

        self._assert_equal_uint8(
            result.data,
            expected_data,
            "Data mismatch vs flashinfer.mxfp8_quantize",
        )
        self._assert_equal_uint8(
            result.scale,
            expected_scale.reshape(result.scale.shape),
            "Scale mismatch vs flashinfer.mxfp8_quantize",
        )


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Skip when cuda is not available",
)
class TestMxDequantize(unittest.TestCase):
    def test_roundtrip_matches_qdq_reference(self) -> None:
        torch.manual_seed(5)
        x = torch.randn(16, 128, dtype=torch.bfloat16, device="cuda")

        for config in (_mxfp4_config("oas"), _mxfp8_config("ceil")):
            with self.subTest(config=config):
                result = native_quantize(x, "weight", config, use_triton=False)
                dequantized = native_dequantize(result, output_dtype=torch.float32)
                expected = qdq(x.float(), "weight", config)

                torch.testing.assert_close(dequantized, expected, rtol=0, atol=0)

    def test_swizzled_scale_layout(self) -> None:
        torch.manual_seed(6)
        x = torch.randn(129, 160, dtype=torch.float32, device="cuda")

        row_major_result = native_quantize(
            x,
            "weight",
            _mxfp4_config("floor"),
            is_sf_swizzled_layout=False,
            use_triton=False,
        )
        swizzled_result = native_quantize(
            x,
            "weight",
            _mxfp4_config("floor"),
            is_sf_swizzled_layout=True,
            use_triton=False,
        )

        self.assertEqual(swizzled_result.scale.ndim, 1)
        self.assertGreater(
            swizzled_result.scale.numel(),
            row_major_result.scale.numel(),
        )

        expected = native_dequantize(row_major_result, output_dtype=torch.float32)
        actual = native_dequantize(
            swizzled_result,
            output_dtype=torch.float32,
            is_sf_swizzled_layout=True,
        )

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _nvfp4_config(global_scale_granularity: str = "tensorwise") -> QuantConfig:
    return _config("nvfp4", global_scale_granularity=global_scale_granularity)


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Skip when cuda is not available",
)
class TestNvfp4NativeQuantize(unittest.TestCase):
    def test_output_format_tensorwise(self) -> None:
        torch.manual_seed(9)
        x = torch.randn(16, 128, dtype=torch.bfloat16, device="cuda")
        result = native_quantize(x, "weight", _nvfp4_config(), use_triton=False)

        self.assertEqual(result.data.dtype, torch.uint8)
        self.assertEqual(result.data.shape, (16, 64))
        self.assertEqual(result.scale.dtype, torch.float8_e4m3fn)
        self.assertEqual(result.scale.shape, (16, 8))
        self.assertIsNotNone(result.global_scale)
        assert result.global_scale is not None
        self.assertEqual(result.global_scale.numel(), 1)

    def test_output_format_rowwise(self) -> None:
        torch.manual_seed(10)
        x = torch.randn(2, 8, 128, dtype=torch.bfloat16, device="cuda")
        result = native_quantize(
            x,
            "weight",
            _nvfp4_config(global_scale_granularity="rowwise"),
            use_triton=False,
        )

        self.assertEqual(result.data.dtype, torch.uint8)
        self.assertEqual(result.data.shape, (2, 8, 64))
        self.assertEqual(result.scale.dtype, torch.float8_e4m3fn)
        self.assertEqual(result.scale.shape, (2, 8, 8))
        self.assertIsNotNone(result.global_scale)
        assert result.global_scale is not None
        self.assertEqual(result.global_scale.shape, (2, 8))

    def test_roundtrip_matches_qdq_tensorwise(self) -> None:
        torch.manual_seed(11)
        x = torch.randn(16, 128, dtype=torch.bfloat16, device="cuda")
        config = _nvfp4_config()

        dequantized = native_dequantize(
            native_quantize(x, "weight", config, use_triton=False),
            output_dtype=torch.float32,
        )
        expected = qdq(x.float(), "weight", config)

        torch.testing.assert_close(dequantized, expected, rtol=0, atol=0)

    def test_roundtrip_matches_qdq_rowwise(self) -> None:
        torch.manual_seed(12)
        x = torch.randn(2, 8, 128, dtype=torch.bfloat16, device="cuda")
        config = _nvfp4_config(global_scale_granularity="rowwise")

        dequantized = native_dequantize(
            native_quantize(x, "weight", config, use_triton=False),
            output_dtype=torch.float32,
        )
        expected = qdq(x.float(), "weight", config)

        torch.testing.assert_close(dequantized, expected, rtol=0, atol=0)

    def test_sqnr_is_reasonable(self) -> None:
        torch.manual_seed(13)
        x = torch.randn(64, 512, dtype=torch.bfloat16, device="cuda")
        dequantized = native_dequantize(
            native_quantize(x, "weight", _nvfp4_config(), use_triton=False),
            output_dtype=torch.float32,
        )

        noise = x.float() - dequantized
        sqnr_db = 10 * torch.log10((x.float() ** 2).mean() / (noise**2).mean())
        self.assertGreater(sqnr_db.item(), 15.0)


@unittest.skipIf(
    not torch.cuda.is_available() or not _requires_sm100(),
    "Skip fused NVFP4 native quant test unless running on B200+ (sm100)",
)
class TestNvfp4NativeQuantizeFusedTriton(unittest.TestCase):
    def _compare_vs_non_triton(self, x: torch.Tensor, config: QuantConfig) -> None:
        expected = native_quantize(x, "weight", config, use_triton=False)
        actual = native_quantize(x, "weight", config, use_triton=True)

        self.assertEqual(actual.data.dtype, expected.data.dtype)
        self.assertEqual(actual.data.shape, expected.data.shape)
        self.assertEqual(actual.scale.dtype, expected.scale.dtype)
        self.assertEqual(actual.scale.shape, expected.scale.shape)
        self.assertTrue(torch.equal(actual.data, expected.data))
        self.assertTrue(
            torch.equal(
                actual.scale.view(torch.uint8),
                expected.scale.view(torch.uint8),
            )
        )
        self.assertIsNotNone(actual.global_scale)
        self.assertIsNotNone(expected.global_scale)
        assert actual.global_scale is not None
        assert expected.global_scale is not None
        torch.testing.assert_close(
            actual.global_scale,
            expected.global_scale,
            rtol=0,
            atol=0,
        )

    def test_fused_native_quant_matches_non_triton_tensorwise(self) -> None:
        torch.manual_seed(14)
        x = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")
        self._compare_vs_non_triton(x, _nvfp4_config())

    def test_fused_native_quant_matches_non_triton_rowwise(self) -> None:
        torch.manual_seed(15)
        x = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")
        self._compare_vs_non_triton(
            x,
            _nvfp4_config(global_scale_granularity="rowwise"),
        )


@unittest.skipIf(
    not torch.cuda.is_available() or not _requires_sm100(),
    "Skip fused NVFP4 native quant test unless running on B200+ (sm100)",
)
class TestNvfp4NativeAllZero(unittest.TestCase):
    def test_all_zero_tensor_and_rows(self) -> None:
        # Both paths give an all-zero tensor, or an all-zero row under ROWWISE,
        # the global scale 1.0 and block scales 0, and dequantize it to zeros.
        torch.manual_seed(0)
        zeros = torch.zeros(4, 64, dtype=torch.bfloat16, device="cuda")
        one_row = zeros.clone()
        one_row[1] = torch.randn(64, dtype=torch.bfloat16, device="cuda")
        for granularity, x in (
            ("tensorwise", zeros),
            ("rowwise", zeros),
            ("rowwise", one_row),
        ):
            config = _nvfp4_config(global_scale_granularity=granularity)
            results = [
                native_quantize(x, "x", config, use_triton=use_triton)
                for use_triton in (True, False)
            ]
            for result, use_triton in zip(results, (True, False)):
                what = f"{granularity} rows={int(x.abs().amax(1).gt(0).sum())} {use_triton=}"
                assert result.global_scale is not None
                zero_rows = x.abs().amax(dim=1) == 0
                gs = result.global_scale.reshape(-1).cpu()
                if granularity == "rowwise":
                    self.assertTrue(bool((gs[zero_rows.cpu()] == 1.0).all()), what)
                else:
                    self.assertEqual(gs.tolist(), [1.0], what)
                scale = result.scale.float().cpu()
                self.assertTrue(bool((scale[zero_rows.cpu()] == 0).all()), what)
                out = native_dequantize(result, output_dtype=torch.float32)
                self.assertFalse(bool(torch.isnan(out).any()), what)
                self.assertTrue(bool((out[zero_rows] == 0).all()), what)
            triton, pure = results
            self.assertTrue(torch.equal(triton.data, pure.data))
            self.assertTrue(
                torch.equal(
                    triton.scale.view(torch.uint8), pure.scale.view(torch.uint8)
                )
            )
            assert triton.global_scale is not None and pure.global_scale is not None
            self.assertTrue(torch.equal(triton.global_scale, pure.global_scale))


@unittest.skipIf(
    not torch.cuda.is_available() or not _requires_sm100(),
    "Skip fused native stochastic rounding test unless running on B200+ (sm100)",
)
class TestNativeQuantizeFusedTritonStochastic(unittest.TestCase):
    """The fused native path rounds stochastically exactly as qdq_triton_mega.

    Both launch the same kernel with the same 1x16 tiling, so under one CUDA RNG
    state they draw the same random bits and must produce the same elements.
    """

    def _native_elements(self, result: NativeQuantResult) -> torch.Tensor:
        if result.config.elem_dtype == ElemDtype.E2M1:
            return packed_uint8_to_float_e2m1(result.data)
        return result.data.to(torch.float32)

    def test_matches_qdq_triton_mega(self) -> None:
        torch.manual_seed(16)
        x = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda")
        for preset in ("mxfp4_sr", "mxfp8_sr", "nvfp4_sr"):
            config = _config(preset)
            for seed in (1234, 5678):
                with self.subTest(preset=preset, seed=seed):
                    torch.cuda.manual_seed(seed)
                    native = native_quantize(x, "weight", config, use_triton=True)
                    torch.cuda.manual_seed(seed)
                    mega = qdq_triton_mega(x, "weight", config, return_quant_info=True)
                    self.assertTrue(
                        torch.equal(self._native_elements(native), mega.quantized_val),
                        f"native and qdq_triton_mega elements differ for {preset}",
                    )

    def test_follows_cuda_rng_state(self) -> None:
        torch.manual_seed(17)
        x = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda")
        config = _config("mxfp8_sr")
        torch.cuda.manual_seed(1234)
        first = native_quantize(x, "weight", config, use_triton=True)
        torch.cuda.manual_seed(5678)
        second = native_quantize(x, "weight", config, use_triton=True)
        self.assertFalse(
            torch.equal(first.data.view(torch.uint8), second.data.view(torch.uint8))
        )


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Skip when cuda is not available",
)
class TestMxScaleSwizzle(unittest.TestCase):
    def test_swizzle_unswizzle_roundtrip(self) -> None:
        scale = torch.randint(0, 255, (128, 8), dtype=torch.uint8, device="cuda")

        swizzled = _swizzle_scale(scale)
        unswizzled = _unswizzle_scale(swizzled, 128, 8)

        self.assertTrue(torch.equal(scale, unswizzled))

    def test_swizzle_unswizzle_non_aligned(self) -> None:
        scale = torch.randint(0, 255, (129, 5), dtype=torch.uint8, device="cuda")

        swizzled = _swizzle_scale(scale)
        unswizzled = _unswizzle_scale(swizzled, 129, 5)

        self.assertTrue(torch.equal(scale, unswizzled))

    def test_quantize_swizzled_scale_matches_row_major_scale(self) -> None:
        torch.manual_seed(7)
        x = torch.randn(129, 160, dtype=torch.bfloat16, device="cuda")

        row_major_result = native_quantize(
            x,
            "weight",
            _mxfp4_config("oas"),
            is_sf_swizzled_layout=False,
            use_triton=False,
        )
        swizzled_result = native_quantize(
            x,
            "weight",
            _mxfp4_config("oas"),
            is_sf_swizzled_layout=True,
            use_triton=False,
        )

        self.assertTrue(torch.equal(row_major_result.data, swizzled_result.data))
        self.assertTrue(
            torch.equal(
                row_major_result.scale,
                _unswizzle_scale(swizzled_result.scale, 129, 5),
            )
        )

    def test_quantize_swizzled_scale_layout_requires_2d(self) -> None:
        torch.manual_seed(8)
        x = torch.randn(2, 8, 128, dtype=torch.bfloat16, device="cuda")

        with self.assertRaisesRegex(NotImplementedError, "only supported for 2D"):
            native_quantize(
                x,
                "weight",
                _mxfp4_config("oas"),
                is_sf_swizzled_layout=True,
                use_triton=False,
            )


class TestNativeMxHelpers(unittest.TestCase):
    """Pure-host helper functions (no CUDA required)."""

    def test_native_data_shape(self) -> None:
        self.assertEqual(
            native_data_shape(torch.Size([4, 32]), ElemDtype.E2M1),
            torch.Size([4, 16]),
        )
        # Odd last dim rounds up.
        self.assertEqual(
            native_data_shape(torch.Size([4, 31]), ElemDtype.E2M1),
            torch.Size([4, 16]),
        )
        self.assertEqual(
            native_data_shape(torch.Size([4, 32]), ElemDtype.E4M3),
            torch.Size([4, 32]),
        )
        with self.assertRaises(NotImplementedError):
            native_data_shape(torch.Size([4, 32]), ElemDtype.E2M3)

    def test_native_dtype_matches(self) -> None:
        u8 = torch.zeros(4, dtype=torch.uint8)
        fp8 = torch.zeros(4, dtype=torch.float8_e4m3fn)
        self.assertTrue(native_dtype_matches(u8, ElemDtype.E2M1))
        self.assertFalse(native_dtype_matches(fp8, ElemDtype.E2M1))
        self.assertTrue(native_dtype_matches(fp8, ElemDtype.E4M3))
        self.assertFalse(native_dtype_matches(u8, ElemDtype.E4M3))
        with self.assertRaises(NotImplementedError):
            native_dtype_matches(u8, ElemDtype.E2M3)


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Skip when cuda is not available",
)
class TestNativeQuantConfigValidation(unittest.TestCase):
    """_validate_native_quant_config error branches, reached through native_quantize."""

    def _quantize_expect_error(self, overrides: dict[str, object], regex: str) -> None:
        config = QuantConfig.from_dict(overrides)
        x = torch.randn(4, 32, dtype=torch.bfloat16, device="cuda")
        with self.assertRaisesRegex(NotImplementedError, regex):
            native_quantize(x, "weight", config, use_triton=False)

    def test_rejects_asymm(self) -> None:
        self._quantize_expect_error(
            {**FAKE_PRESETS["mxfp4"], "quant_mode": "asymm"},
            "only supports SYMM mode",
        )

    def test_rejects_unsupported_elem_dtype(self) -> None:
        self._quantize_expect_error(
            FAKE_PRESETS["mxfp6"],
            "supports E2M1",
        )

    def test_rejects_nvfp4_non_e2m1_elem(self) -> None:
        self._quantize_expect_error(
            {
                "elem_dtype": "e4m3",
                "scale_dtype": "fp8",
                "scale_norm_mode": "amax",
                "scale_round_mode": "none",
                "global_scale_granularity": "tensorwise",
                "global_scale_dtype": "fp32",
                "block_size": "1x16",
            },
            "NVFP4 only supports E2M1",
        )

    def test_rejects_nvfp4_non_tensorwise_rowwise(self) -> None:
        self._quantize_expect_error(
            {
                "elem_dtype": "e2m1",
                "scale_dtype": "fp8",
                "scale_norm_mode": "amax",
                "scale_round_mode": "none",
                "global_scale_granularity": "mbs16",
                "global_scale_dtype": "e0m8",
                "block_size": "1x16",
            },
            "requires TENSORWISE or ROWWISE",
        )

    def test_rejects_unsupported_scale_dtype(self) -> None:
        self._quantize_expect_error(
            {
                "elem_dtype": "e2m1",
                "scale_dtype": "fp16",
                "scale_norm_mode": "amax",
                "scale_round_mode": "none",
                "block_size": "1x16",
            },
            "only supports E8M0 or FP8",
        )

    def test_rejects_non_1xn_block(self) -> None:
        self._quantize_expect_error(
            {**FAKE_PRESETS["mxfp4"], "block_size": "2x16"},
            "only supports 1xN",
        )

    def test_rejects_mbs(self) -> None:
        self._quantize_expect_error(
            FAKE_PRESETS["mxfp4_mbs128"],
            "does not support MBS .* mbs_block_size=128 and mbs_dynamic=False",
        )

    def test_rejects_mbs_dynamic(self) -> None:
        self._quantize_expect_error(
            FAKE_PRESETS["mxfp4_mbs256_dyn"],
            "does not support MBS .* mbs_block_size=256 and mbs_dynamic=True",
        )


class TestNativeE2M1CapabilityGuard(unittest.TestCase):
    """native_quantize rejects fused Triton E2M1 packing below SM100."""

    def _patch_capability(self, capability: tuple[int, int]) -> mock.MagicMock:
        patcher = mock.patch.object(
            torch.cuda, "get_device_capability", return_value=capability
        )
        self.addCleanup(patcher.stop)
        return patcher.start()

    def _patch_fused_triton(self) -> mock.MagicMock:
        patcher = mock.patch.object(
            native_quant_module, "_mx_quantize_fused_triton", autospec=True
        )
        self.addCleanup(patcher.stop)
        return patcher.start()

    @unittest.skipIf(not torch.cuda.is_available(), "requires a CUDA tensor")
    def test_rejects_mxfp4_below_sm100(self) -> None:
        get_capability = self._patch_capability((9, 0))
        fused = self._patch_fused_triton()
        x = torch.randn(4, 32, dtype=torch.bfloat16, device="cuda")
        with self.assertRaisesRegex(
            RuntimeError, r"requires SM100 or newer.*'weight'.*\(SM90\)"
        ):
            native_quantize(x, "weight", _config("mxfp4"))
        get_capability.assert_called_once_with(x.device)
        fused.assert_not_called()

    @unittest.skipIf(not torch.cuda.is_available(), "requires a CUDA tensor")
    def test_rejects_nvfp4_below_sm100(self) -> None:
        self._patch_capability((9, 0))
        fused = self._patch_fused_triton()
        x = torch.randn(4, 32, dtype=torch.bfloat16, device="cuda")
        with self.assertRaisesRegex(RuntimeError, "requires SM100"):
            native_quantize(x, "weight", _config("nvfp4"))
        fused.assert_not_called()

    @unittest.skipIf(not torch.cuda.is_available(), "requires a CUDA tensor")
    def test_allows_mxfp4_on_sm100(self) -> None:
        self._patch_capability((10, 0))
        fused = self._patch_fused_triton()
        x = torch.randn(4, 32, dtype=torch.bfloat16, device="cuda")
        native_quantize(x, "weight", _config("mxfp4"))
        fused.assert_called_once()

    @unittest.skipIf(not torch.cuda.is_available(), "requires a CUDA tensor")
    def test_allows_mxfp8_below_sm100(self) -> None:
        get_capability = self._patch_capability((9, 0))
        fused = self._patch_fused_triton()
        x = torch.randn(4, 32, dtype=torch.bfloat16, device="cuda")
        native_quantize(x, "weight", _config("mxfp8"))
        get_capability.assert_not_called()
        fused.assert_called_once()

    @unittest.skipIf(not torch.cuda.is_available(), "requires a CUDA tensor")
    def test_allows_pure_pytorch_mxfp4_below_sm100(self) -> None:
        get_capability = self._patch_capability((9, 0))
        x = torch.randn(4, 32, dtype=torch.bfloat16, device="cuda")
        result = native_quantize(x, "weight", _config("mxfp4"), use_triton=False)
        self.assertEqual(result.data.shape, (4, 16))
        get_capability.assert_not_called()

    def test_skips_capability_query_for_cpu_tensor(self) -> None:
        get_capability = self._patch_capability((9, 0))
        fused = self._patch_fused_triton()
        x = torch.randn(4, 32, dtype=torch.bfloat16)
        native_quantize(x, "weight", _config("mxfp4"))
        get_capability.assert_not_called()
        fused.assert_called_once()


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Skip when cuda is not available",
)
class TestNativeMxPurePytorchPaths(unittest.TestCase):
    """Under-covered branches of _mx_quantize_pure_pytorch."""

    def test_amax_scale_norm_roundtrip(self) -> None:
        # AMAX (rather than POW2MAX) E8M0 scale branch.
        config = QuantConfig.from_dict(
            {**FAKE_PRESETS["mxfp4"], "scale_norm_mode": "amax"}
        )
        x = torch.randn(16, 64, dtype=torch.bfloat16, device="cuda")
        result = native_quantize(x, "weight", config, use_triton=False)
        dequantized = native_dequantize(result, output_dtype=torch.float32)
        expected = qdq(x.float(), "weight", config)
        torch.testing.assert_close(dequantized, expected, rtol=0, atol=0)

    def test_stochastic_rounding_runs(self) -> None:
        config = QuantConfig.from_dict(FAKE_PRESETS["mxfp4_sr"])
        x = torch.randn(16, 64, dtype=torch.bfloat16, device="cuda")
        result = native_quantize(x, "weight", config, use_triton=False)
        self.assertEqual(result.data.dtype, torch.uint8)
        self.assertEqual(result.data.shape, (16, 32))

    def test_1d_input(self) -> None:
        config = QuantConfig.from_dict(FAKE_PRESETS["mxfp4"])
        x = torch.randn(64, dtype=torch.bfloat16, device="cuda")
        result = native_quantize(x, "weight", config, use_triton=False)
        # 1D input -> data squeezed back to 1D packed FP4.
        self.assertEqual(result.data.ndim, 1)
        self.assertEqual(result.data.shape, (32,))
        self.assertEqual(result.scale.ndim, 1)


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Skip when cuda is not available",
)
class TestMxDequantizeErrors(unittest.TestCase):
    """native_dequantize validation / error branches."""

    def test_unsupported_elem_dtype(self) -> None:
        config = QuantConfig.from_dict(FAKE_PRESETS["mxfp6"])  # e2m3
        dummy = torch.zeros(1, 4, dtype=torch.uint8, device="cuda")
        result = NativeQuantResult(
            data=dummy,
            scale=dummy,
            config=config,
            global_scale=None,
            orig_shape=(1, 4),
        )
        with self.assertRaisesRegex(NotImplementedError, "Unsupported elem dtype"):
            native_dequantize(result)

    def test_swizzled_layout_rejects_fp8(self) -> None:
        x = torch.randn(16, 64, dtype=torch.bfloat16, device="cuda")
        result = native_quantize(x, "weight", _nvfp4_config(), use_triton=False)
        with self.assertRaisesRegex(ValueError, "not supported for FP8 scale dtype"):
            native_dequantize(result, is_sf_swizzled_layout=True)

    def test_swizzled_layout_requires_2d(self) -> None:
        x = torch.randn(2, 8, 64, dtype=torch.bfloat16, device="cuda")
        result = native_quantize(
            x,
            "weight",
            _mxfp4_config("oas"),
            is_sf_swizzled_layout=False,
            use_triton=False,
        )
        with self.assertRaisesRegex(ValueError, "only supported for 2D"):
            native_dequantize(result, is_sf_swizzled_layout=True)

    def test_swizzled_layout_requires_1d_scale(self) -> None:
        x = torch.randn(16, 64, dtype=torch.bfloat16, device="cuda")
        result = native_quantize(
            x,
            "weight",
            _mxfp4_config("oas"),
            is_sf_swizzled_layout=False,
            use_triton=False,
        )
        with self.assertRaisesRegex(ValueError, "Expected flattened 1D scale"):
            native_dequantize(result, is_sf_swizzled_layout=True)

    def test_nvfp4_requires_global_scale(self) -> None:
        x = torch.randn(16, 64, dtype=torch.bfloat16, device="cuda")
        result = native_quantize(x, "weight", _nvfp4_config(), use_triton=False)
        result_no_gs = NativeQuantResult(
            data=result.data,
            scale=result.scale,
            config=result.config,
            global_scale=None,
            orig_shape=result.orig_shape,
        )
        with self.assertRaisesRegex(ValueError, "requires global_scale"):
            native_dequantize(result_no_gs)


if __name__ == "__main__":
    unittest.main()
