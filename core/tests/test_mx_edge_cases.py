# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
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
    ScaleDtype,
    ScaleNormMode,
    ScaleRoundMode,
)
from quant_suite_core.fake_presets import FAKE_PRESETS
from quant_suite_core.qdq_pytorch import qdq
from quant_suite_core.qdq_triton import qdq_triton_mega
from quant_test_utils import e8m0_scale_byte, make_quant_config, skip_if_no_cuda


@skip_if_no_cuda
class TestEdgeCases(unittest.TestCase):
    """Test edge cases like inf, nan, and zero blocks."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.device = "cuda"

    def _create_config(
        self,
        elem_dtype: ElemDtype,
        scale_dtype: ScaleDtype,
        scale_norm_mode: ScaleNormMode,
        scale_round_mode: ScaleRoundMode,
        global_scale_granularity: GlobalScaleGranularity,
    ) -> QuantConfig:
        return make_quant_config(
            scale_dtype=scale_dtype,
            scale_norm_mode=scale_norm_mode,
            scale_round_mode=scale_round_mode,
            global_scale_granularity=global_scale_granularity,
            elem_dtype=elem_dtype,
            elem_round_mode=ElemRoundMode.NEAREST,
            block_size=BlockSize(1, 16),
            global_scale_dtype=(
                ScaleDtype.FP32
                if global_scale_granularity != GlobalScaleGranularity.NONE
                else ScaleDtype.NONE
            ),
        )

    @parameterized.expand(
        [
            # (name, use_triton, elem_dtype, scale_dtype, scale_norm_mode, scale_round_mode, global_scale_granularity)
            (
                "MX_E2M1_triton",
                True,
                ElemDtype.E2M1,
                ScaleDtype.E8M0,
                ScaleNormMode.POW2MAX,
                ScaleRoundMode.OAS,
                GlobalScaleGranularity.NONE,
            ),
            (
                "MX_E2M1_non_triton",
                False,
                ElemDtype.E2M1,
                ScaleDtype.E8M0,
                ScaleNormMode.POW2MAX,
                ScaleRoundMode.OAS,
                GlobalScaleGranularity.NONE,
            ),
            (
                "NV_E2M1_triton",
                True,
                ElemDtype.E2M1,
                ScaleDtype.FP8,
                ScaleNormMode.AMAX,
                ScaleRoundMode.NONE,
                GlobalScaleGranularity.TENSORWISE,
            ),
            (
                "NV_E2M1_non_triton",
                False,
                ElemDtype.E2M1,
                ScaleDtype.FP8,
                ScaleNormMode.AMAX,
                ScaleRoundMode.NONE,
                GlobalScaleGranularity.TENSORWISE,
            ),
        ]
    )
    def test_mixed_inf_nan_normal(
        self,
        _name: str,
        use_triton: bool,
        elem_dtype: ElemDtype,
        scale_dtype: ScaleDtype,
        scale_norm_mode: ScaleNormMode,
        scale_round_mode: ScaleRoundMode,
        global_scale_granularity: GlobalScaleGranularity,
    ) -> None:
        """Mix of inf, nan, and normal values should be handled correctly."""
        tensor = torch.tensor(
            [
                [float("inf"), float("nan"), 1.0, 2.0]
                + [0.5] * 12
                + [float("-inf")]
                + [3.0] * 15
            ],
            dtype=torch.bfloat16,
            device=self.device,
        )
        config = self._create_config(
            elem_dtype=elem_dtype,
            scale_dtype=scale_dtype,
            scale_norm_mode=scale_norm_mode,
            scale_round_mode=scale_round_mode,
            global_scale_granularity=global_scale_granularity,
        )

        result = fake_quantize(
            tensor, "test", config, transpose=False, use_triton=use_triton
        )

        if scale_dtype == ScaleDtype.E8M0:
            # MX: a NaN poisons its entire block.
            # Block 0 (cols 0-15) contains a NaN, so every element -- including
            # the +inf and the normal 1.0/2.0/0.5 values -- dequantizes to NaN.
            # Block 1 (cols 16-31) contains only -inf (no NaN), so -inf
            # saturates to -inf and its finite 3.0 neighbors stay finite.
            self.assertTrue(
                torch.isnan(result[0, 0:16]).all(),
                "A NaN must poison its entire MX block",
            )
            self.assertTrue(
                torch.isinf(result[0, 16]),
                "Negative inf (no NaN in its block) should saturate to -inf",
            )
            self.assertFalse(
                torch.isnan(result[0, 17]) or torch.isinf(result[0, 17]),
                "Finite neighbor in an inf-only block should stay finite",
            )
        else:
            # NVFP4 (and other non-MX formats): Inf/NaN pass through verbatim.
            self.assertTrue(
                torch.isinf(result[0, 0]), "Positive inf should be preserved"
            )
            self.assertTrue(torch.isnan(result[0, 1]), "NaN should be preserved")
            self.assertTrue(
                torch.isinf(result[0, 16]), "Negative inf should be preserved"
            )
            self.assertFalse(
                torch.isnan(result[0, 2]) or torch.isinf(result[0, 2]),
                "Normal values should remain normal",
            )

    @parameterized.expand(
        [
            ("mxfp4", "mxfp4"),
            ("mxfp6", "mxfp6"),
            ("mxfp8", "mxfp8"),
        ]
    )
    def test_mx_nan_poisons_entire_block(self, _name: str, preset_name: str) -> None:
        """A single NaN must poison its whole MX block.

        The MX scale computation returns the E8M0 NaN scale byte (255) when any
        block element is NaN, so the entire block dequantizes to NaN -- not just
        the NaN element.
        """
        config = QuantConfig.from_dict(FAKE_PRESETS[preset_name])
        # 16 values: one NaN plus 15 finite. The NaN must poison the rest.
        vals = [float("nan")] + [float(i) for i in range(1, 16)]
        tensor = torch.tensor([vals], dtype=torch.bfloat16, device=self.device)

        result = qdq(tensor, "test", config)

        self.assertTrue(
            torch.isnan(result).all(),
            f"{preset_name}: one NaN must poison the entire 16-wide block",
        )

    @parameterized.expand(
        [
            # preset, expected E8M0 scale byte, signed max-power-of-two code
            ("mxfp4", 253, 4.0),
            ("mxfp6", 253, 4.0),
            ("mxfp8", 247, 256.0),
        ]
    )
    def test_mx_inf_saturates_to_max_pow2(
        self,
        preset_name: str,
        exp_scale_byte: int,
        exp_code_val: float,
    ) -> None:
        """±Inf saturates to a real E8M0 scale byte + signed max-pow2 code.

        Inf is clamped to ±FLT_MAX before scale selection, so
        the block gets a normal large scale byte (253 for MX4/MX6, 247 for MX8)
        and the Inf element lands on the signed max-power-of-two code
        (±4 / ±256). The fp32 dequant then overflows back to ±Inf, with the sign
        preserved.
        """
        config = QuantConfig.from_dict(FAKE_PRESETS[preset_name])
        for sign in (1.0, -1.0):
            tensor = torch.full(
                (1, 16),
                sign * float("inf"),
                dtype=torch.bfloat16,
                device=self.device,
            )
            info = qdq(tensor, "test", config, return_quant_info=True)

            scale_byte = e8m0_scale_byte(info.local_unscale)
            self.assertTrue(
                bool((scale_byte == exp_scale_byte).all()),
                f"{preset_name} sign={sign}: scale byte "
                f"{int(scale_byte.flatten()[0].item())} != {exp_scale_byte}",
            )
            torch.testing.assert_close(
                info.quantized_val,
                torch.full_like(info.quantized_val, sign * exp_code_val),
            )
            self.assertTrue(
                torch.isinf(info.dequant_data).all()
                and bool((torch.sign(info.dequant_data) == sign).all()),
                f"{preset_name} sign={sign}: dequant should overflow to signed Inf",
            )

    @parameterized.expand(
        [
            # preset, canonical NaN-block element value (sign dropped).
            # E2M1/E2M3 cannot encode NaN -> +max; E4M3 encodes NaN (0x7F).
            ("mxfp4", 6.0),
            ("mxfp6", 7.5),
            ("mxfp8", float("nan")),
        ]
    )
    def test_mx_nan_block_encoding(self, preset_name: str, exp_code_val: float) -> None:
        """NaN block -> E8M0 NaN scale (255) + canonical code, dequant all-NaN.

        Both +NaN and -NaN map to the same positive code; here we lock the
        encoding + poison.
        """
        config = QuantConfig.from_dict(FAKE_PRESETS[preset_name])
        tensor = torch.full(
            (1, 16), float("nan"), dtype=torch.bfloat16, device=self.device
        )

        info = qdq(tensor, "test", config, return_quant_info=True)

        scale_byte = e8m0_scale_byte(info.local_unscale)
        self.assertTrue(
            bool((scale_byte == 255).all()),
            f"{preset_name}: NaN block must use the E8M0 NaN scale byte 255",
        )
        if math.isnan(exp_code_val):
            self.assertTrue(
                torch.isnan(info.quantized_val).all(),
                f"{preset_name}: E4M3 NaN block elements must encode as NaN (0x7F)",
            )
        else:
            torch.testing.assert_close(
                info.quantized_val,
                torch.full_like(info.quantized_val, exp_code_val),
            )
        self.assertTrue(
            torch.isnan(info.dequant_data).all(),
            f"{preset_name}: NaN block must dequantize to all-NaN",
        )

    def _assert_qdq_paths_equal(
        self, ref: QuantInfo, other: QuantInfo, label: str
    ) -> None:
        """Assert two QuantInfo results are bit-identical (NaN positions aligned)."""
        for field in ("dequant_data", "quantized_val", "local_unscale"):
            a = getattr(ref, field)
            b = getattr(other, field)
            self.assertEqual(a.shape, b.shape, f"{label}: {field} shape mismatch")
            self.assertTrue(
                torch.equal(torch.isnan(a), torch.isnan(b)),
                f"{label}: {field} NaN-mask mismatch",
            )
            finite = ~torch.isnan(a)
            torch.testing.assert_close(
                a[finite],
                b[finite],
                rtol=0,
                atol=0,
                msg=f"{label}: {field} value mismatch",
            )

    @parameterized.expand(
        [
            ("mxfp4", "mxfp4"),
            ("mxfp6", "mxfp6"),
            ("mxfp8", "mxfp8"),
        ]
    )
    def test_mx_inf_nan_triton_matches_pure(self, _name: str, preset_name: str) -> None:
        """qdq and qdq_triton_mega agree bit-for-bit on Inf/NaN/finite.

        Combined with the focused tests that pin pure-PyTorch qdq's values, this
        transitively pins the Triton path for the Inf/NaN edge cases (and keeps
        it consistent on finite).
        """
        config = QuantConfig.from_dict(FAKE_PRESETS[preset_name])
        rows = [
            [float("inf")] * 16,  # pure +Inf block
            [float("-inf")] * 16,  # pure -Inf block
            [float("nan")] * 16,  # pure NaN block
            [float("nan")]
            + [1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 4.0] * 2
            + [6.0],  # mixed NaN
            [float("inf")] + [3.0] * 14 + [float("-inf")],  # mixed Inf, no NaN
            [float(i % 7 - 3) for i in range(16)],  # all finite
        ]
        tensor = torch.tensor(rows, dtype=torch.bfloat16, device=self.device)

        ref = qdq(tensor, "test", config, return_quant_info=True)
        mega = qdq_triton_mega(tensor, "test", config, return_quant_info=True)

        self._assert_qdq_paths_equal(ref, mega, f"{preset_name}: qdq_triton_mega")

    @parameterized.expand(
        [
            ("int3", "int3"),
            ("int4", "int4"),
            ("int8", "int8"),
        ]
    )
    def test_int_triton_matches_pure(self, _name: str, elem: str) -> None:
        """qdq and qdq_triton_mega agree bit-for-bit for integer dtypes.

        Covers int3, int4 and int8. The int branch reuses the eager clamp bounds (min_pos/
        max_pos) and round-half-to-even, so both paths must match exactly.
        """
        base = dict(FAKE_PRESETS["mxint8"])
        base["elem_dtype"] = elem
        config = QuantConfig.from_dict(base)
        torch.manual_seed(0)
        tensor = torch.randn(256, 128, dtype=torch.bfloat16, device=self.device)

        ref = qdq(tensor, "test", config, return_quant_info=True)
        mega = qdq_triton_mega(tensor, "test", config, return_quant_info=True)

        self._assert_qdq_paths_equal(ref, mega, f"{elem}: qdq_triton_mega")

    @parameterized.expand(
        [
            # (name, use_triton, elem_dtype, scale_dtype, scale_norm_mode, scale_round_mode, global_scale_granularity)
            (
                "MX_E2M1_triton",
                True,
                ElemDtype.E2M1,
                ScaleDtype.E8M0,
                ScaleNormMode.POW2MAX,
                ScaleRoundMode.OAS,
                GlobalScaleGranularity.NONE,
            ),
            (
                "MX_E2M1_non_triton",
                False,
                ElemDtype.E2M1,
                ScaleDtype.E8M0,
                ScaleNormMode.POW2MAX,
                ScaleRoundMode.OAS,
                GlobalScaleGranularity.NONE,
            ),
            (
                "NV_E2M1_triton",
                True,
                ElemDtype.E2M1,
                ScaleDtype.FP8,
                ScaleNormMode.AMAX,
                ScaleRoundMode.NONE,
                GlobalScaleGranularity.TENSORWISE,
            ),
            (
                "NV_E2M1_non_triton",
                False,
                ElemDtype.E2M1,
                ScaleDtype.FP8,
                ScaleNormMode.AMAX,
                ScaleRoundMode.NONE,
                GlobalScaleGranularity.TENSORWISE,
            ),
        ]
    )
    def test_zero_block_handling(
        self,
        _name: str,
        use_triton: bool,
        elem_dtype: ElemDtype,
        scale_dtype: ScaleDtype,
        scale_norm_mode: ScaleNormMode,
        scale_round_mode: ScaleRoundMode,
        global_scale_granularity: GlobalScaleGranularity,
    ) -> None:
        """All zeros in a block should not cause errors."""
        tensor = torch.zeros(1, 32, dtype=torch.bfloat16, device=self.device)
        config = self._create_config(
            elem_dtype=elem_dtype,
            scale_dtype=scale_dtype,
            scale_norm_mode=scale_norm_mode,
            scale_round_mode=scale_round_mode,
            global_scale_granularity=global_scale_granularity,
        )

        result = fake_quantize(
            tensor, "test", config, transpose=False, use_triton=use_triton
        )

        torch.testing.assert_close(
            result,
            tensor,
            msg="All-zero block should remain all zeros",
        )


@skip_if_no_cuda
class TestOASSRounding(unittest.TestCase):
    """Test OAS_S (Overflow-Aware Scaling with Skipping all-ones) scale rounding for E4M3.

    OAS_S affects E8M0 scale rounding to handle formats where the all-ones
    mantissa (111) is reserved for NaN. Theoretically, it performs OAS considering
    the actual largest representable value.

    E4M3 representation near max:
    - 256: exp=1111, mantissa=000 (1.000 x 2^8)
    - 448: exp=1111, mantissa=110 (1.110 x 2^8) - max normal
    - 480: exp=1111, mantissa=111 (1.111 x 2^8) - NaN (reserved)

    OAS would scale down if amax is larger than 496 while
    OAS_S would scale down if amax is larger than 464.
    """

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.device = "cuda"

    def _create_e4m3_config(self, scale_round_mode: str) -> QuantConfig:
        return make_quant_config(
            scale_dtype=ScaleDtype.E8M0,
            scale_norm_mode=ScaleNormMode.POW2MAX,
            scale_round_mode=ScaleRoundMode(scale_round_mode),
            global_scale_granularity=GlobalScaleGranularity.NONE,
            elem_dtype=ElemDtype.E4M3,
            elem_round_mode=ElemRoundMode.NEAREST,
            block_size=BlockSize(1, 16),
        )

    def test_oas_vs_oas_s_scale_difference(self) -> None:
        """Test OAS_S against OAS on a block with amax 490.

        When amax = 490:
        - OAS_S scale down 490 -> 245, and the value will be quantized to 240.
        - Thus, dequantized value with OAS_S becomes 480.
        - OAS would not scale down, and just clip it to 448, so dequantized value
        - also becomes 448. This also affects the small values in the block, such as 0.001.
        - 0.001 will be flushed to zero with OAS_S while it will be mapped to 0.001953 with OAS.
        """

        # In general, OAS_S has smaller quantization error for the large values,
        # but it has higher flush to zero ratio compared to OAS.
        tensor = torch.tensor(
            [
                [
                    # amax = 490, which is > 448 (max normal in E4M3)
                    490.0,
                    470.0,
                    448.0,
                    256.0,
                    128.0,
                    64.0,
                    32.0,
                    16.0,
                    8.0,
                    4.0,
                    2.0,
                    1.0,
                    0.5,
                    0.002,  # 0.001 is smaller than the smallest magnitude in fp8_e4m3
                    0.001,
                    -470.0,
                    -490.0,
                    # Dummy values for block size of 32
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ],
            dtype=torch.bfloat16,
            device=self.device,
        )

        config_oas_s = self._create_e4m3_config("oas_s")

        result = fake_quantize(
            tensor,
            "test",
            config_oas_s,
            transpose=False,
            use_triton=True,
        )

        # Check 490 after quant-dequant
        self.assertEqual(
            result[0, 0].item(),
            480.0,
            "490.0 should be mapped to 480",
        )

        # Check 470 after quant-dequant
        self.assertEqual(
            result[0, 1].item(),
            480.0,
            "470.0 should be mapped to 480",
        )

        # Check 0.002 after quant-dequant
        self.assertGreater(
            result[0, 13].item(),
            0.0,
            "0.002 should not be flushed to zero",
        )
        # Check 0.001 after quant-dequant
        self.assertEqual(
            result[0, 14].item(),
            0.0,
            "0.001 should be flushed to zero",
        )

        config_oas = self._create_e4m3_config("oas")
        result_oas = fake_quantize(
            tensor,
            "test",
            config_oas,
            transpose=False,
            use_triton=False,
        )
        self.assertEqual(
            result_oas[0, 0].item(),
            448.0,
            "490.0 should be mapped to 448",
        )
        self.assertEqual(
            result_oas[0, 1].item(),
            448.0,
            "470.0 should be mapped to 448",
        )
        self.assertGreater(
            result_oas[0, 13].item(),
            0.0,
            "0.002 should not be flushed to zero.",
        )
        self.assertGreater(
            result_oas[0, 14].item(),
            0.0,
            "0.001 should not be flushed to zero.",
        )


@skip_if_no_cuda
class TestSubnormalBottomOfRangeCpu(unittest.TestCase):
    """CPU / pure-pytorch coverage for the POW2MAX bottom of range.

    A block whose amax is small enough that `amax / max_pow2` is fp32-subnormal
    must land on E8M0 scale byte 0 (the floor), not the 2^0 (byte 127) fallback.
    These exercise only the pytorch `qdq` path
    (`use_triton=False`), so they validate the subnormal scale independently
    of the Triton kernel. The cross-backend (Triton) parity for the same
    patterns lives in `TestSubnormalBottomOfRange`.
    """

    # Deep-subnormal amax: ideal scale <= 2^-127 -> E8M0 floor (byte 0).
    _BOTTOM_PATTERNS = [
        0x0000,  # +0
        0x8000,  # -0
        0x0080,  # 2^-126 (fp32 min normal); amax/4 subnormal
        0x0100,  # 2^-125
        0x0141,  # 1.5078 * 2^-125 (the headline case)
    ]
    # Normal blocks whose scale must NOT be floored to byte 0.
    _NORMAL_PATTERNS = [
        0x0300,  # 2^-121 -> byte 4
        0x4000,  # 2.0
    ]

    @staticmethod
    def _block(bits: int, cols: int) -> torch.Tensor:
        # Map the 16-bit pattern into signed int16 range (0x8000 -> -32768).
        signed = bits - 0x10000 if bits >= 0x8000 else bits
        v = (
            torch.tensor([signed], dtype=torch.int16)
            .view(torch.bfloat16)
            .to(torch.float32)
            .item()
        )
        return torch.full((1, cols), v, dtype=torch.bfloat16, device="cuda")

    @staticmethod
    def _cfg(
        granularity: GlobalScaleGranularity = GlobalScaleGranularity.NONE,
        global_dtype: ScaleDtype = ScaleDtype.NONE,
    ) -> QuantConfig:
        return make_quant_config(
            scale_dtype=ScaleDtype.E8M0,
            scale_norm_mode=ScaleNormMode.POW2MAX,
            scale_round_mode=ScaleRoundMode.OAS,
            global_scale_granularity=granularity,
            global_scale_dtype=global_dtype,
            elem_dtype=ElemDtype.E2M1,
            elem_round_mode=ElemRoundMode.NEAREST,
            block_size=BlockSize(1, 32),
        )

    def test_cpu_bottom_of_range_hits_e8m0_floor(self) -> None:
        cfg = self._cfg()
        for bits in self._BOTTOM_PATTERNS:
            x = self._block(bits, 32)
            qi = fake_quantize(x, "x", cfg, use_triton=False, return_quant_info=True)
            self.assertEqual(
                int(e8m0_scale_byte(qi.local_unscale).reshape(-1)[0].item()),
                0,
                f"expected E8M0 byte 0 at bits={bits:#06x}",
            )
            self.assertTrue(torch.isfinite(qi.quantized_val).all().item())

    def test_cpu_normal_block_not_floored(self) -> None:
        cfg = self._cfg()
        for bits in self._NORMAL_PATTERNS:
            x = self._block(bits, 32)
            qi = fake_quantize(x, "x", cfg, use_triton=False, return_quant_info=True)
            self.assertNotEqual(
                int(e8m0_scale_byte(qi.local_unscale).reshape(-1)[0].item()),
                0,
                f"normal block wrongly floored at bits={bits:#06x}",
            )


if __name__ == "__main__":
    unittest.main()
