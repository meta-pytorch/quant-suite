# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``MxPlus`` reaches both QDQ backends through ``MxPlus.post_process``.

``fake_quantize(use_triton=False)`` runs the eager backend and
``use_triton=True`` the fused one. Needs a GPU; skipped without one.
"""

import unittest

import torch
from mx_plus.config import MxPlus
from mx_plus.presets import MX_PLUS_PRESETS
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
from quant_suite_core.quant_config import DEFAULT_QUANT_CONFIG_BUILDER

_BUILDER = DEFAULT_QUANT_CONFIG_BUILDER.extend((MxPlus,), MX_PLUS_PRESETS)

_BACKENDS: tuple[tuple[str, bool], ...] = (("eager", False), ("triton", True))


def _config(**overrides: object) -> QuantConfig:
    fields: dict[str, object] = {
        "quant_mode": QuantMode.SYMM,
        "scale_dtype": ScaleDtype.E8M0,
        "scale_norm_mode": ScaleNormMode.AMAX,
        "scale_round_mode": ScaleRoundMode.OAS,
        "global_scale_granularity": GlobalScaleGranularity.NONE,
        "elem_dtype": ElemDtype.E2M1,
        "elem_round_mode": ElemRoundMode.NEAREST,
        "block_size": BlockSize(1, 32),
        "global_scale_dtype": ScaleDtype.NONE,
    }
    fields.update(overrides)
    return QuantConfig(**fields)


def _mx_plus_config(**overrides: object) -> QuantConfig:
    return _config(extensions=(MxPlus(extra_mantissa_bits=4),), **overrides)


def _e8m0_scale_byte(local_unscale: torch.Tensor) -> torch.Tensor:
    """The E8M0 exponent byte of a power-of-two fp32 unscale."""
    bits = local_unscale.to(torch.float32).contiguous().view(torch.int32)
    return ((bits & 0x7FFFFFFF) >> 23) & 0xFF


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestMxPlus(unittest.TestCase):
    """What the rule must do: run on both backends, change the output, keep
    the scale, and survive a subnormal block.
    """

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.device = "cuda"

    def test_both_backends_with_mx_plus(self) -> None:
        """MX+ with 4 extra bits preserves extra mantissa bits for amax element."""
        x = torch.randn(32, 64, device=self.device)
        for name, use_triton in _BACKENDS:
            with self.subTest(backend=name):
                result = fake_quantize(
                    x, "test_tensor", _mx_plus_config(), use_triton=use_triton
                )
                self.assertEqual(result.shape, x.shape)
                self.assertFalse(torch.isnan(result).any().item())

    def test_mx_plus_improves_error(self) -> None:
        """The rule has to do something: extra amax mantissa bits reduce error."""
        x = torch.randn(64, 256, device=self.device)
        plain = DEFAULT_QUANT_CONFIG_BUILDER.build(FAKE_PRESETS["mxfp4_ocp"])
        plus = _BUILDER.build_preset("mxfp4_plus2")
        for name, use_triton in _BACKENDS:
            with self.subTest(backend=name):
                err_plain = (
                    (fake_quantize(x, "x", plain, use_triton=use_triton).float() - x)
                    .abs()
                    .sum()
                    .item()
                )
                err_plus = (
                    (fake_quantize(x, "x", plus, use_triton=use_triton).float() - x)
                    .abs()
                    .sum()
                    .item()
                )
                self.assertLess(err_plus, err_plain)

    def test_mx_plus_subnormal_preserves_amax(self) -> None:
        # MX+ recomputes the block scale to preserve the amax element; on a
        # subnormal block it must use the same round-then-shift scale as the
        # mega kernel, otherwise the dequantized amax is corrupted.
        config = _mx_plus_config(scale_norm_mode=ScaleNormMode.POW2MAX)
        for bits in [0x0141, 0x0100, 0x4000]:
            signed = bits - 0x10000 if bits >= 0x8000 else bits
            value = (
                torch.tensor([signed], dtype=torch.int16)
                .view(torch.bfloat16)
                .to(torch.float32)
                .item()
            )
            x = torch.full((1, 32), value, dtype=torch.bfloat16, device=self.device)
            out = fake_quantize(x, "mx_plus", config, use_triton=False)
            self.assertEqual(out.shape, x.shape)
            self.assertTrue(torch.isfinite(out).all().item())
            torch.testing.assert_close(
                out.to(torch.float32),
                x.to(torch.float32),
                rtol=0.5,
                atol=0.0,
                msg=f"MX+ dequant far from input at bits={bits:#06x}",
            )

    def test_post_process_is_a_no_op_without_mx_plus(self) -> None:
        """Knowing the extension changes nothing for a config that does not enable it."""
        x = torch.randn(32, 64, device=self.device)
        preset = FAKE_PRESETS["mxfp4_ocp"]
        for name, use_triton in _BACKENDS:
            with self.subTest(backend=name):
                public = fake_quantize(
                    x, "x", QuantConfig.from_dict(preset), use_triton=use_triton
                )
                known = fake_quantize(
                    x, "x", _BUILDER.build(preset), use_triton=use_triton
                )
                self.assertTrue(torch.equal(public, known))

    def test_e8m0_scale_byte_is_unchanged_by_mx_plus(self) -> None:
        """MX+ rewrites elements, never the block scale it reports."""
        x = torch.randn(32, 64, device=self.device)
        qi_plain = fake_quantize(
            x, "x", _config(), use_triton=False, return_quant_info=True
        )
        qi_plus = fake_quantize(
            x, "x", _mx_plus_config(), use_triton=False, return_quant_info=True
        )
        self.assertIsInstance(qi_plain, QuantInfo)
        self.assertIsInstance(qi_plus, QuantInfo)
        assert isinstance(qi_plain, QuantInfo) and isinstance(qi_plus, QuantInfo)
        torch.testing.assert_close(
            _e8m0_scale_byte(qi_plus.local_unscale),
            _e8m0_scale_byte(qi_plain.local_unscale),
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
