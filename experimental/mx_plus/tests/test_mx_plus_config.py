# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``MxPlus`` as a config extension: options, round-trips, validation and the
amax-element rounding. CPU only."""

import dataclasses
import pickle
import unittest

import torch
from mx_plus.config import MxPlus
from mx_plus.mx_plus import _mx_plus_amax_lp, MAX_EXTRA_MANTISSA_BITS
from mx_plus.presets import MX_PLUS_PRESETS
from quant_suite_core import (
    BlockSize,
    ElemDtype,
    ElemRoundMode,
    fake_quantize,
    GlobalScaleGranularity,
    QuantConfig,
    QuantMode,
    ScaleDtype,
    ScaleNormMode,
    ScaleRoundMode,
)
from quant_suite_core.quant_config import DEFAULT_QUANT_CONFIG_BUILDER, MBSBlockSize

_BUILDER = DEFAULT_QUANT_CONFIG_BUILDER.extend((MxPlus,), MX_PLUS_PRESETS)

_BASE: dict[str, object] = {
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


def _spec(**options: object) -> dict[str, object]:
    """``mxfp4_plus2``'s spec with the given ``mx_plus`` options."""
    return {**MX_PLUS_PRESETS["mxfp4_plus2"], "extensions": {"mx_plus": options}}


class TestDeclaration(unittest.TestCase):
    def test_declaration(self) -> None:
        self.assertEqual(MxPlus.name, "mx_plus")
        self.assertTrue(dataclasses.is_dataclass(MxPlus))
        self.assertEqual(MxPlus().extra_mantissa_bits, 2)
        self.assertRaises(
            dataclasses.FrozenInstanceError,
            setattr,
            MxPlus(),
            "extra_mantissa_bits",
            3,
        )

    def test_public_builder_does_not_know_the_extension(self) -> None:
        with self.assertRaisesRegex(ValueError, "Installed extensions: \\[\\]"):
            DEFAULT_QUANT_CONFIG_BUILDER.build(MX_PLUS_PRESETS["mxfp4_plus2"])

    def test_survives_pickle(self) -> None:
        cfg = _BUILDER.build_preset("mxfp4_plus2")
        self.assertEqual(pickle.loads(pickle.dumps(cfg)), cfg)


class TestSerialization(unittest.TestCase):
    def test_preset(self) -> None:
        cfg = _BUILDER.build_preset("mxfp4_plus2")
        self.assertIs(type(cfg), QuantConfig)
        self.assertEqual(cfg.extensions, (MxPlus(extra_mantissa_bits=2),))
        self.assertTrue(cfg.has_extension("mx_plus"))
        self.assertEqual(cfg.global_scale_granularity, GlobalScaleGranularity.NONE)
        self.assertEqual(cfg.global_scale_dtype, ScaleDtype.NONE)
        self.assertEqual(cfg.get_mbs_block_size(), 0)

    def test_dict_round_trip(self) -> None:
        cfg = _BUILDER.build_preset("mxfp4_plus2")
        d = cfg.to_dict()
        self.assertEqual(d["extensions"], {"mx_plus": {"extra_mantissa_bits": 2}})
        self.assertEqual(d["global_scale_granularity"], "none")
        self.assertEqual(d["global_scale_dtype"], "none")
        self.assertEqual(_BUILDER.build(d), cfg)

    def test_every_supported_bit_count_parses(self) -> None:
        for bits in range(1, MAX_EXTRA_MANTISSA_BITS + 1):
            with self.subTest(bits=bits):
                cfg = _BUILDER.build(_spec(extra_mantissa_bits=bits))
                self.assertEqual(cfg.extensions, (MxPlus(extra_mantissa_bits=bits),))
                self.assertEqual(_BUILDER.build(cfg.to_dict()), cfg)

    def test_options_default(self) -> None:
        self.assertEqual(_BUILDER.build(_spec()), _BUILDER.build_preset("mxfp4_plus2"))

    def test_bit_count_is_part_of_the_identity(self) -> None:
        self.assertNotEqual(
            _BUILDER.build(_spec(extra_mantissa_bits=1)),
            _BUILDER.build(_spec(extra_mantissa_bits=2)),
        )

    def test_rejects_unknown_and_mistyped_options(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown options"):
            _BUILDER.build(_spec(bits=2))
        with self.assertRaisesRegex(ValueError, "must be int"):
            _BUILDER.build(_spec(extra_mantissa_bits="2"))
        with self.assertRaisesRegex(ValueError, "must be int"):
            _BUILDER.build(_spec(extra_mantissa_bits=True))


class TestValidation(unittest.TestCase):
    def _assert_raises(
        self, message: str, extension: MxPlus | None = None, **overrides: object
    ) -> None:
        with self.assertRaisesRegex(ValueError, message):
            QuantConfig(**{**_BASE, **overrides}, extensions=(extension or MxPlus(),))

    def test_accepts_the_base(self) -> None:
        cfg = QuantConfig(**_BASE, extensions=(MxPlus(),))
        self.assertEqual(cfg.extensions, (MxPlus(),))

    def test_mbs_rejected(self) -> None:
        self._assert_raises(
            "MX\\+ requires global_scale_granularity=NONE",
            scale_norm_mode=ScaleNormMode.POW2MAX,
            global_scale_granularity=GlobalScaleGranularity.MBS,
            mbs_block_size=MBSBlockSize.MBS128,
            global_scale_dtype=ScaleDtype.E0M8,
        )

    def test_requires_e8m0(self) -> None:
        self._assert_raises(
            "MX\\+ requires E8M0 scale dtype",
            scale_dtype=ScaleDtype.FP32,
            scale_round_mode=ScaleRoundMode.NONE,
        )

    def test_rejects_bit_counts_out_of_range(self) -> None:
        for bits in (0, -1, MAX_EXTRA_MANTISSA_BITS + 1):
            with self.subTest(bits=bits):
                self._assert_raises(
                    "extra_mantissa_bits to be an int from 1",
                    MxPlus(extra_mantissa_bits=bits),
                )


class TestAmaxRounding(unittest.TestCase):
    """The amax element rounds to nearest-even, as ``fp32_to_fpx`` does."""

    def test_one_ulp_below_a_tie_rounds_down(self) -> None:
        # E2M1 with e0m2 rounds at 3 mantissa bits, and the private exponent
        # clips to 0 below 1.0, so with scale 1 the amax element is multiplied by
        # 8 before rounding: x / 8 lands 1 ULP below the 0/1 tie at 0.5. The
        # block scale is passed directly because a block's own E8M0 scale keeps
        # its amax far above this range.
        below_tie = torch.nextafter(torch.tensor(0.5), torch.tensor(0.0))
        self.assertEqual(below_tie.item(), 0.5 - 2**-25)
        # In float32 the rounding step's + 0.5 gives 1.0, not 1 - 2**-25.
        self.assertEqual((below_tie + 0.5).item(), 1.0)

        blocks = torch.zeros(2, 32)
        blocks[0, 3] = below_tie / 8
        blocks[1, 5] = -below_tie / 8
        cfg = _BUILDER.build_preset("mxfp4_plus2")
        index, amax_lp = _mx_plus_amax_lp(blocks, torch.ones(2), cfg, 2, -6.0, 6.0)

        self.assertEqual(index.tolist(), [3, 5])
        self.assertEqual(amax_lp.tolist(), [0.0, 0.0])
        self.assertEqual(torch.signbit(amax_lp).tolist(), [False, True])

    def test_negative_zero_block_keeps_its_sign(self) -> None:
        x = torch.full((2, 32), -0.0)
        self.assertTrue(bool(torch.signbit(x).all()))
        cfg = _BUILDER.build_preset("mxfp4_plus2")

        out = fake_quantize(x, "w", cfg, use_triton=False)

        self.assertTrue(bool((out == 0).all()))
        self.assertTrue(bool(torch.signbit(out).all()))


if __name__ == "__main__":
    unittest.main()
