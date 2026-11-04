# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest import mock

import torch
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
from quant_suite_core.fake_quant import (
    is_grouped_nvfp4_triton_supported,
    is_triton_mega_supported,
)
from quant_suite_core.qdq_pytorch import (
    _mbs_factor_dynamic,
    _mbs_factor_static,
    _mbs_headroom,
    _mbs_quant_block_peaks,
)
from quant_suite_core.qdq_triton import qdq_triton_mega
from quant_suite_core.quant_config import (
    DEFAULT_QUANT_CONFIG_BUILDER,
    elem_dtype_max_pow2,
    ELEM_DTYPE_MAX_POW2_MAP,
    ELEM_DTYPE_RANGE_MAP,
    F32_MAX,
    MBSBlockSize,
)


# Every built-in granularity token, hardcoded on purpose: these are frozen
# config identities.
_GRANULARITY_TOKENS: list[str] = [
    "none",
    "tensorwise",
    "rowwise",
    "mbs16",
    "mbs32",
    "mbs64",
    "mbs128",
    "mbs256",
    "mbs384",
    "mbs512",
]


def _spec_with_granularity(token: str) -> dict[str, str]:
    """A spec that ``__post_init__`` accepts for the given granularity token."""
    if token in ("tensorwise", "rowwise"):
        return {**FAKE_PRESETS["nvfp4"], "global_scale_granularity": token}
    return {
        **FAKE_PRESETS["mxfp4"],
        "global_scale_granularity": token,
        "global_scale_dtype": "none" if token == "none" else "e0m8",
    }


def _mbs_config(size: MBSBlockSize) -> QuantConfig:
    return QuantConfig.from_dict(_spec_with_granularity(f"mbs{size.value}"))


class TestGroupedNVFP4Support(unittest.TestCase):
    @staticmethod
    def _config(**overrides: str) -> QuantConfig:
        config = {**FAKE_PRESETS["nvfp4"], **overrides}
        return QuantConfig.from_dict(config)

    def test_supported_configs(self) -> None:
        supported = {
            "nvfp4": self._config(),
            "stochastic_rounding": self._config(elem_round_mode="stochastic"),
            "block_1x32": self._config(block_size="1x32"),
            "fp8_scale_floor": self._config(scale_round_mode="floor"),
        }

        for name, config in supported.items():
            with self.subTest(name=name):
                self.assertTrue(is_grouped_nvfp4_triton_supported(config))

    def test_unsupported_configs(self) -> None:
        unsupported = {
            "asymmetric": self._config(quant_mode="asymm"),
            "non_fp8_scale": self._config(
                scale_dtype="fp16",
                global_scale_granularity="none",
                global_scale_dtype="none",
            ),
            "rowwise_global_scale": self._config(global_scale_granularity="rowwise"),
            "non_e2m1_element": self._config(elem_dtype="e2m3"),
            "block_16x16": self._config(block_size="16x16"),
        }

        for name, config in unsupported.items():
            with self.subTest(name=name):
                self.assertFalse(is_grouped_nvfp4_triton_supported(config))


# ******************************************************************************
# *  Config and Utility Function Tests (no GPU required)              *
# ******************************************************************************
class TestConstantsAndConfigs(unittest.TestCase):
    """Tests for config/utility functions that do NOT require GPU."""

    def test_block_size_str(self) -> None:
        bs = BlockSize(m=32, n=16)
        self.assertEqual(str(bs), "32x16")

    def test_block_size_from_str(self) -> None:
        bs = BlockSize.from_str("32x16")
        self.assertEqual(bs.m, 32)
        self.assertEqual(bs.n, 16)

    def test_block_size_roundtrip(self) -> None:
        original = BlockSize(m=1, n=64)
        roundtripped = BlockSize.from_str(str(original))
        self.assertEqual(roundtripped.m, original.m)
        self.assertEqual(roundtripped.n, original.n)

    def test_global_scale_granularity_is_mbs(self) -> None:
        self.assertTrue(GlobalScaleGranularity.MBS.is_mbs())
        self.assertFalse(GlobalScaleGranularity.NONE.is_mbs())
        self.assertFalse(GlobalScaleGranularity.TENSORWISE.is_mbs())
        self.assertFalse(GlobalScaleGranularity.ROWWISE.is_mbs())

    def test_mbs_block_size_members(self) -> None:
        # Note 384: the supported set is not the powers of two, which is why
        # this is an enum and not an int.
        self.assertEqual(
            [s.value for s in MBSBlockSize], [16, 32, 64, 128, 256, 384, 512]
        )

    def test_every_mbs_block_size_maps_to_its_token(self) -> None:
        for size in MBSBlockSize:
            cfg = _mbs_config(size)
            self.assertEqual(cfg.global_scale_granularity, GlobalScaleGranularity.MBS)
            self.assertEqual(cfg.mbs_block_size, size)
            self.assertEqual(cfg.get_mbs_block_size(), size.value)
            self.assertEqual(
                cfg.to_dict()["global_scale_granularity"], f"mbs{size.value}"
            )
            self.assertEqual(QuantConfig.from_dict(cfg.to_dict()), cfg)

    def test_unsupported_mbs_block_size_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported MBS block size"):
            QuantConfig.from_dict(_spec_with_granularity("mbs48"))

    def test_get_mbs_block_size_zero_without_mbs(self) -> None:
        plain = QuantConfig.from_dict(_spec_with_granularity("none"))
        self.assertEqual(plain.get_mbs_block_size(), 0)

    def test_from_dict_rejects_a_bare_mbs_granularity_member(self) -> None:
        # A bare member carries no block size, so the caller has to pass the
        # packed token. Without this the failure is __post_init__'s generic
        # "set exactly when" assertion, which states the invariant but not the
        # fix.
        with self.assertRaisesRegex(ValueError, "pass the packed token"):
            QuantConfig.from_dict(
                {"global_scale_granularity": GlobalScaleGranularity.MBS}
            )

    def test_get_mbs_block_size_raises_rather_than_returning_zero(self) -> None:
        # Returning 0 would satisfy the backends' `width % n == 0` assert and
        # quantize against a zero-width macro block; they guard on is_mbs()
        # alone, so this has to fail here.
        cfg = _mbs_config(MBSBlockSize.MBS128)
        cfg.mbs_block_size = None
        with self.assertRaisesRegex(ValueError, "no macro-block width"):
            cfg.get_mbs_block_size()

    def test_non_canonical_and_unknown_mbs_tokens_are_rejected(self) -> None:
        # Member-name lookup rejects all of these. "mbs0128"/"mbs+128" matter
        # because `int` would have accepted them and they re-serialize as
        # "mbs128", aliasing one config identity onto another; "mbsfoo" is a
        # plain typo and is reported against the granularity enum rather than
        # as a bad block size.
        for token in ("mbs0128", "mbs+128", "mbsfoo"):
            with self.subTest(token=token):
                with self.assertRaises(ValueError):
                    QuantConfig.from_dict(_spec_with_granularity(token))

    def test_mbs_block_size_set_iff_granularity_is_mbs(self) -> None:
        self._assert_post_init_raises(
            "mbs_block_size must be set exactly when",
            global_scale_granularity=GlobalScaleGranularity.MBS,
            global_scale_dtype=ScaleDtype.E0M8,
            mbs_block_size=None,
        )
        self._assert_post_init_raises(
            "mbs_block_size must be set exactly when",
            mbs_block_size=MBSBlockSize.MBS128,
        )

    def test_granularity_token_roundtrip_is_byte_identical(self) -> None:
        """Every granularity token parses and re-emits itself byte-for-byte.

        These 10 tokens are config identities (e.g. recipe dict keys), so the
        serialized value must not change.
        """
        for token in _GRANULARITY_TOKENS:
            with self.subTest(token=token):
                cfg = QuantConfig.from_dict(_spec_with_granularity(token))
                self.assertEqual(cfg.to_dict()["global_scale_granularity"], token)
                self.assertEqual(QuantConfig.from_dict(cfg.to_dict()), cfg)

    def test_quant_config_repr_is_its_dict_form(self) -> None:
        config = QuantConfig(
            quant_mode=QuantMode.SYMM,
            scale_dtype=ScaleDtype.E8M0,
            scale_norm_mode=ScaleNormMode.AMAX,
            scale_round_mode=ScaleRoundMode.OAS,
            global_scale_granularity=GlobalScaleGranularity.NONE,
            elem_dtype=ElemDtype.E2M1,
            elem_round_mode=ElemRoundMode.NEAREST,
            block_size=BlockSize(1, 32),
        )
        self.assertEqual(repr(config), f"QuantConfig({config.to_dict()!r})")
        self.assertEqual(str(config), repr(config))

    def test_quant_config_cannot_be_subclassed(self) -> None:
        with self.assertRaisesRegex(
            TypeError, "QuantConfig cannot be subclassed; add a QuantConfigExtension"
        ):

            class _Sub(QuantConfig):  # pyrefly: ignore[invalid-inheritance]
                pass

    def test_string_specs_are_rejected(self) -> None:
        csv = "symm,e8m0,pow2max,oas,none,e2m1,nearest,1x32,none"
        for build in (QuantConfig.from_dict, DEFAULT_QUANT_CONFIG_BUILDER.build):
            with self.subTest(build=build):
                with self.assertRaisesRegex(TypeError, "mapping of field names"):
                    # pyrefly: ignore[bad-argument-type]
                    build(csv)

    def test_from_dict_rejects_unknown_enum(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown enum value INVALID"):
            QuantConfig.from_dict({"scale_dtype": "INVALID"})

    def test_mbs_dynamic_dict_value(self) -> None:
        base = {
            **FAKE_PRESETS["mxfp4"],
            "global_scale_granularity": "mbs256",
            "global_scale_dtype": "e0m8",
        }
        for value, expected in (
            ("mbs_dynamic", True),
            ("MBS_DYNAMIC", True),
            ("none", False),
            (True, True),
            (False, False),
        ):
            with self.subTest(value=value):
                cfg = QuantConfig.from_dict({**base, "mbs_dynamic": value})
                self.assertIs(cfg.mbs_dynamic, expected)
        for value in ("true", "1", "dynamic", "false", 1, None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "field 'mbs_dynamic' must be 'mbs_dynamic', 'none'"
                ):
                    QuantConfig.from_dict({**base, "mbs_dynamic": value})

    def test_quant_config_mbs_dynamic_roundtrip(self) -> None:
        base = {
            "quant_mode": QuantMode.SYMM,
            "scale_dtype": ScaleDtype.E8M0,
            "scale_norm_mode": ScaleNormMode.POW2MAX,
            "scale_round_mode": ScaleRoundMode.OAS,
            "global_scale_granularity": GlobalScaleGranularity.MBS,
            "mbs_block_size": MBSBlockSize.MBS256,
            "global_scale_dtype": ScaleDtype.E0M8,
            "elem_dtype": ElemDtype.E2M1,
            "elem_round_mode": ElemRoundMode.NEAREST,
            "block_size": BlockSize(1, 16),
        }
        off = QuantConfig(**base)
        self.assertFalse(off.mbs_dynamic)
        self.assertIs(off.to_dict()["mbs_dynamic"], False)
        self.assertFalse(QuantConfig.from_dict(off.to_dict()).mbs_dynamic)

        on = QuantConfig(**base, mbs_dynamic=True)
        self.assertIs(on.to_dict()["mbs_dynamic"], True)
        self.assertTrue(QuantConfig.from_dict(on.to_dict()).mbs_dynamic)

    def test_mbs_dynamic_requires_mbs_granularity(self) -> None:
        # Must fail loudly: an mbs_dynamic that silently fell back to the amax
        # rule would read as "MBS does not help".
        with self.assertRaises(ValueError):
            QuantConfig(
                quant_mode=QuantMode.SYMM,
                scale_dtype=ScaleDtype.E8M0,
                scale_norm_mode=ScaleNormMode.POW2MAX,
                scale_round_mode=ScaleRoundMode.OAS,
                global_scale_granularity=GlobalScaleGranularity.NONE,
                elem_dtype=ElemDtype.E2M1,
                elem_round_mode=ElemRoundMode.NEAREST,
                block_size=BlockSize(1, 16),
                mbs_dynamic=True,
            )

    def test_quant_config_post_init_requires_elem_dtype_range(self) -> None:
        # A quant mode and element dtype without an element range fail when the
        # config is built, not inside qdq().
        ranges = ELEM_DTYPE_RANGE_MAP[QuantMode.ASYMM_MEAN_SHIFT]
        with mock.patch.dict(ranges):
            del ranges[ElemDtype.INT3]
            with self.assertRaisesRegex(ValueError, "does not support elem_dtype"):
                QuantConfig(
                    quant_mode=QuantMode.ASYMM_MEAN_SHIFT,
                    scale_dtype=ScaleDtype.FP16,
                    scale_norm_mode=ScaleNormMode.AMAX,
                    scale_round_mode=ScaleRoundMode.NONE,
                    global_scale_granularity=GlobalScaleGranularity.NONE,
                    elem_dtype=ElemDtype.INT3,
                    elem_round_mode=ElemRoundMode.NEAREST,
                    block_size=BlockSize(1, 64),
                )

    def test_quant_config_post_init_e8m0_requires_round_mode(self) -> None:
        with self.assertRaises(ValueError):
            QuantConfig(
                quant_mode=QuantMode.SYMM,
                scale_dtype=ScaleDtype.E8M0,
                scale_norm_mode=ScaleNormMode.AMAX,
                scale_round_mode=ScaleRoundMode.NONE,
                global_scale_granularity=GlobalScaleGranularity.NONE,
                elem_dtype=ElemDtype.E2M1,
                elem_round_mode=ElemRoundMode.NEAREST,
                block_size=BlockSize(1, 32),
            )

    def test_quant_config_post_init_fp32_requires_none_round(self) -> None:
        with self.assertRaises(ValueError):
            QuantConfig(
                quant_mode=QuantMode.SYMM,
                scale_dtype=ScaleDtype.FP32,
                scale_norm_mode=ScaleNormMode.AMAX,
                scale_round_mode=ScaleRoundMode.OAS,
                global_scale_granularity=GlobalScaleGranularity.NONE,
                elem_dtype=ElemDtype.E2M1,
                elem_round_mode=ElemRoundMode.NEAREST,
                block_size=BlockSize(1, 32),
            )

    # -----------------------------------------------------------------
    # __post_init__ validation branches. Each case starts from a
    # valid baseline and applies exactly one offending override.
    # -----------------------------------------------------------------
    @staticmethod
    def _valid_post_init_kwargs() -> dict[str, object]:
        """A fully valid QuantConfig kwargs dict (E8M0/POW2MAX/OAS/E2M1)."""
        return {
            "quant_mode": QuantMode.SYMM,
            "scale_dtype": ScaleDtype.E8M0,
            "scale_norm_mode": ScaleNormMode.POW2MAX,
            "scale_round_mode": ScaleRoundMode.OAS,
            "global_scale_granularity": GlobalScaleGranularity.NONE,
            "elem_dtype": ElemDtype.E2M1,
            "elem_round_mode": ElemRoundMode.NEAREST,
            "block_size": BlockSize(1, 32),
            "global_scale_dtype": ScaleDtype.NONE,
        }

    def _assert_post_init_raises(self, regex: str, **overrides: object) -> None:
        kwargs = self._valid_post_init_kwargs()
        kwargs.update(overrides)
        with self.assertRaisesRegex(ValueError, regex):
            QuantConfig(**kwargs)

    def test_post_init_baseline_is_valid(self) -> None:
        # Sanity: the baseline used by the negative cases must itself be valid.
        QuantConfig(**self._valid_post_init_kwargs())

    def test_post_init_invalid_scale_dtype(self) -> None:
        self._assert_post_init_raises(
            "scale_dtype must be one of FP32, FP16, FP8, E8M0",
            scale_dtype=ScaleDtype.NONE,
        )

    def test_post_init_invalid_global_scale_dtype(self) -> None:
        self._assert_post_init_raises(
            "global_scale_dtype must be one of NONE, FP32",
            global_scale_dtype=ScaleDtype.FP16,
        )

    def test_post_init_non_mbs_global_scale_dtype_is_none_or_fp32(self) -> None:
        for granularity in (
            GlobalScaleGranularity.TENSORWISE,
            GlobalScaleGranularity.ROWWISE,
        ):
            with self.subTest(granularity=granularity):
                self._assert_post_init_raises(
                    "global_scale_dtype must be one of NONE, FP32 with "
                    f"global_scale_granularity={granularity.value}",
                    scale_dtype=ScaleDtype.FP8,
                    scale_norm_mode=ScaleNormMode.AMAX,
                    scale_round_mode=ScaleRoundMode.NONE,
                    global_scale_granularity=granularity,
                    block_size=BlockSize(1, 16),
                    global_scale_dtype=ScaleDtype.E0M8,
                )

    def test_post_init_block_size_must_be_positive(self) -> None:
        for block_size in (
            BlockSize(0, 32),
            BlockSize(1, 0),
            BlockSize(-1, 32),
            BlockSize(1, -16),
        ):
            with self.subTest(block_size=str(block_size)):
                self._assert_post_init_raises(
                    "block_size must have m >= 1 and n >= 1",
                    block_size=block_size,
                )
        with self.assertRaisesRegex(ValueError, "block_size must have m >= 1"):
            QuantConfig.from_dict({**FAKE_PRESETS["mxfp4"], "block_size": "0x32"})

    def test_post_init_mbs_width_must_be_a_multiple_of_block_n(self) -> None:
        self._assert_post_init_raises(
            r"the MBS block size \(16\) must be a multiple of block_size.n \(32\)",
            global_scale_granularity=GlobalScaleGranularity.MBS,
            mbs_block_size=MBSBlockSize.MBS16,
            global_scale_dtype=ScaleDtype.E0M8,
            block_size=BlockSize(1, 32),
        )
        self._assert_post_init_raises(
            r"the MBS block size \(384\) must be a multiple of block_size.n \(256\)",
            global_scale_granularity=GlobalScaleGranularity.MBS,
            mbs_block_size=MBSBlockSize.MBS384,
            global_scale_dtype=ScaleDtype.FP32,
            block_size=BlockSize(1, 256),
        )
        # A width that is a multiple of n is accepted.
        kwargs = self._valid_post_init_kwargs()
        kwargs.update(
            global_scale_granularity=GlobalScaleGranularity.MBS,
            mbs_block_size=MBSBlockSize.MBS384,
            global_scale_dtype=ScaleDtype.E0M8,
            block_size=BlockSize(1, 128),
        )
        self.assertEqual(QuantConfig(**kwargs).get_mbs_block_size(), 384)

    def test_post_init_mbs_global_scale_dtype_is_e0m8_or_fp32(self) -> None:
        for dtype in ScaleDtype:
            if dtype in (ScaleDtype.E0M8, ScaleDtype.FP32):
                continue
            with self.subTest(global_scale_dtype=dtype):
                self._assert_post_init_raises(
                    "global_scale_dtype must be one of E0M8, FP32 with "
                    "global_scale_granularity=mbs",
                    global_scale_granularity=GlobalScaleGranularity.MBS,
                    mbs_block_size=MBSBlockSize.MBS32,
                    global_scale_dtype=dtype,
                )

    def test_post_init_fp8_scale_round_mode(self) -> None:
        self._assert_post_init_raises(
            "for fp8 scale, scale round mode should be none, floor, or ceil",
            scale_dtype=ScaleDtype.FP8,
            scale_norm_mode=ScaleNormMode.AMAX,
            scale_round_mode=ScaleRoundMode.OAS,
            global_scale_granularity=GlobalScaleGranularity.TENSORWISE,
            global_scale_dtype=ScaleDtype.FP32,
        )

    def test_post_init_fp16_scale_norm_mode(self) -> None:
        self._assert_post_init_raises(
            "they only support amax",
            scale_dtype=ScaleDtype.FP16,
            scale_norm_mode=ScaleNormMode.POW2MAX,
            scale_round_mode=ScaleRoundMode.NONE,
        )

    def test_post_init_pow2max_requires_e8m0(self) -> None:
        self._assert_post_init_raises(
            "POW2MAX requires E8M0 scale_dtype",
            scale_dtype=ScaleDtype.FP32,
            scale_norm_mode=ScaleNormMode.POW2MAX,
            scale_round_mode=ScaleRoundMode.NONE,
        )

    def test_post_init_int_stochastic_rejected(self) -> None:
        # Every integer elem_dtype rejects stochastic element rounding.
        for elem in (ElemDtype.INT3, ElemDtype.INT4, ElemDtype.INT8):
            self._assert_post_init_raises(
                "int3/4/8 quantization does not support",
                elem_dtype=elem,
                elem_round_mode=ElemRoundMode.STOCHASTIC,
            )

    def test_post_init_fp8_without_global_scale_round_trips(self) -> None:
        kwargs = self._valid_post_init_kwargs()
        kwargs.update(
            scale_dtype=ScaleDtype.FP8,
            scale_norm_mode=ScaleNormMode.AMAX,
            scale_round_mode=ScaleRoundMode.NONE,
            global_scale_granularity=GlobalScaleGranularity.NONE,
            global_scale_dtype=ScaleDtype.NONE,
        )
        config = QuantConfig(**kwargs)
        self.assertEqual(config.to_dict()["global_scale_granularity"], "none")
        self.assertEqual(QuantConfig.from_dict(config.to_dict()), config)

    def test_post_init_non_fp8_tensorwise_rejected(self) -> None:
        self._assert_post_init_raises(
            "only FP8 supports tensorwise/rowwise",
            global_scale_granularity=GlobalScaleGranularity.TENSORWISE,
            global_scale_dtype=ScaleDtype.FP32,
        )

    def test_post_init_mbs_requires_block_m_1(self) -> None:
        self._assert_post_init_raises(
            "MBS requires block_size.m == 1",
            global_scale_granularity=GlobalScaleGranularity.MBS,
            mbs_block_size=MBSBlockSize.MBS32,
            global_scale_dtype=ScaleDtype.FP32,
            block_size=BlockSize(2, 32),
        )

    def test_post_init_granularity_none_requires_dtype_none(self) -> None:
        # global_scale_dtype non-NONE while granularity is NONE.
        self._assert_post_init_raises(
            "global_scale_dtype must be NONE when global_scale_granularity is NONE",
            global_scale_granularity=GlobalScaleGranularity.NONE,
            global_scale_dtype=ScaleDtype.FP32,
        )

    def test_post_init_dtype_none_requires_granularity_none(self) -> None:
        # granularity non-NONE while global_scale_dtype is NONE.
        self._assert_post_init_raises(
            "global_scale_granularity must be NONE when global_scale_dtype is NONE",
            scale_dtype=ScaleDtype.FP8,
            scale_norm_mode=ScaleNormMode.AMAX,
            scale_round_mode=ScaleRoundMode.NONE,
            global_scale_granularity=GlobalScaleGranularity.TENSORWISE,
            global_scale_dtype=ScaleDtype.NONE,
        )

    # -----------------------------------------------------------------
    # ScaleDtype.to_torch_dtype error path
    # -----------------------------------------------------------------
    def test_scale_dtype_to_torch_dtype_error(self) -> None:
        for sd in (
            ScaleDtype.E8M0,
            ScaleDtype.FP8,
            ScaleDtype.NONE,
            ScaleDtype.E0M8,
        ):
            with self.subTest(scale_dtype=sd):
                with self.assertRaisesRegex(ValueError, "No torch dtype mapping"):
                    sd.to_torch_dtype()

    # -----------------------------------------------------------------
    # MXFP8 OAS warning: E4M3 + E8M0 + OAS logs a "use OAS_S" warning.
    # -----------------------------------------------------------------
    def test_mxfp8_oas_warning_logged(self) -> None:
        with self.assertLogs("quant_suite_core.quant_config", level="WARNING") as cm:
            QuantConfig(
                quant_mode=QuantMode.SYMM,
                scale_dtype=ScaleDtype.E8M0,
                scale_norm_mode=ScaleNormMode.POW2MAX,
                scale_round_mode=ScaleRoundMode.OAS,
                global_scale_granularity=GlobalScaleGranularity.NONE,
                elem_dtype=ElemDtype.E4M3,
                elem_round_mode=ElemRoundMode.NEAREST,
                block_size=BlockSize(1, 32),
            )
        self.assertTrue(any("OAS_S" in msg for msg in cm.output))


class QuantConfigDictTest(unittest.TestCase):
    """Tests for dict-based QuantConfig API (from_dict, to_dict, round-trip)."""

    def test_from_dict_all_defaults(self) -> None:
        config = QuantConfig.from_dict({})
        self.assertEqual(config.quant_mode, QuantMode.SYMM)
        self.assertEqual(config.scale_dtype, ScaleDtype.E8M0)
        self.assertEqual(config.scale_norm_mode, ScaleNormMode.POW2MAX)
        self.assertEqual(config.scale_round_mode, ScaleRoundMode.OAS)
        self.assertEqual(config.global_scale_granularity, GlobalScaleGranularity.NONE)
        self.assertEqual(config.elem_dtype, ElemDtype.E2M1)
        self.assertEqual(config.elem_round_mode, ElemRoundMode.NEAREST)
        self.assertEqual(config.block_size, BlockSize(1, 16))
        self.assertEqual(config.global_scale_dtype, ScaleDtype.NONE)

    def test_from_dict_partial_overrides(self) -> None:
        config = QuantConfig.from_dict({"elem_dtype": "e4m3", "block_size": "1x32"})
        self.assertEqual(config.elem_dtype, ElemDtype.E4M3)
        self.assertEqual(config.block_size, BlockSize(1, 32))
        # Defaults for non-specified fields
        self.assertEqual(config.quant_mode, QuantMode.SYMM)
        self.assertEqual(config.scale_dtype, ScaleDtype.E8M0)

    def test_from_dict_asymm_mean_shift_quant_mode(self) -> None:
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
        self.assertEqual(config.quant_mode, QuantMode.ASYMM_MEAN_SHIFT)
        self.assertEqual(config.to_dict()["quant_mode"], "asymm_mean_shift")

    def test_from_dict_full(self) -> None:
        config = QuantConfig.from_dict(
            {
                "quant_mode": "symm",
                "scale_dtype": "fp8",
                "scale_norm_mode": "amax",
                "scale_round_mode": "none",
                "global_scale_granularity": "tensorwise",
                "elem_dtype": "e2m1",
                "elem_round_mode": "nearest",
                "block_size": "1x16",
                "global_scale_dtype": "fp32",
            }
        )
        self.assertEqual(config.scale_dtype, ScaleDtype.FP8)
        self.assertEqual(config.scale_norm_mode, ScaleNormMode.AMAX)
        self.assertEqual(config.scale_round_mode, ScaleRoundMode.NONE)
        self.assertEqual(
            config.global_scale_granularity, GlobalScaleGranularity.TENSORWISE
        )
        self.assertEqual(config.global_scale_dtype, ScaleDtype.FP32)

    def test_from_dict_unknown_key_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            QuantConfig.from_dict({"elem_dtypo": "e2m1"})
        self.assertIn("Unknown QuantConfig keys", str(ctx.exception))
        self.assertIn("elem_dtypo", str(ctx.exception))

    def test_to_dict(self) -> None:
        config = QuantConfig.from_dict({"elem_dtype": "e4m3", "block_size": "1x16"})
        d = config.to_dict()
        self.assertEqual(d["elem_dtype"], "e4m3")
        self.assertEqual(d["block_size"], "1x16")
        self.assertEqual(d["quant_mode"], "symm")

    def test_round_trip(self) -> None:
        original = QuantConfig.from_dict(
            {
                "scale_dtype": "fp8",
                "scale_norm_mode": "amax",
                "scale_round_mode": "none",
                "global_scale_granularity": "tensorwise",
                "elem_dtype": "e4m3",
                "block_size": "32x32",
                "global_scale_dtype": "fp32",
            }
        )
        rebuilt = QuantConfig.from_dict(original.to_dict())
        self.assertEqual(original, rebuilt)

    def test_round_trip_all_defaults(self) -> None:
        original = QuantConfig.from_dict({})
        rebuilt = QuantConfig.from_dict(original.to_dict())
        self.assertEqual(original, rebuilt)

    def test_dynamic_supports_transpose_and_quant_info(self) -> None:
        """Dynamic MBS goes through the ordinary `qdq`, so the shared code handles
        transpose and return_quant_info."""
        cfg = QuantConfig.from_dict(
            {
                **FAKE_PRESETS["mxfp4"],
                "global_scale_granularity": "mbs256",
                "global_scale_dtype": "e0m8",
                "mbs_dynamic": "mbs_dynamic",
            }
        )
        torch.manual_seed(0)
        # non-square on purpose: a square input hides transpose shape/layout errors
        x = torch.randn(128, 512, dtype=torch.float32) * 0.05
        plain = fake_quantize(x, "w", cfg, use_triton=False)
        self.assertEqual(plain.shape, x.shape)
        self.assertTrue(bool(torch.isfinite(plain).all()))

        transposed = fake_quantize(x, "w", cfg, transpose=True, use_triton=False)
        self.assertEqual(transposed.shape, x.shape)
        self.assertTrue(bool(torch.isfinite(transposed).all()))

        info = fake_quantize(
            x,
            "w",
            cfg,
            use_triton=False,
            return_quant_info=True,
        )
        # assert the type rather than getattr-with-fallback, which would pass even
        # if a plain tensor came back
        self.assertIsInstance(info, QuantInfo)
        self.assertEqual(info.dequant_data.shape, x.shape)
        self.assertTrue(bool(torch.isfinite(info.dequant_data).all()))

    def test_mega_accepts_dynamic_only_for_e8m0(self) -> None:
        """The fused in-kernel search reproduces the e8m0 scale derivation only.

        For e8m0 the mega kernel implements the rule. For any other scale dtype it
        must refuse rather than search against the wrong scale -- and since the mega
        kernel is the only Triton backend, `use_triton=True` raises instead of
        quietly switching backends. The eager path still serves those configs.
        """
        base = {
            **FAKE_PRESETS["mxfp4"],
            "global_scale_granularity": "mbs256",
            "global_scale_dtype": "e0m8",
            "mbs_dynamic": "mbs_dynamic",
        }
        e8m0 = QuantConfig.from_dict(base)
        self.assertEqual(e8m0.scale_dtype, ScaleDtype.E8M0)
        self.assertTrue(is_triton_mega_supported(e8m0))

        # fp32 scale also requires scale_round_mode=none and rules out pow2max
        other = QuantConfig.from_dict(
            {
                **base,
                "scale_dtype": "fp32",
                "scale_round_mode": "none",
                "scale_norm_mode": "amax",
            }
        )
        self.assertFalse(is_triton_mega_supported(other))

        x = torch.randn(64, 256, dtype=torch.float32) * 0.05
        # A direct call must refuse, or it returns wrong numbers at a
        # plausible-looking SQNR ...
        with self.assertRaises(NotImplementedError):
            qdq_triton_mega(x, "w", other)
        # ... and so must the Triton dispatch, which has no fallback.
        with self.assertRaises(NotImplementedError):
            fake_quantize(x, "w", other, use_triton=True)


class TestMbsFactorDynamic(unittest.TestCase):
    """Contract of the searched MBS rule.

    Deliberately not a brute-force re-implementation of the search: that would be a
    mirrored-implementation test and could not catch a wrong objective. These pin
    properties the rule must satisfy by construction.
    """

    @staticmethod
    def _cfg(**over: object) -> QuantConfig:
        return QuantConfig.from_dict(
            {
                **FAKE_PRESETS["mxfp4"],
                "global_scale_granularity": "mbs256",
                "global_scale_dtype": "e0m8",
                **over,
            }
        )

    def test_factor_in_unit_octave_and_zero_blocks_unscaled(self) -> None:
        cfg = self._cfg(mbs_dynamic="mbs_dynamic")
        torch.manual_seed(0)
        x = torch.randn(4, 3, 256, dtype=torch.float32) * 0.05
        x[1, 1] = 0.0  # an all-zero macro block must not be rescaled
        mbs = _mbs_factor_dynamic(x, "w", cfg)
        self.assertEqual(mbs.shape, (4, 3))
        self.assertTrue(bool(((mbs >= 1.0) & (mbs < 2.0)).all()))
        self.assertEqual(float(mbs[1, 1]), 1.0)

    def test_never_worse_than_no_macro_scale(self) -> None:
        """gg=1.0 is in the candidate grid and means "no scaling", so a search that
        minimises SSE can never do worse than plain mxfp4. A broken objective (wrong
        sign, wrong reduction axis, stale best) shows up here."""
        torch.manual_seed(0)
        x = torch.randn(512, 512, dtype=torch.float32) * 0.05

        def sqnr(got: torch.Tensor) -> float:
            err = (got - x).pow(2).mean()
            return float(10 * torch.log10(x.pow(2).mean() / err))

        plain = QuantConfig.from_dict(dict(FAKE_PRESETS["mxfp4"]))
        s_plain = sqnr(fake_quantize(x, "w", plain, use_triton=False))
        s_dyn = sqnr(
            fake_quantize(
                x,
                "w",
                self._cfg(mbs_dynamic="mbs_dynamic"),
                use_triton=False,
            )
        )
        self.assertGreaterEqual(s_dyn, s_plain)

    def test_non_finite_blocks_do_not_freeze_the_search(self) -> None:
        """A NaN or Inf element must not make every candidate's error non-finite --
        that would leave the block silently pinned to slot 0 with no signal. `qdq`
        masks and clamps those separately, after the factor is chosen."""
        cfg = self._cfg(mbs_dynamic="mbs_dynamic")
        torch.manual_seed(0)
        x = torch.randn(3, 2, 256, dtype=torch.float32) * 0.05
        clean = _mbs_factor_dynamic(x.clone(), "w", cfg)
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                y = x.clone()
                y[0, 0, 5] = bad
                got = _mbs_factor_dynamic(y, "w", cfg)
                self.assertTrue(bool(torch.isfinite(got).all()))
                self.assertTrue(bool(((got >= 1.0) & (got < 2.0)).all()))
                # blocks without the poisoned element must be unaffected
                self.assertTrue(torch.equal(got[1:], clean[1:]))

    def test_rejects_quant_block_larger_than_macro_block(self) -> None:
        """A quant block straddling two macro blocks is rejected at config time."""
        with self.assertRaisesRegex(ValueError, "must be a multiple of block_size.n"):
            self._cfg(
                mbs_dynamic="mbs_dynamic",
                global_scale_granularity="mbs16",
                block_size="1x32",
            )

    def test_rejects_fp8_scale_at_config_time(self) -> None:
        """fp8 + MBS is constructible, but the search disables MBS on its inner
        config, which fp8 forbids. Fail at config time, not inside the helper."""
        with self.assertRaises(ValueError):
            self._cfg(
                mbs_dynamic="mbs_dynamic", scale_dtype="fp8", scale_round_mode="none"
            )


class TestElemDtypeMaxPow2(unittest.TestCase):
    """Pin the POW2MAX normalizer per quant mode and element dtype."""

    def test_table(self) -> None:
        ints = {
            QuantMode.SYMM: {"int3": 2, "int4": 4, "int8": 64},
            QuantMode.ASYMM: {"int3": 4, "int4": 8, "int8": 128},
            QuantMode.ASYMM_MEAN_SHIFT: {"int3": 2, "int4": 4, "int8": 64},
        }
        floats = {"e3m0": 16, "e2m1": 4, "e2m3": 4, "e4m3": 256}
        for mode, table in ints.items():
            for dtype, want in {**table, **floats}.items():
                with self.subTest(mode=mode, dtype=dtype):
                    self.assertEqual(elem_dtype_max_pow2(mode, ElemDtype(dtype)), want)
            for dtype in (ElemDtype.E8M1, ElemDtype.E8M2, ElemDtype.E8M3):
                self.assertEqual(
                    elem_dtype_max_pow2(mode, dtype), ELEM_DTYPE_MAX_POW2_MAP[dtype]
                )


class TestMbsFactorStatic(unittest.TestCase):
    """Pin `_mbs_factor_static` against an independent restatement of the formula."""

    @staticmethod
    def _reference(mbs_reshaped: torch.Tensor, fp_max: float) -> torch.Tensor:
        # sf = fp_max / amax, mantissa extracted so the factor lands in [1, 2)
        max_abs = torch.amax(torch.abs(mbs_reshaped), dim=2)
        safe = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs)
        sf = fp_max / safe
        bits = sf.view(torch.int32)
        bits = (bits & 0x007FFFFF) | (127 << 23)
        mbs = bits.view(torch.float32)
        return torch.where(max_abs == 0, torch.ones_like(mbs), mbs)

    def test_matches_reference_formula_bitwise(self) -> None:
        cfg = QuantConfig.from_dict(
            {
                **FAKE_PRESETS["mxfp4"],
                "global_scale_granularity": "mbs256",
                "global_scale_dtype": "e0m8",
            }
        )
        torch.manual_seed(0)
        for shape in ((4, 3, 256), (1, 1, 256), (8, 12, 256)):
            with self.subTest(shape=shape):
                x = torch.randn(*shape, dtype=torch.float32) * 0.05
                # exercise the zero-block branch too
                x[0, 0] = 0.0
                got = _mbs_factor_static(x, cfg, 6.0)
                want = self._reference(x, 6.0)
                self.assertTrue(
                    torch.equal(got, want),
                    f"static MBS factor changed for shape {shape}",
                )
                self.assertTrue(bool(((got >= 1.0) & (got < 2.0)).all()))

    def test_headroom_cut(self) -> None:
        """4 for a quant block whose finite peak is 2^126 or more, else 1; NaN
        is skipped and an Inf block is left alone."""
        below = 2.0**126 * (1 - 2.0**-24)
        peaks = [0.0, 1.0, below, 2.0**126, 2.0**127, F32_MAX, float("inf")]
        x = torch.zeros(len(peaks), 16)
        x[:, 3] = -torch.tensor(peaks)
        x[1, 5] = float("nan")
        got = _mbs_headroom(_mbs_quant_block_peaks(x, 16)).squeeze(1)
        self.assertEqual(got.tolist(), [1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 1.0])


if __name__ == "__main__":
    unittest.main()
