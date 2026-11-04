# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``QuantConfigExtension`` and ``QuantConfigBuilder``: serialization, validation
and hook dispatch, with toy extensions."""

import dataclasses
import pickle
import unittest
from typing import ClassVar, Optional, Union

import torch
from quant_suite_core import (
    fake_quantize,
    GlobalScaleGranularity,
    QuantConfig,
    QuantInfo,
    ScaleDtype,
    ScaleNormMode,
)
from quant_suite_core.extension import (
    DispatchPhase,
    NVFP4AmaxRule,
    PostProcessPhase,
    QuantConfigExtension,
)
from quant_suite_core.fake_presets import FAKE_PRESETS
from quant_suite_core.fake_quant import is_grouped_nvfp4_triton_supported
from quant_suite_core.quant_config import (
    DEFAULT_QUANT_CONFIG_BUILDER,
    QuantConfigBuilder,
)


class _HalfMaxPosRule:
    """Places each block's amax at half the element-format max."""

    def amax(
        self, blocks: torch.Tensor, max_pos: float
    ) -> tuple[torch.Tensor, float, torch.Tensor]:
        local_amax = blocks.abs().amax(dim=1) / (max_pos / 2.0)
        return local_amax, 1.0, local_amax.amax().reshape(1)


@dataclasses.dataclass(frozen=True)
class _Headroom(QuantConfigExtension):
    """An amax rule on an AMAX-normalized FP8 scale."""

    name: ClassVar[str] = "toy_headroom"

    def validate(self, config: QuantConfig) -> None:
        if config.scale_dtype is not ScaleDtype.FP8:
            raise ValueError("toy_headroom needs an fp8 scale")
        if config.scale_norm_mode is not ScaleNormMode.AMAX:
            raise ValueError("toy_headroom needs scale_norm_mode=amax")

    def amax_rule(self, config: QuantConfig) -> Optional[NVFP4AmaxRule]:
        return _HalfMaxPosRule()


@dataclasses.dataclass(frozen=True)
class _ZeroOutput(QuantConfigExtension):
    """A post-process hook: replaces every element with zero."""

    name: ClassVar[str] = "toy_zero_output"

    def post_process(
        self, config: QuantConfig, phase: PostProcessPhase
    ) -> Optional[torch.Tensor]:
        return torch.zeros_like(phase.data)


@dataclasses.dataclass(frozen=True)
class _DoubleOutput(QuantConfigExtension):
    """A finalize hook: doubles the dequantized output."""

    name: ClassVar[str] = "toy_double_output"

    def finalize(
        self, config: QuantConfig, phase: DispatchPhase
    ) -> Optional[Union[torch.Tensor, QuantInfo]]:
        return phase.with_dequantized(phase.dequantized * 2)


@dataclasses.dataclass(frozen=True)
class _ToyBits(QuantConfigExtension):
    """An int option, checked on construction and per config."""

    name: ClassVar[str] = "toy_bits"
    bits: int = 2

    def __post_init__(self) -> None:
        if not 1 <= self.bits <= 8:
            raise ValueError(f"bits must be in [1, 8], got {self.bits}")

    def validate(self, config: QuantConfig) -> None:
        if config.scale_dtype is not ScaleDtype.E8M0:
            raise ValueError("toy_bits needs an e8m0 scale")


@dataclasses.dataclass(frozen=True)
class _ScaleOutput(QuantConfigExtension):
    """A finalize hook with a float option: scales the dequantized output."""

    name: ClassVar[str] = "toy_scale_output"
    factor: float = 2.0

    def finalize(
        self, config: QuantConfig, phase: DispatchPhase
    ) -> Optional[Union[torch.Tensor, QuantInfo]]:
        return phase.with_dequantized(phase.dequantized * self.factor)


@dataclasses.dataclass(frozen=True)
class _Tag(QuantConfigExtension):
    """No options and no hooks."""

    name: ClassVar[str] = "toy_tag"


_BUILDER: QuantConfigBuilder = DEFAULT_QUANT_CONFIG_BUILDER.extend(
    (_Headroom, _ZeroOutput, _DoubleOutput, _ToyBits, _ScaleOutput, _Tag)
)

_TYPED_SPEC: dict[str, object] = {
    **FAKE_PRESETS["mxfp4"],
    "extensions": {"toy_bits": {"bits": 4}, "toy_tag": {}},
}


def _with(config: QuantConfig, *extensions: QuantConfigExtension) -> QuantConfig:
    return dataclasses.replace(config, extensions=extensions)


class ByteIdentityTest(unittest.TestCase):
    def test_presets_serialize_identically_with_extensions_installed(self) -> None:
        for name, spec in FAKE_PRESETS.items():
            with self.subTest(preset=name):
                plain = QuantConfig.from_dict(spec)
                for builder in (DEFAULT_QUANT_CONFIG_BUILDER, _BUILDER):
                    built = builder.build(spec)
                    self.assertEqual(built.extensions, ())
                    self.assertEqual(
                        list(built.to_dict().items()), list(plain.to_dict().items())
                    )
                    self.assertNotIn("extensions", built.to_dict())
                    self.assertEqual(repr(built), repr(plain))
                    self.assertEqual(builder.build(plain.to_dict()), plain)
                self.assertEqual(DEFAULT_QUANT_CONFIG_BUILDER.build_preset(name), plain)

    def test_pinned_dict(self) -> None:
        self.assertEqual(
            list(QuantConfig.from_dict(FAKE_PRESETS["mxfp4"]).to_dict().items()),
            [
                ("quant_mode", "symm"),
                ("scale_dtype", "e8m0"),
                ("scale_norm_mode", "pow2max"),
                ("scale_round_mode", "oas"),
                ("global_scale_granularity", "none"),
                ("elem_dtype", "e2m1"),
                ("elem_round_mode", "nearest"),
                ("block_size", "1x16"),
                ("global_scale_dtype", "none"),
                ("mbs_dynamic", False),
            ],
        )
        self.assertEqual(
            QuantConfig.from_dict(FAKE_PRESETS["nvfp4"]).to_dict(),
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
                "mbs_dynamic": False,
            },
        )


class SerializationTest(unittest.TestCase):
    def test_dict_round_trip_with_options(self) -> None:
        config = _BUILDER.build(_TYPED_SPEC)
        self.assertEqual(config.extensions, (_ToyBits(bits=4), _Tag()))
        self.assertTrue(config.has_extension("toy_bits"))
        serialized = config.to_dict()
        self.assertEqual(
            serialized["extensions"], {"toy_bits": {"bits": 4}, "toy_tag": {}}
        )
        plain = QuantConfig.from_dict(FAKE_PRESETS["mxfp4"]).to_dict()
        self.assertEqual({k: v for k, v in serialized.items() if k in plain}, plain)
        self.assertEqual(_BUILDER.build(serialized), config)

    def test_repr_includes_extensions(self) -> None:
        config = _BUILDER.build(_TYPED_SPEC)
        self.assertEqual(repr(config), f"QuantConfig({config.to_dict()!r})")
        self.assertIn("'toy_bits': {'bits': 4}", repr(config))

    def test_omitted_options_take_their_defaults_and_serialize(self) -> None:
        config = _BUILDER.build({"extensions": {"toy_bits": {}}})
        self.assertEqual(config.extensions, (_ToyBits(bits=2),))
        self.assertEqual(config.to_dict()["extensions"], {"toy_bits": {"bits": 2}})

    def test_int_option_widens_to_float(self) -> None:
        config = _BUILDER.build({"extensions": {"toy_scale_output": {"factor": 3}}})
        self.assertEqual(
            config.to_dict()["extensions"], {"toy_scale_output": {"factor": 3.0}}
        )
        self.assertIs(
            type(config.to_dict()["extensions"]["toy_scale_output"]["factor"]), float
        )

    def test_value_equality(self) -> None:
        self.assertEqual(_ToyBits(bits=2), _ToyBits(bits=2))
        self.assertEqual(hash(_ToyBits(bits=2)), hash(_ToyBits(bits=2)))
        self.assertNotEqual(_ToyBits(bits=2), _ToyBits(bits=4))
        base = QuantConfig.from_dict(FAKE_PRESETS["mxfp4"])
        self.assertEqual(
            _with(base, _ToyBits(bits=4)),
            _BUILDER.build(
                {**FAKE_PRESETS["mxfp4"], "extensions": {"toy_bits": {"bits": 4}}}
            ),
        )
        self.assertNotEqual(
            _with(base, _ToyBits(bits=2)), _with(base, _ToyBits(bits=4))
        )
        self.assertNotEqual(_with(base, _ToyBits(bits=2)), base)

    def test_unknown_extension_names_the_installed_ones(self) -> None:
        with self.assertRaisesRegex(ValueError, r"'toy_missing'.*toy_bits"):
            _BUILDER.build({"extensions": {"toy_missing": {}}})
        with self.assertRaisesRegex(ValueError, r"Installed extensions: \[\]"):
            QuantConfig.from_dict({"extensions": {"toy_bits": {}}})

    def test_extension_names_are_not_top_level_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown QuantConfig keys"):
            _BUILDER.build({**FAKE_PRESETS["mxfp4"], "toy_tag": True})

    def test_unknown_option_lists_the_options(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\['bitz'\].*'toy_bits'.*\['bits'\]"):
            _BUILDER.build({"extensions": {"toy_bits": {"bitz": 4}}})

    def test_invalid_options_raise(self) -> None:
        for options, pattern in (
            ({"bits": "4"}, "must be int"),
            ({"bits": True}, "must be int"),
            ({"bits": 9}, r"toy_bits.*\[1, 8\]"),
            (4, "must be a mapping"),
        ):
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, pattern):
                    _BUILDER.build({"extensions": {"toy_bits": options}})
        with self.assertRaisesRegex(ValueError, "'extensions' must map"):
            _BUILDER.build({"extensions": ["toy_bits"]})

    def test_string_specs_are_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "mapping of field names"):
            # pyrefly: ignore[bad-argument-type]
            _BUILDER.build("symm,e8m0,pow2max,oas,none,e2m1,nearest,1x16,none")

    def test_pickle(self) -> None:
        config = _BUILDER.build(
            {**_TYPED_SPEC, "extensions": {"toy_bits": {}, "toy_double_output": {}}}
        )
        clone = pickle.loads(pickle.dumps(config))
        self.assertEqual(clone, config)
        self.assertEqual(clone.extensions, config.extensions)
        self.assertEqual(clone.to_dict(), config.to_dict())

    def test_dataclasses_replace(self) -> None:
        config = _BUILDER.build(_TYPED_SPEC)
        rounded = dataclasses.replace(config, scale_round_mode=config.scale_round_mode)
        self.assertEqual(rounded, config)
        changed = dataclasses.replace(config, extensions=(_ToyBits(bits=6),))
        self.assertEqual(changed.to_dict()["extensions"], {"toy_bits": {"bits": 6}})
        with self.assertRaisesRegex(ValueError, "e8m0"):
            dataclasses.replace(
                QuantConfig.from_dict(FAKE_PRESETS["nvfp4"]), extensions=(_ToyBits(),)
            )


class ValidationTest(unittest.TestCase):
    def test_extensions_are_sorted_by_name(self) -> None:
        base = QuantConfig.from_dict(FAKE_PRESETS["mxfp4"])
        config = _with(base, _Tag(), _DoubleOutput())
        self.assertEqual(config.extensions, (_DoubleOutput(), _Tag()))

    def test_one_extension_per_hook(self) -> None:
        base = QuantConfig.from_dict(FAKE_PRESETS["mxfp4"])
        with self.assertRaisesRegex(ValueError, "finalize"):
            _with(base, _DoubleOutput(), _ScaleOutput())
        with self.assertRaisesRegex(ValueError, "finalize"):
            _BUILDER.build(
                {"extensions": {"toy_double_output": {}, "toy_scale_output": {}}}
            )

    def test_names_are_unique_in_a_config(self) -> None:
        base = QuantConfig.from_dict(FAKE_PRESETS["mxfp4"])
        with self.assertRaisesRegex(ValueError, "share the name 'toy_bits'"):
            _with(base, _ToyBits(bits=2), _ToyBits(bits=4))

    def test_extension_validate_runs(self) -> None:
        with self.assertRaisesRegex(ValueError, "e8m0"):
            _BUILDER.build({**FAKE_PRESETS["nvfp4"], "extensions": {"toy_bits": {}}})
        with self.assertRaisesRegex(ValueError, "fp8 scale"):
            _with(QuantConfig.from_dict(FAKE_PRESETS["mxfp4"]), _Headroom())

    def test_global_scale_pairing_has_no_exemption(self) -> None:
        with self.assertRaises(ValueError):
            _BUILDER.build(
                {
                    **FAKE_PRESETS["mxfp4"],
                    "global_scale_dtype": "e0m8",
                    "extensions": {"toy_bits": {}},
                }
            )

    def test_replace_with_mbs_dynamic_keeps_extensions(self) -> None:
        config = _BUILDER.build(
            {
                **FAKE_PRESETS["mxfp4_mbs256_dyn"],
                "extensions": {"toy_double_output": {}},
            }
        )
        # The eager MBS-dynamic search rebuilds the config with MBS off.
        inner = dataclasses.replace(
            config,
            global_scale_granularity=GlobalScaleGranularity.NONE,
            mbs_block_size=None,
            global_scale_dtype=ScaleDtype.NONE,
            mbs_dynamic=False,
        )
        self.assertEqual(inner.extensions, (_DoubleOutput(),))

        torch.manual_seed(0)
        x = torch.randn(16, 512) * 0.05
        plain = fake_quantize(x, "w", _with(config), use_triton=False)
        doubled = fake_quantize(x, "w", config, use_triton=False)
        torch.testing.assert_close(doubled, plain * 2, rtol=0, atol=0)


class HookDispatchTest(unittest.TestCase):
    def test_post_process_hook_changes_output(self) -> None:
        base = QuantConfig.from_dict(FAKE_PRESETS["mxfp4"])
        x = torch.randn(8, 64)
        plain = fake_quantize(x, "a", base, use_triton=False)
        zeroed = fake_quantize(x, "a", _with(base, _ZeroOutput()), use_triton=False)
        self.assertTrue(bool(plain.abs().sum() > 0))
        self.assertTrue(bool((zeroed == 0).all()))

    def test_amax_rule_changes_output(self) -> None:
        base = QuantConfig.from_dict(FAKE_PRESETS["nvfp4"])
        ruled = _with(base, _Headroom())
        x = torch.randn(8, 64)
        self.assertFalse(
            torch.equal(
                fake_quantize(x, "a", base, use_triton=False),
                fake_quantize(x, "a", ruled, use_triton=False),
            )
        )

    def test_finalize_hook_dispatches(self) -> None:
        config = _BUILDER.build(
            {
                **FAKE_PRESETS["mxfp4"],
                "extensions": {"toy_scale_output": {"factor": 3.0}},
            }
        )
        torch.manual_seed(0)
        x = torch.randn(8, 64)
        plain = fake_quantize(x, "a", _with(config), use_triton=False)
        scaled = fake_quantize(x, "a", config, use_triton=False)
        torch.testing.assert_close(scaled, plain * 3.0, rtol=0, atol=0)

    def test_grouped_nvfp4_declines_any_extension(self) -> None:
        base = QuantConfig.from_dict(FAKE_PRESETS["nvfp4"])
        self.assertTrue(is_grouped_nvfp4_triton_supported(base))
        self.assertFalse(is_grouped_nvfp4_triton_supported(_with(base, _Tag())))


class BuilderTest(unittest.TestCase):
    def test_build_preset_from_extension_presets(self) -> None:
        spec = {**FAKE_PRESETS["mxfp4"], "extensions": {"toy_bits": {"bits": 3}}}
        builder = DEFAULT_QUANT_CONFIG_BUILDER.extend(
            (_ToyBits,), presets={"toy_preset": spec}
        )
        self.assertEqual(builder.build_preset("toy_preset"), builder.build(spec))
        with self.assertRaisesRegex(ValueError, "toy_missing"):
            builder.build_preset("toy_missing")

    def test_extend_rejects_name_clash(self) -> None:
        @dataclasses.dataclass(frozen=True)
        class _OtherTag(QuantConfigExtension):
            name: ClassVar[str] = "toy_tag"

        with self.assertRaisesRegex(ValueError, "toy_tag"):
            _BUILDER.extend((_OtherTag,))

    def test_extend_rejects_preset_clash(self) -> None:
        with self.assertRaisesRegex(ValueError, "mxfp4"):
            DEFAULT_QUANT_CONFIG_BUILDER.extend(presets={"mxfp4": {}})

    def test_builder_takes_classes_and_config_takes_instances(self) -> None:
        with self.assertRaisesRegex(TypeError, "QuantConfigExtension subclasses"):
            # pyrefly: ignore[bad-argument-type]
            DEFAULT_QUANT_CONFIG_BUILDER.extend((_ToyBits(),))
        with self.assertRaisesRegex(TypeError, "instance"):
            # pyrefly: ignore[bad-argument-type]
            _with(QuantConfig.from_dict(FAKE_PRESETS["mxfp4"]), _ToyBits)

    def test_rejects_malformed_declarations(self) -> None:
        @dataclasses.dataclass
        class _NotFrozen(QuantConfigExtension):
            name: ClassVar[str] = "toy_not_frozen"

        class _NotDataclass(QuantConfigExtension):
            name: ClassVar[str] = "toy_not_dataclass"

        @dataclasses.dataclass(frozen=True)
        class _NameField(QuantConfigExtension):
            name: str = "toy_name_field"

        @dataclasses.dataclass(frozen=True)
        class _BadName(QuantConfigExtension):
            name: ClassVar[str] = "Toy-Name"

        @dataclasses.dataclass(frozen=True)
        class _TupleOption(QuantConfigExtension):
            name: ClassVar[str] = "toy_tuple_option"
            sizes: tuple[int, ...] = ()

        @dataclasses.dataclass(frozen=True)
        class _OptionalOption(QuantConfigExtension):
            name: ClassVar[str] = "toy_optional_option"
            bits: Optional[int] = None

        for extension, pattern in (
            (_NotFrozen, "frozen"),
            (_NotDataclass, "frozen dataclass"),
            (_NameField, "ClassVar"),
            (_BadName, "lower-case"),
            (_TupleOption, "sizes.*bool, int, float or str"),
            (_OptionalOption, "bits.*bool, int, float or str"),
        ):
            with self.subTest(extension=extension):
                with self.assertRaisesRegex(ValueError, pattern):
                    QuantConfigBuilder(extensions=(extension,))
