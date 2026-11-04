# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantization configuration: the format enums, ``QuantConfig`` and its parsers,
and the element-format range/bit tables those configs are validated against."""

import dataclasses
import logging
import re
import typing
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Optional, TypeVar, Union

import torch
from quant_suite_core.extension import (
    DispatchPhase,
    PostProcessPhase,
    QuantConfigExtension,
    ScalePhase,
)
from quant_suite_core.fake_presets import FAKE_PRESETS

logger = logging.getLogger(__name__)


# ******************************************************************************
# * Enums for Quantization Modes and Data Types                                *
# ******************************************************************************
class QuantMode(Enum):
    """How a block's values are centered before they are scaled.

    - ``SYMM``: no zero point; the scale is derived from the block amax.
    - ``ASYMM``: a per-block zero point -- the midpoint of the block's min and
      max for a signed element range, the block min for an unsigned one -- with
      the scale derived from the block's range. The zero padding of a partial
      block is left out of its min and max.
    - ``ASYMM_MEAN_SHIFT``: a per-block zero point equal to the block mean, with
      the scale derived from the largest deviation from that mean; integer
      codes are signed. A NaN counts as zero in the mean (whose divisor stays
      the block width) and sets no scale, and dequantizes to NaN; an Inf makes
      its whole block NaN.
    """

    SYMM = "symm"
    ASYMM = "asymm"
    ASYMM_MEAN_SHIFT = "asymm_mean_shift"


class ScaleDtype(Enum):
    """Number format of a scaling factor.

    Used for both the per-block scale (``QuantConfig.scale_dtype``) and the
    second-level global scale (``QuantConfig.global_scale_dtype``).

    - ``NONE``: no scale; valid only as ``global_scale_dtype``.
    - ``FP32``, ``FP16``: IEEE floats.
    - ``FP8``: an E4M3 block scale, paired with a tensorwise or rowwise global
      scale (two-level scaling, as in NVFP4).
    - ``E8M0``: an 8-bit power-of-two exponent (the OCP MX scale).
    - ``E0M8``: an MBS macro factor with no exponent, in ``[1, 2)``. The static
      rule keeps all 23 fp32 mantissa bits of ``fp_max / amax``; with
      ``mbs_dynamic`` the factor is searched on the ``1 + i / MBS_DYNAMIC_SLOTS``
      grid, which fits in 8 mantissa bits. Valid only as ``global_scale_dtype``.
    """

    NONE = "none"
    FP32 = "fp32"
    FP16 = "fp16"
    FP8 = "fp8"
    E8M0 = "e8m0"
    E0M8 = "e0m8"

    def to_torch_dtype(self) -> torch.dtype:
        """Convert ScaleDtype to torch.dtype. Used for global_scale_dtype casting."""
        _map = {
            ScaleDtype.FP32: torch.float32,
            ScaleDtype.FP16: torch.float16,
        }
        if self not in _map:
            raise ValueError(f"No torch dtype mapping for ScaleDtype.{self.name}")
        return _map[self]


class ElemDtype(Enum):
    """Number format of the quantized elements.

    - ``INT3``, ``INT4``, ``INT8``: integers; the symmetric range drops the most
      negative code, e.g. ``[-7, 7]`` for ``INT4``.
    - ``E3M0``, ``E2M1`` (FP4), ``E2M3`` (FP6), ``E4M3`` (FP8): minifloats named
      by their exponent and mantissa bits.
    - ``E8M1``, ``E8M2``, ``E8M3``: 8 exponent bits with 1 to 3 mantissa bits.
      The block scale carries the magnitude, so elements saturate at
      ``(2 - 2**-m) * 2**2``: 6.0, 7.0 and 7.5.

    ``ELEM_DTYPE_RANGE_MAP`` and ``ELEM_DTYPE_BITS_MAP`` give each format's range
    and bit widths.
    """

    INT3 = "int3"
    INT4 = "int4"
    INT8 = "int8"
    E3M0 = "e3m0"
    E2M1 = "e2m1"
    E2M3 = "e2m3"
    E4M3 = "e4m3"
    E8M1 = "e8m1"
    E8M2 = "e8m2"
    E8M3 = "e8m3"


class ElemRoundMode(Enum):
    """How scaled values are rounded onto the element grid.

    - ``NEAREST``: round to nearest, ties to even.
    - ``STOCHASTIC``: round up with probability equal to the fractional
      distance to the next grid point. Not supported for integer elements.
    """

    STOCHASTIC = "stochastic"
    NEAREST = "nearest"


class ScaleNormMode(Enum):
    """What the block amax is normalized by to form the scale.

    - ``AMAX``: the element format's largest value, so the block amax maps onto
      it.
    - ``POW2MAX``: the largest power of two in the element range of the quant
      mode (``elem_dtype_max_pow2``). Requires an ``E8M0`` scale.
    """

    AMAX = "amax"
    POW2MAX = "pow2max"


class ScaleRoundMode(Enum):
    """How a block scale is rounded to an ``E8M0`` or ``FP8`` scale dtype.

    - ``NONE``: no rounding step; required for ``FP32``/``FP16`` scales. An
      ``FP8`` scale is then a plain round-to-nearest cast.
    - ``FLOOR``, ``CEIL``: round down or up.
    - ``OAS`` (overflow-aware scaling, ``E8M0`` only): round down to a power of
      two unless the block amax would then round past the element format's
      largest value, in which case round up.
    - ``OAS_S``: ``OAS`` for element formats that reserve the all-ones code
      (e.g. ``E4M3``, where it is NaN), so the largest value is skipped.
    - ``NV_ROUND`` (``E8M0`` only): round up to the next power of two for any
      normal value that is not already one, without passing the largest finite
      exponent; subnormal values round up from one half.

    ``OAS`` and ``OAS_S`` follow "Unveiling the Potential of Quantization with
    MXFP4: Strategies for Quantization Error Reduction" (arXiv 2603.08713).
    """

    NONE = "none"
    CEIL = "ceil"
    OAS = "oas"
    OAS_S = "oas_s"
    FLOOR = "floor"
    NV_ROUND = "nv_round"


class GlobalScaleGranularity(Enum):
    """Extent of the second-level scale applied on top of the block scales.

    - ``NONE``: block scales only. With an ``FP8`` block scale the global scale
      is one, so inputs must already lie in the E4M3 block-scale range; block
      scales beyond it saturate.
    - ``TENSORWISE``, ``ROWWISE``: one global scale per tensor or per row; used
      with an ``FP8`` block scale (NVFP4 two-level scaling).
    - ``MBS`` (macro block scale): a second, coarser scale shared by a run of
      quant blocks along a row, of width ``QuantConfig.mbs_block_size``. Its
      serialized token carries the width, e.g. ``"mbs128"``. MBS follows
      "Unveiling the Potential of Quantization with MXFP4: Strategies for
      Quantization Error Reduction" (arXiv 2603.08713).
    """

    NONE = "none"
    TENSORWISE = "tensorwise"
    ROWWISE = "rowwise"
    MBS = "mbs"

    def is_mbs(self) -> bool:
        """Whether this is the ``MBS`` granularity."""
        return self is GlobalScaleGranularity.MBS


class MBSBlockSize(Enum):
    """The macro-block widths MBS supports.

    An enum and not an ``int`` because the set is closed and not every member is
    a power of two (``MBS384``), so no bit test can validate membership.
    """

    MBS16 = 16
    MBS32 = 32
    MBS64 = 64
    MBS128 = 128
    MBS256 = 256
    MBS384 = 384
    MBS512 = 512


# The dict form writes ``global_scale_granularity`` and ``mbs_block_size`` as
# one token: an MBS granularity is ``"mbs<width>"`` (e.g. ``"mbs128"``) and every
# other granularity is its own value. The tokens are config identities (e.g.
# recipe dict keys), so their spelling is frozen.
_MBS_TOKEN_PREFIX = "mbs"


@dataclass
class BlockSize:
    """The ``m x n`` tile of elements that shares one block scale.

    ``m`` is the number of rows and ``n`` the number of columns; ``str`` and
    ``from_str`` use the ``"MxN"`` form, e.g. ``"1x32"``.
    """

    m: int
    n: int

    def __str__(self):
        return f"{self.m}x{self.n}"

    @classmethod
    def from_str(cls, s):
        m, n = map(int, s.split("x"))
        return cls(m, n)


def _global_scale_granularity_fields(token: str) -> dict[str, Any]:
    """Unpack a granularity token into ``global_scale_granularity`` and
    ``mbs_block_size``."""
    lowered = token.lower()
    # A member-name lookup, not prefix surgery: ``MBSBlockSize`` member
    # names are exactly the tokens upper-cased, so "mbs128" -> ``MBS128``.
    # Spellings ``int`` would have accepted -- "mbs0128", "mbs+128" -- miss,
    # which matters because they re-serialize as "mbs128" and this token is
    # a config identity.
    try:
        return {
            "global_scale_granularity": GlobalScaleGranularity.MBS,
            "mbs_block_size": MBSBlockSize[lowered.upper()],
        }
    except KeyError:
        pass
    # The bare "mbs" resolves here as a real granularity value, and
    # ``__post_init__`` reports the missing block size.
    try:
        granularity = _parse_enum(GlobalScaleGranularity, lowered)
    except ValueError:
        # Neither a block size nor a granularity. An "mbs"-prefixed token
        # was reaching for a block size, so name that -- reporting "mbs48"
        # against the granularity enum sends the reader looking for a
        # granularity that was never the problem.
        if not lowered.startswith(_MBS_TOKEN_PREFIX):
            raise
        raise ValueError(
            f"Unsupported MBS block size in {token!r}; supported: "
            + ", ".join(f"{_MBS_TOKEN_PREFIX}{s.value}" for s in MBSBlockSize)
        ) from None
    return {
        "global_scale_granularity": granularity,
        "mbs_block_size": None,
    }


def _global_scale_granularity_token(config: "QuantConfig") -> str:
    """The token ``config`` writes for its granularity: ``"mbs<width>"`` for
    MBS, else the granularity's value."""
    granularity = config.global_scale_granularity
    if granularity.is_mbs() and config.mbs_block_size is not None:
        return f"{_MBS_TOKEN_PREFIX}{config.mbs_block_size.value}"
    return granularity.value


_QUANT_CONFIG_DEFAULTS: dict[str, str | bool] = {
    "quant_mode": "symm",
    "scale_dtype": "e8m0",
    "scale_norm_mode": "pow2max",
    "scale_round_mode": "oas",
    "global_scale_granularity": "none",
    "elem_dtype": "e2m1",
    "elem_round_mode": "nearest",
    "block_size": "1x16",
    "global_scale_dtype": "none",
    "mbs_dynamic": False,
}

_FIELD_PARSERS: dict[str, Callable[[str], Any]] = {
    "quant_mode": lambda v: _parse_enum(QuantMode, v),
    "scale_dtype": lambda v: _parse_enum(ScaleDtype, v),
    "scale_norm_mode": lambda v: _parse_enum(ScaleNormMode, v),
    "scale_round_mode": lambda v: _parse_enum(ScaleRoundMode, v),
    "elem_dtype": lambda v: _parse_enum(ElemDtype, v),
    "elem_round_mode": lambda v: _parse_enum(ElemRoundMode, v),
    "block_size": lambda v: _parse_block_size(v),
    "global_scale_dtype": lambda v: _parse_enum(ScaleDtype, v),
}

# Candidate count for the MBS SSE search.
MBS_DYNAMIC_SLOTS: int = 16
# The candidate grid is 1 + i/MBS_DYNAMIC_SLOTS. Only a power-of-two slot count makes
# those dyadic rationals; 1/24 for instance is not representable in any finite
# mantissa, so the searched factor could not be stored in the declared global scale.
# <= 256 keeps them within e0m8's 8 mantissa bits. Checked once, at import.
assert (
    MBS_DYNAMIC_SLOTS & (MBS_DYNAMIC_SLOTS - 1)
) == 0 and MBS_DYNAMIC_SLOTS <= 256, (
    f"MBS_DYNAMIC_SLOTS={MBS_DYNAMIC_SLOTS} must be a power of two and <= 256 so the "
    "1 + i/slots grid is exactly representable in an e0m8 global scale"
)


_DerivedScale = TypeVar("_DerivedScale")

_EXTENSION_HOOKS: tuple[str, ...] = ("amax_rule", "post_process", "finalize")
_EXTENSION_NAME_PATTERN: re.Pattern[str] = re.compile(r"[a-z][a-z0-9_]*")

# Option annotations whose values the dict parser checks.
_PLAIN_OPTION_TYPES: tuple[type, ...] = (bool, int, float, str)


def _overrides_hook(extension: QuantConfigExtension, hook: str) -> bool:
    return getattr(type(extension), hook) is not getattr(QuantConfigExtension, hook)


def _option_fields(extension: Any) -> tuple[dataclasses.Field, ...]:
    return tuple(f for f in dataclasses.fields(extension) if f.init)


def _extension_options(extension: QuantConfigExtension) -> dict[str, Any]:
    return {f.name: getattr(extension, f.name) for f in _option_fields(extension)}


def _check_extension_class(cls: object) -> type[QuantConfigExtension]:
    """Raise unless ``cls`` is a well-formed extension class; return it."""
    if not (isinstance(cls, type) and issubclass(cls, QuantConfigExtension)):
        raise TypeError(
            f"extensions must be QuantConfigExtension subclasses, got {cls!r}"
        )
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not _EXTENSION_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            f"{cls.__name__}.name must be lower-case letters, digits "
            f"and underscores, starting with a letter, but got {name!r}"
        )
    params = getattr(cls, "__dataclass_params__", None)
    if params is None or not params.frozen:
        raise ValueError(
            f"extension {name!r} must be a frozen dataclass, so that a config "
            "holding it compares and hashes by value"
        )
    if "name" in {f.name for f in dataclasses.fields(cls)}:
        raise ValueError(
            f"extension {name!r} declares name as a dataclass field; declare it "
            "as ClassVar[str]"
        )
    try:
        hints = typing.get_type_hints(cls)
    except (NameError, TypeError) as error:
        raise TypeError(
            f"cannot resolve the option annotations of extension {name!r}: {error}"
        ) from error
    for field in _option_fields(cls):
        if hints.get(field.name) not in _PLAIN_OPTION_TYPES:
            raise ValueError(
                f"option {field.name!r} of extension {name!r} must be annotated "
                f"bool, int, float or str, but is {hints.get(field.name)!r}"
            )
    return cls


def _check_extension_instance(extension: object) -> QuantConfigExtension:
    """Raise unless ``extension`` is an instance of a well-formed extension."""
    if isinstance(extension, type):
        raise TypeError(
            f"a QuantConfig holds extension instances, but got the class "
            f"{extension.__name__}; instantiate it with its options"
        )
    _check_extension_class(type(extension))
    assert isinstance(extension, QuantConfigExtension)
    return extension


_Extension = TypeVar("_Extension", QuantConfigExtension, type[QuantConfigExtension])


def _sorted_by_name(entries: Iterable[_Extension]) -> tuple[_Extension, ...]:
    """``entries`` sorted by name, raising if two share one."""
    checked = sorted(entries, key=lambda e: e.name)
    for first, second in zip(checked, checked[1:]):
        if first.name == second.name:
            raise ValueError(
                f"two extensions share the name {first.name!r}: {first!r} and "
                f"{second!r}"
            )
    return tuple(checked)


def _installed_extensions_note(available: Iterable[type[QuantConfigExtension]]) -> str:
    names = [e.name for e in available]
    return (
        f"Installed extensions: {names}. An extension parses only through a "
        "QuantConfigBuilder that has the extension."
    )


def _check_option_type(name: str, key: str, value: Any, hint: Any) -> Any:
    """``value`` checked against a plain option annotation; ints widen to float."""
    if hint not in _PLAIN_OPTION_TYPES:
        return value
    if hint is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, hint) and (hint is bool or not isinstance(value, bool)):
        return value
    raise ValueError(
        f"option {key!r} of extension {name!r} must be {hint.__name__}, "
        f"but got {value!r}"
    )


def _extension_from_options(
    cls: type[QuantConfigExtension], options: object
) -> QuantConfigExtension:
    """Construct an extension from its options mapping."""
    name = cls.name
    if not isinstance(options, Mapping):
        raise ValueError(
            f"options of extension {name!r} must be a mapping of option name to "
            f"value, but got {options!r}"
        )
    fields = [f.name for f in _option_fields(cls)]
    unknown = [key for key in options if key not in fields]
    if unknown:
        raise ValueError(
            f"unknown options {unknown} for extension {name!r}; its options are "
            f"{fields}"
        )
    try:
        hints = typing.get_type_hints(cls)
    except (NameError, TypeError) as error:
        raise TypeError(
            f"cannot resolve the option annotations of extension {name!r}: {error}"
        ) from error
    checked = {
        key: _check_option_type(name, key, value, hints.get(key))
        for key, value in options.items()
    }
    try:
        return cls(**checked)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"invalid options {checked} for extension {name!r} (its options are "
            f"{fields}): {error}"
        ) from error


def _parse_extensions(
    raw: object, available: tuple[type[QuantConfigExtension], ...]
) -> list[QuantConfigExtension]:
    """Parse the ``"extensions"`` mapping of extension name to options."""
    if raw is None:
        return []
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"'extensions' must map extension names to option mappings, but got {raw!r}"
        )
    by_name = {e.name: e for e in available}
    enabled = []
    for name, options in raw.items():
        extension_cls = by_name.get(name)
        if extension_cls is None:
            raise ValueError(
                f"Unknown extension {name!r} under 'extensions'. "
                + _installed_extensions_note(available)
            )
        enabled.append(_extension_from_options(extension_cls, options))
    return enabled


def _parse_mbs_dynamic(value: Any) -> bool:
    """``mbs_dynamic`` from the dict form: a bool, or its token."""
    if isinstance(value, bool):
        return value
    token = value.lower() if isinstance(value, str) else None
    if token not in ("mbs_dynamic", "none"):
        raise ValueError(
            "QuantConfig field 'mbs_dynamic' must be 'mbs_dynamic', 'none' or a "
            f"bool, but got {value!r}"
        )
    return token == "mbs_dynamic"


def _parse_granularity(raw: Any) -> dict[str, Any]:
    """``global_scale_granularity`` and ``mbs_block_size`` from the dict form."""
    if not isinstance(raw, Enum):
        return _global_scale_granularity_fields(str(raw))
    # MBS is the one member that carries extra packing. Say so here rather than
    # letting it fall through to ``__post_init__``'s generic "set exactly when"
    # check, which states the invariant but not what to pass instead.
    if raw is GlobalScaleGranularity.MBS:
        raise ValueError(
            "GlobalScaleGranularity.MBS carries no block size, so it cannot be "
            "passed as a bare member; pass the packed token instead, e.g. "
            f"'{_MBS_TOKEN_PREFIX}{MBSBlockSize.MBS384.value}'"
        )
    return {"global_scale_granularity": raw}


def _config_from_dict(
    d: Mapping[str, Any], available: tuple[type[QuantConfigExtension], ...]
) -> "QuantConfig":
    """``QuantConfig.from_dict``, also accepting the ``available`` extensions."""
    if not isinstance(d, Mapping):
        raise TypeError(
            "a QuantConfig spec is a mapping of field names to values, e.g. "
            f"{{'elem_dtype': 'e4m3', 'block_size': '1x32'}}, but got "
            f"{type(d).__name__} {d!r}"
        )
    d = dict(d)
    extensions = _parse_extensions(d.pop("extensions", None), available)
    unknown_keys = set(d) - set(_QUANT_CONFIG_DEFAULTS)
    if unknown_keys:
        raise ValueError(
            f"Unknown QuantConfig keys: {unknown_keys}. "
            f"Valid keys: {sorted({*_QUANT_CONFIG_DEFAULTS, 'extensions'})}."
        )
    merged = {**_QUANT_CONFIG_DEFAULTS, **d}
    parsed: dict[str, Any] = _parse_granularity(merged.pop("global_scale_granularity"))
    parsed["mbs_dynamic"] = _parse_mbs_dynamic(merged.pop("mbs_dynamic"))
    for key, raw_value in merged.items():
        parsed[key] = (
            raw_value
            if isinstance(raw_value, (Enum, BlockSize))
            else _FIELD_PARSERS[key](str(raw_value))
        )
    if extensions:
        parsed["extensions"] = tuple(extensions)
    return QuantConfig(**parsed)


def _hook_holder(config: "QuantConfig", hook: str) -> Optional[QuantConfigExtension]:
    """The enabled extension that overrides ``hook``; ``__post_init__``
    guarantees there is at most one."""
    for extension in config.extensions:
        if _overrides_hook(extension, hook):
            return extension
    return None


@typing.final
@dataclass
class QuantConfig:
    """A complete description of one quantization format.

    Build one with ``QuantConfig.from_dict`` from a mapping, or with a
    ``QuantConfigBuilder`` when the mapping enables extensions;
    ``__post_init__`` rejects field combinations no backend implements.
    ``QuantConfig`` cannot be subclassed: optional behaviour is added as a
    ``QuantConfigExtension``.

    Attributes:
        quant_mode: Symmetric or asymmetric centering (``QuantMode``).
        scale_dtype: Format of the per-block scale.
        scale_norm_mode: What the block amax is normalized by.
        scale_round_mode: How the block scale is rounded to ``scale_dtype``.
        global_scale_granularity: Extent of the second-level scale, if any.
        elem_dtype: Format of the quantized elements.
        elem_round_mode: How values are rounded onto the element grid.
        block_size: The tile that shares one block scale.
        global_scale_dtype: Format of the second-level scale; ``NONE`` exactly
            when there is none.
        mbs_dynamic: How the MBS macro factor is chosen (see below). Only valid
            with an MBS granularity.
        mbs_block_size: Width of the MBS macro block; set exactly when the
            granularity is MBS.
        extensions: The ``QuantConfigExtension`` rules enabled on this config.

    The serialized form is the mapping of ``to_dict`` / ``from_dict``. The MBS
    block size is packed into the granularity token (``"mbs384"``) rather than
    written as a field of its own, and extensions are written under
    ``"extensions"`` as ``{name: {option: value}}``.
    """

    quant_mode: QuantMode
    scale_dtype: ScaleDtype
    scale_norm_mode: ScaleNormMode
    scale_round_mode: ScaleRoundMode
    global_scale_granularity: GlobalScaleGranularity
    elem_dtype: ElemDtype
    elem_round_mode: ElemRoundMode
    block_size: BlockSize
    global_scale_dtype: ScaleDtype = ScaleDtype.NONE
    # MBS macro factor selection: "static" (default) vs "dynamic".
    #   static  -- derived closed-form from the macro-block amax:
    #              sf = fp_max / amax, mantissa extracted to land the block peak
    #              on fp_max. Decided by the single largest element.
    #   dynamic -- chosen by an SSE search over MBS_DYNAMIC_SLOTS candidates in
    #              [1, 2), minimising reconstruction error over the whole macro
    #              block rather than aligning its peak.
    # Only meaningful with an MBS granularity.
    mbs_dynamic: bool = False
    # Width of the MBS macro block. Set exactly when global_scale_granularity
    # is MBS. Serialized packed into the granularity token ("mbs384"), never as
    # a field of its own.
    mbs_block_size: Optional[MBSBlockSize] = None
    # The optional rules enabled on this config. ``__post_init__`` sorts them by
    # name, so the order they are given in is not part of the config.
    extensions: tuple[QuantConfigExtension, ...] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError(
            "QuantConfig cannot be subclassed; add a QuantConfigExtension instead"
        )

    def _derive_scale(self, phase: ScalePhase[_DerivedScale]) -> _DerivedScale:
        """Extension point at the end of the scaling-factor phase.

        Selects the scale rule, then lets the phase finish. A rule cannot be a
        step *after* the scale: it participates in the derivation, as the amax
        the per-block scale is built from, as the magnitude that amax is placed
        on, and as the source the NVFP4 global amax is then reduced from. So the
        hook owns the phase -- it runs where the inputs are fixed and the
        derivation has not, and returns what the derivation produced. Applies
        the ``amax_rule`` of the extension that provides one, if any, then lets
        the phase finish.
        """
        extension = _hook_holder(self, "amax_rule")
        rule = None if extension is None else extension.amax_rule(self)
        if rule is not None:
            phase = phase.with_amax_rule(rule)
        return phase.derive()

    def _post_process(self, phase: PostProcessPhase) -> torch.Tensor:
        """Extension point at the end of a backend's post-process phase.

        Runs once the elements have been cast to the target format, and before
        the backend unwinds what pre-process set up. Returns the tensor that
        unwinding consumes. Runs the ``post_process`` of the extension that
        provides one, if any; otherwise the identity.
        """
        extension = _hook_holder(self, "post_process")
        if extension is not None:
            data = extension.post_process(self, phase)
            if data is not None:
                return data
        return phase.data

    def _finalize(self, phase: DispatchPhase) -> Union[torch.Tensor, "QuantInfo"]:
        """Extension point at the end of ``fake_quantize``.

        Separate from ``_post_process`` because it is a different phase, not a
        different rule: it sees the output at the caller's shape and fires once
        per dispatch, which a backend hook cannot offer. Runs the ``finalize``
        of the extension that provides one, if any; otherwise the identity.
        """
        extension = _hook_holder(self, "finalize")
        if extension is not None:
            result = extension.finalize(self, phase)
            if result is not None:
                return result
        return phase.result

    def has_extension(self, name: str) -> bool:
        """Whether the extension named ``name`` is enabled on this config."""
        return any(e.name == name for e in self.extensions)

    def get_mbs_block_size(self) -> int:
        """The MBS macro-block width, or 0 when MBS is not in use.

        Raises:
            ValueError: if MBS is in use but no block size is set. Returning 0
                there would be worse than failing: the backends guard only on
                ``is_mbs()``, and a width of 0 satisfies their
                ``width % block_size.n == 0`` divisibility assert, so they
                would quantize against a zero-width macro block instead of
                stopping.
        """
        if not self.global_scale_granularity.is_mbs():
            return 0
        if self.mbs_block_size is None:
            raise ValueError(
                "global_scale_granularity is MBS but mbs_block_size is unset, "
                "so there is no macro-block width to return"
            )
        return self.mbs_block_size.value

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "QuantConfig":
        """Create a QuantConfig from a dict, applying defaults for missing keys.

        Args:
            d: Dictionary mapping field names to string values.
                Only non-default fields need to be specified.

        Returns:
            QuantConfig with parsed values.

        Raises:
            TypeError: If ``d`` is not a mapping.
            ValueError: If unknown keys are present (to catch typos early),
                ``"extensions"`` is given (it needs a ``QuantConfigBuilder``
                that has those extensions), or the fields form an invalid
                combination.

        Examples:
            >>> QuantConfig.from_dict({"elem_dtype": "e4m3", "block_size": "1x16"})
            >>> QuantConfig.from_dict({})  # all defaults
        """
        return _config_from_dict(d, ())

    def to_dict(self) -> dict[str, Any]:
        """Serialize QuantConfig to a dict.

        Useful for YAML/JSON serialization and round-tripping through from_dict.

        Returns:
            Dict mapping field names to string representations (a bool for
            ``"mbs_dynamic"``), plus ``"extensions"`` mapping each extension's
            name to its options when the config has any.
        """
        result: dict[str, Any] = {
            "quant_mode": self.quant_mode.value,
            "scale_dtype": self.scale_dtype.value,
            "scale_norm_mode": self.scale_norm_mode.value,
            "scale_round_mode": self.scale_round_mode.value,
            "global_scale_granularity": _global_scale_granularity_token(self),
            "elem_dtype": self.elem_dtype.value,
            "elem_round_mode": self.elem_round_mode.value,
            "block_size": str(self.block_size),
            "global_scale_dtype": self.global_scale_dtype.value,
            "mbs_dynamic": self.mbs_dynamic,
        }
        if self.extensions:
            result["extensions"] = {
                e.name: _extension_options(e) for e in self.extensions
            }
        return result

    def __repr__(self) -> str:
        return f"QuantConfig({self.to_dict()!r})"

    def _check_extensions(self) -> None:
        """Every extension is a well-formed instance, names are unique, and at
        most one extension provides each hook."""
        self.extensions = _sorted_by_name(
            _check_extension_instance(e) for e in self.extensions
        )
        for hook in _EXTENSION_HOOKS:
            holders = [e.name for e in self.extensions if _overrides_hook(e, hook)]
            if len(holders) > 1:
                raise ValueError(
                    f"at most one extension may provide {hook!r}, but {holders} all do"
                )

    def __post_init__(self):
        self._check_extensions()

        # --- block_size: both dimensions positive ---
        if self.block_size.m < 1 or self.block_size.n < 1:
            raise ValueError(
                f"block_size must have m >= 1 and n >= 1, but got {self.block_size}"
            )

        # --- scale_dtype: only FP32, FP16, FP8, E8M0 are valid ---
        if self.scale_dtype not in (
            ScaleDtype.FP32,
            ScaleDtype.FP16,
            ScaleDtype.FP8,
            ScaleDtype.E8M0,
        ):
            raise ValueError(
                f"scale_dtype must be one of FP32, FP16, FP8, E8M0, but got {self.scale_dtype}"
            )

        # --- global_scale_dtype: E0M8 or FP32 with MBS; otherwise NONE or FP32 ---
        if self.global_scale_granularity.is_mbs():
            allowed_global_scale_dtypes = (ScaleDtype.E0M8, ScaleDtype.FP32)
        else:
            allowed_global_scale_dtypes = (ScaleDtype.NONE, ScaleDtype.FP32)
        if self.global_scale_dtype not in allowed_global_scale_dtypes:
            allowed = ", ".join(d.name for d in allowed_global_scale_dtypes)
            raise ValueError(
                f"global_scale_dtype must be one of {allowed} with "
                f"global_scale_granularity={self.global_scale_granularity.value}, "
                f"but got {self.global_scale_dtype}"
            )

        # --- scale_round_mode constraints per scale_dtype ---
        if not (
            self.scale_dtype not in (ScaleDtype.FP16, ScaleDtype.FP32)
            or self.scale_round_mode == ScaleRoundMode.NONE
        ):
            raise ValueError(
                f"for fp16 and fp32 scale, scale round mode should be none, but got {self.scale_round_mode}"
            )
        if not (
            self.scale_dtype != ScaleDtype.FP8
            or self.scale_round_mode
            in (
                ScaleRoundMode.NONE,
                ScaleRoundMode.FLOOR,
                ScaleRoundMode.CEIL,
            )
        ):
            raise ValueError(
                f"for fp8 scale, scale round mode should be none, floor, or ceil, but got {self.scale_round_mode}"
            )
        if not (
            self.scale_dtype not in (ScaleDtype.FP16, ScaleDtype.FP8)
            or self.scale_norm_mode == ScaleNormMode.AMAX
        ):
            raise ValueError(
                f"for fp16 and fp8 scale, they only support amax, but got {self.scale_dtype} {self.scale_norm_mode}"
            )
        if not (
            self.scale_dtype != ScaleDtype.E8M0
            or self.scale_round_mode != ScaleRoundMode.NONE
        ):
            raise ValueError(
                f"for e8m0 scale, scale round mode should not be none, but got {self.scale_round_mode}"
            )

        # --- scale_norm_mode: POW2MAX requires E8M0 scale_dtype ---
        if not (
            self.scale_norm_mode != ScaleNormMode.POW2MAX
            or self.scale_dtype == ScaleDtype.E8M0
        ):
            raise ValueError(
                f"POW2MAX requires E8M0 scale_dtype, but got {self.scale_dtype}"
            )

        # --- elem_dtype must have an element range under quant_mode ---
        if self.elem_dtype not in ELEM_DTYPE_RANGE_MAP[self.quant_mode]:
            raise ValueError(
                f"{self.quant_mode} does not support elem_dtype={self.elem_dtype}; "
                f"supported elem_dtypes: {list(ELEM_DTYPE_RANGE_MAP[self.quant_mode].keys())}"
            )

        # --- POW2MAX requires elem_dtype to be in ELEM_DTYPE_MAX_POW2_MAP ---
        if not (
            self.scale_norm_mode != ScaleNormMode.POW2MAX
            or self.elem_dtype in ELEM_DTYPE_MAX_POW2_MAP
        ):
            raise ValueError(
                f"POW2MAX is not supported for elem_dtype={self.elem_dtype}; "
                f"supported elem_dtypes: {list(ELEM_DTYPE_MAX_POW2_MAP.keys())}"
            )

        # --- elem_round_mode constraints ---
        if not (
            self.elem_dtype not in [ElemDtype.INT3, ElemDtype.INT4, ElemDtype.INT8]
            or self.elem_round_mode != ElemRoundMode.STOCHASTIC
        ):
            raise ValueError(
                f"int3/4/8 quantization does not support elem_round_mode {self.elem_round_mode}"
            )

        # --- global_scale_granularity constraints ---
        # Allow MBS granularity or FP8 with tensorwise/rowwise
        if not (
            self.scale_dtype == ScaleDtype.FP8
            or self.global_scale_granularity == GlobalScaleGranularity.NONE
            or self.global_scale_granularity.is_mbs()
        ):
            raise ValueError(
                f"{self.scale_dtype} scale does not support {self.global_scale_granularity} granularity "
                f"(only FP8 supports tensorwise/rowwise; MBS is supported for all scale dtypes)"
            )
        if not (
            self.global_scale_granularity.is_mbs() == (self.mbs_block_size is not None)
        ):
            raise ValueError(
                "mbs_block_size must be set exactly when global_scale_granularity is "
                f"MBS, but got granularity={self.global_scale_granularity} and "
                f"mbs_block_size={self.mbs_block_size}"
            )
        if not (not self.global_scale_granularity.is_mbs() or self.block_size.m == 1):
            raise ValueError(
                f"MBS requires block_size.m == 1, but got block_size={self.block_size}"
            )
        if (
            self.mbs_block_size is not None
            and self.mbs_block_size.value % self.block_size.n != 0
        ):
            raise ValueError(
                f"the MBS block size ({self.mbs_block_size.value}) must be a "
                f"multiple of block_size.n ({self.block_size.n})"
            )
        # Fail loudly rather than silently falling back to the amax rule: a
        # mbs_dynamic that quietly did nothing would look like "MBS doesn't help".
        # The in-kernel candidate search reproduces only the e8m0 scale derivation.
        if not (not self.mbs_dynamic or self.scale_dtype != ScaleDtype.FP8):
            raise ValueError(
                "mbs_dynamic does not support scale_dtype=fp8: the candidate search "
                "ranks candidates with the e8m0 scale derivation"
            )
        if not (not self.mbs_dynamic or self.global_scale_granularity.is_mbs()):
            raise ValueError(
                "mbs_dynamic requires an MBS global_scale_granularity, but got "
                f"{self.global_scale_granularity}"
            )

        # --- global_scale_granularity and global_scale_dtype must be both NONE or both non-NONE ---
        if not (
            self.global_scale_dtype != ScaleDtype.NONE
            or self.global_scale_granularity == GlobalScaleGranularity.NONE
        ):
            raise ValueError(
                f"global_scale_granularity must be NONE when global_scale_dtype is NONE, "
                f"but got global_scale_granularity={self.global_scale_granularity}"
            )
        if not (
            self.global_scale_granularity != GlobalScaleGranularity.NONE
            or self.global_scale_dtype == ScaleDtype.NONE
        ):
            raise ValueError(
                f"global_scale_dtype must be NONE when global_scale_granularity is NONE, "
                f"but got global_scale_dtype={self.global_scale_dtype}"
            )

        # --- MXFP8 OAS warning ---
        if (
            self.elem_dtype == ElemDtype.E4M3
            and self.scale_dtype == ScaleDtype.E8M0
            and self.scale_round_mode == ScaleRoundMode.OAS
        ):
            logger.warning(
                f"MXFP8 (E4M3 with E8M0 scale) reserves the all-ones exponent (11..1) for NaN, "
                f"so OAS_S (OAS with skipping the last representable value) is recommended "
                f"instead of OAS. Got scale_round_mode={self.scale_round_mode}"
            )

        for extension in self.extensions:
            extension.validate(self)


@dataclass
class QuantInfo:
    """Container for quantization information returned when return_quant_info=True.

    Attributes:
        dequant_data: The dequantized tensor.
        quantized_val: The quantized low-precision values, in the element
            domain.
        local_unscale: The per-block unscale factor (i.e., the inverse of the
            quantization scale). Multiply quantized values by this to
            dequantize.
        global_scale: The global scale (only present for FP8 scale dtype, None otherwise).
    """

    dequant_data: torch.Tensor
    quantized_val: torch.Tensor
    local_unscale: torch.Tensor
    global_scale: Optional[torch.Tensor]


def _parse_enum(enum_cls, value):
    if isinstance(value, str):
        for e in enum_cls:
            if e.value.upper() == value.upper():
                return e
    raise ValueError(
        f"Unknown enum value {value} for {enum_cls}; "
        f"expected one of {[e.value for e in enum_cls]}"
    )


def _parse_block_size(value):
    m, n = map(int, value.split("x"))
    return BlockSize(m, n)


@dataclass(frozen=True)
class QuantConfigBuilder:
    """Builds ``QuantConfig`` objects that may carry extensions.

    A builder knows a set of extension classes, which ``build`` instantiates
    from the options under a spec's ``"extensions"`` key, and a table of named
    presets. It may know several extensions for the same hook; the one-per-hook
    rule applies to each built config. ``extend`` returns a builder that knows
    more, and is how extension packages combine.

    Attributes:
        extensions: The extension classes ``build`` accepts, sorted by name.
        presets: Preset name -> spec, as ``build`` accepts it.
    """

    extensions: tuple[type[QuantConfigExtension], ...] = ()
    presets: Mapping[str, Mapping[str, Any]] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "extensions",
            _sorted_by_name(_check_extension_class(e) for e in self.extensions),
        )
        object.__setattr__(self, "presets", MappingProxyType(dict(self.presets)))

    def build(self, spec: Mapping[str, Any]) -> QuantConfig:
        """Build a ``QuantConfig`` from a mapping of field names to values.

        Accepts everything ``QuantConfig.from_dict`` does, plus this builder's
        extensions under the ``"extensions"`` key, each name mapped to its
        options.

        Raises:
            TypeError: if ``spec`` is not a mapping.
            ValueError: if ``spec`` has an unknown key or extension or an
                invalid value, or the fields form an invalid combination.
        """
        return _config_from_dict(spec, self.extensions)

    def build_preset(self, name: str) -> QuantConfig:
        """Build the preset called ``name``.

        Raises:
            ValueError: if no preset is called ``name``, or the preset's
                fields do not form a valid ``QuantConfig``.
        """
        if name not in self.presets:
            raise ValueError(
                f"Unknown preset {name!r}; available: {sorted(self.presets)}"
            )
        return self.build(self.presets[name])

    def extend(
        self,
        extensions: Iterable[type[QuantConfigExtension]] = (),
        presets: Mapping[str, Mapping[str, Any]] = MappingProxyType({}),
    ) -> "QuantConfigBuilder":
        """A builder that also knows the ``extensions`` classes and ``presets``.

        Raises:
            ValueError: if an extension name or a preset name is already
                known.
            TypeError: if an extension is not a ``QuantConfigExtension``
                subclass.
        """
        extensions = tuple(extensions)
        known = {e.name for e in self.extensions}
        clashes = sorted(known & {getattr(e, "name", None) for e in extensions})
        if clashes:
            raise ValueError(f"extension names already in this builder: {clashes}")
        preset_clashes = sorted(set(self.presets) & set(presets))
        if preset_clashes:
            raise ValueError(f"preset names already in this builder: {preset_clashes}")
        return QuantConfigBuilder(
            extensions=(*self.extensions, *extensions),
            presets={**self.presets, **presets},
        )


DEFAULT_QUANT_CONFIG_BUILDER: QuantConfigBuilder = QuantConfigBuilder(
    presets=FAKE_PRESETS
)

# *****************************************************************************
# * Constants and Maps                                                        *
# *****************************************************************************
SBITS, EBITS_F32, MBITS_F32 = 1, 8, 23
E8M0_EXPONENT_BIAS = F32_EXP_BIAS = 127
F32_MIN_NORMAL = 2 ** (-F32_EXP_BIAS + 1)
# Largest finite fp32 value, i.e. (2 - 2**-23) * 2**127 == FLT_MAX. Used to
# saturate ±Inf inputs to ±FLT_MAX before MX scale selection.
F32_MAX = (2 - 2**-MBITS_F32) * 2**F32_EXP_BIAS
F16_EXP_BIAS = 15
F16_MIN_NORMAL = 2 ** (-F16_EXP_BIAS + 1)
# The smallest positive fp32 and fp16 values: the least FP32 and FP16 block
# scale of a nonzero block.
F32_MIN_SUBNORMAL = 2.0**-149
F16_MIN_SUBNORMAL = 2.0**-24


# Saturation bounds of the E8Mx element formats, (2 - 2**-m) * 2**2 for m
# mantissa bits. They keep the 8-bit exponent of fp32, so the bound is a choice
# of element range rather than a property of the bit layout.
_E8M1_MAX = 6.0  # 1.5 * 2**2
_E8M2_MAX = 7  # 1.75 * 2**2
_E8M3_MAX = 7.5  #  1.875 * 2**2


ELEM_DTYPE_RANGE_MAP = {
    QuantMode.SYMM: {
        # ignore -4.0
        ElemDtype.INT3: (-3.0, 3.0),
        # ignore -8.0
        ElemDtype.INT4: (-7.0, 7.0),
        # ignore -128.0
        ElemDtype.INT8: (-127.0, 127.0),
        ElemDtype.E3M0: (-16.0, 16.0),
        ElemDtype.E2M1: (-6.0, 6.0),
        ElemDtype.E2M3: (-7.5, 7.5),
        ElemDtype.E4M3: (-448.0, 448.0),
        ElemDtype.E8M1: (-_E8M1_MAX, _E8M1_MAX),
        ElemDtype.E8M2: (-_E8M2_MAX, _E8M2_MAX),
        ElemDtype.E8M3: (-_E8M3_MAX, _E8M3_MAX),
    },
    QuantMode.ASYMM: {
        ElemDtype.INT3: (0.0, 7.0),
        ElemDtype.INT4: (0.0, 15.0),
        ElemDtype.INT8: (0.0, 255.0),
        ElemDtype.E3M0: (-16.0, 16.0),
        ElemDtype.E2M1: (-6.0, 6.0),
        ElemDtype.E2M3: (-7.5, 7.5),
        ElemDtype.E4M3: (-448.0, 448.0),
        ElemDtype.E8M1: (-_E8M1_MAX, _E8M1_MAX),
        ElemDtype.E8M2: (-_E8M2_MAX, _E8M2_MAX),
        ElemDtype.E8M3: (-_E8M3_MAX, _E8M3_MAX),
    },
    # Signed: the codes are centered on the block mean.
    QuantMode.ASYMM_MEAN_SHIFT: {
        ElemDtype.INT3: (-3.0, 3.0),
        ElemDtype.INT4: (-7.0, 7.0),
        ElemDtype.INT8: (-127.0, 127.0),
        ElemDtype.E3M0: (-16.0, 16.0),
        ElemDtype.E2M1: (-6.0, 6.0),
        ElemDtype.E2M3: (-7.5, 7.5),
        ElemDtype.E4M3: (-448.0, 448.0),
        ElemDtype.E8M1: (-_E8M1_MAX, _E8M1_MAX),
        ElemDtype.E8M2: (-_E8M2_MAX, _E8M2_MAX),
        ElemDtype.E8M3: (-_E8M3_MAX, _E8M3_MAX),
    },
}

ELEM_DTYPE_MAX_POW2_MAP = {
    ElemDtype.INT3: 2,
    ElemDtype.INT4: 4,
    ElemDtype.INT8: 64,
    ElemDtype.E2M1: 4,
    ElemDtype.E3M0: 16,
    ElemDtype.E2M3: 4,
    ElemDtype.E4M3: 256,
    # For E8MX, we still use the non-power-of-2 range
    ElemDtype.E8M1: _E8M1_MAX,
    ElemDtype.E8M2: _E8M2_MAX,
    ElemDtype.E8M3: _E8M3_MAX,
}


def elem_dtype_max_pow2(quant_mode: QuantMode, elem_dtype: ElemDtype):
    """The POW2MAX normalizer of ``elem_dtype`` under ``quant_mode``.

    The largest power of two the mode's element range reaches
    (``ELEM_DTYPE_RANGE_MAP``). An integer is unsigned in ASYMM (int8: 0..255),
    so it reaches 128 there, and signed in SYMM and ASYMM_MEAN_SHIFT
    (-127..127), where it reaches 64. Float ranges are the same in every mode,
    and E8Mx keep their non-power-of-two maximum (``ELEM_DTYPE_MAX_POW2_MAP``).
    """
    if elem_dtype in (ElemDtype.INT3, ElemDtype.INT4, ElemDtype.INT8):
        _, max_pos = ELEM_DTYPE_RANGE_MAP[quant_mode][elem_dtype]
        return 1 << (int(max_pos).bit_length() - 1)
    return ELEM_DTYPE_MAX_POW2_MAP[elem_dtype]


def pow2max_norm_exp(max_pow2) -> int:
    """`log2(max_pow2)` for the POW2MAX round-then-shift path, else 0 (fall back).

    The round-then-shift path (round amax to a power of two, then fold `/max_pow2`
    in as an integer exponent shift via `norm_exp`) is only valid when `max_pow2`
    is an integer power of two. For the E8Mx formats `max_pow2` is a
    non-power-of-two (`_E8M1_MAX`, ...), so return 0 there and the caller
    pre-divides (`delta / max_pow2`).
    """
    if isinstance(max_pow2, int) and max_pow2 > 0 and (max_pow2 & (max_pow2 - 1)) == 0:
        return max_pow2.bit_length() - 1
    return 0


ELEM_DTYPE_BITS_MAP = {
    # ebits, mbits
    ElemDtype.INT3: (0, 2),
    ElemDtype.INT4: (0, 3),
    ElemDtype.INT8: (0, 7),
    ElemDtype.E3M0: (3, 0),
    ElemDtype.E2M1: (2, 1),
    ElemDtype.E2M3: (2, 3),
    ElemDtype.E4M3: (4, 3),
    ElemDtype.E8M1: (8, 1),
    ElemDtype.E8M2: (8, 2),
    ElemDtype.E8M3: (8, 3),
}
