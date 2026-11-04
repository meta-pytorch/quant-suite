# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields as dataclass_fields
from types import MappingProxyType
from typing import Any, ClassVar, Dict, Optional, Self, TypeVar

from quant_suite_core.native_presets import (
    NATIVE_PRESETS,
    NativeRoundingMode,
    PrecisionType,
)
from quant_suite_core.quant_config import (
    _parse_enum,
    DEFAULT_QUANT_CONFIG_BUILDER,
    QuantConfig,
    QuantConfigBuilder,
)
from quant_suite_core.rht_generator import HadamardMode
from quant_suite_recipe.api.definition import (
    _reject_unknown_keys,
    _unknown_keys_error,
    QuantStageRecipe,
    QuantTensorRecipe,
)


def _check_bool(owner: str, name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise TypeError(
            f"{owner}.{name} must be a bool, got {type(value).__name__} {value!r}"
        )


def _check_int(owner: str, name: str, value: object) -> None:
    # bool is a subclass of int, but True is not a size or a seed.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{owner}.{name} must be an int, got {type(value).__name__} {value!r}"
        )


def _fake_preset_shorthand(quant_config: QuantConfig) -> Optional[str]:
    """Return the shortest built-in fake preset name whose normalized config matches."""
    builder = DEFAULT_QUANT_CONFIG_BUILDER
    quant_config_dict = quant_config.to_dict()
    matching_names = [
        name
        for name in builder.presets
        if builder.build_preset(name).to_dict() == quant_config_dict
    ]
    if not matching_names:
        return None
    return sorted(matching_names, key=lambda name: (len(name), name))[0]


@dataclass
class NoQuantRecipe(QuantTensorRecipe):
    """No quantization - pass through original tensor"""

    discriminator: ClassVar[str] = "none"

    def to_dict(self) -> Dict[str, object]:
        return {self.stage_key: {"type": "none"}}

    def __str__(self) -> str:
        return "NoQuant"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NoQuantRecipe":
        """Build a NoQuantRecipe (the inner payload carries no state)."""
        _reject_unknown_keys(f"{cls.discriminator!r} quantize config", data, {"type"})
        return cls()


@dataclass
class FakeQuantRecipe(QuantTensorRecipe):
    """
    Fake (QDQ) quantization with a ``QuantConfig``.

    Attributes:
        quant_config: The quantization configuration specifying scale dtype,
            element dtype, block size, rounding modes, etc.
        use_triton: Optional flag to use Triton for quantization.
                    If None, defers to fake_quantize's default behavior (True).
    """

    discriminator: ClassVar[str] = "fake"

    quant_config: QuantConfig
    use_triton: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.use_triton is not None:
            _check_bool(type(self).__name__, "use_triton", self.use_triton)

    @property
    def block_size(self) -> tuple[int, ...]:
        return (self.quant_config.block_size.m, self.quant_config.block_size.n)

    def to_dict(self) -> Dict[str, object]:
        if self.use_triton is None:
            preset_name = _fake_preset_shorthand(self.quant_config)
            if preset_name is not None:
                return {self.stage_key: f"fake_{preset_name}"}

        value: Dict[str, object] = {
            "type": "fake",
            "quant_config": self.quant_config.to_dict(),
        }
        if self.use_triton is not None:
            value["use_triton"] = self.use_triton
        return {self.stage_key: value}

    def __str__(self) -> str:
        triton_str = (
            f", use_triton={self.use_triton}" if self.use_triton is not None else ""
        )
        return f"FakeQuant({self.quant_config}{triton_str})"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FakeQuantRecipe":
        """Build a FakeQuantRecipe from the inner payload, with the built-in
        presets and no config extensions."""
        return cls.from_dict_with(data, DEFAULT_QUANT_CONFIG_BUILDER)

    @classmethod
    def from_dict_with(
        cls, data: Mapping[str, Any], config_builder: QuantConfigBuilder
    ) -> "FakeQuantRecipe":
        """Build a FakeQuantRecipe from the inner payload.

        ``quant_config`` is either a preset name of ``config_builder`` or a dict
        of ``QuantConfig`` fields, which ``config_builder`` builds.
        """
        _reject_unknown_keys(
            f"{cls.discriminator!r} quantize config",
            data,
            {"type", "quant_config", "use_triton"},
        )
        raw = data.get("quant_config")
        if isinstance(raw, str) and raw in config_builder.presets:
            quant_config = config_builder.build_preset(raw)
        elif isinstance(raw, Mapping):
            quant_config = config_builder.build(raw)
        else:
            raise ValueError(
                "Fake quantization 'quant_config' must be a preset name (one of "
                f"{sorted(config_builder.presets)}) or a dict of QuantConfig fields; "
                f"got {raw!r}. A whole stage may also be given as "
                "'fake_<preset>'."
            )
        return cls(
            quant_config=quant_config,
            use_triton=data.get("use_triton"),
        )


@dataclass
class NativeQuantRecipe(QuantTensorRecipe):
    """
    Native quantization configuration.

    Supports NVFP4 (block_size=16) and MXFP4 (block_size=32 or 16) with
    configurable rounding modes.
    """

    discriminator: ClassVar[str] = "native"

    # Precision format: NVFP4, MX4_16, MX4_32
    precision: PrecisionType = field(default=PrecisionType.NVFP4)

    # Scale rounding mode: nearest, floor, even, stochastic, ceil.
    # NVFP4 always rounds its FP8 block scales to nearest, so floor, even and
    # ceil have no effect there, and stochastic is rejected.
    scale_round_mode: NativeRoundingMode = field(default=NativeRoundingMode.NEAREST)

    @property
    def block_size(self) -> tuple[int, ...] | None:
        if self.precision == PrecisionType.MX4_32:
            return (1, 32)
        if self.precision in (
            PrecisionType.NVFP4,
            PrecisionType.MX4_16,
        ):
            return (1, 16)
        return None

    def __post_init__(self) -> None:
        if not isinstance(self.scale_round_mode, NativeRoundingMode):
            self.scale_round_mode = _parse_enum(
                NativeRoundingMode, self.scale_round_mode
            )
        if (
            self.precision == PrecisionType.NVFP4
            and self.scale_round_mode == NativeRoundingMode.STOCHASTIC
        ):
            raise ValueError(
                f"Native {self.precision.value} quantization does not support "
                f"scale_round_mode={self.scale_round_mode.value!r}: its FP8 block "
                "scales are always rounded to nearest"
            )

    def to_dict(self) -> Dict[str, object]:
        """Convert to dict with Enum values serialized and type discriminator.

        Manually builds dict to ensure Enum values are properly serialized.
        """
        return {
            self.stage_key: {
                "precision": self.precision.value,
                "scale_round_mode": self.scale_round_mode.value,
                "type": "native",
            }
        }

    def __str__(self) -> str:
        return f"NativeQuant({self.precision.value},{self.scale_round_mode.value})"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NativeQuantRecipe":
        """Deserialize from dict with Enum reconstruction.

        Args:
            data: Dictionary with "precision" and "scale_round_mode" fields

        Returns:
            NativeQuantRecipe instance with Enums properly reconstructed
        """
        _reject_unknown_keys(
            f"{cls.discriminator!r} quantize config",
            data,
            {"type", "precision", "scale_round_mode"},
        )
        precision_value = data.get("precision")
        scale_round_mode_value = data.get("scale_round_mode")

        if precision_value is None:
            raise ValueError(
                f"Native quant config requires a 'precision' field; got {data!r}. "
                f"Valid precisions: {[p.value for p in PrecisionType]}"
            )

        # Convert string to Enum
        if isinstance(precision_value, str):
            precision = PrecisionType(precision_value)
        else:
            precision = precision_value

        # A missing scale_round_mode defaults to nearest.
        if scale_round_mode_value is None:
            return cls(precision=precision)
        return cls(precision=precision, scale_round_mode=scale_round_mode_value)


@dataclass
class RHTRecipe(QuantStageRecipe):
    """Random Hadamard Transform pipeline-stage configuration.

    Applies a (random) Hadamard transform to the tensor as a standalone pipeline
    stage, independent of quantization. Pre-transform modes (``right``/``left``)
    and their inverse post-transform halves (``right_inv``/``left_inv``) compose
    around a quantize stage: e.g. ``[right, quantize, right_inv]`` is the RHT
    round-trip ``Q(X @ R) @ R^T``, and ``[right, right_inv]`` (no quantize) is
    the identity since the Hadamard matrix ``R`` is orthogonal.

    Attributes:
        mode: Hadamard mode (right / right_inv / left / left_inv).
        block_size: Hadamard block size (the transform matrix dimension). Must be
            >= 2 for a real transform -- a 1x1 (or empty) Hadamard block performs
            no mixing, so it is rejected; use ``HadamardMode.NONE`` for a no-op
            stage instead.
        seed: Optional RNG seed for the random sign matrix (reproducibility).
    """

    stage_key: ClassVar[str] = "rht"

    mode: HadamardMode
    block_size: int = 0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        _check_int(type(self).__name__, "block_size", self.block_size)
        if self.seed is not None:
            _check_int(type(self).__name__, "seed", self.seed)
        # A Hadamard block must contain >= 2 elements to mix anything: block_size
        # 0 is empty (and would divide-by-zero in apply_hadamard) and a 1x1 block
        # is just a global +/-1 sign (identity in a round-trip). Reject both for a
        # real mode rather than silently producing a no-op; HadamardMode.NONE is
        # the explicit no-op.
        if self.mode != HadamardMode.NONE and self.block_size <= 1:
            raise ValueError(
                "RHTRecipe.block_size must be >= 2 for a non-NONE RHT (a 1x1 or "
                f"empty block performs no transform); got {self.block_size}. Use "
                "HadamardMode.NONE for a no-op stage."
            )

    def is_noop(self) -> bool:
        return self.mode == HadamardMode.NONE

    def to_dict(self) -> Dict[str, object]:
        value: Dict[str, object] = {"type": self.mode.value}
        if self.block_size > 0:
            value["block_size"] = self.block_size
        if self.seed is not None:
            value["seed"] = self.seed
        return {self.stage_key: value}

    def __str__(self) -> str:
        return f"RHT({self.mode.value}, block_size={self.block_size}, seed={self.seed})"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RHTRecipe":
        """Parse an RHT stage from its inner dict ({type, block_size, seed}).

        ``type`` is the Hadamard mode and is accepted case-insensitively
        (e.g. ``RIGHT_INV`` or ``right_inv``).
        """
        _reject_unknown_keys(
            f"{cls.stage_key!r} stage config", data, {"type", "block_size", "seed"}
        )
        if "type" not in data:
            raise ValueError(
                f"{cls.stage_key!r} stage config {dict(data)!r} has no 'type'. "
                f"Set 'type' to one of {[m.value for m in HadamardMode]}"
            )
        mode_value = data["type"]
        mode = (
            mode_value
            if isinstance(mode_value, HadamardMode)
            else HadamardMode(str(mode_value).lower())
        )
        return cls(
            mode=mode,
            block_size=data.get("block_size", 0),
            seed=data.get("seed"),
        )


@dataclass
class PipelineRecipe:
    """Ordered pipeline of quant stages for a single tensor.

    Wraps a ``list[QuantStageRecipe]``; an empty pipeline is the identity (no
    quantization). Exposes ``len``/iteration/indexing so call sites can treat it
    as a list of stages -- ``len`` also drives truthiness, so
    ``bool(recipe.weight)`` stays "has any stage". ``to_dict`` serializes it;
    ``QuantRecipeRuntime.deserialize`` parses it as part of a complete recipe.
    """

    stages: list[QuantStageRecipe] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.stages)

    def __iter__(self) -> Iterator[QuantStageRecipe]:
        return iter(self.stages)

    def __getitem__(self, index: int) -> QuantStageRecipe:
        return self.stages[index]

    def __str__(self) -> str:
        return "[" + ", ".join(str(stage) for stage in self.stages) + "]"

    def is_noop(self) -> bool:
        return all(stage.is_noop() for stage in self.stages)

    @property
    def has_native_quant(self) -> bool:
        """True if any stage is a native (low-precision) quantize op."""
        return any(
            isinstance(stage, QuantTensorRecipe)
            and stage.discriminator == NativeQuantRecipe.discriminator
            for stage in self.stages
        )

    def to_dict(self) -> Dict[str, object]:
        return {"stages": [stage.to_dict() for stage in self.stages]}


_QUANTIZE_STAGE_KEY = QuantTensorRecipe.stage_key
_STAGE_KEYS: tuple[str, ...] = (
    QuantTensorRecipe.stage_key,
    RHTRecipe.stage_key,
)
# Keys that identify a *bare* (unwrapped) quantize config dict. Used to tell a
# bare quantize node ({"type": ...}) apart from a misspelled stage wrapper
# ({"quantizee": ...}) so the latter fails loudly. A bare node that carries
# quantize fields but no "type" is rejected when the node is parsed.
_QUANTIZE_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "type",
        "quant_config",
        "use_triton",
        "precision",
        "scale_round_mode",
    }
)


def _to_raw_stage_nodes(value: Any) -> list[Any]:
    """Shape a tensor's raw config into a list of stage nodes (structural only).

    Purely structural -- it decides the list shape and never parses a node (that
    is ``QuantRecipeDeserializer.deserialize_stage_recipe``'s job). A scalar or a
    bare quantize dict becomes one ``{"quantize": value}`` node; a bare list is
    passed through. A single non-quantize stage (e.g. a bare ``{"rht": ...}``)
    must use list syntax, so it raises, and a misspelled/unrecognized stage key
    also raises so a typo is never silently wrapped into a no-op quantize stage.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, QuantStageRecipe):
        raise ValueError(
            "A bare QuantStageRecipe is not a valid loose-config input; use list "
            "syntax or construct PipelineRecipe([...]) directly"
        )
    if isinstance(value, Mapping):
        if _QUANTIZE_STAGE_KEY in value:
            return [value]
        stage_keys = [stage_key for stage_key in _STAGE_KEYS if stage_key in value]
        if not stage_keys:
            # No explicit stage-wrapper key (e.g. "quantize"/"rht"): the only
            # thing allowed here is a *bare* quantize dict such as
            # {"type": "fake", "quant_config": {...}}. Require at least one
            # recognized quantize field so a misspelled stage key ("quantizee")
            # fails loudly instead of being silently wrapped into a no-op
            # quantize stage.
            if not any(key in _QUANTIZE_CONFIG_KEYS for key in value):
                raise ValueError(
                    f"Unrecognized stage config {value!r}: expected a stage "
                    f"wrapper (one of {list(_STAGE_KEYS)}) or a bare quantize dict."
                )
            return [{_QUANTIZE_STAGE_KEY: value}]
        # A single non-quantize stage node (e.g. {"rht": ...}) requires list
        # syntax.
        raise ValueError(
            "Only quantization stages can be configured without list syntax; "
            f"got stage wrapper {value!r}"
        )
    return [{_QUANTIZE_STAGE_KEY: value}]


@dataclass
class QuantLinearRecipe:
    """Quantization recipe for a linear layer.

    Each tensor (``activation``/``weight``/``bias``) is a ``PipelineRecipe`` -- an
    ordered pipeline of stage configs (``QuantStageRecipe``): a quantize op
    (``QuantTensorRecipe``) and/or a transform (``RHTRecipe``). An empty pipeline
    is the identity (no quantization). See
    ``quant_suite_recipe.implementation.quant_pipeline`` for how the pipelines
    are turned into runtime ``Pipeline``s.
    """

    activation: PipelineRecipe = field(default_factory=PipelineRecipe)
    weight: PipelineRecipe = field(default_factory=PipelineRecipe)
    bias: PipelineRecipe = field(default_factory=PipelineRecipe)
    init_quantized_weight: bool = field(default=True)

    def __post_init__(self) -> None:
        _check_bool(
            type(self).__name__, "init_quantized_weight", self.init_quantized_weight
        )

    def __str__(self) -> str:
        return (
            f"QuantLinearRecipe(act={self.activation}, "
            f"weight={self.weight}, bias={self.bias}, "
            f"init_quantized_weight={self.init_quantized_weight})"
        )

    @classmethod
    def pipeline_fields(cls) -> tuple[str, ...]:
        return ("activation", "weight", "bias")

    def to_dict(self) -> Dict[str, object]:
        result: Dict[str, object] = {
            field_name: getattr(self, field_name).to_dict()
            for field_name in self.pipeline_fields()
        }
        result["init_quantized_weight"] = self.init_quantized_weight
        return result


@dataclass
class QuantMoERecipe:
    """Quantization recipe for a MoE (Mixture of Experts) layer.

    MoE layers have two separate GEMM operations with different activations:
    1. First GEMM: a1 @ w13 -> gate+up activations (before activation function)
    2. Second GEMM: a2 @ w2 -> final output (after activation and element-wise multiply)

    This recipe allows independent quantization configuration for:
    - a1: Input activations for the first GEMM with w13 (gate+up projection)
    - a2: Intermediate activations for the second GEMM with w2 (down projection)
    - w13: Gate+up projection weights (combined gate and up weights)
    - w2: Down projection weights

    For most cases, a1 and a2 can use the same configuration.
    Use separate configurations when you need different quantization strategies
    for the input vs intermediate activations.
    """

    a1: PipelineRecipe = field(default_factory=PipelineRecipe)
    a2: PipelineRecipe = field(default_factory=PipelineRecipe)
    w13: PipelineRecipe = field(default_factory=PipelineRecipe)
    w2: PipelineRecipe = field(default_factory=PipelineRecipe)
    init_quantized_weight: bool = field(default=True)

    def __post_init__(self) -> None:
        _check_bool(
            type(self).__name__, "init_quantized_weight", self.init_quantized_weight
        )

    def __str__(self) -> str:
        return (
            f"QuantMoERecipe(a1={self.a1}, a2={self.a2}, "
            f"w13={self.w13}, w2={self.w2}, "
            f"init_quantized_weight={self.init_quantized_weight})"
        )

    @classmethod
    def pipeline_fields(cls) -> tuple[str, ...]:
        return ("a1", "a2", "w13", "w2")

    def to_dict(self) -> Dict[str, object]:
        result: Dict[str, object] = {
            field_name: getattr(self, field_name).to_dict()
            for field_name in self.pipeline_fields()
        }
        result["init_quantized_weight"] = self.init_quantized_weight
        return result


@dataclass
class QuantAttentionRecipe:
    """Quantization recipe for an Attention layer.

    Attention layers quantize Q, K, V tensors and attention weights dynamically
    at runtime during forward pass. This recipe allows independent quantization
    configuration for each of these components.

    For KDA (linear attention) layers, the recurrent_state field controls
    quantization of the recurrent hidden state, which is analogous to the
    KV cache in softmax attention.

    Args:
        q: Quantization recipe for query tensor
        k: Quantization recipe for key tensor
        v: Quantization recipe for value tensor
        attn_weights: Quantization recipe for attention weights
            (output of softmax before multiplying with V)
        recurrent_state: Quantization recipe for KDA recurrent state
            (analogous to KV cache in softmax attention)
    """

    q: PipelineRecipe = field(default_factory=PipelineRecipe)
    k: PipelineRecipe = field(default_factory=PipelineRecipe)
    v: PipelineRecipe = field(default_factory=PipelineRecipe)
    attn_weights: PipelineRecipe = field(default_factory=PipelineRecipe)
    recurrent_state: PipelineRecipe = field(default_factory=PipelineRecipe)

    def __str__(self) -> str:
        return (
            f"QuantAttentionRecipe(q={self.q}, k={self.k}, "
            f"v={self.v}, attn_weights={self.attn_weights}, "
            f"recurrent_state={self.recurrent_state})"
        )

    @classmethod
    def pipeline_fields(cls) -> tuple[str, ...]:
        return ("q", "k", "v", "attn_weights", "recurrent_state")

    def requires_custom_attention_backend(self) -> bool:
        return any(
            bool(getattr(self, field_name))
            for field_name in ("q", "k", "v", "attn_weights")
        )

    def to_dict(self) -> Dict[str, object]:
        result: Dict[str, object] = {
            field_name: getattr(self, field_name).to_dict()
            for field_name in self.pipeline_fields()
        }
        return result


def _compile_layer_pattern(section: str, pattern: str) -> re.Pattern[str]:
    """Compile a layer-name pattern of ``section``, raising ``ValueError`` when it
    is not a valid regular expression."""
    try:
        return re.compile(pattern)
    except re.error as error:
        raise ValueError(f"Invalid regex {pattern!r} in {section}: {error}") from error


@dataclass
class QuantRecipe:
    """
    Complete quantization recipe for a model.

    The linear_recipe maps regex patterns to QuantLinearRecipe configurations.
    The moe_recipe maps regex patterns to QuantMoERecipe configurations for MoE layers.
    The attention_recipe maps regex patterns to QuantAttentionRecipe configurations for attention layers.
    This allows different quantization configs for different layers based on
    layer name matching.

    Example:
        {
            "linear.*": QuantLinearRecipe(...),  # All linear layers
            ".*self_attn\\..*_proj": QuantLinearRecipe(...),  # Attention projections
        }

        {
            ".*moe.*": QuantMoERecipe(...),  # All MoE layers
            ".*experts.*": QuantMoERecipe(...),  # All expert layers
        }

        {
            ".*attn.*": QuantAttentionRecipe(...),  # All attention layers
        }

    A section that is omitted, whether in direct construction or in input to
    ``QuantRecipeRuntime.deserialize``, is an empty mapping and leaves that
    module category unaffected. ``QuantRecipeRuntime.deserialize`` reconstructs
    a recipe from ``to_dict`` output or the nested form.
    """

    linear_recipe: Dict[str, QuantLinearRecipe] = field(default_factory=dict)
    moe_recipe: Dict[str, QuantMoERecipe] = field(default_factory=dict)
    attention_recipe: Dict[str, QuantAttentionRecipe] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for section in ("linear_recipe", "moe_recipe", "attention_recipe"):
            for pattern in getattr(self, section):
                _compile_layer_pattern(section, pattern)

    def __str__(self) -> str:
        linear_recipes_str = ", ".join(
            f"{pattern}: {recipe}" for pattern, recipe in self.linear_recipe.items()
        )
        moe_recipes_str = ", ".join(
            f"{pattern}: {recipe}" for pattern, recipe in self.moe_recipe.items()
        )
        attention_recipes_str = ", ".join(
            f"{pattern}: {recipe}" for pattern, recipe in self.attention_recipe.items()
        )
        return f"QuantRecipe(linear=[{linear_recipes_str}], moe=[{moe_recipes_str}], attention=[{attention_recipes_str}])"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert recipe to dictionary with fully qualified paths as keys.

        Calls the custom to_dict() method on QuantLinearRecipe which in turn
        calls to_dict() on each QuantTensorRecipe to ensure proper Enum serialization.

        Returns:
            Dictionary with flattened keys using fully qualified paths
        """
        result: Dict[str, Any] = {}

        # Process linear_recipe
        # Key format: linear_recipe::<regex_pattern>::<field_name>
        # Using :: as separator to avoid ambiguity with dots in regex patterns
        for regex_pattern, quant_linear_recipe in self.linear_recipe.items():
            for field_name, field_value in quant_linear_recipe.to_dict().items():
                qualified_key = f"linear_recipe::{regex_pattern}::{field_name}"
                result[qualified_key] = field_value

        # Process moe_recipe
        # Key format: moe_recipe::<regex_pattern>::<field_name>
        for regex_pattern, quant_linear_recipe in self.moe_recipe.items():
            for field_name, field_value in quant_linear_recipe.to_dict().items():
                qualified_key = f"moe_recipe::{regex_pattern}::{field_name}"
                result[qualified_key] = field_value

        # Process attention_recipe
        # Key format: attention_recipe::<regex_pattern>::<field_name>
        for regex_pattern, quant_attention_recipe in self.attention_recipe.items():
            for field_name, field_value in quant_attention_recipe.to_dict().items():
                qualified_key = f"attention_recipe::{regex_pattern}::{field_name}"
                result[qualified_key] = field_value

        return result

    def get_linear_recipe(self, layer_name: str = "") -> QuantLinearRecipe:
        """
        Get the appropriate QuantLinearRecipe for a given layer name.

        Matches layer_name against regex patterns in linear_recipe dictionary.
        Returns an identity recipe with empty pipelines if no match is found.

        Patterns are tried in insertion order with ``re.match``, which anchors
        at the start of ``layer_name`` but not at the end, and the first match
        wins. A pattern that is not a valid regex raises ``ValueError``.

        Args:
            layer_name: Name of the layer to match. If empty or not provided,
                returns an identity recipe when no pattern matches.

        Returns:
            QuantLinearRecipe matching the layer name

        Examples:
            >>> recipe = QuantRecipe(linear_recipe={"linear.*": QuantLinearRecipe(...)})
            >>> linear_recipe = recipe.get_linear_recipe("linear.fc1")
            >>> # Returns the QuantLinearRecipe for "linear.*" pattern
        """
        for pattern, recipe in self.linear_recipe.items():
            if _compile_layer_pattern("linear_recipe", pattern).match(layer_name):
                return recipe

        # If no match found, return an empty (identity) linear recipe
        return QuantLinearRecipe()

    def has_linear_recipes(self) -> bool:
        """
        Check if this QuantRecipe has any linear quantization recipes configured.

        Returns True if linear_recipe dictionary is non-empty.

        Returns:
            True if linear recipes are configured, False otherwise
        """
        return bool(self.linear_recipe)

    def get_moe_recipe(self, layer_name: str = "") -> QuantMoERecipe:
        """
        Get the appropriate QuantMoERecipe for a given MoE layer name.

        Matches layer_name against regex patterns in moe_recipe dictionary.
        Returns an identity recipe with empty pipelines if no match is found.

        Patterns are tried in insertion order with ``re.match``, which anchors
        at the start of ``layer_name`` but not at the end, and the first match
        wins. A pattern that is not a valid regex raises ``ValueError``.

        Args:
            layer_name: Name of the MoE layer to match. If empty or not provided,
                returns an identity recipe when no pattern matches.

        Returns:
            QuantMoERecipe matching the layer name

        Examples:
            >>> recipe = QuantRecipe(moe_recipe={".*moe.*": QuantMoERecipe(...)})
            >>> moe_recipe = recipe.get_moe_recipe("model.layers.0.block_sparse_moe")
            >>> # Returns the QuantMoERecipe for ".*moe.*" pattern
        """
        for pattern, recipe in self.moe_recipe.items():
            if _compile_layer_pattern("moe_recipe", pattern).match(layer_name):
                return recipe

        # If no match found, return default QuantMoERecipe
        return QuantMoERecipe()

    def has_moe_recipes(self) -> bool:
        """
        Check if this QuantRecipe has any MoE quantization recipes configured.

        Returns True if moe_recipe dictionary is non-empty.

        Returns:
            True if MoE recipes are configured, False otherwise
        """
        return bool(self.moe_recipe)

    def get_attention_recipe(self, layer_name: str = "") -> QuantAttentionRecipe:
        """
        Get the appropriate QuantAttentionRecipe for a given attention layer name.

        Matches layer_name against regex patterns in attention_recipe dictionary.
        Returns an identity recipe with empty pipelines if no match is found.

        Patterns are tried in insertion order with ``re.match``, which anchors
        at the start of ``layer_name`` but not at the end, and the first match
        wins. A pattern that is not a valid regex raises ``ValueError``.

        Args:
            layer_name: Name of the attention layer to match. If empty or not provided,
                returns an identity recipe when no pattern matches.

        Returns:
            QuantAttentionRecipe matching the layer name

        Examples:
            >>> recipe = QuantRecipe(attention_recipe={".*attn.*": QuantAttentionRecipe(...)})
            >>> attn_recipe = recipe.get_attention_recipe("model.layers.0.self_attn")
            >>> # Returns the QuantAttentionRecipe for ".*attn.*" pattern
        """
        for pattern, recipe in self.attention_recipe.items():
            if _compile_layer_pattern("attention_recipe", pattern).match(layer_name):
                return recipe

        # If no match found, return default QuantAttentionRecipe
        return QuantAttentionRecipe()

    def has_attention_recipes(self) -> bool:
        """
        Check if this QuantRecipe has any attention quantization recipes configured.

        Returns True if attention_recipe dictionary is non-empty.

        Returns:
            True if attention recipes are configured, False otherwise
        """
        return bool(self.attention_recipe)


@dataclass(frozen=True)
class QuantModuleRecipeTypes:
    """Concrete module recipe classes used for model-level deserialization."""

    linear: type[QuantLinearRecipe] = QuantLinearRecipe
    moe: type[QuantMoERecipe] = QuantMoERecipe
    attention: type[QuantAttentionRecipe] = QuantAttentionRecipe

    def by_section(
        self,
    ) -> Mapping[
        str,
        type[QuantLinearRecipe] | type[QuantMoERecipe] | type[QuantAttentionRecipe],
    ]:
        return MappingProxyType(
            {
                "linear_recipe": self.linear,
                "moe_recipe": self.moe,
                "attention_recipe": self.attention,
            }
        )


_DEFAULT_MODULE_RECIPE_TYPES = QuantModuleRecipeTypes()
_ModuleRecipeT = TypeVar(
    "_ModuleRecipeT",
    QuantLinearRecipe,
    QuantMoERecipe,
    QuantAttentionRecipe,
)


def _is_model_recipe_key(
    key: object, allowed_fields: Mapping[str, frozenset[str]]
) -> bool:
    if not isinstance(key, str):
        return False
    if key in allowed_fields:
        return True
    for section, fields_for_section in allowed_fields.items():
        prefix = f"{section}::"
        if key.startswith(prefix):
            remainder = key[len(prefix) :]
            if "::" not in remainder:
                return False
            pattern, field_name = remainder.split("::", 1)
            return bool(pattern) and field_name in fields_for_section
    return False


def _validate_model_recipe_data(
    data: Mapping[str, Any],
    module_recipe_types: QuantModuleRecipeTypes,
) -> None:
    allowed_fields = {
        section: frozenset(
            field.name for field in dataclass_fields(recipe_type) if field.init
        )
        for section, recipe_type in module_recipe_types.by_section().items()
    }
    invalid_keys = [
        key for key in data if not _is_model_recipe_key(key, allowed_fields)
    ]
    if invalid_keys:
        raise _unknown_keys_error(
            "a complete model-level quantization recipe",
            invalid_keys,
            [
                *allowed_fields,
                *(f"{section}::<pattern>::<field>" for section in allowed_fields),
            ],
        )

    has_nested_sections = any(section in data for section in allowed_fields)
    has_flattened_fields = any(isinstance(key, str) and "::" in key for key in data)
    if has_nested_sections and has_flattened_fields:
        raise ValueError(
            "A complete model-level quantization recipe cannot mix nested and "
            "flattened representations"
        )

    if not has_nested_sections:
        return
    for section in allowed_fields:
        if section not in data:
            continue
        section_data = data[section]
        if not isinstance(section_data, Mapping):
            raise ValueError(f"Expected {section!r} to be a mapping")
        for pattern, recipe_data in section_data.items():
            if not isinstance(pattern, str) or not isinstance(recipe_data, Mapping):
                raise ValueError(
                    f"Expected each {section!r} entry to map a string pattern to "
                    f"a recipe mapping; got {pattern!r}: {recipe_data!r}"
                )


def _validate_recipe_type(recipe_type: type[QuantTensorRecipe]) -> str:
    """Check that ``recipe_type`` can be registered and return its discriminator."""
    if not (
        isinstance(recipe_type, type) and issubclass(recipe_type, QuantTensorRecipe)
    ):
        raise TypeError(f"Recipe type must extend QuantTensorRecipe: {recipe_type!r}")
    discriminator = getattr(recipe_type, "discriminator", None)
    if not discriminator:
        raise ValueError("Recipe discriminator must not be empty")
    return discriminator


@dataclass(frozen=True)
class QuantRecipeDeserializer:
    """Immutable dispatcher for one explicit set of tensor recipe types."""

    _recipe_types_by_discriminator: Mapping[str, type[QuantTensorRecipe]]
    _module_recipe_types: QuantModuleRecipeTypes
    _config_builder: QuantConfigBuilder

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_recipe_types_by_discriminator",
            MappingProxyType(dict(self._recipe_types_by_discriminator)),
        )

    @classmethod
    def from_recipe_types(
        cls,
        recipe_types: Iterable[type[QuantTensorRecipe]],
        module_recipe_types: QuantModuleRecipeTypes,
        config_builder: QuantConfigBuilder,
    ) -> Self:
        recipe_types_by_discriminator: dict[str, type[QuantTensorRecipe]] = {}
        for recipe_type in recipe_types:
            discriminator = _validate_recipe_type(recipe_type)
            if discriminator in recipe_types_by_discriminator:
                raise ValueError(
                    f"Recipe deserializer already has a parser for {discriminator!r}"
                )
            recipe_types_by_discriminator[discriminator] = recipe_type
        return cls(
            recipe_types_by_discriminator,
            module_recipe_types,
            config_builder,
        )

    def _fake_preset_shorthands(self) -> list[str]:
        return [f"fake_{name}" for name in self._config_builder.presets]

    def deserialize_tensor_recipe(self, data: Any) -> QuantTensorRecipe:  # noqa: C901
        if isinstance(data, QuantTensorRecipe):
            # A typed recipe needs no parsing, but validate it against this
            # deserializer's registrations so it cannot bypass supported-type checks.
            quant_type = data.discriminator
            expected_type = self._recipe_types_by_discriminator.get(quant_type)
            if expected_type is None:
                raise ValueError(
                    f"Invalid config type: {quant_type!r}. "
                    "Expected one of "
                    f"{list(self._recipe_types_by_discriminator)}"
                )
            if not isinstance(data, expected_type):
                raise TypeError(
                    f"Recipe discriminator {quant_type!r} is bound to "
                    f"{expected_type.__name__}, not {type(data).__name__}"
                )
            return data
        if isinstance(data, Mapping) and QuantTensorRecipe.stage_key in data:
            _reject_unknown_keys(
                "a quantize stage", data, (QuantTensorRecipe.stage_key,)
            )
            data = data[QuantTensorRecipe.stage_key]

        if isinstance(data, str):
            shorthand = data.strip().lower()
            if shorthand == "none":
                data = {"type": NoQuantRecipe.discriminator}
            elif shorthand.startswith("fake_"):
                preset_name = shorthand[len("fake_") :]
                if preset_name not in self._config_builder.presets:
                    raise ValueError(
                        f"Unknown fake quant preset '{shorthand}'. "
                        f"Available: {sorted(self._fake_preset_shorthands())}"
                    )
                data = {
                    "type": FakeQuantRecipe.discriminator,
                    "quant_config": preset_name,
                }
            elif shorthand.startswith("native_"):
                preset_name = shorthand[len("native_") :]
                if preset_name not in NATIVE_PRESETS:
                    raise ValueError(
                        f"Unknown native quant preset '{shorthand}'. "
                        f"Available: {['native_' + k for k in sorted(NATIVE_PRESETS)]}"
                    )
                preset = NATIVE_PRESETS[preset_name]
                data = {
                    "type": NativeQuantRecipe.discriminator,
                    "precision": preset["precision"],
                    "scale_round_mode": preset["scale_round_mode"],
                }
            else:
                raise ValueError(
                    f"Unknown quant shorthand '{shorthand}'. "
                    f"Expected 'none', 'fake_<preset>', or 'native_<preset>'. "
                    f"Fake presets: {sorted(self._fake_preset_shorthands())}. "
                    f"Native presets: {['native_' + k for k in sorted(NATIVE_PRESETS)]}"
                )

        if not isinstance(data, Mapping):
            raise ValueError(
                f"Expected a quant config mapping or string shorthand; got {data!r}"
            )

        if "type" not in data:
            raise ValueError(
                f"'{QuantTensorRecipe.stage_key}' stage config {dict(data)!r} has "
                "no 'type'. Set 'type' to one of "
                f"{list(self._recipe_types_by_discriminator)}, or use a "
                "string shorthand: 'none', 'fake_<preset>' or 'native_<preset>'."
            )
        quant_type = data["type"]
        if not isinstance(quant_type, str):
            raise ValueError(
                f"Quantization type discriminator must be a string; got {quant_type!r}"
            )
        recipe_type = self._recipe_types_by_discriminator.get(quant_type)
        if recipe_type is None:
            raise ValueError(
                f"Invalid config type: {quant_type!r}. "
                "Expected one of "
                f"{list(self._recipe_types_by_discriminator)}"
            )
        recipe = recipe_type.from_dict_with(data, self._config_builder)
        if not isinstance(recipe, recipe_type):
            raise TypeError(
                f"Parser for quant type {quant_type!r} returned "
                f"{type(recipe).__name__}; expected {recipe_type.__name__}"
            )
        return recipe

    def deserialize_stage_recipe(self, data: Any) -> QuantStageRecipe:
        if isinstance(data, QuantTensorRecipe):
            return self.deserialize_tensor_recipe(data)
        if isinstance(data, QuantStageRecipe):
            return data
        if not isinstance(data, Mapping):
            raise ValueError(
                f"Expected quant stage mapping or QuantStageRecipe instance; "
                f"got {data!r}"
            )

        stage_keys = [stage_key for stage_key in _STAGE_KEYS if stage_key in data]
        if len(stage_keys) != 1:
            raise ValueError(
                f"Expected exactly one quant stage key from {list(_STAGE_KEYS)}; "
                f"got {data!r}"
            )
        _reject_unknown_keys("a quant stage", data, _STAGE_KEYS)
        stage_key = stage_keys[0]
        payload = data[stage_key]
        if stage_key == RHTRecipe.stage_key:
            if not isinstance(payload, Mapping):
                raise ValueError(f"Expected RHT config mapping; got {payload!r}")
            return RHTRecipe.from_dict(payload)
        return self.deserialize_tensor_recipe(payload)

    def deserialize_pipeline_recipe(
        self,
        value: Any,
    ) -> PipelineRecipe:
        if isinstance(value, PipelineRecipe):
            raw_nodes = list(value)
        elif isinstance(value, Mapping) and "stages" in value:
            _reject_unknown_keys("a pipeline config", value, ("stages",))
            raw_nodes = value["stages"]
        else:
            raw_nodes = _to_raw_stage_nodes(value)
        if not isinstance(raw_nodes, list):
            raise ValueError(f"Pipeline stages must be a list; got {raw_nodes!r}")
        stages = [self.deserialize_stage_recipe(node) for node in raw_nodes]
        return PipelineRecipe(stages)

    def _deserialize_module_recipe(
        self,
        data: Mapping[str, Any],
        recipe_type: type[_ModuleRecipeT],
        where: str = "",
    ) -> _ModuleRecipeT:
        _reject_unknown_keys(
            where or recipe_type.__name__,
            data,
            (
                recipe_field.name
                for recipe_field in dataclass_fields(recipe_type)
                if recipe_field.init
            ),
        )
        pipeline_fields = frozenset(recipe_type.pipeline_fields())
        kwargs: dict[str, Any] = {
            field_name: self.deserialize_pipeline_recipe(data.get(field_name))
            for field_name in pipeline_fields
        }
        kwargs.update(
            {
                recipe_field.name: data[recipe_field.name]
                for recipe_field in dataclass_fields(recipe_type)
                if recipe_field.init
                and recipe_field.name not in pipeline_fields
                and recipe_field.name in data
            }
        )
        return recipe_type(**kwargs)

    def deserialize_quant_recipe(
        self,
        data: Mapping[str, Any],
    ) -> QuantRecipe:  # noqa: C901
        if not isinstance(data, Mapping):
            raise ValueError(
                "Expected a complete model-level quantization recipe mapping; "
                f"got {data!r}"
            )
        _validate_model_recipe_data(data, self._module_recipe_types)
        copied_data: Dict[str, Any] = deepcopy(dict(data))
        if any(
            key.startswith("linear_recipe::")
            or key.startswith("moe_recipe::")
            or key.startswith("attention_recipe::")
            for key in copied_data
        ):
            nested_data: Dict[str, Any] = {
                "linear_recipe": {},
                "moe_recipe": {},
                "attention_recipe": {},
            }
            for key, value in copied_data.items():
                section = None
                for candidate in (
                    "linear_recipe",
                    "moe_recipe",
                    "attention_recipe",
                ):
                    prefix = f"{candidate}::"
                    if key.startswith(prefix):
                        section = candidate
                        remainder = key[len(prefix) :]
                        if "::" in remainder:
                            pattern, field_name = remainder.split("::", 1)
                            section_data = nested_data[section]
                            if pattern not in section_data:
                                section_data[pattern] = {}
                            section_data[pattern][field_name] = value
                        break
                if section is None:
                    nested_data[key] = value
            copied_data = nested_data

        module_types = self._module_recipe_types
        return QuantRecipe(
            linear_recipe={
                pattern: self._deserialize_module_recipe(
                    value, module_types.linear, f"linear_recipe pattern {pattern!r}"
                )
                for pattern, value in copied_data.get("linear_recipe", {}).items()
            },
            moe_recipe={
                pattern: self._deserialize_module_recipe(
                    value, module_types.moe, f"moe_recipe pattern {pattern!r}"
                )
                for pattern, value in copied_data.get("moe_recipe", {}).items()
            },
            attention_recipe={
                pattern: self._deserialize_module_recipe(
                    value,
                    module_types.attention,
                    f"attention_recipe pattern {pattern!r}",
                )
                for pattern, value in copied_data.get("attention_recipe", {}).items()
            },
        )
