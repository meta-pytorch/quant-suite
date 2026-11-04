# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any

import torch
from quant_suite_core.quant_config import (
    DEFAULT_QUANT_CONFIG_BUILDER,
    QuantConfigBuilder,
)
from quant_suite_recipe.api.definition import QuantStageRecipe, QuantTensorRecipe
from quant_suite_recipe.api.implementation import (
    QuantizedTensor,
    QuantizeTensorHandler,
    QuantTensorRecipeRegistration,
)
from quant_suite_recipe.definition.quant_recipe import (
    _DEFAULT_MODULE_RECIPE_TYPES,
    _validate_recipe_type,
    FakeQuantRecipe,
    NativeQuantRecipe,
    NoQuantRecipe,
    PipelineRecipe,
    QuantAttentionRecipe,
    QuantLinearRecipe,
    QuantModuleRecipeTypes,
    QuantMoERecipe,
    QuantRecipe,
    QuantRecipeDeserializer,
    RHTRecipe,
)
from quant_suite_recipe.implementation.attention import QuantizedAttention
from quant_suite_recipe.implementation.linear import QuantizedLinear
from quant_suite_recipe.implementation.moe import QuantizedMoE
from quant_suite_recipe.implementation.quant_pipeline import (
    Pipeline,
    PipelineStage,
    QuantizeStage,
    RHTStage,
)
from quant_suite_recipe.implementation.quantize_tensor import (
    fake_quantize_tensor,
    native_quantize_tensor,
    no_quantize_tensor,
)
from quant_suite_recipe.yaml_to_quant_recipe import _load_quant_recipe_from_yaml


QUANT_RECIPE_REGISTRATIONS: tuple[QuantTensorRecipeRegistration, ...] = (
    (NoQuantRecipe, no_quantize_tensor),
    (FakeQuantRecipe, fake_quantize_tensor),
    (NativeQuantRecipe, native_quantize_tensor),
)
"""The built-in ``none``, ``fake`` and ``native`` tensor recipe types."""


class QuantRecipeRuntime:
    """Immutable deserialization and execution runtime for complete registrations.

    Every argument defaults to the built-in value, so ``QuantRecipeRuntime()``
    is the built-in runtime.

    Args:
        registrations: Every tensor recipe type the runtime accepts, each
            paired with its quantize-tensor handler. To add types, pass
            ``(*QUANT_RECIPE_REGISTRATIONS, *extra)``.
        config_builder: Parses every ``QuantConfig`` the runtime deserializes,
            so its presets and extension tokens are what a recipe may use.
        module_recipe_types: Module recipe classes used for deserialization.
        quantized_linear_type: The ``QuantizedLinear`` class
            ``prepare_linear`` builds.
        quantized_attention_type: The ``QuantizedAttention`` class
            ``prepare_attention`` builds.

    Raises:
        TypeError: if a registered type does not extend ``QuantTensorRecipe``.
        ValueError: if a registered type has an empty discriminator, or two
            registrations share a discriminator.
    """

    _registrations: tuple[QuantTensorRecipeRegistration, ...]
    _config_builder: QuantConfigBuilder
    _recipe_types_by_discriminator: Mapping[str, type[QuantTensorRecipe]]
    _handlers_by_discriminator: Mapping[str, QuantizeTensorHandler]
    _deserializer: QuantRecipeDeserializer
    _quantized_linear_type: type[QuantizedLinear]
    _quantized_attention_type: type[QuantizedAttention]

    def __init__(
        self,
        registrations: Iterable[
            QuantTensorRecipeRegistration
        ] = QUANT_RECIPE_REGISTRATIONS,
        *,
        config_builder: QuantConfigBuilder = DEFAULT_QUANT_CONFIG_BUILDER,
        module_recipe_types: QuantModuleRecipeTypes = _DEFAULT_MODULE_RECIPE_TYPES,
        quantized_linear_type: type[QuantizedLinear] = QuantizedLinear,
        quantized_attention_type: type[QuantizedAttention] = QuantizedAttention,
    ) -> None:
        self._registrations = tuple(registrations)
        self._config_builder = config_builder
        handlers_by_discriminator: dict[str, QuantizeTensorHandler] = {}
        recipe_types_by_discriminator: dict[str, type[QuantTensorRecipe]] = {}
        for recipe_type, handler in self._registrations:
            discriminator = _validate_recipe_type(recipe_type)
            if discriminator in handlers_by_discriminator:
                raise ValueError(
                    "Quant recipe runtime already has an implementation for "
                    f"discriminator {discriminator!r}"
                )
            handlers_by_discriminator[discriminator] = handler
            recipe_types_by_discriminator[discriminator] = recipe_type

        self._handlers_by_discriminator = MappingProxyType(handlers_by_discriminator)
        self._recipe_types_by_discriminator = MappingProxyType(
            recipe_types_by_discriminator
        )
        self._deserializer = QuantRecipeDeserializer.from_recipe_types(
            recipe_types_by_discriminator.values(),
            module_recipe_types,
            config_builder,
        )
        self._quantized_linear_type = quantized_linear_type
        self._quantized_attention_type = quantized_attention_type

    @property
    def registrations(self) -> tuple[QuantTensorRecipeRegistration, ...]:
        """The tensor recipe registrations this runtime was built with."""
        return self._registrations

    @property
    def config_builder(self) -> QuantConfigBuilder:
        """The ``QuantConfigBuilder`` this runtime parses every config with."""
        return self._config_builder

    def deserialize(self, data: Mapping[str, Any]) -> QuantRecipe:
        """Deserialize one complete model-level quantization recipe."""
        return self._deserializer.deserialize_quant_recipe(data)

    def load_quant_recipe_from_yaml(self, yaml_path: str) -> QuantRecipe:
        """Load and deserialize one complete model-level YAML recipe."""
        return _load_quant_recipe_from_yaml(yaml_path, self.deserialize)

    def _dispatch_quantize_tensor_handler(
        self,
        recipe: QuantTensorRecipe,
    ) -> QuantizeTensorHandler:
        discriminator = recipe.discriminator
        handler = self._handlers_by_discriminator.get(discriminator)
        if handler is None:
            raise ValueError(
                f"Quant recipe discriminator {discriminator!r} is not registered "
                f"with this runtime; expected one of "
                f"{list(self._handlers_by_discriminator)}"
            )
        expected_type = self._recipe_types_by_discriminator[discriminator]
        if not isinstance(recipe, expected_type):
            raise TypeError(
                f"Quant recipe discriminator {discriminator!r} is bound to "
                f"{expected_type.__name__}, not {type(recipe).__name__}"
            )
        return handler

    def quantize_tensor(
        self,
        tensor: torch.Tensor,
        tensor_name: str,
        recipe: QuantTensorRecipe,
        *,
        transpose: bool = False,
        return_quant_info: bool = False,
    ) -> QuantizedTensor:
        """Apply one registered tensor-quantization recipe without a pipeline."""
        handler = self._dispatch_quantize_tensor_handler(recipe)
        return handler(
            tensor,
            tensor_name,
            recipe,
            transpose,
            return_quant_info,
        )

    def create_pipeline(
        self,
        recipe: QuantStageRecipe | PipelineRecipe,
        *,
        tensor_name: str = "",
    ) -> Pipeline:
        """Create a pipeline using implementations selected by this runtime."""
        specs = [recipe] if isinstance(recipe, QuantStageRecipe) else recipe
        stages: list[PipelineStage] = []
        for spec in specs:
            if spec.is_noop():
                continue
            if isinstance(spec, RHTRecipe):
                stages.append(
                    RHTStage(
                        mode=spec.mode,
                        block_size=spec.block_size,
                        seed=spec.seed,
                    )
                )
            elif isinstance(spec, QuantTensorRecipe):
                stages.append(
                    QuantizeStage(
                        spec,
                        self._dispatch_quantize_tensor_handler(spec),
                        tensor_name,
                    )
                )
            else:
                raise ValueError(f"Unsupported pipeline stage spec: {spec!r}")
        return Pipeline(*stages)

    def prepare_linear(self, recipe: QuantLinearRecipe) -> QuantizedLinear:
        """Prepare the executable state owned by one Linear module."""
        return self._quantized_linear_type(
            recipe=recipe,
            activation_pipeline=self.create_pipeline(
                recipe.activation,
                tensor_name="activation",
            ),
            weight_pipeline=self.create_pipeline(
                recipe.weight,
                tensor_name="weight",
            ),
            bias_pipeline=self.create_pipeline(
                recipe.bias,
                tensor_name="bias",
            ),
        )

    def prepare_moe(self, recipe: QuantMoERecipe) -> QuantizedMoE:
        """Prepare the executable state owned by one MoE module."""
        return QuantizedMoE(
            recipe=recipe,
            a1_pipeline=self.create_pipeline(recipe.a1, tensor_name="a1"),
            a2_pipeline=self.create_pipeline(recipe.a2, tensor_name="a2"),
            w13_pipeline=self.create_pipeline(recipe.w13, tensor_name="w13"),
            w2_pipeline=self.create_pipeline(recipe.w2, tensor_name="w2"),
        )

    def prepare_attention(self, recipe: QuantAttentionRecipe) -> QuantizedAttention:
        """Prepare the executable state owned by one Attention module."""
        return self._quantized_attention_type(
            recipe=recipe,
            q_pipeline=self.create_pipeline(recipe.q, tensor_name="q"),
            k_pipeline=self.create_pipeline(recipe.k, tensor_name="k"),
            v_pipeline=self.create_pipeline(recipe.v, tensor_name="v"),
            attn_weights_pipeline=self.create_pipeline(
                recipe.attn_weights,
                tensor_name="attn_weights",
            ),
            recurrent_state_pipeline=self.create_pipeline(
                recipe.recurrent_state,
                tensor_name="recurrent_state",
            ),
        )

    def apply_pipeline(
        self,
        pipeline: Pipeline,
        tensor: torch.Tensor,
    ) -> QuantizedTensor:
        """Apply an executable pipeline to a tensor."""
        return pipeline(tensor)
