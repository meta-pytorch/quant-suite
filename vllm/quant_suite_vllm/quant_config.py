# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib import import_module
from typing import Any, cast

import torch
from quant_suite_recipe.definition import QuantRecipe
from quant_suite_recipe.runtime_provider import QUANT_RECIPE_RUNTIME
from vllm.model_executor.layers.quantization import (
    QuantizationMethods,
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)


QUANTIZATION_NAME = cast(QuantizationMethods, "quant_suite")
# vLLM sets ``quant_method`` on every quantization config read from a
# checkpoint; ``quant_recipe`` holds the serialized recipe.
_CONFIG_KEYS: frozenset[str] = frozenset({"quant_method", "quant_recipe"})


def _get_method_provider() -> Any:
    return import_module("quant_suite_vllm.method_provider")


@register_quantization_config(QUANTIZATION_NAME)
class QuantSuiteConfig(QuantizationConfig):
    """vLLM quantization configuration backed by a QuantSuite recipe."""

    def __init__(self, quant_recipe: QuantRecipe) -> None:
        super().__init__()
        self.quant_recipe = quant_recipe

    def __repr__(self) -> str:
        return f"QuantSuiteConfig(quant_recipe={self.quant_recipe})"

    def get_name(self) -> QuantizationMethods:
        return QUANTIZATION_NAME

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float32, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> QuantSuiteConfig:
        """Build the config from a quantization-config mapping.

        The mapping holds the serialized recipe under ``quant_recipe``, plus
        the ``quant_method`` key that vLLM sets on a checkpoint's
        quantization config. Any other key raises ``ValueError``, and a
        ``quant_recipe`` that is not a mapping raises ``TypeError``.
        """
        unknown_keys = sorted(set(config) - _CONFIG_KEYS)
        if unknown_keys:
            raise ValueError(
                f"Unknown QuantSuite quantization config keys {unknown_keys}; "
                f"allowed keys: {sorted(_CONFIG_KEYS)}"
            )
        quant_recipe_data = cls.get_from_keys(config, ["quant_recipe"])
        if not isinstance(quant_recipe_data, Mapping):
            raise TypeError(
                "quant_recipe must be a mapping, got "
                f"{type(quant_recipe_data).__name__}"
            )
        return cls(QUANT_RECIPE_RUNTIME.deserialize(quant_recipe_data))

    @classmethod
    def from_config_dict_json(cls, config_dict_json: str) -> QuantSuiteConfig:
        try:
            config = json.loads(config_dict_json)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Failed to parse quantization config JSON: {error}"
            ) from error
        if not isinstance(config, dict):
            raise ValueError("Quantization config JSON must contain an object")
        return cls.from_config(config)

    @classmethod
    def from_config_file(cls, config_file: str) -> QuantSuiteConfig:
        return cls(QUANT_RECIPE_RUNTIME.load_quant_recipe_from_yaml(config_file))

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        return _get_method_provider().get_quant_method(
            self.quant_recipe,
            self,
            layer,
            prefix,
        )

    def get_scaled_act_names(self) -> list[str]:
        return []


__all__ = ["QUANTIZATION_NAME", "QuantSuiteConfig"]
