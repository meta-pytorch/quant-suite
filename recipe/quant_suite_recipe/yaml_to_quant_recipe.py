# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Utility functions for creating QuantRecipe from YAML configuration files.
"""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml
from quant_suite_recipe.definition.quant_recipe import QuantRecipe


def _require_nonempty_section(section: str, cfg: Any) -> None:
    """Reject a present-but-empty recipe section.

    An empty (``{}``) or ``None`` section almost always means a layer pattern was
    forgotten; silently defaulting it to a no-op recipe for all layers would hide
    that mistake. To intentionally leave a module type unquantized, omit its
    section entirely.
    """
    if cfg is None or cfg == {}:
        raise ValueError(
            f"'{section}' is present but empty. Specify at least one "
            "'<layer-pattern>: <recipe>' mapping, or omit the section entirely to "
            "leave those layers unquantized."
        )


def _load_quant_recipe_from_yaml(
    yaml_path: str,
    deserialize: Callable[[Mapping[str, Any]], QuantRecipe],
) -> QuantRecipe:
    """
    Load a QuantRecipe from a YAML configuration file.

    Args:
        yaml_path: Path to YAML file containing quantization configuration
        deserialize: Parses the loaded mapping into a ``QuantRecipe``, normally
            the runtime's ``deserialize``.

    Returns:
        QuantRecipe: Parsed quantization recipe

    Raises:
        FileNotFoundError: If YAML file does not exist
        ValueError: If YAML format is invalid or the recipe has an unknown key

    Example YAML format (regex-based layer configuration):
        ```yaml
        linear_recipe:
          linear.*:
            activation:
              type: none
            weight:
              type: fake
              quant_config:
                elem_dtype: e2m1
                elem_round_mode: nearest
                block_size: 1x16
                scale_dtype: fp8
                scale_norm_mode: amax
                scale_round_mode: none
                global_scale_granularity: tensorwise
                global_scale_dtype: fp32
            bias:
              type: none
        moe_recipe:
          .*:
            a1: fake_nvfp4
            w13: native_nvfp4
        ```

    """
    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    try:
        with open(yaml_file, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML file {yaml_path}: {error}") from error

    if config is None:
        raise ValueError(f"Empty YAML file: {yaml_path}")
    if not isinstance(config, Mapping):
        raise ValueError(f"YAML recipe must contain a mapping: {yaml_path}")
    if not config:
        raise ValueError(f"YAML recipe mapping must not be empty: {yaml_path}")

    for section in ("linear_recipe", "moe_recipe", "attention_recipe"):
        if section in config:
            _require_nonempty_section(section, config[section])

    return deserialize(config)
