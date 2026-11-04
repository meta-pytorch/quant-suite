#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from importlib import metadata


def main() -> None:
    plugins = tuple(metadata.entry_points(group="vllm.general_plugins"))
    entry_point = next(
        (plugin for plugin in plugins if plugin.name == "quant_suite"),
        None,
    )
    if entry_point is None:
        raise RuntimeError("quant_suite vLLM plugin entry point is not installed")
    print(
        "plugin:",
        entry_point.name,
        entry_point.value,
        entry_point.dist.name,
    )
    entry_point.load()()

    from vllm.model_executor.layers.quantization import get_quantization_config

    config_type = get_quantization_config("quant_suite")
    print("config:", config_type.__module__ + "." + config_type.__qualname__)

    from quant_suite_recipe.runtime_provider import QUANT_RECIPE_RUNTIME

    runtimes = tuple(metadata.entry_points(group="quant_suite.recipe_runtimes"))
    for runtime_entry_point in runtimes:
        print(
            "recipe runtime entry point:",
            runtime_entry_point.name,
            runtime_entry_point.value,
            runtime_entry_point.dist.name if runtime_entry_point.dist else None,
        )
    selected = next(
        (ep.name for ep in runtimes if ep.load() is QUANT_RECIPE_RUNTIME),
        "built-in",
    )
    print("recipe runtime:", selected)


if __name__ == "__main__":
    main()
