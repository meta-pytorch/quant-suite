# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from importlib import import_module
from importlib.util import find_spec
from typing import Any


_KIMI_K3_KDA_MODULE = "vllm.models.kimi_k3.nvidia.kda"


def _has_kimi_k3_kda_module() -> bool:
    module_parts: list[str] = []
    for part in _KIMI_K3_KDA_MODULE.split("."):
        module_parts.append(part)
        if find_spec(".".join(module_parts)) is None:
            return False
    return True


def register_kda_quant_method(quant_config_type: type[Any]) -> bool:
    """Register the KDA adapter matching the installed vLLM API."""
    if not _has_kimi_k3_kda_module():
        return import_module(
            "quant_suite_vllm.kda.constructor_patch"
        ).register_constructor_patch_kda_quant_method(quant_config_type)
    return import_module(
        "quant_suite_vllm.kda.pluggable_layer"
    ).register_pluggable_kda_quant_method(quant_config_type)
