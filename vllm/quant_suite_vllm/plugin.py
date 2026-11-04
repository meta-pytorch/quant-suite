# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM plugin entry point for QuantSuite."""

import os
from importlib import import_module

from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend


_registered = False
_ATTENTION_BACKEND_ENV = "QUANT_SUITE_ATTN_BACKEND"
_MHA_BACKEND = "quant_suite_vllm.attention.backend.QuantSuiteAttentionBackend"
_MLA_BACKEND = "quant_suite_vllm.attention.mla.backend.QuantSuiteMLAAttentionBackend"


def register_attention_backend() -> None:
    """Register the selected QuantSuite implementation as vLLM's CUSTOM backend."""
    backend = (
        _MLA_BACKEND
        if os.environ.get(_ATTENTION_BACKEND_ENV) == "CUSTOM_MLA"
        else _MHA_BACKEND
    )
    register_backend(AttentionBackendEnum.CUSTOM, backend)


def register() -> None:
    """Register the ``quant_suite`` quantization method and its vLLM hooks.

    Resolves the recipe runtime by importing
    ``quant_suite_recipe.runtime_provider``, imports ``QuantSuiteConfig`` (which
    registers the method), selects the ``CUSTOM`` attention backend
    implementation and installs the KDA adapters.
    Idempotent; vLLM calls it through the ``vllm.general_plugins`` entry point.
    """
    global _registered
    if _registered:
        return

    import_module("quant_suite_recipe.runtime_provider")
    quant_config = import_module("quant_suite_vllm.quant_config")
    register_attention_backend()
    import_module("quant_suite_vllm.kda").register_kda_quant_method(
        quant_config.QuantSuiteConfig
    )
    _registered = True


__all__ = ["register", "register_attention_backend"]
