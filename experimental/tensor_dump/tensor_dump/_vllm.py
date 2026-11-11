# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The vLLM state the dump reads.

vLLM is imported inside each function, so this module and the package's
recipe runtime load in environments without vLLM.
"""

from __future__ import annotations

from typing import Any


def forward_context() -> Any | None:
    """vLLM's current forward context, or ``None`` outside a forward."""
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not is_forward_context_available():
        return None
    return get_forward_context()


def attention_class() -> type[Any]:
    """vLLM's decoder attention layer class."""
    from vllm.model_executor.layers.attention import Attention

    return Attention


def tensor_parallel() -> tuple[int, int] | None:
    """``(tp_rank, tp_size)``, or ``None`` when vLLM's groups are not set up."""
    from vllm.distributed.parallel_state import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        model_parallel_is_initialized,
    )

    if not model_parallel_is_initialized():
        return None
    return get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()


def settings_sources() -> tuple[Any | None, bool | None]:
    """vLLM's current config, or ``None`` when none is current, and the value
    of ``VLLM_USE_V2_MODEL_RUNNER``, which is ``None`` when it is unset."""
    import vllm.envs as envs
    from vllm.config import get_current_vllm_config_or_none

    return get_current_vllm_config_or_none(), envs.VLLM_USE_V2_MODEL_RUNNER
