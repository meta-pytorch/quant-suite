# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The per-process dump session, created from the environment on first use."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

import torch
from tensor_dump import _step
from tensor_dump._attention import _AttentionDump
from tensor_dump._moe import _MoEDump
from tensor_dump._naming import _NameResolver
from tensor_dump._options import (
    _ATTENTION_ENV,
    _DIR_ENV,
    _DumpOptions,
    _MOE_ENV,
    _options_from_environ,
    _RANKS_ENV,
)
from tensor_dump._step import _StepTracker, _VllmSettings
from tensor_dump._writer import _Writer

logger: logging.Logger = logging.getLogger(__name__)

# KV-cache dtypes that store keys and values unquantized.
_UNQUANTIZED_KV_CACHE_DTYPES = frozenset({"auto", "bfloat16", "float16"})


class _Session:
    """The options, step tracker, writer, name resolver, and attention and
    MoE hooks of one process.

    Raises:
        RuntimeError: if vLLM's V2 model runner or async scheduling is
            enabled, attention layers are selected while vLLM's prefix
            caching is enabled or its KV cache is quantized, or MoE layers
            are selected and this rank writes but its tensor-parallel rank 0
            does not.
        NotImplementedError: if speculative decoding is configured, or MoE
            layers are selected under context parallelism.
    """

    def __init__(self, options: _DumpOptions, settings: _VllmSettings) -> None:
        if settings.v2_model_runner:
            raise RuntimeError(
                "tensor_dump does not support vLLM's V2 model runner, whose "
                "warmup runs real forwards that cannot be told apart from "
                "requests; launch vLLM with VLLM_USE_V2_MODEL_RUNNER=0"
            )
        if settings.async_scheduling:
            raise RuntimeError(
                "tensor_dump does not support vLLM's async scheduling, which "
                "schedules each forward before the previous step's sampled "
                "tokens are processed, so a request that has stopped runs one "
                "more forward and the dump records a step whose token is never "
                "returned; start vLLM with --no-async-scheduling"
            )
        if settings.speculative:
            raise NotImplementedError(
                "tensor_dump does not support speculative decoding: draft-model "
                "forwards would take step ids"
            )
        pattern = options.attention
        if pattern is not None and settings.prefix_caching:
            raise RuntimeError(
                f"{_ATTENTION_ENV} is set, and attention dumps hold the keys and "
                "values of each step's new tokens, so every token must be "
                "computed; vLLM's prefix caching skips cached prompt tokens. "
                "Start vLLM with --no-enable-prefix-caching"
            )
        kv_cache_dtype = settings.kv_cache_dtype
        if (
            pattern is not None
            and kv_cache_dtype is not None
            and kv_cache_dtype not in _UNQUANTIZED_KV_CACHE_DTYPES
        ):
            raise RuntimeError(
                f"{_ATTENTION_ENV} is set, and vLLM's KV cache dtype is "
                f"{kv_cache_dtype!r}: attention then reads quantized keys and "
                "values and may get a quantized query, so the recorded tensors "
                "would not reproduce its output. Start vLLM with "
                "--kv-cache-dtype auto"
            )
        if options.moe is not None and settings.context_parallel:
            raise NotImplementedError(
                f"{_MOE_ENV} is set, and tensor_dump does not record MoE layers "
                "under context parallelism"
            )
        if options.moe is not None:
            _check_moe_ranks(options)
        self.options = options
        self.settings = settings
        self.resolver = _NameResolver()
        self.writer = _Writer(options)
        self.tracker = _StepTracker(
            options,
            on_step=self._on_step,
            on_selected_step=self.writer.begin_step,
            total_bytes=self.writer.total_bytes,
            context_parallel=settings.context_parallel is True,
        )
        self.attention: _AttentionDump | None = None
        self.moe: _MoEDump | None = None
        visitors: list[Callable[[torch.nn.Module], None]] = []
        if pattern is not None:
            self.attention = _AttentionDump(
                pattern, options.attention_tensors, self.tracker, self.writer
            )
            visitors.append(self.attention.visit)
        if options.moe is not None:
            self.moe = _MoEDump(
                options.moe, options.moe_tensors, self.tracker, self.writer
            )
            visitors.append(self.moe.visit)
        if visitors:
            self.resolver.scan(_visit_all(visitors))

    def _on_step(self, step: int) -> None:
        self.resolver.finish_from_second_step(step)
        if self.moe is not None:
            self.moe.check_hooked(step)


def _check_moe_ranks(options: _DumpOptions) -> None:
    """Refuse a rank filter under which this rank writes MoE records but its
    tensor-parallel rank 0, which alone writes the tensors identical on every
    rank, does not.

    vLLM numbers ranks with the tensor-parallel rank innermost, so a group's
    rank 0 is this rank minus its tensor-parallel rank.
    """
    if options.ranks is None:
        return
    rank = _step._distributed_rank()[0]
    tensor_parallel = _step._tensor_parallel()
    leader = rank - tensor_parallel[0] if tensor_parallel else 0
    if options.selects_rank(rank) and not options.selects_rank(leader):
        raise RuntimeError(
            f"{_MOE_ENV} is set and {_RANKS_ENV} selects rank {rank} but not "
            f"rank {leader}, its tensor-parallel rank 0, which alone writes the "
            "MoE tensors identical on every rank; add rank "
            f"{leader} to {_RANKS_ENV}"
        )


def _visit_all(
    visitors: list[Callable[[torch.nn.Module], None]],
) -> Callable[[torch.nn.Module], None]:
    def visit(module: torch.nn.Module) -> None:
        for visitor in visitors:
            visitor(module)

    return visit


_session: _Session | None = None
_loaded: bool = False
_warned_disabled: bool = False


def _get() -> _Session | None:
    """The session, or ``None`` when dumping is disabled.

    The vLLM settings are read when the session is created, which is when
    the first linear executor is built, while vLLM loads the model.

    Raises:
        ValueError: on the first call, for a malformed environment variable.
        RuntimeError, NotImplementedError: for an unsupported vLLM setting; see
            ``_Session``.
    """
    global _session, _loaded
    if not _loaded:
        options = _options_from_environ(os.environ)
        _session = (
            None if options is None else _Session(options, _step._vllm_settings())
        )
        _loaded = True
    return _session


def _warn_disabled() -> None:
    """Log once that a recipe has ``dump`` stages while dumping is disabled."""
    global _warned_disabled
    if not _warned_disabled:
        _warned_disabled = True
        logger.warning("recipe has dump stages but %s is unset", _DIR_ENV)


def _reset() -> None:
    """Drop the session so the next ``_get`` reads the environment again.

    For tests.
    """
    global _session, _loaded, _warned_disabled
    session = _session
    if session is not None:
        session.resolver.finish()
        attention = session.attention
        if attention is not None:
            attention.remove()
        moe = session.moe
        if moe is not None:
            moe.remove()
        session.writer.close()
    _session = None
    _loaded = False
    _warned_disabled = False
