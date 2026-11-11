# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Step ids from vLLM's forward context, and prefill/decode classification.

vLLM creates one forward context object per model forward. A forward whose
context has no attention metadata is a dummy forward (profiling, warmup): it
gets no step id and writes nothing. Every other forward gets the next step id,
counted per process, so all ranks that run the same forwards agree on it. A
forward keeps its step id for as long as its context object lives, even when
another context is entered and left inside it.
"""

from __future__ import annotations

import importlib
import logging
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from tensor_dump._options import _DumpOptions

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RequestInfo:
    """One request of a step, from the attention metadata.

    Attributes:
        request: Index of the request within the step's batch.
        q_len: Query tokens of the request in this step.
        kv_len: Keys and values the request attends to, cached and new.
        q_start_pos: Absolute position of the request's first query token.
        token_start: Row of the request's first token in the step's tokens.
        phase: ``"decode"`` for one query token after position 0, else
            ``"prefill"``.
    """

    request: int
    q_len: int
    kv_len: int
    q_start_pos: int
    token_start: int
    phase: str


@dataclass(frozen=True)
class _StepInfo:
    """One real forward.

    ``phase``, ``requests`` and ``num_actual_tokens`` are read only for steps
    that pass the step-id and rank filters; otherwise they are empty.
    """

    step: int
    phase: str
    selected: bool
    requests: tuple[_RequestInfo, ...]
    num_actual_tokens: int


@dataclass(frozen=True)
class _VllmSettings:
    """The vLLM settings the dump checks.

    Attributes:
        v2_model_runner: Whether vLLM's V2 model runner is enabled.
        speculative: Whether speculative decoding is configured; ``None`` when
            no vLLM config is current.
        max_num_seqs: The scheduler's request limit per step; ``None`` when no
            vLLM config is current.
    """

    v2_model_runner: bool = False
    speculative: bool | None = None
    max_num_seqs: int | None = None


def _forward_context() -> Any | None:
    """vLLM's current forward context, or ``None`` outside a forward."""
    return importlib.import_module("tensor_dump._vllm").forward_context()


def _tensor_parallel() -> tuple[int, int] | None:
    """``(tp_rank, tp_size)``, or ``None`` when vLLM's groups are not set up."""
    return importlib.import_module("tensor_dump._vllm").tensor_parallel()


def _vllm_settings() -> _VllmSettings:
    config, v2_env = importlib.import_module("tensor_dump._vllm").settings_sources()
    return _settings_from(config, v2_env)


def _settings_from(config: Any | None, v2_env: bool | None) -> _VllmSettings:
    """The settings from vLLM's config, or from ``VLLM_USE_V2_MODEL_RUNNER``
    (``v2_env``, ``None`` when unset) when no config is current.

    The config decides whether the V2 model runner is used, since vLLM may
    select it with the variable unset.
    """
    if config is None:
        return _VllmSettings(v2_model_runner=v2_env is True)
    return _VllmSettings(
        v2_model_runner=bool(getattr(config, "use_v2_model_runner", v2_env is True)),
        speculative=config.speculative_config is not None,
        max_num_seqs=int(config.scheduler_config.max_num_seqs),
    )


def _distributed_rank() -> tuple[int, int]:
    """``(rank, world_size)`` of ``torch.distributed``, or ``(0, 1)``."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _request_phase(q_len: int, q_start_pos: int) -> str:
    return "decode" if q_len == 1 and q_start_pos > 0 else "prefill"


def _classify(requests: tuple[_RequestInfo, ...]) -> str:
    """``"prefill"`` or ``"decode"`` when every request agrees, else ``"mixed"``."""
    phases = {request.phase for request in requests}
    return phases.pop() if len(phases) == 1 else "mixed"


def _has_request_fields(metadata: Any) -> bool:
    return (
        getattr(metadata, "query_start_loc", None) is not None
        and getattr(metadata, "seq_lens", None) is not None
    )


def _parse_requests(metadata: Any) -> tuple[_RequestInfo, ...]:
    """The requests of one attention-metadata object.

    Reads ``query_start_loc`` and ``seq_lens``; ``num_reqs`` and
    ``decode_offset`` are used when present, and derived otherwise.

    Raises:
        RuntimeError: if the metadata lacks ``query_start_loc`` or ``seq_lens``.
    """
    if not _has_request_fields(metadata):
        raise RuntimeError(
            f"tensor_dump needs query_start_loc and seq_lens in the attention "
            f"metadata, which {type(metadata).__name__} does not provide"
        )
    num_reqs = getattr(metadata, "num_reqs", None)
    if num_reqs is None:
        num_reqs = metadata.query_start_loc.numel() - 1
    num_reqs = int(num_reqs)
    starts = metadata.query_start_loc[: num_reqs + 1].tolist()
    kv_lens = metadata.seq_lens[:num_reqs].tolist()
    q_lens = [int(starts[i + 1]) - int(starts[i]) for i in range(num_reqs)]
    offsets = getattr(metadata, "decode_offset", None)
    if offsets is None:
        q_start_positions = [int(kv_lens[i]) - q_lens[i] for i in range(num_reqs)]
    else:
        q_start_positions = offsets[:num_reqs].tolist()
    return tuple(
        _RequestInfo(
            request=index,
            q_len=q_lens[index],
            kv_len=int(kv_lens[index]),
            q_start_pos=int(q_start_positions[index]),
            token_start=int(starts[index]),
            phase=_request_phase(q_lens[index], int(q_start_positions[index])),
        )
        for index in range(num_reqs)
    )


def _step_metadata(attn_metadata: dict[str, Any]) -> Any:
    """The first metadata entry that describes the step's requests."""
    for metadata in attn_metadata.values():
        if _has_request_fields(metadata):
            return metadata
    types = sorted({type(metadata).__name__ for metadata in attn_metadata.values()})
    raise RuntimeError(
        f"tensor_dump needs query_start_loc and seq_lens in the attention "
        f"metadata, which none of {types} provides"
    )


@dataclass
class _Forward:
    """The step of one forward context, and its parsed request metadata."""

    context: weakref.ref[Any]
    info: _StepInfo | None
    requests: dict[int, tuple[_RequestInfo, ...]] = field(default_factory=dict)


class _StepTracker:
    """Assigns step ids to forwards and decides which steps are written.

    Args:
        options: The dump options.
        on_first_step: Called once, at the first real forward.
        on_selected_step: Called at the start of every step that is written.
    """

    def __init__(
        self,
        options: _DumpOptions,
        on_first_step: Callable[[], None],
        on_selected_step: Callable[[_StepInfo], None],
    ) -> None:
        self._options = options
        self._on_first_step = on_first_step
        self._on_selected_step = on_selected_step
        # Keyed by id() and checked against a weak reference, since vLLM's
        # forward contexts are not hashable.
        self._forwards: dict[int, _Forward] = {}
        self._next_step = 0
        self._rank: int | None = None
        self._steps_written = 0
        self._cap_logged = False

    def current(self) -> _StepInfo | None:
        """The step of the running forward; ``None`` for a dummy forward or
        outside a forward.

        Raises:
            NotImplementedError: for dual-batch-overlap micro-batches.
            RuntimeError: when the forward runs with CUDA graphs, or its
                attention metadata does not describe its requests.
        """
        context = _forward_context()
        if context is None:
            return None
        forward = self._lookup(context)
        if forward is None:
            forward = _Forward(weakref.ref(context, self._forget(id(context))), None)
            forward.info = self._start_step(context, forward)
            self._forwards[id(context)] = forward
        return forward.info

    def layer_requests(self, layer_name: str) -> tuple[_RequestInfo, ...]:
        """The requests of the current step in ``layer_name``'s metadata."""
        context = _forward_context()
        forward = None
        metadata = None
        if context is not None:
            forward = self._lookup(context)
            if forward is not None:
                metadata = context.attn_metadata.get(layer_name)
        if forward is None or metadata is None:
            raise RuntimeError(
                f"the forward context has no attention metadata for layer "
                f"{layer_name!r}"
            )
        return self._requests(forward, metadata)

    def _lookup(self, context: Any) -> _Forward | None:
        forward = self._forwards.get(id(context))
        if forward is None or forward.context() is not context:
            return None
        return forward

    def _forget(self, key: int) -> Callable[[weakref.ref[Any]], None]:
        def forget(_: weakref.ref[Any]) -> None:
            self._forwards.pop(key, None)

        return forget

    @staticmethod
    def _requests(forward: _Forward, metadata: Any) -> tuple[_RequestInfo, ...]:
        requests = forward.requests.get(id(metadata))
        if requests is None:
            requests = _parse_requests(metadata)
            forward.requests[id(metadata)] = requests
        return requests

    def _start_step(self, context: Any, forward: _Forward) -> _StepInfo | None:
        attn_metadata = context.attn_metadata
        if isinstance(attn_metadata, list):
            raise NotImplementedError(
                "tensor_dump does not support dual-batch-overlap micro-batches"
            )
        if not attn_metadata:
            return None
        mode = getattr(context, "cudagraph_runtime_mode", None)
        if mode is not None and mode.name != "NONE":
            raise RuntimeError(
                f"tensor_dump requires eager execution, but the forward runs with "
                f"CUDA graph mode {mode.name}; start vLLM with --enforce-eager"
            )
        step = self._next_step
        self._next_step += 1
        if self._rank is None:
            self._rank = _distributed_rank()[0]
            self._on_first_step()
        unselected = _StepInfo(step, "", False, (), 0)
        if not (
            self._options.selects_step(step) and self._options.selects_rank(self._rank)
        ):
            return unselected
        if self._options.max_steps and self._steps_written >= self._options.max_steps:
            if not self._cap_logged:
                self._cap_logged = True
                logger.info(
                    "tensor_dump wrote %d steps, the QUANT_SUITE_TENSOR_DUMP_MAX_STEPS "
                    "cap; nothing more is written",
                    self._steps_written,
                )
            return unselected
        metadata = _step_metadata(attn_metadata)
        requests = self._requests(forward, metadata)
        phase = _classify(requests)
        if phase not in self._options.phases:
            return unselected
        self._steps_written += 1
        num_actual_tokens = getattr(metadata, "num_actual_tokens", None)
        info = _StepInfo(
            step=step,
            phase=phase,
            selected=True,
            requests=requests,
            num_actual_tokens=(
                sum(request.q_len for request in requests)
                if num_actual_tokens is None
                else int(num_actual_tokens)
            ),
        )
        self._on_selected_step(info)
        return info
