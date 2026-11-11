# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Step ids from vLLM's forward context, and prefill/decode classification.

vLLM creates one forward context object per model forward. A forward whose
context has no attention metadata, or whose attention metadata assigns no token
a KV-cache slot, is a dummy forward (profiling, warmup): it gets no step id and
writes nothing. Every other forward gets the next step id, counted per process,
so all ranks that run the same forwards agree on it. A forward keeps its step
id for as long as its context object lives, even when another context is
entered and left inside it.
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
        prefix_caching: Whether vLLM's prefix caching is enabled; ``None`` when
            no vLLM config is current.
        context_parallel: Whether decode or prefill context parallelism is
            configured; ``None`` when no vLLM config is current.
    """

    v2_model_runner: bool = False
    speculative: bool | None = None
    prefix_caching: bool | None = None
    context_parallel: bool | None = None


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
    parallel = getattr(config, "parallel_config", None)
    return _VllmSettings(
        v2_model_runner=bool(getattr(config, "use_v2_model_runner", v2_env is True)),
        speculative=config.speculative_config is not None,
        prefix_caching=bool(config.cache_config.enable_prefix_caching),
        context_parallel=any(
            int(getattr(parallel, name, None) or 1) > 1
            for name in (
                "decode_context_parallel_size",
                "prefill_context_parallel_size",
            )
        ),
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


_FLASHINFER_COUNTS = (
    "num_decodes",
    "num_decode_tokens",
    "num_prefills",
    "num_prefill_tokens",
)


def _is_flashinfer(metadata: Any) -> bool:
    """Whether ``metadata`` has the fields of vLLM's FlashInfer metadata."""
    return (
        all(getattr(metadata, name, None) is not None for name in _FLASHINFER_COUNTS)
        and hasattr(metadata, "prefill")
        and hasattr(metadata, "decode")
    )


def _describes_requests(metadata: Any) -> bool:
    return _has_request_fields(metadata) or _is_flashinfer(metadata)


def _flashinfer_error(metadata: Any, part: str, reason: str) -> RuntimeError:
    return RuntimeError(
        f"tensor_dump cannot read the {part} requests of "
        f"{type(metadata).__name__}: {reason}"
    )


def _flashinfer_decode(metadata: Any) -> Any:
    """The decode part of FlashInfer metadata, if it holds per-request lengths.

    Raises:
        RuntimeError: for FlashInfer's native decode wrapper, or decode
            context parallelism, whose ``seq_lens`` are local to the rank.
    """
    decode = getattr(metadata, "decode", None)
    if decode is None:
        raise _flashinfer_error(metadata, "decode", "it has no decode metadata")
    if (
        getattr(decode, "seq_lens", None) is None
        or getattr(decode, "q_len_per_req", None) is None
    ):
        raise _flashinfer_error(
            metadata,
            "decode",
            f"FlashInfer runs them through {type(decode).__name__}, which keeps "
            "their lengths inside a FlashInfer wrapper. FlashInfer uses its "
            "TRTLLM or XQA decode kernels instead on GPUs of compute capability "
            "9.0, 10.x and 12.x, unless --attention-config.use_trtllm_attention=0 "
            "is set or the layer's head counts or head size rule them out; "
            "otherwise start vLLM with --attention-backend FLASH_ATTN",
        )
    if getattr(decode, "dcp_query_start_loc", None) is not None:
        raise _flashinfer_error(
            metadata,
            "decode",
            "under decode context parallelism its seq_lens hold each rank's "
            "share of the keys and values; start vLLM with "
            "--decode-context-parallel-size 1",
        )
    return decode


def _flashinfer_prefill(metadata: Any) -> Any:
    """The prefill part of FlashInfer metadata, if it holds per-request lengths.

    Raises:
        RuntimeError: for FlashInfer's native prefill wrapper.
    """
    prefill = getattr(metadata, "prefill", None)
    if prefill is None:
        raise _flashinfer_error(metadata, "prefill", "it has no prefill metadata")
    if (
        getattr(prefill, "seq_lens", None) is None
        or getattr(prefill, "cum_seq_lens_q", None) is None
    ):
        raise _flashinfer_error(
            metadata,
            "prefill",
            f"FlashInfer runs them through {type(prefill).__name__}, which keeps "
            "their lengths inside a FlashInfer wrapper. FlashInfer uses its "
            "TRTLLM prefill kernel instead only on GPUs of compute capability "
            "10.x without decode context parallelism, where "
            "--attention-config.use_trtllm_attention=1 selects it; otherwise "
            "start vLLM with --attention-backend FLASH_ATTN",
        )
    return prefill


def _flashinfer_lengths(
    metadata: Any, part: str, name: str, tensor: torch.Tensor, count: int
) -> torch.Tensor:
    """The first ``count`` entries of ``tensor``.

    Raises:
        RuntimeError: if it has fewer.
    """
    lengths = tensor[:count]
    if lengths.numel() != count:
        raise _flashinfer_error(
            metadata, part, f"its {name} has {lengths.numel()} entries, not {count}"
        )
    return lengths


def _parse_flashinfer_requests(metadata: Any) -> tuple[_RequestInfo, ...]:
    """The requests of vLLM's FlashInfer metadata.

    FlashInfer orders the batch decodes first. Decode lengths come from its
    TRTLLM/XQA decode metadata (``seq_lens``, and ``q_cu_seq_lens``, or one
    query token per request when it is ``None``), prefill lengths from its
    TRTLLM prefill metadata (``seq_lens``, and ``cum_seq_lens_q`` counted from
    the first prefill token).

    Raises:
        RuntimeError: for a FlashInfer path whose metadata does not hold
            per-request lengths, decodes of several query tokens without
            ``q_cu_seq_lens``, or counts that disagree with the lengths.
    """
    if getattr(metadata, "use_cascade", False):
        raise _flashinfer_error(
            metadata,
            "batch",
            "FlashInfer cascade attention keeps them inside a FlashInfer "
            "wrapper; start vLLM with --disable-cascade-attn",
        )
    num_decodes = int(metadata.num_decodes)
    num_decode_tokens = int(metadata.num_decode_tokens)
    num_prefills = int(metadata.num_prefills)
    # (part, requests, tokens, first token row, uniform q_len or None)
    parts: list[tuple[str, int, int, int, int | None]] = []
    pieces: list[torch.Tensor] = []
    if num_decodes:
        decode = _flashinfer_decode(metadata)
        q_cu_seq_lens = getattr(decode, "q_cu_seq_lens", None)
        pieces.append(
            _flashinfer_lengths(
                metadata, "decode", "seq_lens", decode.seq_lens, num_decodes
            )
        )
        uniform = None
        if q_cu_seq_lens is not None:
            pieces.append(
                _flashinfer_lengths(
                    metadata, "decode", "q_cu_seq_lens", q_cu_seq_lens, num_decodes + 1
                )
            )
        else:
            uniform = int(decode.q_len_per_req)
            if uniform != 1:
                raise _flashinfer_error(
                    metadata,
                    "decode",
                    f"its q_cu_seq_lens is None and its q_len_per_req is "
                    f"{uniform}, not 1; without speculative decoding every "
                    f"decode request has one query token",
                )
        parts.append(("decode", num_decodes, num_decode_tokens, 0, uniform))
    if num_prefills:
        prefill = _flashinfer_prefill(metadata)
        pieces.append(
            _flashinfer_lengths(
                metadata, "prefill", "seq_lens", prefill.seq_lens, num_prefills
            )
        )
        pieces.append(
            _flashinfer_lengths(
                metadata,
                "prefill",
                "cum_seq_lens_q",
                prefill.cum_seq_lens_q,
                num_prefills + 1,
            )
        )
        tokens = int(metadata.num_prefill_tokens)
        parts.append(("prefill", num_prefills, tokens, num_decode_tokens, None))
    # One host transfer for every length of the step.
    values = [int(value) for value in torch.cat(pieces).tolist()] if pieces else []
    requests: list[_RequestInfo] = []
    taken = 0
    for part, count, tokens, first_token, uniform in parts:
        kv_lens = values[taken : taken + count]
        taken += count
        if uniform is None:
            offsets = values[taken : taken + count + 1]
            taken += count + 1
        else:
            offsets = [index * uniform for index in range(count + 1)]
        if offsets[0] != 0 or offsets[-1] != tokens:
            raise _flashinfer_error(
                metadata,
                part,
                f"its query offsets run from {offsets[0]} to {offsets[-1]}, "
                f"not over its {tokens} tokens",
            )
        for kv_len, start, end in zip(kv_lens, offsets, offsets[1:]):
            q_start_pos = kv_len - (end - start)
            requests.append(
                _RequestInfo(
                    request=len(requests),
                    q_len=end - start,
                    kv_len=kv_len,
                    q_start_pos=q_start_pos,
                    token_start=first_token + start,
                    phase=_request_phase(end - start, q_start_pos),
                )
            )
    return tuple(requests)


def _parse_requests(metadata: Any) -> tuple[_RequestInfo, ...]:
    """The requests of one attention-metadata object.

    Reads ``query_start_loc`` and ``seq_lens``; ``num_reqs`` and
    ``decode_offset`` are used when present, and derived otherwise. Metadata
    without these fields is read as vLLM's FlashInfer metadata when it has
    FlashInfer's fields; see ``_parse_flashinfer_requests``.

    Raises:
        RuntimeError: if the metadata describes no requests this way.
    """
    if not _has_request_fields(metadata):
        if _is_flashinfer(metadata):
            return _parse_flashinfer_requests(metadata)
        raise RuntimeError(
            f"tensor_dump needs query_start_loc and seq_lens, or FlashInfer's "
            f"decode and prefill fields, in the attention metadata, which "
            f"{type(metadata).__name__} does not provide"
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
    """The first metadata entry that describes the step's requests, preferring
    one with ``query_start_loc`` and ``seq_lens``."""
    for describes in (_has_request_fields, _describes_requests):
        for metadata in attn_metadata.values():
            if describes(metadata):
                return metadata
    types = sorted({type(metadata).__name__ for metadata in attn_metadata.values()})
    raise RuntimeError(
        f"tensor_dump needs query_start_loc and seq_lens, or FlashInfer's decode "
        f"and prefill fields, in the attention metadata, which none of {types} "
        f"provides"
    )


def _assigns_no_kv_slot(attn_metadata: dict[str, Any]) -> bool:
    """Whether every ``slot_mapping`` entry of the attention metadata is
    negative, so no token of the forward has a KV-cache slot.

    vLLM marks a token without a slot with ``-1``. A request forward gives each
    of its tokens a slot and marks only padding; a dummy forward marks every
    token. ``False`` when no entry has a non-empty ``slot_mapping``.
    """
    mappings: dict[int, torch.Tensor] = {}
    for metadata in attn_metadata.values():
        slot_mapping = getattr(metadata, "slot_mapping", None)
        if isinstance(slot_mapping, torch.Tensor) and slot_mapping.numel():
            mappings[id(slot_mapping)] = slot_mapping
    if not mappings:
        return False
    # One host transfer for every slot mapping of the forward.
    largest = torch.stack([mapping.max().long() for mapping in mappings.values()])
    return int(largest.max()) < 0


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
        on_step: Called with the step id at the start of every real forward.
        on_selected_step: Called at the start of every step that is written.
        total_bytes: The bytes the rank's dump directory holds, checked
            against the byte budget before a step is written.
        context_parallel: Whether vLLM runs with context parallelism, under
            which a rank may hold no KV-cache slot for any token of a request
            forward; forwards are then not checked for KV-cache slots.
    """

    def __init__(
        self,
        options: _DumpOptions,
        on_step: Callable[[int], None],
        on_selected_step: Callable[[_StepInfo], None],
        total_bytes: Callable[[], int],
        *,
        context_parallel: bool = False,
    ) -> None:
        self._options = options
        self._context_parallel = context_parallel
        self._on_step = on_step
        self._on_selected_step = on_selected_step
        self._total_bytes = total_bytes
        # Keyed by id() and checked against a weak reference, since vLLM's
        # forward contexts are not hashable.
        self._forwards: dict[int, _Forward] = {}
        self._next_step = 0
        self._rank: int | None = None
        self._steps_written = 0
        self._cap_logged = False
        self._budget_logged = False

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
        def forget(ref: weakref.ref[Any]) -> None:
            # Another context may hold the id by now; drop only this entry.
            forward = self._forwards.get(key)
            if forward is not None and forward.context is ref:
                del self._forwards[key]

        return forget

    @staticmethod
    def _requests(forward: _Forward, metadata: Any) -> tuple[_RequestInfo, ...]:
        requests = forward.requests.get(id(metadata))
        if requests is None:
            requests = _parse_requests(metadata)
            forward.requests[id(metadata)] = requests
        return requests

    def _is_dummy(self, context: Any) -> bool:
        """Whether the forward has no attention metadata, or assigns no token a
        KV-cache slot.

        Raises:
            NotImplementedError: for dual-batch-overlap micro-batches.
            RuntimeError: when the forward runs with CUDA graphs.
        """
        attn_metadata = context.attn_metadata
        if isinstance(attn_metadata, list):
            raise NotImplementedError(
                "tensor_dump does not support dual-batch-overlap micro-batches"
            )
        if not attn_metadata:
            return True
        mode = getattr(context, "cudagraph_runtime_mode", None)
        if mode is not None and mode.name != "NONE":
            raise RuntimeError(
                f"tensor_dump requires eager execution, but the forward runs with "
                f"CUDA graph mode {mode.name}; start vLLM with --enforce-eager"
            )
        return not self._context_parallel and _assigns_no_kv_slot(attn_metadata)

    def _start_step(self, context: Any, forward: _Forward) -> _StepInfo | None:
        if self._is_dummy(context):
            return None
        attn_metadata = context.attn_metadata
        step = self._next_step
        self._next_step += 1
        if self._rank is None:
            self._rank = _distributed_rank()[0]
        self._on_step(step)
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
        max_bytes = self._options.max_bytes
        if max_bytes:
            total = self._total_bytes()
            if total >= max_bytes:
                if not self._budget_logged:
                    self._budget_logged = True
                    logger.info(
                        "tensor_dump holds %d bytes, at least the "
                        "QUANT_SUITE_TENSOR_DUMP_MAX_BYTES budget of %d; nothing "
                        "more is written",
                        total,
                        max_bytes,
                    )
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
