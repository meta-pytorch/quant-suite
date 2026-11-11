# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stand-ins for vLLM's forward context and attention layer, for the tests."""

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import torch
from tensor_dump import _attention, _session, _step
from tensor_dump._step import _VllmSettings

_ENV_PREFIX = "QUANT_SUITE_TENSOR_DUMP_"


@dataclass
class FakeAttentionMetadata:
    """The attention-metadata fields the dump reads. ``slot_mapping`` is the
    token's KV-cache slot, ``-1`` for none; ``None`` leaves it out."""

    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    decode_offset: torch.Tensor
    num_reqs: int
    num_actual_tokens: int
    slot_mapping: torch.Tensor | None = None


def fake_metadata(
    q_lens: list[int], q_start_positions: list[int]
) -> FakeAttentionMetadata:
    """Metadata for requests whose queries are the last ``q_len`` positions."""
    starts = [0]
    for q_len in q_lens:
        starts.append(starts[-1] + q_len)
    return FakeAttentionMetadata(
        query_start_loc=torch.tensor(starts),
        seq_lens=torch.tensor(
            [q_len + start for q_len, start in zip(q_lens, q_start_positions)]
        ),
        decode_offset=torch.tensor(q_start_positions),
        num_reqs=len(q_lens),
        num_actual_tokens=sum(q_lens),
    )


@dataclass
class GenericAttentionMetadata:
    """Attention metadata with only the fields every vLLM backend provides."""

    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor | None = None


def generic_metadata(
    q_lens: list[int], q_start_positions: list[int]
) -> GenericAttentionMetadata:
    full = fake_metadata(q_lens, q_start_positions)
    return GenericAttentionMetadata(full.query_start_loc, full.seq_lens)


@dataclass
class FakeTrtllmPrefill:
    """The fields of vLLM's FlashInfer TRTLLM prefill metadata the dump reads."""

    seq_lens: torch.Tensor
    cum_seq_lens_q: torch.Tensor


@dataclass
class FakeTrtllmDecode:
    """The fields of vLLM's FlashInfer TRTLLM/XQA decode metadata the dump
    reads."""

    seq_lens: torch.Tensor
    q_len_per_req: int = 1
    q_cu_seq_lens: torch.Tensor | None = None
    dcp_query_start_loc: torch.Tensor | None = None


@dataclass
class FakeWrapperMetadata:
    """FlashInfer's native prefill or decode metadata, which holds only a
    wrapper."""

    wrapper: object = None


@dataclass
class FakeFlashInferMetadata:
    """The fields of vLLM's FlashInfer metadata the dump reads."""

    num_actual_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    prefill: Any
    decode: Any
    use_cascade: bool = False
    slot_mapping: torch.Tensor | None = None


def flashinfer_metadata(
    q_lens: list[int], q_start_positions: list[int], *, ragged_decode: bool = False
) -> FakeFlashInferMetadata:
    """FlashInfer metadata for requests in batch order, decodes first: the
    leading requests with one query token are decodes, as FlashInfer splits
    them. With ``ragged_decode``, the decodes carry ``q_cu_seq_lens``."""
    num_decodes = 0
    while num_decodes < len(q_lens) and q_lens[num_decodes] == 1:
        num_decodes += 1
    kv_lens = [q_len + start for q_len, start in zip(q_lens, q_start_positions)]
    prefill_lens = q_lens[num_decodes:]
    decode = None
    if num_decodes:
        decode = FakeTrtllmDecode(
            seq_lens=torch.tensor(kv_lens[:num_decodes], dtype=torch.int32),
            q_cu_seq_lens=(
                torch.arange(num_decodes + 1, dtype=torch.int32)
                if ragged_decode
                else None
            ),
        )
    prefill = None
    if prefill_lens:
        offsets = [0]
        for q_len in prefill_lens:
            offsets.append(offsets[-1] + q_len)
        prefill = FakeTrtllmPrefill(
            seq_lens=torch.tensor(kv_lens[num_decodes:], dtype=torch.int32),
            cum_seq_lens_q=torch.tensor(offsets, dtype=torch.int32),
        )
    return FakeFlashInferMetadata(
        num_actual_tokens=sum(q_lens),
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes,
        num_prefills=len(prefill_lens),
        num_prefill_tokens=sum(prefill_lens),
        prefill=prefill,
        decode=decode,
    )


@dataclass(eq=False)
class FakeForwardContext:
    attn_metadata: dict[str, Any] | list[dict[str, Any]] | None
    cudagraph_runtime_mode: Any = None


class FakeForward:
    """The forward context the dump sees; each ``real`` or ``dummy`` call
    starts a new forward."""

    def __init__(self) -> None:
        self.context: FakeForwardContext | None = None
        self.provider: Mock = Mock(side_effect=lambda: self.context)

    def real(self, attn_metadata: dict[str, Any]) -> None:
        self.context = FakeForwardContext(attn_metadata)

    def dummy(self) -> None:
        self.context = FakeForwardContext(None)


class FakeAttention(torch.nn.Module):
    """The attributes of vLLM's ``Attention`` layer that the dump reads.

    ``forward(query, key, value)`` takes ``[tokens, heads * head_size]``
    inputs, as vLLM's layer does, and returns ``2 * query + 1``.
    """

    def __init__(
        self,
        layer_name: str,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        *,
        sliding_window: int | None = None,
        logits_soft_cap: float = 0.0,
    ) -> None:
        super().__init__()
        self.layer_name = layer_name
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.head_size_v = head_size
        self.sliding_window = sliding_window
        self.attn_type = "decoder"
        self.has_sink = False
        self.kv_sharing_target_layer_name: str | None = None
        self.kv_cache_dtype = "auto"
        self.impl = SimpleNamespace(
            scale=head_size**-0.5,
            logits_soft_cap=logits_soft_cap,
            alibi_slopes=None,
        )
        self.attn_backend = SimpleNamespace(get_name=lambda: "FAKE")

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        return query * 2 + 1


@contextmanager
def dump_session(
    directory: Path | None,
    *,
    settings: _VllmSettings | None = None,
    real_vllm: bool = False,
    attention_class: type[torch.nn.Module] = FakeAttention,
    **env: str,
) -> Iterator[FakeForward]:
    """A fresh dump session writing to ``directory``; for ``None``,
    ``QUANT_SUITE_TENSOR_DUMP_DIR`` is unset and dumping is disabled.

    ``env`` sets the other variables by suffix, e.g. ``MAX_STEPS="2"``.
    ``settings`` stands in for the vLLM settings. With ``real_vllm``, the
    forward context, parallel state and settings come from vLLM itself.
    ``attention_class`` stands in for vLLM's ``Attention`` class.
    """
    forward = FakeForward()
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ))
        stack.enter_context(
            patch.object(_attention, "_attention_class", return_value=attention_class)
        )
        if not real_vllm:
            stack.enter_context(
                patch.object(_step, "_forward_context", forward.provider)
            )
            stack.enter_context(
                patch.object(_step, "_tensor_parallel", return_value=None)
            )
            stack.enter_context(
                patch.object(
                    _step,
                    "_vllm_settings",
                    return_value=settings or _VllmSettings(),
                )
            )
        for name in [name for name in os.environ if name.startswith(_ENV_PREFIX)]:
            del os.environ[name]
        if directory is not None:
            os.environ[f"{_ENV_PREFIX}DIR"] = str(directory)
        for suffix, value in env.items():
            os.environ[f"{_ENV_PREFIX}{suffix}"] = value
        _session._reset()
        try:
            yield forward
        finally:
            _session._reset()


def read_manifest(directory: Path, rank: int = 0) -> list[dict[str, Any]]:
    path = directory / f"rank_{rank}" / "manifest.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def load(directory: Path, entry: dict[str, Any], rank: int = 0) -> torch.Tensor:
    """The tensor a manifest ``tensors`` entry names."""
    return torch.load(directory / f"rank_{rank}" / entry["file"])
