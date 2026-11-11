# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stand-ins for vLLM's forward context and attention layer, for the tests."""

import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import torch
from tensor_dump import _session, _step
from tensor_dump._step import _VllmSettings

_ENV_PREFIX = "QUANT_SUITE_TENSOR_DUMP_"


@dataclass
class FakeAttentionMetadata:
    """The attention-metadata fields the dump reads."""

    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    decode_offset: torch.Tensor
    num_reqs: int
    num_actual_tokens: int


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


def generic_metadata(
    q_lens: list[int], q_start_positions: list[int]
) -> GenericAttentionMetadata:
    full = fake_metadata(q_lens, q_start_positions)
    return GenericAttentionMetadata(full.query_start_loc, full.seq_lens)


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


@contextmanager
def dump_session(
    directory: Path | None,
    *,
    settings: _VllmSettings | None = None,
    real_vllm: bool = False,
    **env: str,
) -> Iterator[FakeForward]:
    """A fresh dump session writing to ``directory``; for ``None``,
    ``QUANT_SUITE_TENSOR_DUMP_DIR`` is unset and dumping is disabled.

    ``env`` sets the other variables by suffix, e.g. ``MAX_STEPS="2"``.
    ``settings`` stands in for the vLLM settings. With ``real_vllm``, the
    forward context, parallel state and settings come from vLLM itself.
    """
    forward = FakeForward()
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ))
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


class AttentionLayer(torch.nn.Module):
    """The attributes of a vLLM attention layer that the dump reads.

    ``forward(fn)`` returns ``fn(self.quantized_attention)``, so a test runs
    any sequence of executor calls inside a forward of this layer.
    """

    def __init__(
        self,
        executor: torch.nn.Module,
        layer_name: str,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        micro_batch_size: int | None = None,
    ) -> None:
        super().__init__()
        self.layer_name = layer_name
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.impl = SimpleNamespace(scale=head_size**-0.5)
        if micro_batch_size is not None:
            self.impl.micro_batch_size = micro_batch_size
        self.quantized_attention = executor

    def forward(self, fn: Callable[[torch.nn.Module], torch.Tensor]) -> torch.Tensor:
        return fn(self.quantized_attention)
