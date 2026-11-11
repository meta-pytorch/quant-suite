# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tensor files and the JSON Lines manifest that indexes them.

Layout under the output root, with every manifest path relative to
``rank_<R>/``::

    rank_<R>/manifest.jsonl
    rank_<R>/weights/<layer>/weight.pt          (and bias.pt)
    rank_<R>/step_<NNNNNN>/<layer>/input.pt     (and output.pt)
    rank_<R>/step_<NNNNNN>/<layer>/req<NNN>.<q|k|v|output>.pt

Attention ``k`` and ``v`` hold the rows of the step's new tokens only: rows
``[q_start_pos, kv_len)`` of the request's keys and values.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import torch
from quant_suite_recipe.api import QuantizedTensor
from tensor_dump import _step
from tensor_dump._options import _DumpOptions
from tensor_dump._step import _RequestInfo, _StepInfo

_FORMAT_VERSION = 2
_LAYER_NAME_RE: re.Pattern[str] = re.compile(r"[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class _AttentionLayout:
    """Per-rank shape and attention parameters of one attention layer.

    Attributes:
        layer: vLLM's name for the layer.
        num_heads, num_kv_heads: Query and key/value heads on this rank.
        head_size, head_size_v: Head sizes of Q and K, and of V and the output.
        scale: The softmax scale applied to ``Q @ K^T``.
        sliding_window: vLLM's window: the query at position ``p`` attends the
            keys at positions ``p - sliding_window + 1 .. p``. ``None`` for full
            attention.
        logits_soft_cap: The soft cap on the attention logits; ``None`` for none.
        kv_cache_dtype: vLLM's KV-cache dtype setting.
        backend: vLLM's name for the attention backend, if known.
    """

    layer: str
    num_heads: int
    num_kv_heads: int
    head_size: int
    head_size_v: int
    scale: float
    sliding_window: int | None
    logits_soft_cap: float | None
    kv_cache_dtype: str
    backend: str | None


def _to_host(tensor: torch.Tensor) -> torch.Tensor:
    """A compact host copy that later in-place writes cannot change."""
    return tensor.detach().contiguous().to("cpu", copy=True)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _check_layer_name(name: str) -> None:
    if not _LAYER_NAME_RE.fullmatch(name) or name in (".", ".."):
        raise ValueError(
            f"layer name {name!r} cannot be used as a directory name; expected "
            "letters, digits, '_', '.' and '-'"
        )


class _Writer:
    """Writes one rank's tensor files and manifest."""

    def __init__(self, options: _DumpOptions) -> None:
        self._options = options
        self._root: Path | None = None
        self._manifest: TextIO | None = None
        self._weights_written: set[str] = set()

    def begin_step(self, step: _StepInfo) -> None:
        """Write the ``step`` record, after the ``run`` record on the first call.

        Raises:
            FileExistsError: if the rank's manifest already exists.
        """
        if self._manifest is None:
            self._open()
        self._record(
            {
                "record": "step",
                "step": step.step,
                "phase": step.phase,
                "num_actual_tokens": step.num_actual_tokens,
                "requests": [
                    {
                        "request": request.request,
                        "q_len": request.q_len,
                        "kv_len": request.kv_len,
                        "q_start_pos": request.q_start_pos,
                        "token_start": request.token_start,
                        "phase": request.phase,
                    }
                    for request in step.requests
                ],
            }
        )

    def linear(
        self,
        step: _StepInfo,
        layer: str,
        tensors: frozenset[str],
        x: torch.Tensor,
        weight: QuantizedTensor,
        bias: torch.Tensor | None,
        output: torch.Tensor,
    ) -> None:
        """Write a linear layer's selected rows, and its weight once."""
        _check_layer_name(layer)
        rows = step.num_actual_tokens
        if x.shape[0] < rows:
            raise RuntimeError(
                f"linear layer {layer!r} got {x.shape[0]} rows, fewer than the "
                f"step's {rows} tokens"
            )
        directory = f"step_{step.step:06d}/{layer}"
        entries = {}
        if "input" in tensors:
            entries["input"] = self._save(f"{directory}/input.pt", x[:rows])
        if "output" in tensors:
            entries["output"] = self._save(f"{directory}/output.pt", output[:rows])
        if entries:
            self._record(
                {
                    "record": "linear",
                    "step": step.step,
                    "layer": layer,
                    "tensors": entries,
                }
            )
        if "weight" in tensors and layer not in self._weights_written:
            self._weight(step, layer, weight, bias)

    def attention(
        self,
        step: _StepInfo,
        layout: _AttentionLayout,
        requests: tuple[_RequestInfo, ...],
        tensors: dict[str, torch.Tensor],
        dtype: torch.dtype,
    ) -> None:
        """Write each request's rows of ``tensors``, which hold one row per
        token of the step: ``q`` and ``output`` ``[tokens, heads, head_size]``,
        ``k`` and ``v`` ``[tokens, kv_heads, head_size]``.

        Raises:
            RuntimeError: if a tensor has fewer rows than the requests cover,
                or a request's queries are not its last ``q_len`` positions.
        """
        _check_layer_name(layout.layer)
        rows = max(
            (request.token_start + request.q_len for request in requests), default=0
        )
        for name, tensor in tensors.items():
            if tensor.shape[0] < rows:
                raise RuntimeError(
                    f"attention layer {layout.layer!r} got {tensor.shape[0]} rows "
                    f"of {name}, fewer than the {rows} its requests cover"
                )
        for request in requests:
            if request.q_start_pos != request.kv_len - request.q_len:
                raise RuntimeError(
                    f"request {request.request} of layer {layout.layer!r} has "
                    f"q_start_pos {request.q_start_pos}, q_len {request.q_len} "
                    f"and kv_len {request.kv_len}; the queries must be the last "
                    "q_len positions"
                )
            first = request.token_start
            last = first + request.q_len
            prefix = f"step_{step.step:06d}/{layout.layer}/req{request.request:03d}"
            entries = {
                name: self._save(f"{prefix}.{name}.pt", tensor[first:last])
                for name, tensor in sorted(tensors.items())
            }
            self._record(
                {
                    "record": "attention",
                    "step": step.step,
                    "layer": layout.layer,
                    "request": request.request,
                    "phase": request.phase,
                    "tensors": entries,
                    "replay": {
                        "q_start_pos": request.q_start_pos,
                        "q_len": request.q_len,
                        "kv_len": request.kv_len,
                        "kv_rows": [request.q_start_pos, request.kv_len],
                        "causal": True,
                        "scale": layout.scale,
                        "sliding_window": layout.sliding_window,
                        "logits_soft_cap": layout.logits_soft_cap,
                        "num_q_heads": layout.num_heads,
                        "num_kv_heads": layout.num_kv_heads,
                        "head_size": layout.head_size,
                        "head_size_v": layout.head_size_v,
                        "layout": "NHD",
                        "dtype": _dtype_name(dtype),
                        "kv_cache_dtype": layout.kv_cache_dtype,
                        "backend": layout.backend,
                    },
                }
            )

    def close(self) -> None:
        if self._manifest is not None:
            self._manifest.close()
            self._manifest = None

    def _open(self) -> None:
        rank, world_size = _step._distributed_rank()
        tensor_parallel = _step._tensor_parallel()
        tp_rank, tp_size = tensor_parallel if tensor_parallel else (None, None)
        self._root = self._options.directory / f"rank_{rank}"
        self._root.mkdir(parents=True, exist_ok=True)
        self._manifest = open(self._root / "manifest.jsonl", "x", encoding="utf-8")
        self._record(
            {
                "record": "run",
                "format_version": _FORMAT_VERSION,
                "rank": rank,
                "world_size": world_size,
                "tp_rank": tp_rank,
                "tp_size": tp_size,
                "options": self._options.describe(),
                "torch_version": torch.__version__,
            }
        )

    def _weight(
        self,
        step: _StepInfo,
        layer: str,
        weight: QuantizedTensor,
        bias: torch.Tensor | None,
    ) -> None:
        if weight.precision is not None or weight.tensor is None:
            raise NotImplementedError(
                f"tensor_dump writes unquantized and fake-quantized weights only; "
                f"layer {layer!r} has precision {weight.precision}"
            )
        entries = {"weight": self._save(f"weights/{layer}/weight.pt", weight.tensor)}
        if bias is not None:
            entries["bias"] = self._save(f"weights/{layer}/bias.pt", bias)
        self._weights_written.add(layer)
        self._record(
            {"record": "weight", "step": step.step, "layer": layer, "tensors": entries}
        )

    def _save(self, relative: str, tensor: torch.Tensor) -> dict[str, object]:
        root = self._root
        if root is None:
            raise RuntimeError("tensor_dump wrote a tensor before its first step")
        host = _to_host(tensor)
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(host, path)
        return {
            "file": relative,
            "shape": list(host.shape),
            "dtype": _dtype_name(host.dtype),
        }

    def _record(self, record: dict[str, object]) -> None:
        manifest = self._manifest
        if manifest is None:
            raise RuntimeError("tensor_dump wrote a record before its first step")
        manifest.write(json.dumps(record, sort_keys=True) + "\n")
        manifest.flush()
