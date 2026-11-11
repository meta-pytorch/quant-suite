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
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import torch
from quant_suite_recipe.api import QuantizedTensor
from tensor_dump import _step
from tensor_dump._options import _DumpOptions
from tensor_dump._step import _RequestInfo, _StepInfo

_FORMAT_VERSION = 1
_LAYER_NAME_RE: re.Pattern[str] = re.compile(r"[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class _AttentionLayout:
    """Per-rank shape and scale of one MHA/GQA attention layer."""

    layer: str
    num_heads: int
    num_kv_heads: int
    head_size: int
    scale: float

    @property
    def gqa_group(self) -> int:
        return self.num_heads // self.num_kv_heads


@dataclass
class _PendingAttention:
    """Q and K of one request, held until its ``P @ V`` arrives."""

    step: int
    layout: _AttentionLayout
    request: _RequestInfo
    tensors: frozenset[str]
    dtype: torch.dtype
    host: dict[str, torch.Tensor] = field(default_factory=dict)


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


def _unexpand_kv(
    tensor: torch.Tensor, layout: _AttentionLayout, name: str
) -> torch.Tensor:
    """``[Hq, L, D]`` with every KV head repeated ``gqa_group`` times, as
    ``[L, Hkv, D]``.

    Raises:
        RuntimeError: if the query heads of a group do not hold equal copies.
    """
    group = layout.gqa_group
    num_heads, length, head_size = tensor.shape
    if num_heads != layout.num_heads:
        raise RuntimeError(
            f"{name} of layer {layout.layer!r} has {num_heads} heads, expected "
            f"{layout.num_heads}"
        )
    unexpanded = tensor[::group]
    if group > 1 and not torch.equal(
        tensor.reshape(layout.num_kv_heads, group, length, head_size),
        unexpanded.unsqueeze(1).expand(-1, group, -1, -1),
    ):
        raise RuntimeError(
            f"{name} of layer {layout.layer!r} does not repeat each KV head over "
            f"{group} consecutive query heads"
        )
    return unexpanded.transpose(0, 1)


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

    def attention_qk(
        self,
        step: _StepInfo,
        layout: _AttentionLayout,
        request: _RequestInfo,
        tensors: frozenset[str],
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> _PendingAttention:
        """Copy one request's Q ``[Hq, q_len, D]`` and K ``[Hq, kv_len, D]``
        to the host.

        Raises:
            RuntimeError: if the queries are not the last ``q_len`` of the
                ``kv_len`` positions.
        """
        _check_layer_name(layout.layer)
        if request.q_start_pos != request.kv_len - request.q_len:
            raise RuntimeError(
                f"request {request.request} of layer {layout.layer!r} has "
                f"q_start_pos {request.q_start_pos}, q_len {request.q_len} and "
                f"kv_len {request.kv_len}; the queries must be the last q_len "
                "positions"
            )
        pending = _PendingAttention(step.step, layout, request, tensors, q.dtype)
        if "q" in tensors:
            pending.host["q"] = _to_host(q.transpose(0, 1))
        if "k" in tensors:
            pending.host["k"] = _to_host(_unexpand_kv(k, layout, "K"))
        return pending

    def attention_v_out(
        self, pending: _PendingAttention, v: torch.Tensor, output: torch.Tensor
    ) -> None:
        """Write one request's tensors, given V ``[Hq, kv_len, D]`` and the
        attention output ``[Hq, q_len, D]``."""
        request = pending.request
        layout = pending.layout
        if v.shape[1] != request.kv_len or output.shape[1] != request.q_len:
            raise RuntimeError(
                f"P @ V of layer {layout.layer!r} has V {tuple(v.shape)} and output "
                f"{tuple(output.shape)}, expected kv_len {request.kv_len} and "
                f"q_len {request.q_len}"
            )
        if "v" in pending.tensors:
            pending.host["v"] = _to_host(_unexpand_kv(v, layout, "V"))
        if "output" in pending.tensors:
            pending.host["output"] = _to_host(output.transpose(0, 1))
        prefix = f"step_{pending.step:06d}/{layout.layer}/req{request.request:03d}"
        entries = {
            name: self._save(f"{prefix}.{name}.pt", host, to_host=False)
            for name, host in sorted(pending.host.items())
        }
        dtype = _dtype_name(pending.dtype)
        self._record(
            {
                "record": "attention",
                "step": pending.step,
                "layer": layout.layer,
                "request": request.request,
                "phase": request.phase,
                "tensors": entries,
                "replay": {
                    "q_len": request.q_len,
                    "kv_len": request.kv_len,
                    "q_start_pos": request.q_start_pos,
                    "causal": True,
                    "causal_alignment": "bottom_right",
                    "softmax_scale": layout.scale,
                    "num_q_heads": layout.num_heads,
                    "num_kv_heads": layout.num_kv_heads,
                    "head_dim": layout.head_size,
                    "gqa_group": layout.gqa_group,
                    "sliding_window": None,
                    "logits_soft_cap": None,
                    "alibi": None,
                    "layout": "NHD",
                    "dtype": dtype,
                    "rope_applied": True,
                    "reference": {
                        "qk_accum": "float32",
                        "softmax": "float32",
                        "pv_operand": dtype,
                    },
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

    def _save(
        self, relative: str, tensor: torch.Tensor, *, to_host: bool = True
    ) -> dict[str, object]:
        root = self._root
        if root is None:
            raise RuntimeError("tensor_dump wrote a tensor before its first step")
        host = _to_host(tensor) if to_host else tensor
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
