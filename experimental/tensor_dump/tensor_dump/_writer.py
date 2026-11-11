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

A run appends to the rank's manifest when it exists: it adds a ``run`` record,
numbers its steps after the largest step id already recorded, and never
overwrites a file. A weight already recorded is checked against the layer's
current weight by shape, dtype and SHA-256 instead of being written again.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

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


def _listed_file_size(root: Path, manifest: Path, name: str) -> int:
    file = root / name
    try:
        return file.stat().st_size
    except OSError as error:
        raise RuntimeError(
            f"{manifest} lists {file}, which cannot be read ({error}); "
            "use a new directory"
        ) from error


def _sha256(host: torch.Tensor) -> str:
    """SHA-256 of a contiguous host tensor's bytes, read in place."""
    return hashlib.sha256(host.reshape(-1).view(torch.uint8).numpy().data).hexdigest()


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _check_layer_name(name: str) -> None:
    if not _LAYER_NAME_RE.fullmatch(name) or name in (".", ".."):
        raise ValueError(
            f"layer name {name!r} cannot be used as a directory name; expected "
            "letters, digits, '_', '.' and '-'"
        )


class _Writer:
    """Writes one rank's tensor files and manifest, after any earlier runs'."""

    def __init__(self, options: _DumpOptions) -> None:
        self._options = options
        self._root: Path | None = None
        self._manifest: TextIO | None = None
        self._loaded = False
        self._runs = 0
        self._first_step = 0
        self._bytes = 0
        self._recorded_weights: dict[str, dict[str, Any]] = {}
        self._weights_written: set[str] = set()

    def total_bytes(self) -> int:
        """The bytes of the rank's manifest and of every file it lists,
        earlier runs included.

        Raises:
            RuntimeError: for an existing manifest that cannot be appended to.
        """
        self._load()
        return self._bytes

    def begin_step(self, step: _StepInfo) -> None:
        """Write the ``step`` record, after the ``run`` record on the first call.

        Raises:
            RuntimeError: for an existing manifest that cannot be appended to.
        """
        if self._manifest is None:
            self._open()
        self._record(
            {
                "record": "step",
                "step": self._step_id(step),
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
        directory = f"step_{self._step_id(step):06d}/{layer}"
        entries = {}
        if "input" in tensors:
            entries["input"] = self._save(f"{directory}/input.pt", _to_host(x[:rows]))
        if "output" in tensors:
            entries["output"] = self._save(
                f"{directory}/output.pt", _to_host(output[:rows])
            )
        if entries:
            self._record(
                {
                    "record": "linear",
                    "step": self._step_id(step),
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
            prefix = (
                f"step_{self._step_id(step):06d}/{layout.layer}/"
                f"req{request.request:03d}"
            )
            entries = {
                name: self._save(f"{prefix}.{name}.pt", _to_host(tensor[first:last]))
                for name, tensor in sorted(tensors.items())
            }
            self._record(
                {
                    "record": "attention",
                    "step": self._step_id(step),
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

    def _step_id(self, step: _StepInfo) -> int:
        return self._first_step + step.step

    def _load(self) -> None:
        """Read the rank's existing manifest, if any, once."""
        if self._loaded:
            return
        rank = _step._distributed_rank()[0]
        root = self._options.directory / f"rank_{rank}"
        self._root = root
        path = root / "manifest.jsonl"
        if path.exists():
            self._read_history(root, path)
        self._loaded = True

    def _read_history(self, root: Path, path: Path) -> None:
        self._bytes = path.stat().st_size
        last_step = -1
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines()):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"line {number + 1} of {path} is not JSON ({error}); an earlier "
                    "run may have stopped while writing it. Use a new directory"
                ) from error
            kind = record.get("record")
            if kind == "run":
                self._runs += 1
                version = record.get("format_version")
                if version != _FORMAT_VERSION:
                    raise RuntimeError(
                        f"{path} holds a run in format {version}, not "
                        f"{_FORMAT_VERSION}; use a new directory"
                    )
            elif kind == "step":
                last_step = max(last_step, int(record["step"]))
            elif kind == "weight":
                self._recorded_weights[str(record["layer"])] = record["tensors"]
            for entry in record.get("tensors", {}).values():
                size = entry.get("bytes")
                if size is None:
                    size = _listed_file_size(root, path, entry["file"])
                self._bytes += int(size)
        if text and not text.endswith("\n"):
            raise RuntimeError(
                f"{path} does not end with a newline; an earlier run may have "
                "stopped while writing it. Use a new directory"
            )
        self._first_step = last_step + 1

    def _open(self) -> None:
        self._load()
        root = self._root
        if root is None:
            raise RuntimeError("tensor_dump has no output directory")
        rank, world_size = _step._distributed_rank()
        tensor_parallel = _step._tensor_parallel()
        tp_rank, tp_size = tensor_parallel if tensor_parallel else (None, None)
        root.mkdir(parents=True, exist_ok=True)
        self._manifest = open(root / "manifest.jsonl", "a", encoding="utf-8")
        self._record(
            {
                "record": "run",
                "format_version": _FORMAT_VERSION,
                "run": self._runs,
                "first_step": self._first_step,
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
        """Write the layer's weight and bias, or check them against the ones an
        earlier run recorded.

        Raises:
            NotImplementedError: for a natively quantized weight.
            RuntimeError: if the recorded weight differs, or a weight file
                exists that no record lists.
        """
        tensor = weight.tensor
        if weight.precision is not None or tensor is None:
            raise NotImplementedError(
                f"tensor_dump writes unquantized and fake-quantized weights only; "
                f"layer {layer!r} has precision {weight.precision}"
            )
        hosts = {"weight": _to_host(tensor)}
        if bias is not None:
            hosts["bias"] = _to_host(bias)
        self._weights_written.add(layer)
        recorded = self._recorded_weights.get(layer)
        if recorded is not None:
            self._check_weight(layer, recorded, hosts)
            return
        root = self._root
        if root is not None and (root / f"weights/{layer}").exists():
            raise RuntimeError(
                f"{root / 'weights' / layer} exists but no weight record lists "
                "it; use a new directory"
            )
        entries = {
            name: self._save(f"weights/{layer}/{name}.pt", host, checksum=True)
            for name, host in hosts.items()
        }
        self._record(
            {
                "record": "weight",
                "step": self._step_id(step),
                "layer": layer,
                "tensors": entries,
            }
        )

    def _check_weight(
        self, layer: str, recorded: dict[str, Any], hosts: dict[str, torch.Tensor]
    ) -> None:
        def differs(reason: str) -> RuntimeError:
            return RuntimeError(
                f"layer {layer!r} {reason} in the weights an earlier run recorded "
                f"in {self._options.directory}, so it runs a different model or "
                "recipe; use a new directory"
            )

        if sorted(recorded) != sorted(hosts):
            raise differs(f"has tensors {sorted(hosts)}, not the {sorted(recorded)}")
        for name, host in hosts.items():
            entry = recorded[name]
            if entry.get("shape") != list(host.shape):
                raise differs(
                    f"has {name} shape {list(host.shape)}, not {entry.get('shape')}"
                )
            if entry.get("dtype") != _dtype_name(host.dtype):
                raise differs(
                    f"has {name} dtype {_dtype_name(host.dtype)}, not {entry.get('dtype')}"
                )
            checksum = entry.get("sha256")
            if checksum is None:
                raise RuntimeError(
                    f"the recorded {name} of layer {layer!r} has no SHA-256 to "
                    "check against; use a new directory"
                )
            if checksum != _sha256(host):
                raise differs(f"has a {name} whose SHA-256 differs from the one")

    def _save(
        self, relative: str, host: torch.Tensor, *, checksum: bool = False
    ) -> dict[str, object]:
        """Save a host tensor to a new file; ``checksum`` adds its SHA-256.

        Raises:
            RuntimeError: if the file exists.
        """
        root = self._root
        if root is None:
            raise RuntimeError("tensor_dump wrote a tensor before its first step")
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            file = open(path, "xb")
        except FileExistsError as error:
            raise RuntimeError(
                f"tensor_dump would overwrite {path}: a layer ran twice in one "
                "step, as a module called twice per forward does, or the "
                "directory holds files that no manifest lists"
            ) from error
        with file:
            torch.save(host, file)
        size = path.stat().st_size
        self._bytes += size
        entry: dict[str, object] = {
            "file": relative,
            "shape": list(host.shape),
            "dtype": _dtype_name(host.dtype),
            "bytes": size,
        }
        if checksum:
            entry["sha256"] = _sha256(host)
        return entry

    def _record(self, record: dict[str, object]) -> None:
        manifest = self._manifest
        if manifest is None:
            raise RuntimeError("tensor_dump wrote a record before its first step")
        line = json.dumps(record, sort_keys=True) + "\n"
        manifest.write(line)
        manifest.flush()
        self._bytes += len(line.encode("utf-8"))
