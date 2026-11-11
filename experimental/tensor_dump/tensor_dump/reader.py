# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reads a dump directory: steps, requests, linear operands and attention
inputs with their rebuilt key/value history.

``TensorDump(path)`` parses every ``rank_<R>/manifest.jsonl`` under ``path``,
or, for a dump that ``tensor_dump.merge`` merged, every
``rank_<R>/manifest.json``; both layouts are read through the same API.
Tensors are loaded only when asked for, and each is checked against the shape
and dtype its manifest entry records.
"""

from __future__ import annotations

import json
import pickle
import re
from collections import Counter, OrderedDict
from collections.abc import Container, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

_FORMAT_VERSIONS: frozenset[int] = frozenset({2})
_MERGED_FORMAT_VERSIONS: frozenset[int] = frozenset({1})
_MANIFEST = "manifest.jsonl"
_MERGED_MANIFEST = "manifest.json"
_REQUEST_PHASES: tuple[str, ...] = ("prefill", "decode")
_RANK_DIR_RE: re.Pattern[str] = re.compile(r"rank_([0-9]+)")
_DIGITS_RE: re.Pattern[str] = re.compile(r"([0-9]+)")
# Merged files a reader keeps loaded at once.
_FILE_CACHE_SIZE = 4


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _rank_dirs(root: Path, manifest: str) -> list[int]:
    """The ranks ``R`` whose ``root/rank_<R>/<manifest>`` is a file, in order."""
    ranks: list[int] = []
    for child in root.iterdir() if root.is_dir() else []:
        match = _RANK_DIR_RE.fullmatch(child.name)
        if match is not None and (child / manifest).is_file():
            ranks.append(int(match.group(1)))
    return sorted(ranks)


def _natural_key(name: str) -> tuple[str | int, ...]:
    """Orders names with their digit runs compared as numbers."""
    return tuple(
        int(part) if index % 2 else part
        for index, part in enumerate(_DIGITS_RE.split(name))
    )


def _torch_load(path: Path) -> object:
    """``torch.load`` on the CPU, with only tensors and containers allowed.

    Raises:
        ValueError: if the file cannot be read as a torch file.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (RuntimeError, pickle.UnpicklingError, EOFError) as error:
        raise ValueError(f"{path} cannot be read as a torch file: {error}") from error


def _load(ref: TensorRef | None) -> torch.Tensor | None:
    if ref is None:
        return None
    return ref.load()


@dataclass(frozen=True)
class TensorRef:
    """One tensor file, as its manifest entry describes it.

    Attributes:
        path: The file.
        shape: The recorded shape.
        dtype: The recorded dtype name, e.g. ``"bfloat16"``.
        bytes: The file size, if recorded.
        sha256: The SHA-256 of the tensor's bytes, recorded for weights.
    """

    path: Path
    shape: tuple[int, ...]
    dtype: str
    bytes: int | None = None
    sha256: str | None = None

    def load(self) -> torch.Tensor:
        """The tensor, on the CPU.

        Raises:
            ValueError: if the file cannot be read, or holds another shape or
                dtype.
        """
        tensor = _torch_load(self.path)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{self.path} does not hold a tensor")
        return self._checked(tensor)

    def _checked(self, tensor: torch.Tensor) -> torch.Tensor:
        if tuple(tensor.shape) != self.shape or _dtype_name(tensor.dtype) != self.dtype:
            raise ValueError(
                f"{self.path} holds {_dtype_name(tensor.dtype)} "
                f"{tuple(tensor.shape)}, but its manifest entry records "
                f"{self.dtype} {self.shape}"
            )
        return tensor


class _FileCache:
    """The merged files a reader loaded last, each a dict of tensors."""

    def __init__(self) -> None:
        self._files: OrderedDict[Path, dict[str, object]] = OrderedDict()

    def tensor(self, path: Path, key: str) -> torch.Tensor:
        """Tensor ``key`` of the dict in ``path``.

        Raises:
            ValueError: if the file cannot be read, or holds no tensor ``key``.
        """
        data = self._files.get(path)
        if data is None:
            loaded = _torch_load(path)
            if not isinstance(loaded, dict):
                raise ValueError(f"{path} does not hold a dict of tensors")
            data = loaded
            self._files[path] = data
            if len(self._files) > _FILE_CACHE_SIZE:
                self._files.popitem(last=False)
        else:
            self._files.move_to_end(path)
        tensor = data.get(key)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{path} holds no tensor {key!r}")
        return tensor


@dataclass(frozen=True)
class _RowsRef(TensorRef):
    """Rows of tensors in merged files, concatenated in order.

    ``parts`` lists ``(file, key, start, stop)``: rows ``[start, stop)`` of
    tensor ``key`` of the dict in ``file``, all rows for ``None`` bounds.
    ``load`` returns a new tensor.
    """

    parts: tuple[tuple[Path, str, int | None, int | None], ...] = ()
    cache: _FileCache = field(default_factory=_FileCache, compare=False, repr=False)

    def load(self) -> torch.Tensor:
        """The rows, on the CPU.

        Raises:
            ValueError: if a file cannot be read or lacks the tensor, or the
                rows have another shape or dtype than recorded.
        """
        if not self.parts:
            raise ValueError(f"no rows of {self.path} to load")
        pieces = [
            self.cache.tensor(path, key)[start:stop]
            for path, key, start, stop in self.parts
        ]
        try:
            tensor = torch.cat(pieces)
        except RuntimeError as error:
            raise ValueError(
                f"the rows of {self.path} do not stack: {error}"
            ) from error
        return self._checked(tensor)


@dataclass(frozen=True)
class Run:
    """One run that wrote to a rank's directory.

    Attributes:
        rank: The rank.
        run: Index of the run in the rank's directory, from 0.
        first_step: Step id of the run's first forward; the run's step ids
            are at least this.
        options: The dump options the run used.
    """

    rank: int
    run: int
    first_step: int
    options: Mapping[str, object]


@dataclass(frozen=True)
class RequestSlice:
    """One request's part of one step.

    Attributes:
        step: The step id.
        request: Index of the request in the step's batch.
        q_len: Query tokens of the request in this step.
        kv_len: Keys and values the request attends to, cached and new.
        q_start_pos: Position of the first query token; ``kv_len - q_len``.
        token_start: Row of the request's first token in the step's linear
            tensors.
        phase: ``"decode"`` for one query token after position 0, else
            ``"prefill"``.
    """

    step: int
    request: int
    q_len: int
    kv_len: int
    q_start_pos: int
    token_start: int
    phase: str


@dataclass(frozen=True)
class Step:
    """One written step.

    Attributes:
        rank: The rank.
        run: Index of the run that wrote it.
        step: The step id, unique within the rank's directory.
        phase: ``"prefill"``, ``"decode"`` or ``"mixed"``.
        num_actual_tokens: Rows of the step's linear tensors.
        requests: The step's requests, in batch order.
    """

    rank: int
    run: int
    step: int
    phase: str
    num_actual_tokens: int
    requests: tuple[RequestSlice, ...]


@dataclass(frozen=True)
class Request:
    """One request, followed across the steps of a run.

    Attributes:
        rank: The rank.
        run: Index of the run the request belongs to.
        id: Index of the request among the rank's requests, in order of their
            first step.
        slices: The request's part of each of its steps, in step order.
    """

    rank: int
    run: int
    id: int
    slices: tuple[RequestSlice, ...]

    @property
    def steps(self) -> tuple[int, ...]:
        """The request's step ids."""
        return tuple(piece.step for piece in self.slices)

    @property
    def complete(self) -> bool:
        """Whether the request's first step is in the dump, i.e. its first
        slice starts at position 0."""
        return self.slices[0].q_start_pos == 0

    @property
    def prompt_len(self) -> int:
        """Query tokens over the request's prefill steps."""
        return sum(piece.q_len for piece in self.slices if piece.phase == "prefill")

    @property
    def generated_len(self) -> int:
        """Decode steps of the request: the generated tokens fed back as
        inputs. The last generated token is never an input, so a request that
        ran to completion generated one more."""
        return sum(1 for piece in self.slices if piece.phase == "decode")


@dataclass(frozen=True)
class LinearPair:
    """The files of one linear layer at one step.

    Attributes:
        rank, step, layer: Where the tensors come from.
        phase: The step's phase.
        activation, output: ``[num_actual_tokens, ...]`` input and output, if
            written.
        weight, bias: The layer's weight and bias, if any run wrote them.
    """

    rank: int
    step: int
    phase: str
    layer: str
    activation: TensorRef | None
    output: TensorRef | None
    weight: TensorRef | None
    bias: TensorRef | None


@dataclass(frozen=True)
class LinearTensors:
    """The loaded tensors of a ``LinearPair``; ``None`` where none was written."""

    activation: torch.Tensor | None
    weight: torch.Tensor | None
    bias: torch.Tensor | None
    output: torch.Tensor | None


@dataclass(frozen=True)
class AttentionReplay:
    """What an attention record says about how its output was computed.

    Attributes:
        q_start_pos: Position of the first query, the causal offset.
        q_len, kv_len: Query tokens, and keys and values attended.
        causal: Query ``i`` attends to keys up to ``q_start_pos + i``.
        scale: The softmax scale applied to ``Q @ K^T``.
        sliding_window: vLLM's window ``W``: query ``i`` attends to keys
            ``q_start_pos + i - W + 1 .. q_start_pos + i``; ``None`` for full
            attention.
        logits_soft_cap: The soft cap on ``Q @ K^T``, or ``None``.
        num_q_heads, num_kv_heads: Per-rank head counts.
        head_size, head_size_v: Head sizes of Q and K, and of V and the
            output.
        dtype: The dtype of Q as the layer received it.
        kv_cache_dtype: vLLM's KV-cache dtype setting of the layer.
        backend: vLLM's name for the attention backend, if known.
    """

    q_start_pos: int
    q_len: int
    kv_len: int
    causal: bool
    scale: float
    sliding_window: int | None
    logits_soft_cap: float | None
    num_q_heads: int
    num_kv_heads: int
    head_size: int
    head_size_v: int
    dtype: str
    kv_cache_dtype: str
    backend: str | None


@dataclass(frozen=True)
class AttentionTensors:
    """One request's attention at one step of one layer.

    Attributes:
        q: ``[q_len, num_q_heads, head_size]``, if written.
        k, v: The request's full keys and values, rows ``[0, kv_len)``,
            ``[kv_len, num_kv_heads, head_size]`` (``head_size_v`` for V).
            They may share memory with a cache; do not modify them in place.
        output: ``[q_len, num_q_heads, head_size_v]``, if written.
        replay: The record's replay fields.
    """

    q: torch.Tensor | None
    k: torch.Tensor
    v: torch.Tensor
    output: torch.Tensor | None
    replay: AttentionReplay


@dataclass(frozen=True)
class _AttentionRecord:
    tensors: Mapping[str, TensorRef]
    replay: AttentionReplay


class _History:
    """A request's keys and values, grown one step at a time into buffers
    that double in size, so rebuilding n steps costs O(total rows)."""

    def __init__(self) -> None:
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None
        self.length = 0
        self.slices = 0

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        end = self.length + k.shape[0]
        buffer_k, buffer_v = self.k, self.v
        if buffer_k is None or buffer_v is None or buffer_k.shape[0] < end:
            capacity = end
            if buffer_k is not None:
                capacity = max(end, 2 * buffer_k.shape[0])
            new_k = k.new_empty((capacity, *k.shape[1:]))
            new_v = v.new_empty((capacity, *v.shape[1:]))
            if buffer_k is not None and buffer_v is not None:
                new_k[: self.length] = buffer_k[: self.length]
                new_v[: self.length] = buffer_v[: self.length]
            buffer_k, buffer_v = new_k, new_v
            self.k, self.v = new_k, new_v
        buffer_k[self.length : end] = k
        buffer_v[self.length : end] = v
        self.length = end
        self.slices += 1


def _entry(root: Path, data: Mapping[str, Any]) -> TensorRef:
    size = data.get("bytes")
    checksum = data.get("sha256")
    return TensorRef(
        path=root / str(data["file"]),
        shape=tuple(int(n) for n in data["shape"]),
        dtype=str(data["dtype"]),
        bytes=None if size is None else int(size),
        sha256=None if checksum is None else str(checksum),
    )


def _entries(root: Path, record: Mapping[str, Any]) -> dict[str, TensorRef]:
    return {str(name): _entry(root, data) for name, data in record["tensors"].items()}


def _replay(data: Mapping[str, Any]) -> AttentionReplay:
    window = data["sliding_window"]
    soft_cap = data["logits_soft_cap"]
    backend = data.get("backend")
    return AttentionReplay(
        q_start_pos=int(data["q_start_pos"]),
        q_len=int(data["q_len"]),
        kv_len=int(data["kv_len"]),
        causal=bool(data["causal"]),
        scale=float(data["scale"]),
        sliding_window=None if window is None else int(window),
        logits_soft_cap=None if soft_cap is None else float(soft_cap),
        num_q_heads=int(data["num_q_heads"]),
        num_kv_heads=int(data["num_kv_heads"]),
        head_size=int(data["head_size"]),
        head_size_v=int(data["head_size_v"]),
        dtype=str(data["dtype"]),
        kv_cache_dtype=str(data["kv_cache_dtype"]),
        backend=None if backend is None else str(backend),
    )


def _layer_replay(entry: Mapping[str, Any], row: int) -> Mapping[str, Any]:
    """A merged attention layer's per-layer replay fields at ``row``."""
    replay = entry.get("replay")
    if replay is not None:
        return replay
    for item in entry["replay_ranges"]:
        start, stop = item["rows"]
        if int(start) <= row < int(stop):
            return item["replay"]
    raise ValueError(f"no replay range of {entry['file']} holds row {row}")


def _slice(step: int, data: Mapping[str, Any]) -> RequestSlice:
    return RequestSlice(
        step=step,
        request=int(data["request"]),
        q_len=int(data["q_len"]),
        kv_len=int(data["kv_len"]),
        q_start_pos=int(data["q_start_pos"]),
        token_start=int(data["token_start"]),
        phase=str(data["phase"]),
    )


class TensorDump:
    """A dump directory written by ``tensor_dump``, possibly by several runs,
    or the merge of one that ``tensor_dump.merge`` wrote.

    A directory that holds any ``rank_<R>/manifest.jsonl`` is read as written;
    otherwise its ``rank_<R>/manifest.json`` are read as a merged dump. Both
    give the same steps, requests and tensors. In a merged dump, a step's
    linear tensors are assembled from the files of its requests, and only
    for steps whose every request was merged.

    Args:
        path: The dump directory.

    Raises:
        ValueError: if ``path`` holds no manifest, or a manifest is malformed
            or written in an unsupported format.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._runs: dict[int, list[Run]] = {}
        self._steps: dict[int, dict[int, Step]] = {}
        self._linear: dict[tuple[int, str, int], dict[str, TensorRef]] = {}
        self._weights: dict[tuple[int, str], dict[str, TensorRef]] = {}
        self._attention: dict[tuple[int, str, int, int], _AttentionRecord] = {}
        self._requests: dict[int, tuple[Request, ...]] = {}
        self._histories: OrderedDict[tuple[int, str, int], _History] = OrderedDict()
        self._weight_cache: tuple[tuple[int, str], LinearTensors] | None = None
        self._files = _FileCache()
        self._phase_files: dict[tuple[int, int, str, str], dict[str, TensorRef]] = {}
        ranks = _rank_dirs(self.path, _MANIFEST)
        merged = _rank_dirs(self.path, _MERGED_MANIFEST)
        if ranks:
            for rank in ranks:
                self._read(rank, self.path / f"rank_{rank}")
        elif merged:
            for rank in merged:
                self._read_merged(rank, self.path / f"rank_{rank}")
        else:
            raise ValueError(
                f"{self.path} holds no rank_<R>/{_MANIFEST} or "
                f"rank_<R>/{_MERGED_MANIFEST}"
            )
        layers = len({key[:2] for key in self._attention})
        per_step = Counter(key[:3] for key in self._attention)
        self._history_limit = max(1, layers * max(per_step.values(), default=1))

    @property
    def ranks(self) -> tuple[int, ...]:
        """The ranks that wrote a manifest."""
        return tuple(sorted(self._runs))

    def runs(self, *, rank: int = 0) -> tuple[Run, ...]:
        """The runs that wrote to ``rank``'s directory, in order."""
        return tuple(self._runs_of(rank))

    def steps(self, *, rank: int = 0) -> tuple[Step, ...]:
        """``rank``'s written steps, in step-id order."""
        return tuple(self._steps_of(rank).values())

    def requests(self, *, rank: int = 0) -> tuple[Request, ...]:
        """``rank``'s requests, followed across steps.

        Within each run, in step order, a request slice at position 0 starts
        a request, and a slice at position ``p > 0`` continues the request
        whose last slice ended at ``kv_len == p``. If several requests end at
        ``p``, the one that was in the previous written step is taken. A slice
        that continues no request starts an incomplete request, whose earlier
        steps are not in the dump.

        Raises:
            ValueError: if a slice could continue several requests, or several
                slices of a step continue the same request.
        """
        requests = self._requests.get(rank)
        if requests is None:
            requests = self._segment(rank)
            self._requests[rank] = requests
        return requests

    def linear_pairs(
        self,
        *,
        rank: int | None = None,
        steps: Container[int] | None = None,
        phase: str | None = None,
        layer: str | None = None,
    ) -> Iterator[LinearPair]:
        """The linear layers' files, ordered by rank, layer and step.

        Args:
            rank: Only this rank.
            steps: Only these step ids.
            phase: Only steps of this phase.
            layer: Only layers this regular expression matches with
                ``re.match``.
        """
        pattern = None if layer is None else re.compile(layer)
        for key in sorted(self._linear):
            pair_rank, pair_layer, step = key
            if rank is not None and pair_rank != rank:
                continue
            if steps is not None and step not in steps:
                continue
            if pattern is not None and not pattern.match(pair_layer):
                continue
            pair = self._pair(pair_rank, pair_layer, step)
            if phase is not None and pair.phase != phase:
                continue
            yield pair

    def linear(self, layer: str, step: int, *, rank: int = 0) -> LinearTensors:
        """The activation, weight, bias and output of ``layer`` at ``step``.

        Raises:
            KeyError: if the layer wrote nothing at that step.
        """
        if (rank, layer, step) not in self._linear:
            raise KeyError(f"no linear record for layer {layer!r} at step {step}")
        return self.load_linear(self._pair(rank, layer, step))

    def load_linear(self, pair: LinearPair) -> LinearTensors:
        """Load a pair's tensors. The weight and bias of the last layer loaded
        are kept, so pairs of one layer in a row load them once."""
        key = (pair.rank, pair.layer)
        cached = self._weight_cache
        if cached is None or cached[0] != key:
            weights = LinearTensors(None, _load(pair.weight), _load(pair.bias), None)
            cached = (key, weights)
            self._weight_cache = cached
        return LinearTensors(
            activation=_load(pair.activation),
            weight=cached[1].weight,
            bias=cached[1].bias,
            output=_load(pair.output),
        )

    def attention(self, layer: str, request: Request, step: int) -> AttentionTensors:
        """``request``'s Q and output at ``step`` of ``layer``, with its full
        keys and values rebuilt from the new rows of its steps up to ``step``.

        The rebuilt history is cached per layer and request until the
        request's last step is read. At most ``L * R`` histories are kept,
        the least recently used dropped first, where ``L`` counts the
        attention layers of all ranks and ``R`` is the most requests one
        layer recorded in a step. Reading every step of the requests in
        step order, with any interleaving of requests and layers, so costs
        O(their total rows) when each request is in every step from its
        first to its last.

        Raises:
            KeyError: if ``step`` is not one of the request's steps, or
                ``layer`` wrote no record for it.
            ValueError: if part of the history is not in the dump: the
                request is incomplete, or one of its earlier steps has no
                ``k`` and ``v`` for ``layer``; or if the new rows of its
                steps up to one of them do not add up to that step's
                ``kv_len``.
        """
        position = next(
            (i for i, piece in enumerate(request.slices) if piece.step == step), None
        )
        if position is None:
            raise KeyError(f"step {step} is not a step of request {request.id}")
        piece = request.slices[position]
        record = self._attention.get((request.rank, layer, step, piece.request))
        if record is None:
            raise KeyError(
                f"layer {layer!r} wrote no attention record for request "
                f"{request.id} at step {step}"
            )
        k, v = self._history(layer, request, position)
        return AttentionTensors(
            q=_load(record.tensors.get("q")),
            k=k,
            v=v,
            output=_load(record.tensors.get("output")),
            replay=record.replay,
        )

    def phase_tensors(
        self, layer: str, request: Request | int, phase: str, *, rank: int = 0
    ) -> dict[str, torch.Tensor]:
        """Every tensor ``layer`` wrote for ``request`` in ``phase``, its rows
        of each step stacked along dim 0 in step order: linear ``input`` and
        ``output``, attention ``q``, ``k``, ``v`` and ``output`` (``k`` and
        ``v`` hold the new rows of each step, so the request's full keys at
        a step are its rows of every step up to it).

        Args:
            layer: The layer.
            request: A request, or its ``id`` among ``rank``'s requests.
            phase: ``"prefill"`` or ``"decode"``.
            rank: The rank of a request given by ``id``.

        Raises:
            KeyError: if the request has no rows of ``phase``, or ``layer``
                wrote no record at one of its steps.
            ValueError: for another phase, or tensors whose names, dtypes or
                shapes differ between steps.
        """
        if isinstance(request, int):
            request = self._request(rank, request)
        if phase not in _REQUEST_PHASES:
            raise ValueError(f"phase {phase!r} is not one of {_REQUEST_PHASES}")
        merged = self._phase_files.get((request.rank, request.id, phase, layer))
        if merged is not None:
            return {name: ref.load() for name, ref in sorted(merged.items())}
        slices = [piece for piece in request.slices if piece.phase == phase]
        if not slices:
            raise KeyError(f"request {request.id} has no {phase} rows")
        parts: dict[str, list[torch.Tensor]] = {}
        for piece in slices:
            tensors = self._slice_tensors(layer, request, piece)
            if parts and sorted(tensors) != sorted(parts):
                raise ValueError(
                    f"layer {layer!r} wrote {sorted(tensors)} at step "
                    f"{piece.step} but {sorted(parts)} at an earlier step"
                )
            for name, tensor in tensors.items():
                parts.setdefault(name, []).append(tensor)
        try:
            return {name: torch.cat(pieces) for name, pieces in sorted(parts.items())}
        except RuntimeError as error:
            raise ValueError(
                f"the tensors of layer {layer!r} differ between steps: {error}"
            ) from error

    def attention_layers(self, *, rank: int = 0) -> tuple[str, ...]:
        """The attention layers with records on ``rank``, ordered by name with
        digit runs compared as numbers (``layers.2`` before ``layers.10``)."""
        self._runs_of(rank)
        names = {layer for (owner, layer, _, _) in self._attention if owner == rank}
        return tuple(sorted(names, key=_natural_key))

    def _attention_records(
        self, rank: int
    ) -> Iterator[tuple[str, int, int, Mapping[str, TensorRef]]]:
        """``(layer, step, request, tensors)`` of each of ``rank``'s attention
        records, ``request`` being the index in the step's batch."""
        for (owner, layer, step, request), record in self._attention.items():
            if owner == rank:
                yield layer, step, request, record.tensors

    def _weight_refs(self, rank: int) -> dict[str, Mapping[str, TensorRef]]:
        """``rank``'s weight files per layer."""
        return {
            layer: refs
            for (owner, layer), refs in self._weights.items()
            if owner == rank
        }

    def _request(self, rank: int, request: int) -> Request:
        for candidate in self.requests(rank=rank):
            if candidate.id == request:
                return candidate
        raise KeyError(f"rank {rank} has no request {request}")

    def _slice_tensors(
        self, layer: str, request: Request, piece: RequestSlice
    ) -> dict[str, torch.Tensor]:
        linear = self._linear.get((request.rank, layer, piece.step))
        if linear is not None:
            rows = slice(piece.token_start, piece.token_start + piece.q_len)
            return {name: ref.load()[rows] for name, ref in linear.items()}
        record = self._attention.get((request.rank, layer, piece.step, piece.request))
        if record is not None:
            return {name: ref.load() for name, ref in record.tensors.items()}
        raise KeyError(
            f"layer {layer!r} wrote no record for request {request.id} at step "
            f"{piece.step}"
        )

    def _runs_of(self, rank: int) -> list[Run]:
        runs = self._runs.get(rank)
        if runs is None:
            raise KeyError(f"rank {rank} wrote no manifest; ranks: {self.ranks}")
        return runs

    def _steps_of(self, rank: int) -> dict[int, Step]:
        self._runs_of(rank)
        return self._steps[rank]

    def _read(self, rank: int, root: Path) -> None:
        path = root / "manifest.jsonl"
        runs: list[Run] = []
        steps: dict[int, Step] = {}
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                self._record(rank, root, json.loads(line), runs, steps)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"line {number} of {path} is not a valid record: {error!r}"
                ) from error
        self._runs[rank] = runs
        self._steps[rank] = dict(sorted(steps.items()))

    def _record(
        self,
        rank: int,
        root: Path,
        record: Mapping[str, Any],
        runs: list[Run],
        steps: dict[int, Step],
    ) -> None:
        kind = record["record"]
        if kind == "run":
            version = record["format_version"]
            if version not in _FORMAT_VERSIONS:
                raise ValueError(
                    f"{root} holds a run in format {version}; this reader reads "
                    f"formats {sorted(_FORMAT_VERSIONS)}"
                )
            runs.append(
                Run(
                    rank=rank,
                    run=int(record["run"]),
                    first_step=int(record["first_step"]),
                    options=dict(record["options"]),
                )
            )
            return
        if not runs:
            raise ValueError(f"{root}: a {kind!r} record precedes every run record")
        step = int(record["step"])
        if kind == "step":
            steps[step] = Step(
                rank=rank,
                run=runs[-1].run,
                step=step,
                phase=str(record["phase"]),
                num_actual_tokens=int(record["num_actual_tokens"]),
                requests=tuple(_slice(step, data) for data in record["requests"]),
            )
        elif kind == "linear":
            self._linear[(rank, str(record["layer"]), step)] = _entries(root, record)
        elif kind == "weight":
            self._weights[(rank, str(record["layer"]))] = _entries(root, record)
        elif kind == "attention":
            key = (rank, str(record["layer"]), step, int(record["request"]))
            self._attention[key] = _AttentionRecord(
                _entries(root, record), _replay(record["replay"])
            )

    def _read_merged(self, rank: int, root: Path) -> None:
        path = root / _MERGED_MANIFEST
        try:
            self._merged_manifest(rank, root, json.loads(path.read_text("utf-8")))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{path} is not a valid merged manifest: {error!r}"
            ) from error

    def _merged_manifest(
        self, rank: int, root: Path, manifest: Mapping[str, Any]
    ) -> None:
        version = manifest["merged_format_version"]
        if version not in _MERGED_FORMAT_VERSIONS:
            raise ValueError(
                f"{root} holds a dump in merged format {version}; this reader "
                f"reads merged formats {sorted(_MERGED_FORMAT_VERSIONS)}"
            )
        self._runs[rank] = [
            Run(
                rank=rank,
                run=int(record["run"]),
                first_step=int(record["first_step"]),
                options=dict(record["options"]),
            )
            for record in manifest["runs"]
        ]
        steps: dict[int, Step] = {}
        for record in manifest["steps"]:
            step = int(record["step"])
            steps[step] = Step(
                rank=rank,
                run=int(record["run"]),
                step=step,
                phase=str(record["phase"]),
                num_actual_tokens=int(record["num_actual_tokens"]),
                requests=tuple(_slice(step, data) for data in record["requests"]),
            )
        self._steps[rank] = dict(sorted(steps.items()))
        for layer, entry in manifest["weights"].items():
            weights: dict[str, TensorRef] = {
                str(name): self._rows_ref(root / str(entry["file"]), name, data, None)
                for name, data in entry["tensors"].items()
            }
            self._weights[(rank, str(layer))] = weights
        linear: dict[tuple[str, int], list[tuple[int, dict[str, _RowsRef]]]] = {}
        requests = []
        for entry in manifest["requests"]:
            request = int(entry["request"])
            slices: list[RequestSlice] = []
            for phase, data in entry["phases"].items():
                segments = data["segments"]
                for segment in segments:
                    slices.append(
                        _slice(int(segment["step"]), {**segment, "phase": phase})
                    )
                for layer, layer_entry in data["layers"].items():
                    key = (rank, request, str(phase), str(layer))
                    self._merged_layer(root, key, layer_entry, segments, linear)
            requests.append(
                Request(
                    rank=rank,
                    run=int(entry["run"]),
                    id=request,
                    slices=tuple(sorted(slices, key=lambda piece: piece.step)),
                )
            )
        self._requests[rank] = tuple(sorted(requests, key=lambda r: r.id))
        self._merged_linear(rank, steps, linear)

    def _rows_ref(
        self,
        file: Path,
        name: object,
        data: Mapping[str, Any],
        rows: tuple[int, int] | None,
    ) -> _RowsRef:
        shape = tuple(int(n) for n in data["shape"])
        start: int | None = None
        stop: int | None = None
        if rows is not None:
            start, stop = rows
            shape = (stop - start, *shape[1:])
        checksum = data.get("sha256")
        return _RowsRef(
            path=file,
            shape=shape,
            dtype=str(data["dtype"]),
            sha256=None if checksum is None else str(checksum),
            parts=((file, str(name), start, stop),),
            cache=self._files,
        )

    def _merged_layer(
        self,
        root: Path,
        key: tuple[int, int, str, str],
        entry: Mapping[str, Any],
        segments: list[Mapping[str, Any]],
        linear: dict[tuple[str, int], list[tuple[int, dict[str, _RowsRef]]]],
    ) -> None:
        """Index one merged file by step, as the per-step records would be."""
        rank, _, _, layer = key
        file = root / str(entry["file"])
        tensors = entry["tensors"]
        self._phase_files[key] = {
            str(name): self._rows_ref(file, name, data, None)
            for name, data in tensors.items()
        }
        kind = entry["kind"]
        if kind not in ("linear", "attention"):
            raise ValueError(f"{file} holds a layer of kind {kind!r}")
        for segment in segments:
            start = int(segment["row_start"])
            rows = (start, start + int(segment["q_len"]))
            step = int(segment["step"])
            refs = {
                str(name): self._rows_ref(file, name, data, rows)
                for name, data in tensors.items()
            }
            if kind == "linear":
                linear.setdefault((layer, step), []).append(
                    (int(segment["token_start"]), refs)
                )
                continue
            replay = {
                **_layer_replay(entry, start),
                "q_start_pos": segment["q_start_pos"],
                "q_len": segment["q_len"],
                "kv_len": segment["kv_len"],
            }
            record_key = (rank, layer, step, int(segment["request"]))
            self._attention[record_key] = _AttentionRecord(refs, _replay(replay))

    def _merged_linear(
        self,
        rank: int,
        steps: Mapping[int, Step],
        linear: Mapping[tuple[str, int], list[tuple[int, dict[str, _RowsRef]]]],
    ) -> None:
        """A step's linear records: its requests' rows in row order, for
        steps whose requests' files cover every row."""
        for (layer, step), pieces in linear.items():
            written = steps.get(step)
            if written is None or len(pieces) != len(written.requests):
                continue
            pieces = sorted(pieces, key=lambda piece: piece[0])
            names = sorted(pieces[0][1])
            end = 0
            for token_start, refs in pieces:
                if not names or token_start != end or sorted(refs) != names:
                    end = -1
                    break
                end += refs[names[0]].shape[0]
            if end != written.num_actual_tokens:
                continue
            tensors: dict[str, TensorRef] = {}
            for name in names:
                first = pieces[0][1][name]
                tensors[name] = _RowsRef(
                    path=first.path,
                    shape=(end, *first.shape[1:]),
                    dtype=first.dtype,
                    parts=tuple(
                        part for _, refs in pieces for part in refs[name].parts
                    ),
                    cache=self._files,
                )
            self._linear[(rank, layer, step)] = tensors

    def _pair(self, rank: int, layer: str, step: int) -> LinearPair:
        tensors = self._linear[(rank, layer, step)]
        weights = self._weights.get((rank, layer), {})
        written = self._steps_of(rank).get(step)
        return LinearPair(
            rank=rank,
            step=step,
            phase="" if written is None else written.phase,
            layer=layer,
            activation=tensors.get("input"),
            output=tensors.get("output"),
            weight=weights.get("weight"),
            bias=weights.get("bias"),
        )

    def _segment(self, rank: int) -> tuple[Request, ...]:
        slices: list[list[RequestSlice]] = []
        runs: list[int] = []
        # Per run: index of each open request, its next position, and the
        # step it was last in.
        open_requests: dict[int, tuple[int, int]] = {}
        current_run: int | None = None
        previous_step: int | None = None
        for step in self._steps_of(rank).values():
            if step.run != current_run:
                current_run = step.run
                open_requests = {}
                previous_step = None
            claimed: dict[int, int] = {}
            for piece in step.requests:
                index = None
                if piece.q_start_pos > 0:
                    index = self._continued(
                        piece, open_requests, previous_step, claimed
                    )
                if index is None:
                    index = len(slices)
                    slices.append([])
                    runs.append(step.run)
                claimed[index] = piece.request
                slices[index].append(piece)
            for index in claimed:
                last = slices[index][-1]
                open_requests[index] = (last.kv_len, step.step)
            previous_step = step.step
        return tuple(
            Request(rank=rank, run=run, id=index, slices=tuple(pieces))
            for index, (run, pieces) in enumerate(zip(runs, slices))
        )

    @staticmethod
    def _continued(
        piece: RequestSlice,
        open_requests: Mapping[int, tuple[int, int]],
        previous_step: int | None,
        claimed: Mapping[int, int],
    ) -> int | None:
        candidates = [
            index
            for index, (position, _) in open_requests.items()
            if position == piece.q_start_pos
        ]
        if len(candidates) > 1:
            candidates = [
                index
                for index in candidates
                if open_requests[index][1] == previous_step
            ]
        if len(candidates) > 1:
            raise ValueError(
                f"request {piece.request} of step {piece.step} at position "
                f"{piece.q_start_pos} could continue {len(candidates)} requests; "
                "requests cannot be told apart"
            )
        if not candidates:
            return None
        index = candidates[0]
        if index in claimed:
            raise ValueError(
                f"requests {claimed[index]} and {piece.request} of step "
                f"{piece.step} both continue the request that ended at position "
                f"{piece.q_start_pos}; requests cannot be told apart"
            )
        return index

    def _history(
        self, layer: str, request: Request, position: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not request.complete:
            start = request.slices[0].q_start_pos
            raise ValueError(
                f"request {request.id} starts at position {start} in the dump: "
                f"rows [0, {start}) of its keys and values are in steps that "
                "were not written"
            )
        key = (request.rank, layer, request.id)
        history = self._histories.get(key)
        if history is None:
            history = _History()
        else:
            self._histories.move_to_end(key)
        while history.slices <= position:
            piece = request.slices[history.slices]
            if history.length + piece.q_len != piece.kv_len:
                raise ValueError(
                    f"request {request.id} has kv_len {piece.kv_len} at step "
                    f"{piece.step}, but layer {layer!r} rebuilds "
                    f"{history.length + piece.q_len} rows of keys and values "
                    "up to it"
                )
            record = self._attention.get(
                (request.rank, layer, piece.step, piece.request)
            )
            k_ref: TensorRef | None = None
            v_ref: TensorRef | None = None
            if record is not None:
                k_ref = record.tensors.get("k")
                v_ref = record.tensors.get("v")
            if k_ref is None or v_ref is None:
                raise ValueError(
                    f"layer {layer!r} wrote no k and v for request {request.id} at "
                    f"step {piece.step}, which its keys and values up to "
                    f"position {request.slices[position].kv_len} need"
                )
            k, v = k_ref.load(), v_ref.load()
            if k.shape[0] != piece.q_len or v.shape[0] != piece.q_len:
                raise ValueError(
                    f"k and v of layer {layer!r} at step {piece.step} have "
                    f"{k.shape[0]} and {v.shape[0]} rows, not the step's "
                    f"{piece.q_len} new tokens"
                )
            history.append(k, v)
        buffer_k, buffer_v = history.k, history.v
        if buffer_k is None or buffer_v is None:
            raise ValueError(f"request {request.id} has no steps")
        if position == len(request.slices) - 1:
            self._histories.pop(key, None)
        else:
            self._histories[key] = history
            while len(self._histories) > self._history_limit:
                self._histories.popitem(last=False)
        kv_len = request.slices[position].kv_len
        return buffer_k[:kv_len], buffer_v[:kv_len]


def _reference_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, replay: AttentionReplay
) -> torch.Tensor:
    """Causal attention in fp32 from Q ``[q_len, Hq, D]`` and the full K and V
    ``[kv_len, Hkv, D]``, with the record's scale, window and soft cap; returns
    ``[q_len, Hq, Dv]``."""
    group = replay.num_q_heads // replay.num_kv_heads
    queries = q.float().transpose(0, 1)
    keys = k.float().repeat_interleave(group, dim=1).transpose(0, 1)
    values = v.float().repeat_interleave(group, dim=1).transpose(0, 1)
    scores = queries @ keys.transpose(1, 2) * replay.scale
    soft_cap = replay.logits_soft_cap
    if soft_cap is not None:
        scores = soft_cap * torch.tanh(scores / soft_cap)
    query_pos = torch.arange(replay.q_len)[:, None] + replay.q_start_pos
    key_pos = torch.arange(replay.kv_len)[None, :]
    allowed = key_pos <= query_pos
    window = replay.sliding_window
    if window is not None:
        allowed &= key_pos > query_pos - window
    scores = scores.masked_fill(~allowed, float("-inf"))
    return (torch.softmax(scores, dim=-1) @ values).transpose(0, 1)


__all__ = [
    "AttentionReplay",
    "AttentionTensors",
    "LinearPair",
    "LinearTensors",
    "Request",
    "RequestSlice",
    "Run",
    "Step",
    "TensorDump",
    "TensorRef",
]
