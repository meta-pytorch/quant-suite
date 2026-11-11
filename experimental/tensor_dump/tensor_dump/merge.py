# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Merges a dump's per-step files into one file per request, phase and layer.

``merge(src, dst)`` reads the dump in ``src`` and writes, for every rank::

    rank_<R>/manifest.json
    rank_<R>/weights/<layer>.pt               {"weight": W, "bias": b}
    rank_<R>/req_<NNNN>/<phase>/<layer>.pt    linear: {"input", "output"}
                                              attention: {"q", "k", "v", "output"}

Each ``.pt`` file holds a dict of tensors saved with ``torch.save``; a dict
holds only the tensors the dump wrote. ``<NNNN>`` is the request's index among
the rank's requests, in order of their first step, as
``tensor_dump.reader.TensorDump.requests`` numbers them. ``<phase>`` is
``prefill`` or ``decode``: a request's rows of a step go to the phase the
request has in that step, so the requests of a ``mixed`` step are split too.

Every tensor stacks the request's rows of the phase along dim 0, in step
order, which is position order: linear ``input`` ``[T, in]`` and ``output``
``[T, out]``; attention ``q`` and ``output`` ``[T, num_q_heads, head_size]``,
``k`` and ``v`` ``[T, num_kv_heads, head_size]``. Prefill rows are the prompt
rows of every chunk; decode rows are one per decode step. As the dump writes
only the new rows of K and V, a request whose prefill rows all come before its
decode rows has the full keys of its ``i``-th decode row in
``cat([prefill["k"], decode["k"][: i + 1]])``.

``manifest.json`` (sorted keys) holds:

- ``merged_format_version`` and ``source_format_versions``; ``rank``;
- ``runs``: the dump's ``run`` records; ``steps``: its ``step`` records, each
  with the ``run`` that wrote it;
- ``weights``: per layer, the ``file``, its ``bytes``, the ``step`` the dump
  wrote it at, and ``tensors``: ``shape``, ``dtype`` and ``sha256`` of each;
- ``requests``: per request, ``request``, ``run``, ``steps``,
  ``prompt_len``, ``generated_len``, ``starts_at_zero`` (its first step is
  in the dump), ``may_continue`` (the dump may have stopped writing before
  its last step: its run did not write the next step, and a step filter, the
  step cap or the byte budget would have left that step out), ``complete``
  (``starts_at_zero`` and not ``may_continue``), ``prefill_before_decode``,
  and ``phases``, which maps each phase to:

  - ``rows``: ``T``;
  - ``segments``: the request's part of each step, in row order:
    ``row_start``, ``q_len`` (its rows), ``step``, ``run``, ``request`` (its
    index in the step's batch), ``token_start`` (its first row in the step's
    linear tensors), ``q_start_pos`` and ``kv_len``. Row
    ``row_start + j`` has position ``q_start_pos + j``;
  - ``layers``: per layer, ``kind`` (``linear`` or ``attention``), ``file``,
    ``bytes`` and ``tensors`` (``shape`` and ``dtype`` of each), and for
    attention the dump's ``replay`` fields other than ``q_start_pos``,
    ``q_len``, ``kv_len`` and ``kv_rows``, which the segments give. They are
    the same at every step of a layer; if not, ``replay_ranges`` lists them per
    half-open range of ``rows`` instead of ``replay``.

- ``source_deleted``: whether the dump's manifest has been deleted.

Every merged file is read back and compared bit for bit with the dump's
tensors it came from before the manifest lists it. With ``delete_source``, a
dump file is deleted only once every file merged from it is verified: a step
directory once every request with rows in it is, a weight once its merged copy
is, and the dump's manifest last, once nothing else is left. A merge that
stops part way can be run again: requests and weights the destination
manifest lists are checked against the dump and skipped. With
``delete_source``, the merged files of those requests and weights are read
again and compared bit for bit with the dump's files still there before any of
them is deleted, so only bytes verified against the merged copy are ever
deleted.

``merge`` is I/O-bound and leaves torch's intra-op thread count as the caller
set it; ``python -m tensor_dump.merge`` caps it (``--threads``) to avoid
oversubscription.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from tensor_dump.reader import (
    _dtype_name,
    _rank_dirs,
    LinearPair,
    Request,
    RequestSlice,
    TensorDump,
    TensorRef,
)

logger: logging.Logger = logging.getLogger(__name__)

_MERGED_FORMAT_VERSION = 1
_PHASES: tuple[str, ...] = ("prefill", "decode")
_SOURCE_MANIFEST = "manifest.jsonl"
_MANIFEST = "manifest.json"
# Replay fields that vary per step; the merged manifest derives them from
# the segments.
_STEP_REPLAY_FIELDS: frozenset[str] = frozenset(
    {"q_start_pos", "q_len", "kv_len", "kv_rows"}
)


def _sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(_raw_bytes(tensor).numpy().data).hexdigest()


def _raw_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().reshape(-1).view(torch.uint8)


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Same dtype, shape and bytes; unlike ``torch.equal``, NaNs compare equal
    to themselves and ``-0.0`` differs from ``0.0``."""
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and torch.equal(_raw_bytes(a), _raw_bytes(b))
    )


def _load_file(path: Path) -> object:
    return torch.load(path, map_location="cpu", weights_only=True)


def _load_ref(ref: TensorRef, *, missing_ok: bool) -> torch.Tensor:
    """``ref``'s tensor; with ``missing_ok``, a meta tensor of its shape if
    its file is gone."""
    if missing_ok and not ref.path.exists():
        return torch.empty(ref.shape, device="meta")
    return ref.load()


def _has_data(parts: Mapping[str, Sequence[torch.Tensor]]) -> bool:
    return any(not piece.is_meta for pieces in parts.values() for piece in pieces)


def _save(path: Path, data: object) -> None:
    """Write ``data`` to ``path`` through a temporary file, synced to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as file:
        if isinstance(data, bytes):
            file.write(data)
        else:
            torch.save(data, file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _tensor_entry(tensor: torch.Tensor) -> dict[str, object]:
    return {"shape": list(tensor.shape), "dtype": _dtype_name(tensor.dtype)}


def _in_steps(steps: str, step: int) -> bool:
    """Whether a ``QUANT_SUITE_TENSOR_DUMP_STEPS`` value admits process step
    ``step``."""
    for item in steps.split(","):
        start, colon, stop = item.strip().partition(":")
        first = int(start)
        if not colon:
            inside = step == first
        else:
            inside = first <= step and (not stop or step < int(stop))
        if inside:
            return True
    return False


class _ManifestScan:
    """What a rank's per-step manifest holds beyond the reader's view: the
    ``run`` records, the bytes the rank directory held at the end of each run,
    each attention record's per-layer replay fields and each weight's step."""

    def __init__(self, path: Path) -> None:
        self.runs: list[dict[str, Any]] = []
        self.end_bytes: list[int] = []
        self.replays: dict[tuple[str, int, int], dict[str, Any]] = {}
        self.weight_steps: dict[str, int] = {}
        interned: dict[str, dict[str, Any]] = {}
        total = 0
        with open(path, "rb") as file:
            for line in file:
                if not line.strip():
                    total += len(line)
                    continue
                record = json.loads(line)
                kind = record.get("record")
                if kind == "run":
                    if self.runs:
                        self.end_bytes.append(total)
                    self.runs.append(
                        {key: value for key, value in record.items() if key != "record"}
                    )
                elif kind == "attention":
                    key = (
                        str(record["layer"]),
                        int(record["step"]),
                        int(record["request"]),
                    )
                    self.replays[key] = self._layer_replay(record["replay"], interned)
                elif kind == "weight":
                    self.weight_steps[str(record["layer"])] = int(record["step"])
                total += len(line)
                for entry in record.get("tensors", {}).values():
                    total += int(entry.get("bytes") or 0)
        if self.runs:
            self.end_bytes.append(total)

    @staticmethod
    def _layer_replay(
        replay: Mapping[str, Any], interned: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        rows = [replay["q_start_pos"], replay["kv_len"]]
        if replay.get("kv_rows", rows) != rows:
            raise ValueError(
                f"an attention record has kv_rows {replay.get('kv_rows')}, not "
                f"[q_start_pos, kv_len] = {rows}"
            )
        fields = {
            key: value
            for key, value in replay.items()
            if key not in _STEP_REPLAY_FIELDS
        }
        return interned.setdefault(json.dumps(fields, sort_keys=True), fields)


def _replay_entry(
    replays: Sequence[dict[str, Any]], rows: Sequence[int]
) -> dict[str, object]:
    """``replay`` when every step has the same fields, else ``replay_ranges``."""
    ranges: list[dict[str, Any]] = []
    start = 0
    for replay, count in zip(replays, rows):
        if ranges and ranges[-1]["replay"] == replay:
            ranges[-1]["rows"][1] = start + count
        else:
            ranges.append({"rows": [start, start + count], "replay": replay})
        start += count
    if len(ranges) == 1:
        return {"replay": ranges[0]["replay"]}
    return {"replay_ranges": ranges}


class _RankMerge:
    """Merges one rank's directory."""

    def __init__(
        self,
        dump: TensorDump,
        rank: int,
        source: Path,
        destination: Path,
        delete_source: bool,
    ) -> None:
        self._rank = rank
        self._source = source
        self._destination = destination
        self._delete_source = delete_source
        self._scan = _ManifestScan(source / _SOURCE_MANIFEST)
        self._runs = dump.runs(rank=rank)
        if len(self._scan.runs) != len(self._runs):
            raise ValueError(
                f"{source / _SOURCE_MANIFEST} changed while it was read; merge "
                "a dump that no process writes to"
            )
        self._steps = {step.step: step for step in dump.steps(rank=rank)}
        self._requests = dump.requests(rank=rank)
        self._by_id = {request.id: request for request in self._requests}
        self._linear: dict[int, dict[str, LinearPair]] = {}
        for pair in dump.linear_pairs(rank=rank):
            self._linear.setdefault(pair.step, {})[pair.layer] = pair
        self._attention: dict[tuple[int, int], dict[str, Mapping[str, TensorRef]]] = {}
        for layer, step, request, tensors in dump._attention_records(rank):
            self._attention.setdefault((step, request), {})[layer] = tensors
        self._weights: dict[str, Mapping[str, TensorRef]] = dump._weight_refs(rank)
        self._step_files = self._files_by_step()
        self._step_requests: dict[int, set[int]] = {}
        for request in self._requests:
            for step in request.steps:
                self._step_requests.setdefault(step, set()).add(request.id)
        self._kept = self._steps_to_keep()
        self._deleted: set[int] = set()
        self._done: dict[int, dict[str, Any]] = {}
        # Requests whose merged files this pass wrote or read back.
        self._verified: set[int] = set()
        self._weights_done: dict[str, dict[str, Any]] = {}
        self._source_deleted = False
        self._read_destination()

    def run(self) -> None:
        self._merge_weights()
        if self._delete_source:
            self._delete_steps(sorted(self._steps.keys() | self._step_files.keys()))
        for request in self._requests:
            if request.id in self._done:
                continue
            self._done[request.id] = self._merge_request(request)
            self._verified.add(request.id)
            self._write_manifest()
            logger.info(
                "rank %d: merged request %d of %d",
                self._rank,
                request.id + 1,
                len(self._requests),
            )
            if self._delete_source:
                self._delete_steps(request.steps)
        if self._delete_source:
            self._delete_manifest()

    # Planning.

    def _files_by_step(self) -> dict[int, list[Path]]:
        files: dict[int, list[Path]] = {}
        for step, pairs in self._linear.items():
            for pair in pairs.values():
                for ref in (pair.activation, pair.output):
                    if ref is not None:
                        files.setdefault(step, []).append(ref.path)
        for (step, _), layers in self._attention.items():
            for refs in layers.values():
                files.setdefault(step, []).extend(ref.path for ref in refs.values())
        return files

    def _steps_to_keep(self) -> set[int]:
        """Steps whose files hold rows that no request's merged files hold."""
        kept: set[int] = set()
        for step in self._step_files:
            written = self._steps.get(step)
            if written is None:
                kept.add(step)
                continue
            covered = sorted(
                (piece.token_start, piece.q_len) for piece in written.requests
            )
            end = 0
            for start, count in covered:
                if start != end:
                    break
                end += count
            if self._linear.get(step) and end != written.num_actual_tokens:
                kept.add(step)
        batch = {
            (step, piece.request)
            for step, written in self._steps.items()
            for piece in written.requests
        }
        kept.update(
            step for step, request in self._attention if (step, request) not in batch
        )
        for step in sorted(kept):
            logger.warning(
                "rank %d: step %d holds rows that belong to no request; its "
                "files are kept",
                self._rank,
                step,
            )
        return kept

    def _phase_slices(self, request: Request) -> dict[str, list[RequestSlice]]:
        phases: dict[str, list[RequestSlice]] = {}
        for piece in request.slices:
            if piece.phase not in _PHASES:
                raise ValueError(
                    f"request {request.id} has phase {piece.phase!r} at step "
                    f"{piece.step}; expected one of {_PHASES}"
                )
            phases.setdefault(piece.phase, []).append(piece)
        return phases

    def _segments(self, slices: Sequence[RequestSlice]) -> list[dict[str, int]]:
        segments: list[dict[str, int]] = []
        row = 0
        for piece in slices:
            segments.append(
                {
                    "row_start": row,
                    "q_len": piece.q_len,
                    "step": piece.step,
                    "run": self._steps[piece.step].run,
                    "request": piece.request,
                    "token_start": piece.token_start,
                    "q_start_pos": piece.q_start_pos,
                    "kv_len": piece.kv_len,
                }
            )
            row += piece.q_len
        return segments

    def _layers(
        self, request: Request, slices: Sequence[RequestSlice]
    ) -> tuple[list[str], list[str]]:
        """The linear and attention layers with records at every slice.

        Raises:
            ValueError: if the layers differ between slices.
        """

        def common(kind: str, at: Callable[[RequestSlice], Iterable[str]]) -> list[str]:
            first = sorted(at(slices[0]))
            for piece in slices[1:]:
                names = sorted(at(piece))
                if names != first:
                    raise ValueError(
                        f"request {request.id} has {kind} records of layers "
                        f"{names} at step {piece.step} but {first} at step "
                        f"{slices[0].step}"
                    )
            return first

        linear = common("linear", lambda piece: self._linear.get(piece.step, {}))
        attention = common(
            "attention",
            lambda piece: self._attention.get((piece.step, piece.request), {}),
        )
        both = set(linear) & set(attention)
        if both:
            raise ValueError(f"layers {sorted(both)} have linear and attention records")
        return linear, attention

    def _may_continue(self, request: Request) -> bool:
        """Whether the dump may have stopped writing before the request's
        last step: its run did not write the next step, and a step filter,
        the step cap or the byte budget would have left it out."""
        last = request.slices[-1].step
        following = self._steps.get(last + 1)
        if following is not None and following.run == request.run:
            return False
        index = next(i for i, run in enumerate(self._runs) if run.run == request.run)
        run = self._runs[index]
        options = run.options
        if str(options.get("phase", "all")) != "all":
            return True
        if not _in_steps(str(options.get("steps", "0:")), last + 1 - run.first_step):
            return True
        run_steps = [step for step in self._steps.values() if step.run == run.run]
        max_steps = _option_int(options, "max_steps")
        if max_steps and sum(step.step <= last for step in run_steps) >= max_steps:
            return True
        max_bytes = _option_int(options, "max_bytes")
        return (
            max_bytes > 0
            and last == run_steps[-1].step
            and self._scan.end_bytes[index] >= max_bytes
        )

    # Destination manifest.

    def _read_destination(self) -> None:
        path = self._destination / _MANIFEST
        if not path.exists():
            return
        manifest = json.loads(path.read_text(encoding="utf-8"))
        version = manifest.get("merged_format_version")
        if version != _MERGED_FORMAT_VERSION:
            raise RuntimeError(
                f"{path} is in merged format {version}, not "
                f"{_MERGED_FORMAT_VERSION}; use a new destination"
            )
        recorded_runs = manifest["runs"]
        if (
            manifest.get("rank") != self._rank
            or manifest.get("source_deleted")
            or self._scan.runs[: len(recorded_runs)] != recorded_runs
        ):
            raise RuntimeError(
                f"{path} was merged from another dump; use a new destination"
            )
        for entry in manifest["requests"]:
            request = self._by_id.get(int(entry["request"]))
            if request is None or self._plan(request) != self._recorded_plan(entry):
                raise RuntimeError(
                    f"request {entry['request']} of {path} does not match the "
                    "dump; use a new destination"
                )
            self._done[request.id] = entry
        for layer, entry in manifest["weights"].items():
            refs = self._weights.get(layer)
            if refs is None or self._weight_tensors(refs) != entry["tensors"]:
                raise RuntimeError(
                    f"weight {layer!r} of {path} does not match the dump; use a "
                    "new destination"
                )
            self._weights_done[layer] = entry

    def _plan(self, request: Request) -> dict[str, Any]:
        plan: dict[str, Any] = {}
        for phase, slices in self._phase_slices(request).items():
            linear, attention = self._layers(request, slices)
            plan[phase] = (self._segments(slices), sorted(linear + attention))
        return plan

    @staticmethod
    def _recorded_plan(entry: Mapping[str, Any]) -> dict[str, Any]:
        return {
            phase: (data["segments"], sorted(data["layers"]))
            for phase, data in entry["phases"].items()
        }

    @staticmethod
    def _weight_tensors(refs: Mapping[str, TensorRef]) -> dict[str, dict[str, object]]:
        return {
            name: {"shape": list(ref.shape), "dtype": ref.dtype, "sha256": ref.sha256}
            for name, ref in refs.items()
        }

    def _write_manifest(self) -> None:
        versions = sorted({int(run["format_version"]) for run in self._scan.runs})
        manifest = {
            "merged_format_version": _MERGED_FORMAT_VERSION,
            "source_format_versions": versions,
            "rank": self._rank,
            "runs": self._scan.runs,
            "steps": [
                {
                    "step": step.step,
                    "run": step.run,
                    "phase": step.phase,
                    "num_actual_tokens": step.num_actual_tokens,
                    "requests": [
                        {
                            "request": piece.request,
                            "q_len": piece.q_len,
                            "kv_len": piece.kv_len,
                            "q_start_pos": piece.q_start_pos,
                            "token_start": piece.token_start,
                            "phase": piece.phase,
                        }
                        for piece in step.requests
                    ],
                }
                for step in self._steps.values()
            ],
            "weights": self._weights_done,
            "requests": [self._done[index] for index in sorted(self._done)],
            "source_deleted": self._source_deleted,
        }
        text = json.dumps(manifest, sort_keys=True) + "\n"
        _save(self._destination / _MANIFEST, text.encode("utf-8"))

    # Merging.

    def _merge_weights(self) -> None:
        for layer, refs in sorted(self._weights.items()):
            if layer not in self._weights_done:
                self._merge_weight(layer, refs)
            elif self._delete_source:
                self._reverify_weight(layer, refs)
            if self._delete_source:
                self._remove([ref.path for ref in refs.values()], levels=1)

    def _merge_weight(self, layer: str, refs: Mapping[str, TensorRef]) -> None:
        tensors = {name: ref.load() for name, ref in refs.items()}
        for name, tensor in tensors.items():
            recorded = refs[name].sha256
            if recorded is not None and _sha256(tensor) != recorded:
                raise RuntimeError(
                    f"{refs[name].path} does not match the SHA-256 its manifest "
                    "entry records"
                )
        relative = f"weights/{layer}.pt"
        path = self._destination / relative
        _save(path, tensors)
        _verify(path, {name: [tensor] for name, tensor in tensors.items()})
        self._weights_done[layer] = {
            "file": relative,
            "bytes": path.stat().st_size,
            "step": self._scan.weight_steps.get(layer),
            "tensors": self._weight_tensors(refs),
        }
        self._write_manifest()

    def _reverify_weight(self, layer: str, refs: Mapping[str, TensorRef]) -> None:
        """Compare an earlier pass's merged weight with the dump's files
        still there, bit for bit."""
        parts = {name: [_load_ref(ref, missing_ok=True)] for name, ref in refs.items()}
        if _has_data(parts):
            _verify(self._destination / str(self._weights_done[layer]["file"]), parts)

    def _reverify_request(self, request: Request) -> None:
        """Compare the merged files of a request an earlier pass merged with
        the dump's files still there, bit for bit."""
        phases = self._done[request.id]["phases"]
        for phase, slices in self._phase_slices(request).items():
            linear, attention = self._layers(request, slices)
            layers = phases[phase]["layers"]
            for layer in linear + attention:
                if layer in attention:
                    parts = self._attention_parts(layer, slices, missing_ok=True)
                else:
                    parts = self._linear_parts(layer, slices, missing_ok=True)
                if _has_data(parts):
                    _verify(self._destination / str(layers[layer]["file"]), parts)
        self._verified.add(request.id)

    def _merge_request(self, request: Request) -> dict[str, Any]:
        phases = self._phase_slices(request)
        may_continue = self._may_continue(request)
        entry: dict[str, Any] = {
            "request": request.id,
            "run": request.run,
            "steps": list(request.steps),
            "prompt_len": request.prompt_len,
            "generated_len": request.generated_len,
            "starts_at_zero": request.complete,
            "may_continue": may_continue,
            "complete": request.complete and not may_continue,
            "prefill_before_decode": _prefill_before_decode(request.slices),
            "phases": {},
        }
        for phase, slices in phases.items():
            entry["phases"][phase] = self._merge_phase(request, phase, slices)
        return entry

    def _merge_phase(
        self, request: Request, phase: str, slices: Sequence[RequestSlice]
    ) -> dict[str, Any]:
        linear, attention = self._layers(request, slices)
        directory = f"req_{request.id:04d}/{phase}"
        layers: dict[str, dict[str, object]] = {}
        for layer in linear:
            layers[layer] = self._merge_linear(directory, layer, slices)
        for layer in attention:
            layers[layer] = self._merge_attention(directory, layer, slices)
        return {
            "rows": sum(piece.q_len for piece in slices),
            "segments": self._segments(slices),
            "layers": layers,
        }

    def _merge_linear(
        self, directory: str, layer: str, slices: Sequence[RequestSlice]
    ) -> dict[str, object]:
        parts = self._linear_parts(layer, slices, missing_ok=False)
        return self._write(directory, layer, "linear", parts, {})

    def _linear_parts(
        self, layer: str, slices: Sequence[RequestSlice], *, missing_ok: bool
    ) -> dict[str, list[torch.Tensor]]:
        """The request's rows of each of ``layer``'s tensors, per slice."""
        parts: dict[str, list[torch.Tensor]] = {}
        for piece in slices:
            pair = self._linear[piece.step][layer]
            refs = {"input": pair.activation, "output": pair.output}
            tensors = {name: ref for name, ref in refs.items() if ref is not None}
            _check_names(layer, piece.step, tensors, parts)
            rows = slice(piece.token_start, piece.token_start + piece.q_len)
            for name, ref in tensors.items():
                tensor = _load_ref(ref, missing_ok=missing_ok)
                if tensor.shape[0] < rows.stop:
                    raise ValueError(
                        f"{ref.path} has {tensor.shape[0]} rows; request "
                        f"{piece.request} of step {piece.step} needs rows "
                        f"[{rows.start}, {rows.stop})"
                    )
                parts.setdefault(name, []).append(tensor[rows])
        return parts

    def _merge_attention(
        self, directory: str, layer: str, slices: Sequence[RequestSlice]
    ) -> dict[str, object]:
        parts = self._attention_parts(layer, slices, missing_ok=False)
        replays = [
            self._scan.replays[(layer, piece.step, piece.request)] for piece in slices
        ]
        extra = _replay_entry(replays, [piece.q_len for piece in slices])
        return self._write(directory, layer, "attention", parts, extra)

    def _attention_parts(
        self, layer: str, slices: Sequence[RequestSlice], *, missing_ok: bool
    ) -> dict[str, list[torch.Tensor]]:
        """``layer``'s tensors of the request, per slice."""
        parts: dict[str, list[torch.Tensor]] = {}
        for piece in slices:
            refs = self._attention[(piece.step, piece.request)][layer]
            _check_names(layer, piece.step, refs, parts)
            for name, ref in sorted(refs.items()):
                tensor = _load_ref(ref, missing_ok=missing_ok)
                if tensor.shape[0] != piece.q_len:
                    raise ValueError(
                        f"{ref.path} has {tensor.shape[0]} rows, not the "
                        f"{piece.q_len} of its request"
                    )
                parts.setdefault(name, []).append(tensor)
        return parts

    def _write(
        self,
        directory: str,
        layer: str,
        kind: str,
        parts: Mapping[str, list[torch.Tensor]],
        extra: Mapping[str, object],
    ) -> dict[str, object]:
        """Save ``parts`` concatenated along dim 0, read the file back and
        compare it with ``parts``."""
        for name, pieces in parts.items():
            first = pieces[0]
            for piece in pieces[1:]:
                if piece.dtype != first.dtype or piece.shape[1:] != first.shape[1:]:
                    raise ValueError(
                        f"{name} of layer {layer!r} is {_dtype_name(first.dtype)} "
                        f"{list(first.shape)} at one step and "
                        f"{_dtype_name(piece.dtype)} {list(piece.shape)} at another"
                    )
        relative = f"{directory}/{layer}.pt"
        path = self._destination / relative
        merged = {
            name: torch.cat(pieces).contiguous() for name, pieces in parts.items()
        }
        tensors = {name: _tensor_entry(tensor) for name, tensor in merged.items()}
        _save(path, merged)
        del merged
        _verify(path, parts)
        entry: dict[str, object] = {
            "kind": kind,
            "file": relative,
            "bytes": path.stat().st_size,
            "tensors": tensors,
        }
        entry.update(extra)
        return entry

    # Deleting.

    def _delete_steps(self, steps: Iterable[int]) -> None:
        """Delete the files of each step whose every request is merged, once
        each of those requests is verified in this pass.

        Raises:
            RuntimeError: if a merged file differs from the dump.
        """
        for step in steps:
            if step in self._deleted or step in self._kept:
                continue
            requests = self._step_requests.get(step, set())
            if not requests <= self._done.keys():
                continue
            for request in sorted(requests - self._verified):
                self._reverify_request(self._by_id[request])
            self._remove(self._step_files.get(step, []), levels=2)
            self._deleted.add(step)

    def _remove(self, files: Sequence[Path], *, levels: int) -> None:
        """Delete ``files``, then each directory up to ``levels`` above them
        that is left empty."""
        root = self._source.resolve()
        for path in files:
            if not path.resolve().is_relative_to(root):
                raise RuntimeError(f"{path} is outside {self._source}")
            path.unlink(missing_ok=True)
        directories = {path.parent for path in files}
        for _ in range(levels):
            for directory in sorted(directories, key=lambda d: -len(d.parts)):
                _remove_empty(directory, warn=True)
            directories = {directory.parent for directory in directories}

    def _delete_manifest(self) -> None:
        if len(self._done) != len(self._requests):
            return
        if self._weights.keys() - self._weights_done.keys():
            return
        if (self._steps.keys() | self._step_files.keys()) - self._deleted:
            return
        (self._source / _SOURCE_MANIFEST).unlink()
        self._source_deleted = True
        self._write_manifest()
        _remove_empty(self._source / "weights", warn=False)
        _remove_empty(self._source, warn=True)
        logger.info("rank %d: deleted the merged source %s", self._rank, self._source)


def _option_int(options: Mapping[str, object], key: str) -> int:
    """A non-negative integer option of a run record; 0 when it is absent.

    Raises:
        ValueError: if the option is not a non-negative integer.
    """
    value = options.get(key)
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"a run record has {key}={value!r}; expected a non-negative integer"
        )
    return value


def _prefill_before_decode(slices: Sequence[RequestSlice]) -> bool:
    decoded = False
    for piece in slices:
        if piece.phase == "decode":
            decoded = True
        elif decoded:
            return False
    return True


def _check_names(
    layer: str,
    step: int,
    written: Mapping[str, object],
    parts: Mapping[str, list[torch.Tensor]],
) -> None:
    if parts and sorted(written) != sorted(parts):
        raise ValueError(
            f"layer {layer!r} wrote {sorted(written)} at step {step} but "
            f"{sorted(parts)} at an earlier step of the request"
        )


def _verify(path: Path, parts: Mapping[str, Sequence[torch.Tensor]]) -> None:
    """Check that ``path`` holds the concatenation of ``parts``, bit for bit.
    A meta tensor stands for rows whose dump file is gone; only their number
    is checked.

    Raises:
        RuntimeError: if it does not.
    """
    loaded = _load_file(path)
    if not isinstance(loaded, dict) or sorted(loaded) != sorted(parts):
        raise RuntimeError(f"{path} does not hold the tensors {sorted(parts)}")
    for name, pieces in parts.items():
        tensor = loaded[name]
        rows = sum(piece.shape[0] for piece in pieces)
        if not isinstance(tensor, torch.Tensor) or tensor.shape[:1] != (rows,):
            raise RuntimeError(f"{path} does not hold {name} with {rows} rows")
        start = 0
        for piece in pieces:
            stop = start + piece.shape[0]
            if not piece.is_meta and not _same_bits(tensor[start:stop], piece):
                raise RuntimeError(
                    f"rows [{start}, {stop}) of {name} in {path} differ from the dump's"
                )
            start = stop


def _remove_empty(directory: Path, *, warn: bool) -> None:
    try:
        directory.rmdir()
    except FileNotFoundError:
        pass
    except OSError:
        if warn:
            logger.warning("%s holds files no manifest lists; it is kept", directory)


def merge(src: str | Path, dst: str | Path, *, delete_source: bool = False) -> None:
    """Merge the dump in ``src`` into one file per request, phase and layer
    in ``dst``; see the module docstring for the layout.

    ``merge`` leaves torch's intra-op thread count as the caller set it; only
    ``python -m tensor_dump.merge`` caps it, as merging is I/O-bound.

    Args:
        src: A dump directory, holding ``rank_<R>/manifest.jsonl``.
        dst: The destination; a merge that stopped part way resumes in it.
        delete_source: Delete each dump file once every file merged from it
            is verified, and each rank's dump manifest once nothing else is
            left. Files merged by an earlier run are read back and compared
            with the dump's again first: only bytes verified against the
            merged copy are ever deleted.

    Raises:
        ValueError: if ``src`` is not a dump, ``dst`` is ``src``, or the
            dump's requests or records cannot be split into requests and
            phases, or a run record holds an invalid option.
        RuntimeError: if a merged file differs from the dump, or ``dst``
            holds a merge of another dump.
    """
    source = Path(src)
    destination = Path(dst)
    if source.resolve() == destination.resolve():
        raise ValueError(f"the destination {destination} is the dump directory")
    if not _rank_dirs(source, _SOURCE_MANIFEST):
        if _rank_dirs(destination, _MANIFEST):
            logger.info("%s holds no dump left to merge into %s", source, destination)
            return
        raise ValueError(f"{source} holds no rank_<R>/{_SOURCE_MANIFEST}")
    dump = TensorDump(source)
    for rank in dump.ranks:
        _RankMerge(
            dump,
            rank,
            source / f"rank_{rank}",
            destination / f"rank_{rank}",
            delete_source,
        ).run()


def _main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tensor_dump.merge",
        description="Merge a dump into one file per request, phase and layer.",
    )
    parser.add_argument("src", help="the dump directory")
    parser.add_argument("dst", help="the destination directory")
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="delete each dump file once every file merged from it is verified",
    )
    parser.add_argument(
        "--threads",
        type=int,
        metavar="N",
        help="torch intra-op threads (default: the smaller of 4 and the CPU count)",
    )
    args = parser.parse_args(argv)
    threads: int | None = args.threads
    if threads is not None and threads < 1:
        parser.error("--threads must be at least 1")
    torch.set_num_threads(threads or min(4, os.cpu_count() or 1))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    merge(args.src, args.dst, delete_source=args.delete_source)


__all__ = ["merge"]


if __name__ == "__main__":
    _main()
