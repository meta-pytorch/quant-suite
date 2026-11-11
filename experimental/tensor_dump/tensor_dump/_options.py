# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dump options read from the environment."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_DIR_ENV = "QUANT_SUITE_TENSOR_DUMP_DIR"
_STEPS_ENV = "QUANT_SUITE_TENSOR_DUMP_STEPS"
_PHASE_ENV = "QUANT_SUITE_TENSOR_DUMP_PHASE"
_MAX_STEPS_ENV = "QUANT_SUITE_TENSOR_DUMP_MAX_STEPS"
_RANKS_ENV = "QUANT_SUITE_TENSOR_DUMP_RANKS"

_PHASES: frozenset[str] = frozenset({"prefill", "decode", "mixed"})
_INT_RE: re.Pattern[str] = re.compile(r"[0-9]+")
_RANGE_RE: re.Pattern[str] = re.compile(r"([0-9]+):([0-9]*)")


@dataclass(frozen=True)
class _DumpOptions:
    """Parsed dump options.

    Attributes:
        directory: Output root.
        steps: Half-open step-id ranges ``(start, stop)``; ``stop`` is ``None``
            for an open range.
        phases: Step phases that are written.
        max_steps: Cap on the number of steps written; 0 means no cap.
        ranks: Global ranks that write; ``None`` means every rank.
    """

    directory: Path
    steps: tuple[tuple[int, int | None], ...]
    phases: frozenset[str]
    max_steps: int
    ranks: frozenset[int] | None

    def selects_step(self, step: int) -> bool:
        return any(
            start <= step and (stop is None or step < stop)
            for start, stop in self.steps
        )

    def selects_rank(self, rank: int) -> bool:
        return self.ranks is None or rank in self.ranks

    def describe(self) -> dict[str, object]:
        """The options in the environment-variable grammar, for the manifest."""
        steps = ",".join(
            str(start)
            if stop == start + 1
            else f"{start}:{'' if stop is None else stop}"
            for start, stop in self.steps
        )
        phase = "all" if self.phases == _PHASES else ",".join(sorted(self.phases))
        ranks = "all" if self.ranks is None else ",".join(map(str, sorted(self.ranks)))
        return {
            "steps": steps,
            "phase": phase,
            "max_steps": self.max_steps,
            "ranks": ranks,
        }


def _items(name: str, value: str) -> list[str]:
    items = [item.strip() for item in value.split(",")]
    if not all(items):
        raise ValueError(f"{name}={value!r} has an empty item")
    return items


def _parse_steps(value: str) -> tuple[tuple[int, int | None], ...]:
    ranges: list[tuple[int, int | None]] = []
    for item in _items(_STEPS_ENV, value):
        if _INT_RE.fullmatch(item):
            ranges.append((int(item), int(item) + 1))
            continue
        match = _RANGE_RE.fullmatch(item)
        if match is None:
            raise ValueError(
                f"{_STEPS_ENV}={value!r}: {item!r} is not N, A:B or A: with "
                "non-negative integers"
            )
        start = int(match.group(1))
        stop = int(match.group(2)) if match.group(2) else None
        if stop is not None and stop <= start:
            raise ValueError(f"{_STEPS_ENV}={value!r}: range {item!r} is empty")
        ranges.append((start, stop))
    return tuple(ranges)


def _parse_phases(value: str) -> frozenset[str]:
    phases: set[str] = set()
    for item in _items(_PHASE_ENV, value):
        if item == "all":
            phases |= _PHASES
        elif item in _PHASES:
            phases.add(item)
        else:
            raise ValueError(
                f"{_PHASE_ENV}={value!r}: {item!r} is not one of "
                f"{sorted(_PHASES | {'all'})}"
            )
    return frozenset(phases)


def _parse_max_steps(value: str) -> int:
    if not _INT_RE.fullmatch(value.strip()):
        raise ValueError(f"{_MAX_STEPS_ENV}={value!r} is not a non-negative integer")
    return int(value)


def _parse_ranks(value: str) -> frozenset[int] | None:
    if value.strip() == "all":
        return None
    ranks: set[int] = set()
    for item in _items(_RANKS_ENV, value):
        if not _INT_RE.fullmatch(item):
            raise ValueError(
                f"{_RANKS_ENV}={value!r}: {item!r} is not a non-negative integer"
            )
        ranks.add(int(item))
    return frozenset(ranks)


def _options_from_environ(environ: Mapping[str, str]) -> _DumpOptions | None:
    """The dump options, or ``None`` when ``QUANT_SUITE_TENSOR_DUMP_DIR`` is
    unset or empty, in which case the other variables are ignored.

    Raises:
        ValueError: naming the variable, for a malformed value.
    """
    directory = environ.get(_DIR_ENV, "")
    if not directory:
        return None
    return _DumpOptions(
        directory=Path(directory),
        steps=_parse_steps(environ.get(_STEPS_ENV, "0:")),
        phases=_parse_phases(environ.get(_PHASE_ENV, "all")),
        max_steps=_parse_max_steps(environ.get(_MAX_STEPS_ENV, "1")),
        ranks=_parse_ranks(environ.get(_RANKS_ENV, "all")),
    )
