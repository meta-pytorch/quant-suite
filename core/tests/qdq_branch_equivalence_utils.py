# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared harness for the qdq() trace-branch equivalence tests.

Every shape or dtype check in qdq() takes a trace-safe form when `_is_fx_proxy`
says its argument is a torch.fx Proxy, and the original code otherwise. Tracing
only ever runs the trace-safe form on Proxies, so nothing else checks that the
two forms compute the same thing. `force_trace_branch()` forces the trace-safe
form on real tensors by patching `_is_fx_proxy` to always return True in every
module that imported it; `BranchEquivalenceTestCase` then requires bitwise-equal
outputs (and an unmodified input) from the same call with and without the patch.

Set QUANT_SUITE_BRANCH_FUZZ_SCALE=<k> to draw k times as many random shapes.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import itertools
import os
import random
import sys
import time
import unittest
from types import ModuleType
from typing import Callable, Iterator, NamedTuple
from unittest import mock

import quant_suite_core.quant_primitives as quant_primitives
import torch
from quant_suite_core import fake_quantize, QuantConfig, QuantInfo
from quant_suite_core.fake_presets import FAKE_PRESETS
from quant_suite_core.qdq_pytorch import qdq
from quant_suite_core.qdq_triton import qdq_triton_mega

_SEED = 20261006
FUZZ_SCALE: int = int(os.environ.get("QUANT_SUITE_BRANCH_FUZZ_SCALE", "1"))
_DTYPES: tuple[torch.dtype, ...] = (torch.float32, torch.bfloat16)
_MAX_DIM = 300
# Bounds the padded 2-D size (both orientations, MBS256 padding) so CPU cases
# stay fast.
_MAX_PADDED_NUMEL = 2**17
# Last-dim edge cases: multiples of 16/32/128, their +-1 neighbors, and the
# MBS128/256 boundaries.
_EDGE_N = (1, 15, 16, 17, 31, 32, 33, 96, 100, 127, 128, 129, 255, 256, 257)
_EDGE_M = (1, 15, 16, 17, 33, 128, 129)
_RANDOM_SHAPES_PER_RANK = {1: 4, 2: 8, 3: 6, 4: 6}
# Aligned (no eager padding), unaligned rows and/or cols, and each rank.
MEGA_SHAPES = [(), (33,), (128, 256), (17, 100), (129, 128), (2, 3, 127)]
_INPUT_KINDS = ("finite", "special")
# (transpose, return_quant_info, entry point), rotated across shapes so every
# (preset, dtype, device) sees each combination several times.
_FLAG_COMBOS = list(
    itertools.product((False, True), (False, True), ("qdq", "fake_quantize"))
)
ZERO_WIDTH_SHAPES = ((0,), (5, 0), (2, 3, 0))

ZeroWidthCall = Callable[[torch.Tensor, QuantConfig], object]
EAGER_CALLS: dict[str, ZeroWidthCall] = {
    "qdq": lambda x, cfg: qdq(x, "activation", cfg),
    "fake_quantize": lambda x, cfg: fake_quantize(
        x, "activation", cfg, use_triton=False
    ),
}


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def _padded_cost(shape: tuple[int, ...]) -> int:
    rows = 1
    for d in shape[:-1]:
        rows *= d
    cols = shape[-1] if shape else 1
    return max(rows * _round_up(cols, 256), cols * _round_up(rows, 256))


def _random_shape(rng: random.Random, rank: int) -> tuple[int, ...]:
    shape = [
        rng.choice((rng.randint(1, 8), rng.randint(1, _MAX_DIM))) for _ in range(rank)
    ]
    while _padded_cost(tuple(shape)) > _MAX_PADDED_NUMEL:
        i = max(range(rank - 1), key=lambda j: shape[j]) if rank > 1 else 0
        shape[i] = rng.randint(1, max(1, shape[i] // 2))
    return tuple(shape)


def _shapes() -> list[tuple[int, ...]]:
    shapes: list[tuple[int, ...]] = [()]
    shapes += [(n,) for n in _EDGE_N]
    for i, n in enumerate(_EDGE_N):
        shapes += [(1, n), (_EDGE_M[i % len(_EDGE_M)], n)]
    shapes += [(129, 128), (128, 257), (3, 5, 96), (2, 3, 127), (1, 1, 129)]
    shapes += [(4, 17, 33), (2, 3, 4, 64), (1, 2, 16, 129), (2, 1, 1, 1)]
    rng = random.Random(_SEED)
    for rank, count in _RANDOM_SHAPES_PER_RANK.items():
        shapes += [_random_shape(rng, rank) for _ in range(count * FUZZ_SCALE)]
    return shapes


def _make_input(
    shape: tuple[int, ...],
    kind: str,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g) * torch.exp2(
        torch.randint(-6, 7, shape, generator=g).float()
    )
    if kind == "special":
        flat = x.view(-1)
        n = flat.numel()
        start = int(torch.randint(0, n, (), generator=g))
        flat[start : start + int(torch.randint(16, 300, (), generator=g))] = 0.0
        idx = torch.randint(0, n, (max(1, n // 40),), generator=g)
        specials = torch.tensor([float("inf"), float("-inf"), float("nan"), -0.0])
        flat[idx] = specials[torch.randint(0, 4, idx.shape, generator=g)]
    return x.to(dtype).to(device)


def _bytes(t: torch.Tensor) -> torch.Tensor:
    """Raw bytes, so NaN payloads and -0.0 compare exactly."""
    # clone, not contiguous(): a one-element view can keep a stride != 1 and
    # still count as contiguous, which view(uint8) rejects.
    return t.reshape(-1).clone(memory_format=torch.contiguous_format).view(torch.uint8)


def _first_diff(got: torch.Tensor, want: torch.Tensor) -> str:
    width = got.element_size()
    differs = (_bytes(got) != _bytes(want)).view(-1, width).any(dim=1)
    flat_idx = differs.nonzero().flatten()[:5].tolist()
    shape = got.shape if got.ndim else torch.Size([1])
    entries = [
        f"{tuple(int(c) for c in torch.unravel_index(torch.tensor(i), shape))}: "
        f"forced={got.reshape(-1)[i].item()!r} eager={want.reshape(-1)[i].item()!r}"
        for i in flat_idx
    ]
    return f"{int(differs.sum())} differing elements; first: " + "; ".join(entries)


def _diff_tensors(
    name: str, got: torch.Tensor | None, want: torch.Tensor | None
) -> list[str]:
    if got is None or want is None:
        return [] if got is None and want is None else [f"{name}: None mismatch"]
    if got.shape != want.shape or got.dtype != want.dtype:
        return [f"{name}: {got.shape}/{got.dtype} vs eager {want.shape}/{want.dtype}"]
    if torch.equal(_bytes(got), _bytes(want)):
        return []
    return [f"{name}: {_first_diff(got, want)}"]


def _diff_results(got: object, want: object) -> list[str]:
    if isinstance(want, QuantInfo) and isinstance(got, QuantInfo):
        return [
            d
            for f in dataclasses.fields(QuantInfo)
            for d in _diff_tensors(f.name, getattr(got, f.name), getattr(want, f.name))
        ]
    if isinstance(want, torch.Tensor) and isinstance(got, torch.Tensor):
        return _diff_tensors("output", got, want)
    return [f"result type {type(got).__name__} vs eager {type(want).__name__}"]


@functools.cache
def importers() -> tuple[ModuleType, ...]:
    """Every loaded module holding the original `_is_fx_proxy` by name."""
    original = quant_primitives._is_fx_proxy
    return tuple(
        m
        for name, m in list(sys.modules.items())
        if name.startswith("quant_suite_core")
        and getattr(m, "_is_fx_proxy", None) is original
    )


@contextlib.contextmanager
def force_trace_branch() -> Iterator[None]:
    with contextlib.ExitStack() as stack:
        for module in importers():
            stack.enter_context(
                mock.patch.object(module, "_is_fx_proxy", lambda x: True)
            )
        yield


class Case(NamedTuple):
    preset: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    transpose: bool
    return_quant_info: bool
    entry: str
    kind: str


def cases(
    entries: tuple[str, ...], shapes: list[tuple[int, ...]] | None = None
) -> Iterator[Case]:
    combos = [c for c in _FLAG_COMBOS if c[2] in entries]
    shapes = _shapes() if shapes is None else shapes
    for p, preset in enumerate(sorted(FAKE_PRESETS)):
        for d, dtype in enumerate(_DTYPES):
            for s, shape in enumerate(shapes):
                transpose, quant_info, entry = combos[(s + p + 3 * d) % len(combos)]
                for kind in _INPUT_KINDS:
                    yield Case(preset, dtype, shape, transpose, quant_info, entry, kind)


def _call(case: Case, cfg: QuantConfig, x: torch.Tensor) -> object:
    if case.entry == "qdq":
        return qdq(x, "activation", cfg, case.transpose, case.return_quant_info)
    if case.entry == "fake_quantize":
        return fake_quantize(
            x,
            "activation",
            cfg,
            transpose=case.transpose,
            use_triton=False,
            return_quant_info=case.return_quant_info,
        )
    return qdq_triton_mega(x, "activation", cfg, case.transpose, case.return_quant_info)


def _run_side(
    case: Case, cfg: QuantConfig, x: torch.Tensor, seed: int, forced: bool
) -> tuple[object, list[str]]:
    """One call on a private clone of `x`; returns (result or exception, problems)."""
    x = x.clone()
    x_before = x.clone()
    torch.manual_seed(seed)
    try:
        with force_trace_branch() if forced else contextlib.nullcontext():
            result: object = _call(case, cfg, x)
    except Exception as e:
        result = e
    side = "forced" if forced else "eager"
    mutated = not torch.equal(_bytes(x), _bytes(x_before))
    return result, [f"{side} call mutated its input"] if mutated else []


class BranchEquivalenceTestCase(unittest.TestCase):
    def sweep(self, cases: Iterator[Case], device: torch.device, label: str) -> None:
        configs = {
            name: QuantConfig.from_dict(spec) for name, spec in FAKE_PRESETS.items()
        }
        n_cases = n_compared = 0
        raised: dict[str, int] = {}
        failures: list[str] = []
        start = time.monotonic()
        for i, case in enumerate(cases):
            n_cases += 1
            cfg = configs[case.preset]
            x = _make_input(case.shape, case.kind, case.dtype, device, seed=i)
            want, problems = _run_side(case, cfg, x, seed=i, forced=False)
            got, forced_problems = _run_side(case, cfg, x, seed=i, forced=True)
            problems += forced_problems
            if isinstance(want, Exception) or isinstance(got, Exception):
                if type(got) is not type(want):
                    problems.append(f"raised {got!r} vs eager {want!r}")
                key = f"{case.preset}: {type(want).__name__}: {str(want)[:100]}"
                raised[key] = raised.get(key, 0) + 1
            else:
                n_compared += 1
                problems += _diff_results(got, want)
            failures += [f"{case} on {device}: {p}" for p in problems]
        print(
            f"[branch-equivalence] {label} on {device}: {n_cases} cases, "
            f"{n_compared} compared, {sum(raised.values())} raised on both branches, "
            f"{len(failures)} failures, {time.monotonic() - start:.1f}s "
            f"(fuzz scale {FUZZ_SCALE})",
            file=sys.stderr,
        )
        for key, count in sorted(raised.items()):
            print(f"[branch-equivalence]   raised x{count}: {key}", file=sys.stderr)
        self.assertGreater(n_compared, 0)
        self.assertEqual(failures, [], "\n".join(failures[:20]))

    def check_zero_width_raises(
        self, calls: dict[str, ZeroWidthCall], device: torch.device
    ) -> None:
        for preset, spec in sorted(FAKE_PRESETS.items()):
            cfg = QuantConfig.from_dict(spec)
            for (name, call), shape in itertools.product(
                calls.items(), ZERO_WIDTH_SHAPES
            ):
                x = torch.empty(shape, device=device)
                with self.subTest(
                    preset=preset, entry=name, device=str(device), shape=shape
                ):
                    with self.assertRaisesRegex(ValueError, "zero-size last dim"):
                        call(x, cfg)
                    with force_trace_branch(), self.assertRaises(Exception):
                        call(x, cfg)
