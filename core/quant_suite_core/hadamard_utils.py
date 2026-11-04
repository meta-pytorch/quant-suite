# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Orthonormal Hadamard transforms built from the Sylvester recursion.

The Sylvester construction is defined only for power-of-two orders::

    H_1 = [[1]]
    H_2n = [[H_n,  H_n],
            [H_n, -H_n]]

so every entry of ``H_n`` is ``+-1``, ``H_n`` is symmetric, and
``H_n @ H_n.T = n * I``. Dividing by ``sqrt(n)`` makes it orthonormal.

Sizes that are not a power of two are rejected with ``ValueError``; there is no
fallback to a tabulated Hadamard matrix of some other order.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def is_pow2(n: int) -> bool:
    """Return True iff ``n`` is a positive integer power of two."""
    return n > 0 and (n & (n - 1)) == 0


def _check_pow2(n: int) -> None:
    if not is_pow2(n):
        raise ValueError(
            f"Hadamard order must be a positive power of two, got {n}. "
            "Only the Sylvester construction is supported."
        )


def _normalizer(n: int) -> float:
    """``sqrt(n)`` in double precision -- the divisor that orthonormalizes ``H_n``.

    Double rather than float32: the matrices this module returns are float64, and
    a float32 divisor would leave them orthogonal only to float32 precision at the
    orders where ``sqrt(n)`` is irrational (32, 128, 2048, ...).
    """
    return math.sqrt(n)


def sylvester_matrix(
    n: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Unnormalized Sylvester Hadamard matrix of order ``n``, entries ``+-1``.

    Built by repeated quadrant doubling, so it costs ``O(n^2)`` writes rather
    than the ``O(n^2 log n)`` of transforming an identity matrix.
    """
    _check_pow2(n)
    h = torch.empty((n, n), device=device, dtype=dtype)
    h[0, 0] = 1
    order = 1
    while order < n:
        block = h[:order, :order]
        h[:order, order : 2 * order] = block
        h[order : 2 * order, :order] = block
        h[order : 2 * order, order : 2 * order] = -block
        order *= 2
    return h


def _walsh_hadamard_transform(x: Tensor) -> Tensor:
    """Unnormalized Hadamard transform of the last axis of ``x``.

    Applies the Sylvester recursion in reverse: ``H_2n`` acting on a vector is
    ``H_n`` acting independently on each half followed by a sum/difference of
    the two halves, so ``log2(n)`` sweeps of butterflies over index bit ``k``
    realize the full transform. The bit sweeps commute, so the stage order does
    not change the result.
    """
    n = x.shape[-1]
    src = x.reshape(-1, n).clone()
    dst = torch.empty_like(src)
    rows = src.shape[0]
    stride = 1
    while stride < n:
        groups = n // (2 * stride)
        pairs = src.view(rows, groups, 2, stride)
        out = dst.view(rows, groups, 2, stride)
        lo = pairs[:, :, 0, :]
        hi = pairs[:, :, 1, :]
        torch.add(lo, hi, out=out[:, :, 0, :])
        torch.sub(lo, hi, out=out[:, :, 1, :])
        src, dst = dst, src
        stride *= 2
    return src.view(x.shape)


def fast_hadamard_transform(X: Tensor) -> Tensor:
    """Apply the orthonormal Hadamard transform to the last axis of ``X``.

    Equivalent to ``X @ H_n / _normalizer(n)`` with ``n = X.shape[-1]``,
    computed in ``O(n log n)`` per row. ``X.shape[-1]`` must be a power of two.
    """
    n = X.shape[-1]
    _check_pow2(n)
    return _walsh_hadamard_transform(X) / _normalizer(n)


def random_hadamard_matrix(
    size: int,
    device: torch.device | str,
    seed: int | None = None,
) -> Tensor:
    """Randomized Hadamard matrix ``diag(s) @ H_size / _normalizer(size)``.

    ``s`` is a uniform ``+-1`` sign vector. The returned matrix is ``float64``
    and orthogonal to double precision. ``size`` must be a power of two.

    Seeded draws use a private generator on the requested device. With
    ``seed=None`` the signs come from the global RNG's current state.
    """
    _check_pow2(size)
    generator = (
        torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    )
    bits = torch.randint(
        low=0, high=2, size=(size,), device=device, generator=generator
    ).to(torch.float64)
    signs = bits * 2 - 1
    hadamard = sylvester_matrix(size, device=device, dtype=torch.float64)
    # Broadcasting, not `torch.diag(signs) @ hadamard`, which would be O(n^3).
    return signs.unsqueeze(1) * hadamard / _normalizer(size)
