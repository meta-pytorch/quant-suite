# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Random Hadamard Transform (RHT) generator.

``RHTGenerator`` produces the random Hadamard matrix ``R`` used in quantization
transforms. It is an ``nn.Module`` whose ``R`` rides ``model.to(...)`` and is read
as a buffer inside a dynamo-traceable ``forward``:

    generator = RHTGenerator(block_size=16, seed=42)  # cached singleton
    R = generator.get_matrix(device=tensor.device, dtype=tensor.dtype)
"""

from enum import Enum
from typing import ClassVar, Dict, Optional, Tuple

import torch
import torch.nn as nn
from quant_suite_core.hadamard_utils import random_hadamard_matrix


def _default_device() -> torch.device:
    """The current CUDA device, else CPU."""
    return (
        torch.device(torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


class HadamardMode(Enum):
    """Where a Hadamard rotation ``R`` is applied around quantization ``Q``.

    The left/right and inverse variants follow HALO (Ashkboos et al., 2025,
    "HALO: Hadamard-Assisted Lower-Precision Optimization for LLMs").
    """

    NONE = "none"  # No Hadamard transform
    LEFT = "left"  # Left Hadamard: Q((X.T @ R).T)
    RIGHT = "right"  # Right Hadamard: Q(X @ R)
    LEFT_INV = "left_inv"  # Left Hadamard + inverse after quant: R @ Q(R.T @ X)
    RIGHT_INV = "right_inv"  # Right Hadamard + inverse after quant: Q(X @ R) @ R^T

    @property
    def is_left(self) -> bool:
        """Transform row blocks (LEFT/LEFT_INV) rather than column blocks."""
        return self in (HadamardMode.LEFT, HadamardMode.LEFT_INV)

    @property
    def is_inverse(self) -> bool:
        """Apply an inverse (R.T) step after quantization (LEFT_INV/RIGHT_INV)."""
        return self in (HadamardMode.LEFT_INV, HadamardMode.RIGHT_INV)


class RHTGenerator(nn.Module):
    """RHT generator: one ``R = diag(signs) @ H / sqrt(n)`` per ``(block_size, seed)``.

    Each ``(block_size, seed)`` is a process-wide singleton holding ``R`` as a
    ``persistent=False``, ``requires_grad=False`` buffer, so memory is
    O(distinct configs) and the buffer rides ``model.to(...)`` and is read directly
    in a traceable ``forward``. ``seed`` sets reproducibility (``None`` = a fresh
    draw).
    """

    # Registry of cached singletons -- the single caching mechanism.
    _instances: ClassVar[Dict[Tuple[int, Optional[int]], "RHTGenerator"]] = {}
    _initialized: bool
    block_size: int
    matrix: torch.Tensor  # registered buffer

    def __new__(cls, block_size: int, seed: Optional[int] = None) -> "RHTGenerator":
        existing = cls._instances.get((block_size, seed))
        if existing is not None:
            return existing
        instance = super().__new__(cls)
        instance._initialized = False
        return instance

    def __init__(self, block_size: int, seed: Optional[int] = None) -> None:
        # __init__ runs on every call, but a cached singleton builds its buffer once.
        if self._initialized:
            return
        super().__init__()
        self.block_size = block_size
        self.seed = seed
        matrix = random_hadamard_matrix(
            block_size, device=_default_device(), seed=seed
        ).detach()
        matrix.requires_grad_(False)
        self.register_buffer("matrix", matrix, persistent=False)
        self._initialized = True
        type(self)._instances[(block_size, seed)] = self

    def get_matrix(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Return the shared ``R`` on ``device``/``dtype`` (unchanged if either is None)."""
        return self.matrix.to(device=device, dtype=dtype)

    @classmethod
    def clear_cache(cls) -> None:
        """Drop all cached singletons (test isolation; cache is process-global)."""
        cls._instances.clear()


def apply_block_hadamard(
    tensor: torch.Tensor,
    matrix: torch.Tensor,
    *,
    is_left: bool,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply a block-diagonal Hadamard transform with a materialized matrix.

    Pure, dynamo-traceable core: applies ``matrix`` (the ``block_size x block_size``
    ``R``, already on ``tensor``'s device/dtype) independently to each
    ``block_size`` block -- columns for ``is_left=False`` (``X @ R``) or rows for
    ``is_left=True`` (``R.T @ X``). Blocks never mix, so plain LEFT/RIGHT compose for
    cross-GEMM cancellation; ``inverse=True`` uses ``R.T`` -- a true round-trip
    inverse since ``R`` is orthogonal.
    """
    block_size = matrix.shape[0]
    dim = tensor.shape[0] if is_left else tensor.shape[1]
    if dim % block_size != 0:
        raise ValueError(
            f"Tensor dimension ({dim}) must be divisible by block_size ({block_size})."
        )

    matrix = matrix.T if inverse else matrix
    if is_left:
        # block-diagonal on rows: (X.T @ M).T == block-diag(M.T) @ X
        return (
            (tensor.T.reshape(-1, block_size) @ matrix)
            .reshape(tensor.T.shape)
            .T.contiguous()
        )
    # block-diagonal on cols: X @ M
    return (tensor.reshape(-1, block_size) @ matrix).reshape(tensor.shape)
