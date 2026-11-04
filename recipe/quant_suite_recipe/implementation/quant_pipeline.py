# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
from quant_suite_core.rht_generator import (
    apply_block_hadamard,
    HadamardMode,
    RHTGenerator,
)
from quant_suite_recipe.api.definition import QuantTensorRecipe
from quant_suite_recipe.api.implementation import QuantizedTensor, QuantizeTensorHandler


class PipelineStage(nn.Module, ABC):
    """Base class for a quant pipeline stage.

    Being an ``nn.Module`` lets stages live in an ``nn.Sequential`` without
    casts, while the abstract ``forward`` enforces the contract that every stage
    maps a ``QuantizedTensor`` to a ``QuantizedTensor``.
    """

    @abstractmethod
    def forward(self, quantized: QuantizedTensor) -> QuantizedTensor: ...


class QuantizeStage(PipelineStage):
    """Pipeline stage that quantizes the wrapped tensor per a QuantTensorRecipe."""

    quant_recipe: QuantTensorRecipe
    _quantize_tensor: QuantizeTensorHandler
    tensor_name: str

    def __init__(
        self,
        quant_recipe: QuantTensorRecipe,
        quantize_tensor: QuantizeTensorHandler,
        tensor_name: str = "",
    ) -> None:
        super().__init__()
        self.quant_recipe = quant_recipe
        self._quantize_tensor = quantize_tensor
        self.tensor_name = tensor_name

    @property
    def block_size(self) -> tuple[int, ...] | None:
        return self.quant_recipe.block_size

    def forward(self, quantized: QuantizedTensor) -> QuantizedTensor:
        assert quantized.tensor is not None, "QuantizeStage requires an input tensor"
        return self._quantize_tensor(
            quantized.tensor,
            self.tensor_name,
            self.quant_recipe,
            False,
            False,
        )

    def extra_repr(self) -> str:
        return str(self.quant_recipe)


class RHTStage(PipelineStage):
    """Pipeline stage that applies a (random) Hadamard transform to the tensor.

    Modes compose so a quantization round-trip is expressed explicitly:

    - ``right``     (right pre):  ``X @ R``
    - ``right_inv`` (right post): ``X @ R^T``
    - ``left``      (left pre):   ``R^T @ X``
    - ``left_inv``  (left post):  ``R @ X``

    so ``[right, quantize, right_inv]`` == ``Q(X @ R) @ R^T`` is the RHT
    round-trip (``[left, quantize, left_inv]`` is ``R @ Q(R^T @ X)``), and
    ``[right, right_inv]`` / ``[left, left_inv]`` (no quantize) are the identity
    since ``R`` is orthogonal. The transform runs on
    the real (non-native) tensor.

    The matrix ``R`` is provided by a shared :class:`RHTGenerator` sub-module
    (a process-wide singleton per ``(block_size, seed)`` holding ``R`` as a
    ``requires_grad=False`` buffer). ``forward`` reads that buffer via
    ``get_matrix()``, so it is fully dynamo-traceable (no RNG / branches /
    global dict in the graph) and shares one ``R`` across every stage with the
    same config.
    """

    mode: HadamardMode
    block_size: int
    seed: int | None

    def __init__(
        self, mode: HadamardMode, block_size: int, seed: int | None = None
    ) -> None:
        super().__init__()
        assert mode != HadamardMode.NONE, "RHTStage requires a non-NONE mode"
        self.mode = mode
        self.block_size = block_size
        self.seed = seed
        # Shared singleton sub-module: rides device/dtype moves; one R per config.
        self._generator = RHTGenerator(block_size=block_size, seed=seed)

    def forward(self, quantized: QuantizedTensor) -> QuantizedTensor:
        tensor = quantized.tensor
        assert tensor is not None, "RHTStage requires an input tensor"
        if quantized.precision is not None:
            raise ValueError(
                "RHTStage operates on real tensors; it cannot transform a natively "
                "quantized tensor"
            )

        matrix = self._generator.get_matrix(device=tensor.device, dtype=tensor.dtype)
        transformed = apply_block_hadamard(
            tensor,
            matrix,
            is_left=self.mode.is_left,
            inverse=self.mode.is_inverse,
        )
        return QuantizedTensor(tensor=transformed)

    def extra_repr(self) -> str:
        return f"mode={self.mode.value}, block_size={self.block_size}, seed={self.seed}"


class Pipeline(nn.Module):
    """An ordered quantization pipeline.

    Holds an ``nn.Sequential`` of ``PipelineStage``s. ``forward`` takes a raw
    ``torch.Tensor``, wraps it in a ``QuantizedTensor``, runs it through the
    stages, and ALWAYS returns a ``QuantizedTensor`` — so call sites never have
    to disambiguate the output type. An empty pipeline is the identity.
    """

    stages: nn.Sequential

    def __init__(self, *stages: PipelineStage) -> None:
        super().__init__()
        self.stages = nn.Sequential(*stages)

    def forward(self, tensor: torch.Tensor) -> QuantizedTensor:
        return self.stages(QuantizedTensor(tensor=tensor))

    def is_noop(self) -> bool:
        return len(self.stages) == 0


@dataclass(frozen=True)
class PipelineSlice:
    """A pipeline split around its final quantization stage."""

    head: Pipeline
    qdq: QuantizeStage | None
    tail: Pipeline


def split_pipeline(pipeline: Pipeline) -> PipelineSlice:
    """Split ``pipeline`` around its final ``QuantizeStage``.

    A pipeline without a QDQ is returned entirely as the head.
    """
    stages = tuple(cast(PipelineStage, stage) for stage in pipeline.stages)
    qdq_index = next(
        (
            index
            for index in range(len(stages) - 1, -1, -1)
            if isinstance(stages[index], QuantizeStage)
        ),
        None,
    )
    if qdq_index is None:
        return PipelineSlice(
            head=Pipeline(*stages),
            qdq=None,
            tail=Pipeline(),
        )
    qdq = stages[qdq_index]
    assert isinstance(qdq, QuantizeStage)
    return PipelineSlice(
        head=Pipeline(*stages[:qdq_index]),
        qdq=qdq,
        tail=Pipeline(*stages[qdq_index + 1 :]),
    )
