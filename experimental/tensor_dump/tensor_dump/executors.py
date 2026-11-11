# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The linear executor that writes the tensors a ``dump`` stage selects.

``DumpLinear`` computes exactly what ``QuantizedLinear`` computes. With
``QUANT_SUITE_TENSOR_DUMP_DIR`` set, one whose recipe holds a ``dump`` stage
also writes its selected tensors at the selected steps. Building any
``DumpLinear`` starts the dump session, which also hooks the attention layers
that ``QUANT_SUITE_TENSOR_DUMP_ATTENTION`` selects.
"""

from __future__ import annotations

from typing import Any

import torch
from quant_suite_recipe.api import QuantizedTensor
from quant_suite_recipe.definition import QuantLinearRecipe
from quant_suite_recipe.implementation.linear import QuantizedLinear
from tensor_dump import _session
from tensor_dump.recipe import _LINEAR_TENSORS, DumpRecipe


def _dump_tensors(recipe: QuantLinearRecipe) -> frozenset[str]:
    """The tensors the recipe's ``dump`` stage selects; empty without one.

    Raises:
        ValueError: if the recipe holds more than one ``dump`` stage.
    """
    stages = [
        stage
        for name in recipe.pipeline_fields()
        for stage in getattr(recipe, name)
        if isinstance(stage, DumpRecipe)
    ]
    if not stages:
        return frozenset()
    if len(stages) > 1:
        raise ValueError(
            f"linear recipe holds {len(stages)} dump stages; at most one is allowed"
        )
    selected = stages[0].tensors
    return _LINEAR_TENSORS if selected is None else frozenset(selected)


class DumpLinear(QuantizedLinear):
    """``QuantizedLinear`` that writes its input, weight and output.

    Takes the arguments of ``QuantizedLinear``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tensors: frozenset[str] = _dump_tensors(self.recipe)
        self._layer_name: str | None = None
        session = _session._get()
        if not self._tensors:
            return
        if session is None:
            _session._warn_disabled()
        else:
            session.resolver.track(self)

    def forward(
        self,
        x: torch.Tensor,
        weight: QuantizedTensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        output = super().forward(x, weight, bias)
        session = _session._get()
        if session is None:
            return output
        if torch.compiler.is_compiling():
            raise RuntimeError("tensor_dump requires eager execution (--enforce-eager)")
        # Called by every executor, so the step id advances whatever is dumped.
        step = session.tracker.current()
        if step is None or not step.selected or not self._tensors:
            return output
        session.writer.linear(
            step, self._require_name(), self._tensors, x, weight, bias, output
        )
        return output

    def _bind(self, parent: torch.nn.Module, path: str) -> None:
        prefix = getattr(parent, "prefix", None)
        name = prefix if isinstance(prefix, str) and prefix else path
        self._layer_name = name or None

    def _require_name(self) -> str:
        if self._layer_name is None:
            raise RuntimeError(
                "tensor_dump could not resolve the name of a linear layer with a "
                "dump stage"
            )
        return self._layer_name


__all__ = ["DumpLinear"]
