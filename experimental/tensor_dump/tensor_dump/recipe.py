# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The ``dump`` tensor recipe type and its identity handler.

A ``dump`` stage leaves its tensor unchanged. Its presence in a module recipe
marks the module for dumping, and its ``tensors`` option selects what the
module writes::

    linear_recipe:
      '.*':
        activation:
          type: dump
          tensors: [input, output]
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from quant_suite_recipe.api import QuantizedTensor, QuantTensorRecipe

_LINEAR_TENSORS: frozenset[str] = frozenset({"input", "weight", "output"})

_ATTENTION_TENSORS: frozenset[str] = frozenset({"q", "k", "v", "output"})

_KEYS: frozenset[str] = frozenset({"type", "tensors"})


@dataclass
class DumpRecipe(QuantTensorRecipe):
    """Identity stage that marks its module for dumping.

    ``tensors`` lists the tensors to write: ``input``, ``weight`` and ``output``
    for a linear module, ``q``, ``k``, ``v`` and ``output`` for an attention
    module; ``None`` writes every tensor of the module's kind. The names are
    stored sorted.
    """

    discriminator: ClassVar[str] = "dump"
    tensors: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.tensors is None:
            return
        if not isinstance(self.tensors, tuple) or not all(
            isinstance(name, str) for name in self.tensors
        ):
            raise TypeError(
                f"DumpRecipe.tensors must be a tuple of str or None, got "
                f"{self.tensors!r}"
            )
        if not self.tensors:
            raise ValueError("DumpRecipe.tensors must not be empty; omit it instead")
        duplicates = sorted(
            {name for name in self.tensors if self.tensors.count(name) > 1}
        )
        if duplicates:
            raise ValueError(f"DumpRecipe.tensors repeats {duplicates}")
        unknown = sorted(set(self.tensors) - _LINEAR_TENSORS - _ATTENTION_TENSORS)
        if unknown:
            raise ValueError(
                f"DumpRecipe.tensors has unknown names {unknown}; expected names "
                f"from {sorted(_LINEAR_TENSORS | _ATTENTION_TENSORS)}"
            )
        self.tensors = tuple(sorted(self.tensors))

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"type": self.discriminator}
        if self.tensors is not None:
            payload["tensors"] = list(self.tensors)
        return {self.stage_key: payload}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DumpRecipe:
        """Build a ``DumpRecipe`` from ``{type: dump, tensors: [...]}``.

        Raises:
            ValueError: for a key other than ``type`` and ``tensors``, or a
                ``tensors`` value that is not a list of str.
        """
        unknown = [key for key in data if key not in _KEYS]
        if unknown:
            raise ValueError(
                f"Unknown key(s) {unknown!r} in a 'dump' quantize config; allowed "
                f"keys: {sorted(_KEYS)}"
            )
        tensors = data.get("tensors")
        if tensors is None:
            return cls()
        if not isinstance(tensors, list) or not all(
            isinstance(name, str) for name in tensors
        ):
            raise ValueError(
                f"'dump' option 'tensors' must be a list of str, got {tensors!r}"
            )
        return cls(tensors=tuple(tensors))


def dump_tensor(
    tensor: torch.Tensor,
    tensor_name: str,
    recipe: QuantTensorRecipe,
    transpose: bool,
    return_quant_info: bool,
) -> QuantizedTensor:
    """The ``dump`` handler: returns ``tensor`` itself, with no copy or cast."""
    if not isinstance(recipe, DumpRecipe):
        raise TypeError(
            f"dump_tensor expects a DumpRecipe, got {type(recipe).__name__}"
        )
    return QuantizedTensor(tensor=tensor)


__all__ = ["DumpRecipe", "dump_tensor"]
