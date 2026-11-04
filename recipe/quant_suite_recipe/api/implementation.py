# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, cast

import torch
import torch.nn as nn
from quant_suite_recipe.api.definition import QuantTensorRecipe


@dataclass
class QuantizedTensor:
    """Tensor quantization result shared by all recipe implementations."""

    tensor: torch.Tensor | None = None
    scale: torch.Tensor | None = None
    global_scale: torch.Tensor | None = None
    precision: Enum | None = None
    quantized_val: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def _is_typed_field(self, key: str) -> bool:
        return any(key == dataclass_field.name for dataclass_field in fields(self))

    def __getitem__(self, key: str) -> Any:
        if self._is_typed_field(key):
            return getattr(self, key)
        return self.metadata.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if self._is_typed_field(key):
            setattr(self, key, value)
        else:
            self.metadata[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        if self._is_typed_field(key):
            return getattr(self, key)
        return self.metadata.get(key, default)

    @staticmethod
    def _as_parameter(tensor: torch.Tensor, requires_grad: bool) -> nn.Parameter:
        if isinstance(tensor, nn.Parameter):
            tensor.requires_grad_(requires_grad)
            return tensor
        return nn.Parameter(tensor, requires_grad=requires_grad)

    def register_parameters(
        self,
        module: nn.Module,
        name: str,
        *,
        requires_grad: bool = False,
        tensor_attrs: dict[str, Any] | None = None,
        set_tensor_attrs: Callable[[torch.Tensor, dict[str, Any]], None] | None = None,
    ) -> nn.Parameter:
        """Register the tensor and any scale metadata on ``module``."""
        assert self.tensor is not None, f"{name} tensor must be provided"

        tensor_parameter = self._as_parameter(self.tensor, requires_grad)
        if tensor_attrs is not None and set_tensor_attrs is not None:
            set_tensor_attrs(tensor_parameter, tensor_attrs)
        module.register_parameter(name, tensor_parameter)

        if self.scale is not None:
            module.register_parameter(
                f"{name}_scale",
                self._as_parameter(self.scale, requires_grad),
            )
        if self.global_scale is not None:
            module.register_parameter(
                f"{name}_global_scale",
                self._as_parameter(self.global_scale, requires_grad),
            )

        precision_attr = f"{name}_precision"
        if self.precision is not None:
            setattr(module, precision_attr, self.precision)
        elif hasattr(module, precision_attr):
            delattr(module, precision_attr)
        return tensor_parameter

    @classmethod
    def from_registered_parameters(
        cls,
        module: nn.Module,
        name: str,
    ) -> "QuantizedTensor":
        return cls(
            tensor=cast(torch.Tensor, getattr(module, name)),
            scale=cast(torch.Tensor | None, getattr(module, f"{name}_scale", None)),
            global_scale=cast(
                torch.Tensor | None, getattr(module, f"{name}_global_scale", None)
            ),
            precision=getattr(module, f"{name}_precision", None),
        )


QuantizeTensorHandler = Callable[
    [torch.Tensor, str, QuantTensorRecipe, bool, bool], QuantizedTensor
]


QuantTensorRecipeRegistration = tuple[type[QuantTensorRecipe], QuantizeTensorHandler]
