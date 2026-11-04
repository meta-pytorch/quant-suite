# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Self

from quant_suite_core.quant_config import QuantConfigBuilder


def _unknown_keys_error(
    where: str, unknown: Iterable[object], allowed: Iterable[str]
) -> ValueError:
    return ValueError(
        f"Unknown key(s) {list(unknown)!r} in {where}; allowed keys: {sorted(allowed)}"
    )


def _reject_unknown_keys(
    where: str, data: Mapping[Any, Any], allowed: Iterable[str]
) -> None:
    """Raise ``ValueError`` naming every key of ``data`` outside ``allowed``."""
    allowed_keys = frozenset(allowed)
    unknown = [key for key in data if key not in allowed_keys]
    if unknown:
        raise _unknown_keys_error(where, unknown, allowed_keys)


@dataclass
class QuantStageRecipe(ABC):
    """Stable contract for one serializable quantization pipeline stage."""

    stage_key: ClassVar[str]

    @abstractmethod
    def to_dict(self) -> dict[str, object]: ...

    def is_noop(self) -> bool:
        return False


@dataclass
class QuantTensorRecipe(QuantStageRecipe):
    """Stable contract for a tensor quantization recipe definition.

    Each concrete recipe sets ``discriminator`` to the stable ``type`` string its
    serialized payload carries.
    """

    stage_key: ClassVar[str] = "quantize"
    discriminator: ClassVar[str]

    @property
    def block_size(self) -> tuple[int, ...] | None:
        return None

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build the recipe from its inner payload, which may hold ``type``.

        Raises ``ValueError`` for a key the recipe does not define.
        """

    @classmethod
    def from_dict_with(
        cls, data: Mapping[str, Any], config_builder: QuantConfigBuilder
    ) -> Self:
        """Build the recipe from its inner payload with a runtime's config builder.

        The runtime's deserializer calls this. The default ignores
        ``config_builder`` and calls ``from_dict``; a recipe whose payload holds
        a ``QuantConfig`` overrides it to parse through ``config_builder``.
        """
        return cls.from_dict(data)

    def is_noop(self) -> bool:
        return self.discriminator == "none"
