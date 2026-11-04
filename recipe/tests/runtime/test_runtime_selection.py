# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import os
import unittest
from dataclasses import dataclass
from typing import ClassVar
from unittest.mock import patch

from quant_suite_core.extension import QuantConfigExtension
from quant_suite_core.quant_config import (
    DEFAULT_QUANT_CONFIG_BUILDER,
    QuantConfigBuilder,
)
from quant_suite_recipe import _runtime_selection
from quant_suite_recipe.runtime import QUANT_RECIPE_REGISTRATIONS, QuantRecipeRuntime


@dataclass(frozen=True)
class _ToyBits(QuantConfigExtension):
    name: ClassVar[str] = "toy_bits"
    bits: int = 2


@dataclass(frozen=True)
class _OtherBits(QuantConfigExtension):
    name: ClassVar[str] = "other_bits"
    bits: int = 2


_TOY_PRESETS: dict[str, dict[str, object]] = {
    "mxfp4_toy_bits": {
        "elem_dtype": "e2m1",
        "block_size": "1x32",
        "scale_dtype": "e8m0",
        "extensions": {"toy_bits": {"bits": 3}},
    }
}


def _toy_runtime() -> QuantRecipeRuntime:
    return QuantRecipeRuntime(
        config_builder=DEFAULT_QUANT_CONFIG_BUILDER.extend((_ToyBits,), _TOY_PRESETS)
    )


@dataclass(frozen=True)
class _FakeDistribution:
    name: str


class _FakeEntryPoint:
    def __init__(self, name: str, loaded: object) -> None:
        self.name = name
        self.value = f"{name}_pkg.runtime:RUNTIME"
        self.dist: _FakeDistribution | None = _FakeDistribution(f"{name}_pkg")
        self._loaded = loaded

    def load(self) -> object:
        if isinstance(self._loaded, BaseException):
            raise self._loaded
        return self._loaded


class ResolveRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("QUANT_SUITE_RECIPE_RUNTIME", None)
        self.base = QuantRecipeRuntime()

    def resolve(self, *entry_points: _FakeEntryPoint) -> QuantRecipeRuntime:
        with patch.object(
            _runtime_selection.metadata, "entry_points", return_value=entry_points
        ) as entry_points_mock:
            runtime = _runtime_selection.resolve_runtime(self.base)
        entry_points_mock.assert_called_once_with(group="quant_suite.recipe_runtimes")
        return runtime

    def test_no_entry_point_uses_the_base(self) -> None:
        self.assertIs(self.resolve(), self.base)

    def test_one_entry_point_is_used(self) -> None:
        toy = _toy_runtime()

        self.assertIs(self.resolve(_FakeEntryPoint("toy", toy)), toy)

    def test_base_that_covers_the_entry_point_is_kept(self) -> None:
        self.base = _toy_runtime()

        runtime = self.resolve(_FakeEntryPoint("built_in", QuantRecipeRuntime()))

        self.assertIs(runtime, self.base)

    def test_equivalent_entry_point_keeps_the_base(self) -> None:
        runtime = self.resolve(_FakeEntryPoint("same", QuantRecipeRuntime()))

        self.assertIs(runtime, self.base)

    def test_two_entry_points_raise_and_list_both(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "QUANT_SUITE_RECIPE_RUNTIME.*first=first_pkg.runtime:RUNTIME from "
            "first_pkg; second=second_pkg.runtime:RUNTIME from second_pkg",
        ):
            self.resolve(
                _FakeEntryPoint("first", _toy_runtime()),
                _FakeEntryPoint("second", _toy_runtime()),
            )

    def test_env_var_selects_one_of_several(self) -> None:
        second = _toy_runtime()
        os.environ["QUANT_SUITE_RECIPE_RUNTIME"] = "second"

        runtime = self.resolve(
            _FakeEntryPoint("first", _toy_runtime()),
            _FakeEntryPoint("second", second),
        )

        self.assertIs(runtime, second)

    def test_env_var_with_an_unknown_name_lists_the_installed_names(self) -> None:
        os.environ["QUANT_SUITE_RECIPE_RUNTIME"] = "missing"

        with self.assertRaisesRegex(
            RuntimeError, "'missing' names no runtime.*toy=toy_pkg.runtime:RUNTIME"
        ):
            self.resolve(_FakeEntryPoint("toy", _toy_runtime()))

    def test_broken_entry_point_names_the_entry_point_and_package(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "broken=broken_pkg.runtime:RUNTIME from broken_pkg failed to load: "
            "ImportError: no module named toy",
        ):
            self.resolve(_FakeEntryPoint("broken", ImportError("no module named toy")))

    def test_non_runtime_object_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            TypeError,
            "bad=bad_pkg.runtime:RUNTIME from bad_pkg resolved to object, "
            "expected QuantRecipeRuntime",
        ):
            self.resolve(_FakeEntryPoint("bad", object()))

    def test_runtime_missing_a_base_recipe_type_is_rejected(self) -> None:
        runtime = QuantRecipeRuntime(
            (
                registration
                for registration in QUANT_RECIPE_REGISTRATIONS
                if registration[0].discriminator != "native"
            ),
            config_builder=DEFAULT_QUANT_CONFIG_BUILDER.extend((_ToyBits,)),
        )

        with self.assertRaisesRegex(
            RuntimeError, r"narrow=.*from narrow_pkg.*recipe types \['native'\]"
        ):
            self.resolve(_FakeEntryPoint("narrow", runtime))

    def test_runtime_missing_base_presets_is_rejected(self) -> None:
        runtime = QuantRecipeRuntime(
            config_builder=QuantConfigBuilder(extensions=(_ToyBits,))
        )

        with self.assertRaisesRegex(RuntimeError, r"presets \[.*'mxfp4'"):
            self.resolve(_FakeEntryPoint("narrow", runtime))

    def test_runtime_missing_a_base_extension_is_rejected(self) -> None:
        self.base = QuantRecipeRuntime(
            config_builder=DEFAULT_QUANT_CONFIG_BUILDER.extend((_ToyBits,))
        )
        runtime = QuantRecipeRuntime(
            config_builder=DEFAULT_QUANT_CONFIG_BUILDER.extend((_OtherBits,))
        )

        with self.assertRaisesRegex(RuntimeError, r"config extensions \['toy_bits'\]"):
            self.resolve(_FakeEntryPoint("narrow", runtime))


if __name__ == "__main__":
    unittest.main()
