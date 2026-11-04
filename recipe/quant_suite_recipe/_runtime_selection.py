# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Selection of the environment's recipe runtime from installed packages.

A package publishes one complete ``QuantRecipeRuntime`` in the
``quant_suite.recipe_runtimes`` entry-point group. ``resolve_runtime(base)``
selects:

- ``base`` when no runtime is published;
- the published runtime when there is exactly one;
- the one named by ``QUANT_SUITE_RECIPE_RUNTIME`` when there are several.

``base`` is kept when it already accepts every recipe type, config extension
and preset the selected runtime accepts. Otherwise the selected runtime must
accept everything ``base`` accepts.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from importlib import metadata
from typing import Final

from quant_suite_recipe.runtime import QuantRecipeRuntime


_ENTRY_POINT_GROUP: Final = "quant_suite.recipe_runtimes"
_RUNTIME_ENV: Final = "QUANT_SUITE_RECIPE_RUNTIME"


def _describe(entry_point: metadata.EntryPoint) -> str:
    dist = entry_point.dist
    package = dist.name if dist is not None else "an unknown package"
    return f"{entry_point.name}={entry_point.value} from {package}"


def _select(entry_points: Sequence[metadata.EntryPoint]) -> metadata.EntryPoint | None:
    installed = "; ".join(_describe(ep) for ep in entry_points) or "none"
    name = os.environ.get(_RUNTIME_ENV)
    if name is None:
        if len(entry_points) > 1:
            raise RuntimeError(
                f"Several QuantSuite recipe runtimes are installed in the "
                f"{_ENTRY_POINT_GROUP!r} entry-point group; set {_RUNTIME_ENV} to "
                f"the name of one. Installed: {installed}"
            )
        return entry_points[0] if entry_points else None
    matches = [ep for ep in entry_points if ep.name == name]
    if len(matches) != 1:
        problem = "names no runtime" if not matches else "names several runtimes"
        raise RuntimeError(
            f"{_RUNTIME_ENV}={name!r} {problem} in the {_ENTRY_POINT_GROUP!r} "
            f"entry-point group. Installed: {installed}"
        )
    return matches[0]


def _load(entry_point: metadata.EntryPoint) -> QuantRecipeRuntime:
    try:
        runtime = entry_point.load()
    except Exception as error:
        raise RuntimeError(
            f"QuantSuite recipe runtime entry point {_describe(entry_point)} "
            f"failed to load: {type(error).__name__}: {error}"
        ) from error
    if not isinstance(runtime, QuantRecipeRuntime):
        raise TypeError(
            f"QuantSuite recipe runtime entry point {_describe(entry_point)} "
            f"resolved to {type(runtime).__name__}, expected QuantRecipeRuntime"
        )
    return runtime


def _missing(runtime: QuantRecipeRuntime, required: QuantRecipeRuntime) -> list[str]:
    """What ``required`` accepts that ``runtime`` does not, one entry per kind."""

    def discriminators(r: QuantRecipeRuntime) -> set[str]:
        return {recipe_type.discriminator for recipe_type, _ in r.registrations}

    builder = runtime.config_builder
    required_builder = required.config_builder
    missing = {
        "recipe types": sorted(discriminators(required) - discriminators(runtime)),
        "config extensions": sorted(
            e.name for e in set(required_builder.extensions) - set(builder.extensions)
        ),
        "presets": sorted(
            name
            for name, spec in required_builder.presets.items()
            if builder.presets.get(name) != spec
        ),
    }
    return [f"{kind} {names}" for kind, names in missing.items() if names]


def resolve_runtime(base: QuantRecipeRuntime) -> QuantRecipeRuntime:
    """The environment's recipe runtime, starting from the base runtime ``base``.

    Raises:
        RuntimeError: if several runtimes are installed and
            ``QUANT_SUITE_RECIPE_RUNTIME`` does not name exactly one of them, if
            the selected entry point fails to load, or if neither the selected
            runtime nor ``base`` accepts every recipe type, config extension and
            preset of the other.
        TypeError: if the selected entry point is not a ``QuantRecipeRuntime``.
    """
    entry_point = _select(tuple(metadata.entry_points(group=_ENTRY_POINT_GROUP)))
    if entry_point is None:
        return base
    runtime = _load(entry_point)
    if not _missing(base, runtime):
        return base
    problems = _missing(runtime, base)
    if problems:
        raise RuntimeError(
            f"QuantSuite recipe runtime entry point {_describe(entry_point)} does "
            f"not accept everything the base runtime does; missing or "
            f"changed: {'; '.join(problems)}"
        )
    return runtime
