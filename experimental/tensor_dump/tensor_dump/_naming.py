# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Layer names for executors, which are never told their parent module.

An executor is a registered child of the layer it computes for. A global
forward pre-hook walks the module tree at the first forward, finds each tracked
executor's parent and binds the executor to it, and shows every module it
walks to an optional visitor. The hook removes itself once every tracked
executor is named and no visitor is set, or when ``finish`` is called.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import Protocol

import torch
from torch.utils.hooks import RemovableHandle


class _Bindable(Protocol):
    def _bind(self, parent: torch.nn.Module, path: str) -> None: ...


class _NameResolver:
    """Binds tracked executors to their parent modules."""

    def __init__(self) -> None:
        self._pending: weakref.WeakSet[_Bindable] = weakref.WeakSet()
        self._visited: set[int] = set()
        self._visitor: Callable[[torch.nn.Module], None] | None = None
        self._handle: RemovableHandle | None = None

    def track(self, executor: _Bindable) -> None:
        """Bind ``executor`` at the next forward of a module that contains it."""
        self._pending.add(executor)
        self._install()

    def scan(self, visitor: Callable[[torch.nn.Module], None]) -> None:
        """Call ``visitor`` once on every module the walks reach, until
        ``finish``."""
        self._visitor = visitor
        self._install()

    def finish(self) -> None:
        """Remove the hook; executors still unnamed stay unnamed."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._visited.clear()
        self._visitor = None

    def finish_from_second_step(self, step: int) -> None:
        """``finish`` at step 1 and later, by when every module that runs has
        been walked."""
        if step > 0:
            self.finish()

    def _install(self) -> None:
        if self._handle is None:
            self._handle = torch.nn.modules.module.register_module_forward_pre_hook(
                self._hook
            )

    def _hook(self, module: torch.nn.Module, args: object) -> None:
        if id(module) in self._visited:
            return
        pending = {id(executor): executor for executor in self._pending}
        visitor = self._visitor
        for path, parent in module.named_modules():
            if visitor is not None and id(parent) not in self._visited:
                visitor(parent)
            self._visited.add(id(parent))
            for child in parent.children():
                executor = pending.pop(id(child), None)
                if executor is not None:
                    self._pending.discard(executor)
                    executor._bind(parent, path)
        if not self._pending and self._visitor is None:
            self.finish()
