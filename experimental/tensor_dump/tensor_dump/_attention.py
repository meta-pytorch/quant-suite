# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Attention dumps from forward hooks on vLLM's ``Attention`` layers.

The name resolver's walk shows every module to ``_AttentionDump.visit``, which
installs a forward hook on each ``Attention`` layer whose name the
``QUANT_SUITE_TENSOR_DUMP_ATTENTION`` pattern matches. The hook sees the
layer's ``query``, ``key`` and ``value`` arguments, which hold the step's new
tokens, and the output the layer returns, so it works with any attention
backend.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from typing import Any

import torch
from tensor_dump._options import _ATTENTION_ENV
from tensor_dump._step import _StepTracker
from tensor_dump._writer import _AttentionLayout, _check_layer_name, _Writer
from torch.utils.hooks import RemovableHandle

_DECODER = "decoder"


def _attention_class() -> type[Any]:
    """vLLM's ``Attention`` layer class."""
    return importlib.import_module("tensor_dump._vllm").attention_class()


def _unsupported(layer: str, reason: str) -> NotImplementedError:
    return NotImplementedError(
        f"tensor_dump cannot record attention layer {layer!r}: {reason}; "
        f"exclude it from {_ATTENTION_ENV}"
    )


def _layout(module: torch.nn.Module, layer: str) -> _AttentionLayout:
    """The layout of a causal MHA/GQA decoder attention layer.

    Raises:
        NotImplementedError: for an attention variant that the record's
            fields do not describe.
    """
    attn_type = getattr(module, "attn_type", None)
    if attn_type != _DECODER:
        raise _unsupported(layer, f"attention type {attn_type!r} is not causal")
    impl = getattr(module, "impl", None)
    if getattr(impl, "alibi_slopes", None) is not None:
        raise _unsupported(layer, "ALiBi is not supported")
    if getattr(module, "has_sink", False):
        raise _unsupported(layer, "attention sinks are not supported")
    if getattr(module, "kv_sharing_target_layer_name", None) is not None:
        raise _unsupported(layer, "it shares the KV cache of another layer")
    num_heads = getattr(module, "num_heads", None)
    num_kv_heads = getattr(module, "num_kv_heads", None)
    head_size = getattr(module, "head_size", None)
    scale = getattr(impl, "scale", None)
    if num_heads is None or num_kv_heads is None or head_size is None:
        raise _unsupported(layer, "it lacks num_heads, num_kv_heads or head_size")
    if scale is None:
        raise _unsupported(layer, "its implementation has no scale")
    head_size_v = getattr(module, "head_size_v", None)
    if head_size_v is None:
        head_size_v = head_size
    sliding_window = getattr(module, "sliding_window", None)
    # Some backends store 0 for no soft cap.
    soft_cap = getattr(impl, "logits_soft_cap", None)
    backend_name = getattr(getattr(module, "attn_backend", None), "get_name", None)
    return _AttentionLayout(
        layer=layer,
        num_heads=int(num_heads),
        num_kv_heads=int(num_kv_heads),
        head_size=int(head_size),
        head_size_v=int(head_size_v),
        scale=float(scale),
        sliding_window=None if sliding_window is None else int(sliding_window),
        logits_soft_cap=float(soft_cap) if soft_cap else None,
        kv_cache_dtype=str(getattr(module, "kv_cache_dtype", "auto")),
        backend=str(backend_name()) if callable(backend_name) else None,
    )


def _argument(
    args: tuple[Any, ...], kwargs: dict[str, Any], index: int, name: str
) -> Any:
    return args[index] if len(args) > index else kwargs.get(name)


class _AttentionDump:
    """Hooks the selected attention layers and writes their tensors.

    Args:
        pattern: Selects layers by vLLM layer name, with ``re.match``.
        tensors: The tensors each selected layer writes.
        tracker: The step tracker shared with the linear executors.
        writer: The rank's writer.
    """

    def __init__(
        self,
        pattern: re.Pattern[str],
        tensors: frozenset[str],
        tracker: _StepTracker,
        writer: _Writer,
    ) -> None:
        self._pattern = pattern
        self._tensors = tensors
        self._tracker = tracker
        self._writer = writer
        self._attention_type: type[Any] | None = None
        self._handles: dict[int, RemovableHandle] = {}

    def visit(self, module: torch.nn.Module) -> None:
        """Hook ``module`` if it is a selected attention layer.

        Raises:
            NotImplementedError: for a selected layer that cannot be recorded.
        """
        attention_type = self._attention_type
        if attention_type is None:
            attention_type = _attention_class()
            self._attention_type = attention_type
        if not isinstance(module, attention_type) or id(module) in self._handles:
            return
        layer = getattr(module, "layer_name", None)
        if not isinstance(layer, str) or not self._pattern.match(layer):
            return
        if type(module) is not attention_type:
            raise _unsupported(
                layer, f"{type(module).__name__} is not {attention_type.__name__}"
            )
        _check_layer_name(layer)
        layout = _layout(module, layer)
        self._handles[id(module)] = module.register_forward_hook(
            self._hook(layout), with_kwargs=True
        )

    def remove(self) -> None:
        """Remove every hook."""
        for handle in self._handles.values():
            handle.remove()
        self._handles.clear()

    def _hook(
        self, layout: _AttentionLayout
    ) -> Callable[[torch.nn.Module, tuple[Any, ...], dict[str, Any], Any], None]:
        def hook(
            module: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            self._write(layout, args, kwargs, output)

        return hook

    def _write(
        self,
        layout: _AttentionLayout,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        if torch.compiler.is_compiling():
            raise RuntimeError("tensor_dump requires eager execution (--enforce-eager)")
        step = self._tracker.current()
        if step is None or not step.selected:
            return
        requests = self._tracker.layer_requests(layout.layer)
        query = _argument(args, kwargs, 0, "query")
        if not isinstance(query, torch.Tensor) or not isinstance(output, torch.Tensor):
            raise RuntimeError(
                f"attention layer {layout.layer!r} was called without a query "
                "tensor or returned no tensor"
            )
        shapes = {
            "q": (0, "query", layout.num_heads, layout.head_size),
            "k": (1, "key", layout.num_kv_heads, layout.head_size),
            "v": (2, "value", layout.num_kv_heads, layout.head_size_v),
        }
        tensors: dict[str, torch.Tensor] = {}
        for name in sorted(self._tensors):
            if name == "output":
                tensors[name] = output.reshape(-1, layout.num_heads, layout.head_size_v)
                continue
            index, argument, heads, size = shapes[name]
            tensor = _argument(args, kwargs, index, argument)
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(
                    f"attention layer {layout.layer!r} was called without a "
                    f"{argument} tensor"
                )
            tensors[name] = tensor.reshape(-1, heads, size)
        self._writer.attention(step, layout, requests, tensors, query.dtype)
