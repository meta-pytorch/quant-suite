# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Executors that write the tensors a ``dump`` stage selects.

``DumpLinear`` and ``DumpAttention`` compute exactly what their base classes
compute. With ``QUANT_SUITE_TENSOR_DUMP_DIR`` set, an executor whose recipe
holds a ``dump`` stage also writes its selected tensors at the selected steps.

``DumpAttention`` reads the operands of the two attention products:
``Q @ K^T`` gives Q and K, and ``P @ V`` gives V and the attention output. It
requires one request per attention micro-batch, so that the operands are
unpadded.
"""

from __future__ import annotations

from typing import Any

import torch
from quant_suite_recipe.api import QuantizedTensor
from quant_suite_recipe.definition import QuantAttentionRecipe, QuantLinearRecipe
from quant_suite_recipe.implementation.attention import QuantizedAttention
from quant_suite_recipe.implementation.linear import QuantizedLinear
from tensor_dump import _session
from tensor_dump._step import _RequestInfo, _StepInfo
from tensor_dump._writer import _AttentionLayout, _PendingAttention
from tensor_dump.recipe import _ATTENTION_TENSORS, _LINEAR_TENSORS, DumpRecipe


_ONE_REQUEST = (
    "set QUANT_SUITE_MHA_MICRO_BATCH_SIZE=1 or start vLLM with --max-num-seqs 1"
)


def _dump_tensors(
    recipe: QuantLinearRecipe | QuantAttentionRecipe,
    allowed: frozenset[str],
    kind: str,
) -> frozenset[str]:
    """The tensors the recipe's ``dump`` stage selects; empty without one.

    Raises:
        ValueError: if the recipe holds more than one ``dump`` stage, or its
            stage names a tensor outside ``allowed``.
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
            f"{kind} recipe holds {len(stages)} dump stages; at most one is allowed"
        )
    selected = stages[0].tensors
    tensors = allowed if selected is None else frozenset(selected)
    if not tensors <= allowed:
        raise ValueError(
            f"dump tensors {sorted(tensors - allowed)} are not valid for {kind} "
            f"layers; expected a subset of {sorted(allowed)}"
        )
    return tensors


def _check_one_request(layer: str, micro_batch_size: int | None) -> None:
    """Raise at startup if an attention product can hold several requests.

    Uses the backend's micro-batch size and vLLM's ``max_num_seqs``. When the
    micro-batch size is unknown and ``max_num_seqs`` is not 1, each written
    product is checked instead.
    """
    session = _session._get()
    max_num_seqs = None if session is None else session.settings.max_num_seqs
    if micro_batch_size is None or micro_batch_size == 1 or max_num_seqs == 1:
        return
    raise RuntimeError(
        f"tensor_dump needs one request per attention product, but layer {layer!r} "
        f"multiplies up to {micro_batch_size} requests at once; {_ONE_REQUEST}"
    )


def _track(executor: DumpLinear | DumpAttention) -> None:
    if not executor._tensors:
        return
    session = _session._get()
    if session is None:
        _session._warn_disabled()
    else:
        session.resolver.track(executor)


class DumpLinear(QuantizedLinear):
    """``QuantizedLinear`` that writes its input, weight and output.

    Takes the arguments of ``QuantizedLinear``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tensors: frozenset[str] = _dump_tensors(
            self.recipe, _LINEAR_TENSORS, "linear"
        )
        self._layer_name: str | None = None
        _track(self)

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


class DumpAttention(QuantizedAttention):
    """``QuantizedAttention`` that writes Q, K, V and the attention output.

    Takes the arguments of ``QuantizedAttention``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tensors: frozenset[str] = _dump_tensors(
            self.recipe, _ATTENTION_TENSORS, "attention"
        )
        self._layout: _AttentionLayout | None = None
        self._pending: _PendingAttention | None = None
        self._request_step: int | None = None
        self._request_index = 0
        _track(self)

    def bmm(
        self,
        left: QuantizedTensor,
        right: QuantizedTensor,
        *,
        transpose_right: bool = False,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        output = super().bmm(
            left, right, transpose_right=transpose_right, out_dtype=out_dtype
        )
        session = _session._get()
        if session is None:
            return output
        step = session.tracker.current()
        if step is None or not step.selected or not self._tensors:
            return output
        layout = self._require_layout()
        left_tensor, right_tensor = left.tensor, right.tensor
        if left_tensor is None or right_tensor is None:
            raise ValueError("tensor_dump needs both attention operands")
        if left_tensor.shape[0] != layout.num_heads:
            raise RuntimeError(
                f"attention layer {layout.layer!r} multiplies "
                f"{left_tensor.shape[0] // layout.num_heads} requests at once; "
                f"tensor_dump needs one request per attention product; {_ONE_REQUEST}"
            )
        if transpose_right:
            if self._pending is not None:
                raise RuntimeError(
                    f"attention layer {layout.layer!r} computed Q @ K^T twice "
                    "without P @ V"
                )
            request = self._next_request(session, step, layout)
            if (
                left_tensor.shape[1] != request.q_len
                or right_tensor.shape[1] != request.kv_len
            ):
                raise RuntimeError(
                    f"attention layer {layout.layer!r} got Q {tuple(left_tensor.shape)} "
                    f"and K {tuple(right_tensor.shape)} for request {request.request}, "
                    f"expected q_len {request.q_len} and kv_len {request.kv_len}"
                )
            self._pending = session.writer.attention_qk(
                step, layout, request, self._tensors, left_tensor, right_tensor
            )
        else:
            pending = self._pending
            if pending is None:
                raise RuntimeError(
                    f"attention layer {layout.layer!r} computed P @ V without Q @ K^T"
                )
            self._pending = None
            session.writer.attention_v_out(pending, right_tensor, output)
        return output

    def _next_request(
        self, session: _session._Session, step: _StepInfo, layout: _AttentionLayout
    ) -> _RequestInfo:
        if self._request_step != step.step:
            self._request_step = step.step
            self._request_index = 0
        requests = session.tracker.layer_requests(layout.layer)
        if self._request_index >= len(requests):
            raise RuntimeError(
                f"attention layer {layout.layer!r} computed more Q @ K^T products "
                f"than the step's {len(requests)} requests"
            )
        request = requests[self._request_index]
        self._request_index += 1
        return request

    def _bind(self, parent: torch.nn.Module, path: str) -> None:
        impl = getattr(parent, "impl", None)
        num_heads = getattr(parent, "num_heads", None)
        num_kv_heads = getattr(parent, "num_kv_heads", None)
        head_size = getattr(parent, "head_size", None)
        scale = getattr(impl, "scale", None)
        if (
            num_heads is None
            or num_kv_heads is None
            or head_size is None
            or scale is None
        ):
            raise NotImplementedError(
                f"tensor_dump records MHA/GQA attention only; layer {path!r} "
                "lacks num_heads, num_kv_heads, head_size or impl.scale"
            )
        layer_name = getattr(parent, "layer_name", None)
        layer = layer_name if isinstance(layer_name, str) and layer_name else path
        _check_one_request(layer, getattr(impl, "micro_batch_size", None))
        self._layout = _AttentionLayout(
            layer=layer,
            num_heads=int(num_heads),
            num_kv_heads=int(num_kv_heads),
            head_size=int(head_size),
            scale=float(scale),
        )

    def _require_layout(self) -> _AttentionLayout:
        if self._layout is None:
            raise RuntimeError(
                "tensor_dump could not resolve the name of an attention layer with "
                "a dump stage"
            )
        return self._layout


__all__ = ["DumpAttention", "DumpLinear"]
