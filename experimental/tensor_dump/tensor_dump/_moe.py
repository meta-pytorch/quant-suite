# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoE dumps from forward hooks on vLLM's MoE layers (``MoERunner``).

The name resolver's walk shows every module to ``_MoEDump.visit``, which hooks
each ``MoERunner`` whose layer name the ``QUANT_SUITE_TENSOR_DUMP_MOE`` pattern
matches. A forward hook records the layer's ``hidden_states`` and
``router_logits`` arguments and its output. The tensors the experts kernel
computes inside the layer come from the kernel's own outputs:

- ``topk_ids``: the expert ids each token is routed to, through vLLM's
  routed-experts capture, which a kernel that routes the tokens itself fills
  from inside the kernel;
- ``topk_weights``: the routing weights, from the unfinalized kernel output a
  ``post_gemm2_hook`` receives, or from the arguments of the experts call when
  the routing runs before the kernel;
- ``gemm2``: the unweighted output of the second expert GEMM of each
  ``(token, k)`` route, ``[tokens, top_k, hidden]``, from the
  ``post_gemm2_hook``, which receives it before the top-k reduction. Under
  tensor parallelism each rank holds the partial sum over its share of the
  intermediate dimension. The rows are copied to the host inside the hook, at
  most ``_GEMM2_CHUNK_BYTES`` of them at a time, so the GPU holds no more than
  that beyond what the layer itself allocates.

The hooks only copy tensors; the layer computes what it computes without them.
A model that calls the experts kernel without the layer's forward, as fast
paths for small batches can, is not recorded.
"""

from __future__ import annotations

import importlib
import logging
import re
from collections.abc import Callable
from typing import Any

import torch
from tensor_dump._options import _MOE_ENV, _MOE_TENSORS_ENV
from tensor_dump._step import _StepInfo, _StepTracker
from tensor_dump._writer import _check_layer_name, _dtype_name, _MoELayout, _Writer
from torch.utils.hooks import RemovableHandle

logger: logging.Logger = logging.getLogger(__name__)

# Tensors identical on every tensor-parallel rank when the layer's tokens are.
_TOKEN_TENSORS: frozenset[str] = frozenset({"input", "router_logits"})
# GPU bytes of GEMM2 rows gathered at once before they are copied to the host.
_GEMM2_CHUNK_BYTES = 32 * 2**20
# Captured tensors that ``_gather_to_host`` already copies to the host.
_HOST_COPIES: frozenset[str] = frozenset({"gemm2"})


def _moe_runner_class() -> type[Any]:
    """vLLM's ``MoERunner`` class."""
    return importlib.import_module("tensor_dump._vllm").moe_runner_class()


def _bind_routing_capture(module: torch.nn.Module, capturer: Any) -> None:
    """Have ``module`` call ``capturer.capture(layer_id, topk_ids)``."""
    importlib.import_module("tensor_dump._vllm").bind_routed_experts_capturer(
        module, capturer
    )


def _unsupported(layer: str, reason: str) -> NotImplementedError:
    return NotImplementedError(
        f"tensor_dump cannot record MoE layer {layer!r}: {reason}; exclude it "
        f"from {_MOE_ENV}"
    )


def _argument(
    args: tuple[Any, ...], kwargs: dict[str, Any], index: int, name: str
) -> Any:
    return args[index] if len(args) > index else kwargs.get(name)


def _moe_config(module: torch.nn.Module) -> Any:
    """The layer's vLLM ``FusedMoEConfig``."""
    return module.moe_config


def _routed_experts(module: torch.nn.Module) -> Any:
    """The layer's routed experts module."""
    return module.routed_experts


def _fused_experts(module: torch.nn.Module) -> Any | None:
    """The experts kernel object of the layer's quantization method, if any."""
    quant_method = getattr(_routed_experts(module), "quant_method", None)
    kernel = getattr(quant_method, "moe_kernel", None)
    return getattr(kernel, "fused_experts", None)


def _is_monolithic(module: torch.nn.Module) -> bool:
    quant_method = getattr(_routed_experts(module), "quant_method", None)
    return bool(getattr(quant_method, "is_monolithic", False))


def _name(value: object) -> str | None:
    """An enum's value or an object's string; ``None`` stays ``None``."""
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _dtype(value: object) -> str | None:
    return _dtype_name(value) if isinstance(value, torch.dtype) else None


def _float(value: Any) -> float | None:
    return None if value is None else float(value)


def _int(value: Any) -> int | None:
    return None if value is None else int(value)


def _check_supported(
    module: torch.nn.Module, layer: str, tensors: frozenset[str]
) -> None:
    """Refuse the layer variants whose recorded tensors would not describe
    the computation.

    Raises:
        NotImplementedError: for expert, data, sequence or context
            parallelism, expert load balancing, LoRA, routed input or output
            transforms, a gate inside the layer when ``router_logits`` is
            selected, or shared experts when ``output`` is selected.
    """
    config = _moe_config(module)
    parallel = config.moe_parallel_config
    if getattr(parallel, "use_ep", False) or int(getattr(parallel, "ep_size", 1)) > 1:
        raise _unsupported(layer, "expert parallelism is not supported")
    if int(getattr(parallel, "dp_size", 1)) > 1:
        raise _unsupported(layer, "data parallelism is not supported")
    if int(getattr(parallel, "pcp_size", 1)) > 1:
        raise _unsupported(layer, "context parallelism is not supported")
    if getattr(parallel, "is_sequence_parallel", False):
        raise _unsupported(layer, "sequence-parallel MoE is not supported")
    if getattr(parallel, "enable_eplb", False):
        raise _unsupported(layer, "expert load balancing is not supported")
    if getattr(config, "is_lora_enabled", False):
        raise _unsupported(layer, "MoE LoRA is not supported")
    if (
        getattr(module, "routed_input_transform", None) is not None
        or getattr(module, "routed_output_transform", None) is not None
    ):
        raise _unsupported(layer, "routed input or output transforms are not supported")
    if "router_logits" in tensors and getattr(module, "gate", None) is not None:
        raise _unsupported(
            layer,
            "the layer computes its router logits itself, so router_logits "
            f"cannot be recorded; set {_MOE_TENSORS_ENV} to a list without it "
            "(its default includes it)",
        )
    if "output" in tensors and getattr(module, "shared_experts", None) is not None:
        raise _unsupported(
            layer,
            "its output includes the shared experts' output, so output cannot "
            f"be recorded; set {_MOE_TENSORS_ENV} to a list without it (its "
            "default includes it)",
        )


def _output_reduced(module: torch.nn.Module) -> bool:
    """Whether the layer returns its output summed over the tensor-parallel
    ranks, so every rank returns the same output."""
    config = _moe_config(module)
    return int(config.moe_parallel_config.tp_size) == 1 or not bool(
        getattr(config, "skip_final_all_reduce", False)
    )


def _layout(module: torch.nn.Module, layer: str) -> _MoELayout:
    """What a reader needs to pair the layer's tensors with the expert
    weights of the checkpoint and to redo its routing."""
    config = _moe_config(module)
    parallel = config.moe_parallel_config
    routed = _routed_experts(module)
    fused_experts = _fused_experts(module)
    hook = getattr(fused_experts, "post_gemm2_hook", None)
    quant_method = getattr(routed, "quant_method", None)
    custom_routing = getattr(routed, "custom_routing_function", None)
    bias = getattr(routed, "e_score_correction_bias", None)
    fields: dict[str, object] = {
        "num_experts": int(config.num_experts),
        "num_local_experts": int(config.num_local_experts),
        "top_k": int(config.experts_per_token),
        "hidden_size": int(config.hidden_dim),
        "intermediate_size": int(config.intermediate_size),
        "intermediate_size_per_partition": int(config.intermediate_size_per_partition),
        "activation": _name(config.activation),
        "gated": bool(getattr(config, "is_act_and_mul", True)),
        "has_bias": bool(getattr(config, "has_bias", False)),
        "dtype": _dtype_name(config.in_dtype),
        "weight_dtype": _dtype(getattr(routed, "params_dtype", None)),
        "tp_rank": int(parallel.tp_rank),
        "tp_size": int(parallel.tp_size),
        "ep_rank": int(getattr(parallel, "ep_rank", 0)),
        "ep_size": int(getattr(parallel, "ep_size", 1)),
        "routing_method": getattr(config.routing_method, "name", None),
        "scoring_func": _name(getattr(routed, "scoring_func", None)),
        "renormalize": bool(getattr(routed, "renormalize", False)),
        "use_grouped_topk": bool(getattr(routed, "use_grouped_topk", False)),
        "num_expert_group": _int(getattr(routed, "num_expert_group", None)),
        "topk_group": _int(getattr(routed, "topk_group", None)),
        "has_e_score_correction_bias": bias is not None,
        "routed_scaling_factor": _float(getattr(routed, "routed_scaling_factor", None)),
        "output_scaling_factor": _float(getattr(module, "routed_scaling_factor", 1.0)),
        "apply_router_weight_on_input": bool(
            getattr(routed, "apply_router_weight_on_input", False)
        ),
        "custom_routing_function": (
            None if custom_routing is None else getattr(custom_routing, "__name__", "?")
        ),
        "hash_routing": bool(getattr(config, "has_hash_routing", False)),
        "swiglu_limit": _float(getattr(config, "swiglu_limit", None)),
        "swiglu_alpha": _float(getattr(config, "swiglu_alpha", None)),
        "swiglu_beta": _float(getattr(config, "swiglu_beta", None)),
        "monolithic": _is_monolithic(module),
        "quant_method": None if quant_method is None else type(quant_method).__name__,
        "experts_kernel": None
        if fused_experts is None
        else type(fused_experts).__name__,
        "post_gemm2_hook": None if hook is None else type(hook).__name__,
        "output_reduced": _output_reduced(module),
    }
    return _MoELayout(
        layer=layer,
        fields=fields,
        bias=bias if isinstance(bias, torch.Tensor) else None,
    )


class _MoELayer:
    """One selected MoE layer: its hooks, and what they captured during the
    layer's current call.

    vLLM calls ``capture`` with the routed expert ids; the wrapped
    ``post_gemm2_hook`` and experts call keep the routing weights and the
    GEMM2 rows. All are kept only for calls at selected steps, from the
    layer's forward pre-hook to its forward hook, which writes them.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        layer: str,
        tensors: frozenset[str],
        tracker: _StepTracker,
        writer: _Writer,
    ) -> None:
        self._module = module
        self._layer = layer
        self._tracker = tracker
        self._writer = writer
        fused_experts = _fused_experts(module)
        has_hook = getattr(fused_experts, "post_gemm2_hook", None) is not None
        if "gemm2" in tensors and not has_hook:
            raise _unsupported(
                layer,
                "its experts kernel hands no unweighted GEMM2 rows to a "
                f"post_gemm2_hook; remove gemm2 from {_MOE_TENSORS_ENV}",
            )
        weights_source: str | None = None
        if "topk_weights" in tensors:
            if has_hook:
                weights_source = "post_gemm2_hook"
            elif not _is_monolithic(module):
                weights_source = "experts_call"
            else:
                logger.warning(
                    "tensor_dump cannot record the top-k weights of MoE layer "
                    "%s: its experts kernel routes and reduces the tokens "
                    "itself; its records have no topk_weights",
                    layer,
                )
                tensors = tensors - {"topk_weights"}
        parallel = _moe_config(module).moe_parallel_config
        replicated = _TOKEN_TENSORS
        if _output_reduced(module):
            replicated = replicated | {"output"}
        replicated = replicated & tensors
        self.layout: _MoELayout = _layout(module, layer)
        tp_size = int(parallel.tp_size)
        # Fields of every record, which depend on the selected tensors.
        self._record_fields: dict[str, object] = {
            "replicated_on": 0 if tp_size > 1 else None,
            "replicated_tensors": sorted(replicated),
            "topk_weights_source": weights_source,
        }
        self._tensors: frozenset[str] = (
            tensors if int(parallel.tp_rank) == 0 else tensors - replicated
        )
        self._weights_source = weights_source
        self._handles: list[RemovableHandle] = []
        self._original_hook: Callable[..., None] | None = None
        self._wrapped_hook: Callable[..., None] | None = None
        self._original_forward_modular: Callable[..., Any] | None = None
        self._bound_capture = False
        self._in_call = False
        self._step: _StepInfo | None = None
        self._captured: dict[str, torch.Tensor] = {}
        if "topk_ids" in self._tensors:
            self._bind_capture()
        if "gemm2" in self._tensors or weights_source == "post_gemm2_hook":
            self._wrap_hook()
        if weights_source == "experts_call" and "topk_weights" in self._tensors:
            self._wrap_forward_modular()
        self._handles.append(
            module.register_forward_pre_hook(self._before, with_kwargs=True)
        )
        self._handles.append(
            module.register_forward_hook(self._after, with_kwargs=True)
        )

    # vLLM's routed-experts capture contract.

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        """Keep the call's routed expert ids, ``[tokens, top_k]``."""
        rows = self._rows("topk_ids")
        if rows is not None:
            self._keep("topk_ids", topk_ids, rows)

    def remove(self) -> None:
        """Remove the hooks and restore what the layer had before."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        fused_experts = _fused_experts(self._module)
        if (
            fused_experts is not None
            and self._wrapped_hook is not None
            and fused_experts.post_gemm2_hook is self._wrapped_hook
        ):
            fused_experts.post_gemm2_hook = self._original_hook
        if self._original_forward_modular is not None:
            del _routed_experts(self._module).forward_modular
        if self._bound_capture:
            routed = _routed_experts(self._module)
            if getattr(routed, "routing_sink", None) is not None:
                routed.routing_sink = None
            router = getattr(self._module, "router", None)
            if router is not None and getattr(router, "capture_fn", None) is not None:
                router.set_capture_fn(None)
        self._captured.clear()

    def _bind_capture(self) -> None:
        module = self._module
        if _is_monolithic(module):
            bound = getattr(_routed_experts(module), "routing_sink", None) is not None
        else:
            bound = getattr(module.router, "capture_fn", None) is not None
        if bound:
            raise _unsupported(
                self._layer,
                "another routed-experts capture is bound to it, as vLLM's "
                "--enable-return-routed-experts binds one; remove topk_ids from "
                f"{_MOE_TENSORS_ENV} or turn that off",
            )
        try:
            _bind_routing_capture(module, self)
        except (ValueError, AssertionError) as error:
            # vLLM asserts that the layer name holds exactly one layer index.
            raise _unsupported(
                self._layer,
                f"vLLM's routed-experts capture cannot be bound to it ({error}); "
                f"remove topk_ids from {_MOE_TENSORS_ENV}",
            ) from error
        self._bound_capture = True

    def _wrap_hook(self) -> None:
        """Wrap the kernel's current ``post_gemm2_hook``, unless it is ours.

        Raises:
            RuntimeError: if the kernel has no hook any more.
        """
        fused_experts = _fused_experts(self._module)
        current = getattr(fused_experts, "post_gemm2_hook", None)
        if fused_experts is None or current is None:
            raise RuntimeError(
                f"the experts kernel of MoE layer {self._layer!r} has no "
                "post_gemm2_hook any more; the dump needs it for "
                f"{sorted(self._tensors & {'gemm2', 'topk_weights'})}"
            )
        if current is self._wrapped_hook:
            return
        self._original_hook = current

        def post_gemm2_hook(unfinalized: Any, output: torch.Tensor) -> None:
            self._gemm2_rows(unfinalized)
            current(unfinalized, output)

        self._wrapped_hook = post_gemm2_hook
        # The kernel calls the instance attribute as a plain function.
        fused_experts.post_gemm2_hook = post_gemm2_hook

    def _wrap_forward_modular(self) -> None:
        routed = _routed_experts(self._module)
        original = routed.forward_modular
        self._original_forward_modular = original

        def forward_modular(*args: Any, **kwargs: Any) -> Any:
            rows = self._rows("topk_weights")
            weights = kwargs.get("topk_weights")
            if rows is not None and isinstance(weights, torch.Tensor):
                self._keep("topk_weights", weights, rows)
            return original(*args, **kwargs)

        routed.forward_modular = forward_modular

    def _rows(self, name: str) -> int | None:
        """The rows to keep of a tensor captured inside the layer, or ``None``
        when nothing is kept.

        Raises:
            RuntimeError: if the experts run outside the layer's forward at a
                selected step, so the forward hook would never write them.
        """
        if name not in self._tensors:
            return None
        if not self._in_call:
            step = self._tracker.current()
            if step is not None and step.selected:
                raise RuntimeError(
                    f"the experts of MoE layer {self._layer!r} ran without the "
                    "layer's forward, which the dump hooks, as a model's fast "
                    "path that calls the experts kernel directly (for example "
                    "for small batches of quantized inputs) does; such paths "
                    f"are not supported. Exclude the layer from {_MOE_ENV}"
                )
            return None
        step = self._step
        return None if step is None else step.num_actual_tokens

    def _keep(
        self,
        name: str,
        tensor: torch.Tensor,
        rows: int,
        copy: Callable[[torch.Tensor], torch.Tensor] = torch.clone,
    ) -> None:
        """Keep a copy of the first ``rows`` rows of ``tensor``, made by
        ``copy``."""
        if tensor.shape[0] < rows:
            raise RuntimeError(
                f"MoE layer {self._layer!r} produced {tensor.shape[0]} rows of "
                f"{name}, fewer than the step's {rows} tokens"
            )
        if name in self._captured:
            raise RuntimeError(
                f"MoE layer {self._layer!r} produced {name} twice in one call"
            )
        self._captured[name] = copy(tensor[:rows])

    def _gemm2_rows(self, unfinalized: Any) -> None:
        """Keep the route-major GEMM2 rows and the routing weights of an
        unfinalized kernel output."""
        rows = self._rows("gemm2")
        weight_rows = (
            self._rows("topk_weights")
            if self._weights_source == "post_gemm2_hook"
            else None
        )
        if weight_rows is not None:
            self._keep("topk_weights", unfinalized.expert_weights, weight_rows)
        if rows is None:
            return
        gemm2 = unfinalized.gemm2_permuted

        def gather(permuted: torch.Tensor) -> torch.Tensor:
            # Every slot names a row: expert parallelism, under which a slot
            # can be -1, is refused when the layer is hooked.
            return _gather_to_host(gemm2, permuted)

        self._keep("gemm2", unfinalized.expanded_idx_to_permuted_idx, rows, gather)

    def _before(
        self, module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        if torch.compiler.is_compiling():
            raise RuntimeError("tensor_dump requires eager execution (--enforce-eager)")
        self._captured.clear()
        step = self._tracker.current()
        self._step = step if step is not None and step.selected else None
        if self._wrapped_hook is not None:
            # A weight reload may rebuild the kernel and attach a new hook.
            self._wrap_hook()
        self._in_call = True

    def _after(
        self,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        self._in_call = False
        step = self._step
        self._step = None
        captured = self._captured
        self._captured = {}
        if step is None:
            return
        rows = step.num_actual_tokens
        given = {
            "input": _argument(args, kwargs, 0, "hidden_states"),
            "router_logits": _argument(args, kwargs, 1, "router_logits"),
            "output": output,
        }
        tensors: dict[str, torch.Tensor] = {}
        for name in sorted(self._tensors):
            tensor = captured.get(name, given.get(name))
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(
                    f"MoE layer {self._layer!r} produced no {name} at step {step.step}"
                )
            if tensor.shape[0] < rows:
                raise RuntimeError(
                    f"MoE layer {self._layer!r} has {tensor.shape[0]} rows of "
                    f"{name}, fewer than the step's {rows} tokens"
                )
            tensors[name] = tensor[:rows]
        self._writer.moe(
            step, self.layout, tensors, self._record_fields, host_copies=_HOST_COPIES
        )


def _gather_to_host(rows: torch.Tensor, permuted: torch.Tensor) -> torch.Tensor:
    """``rows[permuted]`` as ``[tokens, top_k, hidden]`` on the host, gathered
    on the device at most ``_GEMM2_CHUNK_BYTES`` at a time.

    Args:
        rows: ``[permuted rows, hidden]``, the kernel's permuted GEMM2 rows.
        permuted: ``[tokens, top_k]``, the row of each route.
    """
    tokens, top_k = permuted.shape
    hidden = rows.shape[-1]
    host = torch.empty((tokens, top_k, hidden), dtype=rows.dtype, device="cpu")
    token_bytes = top_k * hidden * rows.element_size()
    chunk = max(1, _GEMM2_CHUNK_BYTES // token_bytes)
    for start in range(0, tokens, chunk):
        index = permuted[start : start + chunk].reshape(-1).long()
        host[start : start + chunk] = rows.index_select(0, index).view(
            -1, top_k, hidden
        )
    return host


class _MoEDump:
    """Hooks the selected MoE layers and writes their tensors.

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
        self._runner_type: type[Any] | None = None
        self._layers: dict[int, _MoELayer] = {}
        self._warned_none = False

    def check_hooked(self, step: int) -> None:
        """Warn once, from step 1 on, by when every module that runs has been
        walked, if no MoE layer matched the pattern."""
        if step > 0 and not self._layers and not self._warned_none:
            self._warned_none = True
            logger.warning(
                "%s=%r matched no vLLM MoERunner layer by step %d; no MoE "
                "tensor is recorded. Experts that are not MoERunner layers are "
                "not supported",
                _MOE_ENV,
                self._pattern.pattern,
                step,
            )

    def visit(self, module: torch.nn.Module) -> None:
        """Hook ``module`` if it is a selected MoE layer.

        Raises:
            NotImplementedError: for a selected layer that cannot be recorded.
        """
        runner_type = self._runner_type
        if runner_type is None:
            runner_type = _moe_runner_class()
            self._runner_type = runner_type
        if not isinstance(module, runner_type) or id(module) in self._layers:
            return
        layer = getattr(module, "layer_name", None)
        if not isinstance(layer, str) or not self._pattern.match(layer):
            return
        _check_layer_name(layer)
        _check_supported(module, layer, self._tensors)
        self._layers[id(module)] = _MoELayer(
            module, layer, self._tensors, self._tracker, self._writer
        )

    def remove(self) -> None:
        """Remove every hook."""
        for layer in self._layers.values():
            layer.remove()
        self._layers.clear()
