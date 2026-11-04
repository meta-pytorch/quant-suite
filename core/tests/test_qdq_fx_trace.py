# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""qdq() and fake_quantize(..., use_triton=False) trace under torch.fx.symbolic_trace.

Graph-lowering pipelines that trace a module with a torch.fx Tracer (and then
torch.export the result) need a qdq() call inside the module's forward to trace.
Without trace-safe checks, tracing fails with:

    symbolically traced variables cannot be used as inputs to control flow

The shape and dtype checks in qdq() take a trace-safe form only when they see a
torch.fx Proxy, so eager numerics and speed are unchanged. This file pins the
plain symbolic_trace contract, bitwise against eager.

The Triton backend (fake_quantize's default use_triton=True, qdq_triton_mega) does
not trace: its raw kernel launch cannot take Proxies. Those tests are expected
failures.
"""

from __future__ import annotations

import json
import unittest

import torch
from quant_suite_core import fake_quantize, QuantConfig
from quant_suite_core.qdq_pytorch import qdq
from quant_suite_core.qdq_triton import qdq_triton_mega

# An MXFP8 fake-quant config.
MXFP8_CONFIG: dict[str, str] = {
    "elem_dtype": "e4m3",
    "elem_round_mode": "nearest",
    "block_size": "1x32",
    "scale_dtype": "e8m0",
    "scale_norm_mode": "pow2max",
    "scale_round_mode": "oas_s",
}

# Four representative configs: MX e4m3 / e2m3 / e2m1 and int8.
REPRESENTATIVE_CONFIGS: dict[str, dict[str, str]] = {
    "mxfp8_e4m3_oas_s": MXFP8_CONFIG,
    "mxfp6_e2m3_oas": {**MXFP8_CONFIG, "elem_dtype": "e2m3", "scale_round_mode": "oas"},
    "mxfp4_e2m1_oas": {**MXFP8_CONFIG, "elem_dtype": "e2m1", "scale_round_mode": "oas"},
    "mxint8_oas": {**MXFP8_CONFIG, "elem_dtype": "int8", "scale_round_mode": "oas"},
}

# 2-D and 3-D activations, block-aligned along the last dim.
ACTIVATION_SHAPES: list[tuple[int, ...]] = [(4, 128), (2, 5, 96)]
# One traced graph serves every shape, including ones the trace never saw:
# 0-D, 1-D, 4-D, and both dims needing padding.
OTHER_SHAPES: list[tuple[int, ...]] = [(), (37,), (33, 100), (2, 3, 4, 64)]


def _devices() -> list[torch.device]:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda", torch.cuda.current_device()))
    return devices


def _bits(t: torch.Tensor) -> torch.Tensor:
    """Reinterpret as integers so NaN payloads and -0.0 compare exactly."""
    int_dtype = {
        torch.float32: torch.int32,
        torch.bfloat16: torch.int16,
        torch.float16: torch.int16,
    }[t.dtype]
    return t.contiguous().view(int_dtype)


class _QdqModule(torch.nn.Module):
    """Stand-in for a fake-quantized linear's forward."""

    def __init__(self, config: QuantConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return qdq(x, "activation", self.config)


class _FakeQuantEagerModule(torch.nn.Module):
    """Same forward as _QdqModule, through fake_quantize()'s eager backend."""

    def __init__(self, config: QuantConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fake_quantize(x, "activation", self.config, use_triton=False)


class QdqFxTraceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = QuantConfig.from_dict(MXFP8_CONFIG)
        self.module = _QdqModule(self.config)
        torch.manual_seed(0)
        self.x = torch.randn(4, 128)

    def assert_bitwise_equal(self, got: torch.Tensor, want: torch.Tensor) -> None:
        self.assertEqual(got.shape, want.shape)
        self.assertEqual(got.dtype, want.dtype)
        self.assertTrue(torch.equal(_bits(got), _bits(want)))

    def _check_traced_matches_eager(self, module_cls: type[torch.nn.Module]) -> None:
        for name, cfg_dict in REPRESENTATIVE_CONFIGS.items():
            module = module_cls(QuantConfig.from_dict(cfg_dict))
            traced = torch.fx.symbolic_trace(module)
            for device in _devices():
                for dtype in (torch.float32, torch.bfloat16):
                    for shape in ACTIVATION_SHAPES + OTHER_SHAPES:
                        with self.subTest(
                            config=name, device=str(device), dtype=dtype, shape=shape
                        ):
                            x = (torch.randn(shape) * 3).to(dtype).to(device)
                            self.assert_bitwise_equal(traced(x), module(x))

    def test_qdq_is_fx_traceable(self) -> None:
        traced = torch.fx.symbolic_trace(self.module)
        self.assert_bitwise_equal(traced(self.x), self.module(self.x))

    def test_qdq_traced_matches_eager_bitwise(self) -> None:
        self._check_traced_matches_eager(_QdqModule)

    def test_fake_quant_eager_traced_matches_eager_bitwise(self) -> None:
        self._check_traced_matches_eager(_FakeQuantEagerModule)

    def test_qdq_is_exportable(self) -> None:
        """qdq() exports with torch.export.export."""
        exported = torch.export.export(self.module, (self.x,))
        self.assert_bitwise_equal(exported.module()(self.x), self.module(self.x))

    def test_traced_module_exports_with_dynamic_batch(self) -> None:
        for name, cfg_dict in REPRESENTATIVE_CONFIGS.items():
            with self.subTest(config=name):
                module = _QdqModule(QuantConfig.from_dict(cfg_dict))
                traced = torch.fx.symbolic_trace(module)
                x = torch.randn(4, 3, 96)
                batch = torch.export.Dim("batch", min=2, max=1024)
                exported = torch.export.export(
                    traced, (x,), dynamic_shapes={"x": {0: batch}}, strict=True
                )
                x_other = torch.randn(7, 3, 96)
                self.assert_bitwise_equal(exported.module()(x_other), module(x_other))


class _FakeQuantModule(torch.nn.Module):
    """fake_quantize() with its default backend, use_triton=True."""

    def __init__(self, config: QuantConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fake_quantize(x, "activation", self.config)


class _QdqTritonMegaModule(torch.nn.Module):
    """Calls the Triton backend directly, skipping fake_quantize's preamble."""

    def __init__(self, config: QuantConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return qdq_triton_mega(x, "activation", self.config)


def _fake_quant_leaf(x: torch.Tensor, config_str: str) -> torch.Tensor:
    return fake_quantize(x, "activation", QuantConfig.from_dict(json.loads(config_str)))


# Caller-side workaround, not library behavior: symbolic_trace records this
# function as one opaque call_function node instead of tracing into it. The
# config goes in as a JSON string: a QuantConfig argument would be recorded as
# BlockSize(...) and QuantConfig(...) constructor nodes.
torch.fx.wrap("_fake_quant_leaf")


class _FakeQuantLeafModule(torch.nn.Module):
    def __init__(self, config: QuantConfig) -> None:
        super().__init__()
        self.config_str: str = json.dumps(config.to_dict())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _fake_quant_leaf(x, self.config_str)


def _call_function_targets(gm: torch.fx.GraphModule) -> list[str]:
    return [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]


@unittest.skipIf(
    not torch.cuda.is_available(), "fake_quantize's Triton backend needs CUDA"
)
class FakeQuantTritonFxTraceTest(unittest.TestCase):
    """fake_quantize()'s default Triton backend (qdq_triton_mega)."""

    def setUp(self) -> None:
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.config = QuantConfig.from_dict(MXFP8_CONFIG)
        torch.manual_seed(0)
        self.x_cpu = torch.randn(4, 128)
        self.x_cuda = self.x_cpu.to(self.device)

    def test_fake_quant_cuda_bitwise_matches_qdq(self) -> None:
        got = fake_quantize(self.x_cuda, "activation", self.config)
        want_cuda = qdq(self.x_cuda, "activation", self.config)
        want_cpu = qdq(self.x_cpu, "activation", self.config)
        self.assertTrue(torch.equal(_bits(got), _bits(want_cuda)))
        self.assertTrue(torch.equal(_bits(got.cpu()), _bits(want_cpu)))

    # The Triton kernel launch cannot bind Proxy arguments (KeyError on the
    # Proxy). Making it traceable needs torch.library.triton_op + wrap_triton.
    @unittest.expectedFailure
    def test_fake_quant_is_fx_traceable(self) -> None:
        module = _FakeQuantModule(self.config)
        traced = torch.fx.symbolic_trace(module)
        self.assertTrue(torch.equal(traced(self.x_cuda), module(self.x_cuda)))

    @unittest.expectedFailure
    def test_qdq_triton_mega_is_fx_traceable(self) -> None:
        module = _QdqTritonMegaModule(self.config)
        traced = torch.fx.symbolic_trace(module)
        self.assertTrue(torch.equal(traced(self.x_cuda), module(self.x_cuda)))

    # Export traces into the raw Triton launch, which needs a real data pointer.
    @unittest.expectedFailure
    def test_fake_quant_is_exportable(self) -> None:
        module = _FakeQuantModule(self.config)
        exported = torch.export.export(module, (self.x_cuda,))
        self.assertTrue(
            torch.equal(exported.module()(self.x_cuda), module(self.x_cuda))
        )

    def test_fx_wrap_leaf_traces_as_one_node(self) -> None:
        module = _FakeQuantLeafModule(self.config)
        traced = torch.fx.symbolic_trace(module)
        targets = _call_function_targets(traced)
        self.assertEqual(len(targets), 1, targets)
        self.assertIn("_fake_quant_leaf", targets[0])
        self.assertTrue(
            torch.equal(_bits(traced(self.x_cuda)), _bits(module(self.x_cuda)))
        )
