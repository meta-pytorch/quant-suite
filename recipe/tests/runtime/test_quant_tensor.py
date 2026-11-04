# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import importlib.util
import math
import unittest

import torch
from parameterized import parameterized
from quant_suite_recipe.api.implementation import QuantizedTensor
from quant_suite_recipe.definition import QuantRecipe
from quant_suite_recipe.implementation.quantize_tensor import (
    dequantize_tensor,
    fp4_to_float,
)
from quant_suite_recipe.runtime import QuantRecipeRuntime

_HAS_MSLK: bool = importlib.util.find_spec("mslk") is not None
if _HAS_MSLK:
    from mslk.quantize.triton.fp4_quantize import _from_blocked


_BUILTIN_RUNTIME = QuantRecipeRuntime()


def _linear_quant_recipe(
    activation: dict[str, object], weight: dict[str, object]
) -> QuantRecipe:
    return _BUILTIN_RUNTIME.deserialize(
        {"linear_recipe": {".*": {"activation": activation, "weight": weight}}}
    )


def _is_blackwell_or_newer() -> bool:
    """Return True if the current CUDA device is Blackwell (sm_100) or newer."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 10


def _calculate_sqnr(tensor: torch.Tensor, reference: torch.Tensor) -> float:
    reference = reference.to(torch.double)
    error = reference - tensor.to(torch.double)
    error_power = torch.mean(error * error)
    if error_power == 0:
        return math.inf
    signal_power = torch.mean(reference * reference)
    return 10.0 * torch.log10(signal_power / error_power).item()


class QuantizedTensorDictAccessTest(unittest.TestCase):
    """Dict-like access to QuantizedTensor (no GPU required)."""

    def test_getitem_metadata_returns_dict(self) -> None:
        qt = QuantizedTensor(metadata={"offset": 1})
        # "metadata" is a typed field, so it returns the dict itself, not a
        # nested self.metadata["metadata"] lookup.
        self.assertIs(qt["metadata"], qt.metadata)
        self.assertEqual(qt["metadata"], {"offset": 1})
        self.assertEqual(qt["offset"], 1)

    def test_setitem_metadata_replaces_dict(self) -> None:
        qt = QuantizedTensor(metadata={"offset": 1})
        qt["metadata"] = {"offset": 2}
        self.assertEqual(qt.metadata, {"offset": 2})

    def test_arbitrary_key_round_trips_through_metadata_dict(self) -> None:
        qt = QuantizedTensor()
        qt["offset"] = 7
        self.assertEqual(qt["offset"], 7)
        self.assertEqual(qt.get("offset"), 7)
        self.assertEqual(qt.metadata, {"offset": 7})
        self.assertEqual(qt.get("missing", 42), 42)
        self.assertIsNone(qt["missing"])


@unittest.skipUnless(_HAS_MSLK, "mslk is not available")
@unittest.skipUnless(
    _is_blackwell_or_newer(),
    "native FP4 (e2m1) cvt requires a Blackwell (sm_100+) GPU; "
    "the triton MX4/NVFP4 kernels do not assemble on Hopper/H100 (sm_90a)",
)
class QuantTensorTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.device = "cuda"

    @parameterized.expand(
        [
            ("1x128", (1, 128)),
            ("128x2048", (128, 2048)),
            ("4096x10240", (4096, 10240)),
        ]
    )
    def test_mx4_32_native_vs_fake_quant(
        self, name: str, shape: tuple[int, int]
    ) -> None:
        """Test native mx4-32 and fake quant mx4-32 are close, and also both are close to bf16"""
        M, N = shape

        torch.manual_seed(0)
        bf16_tensor = torch.randn(M, N, dtype=torch.bfloat16, device=self.device)

        # Create a quant recipe for fake quantization using mx4
        fake_mx4_config = _linear_quant_recipe(
            activation={
                "type": "fake",
                "quant_config": {
                    "elem_dtype": "e2m1",
                    "elem_round_mode": "nearest",
                    "block_size": "1x32",
                    "scale_dtype": "e8m0",
                    "scale_norm_mode": "pow2max",
                    "scale_round_mode": "oas",
                },
            },
            weight={
                "type": "fake",
                "quant_config": {
                    "elem_dtype": "e2m1",
                    "elem_round_mode": "nearest",
                    "block_size": "1x32",
                    "scale_dtype": "e8m0",
                    "scale_norm_mode": "pow2max",
                    "scale_round_mode": "oas",
                },
            },
        )
        fake_mx4_tensor_recipe = fake_mx4_config.get_linear_recipe("test").activation[0]

        # Quantize and dequantize using fake mx4
        fake_mx4_quantized = _BUILTIN_RUNTIME.quantize_tensor(
            bf16_tensor, "test_tensor", fake_mx4_tensor_recipe
        )
        fake_mx4_dequant = dequantize_tensor(fake_mx4_quantized)

        # Create a quant recipe for native mx4 quantization
        native_mx4_config = _linear_quant_recipe(
            activation={
                "type": "native",
                "precision": "mx4-32",
                "scale_round_mode": "even",
            },
            weight={
                "type": "native",
                "precision": "mx4-32",
                "scale_round_mode": "even",
            },
        )
        native_mx4_tensor_recipe = native_mx4_config.get_linear_recipe(
            "test"
        ).activation[0]

        # Quantize and dequantize using native mx4
        native_mx4_quantized = _BUILTIN_RUNTIME.quantize_tensor(
            bf16_tensor, "test_tensor", native_mx4_tensor_recipe
        )
        native_mx4_dequant = dequantize_tensor(
            native_mx4_quantized,
            orig_shape=bf16_tensor.shape,
            orig_dtype=bf16_tensor.dtype,
        )

        # Compare the results
        self.assertTrue(torch.equal(fake_mx4_dequant, native_mx4_dequant))

        # calculate SQNR between fake/native mx4 and bf16
        sqnr_fake_bf16 = _calculate_sqnr(fake_mx4_dequant, bf16_tensor)
        sqnr_native_bf16 = _calculate_sqnr(native_mx4_dequant, bf16_tensor)
        self.assertGreaterEqual(sqnr_fake_bf16, 15.0)
        self.assertGreaterEqual(sqnr_native_bf16, 15.0)

    @parameterized.expand(
        [
            ("1x128", (1, 128)),
            ("128x2048", (128, 2048)),
            ("4096x10240", (4096, 10240)),
        ]
    )
    def test_mx4_16_native_vs_fake_quant(
        self, name: str, shape: tuple[int, int]
    ) -> None:
        """Test native mx4-16 and fake quant mx4-16 are close, and also both are close to bf16"""
        M, N = shape

        torch.manual_seed(0)
        bf16_tensor = torch.randn(M, N, dtype=torch.bfloat16, device=self.device)

        # Create a quant recipe for fake quantization using mx4
        fake_mx4_config = _linear_quant_recipe(
            activation={
                "type": "fake",
                "quant_config": {
                    "elem_dtype": "e2m1",
                    "elem_round_mode": "nearest",
                    "block_size": "1x16",
                    "scale_dtype": "e8m0",
                    "scale_norm_mode": "pow2max",
                    "scale_round_mode": "oas",
                },
            },
            weight={
                "type": "fake",
                "quant_config": {
                    "elem_dtype": "e2m1",
                    "elem_round_mode": "nearest",
                    "block_size": "1x16",
                    "scale_dtype": "e8m0",
                    "scale_norm_mode": "pow2max",
                    "scale_round_mode": "oas",
                },
            },
        )
        fake_mx4_tensor_recipe = fake_mx4_config.get_linear_recipe("test").activation[0]

        # Quantize and dequantize using fake mx4
        fake_mx4_quantized = _BUILTIN_RUNTIME.quantize_tensor(
            bf16_tensor, "test_tensor", fake_mx4_tensor_recipe
        )
        fake_mx4_dequant = dequantize_tensor(fake_mx4_quantized)

        # Create a quant recipe for native mx4 quantization
        native_mx4_config = _linear_quant_recipe(
            activation={
                "type": "native",
                "precision": "mx4-16",
                "scale_round_mode": "even",
            },
            weight={
                "type": "native",
                "precision": "mx4-16",
                "scale_round_mode": "even",
            },
        )
        native_mx4_tensor_recipe = native_mx4_config.get_linear_recipe(
            "test"
        ).activation[0]

        # Quantize and dequantize using native mx4
        native_mx4_quantized = _BUILTIN_RUNTIME.quantize_tensor(
            bf16_tensor, "test_tensor", native_mx4_tensor_recipe
        )
        native_mx4_dequant = dequantize_tensor(
            native_mx4_quantized,
            orig_shape=bf16_tensor.shape,
            orig_dtype=bf16_tensor.dtype,
        )

        # Compare the results of fake and native mx4, should gaurantee bit-wise match
        self.assertTrue(torch.equal(fake_mx4_dequant, native_mx4_dequant))

        # calculate SQNR between fake/native mx4 and bf16
        sqnr_fake_bf16 = _calculate_sqnr(fake_mx4_dequant, bf16_tensor)
        sqnr_native_bf16 = _calculate_sqnr(native_mx4_dequant, bf16_tensor)
        self.assertGreaterEqual(
            sqnr_fake_bf16,
            17.0,
            f"sqnr(fake_mx4_16, bf16)={sqnr_fake_bf16} should be >= 17.0",
        )
        self.assertGreaterEqual(
            sqnr_native_bf16,
            17.0,
            f"sqnr(native_mx4_16, bf16)={sqnr_native_bf16} should be >= 17.0",
        )

    @parameterized.expand(
        [
            ("1x128", (1, 128)),
            ("128x2048", (128, 2048)),
            ("4096x10240", (4096, 10240)),
        ]
    )
    def test_nvfp4_native_vs_fake_quant(
        self, name: str, shape: tuple[int, int]
    ) -> None:
        """Test native nvfp4 and fake quant nvfp4 are close, and also both are close to bf16"""
        M, N = shape

        torch.manual_seed(0)
        bf16_tensor = torch.randn(M, N, dtype=torch.bfloat16, device=self.device)

        # Create a quant recipe for fake quantization using nvfp4
        fake_nvfp4_config = _linear_quant_recipe(
            activation={
                "type": "fake",
                "quant_config": {
                    "elem_dtype": "e2m1",
                    "elem_round_mode": "nearest",
                    "block_size": "1x16",
                    "scale_dtype": "fp8",
                    "scale_norm_mode": "amax",
                    "scale_round_mode": "none",
                    "global_scale_granularity": "tensorwise",
                    "global_scale_dtype": "fp32",
                },
            },
            weight={
                "type": "fake",
                "quant_config": {
                    "elem_dtype": "e2m1",
                    "elem_round_mode": "nearest",
                    "block_size": "1x16",
                    "scale_dtype": "fp8",
                    "scale_norm_mode": "amax",
                    "scale_round_mode": "none",
                    "global_scale_granularity": "tensorwise",
                    "global_scale_dtype": "fp32",
                },
            },
        )
        fake_nvfp4_tensor_recipe = fake_nvfp4_config.get_linear_recipe(
            "test"
        ).activation[0]

        # Quantize and dequantize using fake nvfp4
        fake_nvfp4_quantized = _BUILTIN_RUNTIME.quantize_tensor(
            bf16_tensor, "test_tensor", fake_nvfp4_tensor_recipe
        )
        fake_nvfp4_dequant = dequantize_tensor(fake_nvfp4_quantized)

        # Create a quant recipe for native nvfp4 quantization
        native_nvfp4_config = _linear_quant_recipe(
            activation={
                "type": "native",
                "precision": "nvfp4",
                "scale_round_mode": "nearest",
            },
            weight={
                "type": "native",
                "precision": "nvfp4",
                "scale_round_mode": "nearest",
            },
        )
        native_nvfp4_tensor_recipe = native_nvfp4_config.get_linear_recipe(
            "test"
        ).activation[0]

        # Quantize and dequantize using native nvfp4
        native_nvfp4_quantized = _BUILTIN_RUNTIME.quantize_tensor(
            bf16_tensor,
            "test_tensor",
            native_nvfp4_tensor_recipe,
        )
        native_nvfp4_dequant = dequantize_tensor(
            native_nvfp4_quantized,
            orig_shape=bf16_tensor.shape,
            orig_dtype=bf16_tensor.dtype,
        )

        sqnr_fake_native = _calculate_sqnr(fake_nvfp4_dequant, native_nvfp4_dequant)
        self.assertGreater(
            sqnr_fake_native, 40.0, f"sqnr_fake_native={sqnr_fake_native} < 40.0"
        )

    def test_nvfp4_native_tiny_tensor(self) -> None:
        """A tensor whose amax is below 2688 / FLT_MAX (about 7.9e-36) gets the
        global scale FLT_MAX and quantizes like any other, not to zeros."""
        torch.manual_seed(0)
        tensor = (torch.randn(128, 256) * 1e-37).to(torch.bfloat16).to(self.device)
        native_spec = {
            "type": "native",
            "precision": "nvfp4",
            "scale_round_mode": "nearest",
        }
        recipe = (
            _linear_quant_recipe(activation=native_spec, weight=native_spec)
            .get_linear_recipe("test")
            .activation[0]
        )
        quantized = _BUILTIN_RUNTIME.quantize_tensor(tensor, "tensor", recipe)
        assert quantized.global_scale is not None
        self.assertEqual(quantized.global_scale.item(), torch.finfo(torch.float32).max)
        dequantized = dequantize_tensor(
            quantized, orig_shape=tensor.shape, orig_dtype=torch.float32
        )
        sqnr = _calculate_sqnr(dequantized, tensor.float())
        self.assertGreater(sqnr, 15.0, f"sqnr={sqnr}")

    def test_nvfp4_native_all_zero_tensor(self) -> None:
        """An all-zero tensor gets the global scale 1.0, finite block scales
        and dequantizes to zeros."""
        tensor = torch.zeros(128, 256, dtype=torch.bfloat16, device=self.device)
        native_spec = {
            "type": "native",
            "precision": "nvfp4",
            "scale_round_mode": "nearest",
        }
        recipe = (
            _linear_quant_recipe(activation=native_spec, weight=native_spec)
            .get_linear_recipe("test")
            .activation[0]
        )
        quantized = _BUILTIN_RUNTIME.quantize_tensor(tensor, "tensor", recipe)
        assert quantized.global_scale is not None
        assert quantized.scale is not None
        self.assertEqual(quantized.global_scale.item(), 1.0)
        block_scales = quantized.scale.view(torch.float8_e4m3fn).float()
        self.assertTrue(torch.isfinite(block_scales).all())
        dequantized = dequantize_tensor(
            quantized, orig_shape=tensor.shape, orig_dtype=torch.float32
        )
        self.assertTrue(torch.isfinite(dequantized).all())
        self.assertTrue(torch.equal(dequantized, torch.zeros_like(dequantized)))

    @parameterized.expand(
        [
            ("1x128", (1, 128)),
            ("128x2048", (128, 2048)),
            ("4096x10240", (4096, 10240)),
        ]
    )
    def test_nvfp4_native_vs_fake_quant_quant_info(
        self, name: str, shape: tuple[int, int]
    ) -> None:
        """Test native nvfp4 and fake quant nvfp4 quant info match"""
        M, N = shape

        torch.manual_seed(0)
        bf16_tensor = torch.randn(M, N, dtype=torch.bfloat16, device=self.device)

        # Create a quant recipe for fake quantization using nvfp4
        fake_nvfp4_config = _linear_quant_recipe(
            activation={
                "type": "fake",
                "quant_config": {
                    "elem_dtype": "e2m1",
                    "elem_round_mode": "nearest",
                    "block_size": "1x16",
                    "scale_dtype": "fp8",
                    "scale_norm_mode": "amax",
                    "scale_round_mode": "none",
                    "global_scale_granularity": "tensorwise",
                    "global_scale_dtype": "fp32",
                },
            },
            weight={
                "type": "fake",
                "quant_config": {
                    "elem_dtype": "e2m1",
                    "elem_round_mode": "nearest",
                    "block_size": "1x16",
                    "scale_dtype": "fp8",
                    "scale_norm_mode": "amax",
                    "scale_round_mode": "none",
                    "global_scale_granularity": "tensorwise",
                    "global_scale_dtype": "fp32",
                },
            },
        )
        fake_nvfp4_tensor_recipe = fake_nvfp4_config.get_linear_recipe(
            "test"
        ).activation[0]

        # Quantize and dequantize using fake nvfp4
        fake_nvfp4_quantized = _BUILTIN_RUNTIME.quantize_tensor(
            bf16_tensor,
            "test_tensor",
            fake_nvfp4_tensor_recipe,
            return_quant_info=True,
        )
        fake_local_scale = fake_nvfp4_quantized.scale
        fake_global_scale = fake_nvfp4_quantized.global_scale
        fake_quant_val = fake_nvfp4_quantized.quantized_val

        # Create a quant recipe for native nvfp4 quantization
        native_nvfp4_config = _linear_quant_recipe(
            activation={
                "type": "native",
                "precision": "nvfp4",
                "scale_round_mode": "nearest",
            },
            weight={
                "type": "native",
                "precision": "nvfp4",
                "scale_round_mode": "nearest",
            },
        )
        native_nvfp4_tensor_recipe = native_nvfp4_config.get_linear_recipe(
            "test"
        ).activation[0]

        # Quantize and dequantize using native nvfp4
        native_nvfp4_quantized = _BUILTIN_RUNTIME.quantize_tensor(
            bf16_tensor,
            "test_tensor",
            native_nvfp4_tensor_recipe,
        )
        group_size = 16
        num_groups = math.ceil(N / group_size)
        # Convert blocked scale format back to (M, num_groups) layout.
        native_local_scale = (
            _from_blocked(native_nvfp4_quantized.scale, (M, num_groups))
            .flatten()
            .view(torch.float8_e4m3fn)
            .to(torch.float)
        )
        native_global_scale = native_nvfp4_quantized.global_scale
        native_quant_val = fp4_to_float(native_nvfp4_quantized.tensor).reshape(M, N)

        # Compare the results of fake and native nvfp4
        # Check that mismatch percentage is less than 1%
        num_unmatched = (fake_quant_val != native_quant_val).sum().item()
        mismatch_frac = num_unmatched / (M * N)
        self.assertLess(
            mismatch_frac,
            0.01,
            f"Mismatch frac {mismatch_frac * 100:.2f}% >= 1% ({num_unmatched}/{M * N} unmatched)",
        )
        self.assertTrue(torch.allclose(fake_local_scale, native_local_scale))
        self.assertTrue(
            torch.allclose(fake_global_scale.to(torch.float), native_global_scale)
        )
