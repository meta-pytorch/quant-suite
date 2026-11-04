# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
import unittest
from unittest import mock

import torch
from quant_suite_core import QuantConfig, QuantInfo
from quant_suite_core.native_presets import PrecisionType
from quant_suite_recipe.api import QuantizedTensor
from quant_suite_recipe.definition import (
    FakeQuantRecipe,
    NativeQuantRecipe,
    NoQuantRecipe,
)
from quant_suite_recipe.implementation.quantize_tensor import (
    fake_quantize_tensor,
    native_quantize_tensor,
    no_quantize_tensor,
)


class QuantizeTensorTest(unittest.TestCase):
    def test_no_quant_handler_preserves_tensor_without_precision(self) -> None:
        tensor = torch.randn(2, 4)

        result = no_quantize_tensor(
            tensor,
            "activation",
            NoQuantRecipe(),
            transpose=False,
            return_quant_info=False,
        )

        self.assertIs(result.tensor, tensor)
        self.assertIsNone(result.precision)
        self.assertNotIn("precision", result.metadata)

    def test_fake_quant_forwards_options_and_metadata(self) -> None:
        tensor = torch.randn(2, 16)
        recipe = FakeQuantRecipe(
            quant_config=QuantConfig.from_dict({}), use_triton=None
        )
        quant_info = QuantInfo(
            dequant_data=tensor + 1,
            quantized_val=torch.ones_like(tensor),
            local_unscale=torch.tensor([2.0]),
            global_scale=torch.tensor(3.0),
        )

        with mock.patch(
            "quant_suite_recipe.implementation.quantize_tensor.fake_quantize",
            return_value=quant_info,
        ) as fake_quant_mock:
            result = fake_quantize_tensor(
                tensor,
                "activation",
                recipe,
                True,
                True,
            )

        fake_quant_mock.assert_called_once_with(
            tensor=tensor,
            tensor_name="activation",
            config=recipe.quant_config,
            transpose=True,
            use_triton=True,
            return_quant_info=True,
        )
        self.assertIs(result.tensor, quant_info.dequant_data)
        self.assertIs(result.quantized_val, quant_info.quantized_val)
        self.assertIs(result.scale, quant_info.local_unscale)
        self.assertIs(result.global_scale, quant_info.global_scale)

    def test_handlers_accept_fp16(self) -> None:
        tensor = torch.randn(2, 16, dtype=torch.float16)
        recipe = FakeQuantRecipe(
            quant_config=QuantConfig.from_dict({}), use_triton=False
        )

        no_quant = no_quantize_tensor(
            tensor,
            "activation",
            NoQuantRecipe(),
            transpose=False,
            return_quant_info=False,
        )
        with mock.patch(
            "quant_suite_recipe.implementation.quantize_tensor.fake_quantize",
            return_value=tensor,
        ) as fake_quant_mock:
            fake_quant = fake_quantize_tensor(
                tensor,
                "activation",
                recipe,
                transpose=False,
                return_quant_info=False,
            )

        self.assertIs(no_quant.tensor, tensor)
        self.assertIs(fake_quant.tensor, tensor)
        fake_quant_mock.assert_called_once()

    def test_handlers_reject_unsupported_dtype(self) -> None:
        tensor = torch.ones(2, 16, dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "torch.int32"):
            no_quantize_tensor(
                tensor,
                "activation",
                NoQuantRecipe(),
                transpose=False,
                return_quant_info=False,
            )

    def test_native_mx4_rejects_fp32(self) -> None:
        tensor = torch.randn(2, 32, dtype=torch.float32)

        with self.assertRaisesRegex(NotImplementedError, "bfloat16 or float16"):
            native_quantize_tensor(
                tensor,
                "activation",
                NativeQuantRecipe(PrecisionType.MX4_32),
                transpose=False,
                return_quant_info=False,
            )

    def test_native_rejects_transpose(self) -> None:
        tensor = torch.randn(32, 32, dtype=torch.bfloat16)
        for precision in PrecisionType:
            with self.subTest(precision=precision):
                with self.assertRaisesRegex(
                    NotImplementedError, "does not support transpose=True"
                ):
                    native_quantize_tensor(
                        tensor,
                        "activation",
                        NativeQuantRecipe(precision),
                        transpose=True,
                        return_quant_info=False,
                    )

    def test_native_returns_scales_with_or_without_return_quant_info(self) -> None:
        tensor = torch.randn(2, 32, dtype=torch.bfloat16)
        xq, x_scale = torch.empty(2, 16, dtype=torch.uint8), torch.ones(2, 2)
        for return_quant_info in (False, True):
            with self.subTest(return_quant_info=return_quant_info):
                with mock.patch(
                    "quant_suite_recipe.implementation.quantize_tensor."
                    "triton_quantize_mx4_unpack",
                    return_value=(xq, x_scale),
                ):
                    result = native_quantize_tensor(
                        tensor,
                        "activation",
                        NativeQuantRecipe(PrecisionType.MX4_16),
                        transpose=False,
                        return_quant_info=return_quant_info,
                    )
                self.assertIs(result.tensor, xq)
                self.assertIs(result.scale, x_scale)
                self.assertEqual(result.precision, PrecisionType.MX4_16)

    def test_quantized_tensor_metadata_access_is_preserved(self) -> None:
        result = QuantizedTensor(metadata={"offset": 1})

        result["offset"] = 2

        self.assertEqual(2, result["offset"])
        self.assertEqual({"offset": 2}, result["metadata"])
        self.assertEqual("fallback", result.get("missing", "fallback"))


class NativeNvfp4GlobalScaleTest(unittest.TestCase):
    """The native NVFP4 global scale for an amax of 0, Inf, NaN or below
    2688 / FLT_MAX."""

    def _global_scales(self, tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """The global scale the recipe returns and the one it gives the kernel."""
        with mock.patch(
            "quant_suite_recipe.implementation.quantize_tensor.triton_quantize_nvfp4",
            return_value=(torch.empty(0), torch.empty(0)),
        ) as kernel_mock:
            result = native_quantize_tensor(
                tensor,
                "activation",
                NativeQuantRecipe(precision=PrecisionType.NVFP4),
                transpose=False,
                return_quant_info=False,
            )
        assert result.global_scale is not None
        return result.global_scale, kernel_mock.call_args.args[1]

    def test_invalid_and_tiny_global_amax(self) -> None:
        cases = [
            # An all-zero tensor gets 1.0.
            ([0.0, 0.0], 1.0),
            ([1.0, math.inf], 0.0),
            ([1.0, math.nan], math.nan),
            # Below 2688 / FLT_MAX the scale saturates to FLT_MAX.
            ([1e-36, 0.0], torch.finfo(torch.float32).max),
            ([1e-44, 0.0], torch.finfo(torch.float32).max),
        ]
        for values, want in cases:
            for scale in self._global_scales(torch.tensor([values])):
                self.assertEqual(scale.dtype, torch.float32)
                self.assertEqual(scale.shape, ())
                if math.isnan(want):
                    self.assertTrue(math.isnan(scale.item()), values)
                else:
                    self.assertEqual(scale.item(), want, values)
        # An ordinary amax (5.0) is unaffected: the scale is the reciprocal times 2688.
        scale, _ = self._global_scales(torch.tensor([[5.0, -1.0]]))
        want = torch.tensor(5.0).reciprocal() * 2688.0
        self.assertEqual(scale.item(), want.item())
