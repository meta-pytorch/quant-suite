# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import importlib.util
import unittest
from unittest import mock

import torch
from quant_suite_core.native_presets import PrecisionType
from quant_suite_recipe.api.implementation import QuantizedTensor
from quant_suite_recipe.definition import (
    FakeQuantRecipe,
    NativeQuantRecipe,
    NoQuantRecipe,
    PipelineRecipe,
    QuantLinearRecipe,
)
from quant_suite_recipe.implementation.linear import (
    maybe_quantize_weight,
    quantized_matmul,
    QuantizedLinear,
)
from quant_suite_recipe.implementation.quant_pipeline import Pipeline
from quant_suite_recipe.runtime import QuantRecipeRuntime

_HAS_MSLK: bool = importlib.util.find_spec("mslk") is not None
if _HAS_MSLK:
    import mslk.gemm  # noqa: F401
    import mslk.quantize  # noqa: F401


_BUILTIN_RUNTIME = QuantRecipeRuntime()


class _OffsetPipeline(Pipeline):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset
        self.call_count = 0

    def forward(self, tensor: torch.Tensor) -> QuantizedTensor:
        self.call_count += 1
        return QuantizedTensor(tensor=tensor + self.offset)


@unittest.skipUnless(_HAS_MSLK, "mslk is not available")
class LinearTest(unittest.TestCase):
    def _quantized_matmul(
        self,
        quantized_x: QuantizedTensor,
        quantized_w: QuantizedTensor,
        quantized_bias: QuantizedTensor | None = None,
    ) -> torch.Tensor:
        return quantized_matmul(
            quantized_x,
            quantized_w,
            quantized_bias,
            QuantLinearRecipe(),
        )

    def _native_tensor(
        self,
        precision: PrecisionType,
        *,
        global_scale: float | None = None,
    ) -> QuantizedTensor:
        return QuantizedTensor(
            tensor=torch.randn(2, 4),
            scale=torch.randn(2, 1),
            global_scale=(
                torch.tensor(global_scale) if global_scale is not None else None
            ),
            precision=precision,
        )

    def test_maybe_quantize_weight_skips_pipeline_when_disabled(self) -> None:
        weight = torch.randn(4, 8, dtype=torch.bfloat16)
        pipeline = _OffsetPipeline(1.0)

        result = maybe_quantize_weight(weight, pipeline, False)

        self.assertIs(result.tensor, weight)
        self.assertEqual(pipeline.call_count, 0)

    def test_maybe_quantize_weight_applies_pipeline_when_enabled(self) -> None:
        weight = torch.randn(4, 8, dtype=torch.bfloat16)
        pipeline = _OffsetPipeline(1.0)

        result = maybe_quantize_weight(weight, pipeline, True)

        assert result.tensor is not None
        torch.testing.assert_close(result.tensor, weight + 1.0)
        self.assertEqual(pipeline.call_count, 1)

    def test_maybe_quantize_weight_accepts_fp16_and_fp32(self) -> None:
        pipeline = _BUILTIN_RUNTIME.create_pipeline(PipelineRecipe([NoQuantRecipe()]))
        for dtype in (torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                weight = torch.randn(4, 8, dtype=dtype)

                result = maybe_quantize_weight(weight, pipeline, True)

                self.assertIs(result.tensor, weight)

    def test_maybe_quantize_weight_checks_dtype_only_when_enabled(self) -> None:
        weight = torch.ones(4, 8, dtype=torch.int8)
        pipeline = _OffsetPipeline(1.0)

        result = maybe_quantize_weight(weight, pipeline, False)
        self.assertIs(result.tensor, weight)

        with self.assertRaisesRegex(ValueError, "torch.int8"):
            maybe_quantize_weight(weight, pipeline, True)
        self.assertEqual(pipeline.call_count, 0)

    def test_prepared_linear_applies_activation_and_bias_pipelines(self) -> None:
        x = torch.randn(3, 4)
        weight = torch.randn(5, 4)
        bias = torch.randn(5)
        activation_pipeline = _OffsetPipeline(1.0)
        bias_pipeline = _OffsetPipeline(2.0)

        quantized_linear = QuantizedLinear(
            recipe=QuantLinearRecipe(),
            activation_pipeline=activation_pipeline,
            weight_pipeline=Pipeline(),
            bias_pipeline=bias_pipeline,
        )

        result = quantized_linear(
            x,
            QuantizedTensor(tensor=weight),
            bias,
        )

        expected = torch.matmul(x + 1.0, weight.t()) + bias + 2.0
        torch.testing.assert_close(result, expected)
        self.assertEqual(activation_pipeline.call_count, 1)
        self.assertEqual(bias_pipeline.call_count, 1)

    def test_plain_matmul_adds_bias(self) -> None:
        x = torch.randn(3, 4)
        weight = torch.randn(5, 4)
        bias = torch.randn(5)

        result = self._quantized_matmul(
            QuantizedTensor(tensor=x),
            QuantizedTensor(tensor=weight),
            QuantizedTensor(tensor=bias),
        )

        torch.testing.assert_close(result, torch.matmul(x, weight.t()) + bias)

    def test_mx4_native_precisions_forward_group_size(self) -> None:
        for precision, group_size in (
            (PrecisionType.MX4_32, 32),
            (PrecisionType.MX4_16, 16),
        ):
            with self.subTest(precision=precision):
                quantized_x = self._native_tensor(precision)
                quantized_weight = self._native_tensor(precision)
                expected = torch.randn(2, 2)

                with mock.patch.object(
                    torch.ops.mslk,
                    "f4f4bf16",
                    return_value=expected,
                ) as native_matmul:
                    result = self._quantized_matmul(quantized_x, quantized_weight)

                self.assertIs(result, expected)
                native_matmul.assert_called_once()
                args = native_matmul.call_args.args
                self.assertIs(args[0], quantized_x.tensor)
                self.assertIs(args[1], quantized_weight.tensor)
                self.assertIs(args[2], quantized_x.scale)
                self.assertIs(args[3], quantized_weight.scale)
                self.assertIsNone(args[4])
                self.assertIsNone(args[5])
                self.assertEqual(args[6], group_size)

    def test_global_scale_precisions_forward_reciprocal_product(self) -> None:
        quantized_x = self._native_tensor(PrecisionType.NVFP4, global_scale=2.0)
        quantized_weight = self._native_tensor(PrecisionType.NVFP4, global_scale=4.0)
        expected = torch.randn(2, 2)

        with mock.patch.object(
            torch.ops.mslk,
            "f4f4bf16",
            return_value=expected,
        ) as native_matmul:
            result = self._quantized_matmul(quantized_x, quantized_weight)

        self.assertIs(result, expected)
        args = native_matmul.call_args.args
        self.assertIs(args[0], quantized_x.tensor)
        self.assertIs(args[1], quantized_weight.tensor)
        self.assertIs(args[2], quantized_x.scale)
        self.assertIs(args[3], quantized_weight.scale)
        self.assertIsNone(args[4])
        torch.testing.assert_close(args[5], torch.tensor(0.125))

    def test_native_matmul_adds_non_native_bias(self) -> None:
        quantized_x = self._native_tensor(PrecisionType.MX4_32)
        quantized_weight = self._native_tensor(PrecisionType.MX4_32)
        bias = torch.tensor([1.0, 2.0])
        matmul_result = torch.zeros(2, 2)

        with mock.patch.object(
            torch.ops.mslk,
            "f4f4bf16",
            return_value=matmul_result,
        ):
            result = self._quantized_matmul(
                quantized_x,
                quantized_weight,
                QuantizedTensor(tensor=bias),
            )

        torch.testing.assert_close(result, matmul_result + bias)

    def test_native_bias_is_rejected(self) -> None:
        native_bias = self._native_tensor(PrecisionType.MX4_32)
        for quantized_x, quantized_weight in (
            (
                self._native_tensor(PrecisionType.MX4_32),
                self._native_tensor(PrecisionType.MX4_32),
            ),
            (
                QuantizedTensor(tensor=torch.randn(2, 4)),
                QuantizedTensor(tensor=torch.randn(2, 4)),
            ),
        ):
            with self.subTest(precision=quantized_x.precision):
                with self.assertRaisesRegex(
                    NotImplementedError, "Native quantization of a Linear bias"
                ):
                    self._quantized_matmul(quantized_x, quantized_weight, native_bias)

    def test_missing_input_tensor_is_rejected(self) -> None:
        with self.assertRaisesRegex(AssertionError, "quantized_x.tensor"):
            self._quantized_matmul(
                QuantizedTensor(),
                QuantizedTensor(tensor=torch.randn(2, 4)),
            )

        with self.assertRaisesRegex(AssertionError, "quantized_w.tensor"):
            self._quantized_matmul(
                QuantizedTensor(tensor=torch.randn(2, 4)),
                QuantizedTensor(),
            )

    def test_missing_native_scale_is_rejected(self) -> None:
        valid = self._native_tensor(PrecisionType.MX4_32)
        missing_scale = QuantizedTensor(
            tensor=torch.randn(2, 4),
            precision=PrecisionType.MX4_32,
        )

        with self.assertRaisesRegex(AssertionError, "x_scale must be provided"):
            self._quantized_matmul(missing_scale, valid)
        with self.assertRaisesRegex(AssertionError, "w_scale must be provided"):
            self._quantized_matmul(valid, missing_scale)

    def test_missing_global_scale_is_rejected(self) -> None:
        for missing_side in ("activation", "weight"):
            with self.subTest(missing_side=missing_side):
                quantized_x = self._native_tensor(
                    PrecisionType.NVFP4,
                    global_scale=2.0 if missing_side == "weight" else None,
                )
                quantized_weight = self._native_tensor(
                    PrecisionType.NVFP4,
                    global_scale=4.0 if missing_side == "activation" else None,
                )

                with self.assertRaisesRegex(
                    AssertionError,
                    f"{('x' if missing_side == 'activation' else 'w')}_global_scale",
                ):
                    self._quantized_matmul(quantized_x, quantized_weight)

    def test_different_native_precisions_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "same native precision"):
            self._quantized_matmul(
                self._native_tensor(PrecisionType.MX4_32),
                self._native_tensor(PrecisionType.NVFP4, global_scale=1.0),
            )

    def test_native_and_unquantized_inputs_are_rejected(self) -> None:
        native = self._native_tensor(PrecisionType.MX4_32)
        unquantized = QuantizedTensor(tensor=torch.randn(2, 4))

        with self.assertRaisesRegex(ValueError, "same native precision"):
            self._quantized_matmul(native, unquantized)
        with self.assertRaisesRegex(ValueError, "same native precision"):
            self._quantized_matmul(unquantized, native)

    def test_no_quant_pipelines_match_plain_matmul(self) -> None:
        x = torch.randn(3, 8)
        weight = torch.randn(4, 8)
        bias = torch.randn(4)

        output = _BUILTIN_RUNTIME.prepare_linear(QuantLinearRecipe())(
            x,
            QuantizedTensor(tensor=weight),
            bias,
        )

        torch.testing.assert_close(output, torch.matmul(x, weight.t()) + bias)

    @unittest.skipUnless(torch.cuda.is_available(), "Skip when cuda is not available")
    def test_fake_activation_pipeline_matches_direct_runtime_application(self) -> None:
        input_tensor = torch.ones(1, 16, dtype=torch.bfloat16, device="cuda") * 6.0
        input_tensor[:, -1] = 0.4
        weight = torch.zeros(1, 16, dtype=torch.bfloat16, device="cuda")
        weight[:, -1] = 1.0
        recipe = QuantLinearRecipe(
            activation=PipelineRecipe(
                [
                    FakeQuantRecipe.from_dict(
                        {"quant_config": "nvfp4", "use_triton": False}
                    )
                ]
            )
        )
        quantized_linear = _BUILTIN_RUNTIME.prepare_linear(recipe)
        activation_pipeline = quantized_linear.activation_pipeline
        quantized_activation = _BUILTIN_RUNTIME.apply_pipeline(
            activation_pipeline,
            input_tensor,
        )
        assert quantized_activation.tensor is not None

        output = quantized_linear(
            input_tensor,
            QuantizedTensor(tensor=weight),
            None,
        )

        torch.testing.assert_close(
            output,
            torch.matmul(quantized_activation.tensor, weight.t()),
        )

    @unittest.skipUnless(torch.cuda.is_available(), "Skip when cuda is not available")
    def test_maybe_quantize_weight_with_fake_quantization(self) -> None:
        weight = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
        recipe = QuantLinearRecipe(
            weight=PipelineRecipe(
                [
                    FakeQuantRecipe.from_dict(
                        {"quant_config": "nvfp4", "use_triton": False}
                    )
                ]
            ),
            init_quantized_weight=True,
        )

        result = maybe_quantize_weight(
            weight,
            _BUILTIN_RUNTIME.create_pipeline(recipe.weight),
            recipe.init_quantized_weight,
        )

        self.assertIsNone(result.precision)
        assert result.tensor is not None
        self.assertFalse(torch.equal(result.tensor, weight))

    @unittest.skipIf(
        not torch.cuda.is_available()
        or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 10,
        "Skip when B200 is not available",
    )
    def test_maybe_quantize_weight_with_native_quantization(self) -> None:
        weight = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
        recipe = QuantLinearRecipe(
            weight=PipelineRecipe([NativeQuantRecipe(PrecisionType.NVFP4)]),
            init_quantized_weight=True,
        )

        result = maybe_quantize_weight(
            weight,
            _BUILTIN_RUNTIME.create_pipeline(recipe.weight),
            recipe.init_quantized_weight,
        )

        self.assertIs(result.precision, PrecisionType.NVFP4)
        assert result.tensor is not None
        assert result.scale is not None
        assert result.global_scale is not None
        self.assertEqual(result.tensor.shape, (64, 64))
        self.assertEqual(result.tensor.dtype, torch.float4_e2m1fn_x2)
        self.assertEqual(result.global_scale.numel(), 1)
