# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import List

import torch
from parameterized import parameterized
from quant_suite_core import (
    BlockSize,
    ElemDtype,
    ElemRoundMode,
    fake_quantize,
    GlobalScaleGranularity,
    QuantConfig,
    ScaleDtype,
    ScaleNormMode,
    ScaleRoundMode,
)
from quant_suite_core.fake_presets import FAKE_PRESETS
from quant_suite_core.qdq_triton import qdq_triton_mega
from quant_suite_core.quant_config import ELEM_DTYPE_BITS_MAP
from quant_test_utils import make_quant_config, MX_NV_ELEM_CONFIGS, skip_if_no_cuda


@skip_if_no_cuda
class TestStochasticRounding(unittest.TestCase):
    """Test stochastic rounding behavior in MX quantization."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.device = "cuda"

    def _create_sr_config(
        self,
        use_stochastic: bool,
        elem_dtype: ElemDtype,
        scale_dtype: ScaleDtype,
        scale_norm_mode: ScaleNormMode,
        scale_round_mode: ScaleRoundMode,
        global_scale_granularity: GlobalScaleGranularity,
    ) -> QuantConfig:
        """Create a config with stochastic or nearest rounding."""
        return make_quant_config(
            scale_dtype=scale_dtype,
            scale_norm_mode=scale_norm_mode,
            scale_round_mode=scale_round_mode,
            global_scale_granularity=global_scale_granularity,
            elem_dtype=elem_dtype,
            elem_round_mode=(
                ElemRoundMode.STOCHASTIC if use_stochastic else ElemRoundMode.NEAREST
            ),
            block_size=BlockSize(1, 16),
            global_scale_dtype=(
                ScaleDtype.FP32
                if global_scale_granularity != GlobalScaleGranularity.NONE
                else ScaleDtype.NONE
            ),
        )

    @parameterized.expand(MX_NV_ELEM_CONFIGS)
    def test_stochastic_rounding_correctness(
        self,
        _name: str,
        use_triton: bool,
        elem_dtype: ElemDtype,
        scale_dtype: ScaleDtype,
        scale_norm_mode: ScaleNormMode,
        scale_round_mode: ScaleRoundMode,
        global_scale_granularity: GlobalScaleGranularity,
    ) -> None:
        """Test that stochastic rounding differs from RTN by at most 1 ULP."""
        tensor = torch.randn(128, 128, dtype=torch.bfloat16, device=self.device)

        config_rtn = self._create_sr_config(
            use_stochastic=False,
            elem_dtype=elem_dtype,
            scale_dtype=scale_dtype,
            scale_norm_mode=scale_norm_mode,
            scale_round_mode=scale_round_mode,
            global_scale_granularity=global_scale_granularity,
        )
        config_sr = self._create_sr_config(
            use_stochastic=True,
            elem_dtype=elem_dtype,
            scale_dtype=scale_dtype,
            scale_norm_mode=scale_norm_mode,
            scale_round_mode=scale_round_mode,
            global_scale_granularity=global_scale_granularity,
        )

        # Use return_quant_info=True to get quantized_val
        result_rtn_info = fake_quantize(
            tensor,
            "test",
            config_rtn,
            transpose=False,
            use_triton=use_triton,
            return_quant_info=True,
        )
        result_sr_info = fake_quantize(
            tensor,
            "test",
            config_sr,
            transpose=False,
            use_triton=use_triton,
            return_quant_info=True,
        )
        elem_rtn = result_rtn_info.quantized_val.float()
        elem_sr = result_sr_info.quantized_val.float()

        ebits, mbits = ELEM_DTYPE_BITS_MAP[config_sr.elem_dtype]
        exp_bias = (1 << (ebits - 1)) - 1  # _n_ones(ebits - 1)
        shift_amount = 23 - mbits

        # Compute min_normal for shifting subnormals to normal range
        min_normal = 2.0 ** (1 - exp_bias)

        # Handle zero cases separately: 0 is special
        zero_mask = (elem_rtn == 0) | (elem_sr == 0)
        if zero_mask.any():
            zero_rtn = elem_rtn[zero_mask]
            zero_sr = elem_sr[zero_mask]
            # For zero cases, both should be zero or one should be min_subnormal
            min_subnormal = min_normal / (2**mbits)
            valid_zero = (
                ((zero_rtn == 0) & (zero_sr.abs() <= min_subnormal))
                | ((zero_sr == 0) & (zero_rtn.abs() <= min_subnormal))
                | ((zero_rtn == 0) & (zero_sr == 0))
            )
            self.assertTrue(
                valid_zero.all(),
                f"Zero case: expected 0 or ±{min_subnormal}, got rtn={zero_rtn}, sr={zero_sr}",
            )

        # For non-zero cases: use the shifting trick to handle subnormals
        non_zero_mask = ~zero_mask
        if non_zero_mask.any():
            non_zero_rtn = elem_rtn[non_zero_mask].abs()
            non_zero_sr = elem_sr[non_zero_mask].abs()

            # Check signs match (SR should not flip sign)
            signs_match = torch.sign(elem_rtn[non_zero_mask]) == torch.sign(
                elem_sr[non_zero_mask]
            )
            self.assertTrue(signs_match.all(), "SR should not flip sign")

            # Shift subnormal values to normal range before bit comparison
            # Key: if EITHER value is subnormal, shift BOTH to keep relative position
            subnormal_mask_rtn = non_zero_rtn < min_normal
            subnormal_mask_sr = non_zero_sr < min_normal
            any_subnormal = subnormal_mask_rtn | subnormal_mask_sr

            # Shift both when either is subnormal (handles boundary case)
            shifted_rtn = torch.where(
                any_subnormal, non_zero_rtn + min_normal, non_zero_rtn
            )
            shifted_sr = torch.where(
                any_subnormal, non_zero_sr + min_normal, non_zero_sr
            )

            # Now compare bits - all values are in normal range
            rtn_bits = shifted_rtn.view(torch.int32) >> shift_amount
            sr_bits = shifted_sr.view(torch.int32) >> shift_amount
            bit_diff = (rtn_bits - sr_bits).abs()
            max_bit_diff = bit_diff.max().item()

            # Debug: print values that don't satisfy <= 1 condition
            bad_mask = bit_diff > 1
            if bad_mask.any():
                bad_rtn = non_zero_rtn[bad_mask]
                bad_sr = non_zero_sr[bad_mask]
                bad_rtn_bits = rtn_bits[bad_mask]
                bad_sr_bits = sr_bits[bad_mask]
                bad_diff = bit_diff[bad_mask]
                print("=== Values exceeding 1 ULP ===")
                print(f"RTN values: {bad_rtn}")
                print(f"SR values: {bad_sr}")
                print(f"RTN bits (>>{shift_amount}): {bad_rtn_bits}")
                print(f"SR bits (>>{shift_amount}): {bad_sr_bits}")
                print(f"Bit diff: {bad_diff}")
                print(f"RTN int32: {bad_rtn.view(torch.int32)}")
                print(f"SR int32: {bad_sr.view(torch.int32)}")

            self.assertLessEqual(
                max_bit_diff,
                1,
                f"SR exceeded 1 ULP: max bit diff = {max_bit_diff}",
            )

    @parameterized.expand(MX_NV_ELEM_CONFIGS)
    def test_stochastic_rounding_distribution(
        self,
        _name: str,
        use_triton: bool,
        elem_dtype: ElemDtype,
        scale_dtype: ScaleDtype,
        scale_norm_mode: ScaleNormMode,
        scale_round_mode: ScaleRoundMode,
        global_scale_granularity: GlobalScaleGranularity,
    ) -> None:
        """Run multiple times on values at rounding boundaries.

        Verify stochastic behavior (not always rounding same direction).
        """
        tensor = torch.randn(1, 16, dtype=torch.bfloat16, device=self.device)

        config = self._create_sr_config(
            use_stochastic=True,
            elem_dtype=elem_dtype,
            scale_dtype=scale_dtype,
            scale_norm_mode=scale_norm_mode,
            scale_round_mode=scale_round_mode,
            global_scale_granularity=global_scale_granularity,
        )

        results: List[tuple[float, ...]] = []
        for i in range(10):
            torch.manual_seed(i)
            result = fake_quantize(
                tensor, "test", config, transpose=False, use_triton=use_triton
            )
            print(f"Round {i}: {result}")
            # Convert whole tensor to tuple for uniqueness check
            results.append(tuple(result.flatten().tolist()))

        unique_tensors = set(results)
        self.assertGreater(
            len(unique_tensors),
            1,
            f"Stochastic rounding should produce varying results, got {unique_tensors}",
        )

    def test_triton_seed_advances_and_follows_manual_seed(self) -> None:
        """Each Triton call draws fresh bits; the same CUDA seed repeats them."""
        tensor = torch.randn(256, 512, dtype=torch.bfloat16, device=self.device)
        config = QuantConfig.from_dict(FAKE_PRESETS["mxfp4_sr"])

        first = qdq_triton_mega(tensor, "test", config)
        second = qdq_triton_mega(tensor, "test", config)
        self.assertFalse(torch.equal(first, second))

        torch.cuda.manual_seed(1234)
        seeded = qdq_triton_mega(tensor, "test", config)
        torch.cuda.manual_seed(1234)
        reseeded = qdq_triton_mega(tensor, "test", config)
        self.assertTrue(torch.equal(seeded, reseeded))

    def test_triton_graph_replay_draws_fresh_seed(self) -> None:
        """Each replay of a captured Triton call rounds with a new seed."""
        tensor = torch.randn(256, 512, dtype=torch.bfloat16, device=self.device)
        config = QuantConfig.from_dict(FAKE_PRESETS["mxfp4_sr"])

        # Compile the device-seed kernel variant before the checked capture.
        warmup = torch.cuda.CUDAGraph()
        with torch.cuda.graph(warmup, capture_error_mode="relaxed"):
            qdq_triton_mega(tensor, "test", config)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = qdq_triton_mega(tensor, "test", config)
        graph.replay()
        first = out.clone()
        graph.replay()
        self.assertFalse(torch.equal(first, out))


if __name__ == "__main__":
    unittest.main()
