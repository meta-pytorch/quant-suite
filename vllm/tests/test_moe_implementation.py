# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the QuantSuite vLLM fused-experts implementation."""

from __future__ import annotations

import unittest

import torch
from quant_suite_core.rht_generator import HadamardMode
from quant_suite_recipe.definition import (
    FakeQuantRecipe,
    PipelineRecipe,
    QuantMoERecipe,
    RHTRecipe,
)
from quant_suite_recipe.implementation.moe import QuantizedMoE
from quant_suite_recipe.runtime_provider import QUANT_RECIPE_RUNTIME
from quant_suite_vllm.moe.implementation import apply_quantized_fused_experts


def _calculate_sqnr(actual: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    signal_power = torch.mean(reference.float().square())
    noise_power = torch.mean((reference.float() - actual.float()).square())
    return 10 * torch.log10(signal_power / noise_power)


def _prepare_moe(moe_recipe: QuantMoERecipe) -> QuantizedMoE:
    return QUANT_RECIPE_RUNTIME.prepare_moe(moe_recipe)


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Skip when cuda is not available",
)
class QuantizedFusedExpertsTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.device = "cuda"

    def test_basic_functionality(self) -> None:
        """The fused implementation preserves output shape and dtype."""
        # Setup: Create test tensors for MoE computation
        num_tokens = 32
        hidden_size = 128
        num_experts = 4
        intermediate_size = 256
        top_k = 2

        hidden_states = torch.randn(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=self.device
        )
        original_hidden_states = hidden_states.clone()
        topk_weights = torch.rand(
            num_tokens, top_k, dtype=torch.bfloat16, device=self.device
        )
        topk_ids = torch.randint(
            0, num_experts, (num_tokens, top_k), device=self.device
        )

        w13_weight = torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            dtype=torch.bfloat16,
            device=self.device,
        )
        w2_weight = torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
            device=self.device,
        )

        quantized_moe = _prepare_moe(QuantMoERecipe())

        output = apply_quantized_fused_experts(
            hidden_states=hidden_states,
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quantized_moe=quantized_moe,
            activation="silu",
        )

        # Assert: Output should have correct shape and dtype
        self.assertEqual(output.shape, (num_tokens, hidden_size))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertFalse(torch.isnan(output).any())
        self.assertNotEqual(output.data_ptr(), hidden_states.data_ptr())
        torch.testing.assert_close(hidden_states, original_hidden_states)

    def test_fake_quantization(self) -> None:
        # Setup: Create test data and fake quantization recipe
        num_tokens = 128
        hidden_size = 1024
        num_experts = 4
        intermediate_size = 1024
        top_k = 2

        hidden_states = torch.randn(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=self.device
        )
        topk_weights = torch.rand(
            num_tokens, top_k, dtype=torch.bfloat16, device=self.device
        )
        topk_ids = torch.randint(
            0, num_experts, (num_tokens, top_k), device=self.device
        )

        w13_weight = torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            dtype=torch.bfloat16,
            device=self.device,
        )
        w2_weight = torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
            device=self.device,
        )

        # Create MoE recipe with fake quantization
        _nvfp4_fake_config = {
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
        }
        quant_recipe = QUANT_RECIPE_RUNTIME.deserialize(
            {
                "moe_recipe": {
                    ".*": {
                        "a1": _nvfp4_fake_config,
                        "a2": _nvfp4_fake_config,
                        "w13": _nvfp4_fake_config,
                        "w2": _nvfp4_fake_config,
                    }
                }
            }
        )
        moe_recipe = quant_recipe.moe_recipe[".*"]

        no_quant = _prepare_moe(QuantMoERecipe())
        fake_quant = _prepare_moe(moe_recipe)

        # Execute: Run with no quantization
        output_no_quant = apply_quantized_fused_experts(
            hidden_states=hidden_states,
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quantized_moe=no_quant,
            activation="silu",
        )

        # Execute: Run with fake quantization
        output_fake_quant = apply_quantized_fused_experts(
            hidden_states=hidden_states,
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quantized_moe=fake_quant,
            activation="silu",
        )

        # Assert: Both should have same shape
        self.assertEqual(output_no_quant.shape, output_fake_quant.shape)

        # Assert: Outputs should be similar but not identical due to quantization
        self.assertFalse(torch.equal(output_no_quant, output_fake_quant))

        # Assert: SQNR should be reasonable for fake quantization
        sqnr = _calculate_sqnr(output_fake_quant, output_no_quant)
        self.assertGreaterEqual(sqnr, 15.0)

    def test_rht_activation_pipeline(self) -> None:
        """A multi-stage a1 pipeline ([R, quantize, R_inv]) runs end-to-end.

        MoE gets RHT (and any other stage) through the generic pipeline, with
        no new stage class -- the same
        ``[R, quantize, R_inv]`` round-trip the linear path supports.
        """
        num_tokens = 128
        hidden_size = 1024
        num_experts = 4
        intermediate_size = 1024
        top_k = 2
        block_size = 128

        hidden_states = torch.randn(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=self.device
        )
        topk_weights = torch.rand(
            num_tokens, top_k, dtype=torch.bfloat16, device=self.device
        )
        topk_ids = torch.randint(
            0, num_experts, (num_tokens, top_k), device=self.device
        )
        w13_weight = torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            dtype=torch.bfloat16,
            device=self.device,
        )
        w2_weight = torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
            device=self.device,
        )

        fake_config = {
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
        }
        rht_a1 = PipelineRecipe(
            [
                RHTRecipe(mode=HadamardMode.RIGHT, block_size=block_size, seed=0),
                FakeQuantRecipe.from_dict(fake_config),
                RHTRecipe(mode=HadamardMode.RIGHT_INV, block_size=block_size, seed=0),
            ]
        )
        rht_recipe = QuantMoERecipe(a1=rht_a1)

        no_quant = _prepare_moe(QuantMoERecipe())
        rht_quant = _prepare_moe(rht_recipe)

        output_no_quant = apply_quantized_fused_experts(
            hidden_states=hidden_states,
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quantized_moe=no_quant,
            activation="silu",
        )
        output_rht = apply_quantized_fused_experts(
            hidden_states=hidden_states,
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quantized_moe=rht_quant,
            activation="silu",
        )

        # The RHT+quantize stages actually ran: output is finite, shaped
        # correctly, and differs from the unquantized baseline.
        self.assertEqual(output_rht.shape, (num_tokens, hidden_size))
        self.assertEqual(output_rht.dtype, torch.bfloat16)
        self.assertFalse(torch.isnan(output_rht).any())
        self.assertFalse(torch.equal(output_no_quant, output_rht))
        # The round-trip keeps the quantized output close to the baseline.
        sqnr = _calculate_sqnr(output_rht, output_no_quant)
        self.assertGreaterEqual(sqnr, 15.0)

    def test_correctness_against_reference(self) -> None:
        # Setup: Create small test case for easy verification
        num_tokens = 128
        hidden_size = 1024
        num_experts = 128
        intermediate_size = 1024
        top_k = 1

        hidden_states = torch.randn(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=self.device
        )
        topk_weights = torch.softmax(
            torch.rand(num_tokens, top_k, dtype=torch.bfloat16, device=self.device),
            dim=-1,
        )
        topk_ids = torch.randint(
            0, num_experts, (num_tokens, top_k), device=self.device
        )

        w13_weight = torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            dtype=torch.bfloat16,
            device=self.device,
        )
        w2_weight = torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
            device=self.device,
        )

        quantized_moe = _prepare_moe(QuantMoERecipe())
        output = apply_quantized_fused_experts(
            hidden_states=hidden_states,
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quantized_moe=quantized_moe,
            activation="silu",
        )

        # Execute: Run reference implementation (simple, obvious correctness)
        reference_output = self._reference_moe_implementation(
            hidden_states,
            topk_weights,
            topk_ids,
            w13_weight,
            w2_weight,
            intermediate_size,
        )

        # Assert: Outputs should match within numerical tolerance
        self.assertEqual(output.shape, reference_output.shape)
        torch.testing.assert_close(
            output,
            reference_output,
            rtol=1e-3,
            atol=1e-3,
            msg="Quantized fused experts should match the reference implementation",
        )

    def _reference_moe_implementation(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        intermediate_size: int,
    ) -> torch.Tensor:
        """Simple reference MoE implementation for correctness testing.

        This is intentionally simple and easy to verify by inspection.
        Each token is processed independently through its top-k experts.
        """
        num_tokens, hidden_size = x.shape
        top_k = topk_ids.shape[1]

        # Initialize output
        output = torch.zeros_like(x)

        # Process each token independently
        for token_idx in range(num_tokens):
            token_input = x[token_idx]  # [hidden_size]
            token_output = torch.zeros_like(token_input)

            # Process each of the top-k experts for this token
            for k_idx in range(top_k):
                expert_id = topk_ids[token_idx, k_idx].item()
                expert_weight = topk_weights[token_idx, k_idx]

                # Get expert weights
                w13 = w13_weight[expert_id]  # [2*intermediate_size, hidden_size]
                w2 = w2_weight[expert_id]  # [hidden_size, intermediate_size]

                # Forward pass through this expert
                # 1. Gate and up projection
                gate_up = torch.matmul(w13, token_input)  # [2*intermediate_size]
                gate = gate_up[:intermediate_size]
                up = gate_up[intermediate_size:]

                # 2. Activation (SiLU)
                intermediate = torch.nn.functional.silu(gate) * up

                # 3. Down projection
                expert_output = torch.matmul(w2, intermediate)  # [hidden_size]

                # 4. Apply routing weight and accumulate
                token_output += expert_weight * expert_output

            output[token_idx] = token_output

        return output
