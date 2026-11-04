# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from dataclasses import replace
from inspect import signature
from typing import Any, cast
from unittest.mock import patch

import torch
from quant_suite_core.native_presets import PrecisionType
from quant_suite_recipe.definition import (
    NativeQuantRecipe,
    PipelineRecipe,
    QuantMoERecipe,
    QuantRecipe,
)
from quant_suite_recipe.implementation.moe import QuantizedMoE
from quant_suite_recipe.runtime import QuantRecipeRuntime
from quant_suite_vllm.moe.method import QuantSuiteMoEMethod
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    MoEActivation,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig


class QuantSuiteMoEMethodTest(unittest.TestCase):
    def make_moe_config(
        self,
        *,
        activation: MoEActivation,
        intermediate_size: int,
        hidden_size: int = 4,
        num_experts: int = 2,
        tp_size: int = 1,
        tp_rank: int = 0,
    ) -> FusedMoEConfig:
        parallel_config = replace(
            FusedMoEParallelConfig.make_no_parallel(),
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        config_type: Any = FusedMoEConfig
        size_argument = (
            {"intermediate_size": intermediate_size}
            if "intermediate_size" in signature(config_type).parameters
            else {"intermediate_size_per_partition": intermediate_size // tp_size}
        )
        with patch(
            "vllm.model_executor.layers.fused_moe.config."
            "current_platform.is_cuda_alike",
            return_value=True,
        ):
            return cast(
                FusedMoEConfig,
                config_type(
                    num_experts=num_experts,
                    experts_per_token=1,
                    hidden_dim=hidden_size,
                    num_local_experts=num_experts,
                    num_logical_experts=num_experts,
                    activation=activation,
                    is_act_and_mul=activation.is_gated,
                    device="cpu",
                    routing_method=RoutingMethodType.Default,
                    moe_parallel_config=parallel_config,
                    in_dtype=torch.bfloat16,
                    **size_argument,
                ),
            )

    def make_layer(
        self,
        moe_config: FusedMoEConfig,
        recipe: QuantMoERecipe,
    ) -> tuple[QuantSuiteMoEMethod, torch.nn.Module]:
        method = QuantSuiteMoEMethod(
            QuantRecipe(moe_recipe={"model.layers.0.mlp": recipe}),
            "model.layers.0.mlp",
            moe_config=moe_config,
        )

        def weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: int,
        ) -> None:
            shard_dim = 1 if shard_id == "w2" else 0
            num_shards = 2 if method.moe.is_act_and_mul else 1
            shard_size = (
                param.shape[2] if shard_id == "w2" else param.shape[1] // num_shards
            )
            tp_rank = moe_config.moe_parallel_config.tp_rank
            source = loaded_weight.narrow(
                shard_dim,
                tp_rank * shard_size,
                shard_size,
            )
            destination = param.data[expert_id]
            if shard_id in ("w1", "w3"):
                offset = shard_size if shard_id == "w3" else 0
                destination = destination.narrow(shard_dim, offset, shard_size)
            destination.copy_(source)

        layer = torch.nn.Module()
        method.create_weights(
            layer,
            num_experts=moe_config.num_experts,
            hidden_size=moe_config.hidden_dim,
            intermediate_size_per_partition=(
                moe_config.intermediate_size_per_partition
            ),
            params_dtype=torch.bfloat16,
            weight_loader=weight_loader,
        )
        return method, layer

    def setUp(self) -> None:
        self.recipe = QuantMoERecipe()
        quant_recipe = QuantRecipe(moe_recipe={"model.layers.0.mlp": self.recipe})
        self.moe_config = self.make_moe_config(
            activation=MoEActivation.SILU,
            intermediate_size=3,
        )
        self.method = QuantSuiteMoEMethod(
            quant_recipe,
            "model.layers.0.mlp",
            moe_config=self.moe_config,
        )
        self.layer = torch.nn.Module()
        self.method.create_weights(
            self.layer,
            num_experts=2,
            hidden_size=4,
            intermediate_size_per_partition=3,
            params_dtype=torch.bfloat16,
        )

    def test_prepares_moe_pipelines_and_preserves_noop_weights(self) -> None:
        w13_weight = self.layer.get_parameter("w13_weight")
        w2_weight = self.layer.get_parameter("w2_weight")

        self.method.process_weights_after_loading(self.layer)

        self.assertIs(self.method.moe_recipe, self.recipe)
        self.assertIs(self.layer.get_parameter("w13_weight"), w13_weight)
        self.assertIs(self.layer.get_parameter("w2_weight"), w2_weight)
        dynamic_layer: Any = self.layer
        self.assertIsInstance(dynamic_layer.quantized_moe, QuantizedMoE)
        self.assertIsNone(self.method.get_fused_moe_quant_config(self.layer))

    def test_creates_vllm_moe_weight_layout(self) -> None:
        self.assertEqual(self.layer.get_parameter("w13_weight").shape, (2, 6, 4))
        self.assertEqual(self.layer.get_parameter("w2_weight").shape, (2, 4, 3))

    def test_supports_nemotron_relu2_without_allocating_a_gate_shard(self) -> None:
        non_gated_config = self.make_moe_config(
            activation=MoEActivation["RELU2_NO_MUL"],
            intermediate_size=3,
        )
        method = QuantSuiteMoEMethod(
            QuantRecipe(moe_recipe={"model.layers.0.mlp": self.recipe}),
            "model.layers.0.mlp",
            moe_config=non_gated_config,
        )
        layer = torch.nn.Module()

        method.create_weights(
            layer,
            num_experts=2,
            hidden_size=4,
            intermediate_size_per_partition=3,
            params_dtype=torch.bfloat16,
        )
        method.process_weights_after_loading(layer)

        self.assertEqual(layer.get_parameter("w13_weight").shape, (2, 3, 4))
        self.assertEqual(layer.get_parameter("w2_weight").shape, (2, 4, 3))
        dynamic_layer: Any = layer
        self.assertIsInstance(dynamic_layer.quantized_moe, QuantizedMoE)

    def test_tp4_loads_nemotron_shards_and_preserves_local_pipeline_shape(
        self,
    ) -> None:
        global_intermediate_size = 2688
        local_intermediate_size = 672
        hidden_size = 1024
        up_proj = (
            torch.arange(
                global_intermediate_size,
                dtype=torch.bfloat16,
            )
            .unsqueeze(1)
            .expand(-1, hidden_size)
        )
        down_proj = (
            torch.arange(
                global_intermediate_size,
                dtype=torch.bfloat16,
            )
            .unsqueeze(0)
            .expand(hidden_size, -1)
        )
        quant_recipe = QuantRecipeRuntime().deserialize(
            {"moe_recipe": {".*": {"w13": "fake_mxfp8"}}}
        )
        recipe = quant_recipe.moe_recipe[".*"]

        for tp_rank in range(4):
            with self.subTest(tp_rank=tp_rank):
                moe_config = self.make_moe_config(
                    activation=MoEActivation["RELU2_NO_MUL"],
                    intermediate_size=global_intermediate_size,
                    hidden_size=hidden_size,
                    num_experts=1,
                    tp_size=4,
                    tp_rank=tp_rank,
                )
                method, layer = self.make_layer(moe_config, recipe)
                w13_weight = layer.get_parameter("w13_weight")
                w2_weight = layer.get_parameter("w2_weight")

                cast(Any, w13_weight).weight_loader(
                    param=w13_weight,
                    loaded_weight=up_proj,
                    weight_name="w13_weight",
                    shard_id="w1",
                    expert_id=0,
                )
                cast(Any, w2_weight).weight_loader(
                    param=w2_weight,
                    loaded_weight=down_proj,
                    weight_name="w2_weight",
                    shard_id="w2",
                    expert_id=0,
                )

                start = tp_rank * local_intermediate_size
                end = start + local_intermediate_size
                torch.testing.assert_close(w13_weight[0], up_proj[start:end])
                torch.testing.assert_close(w2_weight[0], down_proj[:, start:end])

                with patch(
                    "quant_suite_core.fake_quant.qdq_triton_mega",
                    side_effect=lambda tensor, *_args, **_kwargs: tensor,
                ) as qdq:
                    method.process_weights_after_loading(layer)

                self.assertEqual(
                    layer.get_parameter("w13_weight").shape,
                    (1, local_intermediate_size, hidden_size),
                )
                self.assertEqual(
                    layer.get_parameter("w2_weight").shape,
                    (1, hidden_size, local_intermediate_size),
                )
                self.assertEqual(
                    qdq.call_args.args[0].shape,
                    (1, local_intermediate_size, hidden_size),
                )
                self.assertFalse(moe_config.moe_parallel_config.use_ep)

    def test_gated_loader_keeps_w1_and_w3_in_separate_halves(self) -> None:
        method, layer = self.make_layer(self.moe_config, self.recipe)
        w13_weight = layer.get_parameter("w13_weight")
        w1 = torch.arange(12, dtype=torch.bfloat16).view(3, 4)
        w3 = w1 + 100

        cast(Any, w13_weight).weight_loader(
            param=w13_weight,
            loaded_weight=w1,
            weight_name="w13_weight",
            shard_id="w1",
            expert_id=0,
        )
        cast(Any, w13_weight).weight_loader(
            param=w13_weight,
            loaded_weight=w3,
            weight_name="w13_weight",
            shard_id="w3",
            expert_id=0,
        )

        torch.testing.assert_close(w13_weight[0, :3], w1)
        torch.testing.assert_close(w13_weight[0, 3:], w3)
        method.process_weights_after_loading(layer)
        self.assertEqual(layer.get_parameter("w13_weight").shape, (2, 6, 4))

    def test_non_gated_expert_parallel_forwards_local_layout_and_map(self) -> None:
        global_num_experts = 4
        local_num_experts = 2
        base_config = self.make_moe_config(
            activation=MoEActivation["RELU2_NO_MUL"],
            intermediate_size=3,
            num_experts=global_num_experts,
        )
        parallel_config = replace(
            base_config.moe_parallel_config,
            use_ep=True,
            ep_size=2,
        )
        moe_config = replace(
            base_config,
            num_local_experts=local_num_experts,
            moe_parallel_config=parallel_config,
        )
        method = QuantSuiteMoEMethod(
            QuantRecipe(moe_recipe={"model.layers.0.mlp": self.recipe}),
            "model.layers.0.mlp",
            moe_config=moe_config,
        )
        layer = torch.nn.Module()

        def weight_loader(*_args: object, **_kwargs: object) -> None:
            pass

        method.create_weights(
            layer,
            num_experts=local_num_experts,
            hidden_size=4,
            intermediate_size_per_partition=3,
            params_dtype=torch.bfloat16,
            weight_loader=weight_loader,
        )
        w13_weight = layer.get_parameter("w13_weight")
        w2_weight = layer.get_parameter("w2_weight")
        self.assertEqual(w13_weight.shape, (local_num_experts, 3, 4))
        self.assertEqual(w2_weight.shape, (local_num_experts, 4, 3))
        self.assertIs(cast(Any, w13_weight).weight_loader, weight_loader)
        self.assertIs(cast(Any, w2_weight).weight_loader, weight_loader)

        method.process_weights_after_loading(layer)
        expert_map = torch.tensor([-1, -1, 0, 1], dtype=torch.int32)
        dynamic_layer: Any = layer
        dynamic_layer.activation = MoEActivation["RELU2_NO_MUL"]
        dynamic_layer.global_num_experts = global_num_experts
        dynamic_layer.apply_router_weight_on_input = False
        dynamic_layer.expert_map = expert_map
        hidden_states = torch.empty(1, 4, dtype=torch.bfloat16)
        topk_weights = torch.ones(1, 1, dtype=torch.bfloat16)
        topk_ids = torch.zeros(1, 1, dtype=torch.int32)
        expected = torch.empty_like(hidden_states)

        with patch(
            "quant_suite_vllm.moe.method.apply_quantized_fused_experts",
            return_value=expected,
        ) as apply_fused_experts:
            result = method.apply(
                layer,
                hidden_states,
                topk_weights,
                topk_ids,
            )

        self.assertIs(result, expected)
        self.assertTrue(method.moe.use_ep)
        self.assertIs(apply_fused_experts.call_args.kwargs["expert_map"], expert_map)
        self.assertEqual(
            apply_fused_experts.call_args.kwargs["global_num_experts"],
            global_num_experts,
        )
        self.assertEqual(
            apply_fused_experts.call_args.kwargs["activation"],
            MoEActivation["RELU2_NO_MUL"],
        )

    def test_rejects_unsupported_vllm_moe_variants(self) -> None:
        cases = (
            (
                "expert bias",
                replace(self.moe_config, has_bias=True),
                "does not support MoE bias",
            ),
        )

        for name, moe_config, message in cases:
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(
                    NotImplementedError,
                    message,
                ),
            ):
                QuantSuiteMoEMethod(
                    QuantRecipe(),
                    "model.layers.0.mlp",
                    moe_config=moe_config,
                )

    def test_rejects_native_weight_pipelines_that_are_initialized(self) -> None:
        native_pipeline = PipelineRecipe([NativeQuantRecipe(PrecisionType.NVFP4)])
        cases = (
            ("w13", QuantMoERecipe(w13=native_pipeline)),
            ("w2", QuantMoERecipe(w2=native_pipeline)),
        )

        for name, recipe in cases:
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(
                    NotImplementedError,
                    "does not support native weight quantization",
                ),
            ):
                QuantSuiteMoEMethod(
                    QuantRecipe(moe_recipe={"model.layers.0.mlp": recipe}),
                    "model.layers.0.mlp",
                    moe_config=self.moe_config,
                )

    def test_allows_native_weight_recipe_when_initialization_is_disabled(self) -> None:
        recipe = QuantMoERecipe(
            w13=PipelineRecipe([NativeQuantRecipe(PrecisionType.NVFP4)]),
            init_quantized_weight=False,
        )

        method = QuantSuiteMoEMethod(
            QuantRecipe(moe_recipe={"model.layers.0.mlp": recipe}),
            "model.layers.0.mlp",
            moe_config=self.moe_config,
        )

        self.assertIs(method.moe_recipe, recipe)

    def test_apply_accepts_shared_experts_and_uses_non_inplace_execution(self) -> None:
        method = QuantSuiteMoEMethod(
            QuantRecipe(moe_recipe={"model.layers.0.mlp": self.recipe}),
            "model.layers.0.mlp",
            moe_config=self.moe_config,
        )
        method.process_weights_after_loading(self.layer)
        dynamic_layer: Any = self.layer
        dynamic_layer.activation = MoEActivation.SILU
        dynamic_layer.global_num_experts = 2
        dynamic_layer.apply_router_weight_on_input = False
        dynamic_layer.expert_map = None
        hidden_states = torch.empty(1, 4, dtype=torch.bfloat16)
        topk_weights = torch.ones(1, 1, dtype=torch.bfloat16)
        topk_ids = torch.zeros(1, 1, dtype=torch.int32)
        expected = torch.empty_like(hidden_states)

        with patch(
            "quant_suite_vllm.moe.method.apply_quantized_fused_experts",
            return_value=expected,
        ) as apply_fused_experts:
            result = method.apply(
                self.layer,
                hidden_states,
                topk_weights,
                topk_ids,
                shared_experts=object(),
                shared_experts_input=hidden_states,
            )

        self.assertIs(result, expected)
        self.assertNotIn("inplace", apply_fused_experts.call_args.kwargs)

    def test_apply_preserves_non_gated_activation_and_pipeline_owner(self) -> None:
        non_gated_config = self.make_moe_config(
            activation=MoEActivation["RELU2_NO_MUL"],
            intermediate_size=3,
        )
        method = QuantSuiteMoEMethod(
            QuantRecipe(moe_recipe={"model.layers.0.mlp": self.recipe}),
            "model.layers.0.mlp",
            moe_config=non_gated_config,
        )
        layer = torch.nn.Module()
        method.create_weights(
            layer,
            num_experts=2,
            hidden_size=4,
            intermediate_size_per_partition=3,
            params_dtype=torch.bfloat16,
        )
        method.process_weights_after_loading(layer)
        dynamic_layer: Any = layer
        dynamic_layer.activation = MoEActivation["RELU2_NO_MUL"]
        dynamic_layer.global_num_experts = 2
        dynamic_layer.apply_router_weight_on_input = False
        dynamic_layer.expert_map = None
        hidden_states = torch.empty(1, 4, dtype=torch.bfloat16)
        topk_weights = torch.ones(1, 1, dtype=torch.bfloat16)
        topk_ids = torch.zeros(1, 1, dtype=torch.int32)
        expected = torch.empty_like(hidden_states)

        with patch(
            "quant_suite_vllm.moe.method.apply_quantized_fused_experts",
            return_value=expected,
        ) as apply_fused_experts:
            result = method.apply(
                layer,
                hidden_states,
                topk_weights,
                topk_ids,
            )

        self.assertIs(result, expected)
        self.assertIs(
            apply_fused_experts.call_args.kwargs["quantized_moe"],
            dynamic_layer.quantized_moe,
        )
        self.assertEqual(
            apply_fused_experts.call_args.kwargs["activation"],
            MoEActivation["RELU2_NO_MUL"],
        )
