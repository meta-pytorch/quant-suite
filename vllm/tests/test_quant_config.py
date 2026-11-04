# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import quant_suite_vllm.method_provider as method_provider
import quant_suite_vllm.plugin as plugin
import torch
from quant_suite_recipe.definition import (
    QuantAttentionRecipe,
    QuantLinearRecipe,
    QuantMoERecipe,
    QuantRecipe,
)
from quant_suite_recipe.runtime import QuantRecipeRuntime
from quant_suite_vllm.attention.method import QuantSuiteAttentionMethod
from quant_suite_vllm.linear.method import QuantSuiteLinearMethod
from quant_suite_vllm.moe.method import QuantSuiteMoEMethod
from quant_suite_vllm.quant_config import QUANTIZATION_NAME, QuantSuiteConfig
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod


class _AttentionLayer(torch.nn.Module):
    pass


class _MoELayer(torch.nn.Module):
    def __init__(self, moe_config: object) -> None:
        super().__init__()
        self.moe_config = moe_config


class QuantSuiteConfigTest(unittest.TestCase):
    def write_recipe(self, content: str) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "recipe.yaml"
        path.write_text(content)
        return path

    def test_declares_vllm_contract(self) -> None:
        config = QuantSuiteConfig(QuantRecipe())

        self.assertEqual(config.get_name(), QUANTIZATION_NAME)
        self.assertEqual(config.get_min_capability(), 90)
        self.assertEqual(config.get_config_filenames(), [])
        self.assertEqual(
            config.get_supported_act_dtypes(),
            [torch.float32, torch.float16, torch.bfloat16],
        )
        self.assertEqual(config.get_scaled_act_names(), [])
        self.assertEqual(config.packed_modules_mapping, {})

    def test_deserializes_config_and_json(self) -> None:
        recipe = QuantRecipe(linear_recipe={".*": QuantLinearRecipe()})
        data = {"quant_recipe": recipe.to_dict()}

        from_mapping = QuantSuiteConfig.from_config(data)
        from_json = QuantSuiteConfig.from_config_dict_json(json.dumps(data))

        self.assertEqual(from_mapping.quant_recipe, recipe)
        self.assertEqual(from_json.quant_recipe, recipe)

    def test_from_config_parses_with_the_environment_runtime(self) -> None:
        data = {"linear_recipe": {".*": {"weight": "fake_nvfp4"}}}
        runtime = Mock(spec=QuantRecipeRuntime)
        runtime.deserialize.return_value = QuantRecipe()

        with patch("quant_suite_vllm.quant_config.QUANT_RECIPE_RUNTIME", runtime):
            config = QuantSuiteConfig.from_config({"quant_recipe": data})

        runtime.deserialize.assert_called_once_with(data)
        self.assertIs(config.quant_recipe, runtime.deserialize.return_value)

    def test_from_config_file_parses_with_the_environment_runtime(self) -> None:
        runtime = Mock(spec=QuantRecipeRuntime)
        runtime.load_quant_recipe_from_yaml.return_value = QuantRecipe()

        with patch("quant_suite_vllm.quant_config.QUANT_RECIPE_RUNTIME", runtime):
            config = QuantSuiteConfig.from_config_file("recipe.yaml")

        runtime.load_quant_recipe_from_yaml.assert_called_once_with("recipe.yaml")
        self.assertIs(
            config.quant_recipe, runtime.load_quant_recipe_from_yaml.return_value
        )

    def test_loads_yaml_preserving_selector_order(self) -> None:
        recipe_file = self.write_recipe(
            "attention_recipe:\n"
            "  '^model\\.selected$':\n"
            "    q: fake_mxfp4\n"
            "  '.*':\n"
            "    q: none\n"
        )

        recipe = QuantSuiteConfig.from_config_file(str(recipe_file)).quant_recipe

        self.assertFalse(recipe.get_attention_recipe("model.selected").q.is_noop())
        self.assertTrue(recipe.get_attention_recipe("model.other").q.is_noop())

    def test_yaml_selector_lookup_uses_first_match(self) -> None:
        recipe_file = self.write_recipe(
            "attention_recipe:\n"
            "  '.*':\n"
            "    q: none\n"
            "  '^model\\.selected$':\n"
            "    q: fake_mxfp4\n"
        )

        recipe = QuantSuiteConfig.from_config_file(str(recipe_file)).quant_recipe

        self.assertTrue(recipe.get_attention_recipe("model.selected").q.is_noop())

    def test_explicit_no_quant_attention_survives_vllm_reconstruction(self) -> None:
        recipe_file = self.write_recipe(
            "attention_recipe:\n"
            "  '.*':\n"
            "    q: none\n"
            "    k: none\n"
            "    v: none\n"
            "    attn_weights: none\n"
        )
        loaded = QuantSuiteConfig.from_config_file(str(recipe_file)).quant_recipe
        reconstructed = QuantSuiteConfig.from_config(
            {"quant_recipe": loaded.to_dict()}
        ).quant_recipe

        for recipe in (loaded, reconstructed):
            with self.subTest(recipe=recipe):
                self.assertEqual(recipe.linear_recipe, {})
                self.assertEqual(recipe.moe_recipe, {})
                attention = recipe.get_attention_recipe("model.attention")
                self.assertTrue(attention.requires_custom_attention_backend())
                for field_name in ("q", "k", "v", "attn_weights"):
                    self.assertTrue(getattr(attention, field_name).is_noop())

        config = QuantSuiteConfig(reconstructed)
        moe_config = SimpleNamespace(is_act_and_mul=False, has_bias=False)
        with (
            patch.object(method_provider, "LinearBase", torch.nn.Linear),
            patch.object(method_provider, "_VLLM_MOE_LAYER_TYPE", _MoELayer),
            patch.object(method_provider, "AttentionLayerBase", _AttentionLayer),
        ):
            linear_method = config.get_quant_method(
                torch.nn.Linear(4, 4, bias=False),
                "model.linear",
            )
            moe_method = config.get_quant_method(
                _MoELayer(moe_config),
                "model.moe",
            )
            attention_method = config.get_quant_method(
                _AttentionLayer(),
                "model.attention",
            )

        self.assertIsInstance(linear_method, UnquantizedLinearMethod)
        self.assertIsNone(moe_method)
        self.assertIsInstance(attention_method, QuantSuiteAttentionMethod)

    def test_recurrent_state_only_stays_backend_agnostic_after_reconstruction(
        self,
    ) -> None:
        recipe_file = self.write_recipe(
            "attention_recipe:\n  '.*':\n    recurrent_state: fake_mxfp4\n"
        )
        loaded = QuantSuiteConfig.from_config_file(str(recipe_file)).quant_recipe
        reconstructed = QuantSuiteConfig.from_config(
            {"quant_recipe": loaded.to_dict()}
        ).quant_recipe

        for recipe in (loaded, reconstructed):
            with self.subTest(recipe=recipe):
                attention = recipe.get_attention_recipe("model.attention")
                self.assertFalse(attention.recurrent_state.is_noop())
                self.assertFalse(attention.requires_custom_attention_backend())

    def test_unmatched_yaml_modules_default_to_identity(self) -> None:
        recipe_file = self.write_recipe(
            "linear_recipe:\n"
            "  '^selected\\.linear$':\n"
            "    weight: fake_mxfp4\n"
            "moe_recipe:\n"
            "  '^selected\\.moe$':\n"
            "    w13: fake_mxfp4\n"
            "attention_recipe:\n"
            "  '^selected\\.attention$':\n"
            "    recurrent_state: fake_mxfp4\n"
        )

        recipe = QuantSuiteConfig.from_config_file(str(recipe_file)).quant_recipe

        self.assertTrue(recipe.get_linear_recipe("other.linear").weight.is_noop())
        self.assertTrue(recipe.get_moe_recipe("other.moe").w13.is_noop())
        self.assertTrue(
            recipe.get_attention_recipe("other.attention").recurrent_state.is_noop()
        )

    def test_rejects_empty_yaml_file(self) -> None:
        recipe_file = self.write_recipe("")

        with self.assertRaisesRegex(ValueError, "Empty YAML file"):
            QuantSuiteConfig.from_config_file(str(recipe_file))

    def test_rejects_malformed_yaml_file(self) -> None:
        recipe_file = self.write_recipe("[\n")

        with self.assertRaisesRegex(ValueError, "Invalid YAML file"):
            QuantSuiteConfig.from_config_file(str(recipe_file))

    def test_rejects_non_mapping_yaml_file(self) -> None:
        for content in ("[]\n", "false\n"):
            with self.subTest(content=content):
                recipe_file = self.write_recipe(content)
                with self.assertRaisesRegex(ValueError, "must contain a mapping"):
                    QuantSuiteConfig.from_config_file(str(recipe_file))

    def test_dispatches_linear_moe_and_attention(self) -> None:
        moe_config = SimpleNamespace(is_act_and_mul=True, has_bias=False)
        recipe = QuantRecipe(
            linear_recipe={".*": QuantLinearRecipe()},
            moe_recipe={".*": QuantMoERecipe()},
            attention_recipe={".*": QuantAttentionRecipe()},
        )
        config = QuantSuiteConfig(recipe)

        linear = ReplicatedLinear(
            input_size=4,
            output_size=4,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=config,
            prefix="model.linear",
            return_bias=False,
            disable_tp=True,
        )
        with (
            patch.object(method_provider, "_VLLM_MOE_LAYER_TYPE", _MoELayer),
            patch.object(method_provider, "AttentionLayerBase", _AttentionLayer),
        ):
            moe_method = config.get_quant_method(
                _MoELayer(moe_config),
                "model.moe",
            )
            attention_method = config.get_quant_method(
                _AttentionLayer(),
                "model.attention",
            )

        self.assertIsInstance(linear.quant_method, QuantSuiteLinearMethod)
        self.assertIsInstance(moe_method, QuantSuiteMoEMethod)
        self.assertIsInstance(attention_method, QuantSuiteAttentionMethod)

    def test_returns_none_for_unsupported_layer(self) -> None:
        config = QuantSuiteConfig(
            QuantRecipe(attention_recipe={".*": QuantAttentionRecipe()})
        )

        self.assertIsNone(config.get_quant_method(torch.nn.Identity(), "model.other"))

    def test_linear_without_recipe_uses_unquantized_method(self) -> None:
        config = QuantSuiteConfig(QuantRecipe(linear_recipe={}))
        linear = torch.nn.Linear(4, 4, bias=False)

        with patch.object(method_provider, "LinearBase", torch.nn.Linear):
            quant_method = config.get_quant_method(linear, "model.linear")

        self.assertIsInstance(quant_method, UnquantizedLinearMethod)

    def test_rejects_non_mapping_recipe(self) -> None:
        for value in (QuantRecipe(), "recipe.yaml", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    TypeError, "quant_recipe must be a mapping"
                ):
                    QuantSuiteConfig.from_config({"quant_recipe": value})

    def test_accepts_the_quant_method_vllm_sets(self) -> None:
        recipe = QuantRecipe(linear_recipe={".*": QuantLinearRecipe()})
        config = QuantSuiteConfig.from_config(
            {"quant_method": "quant_suite", "quant_recipe": recipe.to_dict()}
        )
        self.assertEqual(config.quant_recipe, recipe)

    def test_rejects_unknown_keys(self) -> None:
        data = {"quant_recipe": {}, "skip_modules": [], "quant_recipes": {}}
        with self.assertRaisesRegex(
            ValueError, r"keys \['quant_recipes', 'skip_modules'\]"
        ):
            QuantSuiteConfig.from_config(data)
        with self.assertRaisesRegex(ValueError, "skip_modules"):
            QuantSuiteConfig.from_config_dict_json(json.dumps(data))

    def test_rejects_non_object_json(self) -> None:
        with self.assertRaisesRegex(ValueError, "must contain an object"):
            QuantSuiteConfig.from_config_dict_json("[]")

    def test_attention_registration_defaults_to_mha(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(
                plugin,
                "register_backend",
            ) as register_backend,
        ):
            plugin.register_attention_backend()

        register_backend.assert_called_once_with(
            plugin.AttentionBackendEnum.CUSTOM,
            "quant_suite_vllm.attention.backend.QuantSuiteAttentionBackend",
        )


class QuantSuitePluginTest(unittest.TestCase):
    def test_registration_is_idempotent(self) -> None:
        was_registered = plugin._registered
        self.addCleanup(setattr, plugin, "_registered", was_registered)
        plugin._registered = False
        quant_config = SimpleNamespace(QuantSuiteConfig=QuantSuiteConfig)
        kda = Mock()
        modules = {
            "quant_suite_recipe.runtime_provider": Mock(),
            "quant_suite_vllm.quant_config": quant_config,
            "quant_suite_vllm.kda": kda,
        }
        with (
            patch.object(
                plugin,
                "import_module",
                side_effect=modules.__getitem__,
            ) as import_module,
            patch.object(plugin, "register_attention_backend") as register_backend,
        ):
            plugin.register()
            plugin.register()

        self.assertEqual(
            [call.args[0] for call in import_module.call_args_list], list(modules)
        )
        register_backend.assert_called_once_with()
        kda.register_kda_quant_method.assert_called_once_with(QuantSuiteConfig)
