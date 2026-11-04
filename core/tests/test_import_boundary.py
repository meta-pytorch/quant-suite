# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Permanent guards on the public surface.

Three invariants:
  (a) `quant_suite_core` resolves as a real module, not an implicit namespace
      package, and its `__all__` is exactly what it claims to export.
  (b) An extension built from nothing but documented public modules actually
      works: the root facade, `quant_suite_core.extension`,
      `quant_suite_core.quant_config` and `quant_suite_core.fake_presets`.
  (c) `quant_suite_core` imports no other QuantSuite package: not recipe or
      vllm, and no extension package, including the ones under
      `experimental/`. Extensions depend on core, never the reverse.
"""

from __future__ import annotations

import ast
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional

import torch
from quant_suite_core import fake_quantize, QuantConfig
from quant_suite_core.extension import NVFP4AmaxRule, QuantConfigExtension
from quant_suite_core.fake_presets import FAKE_PRESETS
from quant_suite_core.quant_config import DEFAULT_QUANT_CONFIG_BUILDER


class FacadeModuleTest(unittest.TestCase):
    """Guards the failure a namespace package makes invisible: every existing
    test imports submodules directly, so nothing else would have noticed the
    facade shipping nowhere.
    """

    def test_facade_is_a_real_module_with_a_complete_all(self) -> None:
        import quant_suite_core

        self.assertIsNotNone(quant_suite_core.__file__)
        assert quant_suite_core.__file__ is not None
        self.assertTrue(quant_suite_core.__file__.endswith("__init__.py"))

        self.assertEqual(len(quant_suite_core.__all__), 15)
        for name in quant_suite_core.__all__:
            getattr(quant_suite_core, name)
            if name != "__version__":
                self.assertFalse(name.startswith("_"), name)


class NoReverseImportTest(unittest.TestCase):
    def test_core_imports_no_other_quant_suite_package(self) -> None:
        import quant_suite_core

        assert quant_suite_core.__file__ is not None
        sources = sorted(Path(quant_suite_core.__file__).parent.glob("*.py"))
        self.assertIn("quant_config.py", [path.name for path in sources])
        for path in sources:
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    modules = [node.module or ""]
                else:
                    continue
                for module in modules:
                    root = module.split(".")[0]
                    if root.startswith("quant_suite"):
                        self.assertEqual(
                            root, "quant_suite_core", f"{path.name}: {module}"
                        )


class PublicExtensionPointTest(unittest.TestCase):
    """An extension built from documented public modules must work end-to-end.

    A real extension may reach for private helpers, which proves nothing
    about the public path. This toy `NVFP4AmaxRule`, shipped as a
    `QuantConfigExtension` and parsed by a `QuantConfigBuilder`, is constructed
    from nothing but the documented public modules.
    """

    def test_toy_amax_rule_runs_through_fake_quantize(self) -> None:
        x = torch.randn(8, 64)

        plain = fake_quantize(
            x, "activation", _toy_config(use_half_max_pos=False), use_triton=False
        )
        ruled = fake_quantize(
            x, "activation", _toy_config(use_half_max_pos=True), use_triton=False
        )

        self.assertTrue(torch.isfinite(ruled).all())
        self.assertFalse(torch.equal(plain, ruled))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
    def test_the_same_rule_object_serves_both_backends(self) -> None:
        """The rule defines one method, and both backends accept it unchanged.

        This is the property the single-method protocol buys: an extension
        author writes `amax` once. A per-backend protocol would make an
        eager-only author supply a stub for the form they never call.
        """
        x = torch.randn(8, 64, device="cuda")
        cfg = _toy_config(use_half_max_pos=True)

        eager = fake_quantize(x, "activation", cfg, use_triton=False)
        fused = fake_quantize(x, "activation", cfg, use_triton=True)

        self.assertTrue(torch.isfinite(eager).all())
        self.assertTrue(torch.isfinite(fused).all())


class _HalfMaxPosRule:
    """Toy `NVFP4AmaxRule`: places each block's amax at half the
    element-format max instead of the format max itself. A synthetic
    "extra headroom" rule, not a real numerics recipe -- it exists only to
    prove the extension point is reachable from outside the package.

    One method, and nothing in it is backend-specific.
    """

    def amax(
        self, blocks: torch.Tensor, max_pos: float
    ) -> tuple[torch.Tensor, float, torch.Tensor]:
        block_amax = blocks.abs().amax(dim=1)
        local_amax = block_amax / (max_pos / 2.0)
        # Reduced from `local_amax`, not from the raw tensor, so the per-block
        # choice above reaches the global level too. That coupling is why both
        # come back from one call.
        return local_amax, 1.0, local_amax.amax().reshape(1)


@dataclass(frozen=True)
class _HalfMaxPos(QuantConfigExtension):
    """Binds `_HalfMaxPosRule` as an extension, using nothing outside the
    documented public modules."""

    name: ClassVar[str] = "toy_half_max_pos"

    def amax_rule(self, config: QuantConfig) -> Optional[NVFP4AmaxRule]:
        return _HalfMaxPosRule()


_TOY_BUILDER = DEFAULT_QUANT_CONFIG_BUILDER.extend((_HalfMaxPos,))


def _toy_config(use_half_max_pos: bool) -> QuantConfig:
    extensions = {"toy_half_max_pos": {}} if use_half_max_pos else {}
    return _TOY_BUILDER.build({**FAKE_PRESETS["nvfp4"], "extensions": extensions})
