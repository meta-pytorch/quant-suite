# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The package imports only documented public QuantSuite modules, and vLLM
only inside functions.

An experimental package must be buildable from the public surface alone: no
private names and no other experimental package. The recipe runtime is loaded
in every process of an environment, including ones without vLLM, so no module
imports vLLM at import time.
"""

import ast
import unittest
from pathlib import Path

import tensor_dump

_PACKAGE = "tensor_dump"

# Documented public QuantSuite modules the package may import from. The two
# executor modules define the executor base classes a runtime is built with.
_ALLOWED_QUANT_SUITE_MODULES: frozenset[str] = frozenset(
    {
        "quant_suite_core",
        "quant_suite_core.extension",
        "quant_suite_core.fake_presets",
        "quant_suite_core.quant_config",
        "quant_suite_recipe.api",
        "quant_suite_recipe.api.definition",
        "quant_suite_recipe.api.implementation",
        "quant_suite_recipe.definition",
        "quant_suite_recipe.implementation.attention",
        "quant_suite_recipe.implementation.linear",
        "quant_suite_recipe.runtime",
    }
)

_ALLOWED_VLLM_MODULES: frozenset[str] = frozenset(
    {
        "vllm.config",
        "vllm.distributed.parallel_state",
        "vllm.envs",
        "vllm.forward_context",
    }
)

_OTHER_ALLOWED_ROOTS: frozenset[str] = frozenset(
    {
        "__future__",
        "collections",
        "dataclasses",
        "importlib",
        "json",
        "logging",
        "os",
        "pathlib",
        "re",
        "torch",
        "types",
        "typing",
        "weakref",
    }
)


def _imports(nodes: list[ast.AST]) -> list[tuple[str, list[str]]]:
    """(module, imported names) for every import statement in ``nodes``."""
    found: list[tuple[str, list[str]]] = []
    for node in nodes:
        if isinstance(node, ast.Import):
            found.extend((alias.name, []) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative imports are not used here"
            found.append((node.module or "", [alias.name for alias in node.names]))
    return found


def _sources() -> list[Path]:
    """The package sources: the source tree beside the tests when present, so
    that modules a test environment does not install are scanned too."""
    source_dir = Path(__file__).resolve().parents[1] / _PACKAGE
    if not source_dir.is_dir():
        source_dir = Path(tensor_dump.__file__).parent
    return sorted(source_dir.glob("*.py"))


class ImportBoundaryTest(unittest.TestCase):
    def test_imports_only_public_modules(self) -> None:
        sources = _sources()
        self.assertIn("executors.py", [path.name for path in sources])
        for path in sources:
            tree = ast.parse(path.read_text())
            for module, names in _imports(list(ast.walk(tree))):
                with self.subTest(file=path.name, module=module):
                    root = module.split(".")[0]
                    if root == _PACKAGE or root in _OTHER_ALLOWED_ROOTS:
                        continue
                    if root == "vllm":
                        self.assertIn(module, _ALLOWED_VLLM_MODULES)
                        continue
                    self.assertIn(module, _ALLOWED_QUANT_SUITE_MODULES)
                    private = [name for name in names if name.startswith("_")]
                    self.assertEqual(private, [])

    def test_no_module_level_vllm_import(self) -> None:
        sources = _sources()
        self.assertIn("_step.py", [path.name for path in sources])
        for path in sources:
            tree = ast.parse(path.read_text())
            for module, _ in _imports(list(tree.body)):
                with self.subTest(file=path.name, module=module):
                    self.assertNotEqual(module.split(".")[0], "vllm")


if __name__ == "__main__":
    unittest.main()
