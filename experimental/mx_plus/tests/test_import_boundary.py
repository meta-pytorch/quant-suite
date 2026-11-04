# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The package imports only documented public QuantSuite modules.

An experimental package is a worked example for contributors, so it must be
buildable from the public surface alone: no private names and no other
experimental package.
"""

import ast
import unittest
from pathlib import Path

import mx_plus

_PACKAGE = "mx_plus"

# Documented public QuantSuite modules an extension may import from.
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
        "quant_suite_recipe.runtime",
    }
)

_OTHER_ALLOWED_ROOTS: frozenset[str] = frozenset(
    {"dataclasses", "torch", "types", "typing"}
)


def _imports(source: str) -> list[tuple[str, list[str]]]:
    """(module, imported names) for every import statement in ``source``."""
    found: list[tuple[str, list[str]]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.extend((alias.name, []) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative imports are not used here"
            found.append((node.module or "", [alias.name for alias in node.names]))
    return found


class ImportBoundaryTest(unittest.TestCase):
    def test_imports_only_public_modules(self) -> None:
        package_dir = Path(mx_plus.__file__).parent
        sources = sorted(package_dir.glob("*.py"))
        self.assertIn("mx_plus.py", [path.name for path in sources])
        for path in sources:
            for module, names in _imports(path.read_text()):
                with self.subTest(file=path.name, module=module):
                    root = module.split(".")[0]
                    if root == _PACKAGE or root in _OTHER_ALLOWED_ROOTS:
                        continue
                    self.assertIn(module, _ALLOWED_QUANT_SUITE_MODULES)
                    private = [name for name in names if name.startswith("_")]
                    self.assertEqual(private, [])


if __name__ == "__main__":
    unittest.main()
