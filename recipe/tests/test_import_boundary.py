# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import subprocess
import sys
import unittest


class ImportBoundaryTest(unittest.TestCase):
    def test_import_does_not_load_mslk(self) -> None:
        blocked_prefixes = ("mslk",)
        child_code = """
import importlib
import json
import sys

sys.path[:] = json.loads(sys.argv[1])
blocked_prefixes = tuple(json.loads(sys.argv[2]))
importlib.import_module("quant_suite_recipe.definition")
importlib.import_module("quant_suite_recipe.api")
loaded = sorted(
    name
    for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in blocked_prefixes)
)
print(json.dumps(loaded))
"""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                child_code,
                json.dumps(sys.path),
                json.dumps(blocked_prefixes),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(json.loads(result.stdout), [])
