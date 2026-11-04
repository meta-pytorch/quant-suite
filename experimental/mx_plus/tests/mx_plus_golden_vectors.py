# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The golden-vector refactor check over ``MX_PLUS_PRESETS``.

Same tool, same inputs, same digest format as the check in ``recipe/tests`` --
see its ``golden_vectors`` module docstring for how it works and why it is a
hand-run tool and not a test. The only difference is the suite: these presets
enable ``MxPlus``, so ``QuantConfig.from_dict`` cannot build them; this
package's ``RUNTIME`` builder does.

Running it
----------
With ``quant_suite_core``, ``quant_suite_recipe`` and this package installed,
put ``recipe/tests`` (which holds the harness) on the path and run the module
from this directory, on a host with a CUDA device::

    cd experimental/mx_plus/tests
    PYTHONPATH=../../../recipe/tests python mx_plus_golden_vectors.py check
    PYTHONPATH=../../../recipe/tests python mx_plus_golden_vectors.py record
"""

from __future__ import annotations

import mx_plus_golden_vectors_data
from golden_vectors import GoldenSuite, run
from mx_plus.presets import MX_PLUS_PRESETS
from mx_plus.runtime import RUNTIME

MX_PLUS_SUITE: GoldenSuite = GoldenSuite(
    presets=MX_PLUS_PRESETS,
    build_config=RUNTIME.config_builder.build,
    table=mx_plus_golden_vectors_data,
    header=(
        "# Copyright (c) Meta Platforms, Inc. and affiliates.\n"
        "# All rights reserved.\n"
        "#\n"
        "# This source code is licensed under the BSD-style license found in the\n"
        "# LICENSE file in the root directory of this source tree.\n"
    ),
)


def main() -> None:
    run(MX_PLUS_SUITE, prog="mx_plus_golden_vectors")


if __name__ == "__main__":
    main()
