# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Records linear and attention tensors of a QuantSuite model.

- ``tensor_dump.recipe.DumpRecipe``: the ``dump`` tensor recipe type, an
  identity stage whose options select the tensors a layer writes.
- ``tensor_dump.executors.DumpLinear`` and ``DumpAttention``: executors that
  write the selected tensors.
- ``tensor_dump.runtime.RUNTIME``: the built-in recipe runtime extended with
  both, published in the ``quant_suite.recipe_runtimes`` entry-point group.
"""

__version__ = "0.1.0"
