# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Records linear and attention tensors of a QuantSuite model run by vLLM.

- ``tensor_dump.recipe.DumpRecipe``: the ``dump`` tensor recipe type, an
  identity stage whose options select the tensors a linear layer writes.
- ``tensor_dump.executors.DumpLinear``: the linear executor that writes them.
- ``tensor_dump.runtime.RUNTIME``: the built-in recipe runtime extended with
  both, published in the ``quant_suite.recipe_runtimes`` entry-point group.

Attention layers are recorded by forward hooks on vLLM's ``Attention`` layers,
selected with environment variables.
"""

__version__ = "0.1.0"
