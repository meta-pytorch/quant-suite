# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Records linear, attention and MoE tensors of a QuantSuite model run by vLLM.

- ``tensor_dump.recipe.DumpRecipe``: the ``dump`` tensor recipe type, an
  identity stage whose options select the tensors a linear layer writes.
- ``tensor_dump.executors.DumpLinear``: the linear executor that writes them.
- ``tensor_dump.runtime.RUNTIME``: the built-in recipe runtime extended with
  both, published in the ``quant_suite.recipe_runtimes`` entry-point group.

Attention and MoE layers are recorded by forward hooks on vLLM's ``Attention``
and ``MoERunner`` layers, selected with environment variables.

``tensor_dump.merge`` merges a dump into one file per request, phase and
layer (``python -m tensor_dump.merge SRC DST``).

``tensor_dump.reader.TensorDump`` reads a dump back, as written or merged:
steps, requests, linear operands, attention inputs with each request's
rebuilt key/value history, and MoE tensors with each layer's layout.
"""

__version__ = "0.1.0"
