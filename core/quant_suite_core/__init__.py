# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tensor-level quantization building blocks.

Entry points: ``QuantConfig.from_dict`` turns a mapping into a ``QuantConfig``; ``fake_quantize`` quantizes and dequantizes a tensor with it;
``native_quantize`` / ``native_dequantize`` convert to and from packed MXFP4,
MXFP8 and NVFP4 layouts. The QDQ backends, preset tables and extension API
live in their own modules (see the package README).
"""

from quant_suite_core.fake_quant import fake_quantize
from quant_suite_core.native_quant import (
    native_dequantize,
    native_quantize,
    NativeQuantResult,
)
from quant_suite_core.quant_config import (
    BlockSize,
    ElemDtype,
    ElemRoundMode,
    GlobalScaleGranularity,
    QuantConfig,
    QuantInfo,
    QuantMode,
    ScaleDtype,
    ScaleNormMode,
    ScaleRoundMode,
)

__version__ = "0.1.0"

__all__ = [
    "BlockSize",
    "ElemDtype",
    "ElemRoundMode",
    "GlobalScaleGranularity",
    "NativeQuantResult",
    "QuantConfig",
    "QuantInfo",
    "QuantMode",
    "ScaleDtype",
    "ScaleNormMode",
    "ScaleRoundMode",
    "__version__",
    "fake_quantize",
    "native_dequantize",
    "native_quantize",
]
