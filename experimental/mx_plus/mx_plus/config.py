# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``MxPlus``: the MX+ rule as a ``QuantConfigExtension``.

Depends on ``quant_suite_core`` only, so a core-only user can enable it on a
``QuantConfigBuilder`` without importing ``quant_suite_recipe``::

    from quant_suite_core.quant_config import DEFAULT_QUANT_CONFIG_BUILDER
    from mx_plus.config import MxPlus
    from mx_plus.presets import MX_PLUS_PRESETS

    builder = DEFAULT_QUANT_CONFIG_BUILDER.extend((MxPlus,), MX_PLUS_PRESETS)
    config = builder.build_preset("mxfp4_plus2")
"""

from dataclasses import dataclass
from typing import ClassVar, Optional

import torch
from mx_plus.mx_plus import apply_mx_plus, MAX_EXTRA_MANTISSA_BITS
from quant_suite_core.extension import PostProcessPhase, QuantConfigExtension
from quant_suite_core.quant_config import (
    GlobalScaleGranularity,
    QuantConfig,
    ScaleDtype,
)


@dataclass(frozen=True)
class MxPlus(QuantConfigExtension):
    """MX+: keep extra mantissa bits on the amax element of each block.

    Enabled in a config dict as
    ``"extensions": {"mx_plus": {"extra_mantissa_bits": 2}}``.

    Attributes:
        extra_mantissa_bits: Mantissa bits the amax element keeps beyond the
            element format's own, from 1 to ``MAX_EXTRA_MANTISSA_BITS``.
    """

    name: ClassVar[str] = "mx_plus"
    extra_mantissa_bits: int = 2

    def validate(self, config: QuantConfig) -> None:
        # Power-of-two scaling preserves the extra mantissa bits; MX+ is only
        # tested with E8M0.
        if config.scale_dtype != ScaleDtype.E8M0:
            raise ValueError(
                "MX+ requires E8M0 scale dtype (power-of-two scaling "
                f"preserves mantissa bits), but got {config.scale_dtype}"
            )
        if (
            config.global_scale_granularity != GlobalScaleGranularity.NONE
            or config.global_scale_dtype != ScaleDtype.NONE
        ):
            raise ValueError(
                "MX+ requires global_scale_granularity=NONE and "
                "global_scale_dtype=NONE (no MBS or global scale), but got "
                f"global_scale_granularity={config.global_scale_granularity}, "
                f"global_scale_dtype={config.global_scale_dtype}"
            )
        bits = self.extra_mantissa_bits
        if (
            not isinstance(bits, int)
            or isinstance(bits, bool)
            or not 1 <= bits <= MAX_EXTRA_MANTISSA_BITS
        ):
            raise ValueError(
                "MX+ requires extra_mantissa_bits to be an int from 1 to "
                f"{MAX_EXTRA_MANTISSA_BITS}, but got {bits!r}"
            )

    def post_process(
        self, config: QuantConfig, phase: PostProcessPhase
    ) -> Optional[torch.Tensor]:
        return apply_mx_plus(config, phase, self.extra_mantissa_bits)
