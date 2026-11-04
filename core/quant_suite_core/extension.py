# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The extension-author API of ``quant_suite_core``.

This module is the documented home for everything a package needs to extend
quantization behaviour without touching the QDQ backends. It defines
``QuantConfigExtension``, the unit an extension package ships: one optional
rule with typed options, enabled on a config through ``QuantConfig.extensions``
and made parseable by adding its class to a ``QuantConfigBuilder``.

It also contains the hook protocols the backends hand to an extension's hooks:

- ``ScalePhase``: a backend's scaling-factor phase, to which the config
  applies the rule returned by ``QuantConfigExtension.amax_rule``.
- ``NVFP4AmaxRule``: a replacement for NVFP4's two-level amax derivation,
  applied through ``ScalePhase.with_amax_rule``.
- ``PostProcessPhase``: a backend's working state at the end of its
  post-process phase, passed to ``QuantConfigExtension.post_process``.
- ``DispatchPhase``: ``fake_quantize``'s state once a backend has returned,
  passed to ``QuantConfigExtension.finalize``.
"""

from typing import ClassVar, Optional, Protocol, TYPE_CHECKING, TypeVar, Union

import torch

if TYPE_CHECKING:
    from quant_suite_core.quant_config import QuantConfig, QuantInfo  # @manual


_DerivedScale = TypeVar("_DerivedScale")


class NVFP4AmaxRule(Protocol):
    """Computes NVFP4's two-level amax in place of the default derivation.

    The default builds the per-block scale from ``local_amax / max_pos`` with
    ``max_pos`` the scalar element-format max, then reduces the tensor-global
    amax from the result. A rule here replaces that whole step and returns the
    three quantities the NVFP4 scale is built from -- the same three
    ``calculate_fp8_scale_for_nvfp4`` takes.

    One method, not one per backend. Both levels come from one call because
    choosing a per-block clip changes what the global reduction should see;
    returning them separately would oblige each backend to re-derive the global
    consistently, which is a correctness burden the signature can simply remove.
    """

    def amax(
        self, blocks: torch.Tensor, max_pos: float
    ) -> tuple[torch.Tensor, float, torch.Tensor]:
        """``(local_amax, local_max_pos, global_amax)``.

        ``blocks`` is ``(num_blocks, block_numel)`` in fp32; ``max_pos`` is the
        element-format max the default would have used.

        ``local_amax`` is the per-block amax the scale is built from, shape
        ``(num_blocks,)``, and ``local_max_pos`` the magnitude it is placed on
        -- a rule that folds its choice into the amax returns 1.0 here.
        ``global_amax``, shape ``(1,)``, is reduced from ``local_amax`` and not
        from the raw tensor, so the rule's per-block choice reaches it.
        """
        ...


class ScalePhase(Protocol[_DerivedScale]):
    """A QDQ backend's scaling-factor phase: the inputs, and ``derive`` to finish it.

    The hook *owns* this phase rather than adjusting its result. A scale rule
    reaches both levels of the scale and can change how the global one is
    *derived*, not just its value -- see ``NVFP4AmaxRule`` -- which an
    adjust-the-result hook could not express. A rule that needs neither may
    replace ``derive`` outright; the hook returns whatever it produced.

    The concrete type differs per backend -- ``qdq`` derives the per-block scale
    itself, ``qdq_triton_mega`` leaves it to its kernel -- which is why applying
    a rule is a method here and not arithmetic on the phase's fields. What the
    rule returns is the same for both.
    """

    def with_amax_rule(self, rule: NVFP4AmaxRule) -> "ScalePhase[_DerivedScale]": ...

    def derive(self) -> _DerivedScale: ...


class PostProcessPhase(Protocol):
    """A QDQ backend's working state at the end of its post-process phase.

    ``data`` is the tensor the backend is about to unwind (unscale and reshape
    back); a hook returns the replacement.

    The members below are what a *per-block element correction* needs, which is
    the only shape of post-process step either backend supports: read the blocks
    and the scale they were quantized with, decide one replacement element per
    block, write it back. Each of the three is something only the backend can
    answer, and none of them names the rule -- the rule is the hook's.

    That split is what keeps the domain difference out of the rule. ``qdq`` is
    still in the element domain, so its ``scatter_block_elements`` writes low-
    precision values that pick up the shared unscale and NaN-poisoning tail;
    ``qdq_triton_mega`` is already in the dequantized domain, so its own
    implementation applies the unscale as it writes. ``block_scale`` is likewise
    backend knowledge: ``qdq`` hands back the scale its main path computed, while
    the mega kernel does not expose one and its phase reconstructs it.
    """

    data: torch.Tensor
    # The element format's saturation bounds, as the backend applied them.
    min_pos: float
    max_pos: float

    def block_view(self) -> torch.Tensor:
        """The high-precision blocks, shape ``(num_blocks, block_numel)``."""
        ...

    def block_scale(self) -> torch.Tensor:
        """The per-block scale ``data`` was quantized with, shape ``(num_blocks,)``."""
        ...

    def scatter_block_elements(
        self, index: torch.Tensor, value_lp: torch.Tensor
    ) -> torch.Tensor:
        """Replace one element per block, given its position within the block.

        ``index`` and ``value_lp`` are both ``(num_blocks,)``; ``value_lp`` is in
        the quantized (element) domain and the backend converts it into whatever
        domain ``data`` is in.
        """
        ...


class DispatchPhase(Protocol):
    """``fake_quantize``'s state once a backend has returned.

    The counterpart of ``PostProcessPhase`` one level up. A step here sees the
    dequantized output at its original shape and orientation, with padding and
    the transpose already unwound, and runs exactly once per ``fake_quantize``
    -- neither of which holds inside a backend, since ``qdq`` re-enters itself
    through ``_mbs_factor_dynamic`` and both backends are also called directly.

    ``with_dequantized`` exists because the call may be returning either a bare
    tensor or a ``QuantInfo``; the phase owns that distinction so the step does
    not have to.
    """

    source: torch.Tensor
    result: Union[torch.Tensor, "QuantInfo"]

    @property
    def dequantized(self) -> torch.Tensor:
        """The dequantized output, at the caller's shape."""
        ...

    def with_dequantized(
        self, dequantized: torch.Tensor
    ) -> Union[torch.Tensor, "QuantInfo"]:
        """``result`` with its dequantized output replaced."""
        ...


class QuantConfigExtension:
    """One optional rule layered on ``QuantConfig``.

    Declare an extension as a frozen dataclass subclass. The class attribute
    ``name`` identifies it and the dataclass fields are its typed options::

        @dataclass(frozen=True)
        class ExtraBits(QuantConfigExtension):
            name: ClassVar[str] = "extra_bits"
            bits: int = 2

            def validate(self, config): ...

    A builder knows the class, and a config carries an instance: two instances
    are equal exactly when their options are. A config dict enables an
    extension under the ``"extensions"`` key, as ``{"extra_bits": {"bits": 2}}``,
    and ``to_dict`` writes it back the same way. Options are plain values
    (bool, int, float, str).

    Class attributes:
        name: Unique among the extensions a builder knows. Lower-case letters,
            digits and underscores, starting with a letter.

    A config enables at most one extension per hook. A hook counts as used
    when the subclass overrides it.
    """

    name: ClassVar[str]

    def validate(self, config: "QuantConfig") -> None:
        """Raise ``ValueError`` if ``config`` cannot carry this rule.

        Runs at the end of ``QuantConfig.__post_init__``, after the base
        checks.
        """
        return None

    def amax_rule(self, config: "QuantConfig") -> Optional[NVFP4AmaxRule]:
        """The ``NVFP4AmaxRule`` the config applies to the backend's
        ``ScalePhase``, or None for none."""
        return None

    def post_process(
        self, config: "QuantConfig", phase: PostProcessPhase
    ) -> Optional[torch.Tensor]:
        """The tensor that replaces ``phase.data``, or None to leave it."""
        return None

    def finalize(
        self, config: "QuantConfig", phase: DispatchPhase
    ) -> Optional[Union[torch.Tensor, "QuantInfo"]]:
        """The value that replaces ``phase.result``, or None to leave it."""
        return None
