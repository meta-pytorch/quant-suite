# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The per-process dump session, created from the environment on first use."""

from __future__ import annotations

import logging
import os

from tensor_dump import _step
from tensor_dump._naming import _NameResolver
from tensor_dump._options import _DIR_ENV, _DumpOptions, _options_from_environ
from tensor_dump._step import _StepTracker, _VllmSettings
from tensor_dump._writer import _Writer

logger: logging.Logger = logging.getLogger(__name__)


class _Session:
    """The options, step tracker, writer and name resolver of one process.

    Raises:
        RuntimeError: if vLLM's V2 model runner is enabled.
        NotImplementedError: if speculative decoding is configured.
    """

    def __init__(self, options: _DumpOptions, settings: _VllmSettings) -> None:
        if settings.v2_model_runner:
            raise RuntimeError(
                "tensor_dump does not support vLLM's V2 model runner, whose "
                "warmup runs real forwards that cannot be told apart from "
                "requests; launch vLLM with VLLM_USE_V2_MODEL_RUNNER=0"
            )
        if settings.speculative:
            raise NotImplementedError(
                "tensor_dump does not support speculative decoding: draft-model "
                "forwards would take step ids"
            )
        self.options = options
        self.settings = settings
        self.resolver = _NameResolver()
        self.writer = _Writer(options)
        self.tracker = _StepTracker(
            options,
            on_first_step=self.resolver.finish,
            on_selected_step=self.writer.begin_step,
        )


_session: _Session | None = None
_loaded: bool = False
_warned_disabled: bool = False


def _get() -> _Session | None:
    """The session, or ``None`` when dumping is disabled.

    The vLLM settings are read when the session is created, which is when
    the first executor with a ``dump`` stage is built, while vLLM loads the
    model.

    Raises:
        ValueError: on the first call, for a malformed environment variable.
        RuntimeError, NotImplementedError: for an unsupported vLLM setting; see
            ``_Session``.
    """
    global _session, _loaded
    if not _loaded:
        options = _options_from_environ(os.environ)
        _session = (
            None if options is None else _Session(options, _step._vllm_settings())
        )
        _loaded = True
    return _session


def _warn_disabled() -> None:
    """Log once that a recipe has ``dump`` stages while dumping is disabled."""
    global _warned_disabled
    if not _warned_disabled:
        _warned_disabled = True
        logger.warning("recipe has dump stages but %s is unset", _DIR_ENV)


def _reset() -> None:
    """Drop the session so the next ``_get`` reads the environment again.

    For tests.
    """
    global _session, _loaded, _warned_disabled
    if _session is not None:
        _session.resolver.finish()
        _session.writer.close()
    _session = None
    _loaded = False
    _warned_disabled = False
