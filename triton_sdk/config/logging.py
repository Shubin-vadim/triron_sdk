"""Opt-in logging setup for examples, scripts and tests. The library itself never calls this."""

from __future__ import annotations

import logging

from ..core.constants import LOG_COLORS
from ..core.optional import try_import


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger with a colored (or plain) console handler.

    Replaces any existing root handlers and quiets the noisy ``httpx``, ``httpcore`` and ``grpc``
    loggers to ``WARNING``. Intended for examples and scripts, not for library code.

    :param level: Log level name (e.g. ``"DEBUG"``, ``"INFO"``) to set on the root logger.
    """
    handler: logging.Handler
    colorlog = try_import("colorlog")
    if colorlog is not None:
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s %(levelname)-8s %(name)s: %(message)s", log_colors=LOG_COLORS
            )
        )
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "grpc"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
