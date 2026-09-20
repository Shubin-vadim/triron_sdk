"""Helpers for importing optional third-party dependencies (grpc/chat extras, dev-only niceties)."""

from __future__ import annotations

import importlib
from types import ModuleType

from ..exceptions import MissingDependencyError


def require_import(name: str, extra: str) -> ModuleType:
    """Import an optional dependency, or fail with an actionable install hint.

    Use for a dependency the caller cannot proceed without (e.g. ``grpcio`` for
    :class:`~triton_sdk.transport.grpc.GrpcTransport`, ``transformers`` for
    :class:`~triton_sdk.chat.HFChatTemplate`).

    :param name: Module name to import (e.g. ``"transformers"``).
    :param extra: Name of the triton-sdk extra that installs it (e.g. ``"chat"``), used to build
        the install hint in the resulting error.
    :return: The imported module.
    :raises MissingDependencyError: If the module is not installed.
    """
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise MissingDependencyError(name, extra) from exc


def try_import(name: str) -> ModuleType | None:
    """Import a module if available, without raising when it is not installed.

    Use for a dependency that only improves behaviour when present (e.g. ``colorlog`` for nicer
    console output), where the caller has a reasonable fallback.

    :param name: Module name to import.
    :return: The imported module, or ``None`` if it is not installed.
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        return None
