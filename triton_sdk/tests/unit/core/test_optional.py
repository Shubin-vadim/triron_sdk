import json

import pytest

from triton_sdk.core.optional import require_import, try_import
from triton_sdk.exceptions import MissingDependencyError


def test_require_import_returns_module_when_installed() -> None:
    assert require_import("json", "irrelevant") is json


def test_require_import_raises_with_install_hint_when_missing() -> None:
    with pytest.raises(MissingDependencyError) as info:
        require_import("definitely_not_a_real_module_xyz", "chat")
    assert "definitely_not_a_real_module_xyz" in info.value.message
    assert "pip install triton-sdk[chat]" in info.value.message


def test_try_import_returns_module_when_installed() -> None:
    assert try_import("json") is json


def test_try_import_returns_none_when_missing() -> None:
    assert try_import("definitely_not_a_real_module_xyz") is None
