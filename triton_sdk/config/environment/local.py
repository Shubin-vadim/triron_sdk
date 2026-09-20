"""Local defaults for examples / integration tests. Sets a variable only if it is not already defined."""

from __future__ import annotations

import os

from dotenv import load_dotenv


LOCAL_DEFAULTS: dict[str, str] = {
    "TRITON_URL": "localhost:8000",
    "TRITON_PROTOCOL": "http",
    "TRITON_DEFAULT_MODEL": "vllm_model",
    "TRITON_LOG_LEVEL": "DEBUG",
}


def apply_local_env() -> None:
    """Load a local .env file and fill in any unset TRITON_* defaults from :data:`LOCAL_DEFAULTS`.

    Existing environment variables are never overwritten.
    """
    load_dotenv()
    for key, value in LOCAL_DEFAULTS.items():
        os.environ.setdefault(key, value)
