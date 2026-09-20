"""KServe v2 HTTP paths. Format with model= and optionally version=."""

from enum import StrEnum


class Endpoint(StrEnum):
    """KServe v2 HTTP path templates."""

    SERVER_LIVE = "/v2/health/live"
    SERVER_READY = "/v2/health/ready"
    SERVER_METADATA = "/v2"
    MODEL_READY = "/v2/models/{model}/ready"
    MODEL_METADATA = "/v2/models/{model}"
    MODEL_CONFIG = "/v2/models/{model}/config"
    MODEL_INFER = "/v2/models/{model}/infer"
    MODEL_GENERATE = "/v2/models/{model}/generate"
    MODEL_GENERATE_STREAM = "/v2/models/{model}/generate_stream"
    REPOSITORY_INDEX = "/v2/repository/index"
    METRICS = "/metrics"


def versioned(endpoint: Endpoint, model: str, version: str | None = None) -> str:
    """Format an :class:`Endpoint` template and insert a version segment when given.

    :param endpoint: Path template containing a ``{model}`` placeholder.
    :param model: Model name to substitute into the template.
    :param version: Optional model version; when set, rewrites ``/models/{model}`` to
        ``/models/{model}/versions/{version}``.
    :return: The fully formatted path.
    """
    path = endpoint.format(model=model)
    if version:
        path = path.replace(f"/models/{model}", f"/models/{model}/versions/{version}", 1)
    return path
