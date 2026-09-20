"""grpc.StatusCode names -> HTTP-like codes, used to unify error reporting across transports."""

GRPC_STATUS_TO_HTTP: dict[str, int] = {
    "OK": 200,
    "CANCELLED": 499,
    "INVALID_ARGUMENT": 400,
    "DEADLINE_EXCEEDED": 504,
    "NOT_FOUND": 404,
    "PERMISSION_DENIED": 403,
    "UNAUTHENTICATED": 401,
    "RESOURCE_EXHAUSTED": 429,
    "UNIMPLEMENTED": 501,
    "INTERNAL": 500,
    "UNAVAILABLE": 503,
}
