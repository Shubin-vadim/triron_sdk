from triton_sdk.core.utils import decode_bytes_tensor, encode_bytes_tensor, is_loopback_url, normalize_url


def test_normalize_url_adds_scheme() -> None:
    assert normalize_url("localhost:8000") == "http://localhost:8000"
    assert normalize_url("https://triton.local/") == "https://triton.local"


def test_is_loopback_url() -> None:
    assert is_loopback_url("localhost:8000")
    assert is_loopback_url("http://127.0.0.1:9000")
    assert is_loopback_url("http://[::1]:8000")
    assert is_loopback_url("http://foo.localhost:8000")
    assert not is_loopback_url("triton.example:8000")
    assert not is_loopback_url("https://triton.internal")


def test_bytes_tensor_roundtrip() -> None:
    values = ["hello", "мир", ""]
    assert decode_bytes_tensor(encode_bytes_tensor(values)) == values
