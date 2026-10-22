"""Tests for the download step of the RabbitMQ library consumer (no network, no models)."""

import io
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from app import converter
from app import rabbitmq_consumer as consumer


class _FakeResponse:
    def __init__(self, body: bytes, headers: dict[str, str] | None = None) -> None:
        self._buffer = io.BytesIO(body)
        self.headers: dict[str, str] = headers or {}

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


@pytest.fixture
def temp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Points tempfile at an isolated directory so leftovers are easy to detect."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


def _serve(monkeypatch: pytest.MonkeyPatch, response: _FakeResponse) -> None:
    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        return response

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def test_download_streams_file_with_unique_name(
    monkeypatch: pytest.MonkeyPatch, temp_dir: Path
) -> None:
    body = b"lecture-bytes" * 1000
    _serve(monkeypatch, _FakeResponse(body))

    target = consumer.download_resource_file(
        "https://storage.example/studymind/item-1/lecture.mp4?v=1", "item-1"
    )

    try:
        assert target.read_bytes() == body
        assert target.suffix == ".mp4"
        assert target.name.startswith("studymind-item-1-")
    finally:
        target.unlink(missing_ok=True)


def test_download_drops_unsafe_suffix(monkeypatch: pytest.MonkeyPatch, temp_dir: Path) -> None:
    _serve(monkeypatch, _FakeResponse(b"data"))

    target = consumer.download_resource_file("https://storage.example/a/file.a-b!c", "item-2")

    try:
        assert target.suffix == ""
    finally:
        target.unlink(missing_ok=True)


def test_download_rejects_oversize_body_and_removes_partial_file(
    monkeypatch: pytest.MonkeyPatch, temp_dir: Path
) -> None:
    monkeypatch.setattr(converter, "max_file_size_bytes", lambda: 10)
    _serve(monkeypatch, _FakeResponse(b"x" * 100))

    with pytest.raises(ValueError, match="byte limit"):
        consumer.download_resource_file("https://storage.example/a/big.pdf", "item-3")

    assert list(temp_dir.iterdir()) == []


def test_download_rejects_declared_oversize_before_reading(
    monkeypatch: pytest.MonkeyPatch, temp_dir: Path
) -> None:
    monkeypatch.setattr(converter, "max_file_size_bytes", lambda: 10)
    _serve(monkeypatch, _FakeResponse(b"x" * 100, headers={"Content-Length": "100"}))

    with pytest.raises(ValueError, match="limit is 10"):
        consumer.download_resource_file("https://storage.example/a/big.pdf", "item-4")

    assert list(temp_dir.iterdir()) == []
