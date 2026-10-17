"""Tests for the StudyMind AI tool endpoints (/markdown-to-pdf, /text-to-audio)."""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import main

client = TestClient(main.app)

# Two MPEG-1 Layer III frames (128 kbps, 44.1 kHz) — the minimum mutagen needs to
# sync and report a duration.
_MP3_FRAME = bytes([0xFF, 0xFB, 0x90, 0x00]) + b"\x00" * 413
_MP3_BYTES = _MP3_FRAME * 2
_MP3_DURATION = 0.052125


class _FakeBucket:
    def __init__(self, *, return_path: bool = True) -> None:
        self.return_path = return_path
        self.uploads: list[dict[str, Any]] = []

    def upload(self, *, file: Any, path: str, file_options: dict[str, str]) -> SimpleNamespace:
        self.uploads.append({"path": path, "data": file.read(), "file_options": file_options})
        if not self.return_path:
            return SimpleNamespace(path="", fullPath="")
        return SimpleNamespace(path=path, fullPath=f"studymind/{path}")

    def get_public_url(self, path: str) -> str:
        return f"https://example.supabase.co/storage/v1/object/public/studymind/{path}"


class _FakeStorage:
    def __init__(self, *, return_path: bool = True) -> None:
        self.bucket = _FakeBucket(return_path=return_path)
        self.bucket_names: list[str] = []

    def from_(self, bucket_name: str) -> _FakeBucket:
        self.bucket_names.append(bucket_name)
        return self.bucket


class _FakeSupabase:
    def __init__(self, *, return_path: bool = True) -> None:
        self.storage = _FakeStorage(return_path=return_path)


@pytest.fixture
def fake_supabase(monkeypatch: pytest.MonkeyPatch) -> _FakeSupabase:
    """Replace the module-level Supabase client so no env vars or network are needed."""
    fake = _FakeSupabase()
    monkeypatch.setattr(main, "_supabase", fake)
    return fake


class _FakeGtts:
    """Offline stand-in for gTTS that writes a playable MP3 frame."""

    def __init__(self, *, text: str, lang: str = "en", **kwargs: Any) -> None:
        self.text = text
        self.lang = lang

    def save(self, path: str) -> None:
        Path(path).write_bytes(_MP3_BYTES)


def _temp_artifacts(suffix: str) -> set[Path]:
    return set(Path(tempfile.gettempdir()).glob(f"tmp*{suffix}"))


def test_markdown_to_pdf(fake_supabase: _FakeSupabase) -> None:
    before = _temp_artifacts(".pdf")

    response = client.post(
        "/markdown-to-pdf",
        json={"contents": "# Heading\n\nSome **body** text.", "filePath": "notes.pdf"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["message"] == "PDF generated successfully"
    assert body["data"]["filePath"] == "notes.pdf"
    assert body["data"]["fileUrl"].startswith("https://example.supabase.co/")
    assert "/studymind/notes.pdf?v=" in body["data"]["fileUrl"]
    assert body["data"]["fileSize"] > 0

    upload = fake_supabase.storage.bucket.uploads[0]
    assert upload["path"] == "notes.pdf"
    assert upload["file_options"] == {"content-type": "application/pdf", "upsert": "true"}
    assert upload["data"].startswith(b"%PDF")
    assert set(fake_supabase.storage.bucket_names) == {main.BUCKET_NAME}
    assert _temp_artifacts(".pdf") == before, "temporary PDF should be cleaned up"


def test_markdown_to_pdf_upload_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "_supabase", _FakeSupabase(return_path=False))

    response = client.post(
        "/markdown-to-pdf",
        json={"contents": "# Heading", "filePath": "notes.pdf"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Failed to upload PDF to Supabase"


def test_text_to_audio(fake_supabase: _FakeSupabase, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "gTTS", _FakeGtts)
    before = _temp_artifacts(".mp3")

    response = client.post(
        "/text-to-audio",
        json={"contents": "Hello there", "filePath": "lesson.mp3", "language": "en"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["message"] == "Audio generated successfully"
    assert body["data"]["filePath"] == "lesson.mp3"
    assert "/studymind/lesson.mp3?v=" in body["data"]["fileUrl"]
    assert body["data"]["fileSize"] == len(_MP3_BYTES)
    assert body["data"]["duration"] == round(_MP3_DURATION)

    upload = fake_supabase.storage.bucket.uploads[0]
    assert upload["path"] == "lesson.mp3"
    assert upload["file_options"] == {"content-type": "audio/mpeg", "upsert": "true"}
    assert upload["data"] == _MP3_BYTES
    assert _temp_artifacts(".mp3") == before, "temporary MP3 should be cleaned up"


def test_text_to_audio_defaults_language(
    fake_supabase: _FakeSupabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main, "gTTS", _FakeGtts)

    response = client.post("/text-to-audio", json={"contents": "Hi", "filePath": "a.mp3"})

    assert response.status_code == 200
    assert response.json()["data"]["duration"] == round(_MP3_DURATION)


def test_missing_supabase_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "_supabase", None)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)

    response = client.post("/markdown-to-pdf", json={"contents": "x", "filePath": "a.pdf"})

    assert response.status_code == 500
    assert "SUPABASE_URL" in response.json()["detail"]


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/markdown-to-pdf", {"contents": "", "filePath": "a.pdf"}),
        ("/markdown-to-pdf", {"filePath": "a.pdf"}),
        ("/text-to-audio", {"contents": "hi", "filePath": ""}),
    ],
)
def test_request_validation(path: str, payload: dict[str, str]) -> None:
    response = client.post(path, json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("markdown", "note"),
    [
        ("## Only an H2\n\ntext", "first heading is H2"),
        ("## One\n\ntext\n\n## Two\n\nmore", "all headings are H2"),
        ("# H1\n\n## H2\n\ntext", "normal H1 + H2"),
        ("### H3 only\n\ntext", "headings deeper than toc_level"),
        ("no headings at all", "no TOC entries"),
    ],
)
def test_markdown_to_pdf_heading_shapes(
    fake_supabase: _FakeSupabase, markdown: str, note: str
) -> None:
    """Regression: PyMuPDF rejects a TOC whose first entry is not level 1."""
    response = client.post(
        "/markdown-to-pdf", json={"contents": markdown, "filePath": "shapes.pdf"}
    )

    assert response.status_code == 200, f"{note}: {response.text}"
    assert fake_supabase.storage.bucket.uploads[0]["data"].startswith(b"%PDF")


def test_file_url_version_tracks_content(fake_supabase: _FakeSupabase) -> None:
    """Regression: an overwritten name must not keep serving cached bytes."""

    def url_for(contents: str) -> str:
        response = client.post(
            "/markdown-to-pdf", json={"contents": contents, "filePath": "same.pdf"}
        )
        assert response.status_code == 200
        return str(response.json()["data"]["fileUrl"])

    first = url_for("# One")
    assert url_for("# One") == first, "same content should reuse the cached URL"
    assert url_for("# Two") != first, "changed content must produce a new URL"
