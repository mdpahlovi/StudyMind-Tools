"""Unit tests for app.converter. These run without downloading any Docling models."""

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from docling.datamodel import asr_model_specs
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.pipeline.video_pipeline import VideoPipeline

from app import converter


class _FakeDocument:
    def __init__(self, text: str = "", pages: int = 2) -> None:
        self._text = text
        self._pages = pages
        self.pictures: list[Any] = []
        self.pages: dict[int, Any] = {}

    def num_pages(self) -> int:
        return self._pages

    def export_to_markdown(self, image_mode: Any = None) -> str:
        return self._text

    def export_to_dict(self) -> dict[str, Any]:
        return {"schema_name": "DoclingDocument"}


class _FakeConverter:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def convert(self, source: str, **kwargs: Any) -> Any:
        self.calls.append({"source": source, **kwargs})
        return self.result


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep tests independent of the developer's .env and of any cached converters.
    for name in (
        "DOCLING_ASR_MODEL",
        "DOCLING_DOCUMENT_TIMEOUT",
        "DOCLING_MAX_FILE_SIZE_MB",
        "DOCLING_MAX_NUM_PAGES",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(converter, "load_environment", lambda: None)


def test_converter_kind_maps_item_types() -> None:
    assert converter.converter_kind("AUDIO") == "asr"
    assert converter.converter_kind("VIDEO") == "asr"
    assert converter.converter_kind("IMAGE") == "picture"
    assert converter.converter_kind("DOCUMENT") == "picture"
    assert converter.converter_kind(None) == "document"


def test_asr_model_is_resolved_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCLING_ASR_MODEL", "whisper_tiny")
    name, spec = converter.resolve_asr_model()
    assert name == "WHISPER_TINY"
    assert spec is asr_model_specs.WHISPER_TINY


def test_unknown_asr_model_falls_back_and_reports_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCLING_ASR_MODEL", "NOT_A_MODEL")
    name, spec = converter.resolve_asr_model()
    assert name == converter.DEFAULT_ASR_MODEL
    assert spec is asr_model_specs.WHISPER_BASE

    meta = converter.document_meta(_FakeDocument(), "AUDIO")
    assert meta["asrModel"] == converter.DEFAULT_ASR_MODEL


def test_video_uses_video_pipeline_with_configured_model_and_no_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DOCLING_ASR_MODEL", "WHISPER_TINY")
    built = converter._build_converter("asr")

    video = built.format_to_options[InputFormat.VIDEO]
    assert video.pipeline_cls is VideoPipeline
    assert video.pipeline_options.asr_options is asr_model_specs.WHISPER_TINY
    assert video.pipeline_options.generate_frame_images is False

    audio = built.format_to_options[InputFormat.AUDIO]
    assert audio.pipeline_options.asr_options is asr_model_specs.WHISPER_TINY


def test_document_meta_reports_model_actually_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCLING_ASR_MODEL", "WHISPER_TINY")
    assert converter.document_meta(_FakeDocument(), "VIDEO")["asrModel"] == "WHISPER_TINY"
    assert "asrModel" not in converter.document_meta(_FakeDocument(), "IMAGE")


def test_limits_come_from_env_and_ignore_invalid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    assert converter.max_file_size_bytes() == 200 * 1024 * 1024
    assert converter.max_num_pages() == 1000
    assert converter.document_timeout() == 600.0

    monkeypatch.setenv("DOCLING_MAX_FILE_SIZE_MB", "5")
    monkeypatch.setenv("DOCLING_MAX_NUM_PAGES", "abc")
    monkeypatch.setenv("DOCLING_DOCUMENT_TIMEOUT", "-3")
    assert converter.max_file_size_bytes() == 5 * 1024 * 1024
    assert converter.max_num_pages() == 1000
    assert converter.document_timeout() == 600.0


def test_convert_file_returns_markdown_and_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeConverter(
        SimpleNamespace(
            status=ConversionStatus.SUCCESS,
            errors=[],
            document=_FakeDocument(text="# Title", pages=3),
        )
    )
    monkeypatch.setattr(converter, "get_converter", lambda kind: fake)

    out = converter.convert_file(Path("notes.pdf"), "DOCUMENT")

    assert out["text"] == "# Title"
    assert out["meta"]["numPages"] == 3
    assert out["pictures"] == []
    assert out["structured"] == {"schema_name": "DoclingDocument"}
    assert fake.calls[0]["max_file_size"] == converter.max_file_size_bytes()
    assert fake.calls[0]["max_num_pages"] == converter.max_num_pages()


@pytest.mark.parametrize("status", [ConversionStatus.PARTIAL_SUCCESS, ConversionStatus.SKIPPED])
def test_convert_file_fails_instead_of_publishing_partial_results(
    monkeypatch: pytest.MonkeyPatch, status: ConversionStatus
) -> None:
    fake = _FakeConverter(
        SimpleNamespace(
            status=status,
            errors=[SimpleNamespace(error_message="document timed out")],
            document=_FakeDocument(text="partial"),
        )
    )
    monkeypatch.setattr(converter, "get_converter", lambda kind: fake)

    with pytest.raises(RuntimeError, match="document timed out"):
        converter.convert_file(Path("long.mp4"), "VIDEO")


@pytest.mark.skipif(
    os.getenv("RUN_DOCLING_INTEGRATION") != "1",
    reason="downloads Docling models; set RUN_DOCLING_INTEGRATION=1 to run",
)
def test_image_conversion_extracts_picture(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw

    # A solid-colour image has no content, so Docling emits no picture for it.
    figure = Image.new("RGB", (400, 250), "white")
    draw = ImageDraw.Draw(figure)
    draw.rectangle([20, 20, 380, 230], outline="blue", width=6)
    draw.ellipse([80, 60, 320, 200], fill="orange")
    png = tmp_path / "figure.png"
    figure.save(png)

    out = converter.convert_file(png, "IMAGE")

    assert "](picture/picture-1.png)" in out["text"]
    assert [name for name, _ in out["pictures"]] == ["picture-1.png"]
