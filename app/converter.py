"""Docling-based conversion of library items into AI-ready structured data.

Files uploaded through the NestJS API are stored in Supabase Storage and their
public URL is queued on ``library.processing``. The consumer downloads the file
and hands the local path to :func:`convert_file`, which runs it through Docling
and returns structured output (Markdown text plus the lossless Docling JSON document)
suitable as contextual input for AI models and vector chunking.

Supported library item types:
* ``DOCUMENT`` — PDF, DOCX, PPTX, XLSX, HTML, Markdown, EPUB, …
* ``IMAGE``    — PNG, JPEG, TIFF, BMP, WEBP (single-page pipeline with OCR)
* ``AUDIO``    — WAV, MP3, M4A, AAC, OGG, FLAC (Whisper ASR pipeline)
* ``VIDEO``    — MP4, AVI, MOV, MKV, WEBM through Docling's video pipeline: the
  audio track is transcribed with the same ASR model as ``AUDIO``. Frame sampling
  is disabled, so only the transcript is produced.

Environment settings (all optional):
* ``DOCLING_ASR_MODEL``          — ASR preset name, default ``WHISPER_BASE``
* ``DOCLING_DOCUMENT_TIMEOUT``   — seconds per document, default 600
* ``DOCLING_MAX_FILE_SIZE_MB``   — largest accepted input, default 200
* ``DOCLING_MAX_NUM_PAGES``      — largest accepted page count, default 1000
* ``DOCLING_ARTIFACTS_PATH``     — pre-downloaded model directory (read by Docling)
"""

from __future__ import annotations

import io
import logging
import os
import threading
from pathlib import Path
from typing import Any

from app.config import load_environment

LOGGER = logging.getLogger(__name__)

SUPPORTED_FILE_TYPES = frozenset({"DOCUMENT", "IMAGE", "AUDIO", "VIDEO"})

_ASR_FILE_TYPES = frozenset({"AUDIO", "VIDEO"})
_PICTURE_FILE_TYPES = frozenset({"DOCUMENT", "IMAGE"})

_CONVERTER_LOCK = threading.Lock()
_CONVERTERS: dict[str, Any] = {}

DEFAULT_ASR_MODEL = "WHISPER_BASE"
DEFAULT_DOCUMENT_TIMEOUT = 600.0
DEFAULT_MAX_FILE_SIZE_MB = 200.0
DEFAULT_MAX_NUM_PAGES = 1000
PICTURE_SCALE = 2.0
_PICTURE_REF_PLACEHOLDER = "picture/{}"


def _ensure_utf8_io() -> None:
    """Ensure standard output/error streams handle UTF-8 on Windows."""
    import sys

    for stream in (sys.stdout, sys.stderr):
        if stream and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _patch_docling_audio_sniffing() -> None:
    """Ensure Docling classifies audio containers like .m4a as AUDIO instead of VIDEO."""
    try:
        from docling.datamodel.base_models import FormatToExtensions, InputFormat
        from docling.document_converter import _DocumentConversionInput

        audio_exts = {
            f".{ext.lower()}" for ext in FormatToExtensions.get(InputFormat.AUDIO, [])
        }
        orig_guess = getattr(_DocumentConversionInput, "_guess_format", None)
        if orig_guess is not None and not getattr(orig_guess, "_studybuddy_patched", False):

            def _patched_guess(self: Any, obj: Any) -> Any:
                ext = getattr(obj, "suffix", "") or (
                    Path(obj.name).suffix if hasattr(obj, "name") else ""
                )
                if ext.lower() in audio_exts:
                    return InputFormat.AUDIO
                return orig_guess(self, obj)

            _patched_guess._studybuddy_patched = True  # type: ignore[attr-defined]
            _DocumentConversionInput._guess_format = _patched_guess
    except Exception as exc:
        LOGGER.debug("Could not patch docling audio format guesser: %s", exc)


def converter_kind(file_type: str | None) -> str:
    if file_type in _ASR_FILE_TYPES:
        return "asr"
    if file_type in _PICTURE_FILE_TYPES:
        return "picture"
    return "document"


def _env_positive(name: str, default: float) -> float:
    """Reads a positive number from the environment, falling back to ``default``."""
    load_environment()
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = 0.0
    if value <= 0:
        LOGGER.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default
    return value


def document_timeout() -> float:
    """Per-document processing budget in seconds (Docling's ``document_timeout``)."""
    return _env_positive("DOCLING_DOCUMENT_TIMEOUT", DEFAULT_DOCUMENT_TIMEOUT)


def max_file_size_bytes() -> int:
    return int(_env_positive("DOCLING_MAX_FILE_SIZE_MB", DEFAULT_MAX_FILE_SIZE_MB) * 1024 * 1024)


def max_num_pages() -> int:
    return int(_env_positive("DOCLING_MAX_NUM_PAGES", DEFAULT_MAX_NUM_PAGES))


def resolve_asr_model() -> tuple[str, Any]:
    """Returns the ASR preset name and spec that will actually be used.

    Unknown names fall back to ``DEFAULT_ASR_MODEL``.
    """
    from docling.datamodel import asr_model_specs
    from docling.datamodel.pipeline_options_asr_model import ApiAsrOptions, InlineAsrOptions

    load_environment()
    requested = os.getenv("DOCLING_ASR_MODEL", DEFAULT_ASR_MODEL).upper().strip()
    requested = requested or DEFAULT_ASR_MODEL
    spec = getattr(asr_model_specs, requested, None)
    if not isinstance(spec, InlineAsrOptions | ApiAsrOptions):
        LOGGER.warning(
            "Unknown DOCLING_ASR_MODEL '%s'; falling back to %s",
            requested,
            DEFAULT_ASR_MODEL,
        )
        requested = DEFAULT_ASR_MODEL
        spec = getattr(asr_model_specs, DEFAULT_ASR_MODEL)
    return requested, spec


def _build_converter(kind: str) -> Any:
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter

    timeout = document_timeout()

    if kind == "asr":
        from docling.datamodel.pipeline_options import AsrPipelineOptions, VideoPipelineOptions
        from docling.document_converter import AudioFormatOption, VideoFormatOption
        from docling.pipeline.asr_pipeline import AsrPipeline

        _, asr_spec = resolve_asr_model()
        audio_options = AsrPipelineOptions(asr_options=asr_spec, document_timeout=timeout)
        video_options = VideoPipelineOptions(
            asr_options=asr_spec,
            document_timeout=timeout,
            generate_frame_images=False,
        )
        return DocumentConverter(
            format_options={
                InputFormat.AUDIO: AudioFormatOption(
                    pipeline_cls=AsrPipeline,
                    pipeline_options=audio_options,
                ),
                InputFormat.VIDEO: VideoFormatOption(pipeline_options=video_options),
            }
        )

    if kind == "picture":
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import (
            ImageFormatOption,
            PdfFormatOption,
        )

        pipeline_options = PdfPipelineOptions(
            document_timeout=timeout,
            generate_picture_images=True,
            images_scale=PICTURE_SCALE,
        )
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
            }
        )

    return DocumentConverter()


def get_converter(kind: str) -> Any:
    """Returns a cached converter instance, creating it on first use."""
    with _CONVERTER_LOCK:
        converter = _CONVERTERS.get(kind)
        if converter is None:
            LOGGER.info(
                "Creating Docling %s converter (this may load models on first use)",
                kind,
            )
            converter = _build_converter(kind)
            _CONVERTERS[kind] = converter
        return converter


def document_meta(document: Any, file_type: str | None) -> dict[str, Any]:
    """Best-effort metadata extracted from a converted DoclingDocument."""
    meta: dict[str, Any] = {"fileType": file_type, "engine": "docling"}
    try:
        meta["numPages"] = document.num_pages()
    except Exception:
        meta["numPages"] = None
    if converter_kind(file_type) == "asr":
        meta["asrModel"], _ = resolve_asr_model()
    return meta


def extract_pictures(document: Any) -> list[tuple[str, bytes]]:
    pictures: list[tuple[str, bytes]] = []
    for index, picture in enumerate(document.pictures):
        image = picture.get_image(document)
        if image is None:
            continue
        name = f"picture-{index + 1}.png"
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        pictures.append((name, buffer.getvalue()))
        picture.image = picture.image.model_copy(
            update={"uri": _PICTURE_REF_PLACEHOLDER.format(name)}
        )
    return pictures


def convert_file(path: Path, file_type: str | None) -> dict[str, Any]:
    _ensure_utf8_io()
    _patch_docling_audio_sniffing()

    from docling.datamodel.base_models import ConversionStatus
    from docling_core.types.doc.base import ImageRefMode

    kind = converter_kind(file_type)
    converter = get_converter(kind)
    LOGGER.info("Converting %s resource with Docling %s pipeline: %s", file_type, kind, path)

    result = converter.convert(
        str(path),
        max_file_size=max_file_size_bytes(),
        max_num_pages=max_num_pages(),
    )
    if result.status != ConversionStatus.SUCCESS:
        details = "; ".join(err.error_message for err in result.errors) or "no details"
        raise RuntimeError(
            f"Docling conversion did not succeed for {path} "
            f"(status={result.status.value}): {details}"
        )
    document = result.document
    if document is None:
        raise RuntimeError(
            f"Docling conversion produced no document for {path} (status={result.status})"
        )

    pictures = extract_pictures(document)
    text = document.export_to_markdown(image_mode=ImageRefMode.REFERENCED)

    return {
        "text": text,
        "structured": document.export_to_dict(),
        "meta": document_meta(document, file_type),
        "pictures": pictures,
    }


def reset_converters() -> None:
    with _CONVERTER_LOCK:
        _CONVERTERS.clear()
