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
* ``VIDEO``    — transcribed through the same ASR pipeline (audio track only)
"""

from __future__ import annotations

import io
import logging
import os
import threading
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

SUPPORTED_FILE_TYPES = frozenset({"DOCUMENT", "IMAGE", "AUDIO", "VIDEO"})

_ASR_FILE_TYPES = frozenset({"AUDIO", "VIDEO"})
_PICTURE_FILE_TYPES = frozenset({"DOCUMENT", "IMAGE"})

_CONVERTER_LOCK = threading.Lock()
_CONVERTERS: dict[str, Any] = {}

DEFAULT_ASR_MODEL = "WHISPER_BASE"
PICTURE_SCALE = 2.0
_PICTURE_REF_PLACEHOLDER = "picture/{}"


def converter_kind(file_type: str | None) -> str:
    if file_type in _ASR_FILE_TYPES:
        return "asr"
    if file_type in _PICTURE_FILE_TYPES:
        return "picture"
    return "document"


def asr_model_spec() -> Any:
    from docling.datamodel import asr_model_specs

    name = os.getenv("DOCLING_ASR_MODEL", DEFAULT_ASR_MODEL).upper().strip()
    spec = getattr(asr_model_specs, name, None)
    if spec is None:
        LOGGER.warning(
            "Unknown DOCLING_ASR_MODEL '%s'; falling back to %s",
            name,
            DEFAULT_ASR_MODEL,
        )
        spec = getattr(asr_model_specs, DEFAULT_ASR_MODEL)
    return spec


def _build_converter(kind: str) -> Any:
    from docling.document_converter import DocumentConverter

    if kind == "asr":
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import AsrPipelineOptions
        from docling.document_converter import AudioFormatOption
        from docling.pipeline.asr_pipeline import AsrPipeline

        pipeline_options = AsrPipelineOptions()
        pipeline_options.asr_options = asr_model_spec()
        return DocumentConverter(
            format_options={
                InputFormat.AUDIO: AudioFormatOption(
                    pipeline_cls=AsrPipeline,
                    pipeline_options=pipeline_options,
                )
            }
        )

    if kind == "picture":
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import (
            ImageFormatOption,
            PdfFormatOption,
        )

        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = PICTURE_SCALE
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
        meta["asrModel"] = (
            os.getenv("DOCLING_ASR_MODEL", DEFAULT_ASR_MODEL).upper().strip()
            or DEFAULT_ASR_MODEL
        )
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
    for page in document.pages.values():
        page.image = None
    return pictures


def convert_file(path: Path, file_type: str | None) -> dict[str, Any]:
    kind = converter_kind(file_type)
    converter = get_converter(kind)
    LOGGER.info(
        "Converting %s resource with Docling %s pipeline: %s", file_type, kind, path
    )

    result = converter.convert(str(path))
    document = result.document
    if document is None:
        raise RuntimeError(
            f"Docling conversion produced no document for {path} (status={result.status})"
        )

    pictures = extract_pictures(document)

    from docling_core.types.doc.base import ImageRefMode

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
