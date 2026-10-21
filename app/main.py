"""Tools for StudyMind AI.

Endpoints:
- /markdown-to-pdf: Generate pdf from markdown and upload to Supabase.
- /text-to-audio: Generate audio from text and upload to Supabase.
"""

import hashlib
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from gtts import gTTS
from markdown_pdf import MarkdownPdf, Section
from mutagen.mp3 import MP3, MPEGInfo
from pydantic import BaseModel, Field
from supabase import Client, create_client

from app.rabbitmq_consumer import RabbitMQConsumerThread

load_dotenv()

LOGGER = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
BUCKET_NAME = os.environ.get("BUCKET_NAME", "studymind")


@asynccontextmanager
async def lifespan(_: FastAPI):
    consumer: RabbitMQConsumerThread | None = None
    if os.getenv("RABBITMQ_URL"):
        consumer = RabbitMQConsumerThread()
        consumer.start()
        LOGGER.info("Library RabbitMQ consumer started")
    try:
        yield
    finally:
        if consumer is not None:
            consumer.stop()
            LOGGER.info("Library RabbitMQ consumer stopped")


app = FastAPI(title="Tools for StudyMind AI", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_supabase: Client | None = None


def get_supabase() -> Client:
    """Return the Supabase client, creating it on first use."""
    global _supabase
    if _supabase is None:
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        if not supabase_url or not supabase_key:
            raise HTTPException(
                status_code=500,
                detail="SUPABASE_URL and SUPABASE_KEY must be set",
            )
        _supabase = create_client(supabase_url, supabase_key)
    return _supabase


def get_file_url(file_path: str, version: str) -> str:
    """Get public URL for uploaded file from Supabase, tagged with its content hash."""
    response = get_supabase().storage.from_(BUCKET_NAME).get_public_url(file_path)
    return f"{response}?v={version}" if response else ""


def get_duration(file_path: str) -> float:
    """Get duration of audio file in seconds."""
    audio = MP3(file_path)  # type: ignore[no-untyped-call]
    info = cast(MPEGInfo, audio.info)
    return info.length if info.length else 0


def normalize_toc(toc: list[tuple[int, str, int, float]]) -> list[tuple[int, str, int, float]]:
    """Clamp heading levels so PyMuPDF's set_toc accepts them.

    PyMuPDF requires the first entry to be level 1 and allows an entry to be at most
    one level deeper than its predecessor. markdown_pdf forwards heading depths
    verbatim, so a document whose first heading is `##` (never `#`) would otherwise
    fail to save.
    """
    normalized: list[tuple[int, str, int, float]] = []
    for level, title, page, top in toc:
        level = 1 if not normalized else min(level, normalized[-1][0] + 1)
        normalized.append((level, title, page, top))
    return normalized


def get_file_version(file_path: str) -> str:
    """Short content hash, used to serve the right bytes when a name is overwritten."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> RedirectResponse:
    """Send browsers' default favicon request to the mounted static file."""
    return RedirectResponse("/static/favicon.png", status_code=307)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"Tools for StudyMind AI": "API is live!"}


class MarkdownToPdfIn(BaseModel):
    """Request body for the /markdown-to-pdf endpoint."""

    contents: str = Field(..., min_length=1, max_length=500000)
    filePath: str = Field(..., min_length=1, max_length=100)


@app.post("/markdown-to-pdf")
def markdown_to_pdf(payload: MarkdownToPdfIn) -> dict[str, Any]:
    """Generate pdf from markdown and upload to Supabase."""

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_path = temp_file.name

        pdf = MarkdownPdf(toc_level=2, optimize=True)
        pdf.add_section(Section(payload.contents))
        pdf.toc = normalize_toc(pdf.toc)
        pdf.save(temp_path)

        with open(temp_path, "rb") as f:
            response = (
                get_supabase()
                .storage.from_(BUCKET_NAME)
                .upload(
                    file=f,
                    path=payload.filePath,
                    file_options={"content-type": "application/pdf", "upsert": "true"},
                )
            )

        if not response.path or not response.fullPath:
            raise HTTPException(status_code=400, detail="Failed to upload PDF to Supabase")

        return {
            "success": True,
            "message": "PDF generated successfully",
            "data": {
                "filePath": payload.filePath,
                "fileUrl": get_file_url(payload.filePath, get_file_version(temp_path)),
                "fileSize": os.path.getsize(temp_path),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {e}") from e

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


class TextToAudioIn(BaseModel):
    """Request body for the /text-to-audio endpoint."""

    contents: str = Field(..., min_length=1, max_length=500000)
    filePath: str = Field(..., min_length=1, max_length=100)
    language: str = Field(default="en")


@app.post("/text-to-audio")
def text_to_audio(payload: TextToAudioIn) -> dict[str, Any]:
    """Generate audio from text and upload to supabase."""

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = temp_file.name

        tts = gTTS(text=payload.contents, lang=payload.language)
        tts.save(temp_path)

        with open(temp_path, "rb") as f:
            response = (
                get_supabase()
                .storage.from_(BUCKET_NAME)
                .upload(
                    file=f,
                    path=payload.filePath,
                    file_options={"content-type": "audio/mpeg", "upsert": "true"},
                )
            )

        if not response.path or not response.fullPath:
            raise HTTPException(status_code=400, detail="Failed to upload audio to Supabase")

        return {
            "success": True,
            "message": "Audio generated successfully",
            "data": {
                "duration": round(get_duration(temp_path)),
                "filePath": payload.filePath,
                "fileUrl": get_file_url(payload.filePath, get_file_version(temp_path)),
                "fileSize": os.path.getsize(temp_path),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {e}") from e

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
