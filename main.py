"""StudyMind Tools API.

FastAPI service providing:
- /markdown-to-pdf: Convert Markdown to PDF and upload to Supabase storage.
- /text-to-audio: Convert text to MP3 via gTTS and upload to Supabase.

Environment variables:
- SUPABASE_URL
- SUPABASE_KEY
- BUCKET_NAME (optional, default 'studymind')
"""

import os
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from supabase import create_client, Client
from markdown_pdf import MarkdownPdf, Section
from gtts import gTTS

load_dotenv()

app = FastAPI(title="StudyMind Tools")

# Supabase client
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
BUCKET_NAME = os.environ.get("BUCKET_NAME", "studymind")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Return favicon.ico"""
    return FileResponse("static/favicon.png")


@app.get("/")
def root():
    """Check if the service is running."""
    return {"message": "StudyMind Tools API is running"}


# Endpoints
class MarkdownToPdfIn(BaseModel):
    """Request body for the /markdown-to-pdf endpoint."""

    contents: str = Field(..., min_length=1, max_length=500000)
    filename: str = Field(..., min_length=1, max_length=100)


@app.post("/markdown-to-pdf")
def markdown_to_pdf(payload: MarkdownToPdfIn):
    """Generate pdf from markdown and upload to Supabase."""

    temp_path = None

    try:
        # Create PDF in temporary file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_path = temp_file.name

        pdf = MarkdownPdf(toc_level=2, optimize=True)
        pdf.add_section(Section(payload.contents))
        pdf.save(temp_path)

        # Upload to Supabase
        with open(temp_path, "rb") as f:
            response = supabase.storage.from_(BUCKET_NAME).upload(
                file=f,
                path=payload.filename,
                file_options={"content-type": "application/pdf", "upsert": "true"},
            )

        # Handle response errors
        if isinstance(response, dict) and response.get("error"):
            raise Exception(response.get("error"))

        if not response:
            raise Exception("Failed to upload PDF to Supabase")

        return {
            "success": True,
            "message": "PDF generated successfully",
            "filename": payload.filename,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


class TextToAudioIn(BaseModel):
    """Request body for the /text-to-audio endpoint."""

    contents: str = Field(..., min_length=1, max_length=500000)
    filename: str = Field(..., min_length=1, max_length=100)
    language: str = Field(default="en")


@app.post("/text-to-audio")
def text_to_audio(payload: TextToAudioIn):
    """Generate audio from text and upload to supabase."""

    temp_path = None

    try:
        # Create audio in temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = temp_file.name

        # Generate audio
        tts = gTTS(text=payload.contents, lang=payload.language)
        tts.save(temp_path)

        # Upload to Supabase
        with open(temp_path, "rb") as f:
            response = supabase.storage.from_(BUCKET_NAME).upload(
                file=f,
                path=payload.filename,
                file_options={"content-type": "audio/mpeg", "upsert": "true"},
            )

        # Handle response errors
        if isinstance(response, dict) and response.get("error"):
            raise Exception(response.get("error"))

        if not response:
            raise Exception("Failed to upload audio to Supabase")

        return {
            "success": True,
            "message": "Audio generated successfully",
            "filename": payload.filename,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
