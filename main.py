import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import asyncio
from functools import wraps

from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from gtts import gTTS
from markdownpdf import MarkdownPdf
from supabase import create_client, Client
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="StudyMind Tools", description="Tools for StudyMind", version="1.0.0"
)

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET_NAME = os.getenv("SUPABASE_BUCKET_NAME")

# Validate environment variables
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables are required")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# Pydantic models
class MarkdownRequest(BaseModel):
    markdown_text: str = Field(..., description="Markdown text to convert to PDF")
    filename: str = Field(None, description="Optional filename (without extension)")


class AudioRequest(BaseModel):
    script_text: str = Field(..., description="Text script to convert to audio")
    filename: str = Field(None, description="Optional filename (without extension)")
    language: str = Field("en", description="Language code for text-to-speech")
    slow: bool = Field(False, description="Whether to speak slowly")


class ProcessingResponse(BaseModel):
    success: bool
    file_url: str = None
    filename: str = None
    message: str = None
    file_size: int = None


# Utility functions
def async_endpoint(func):
    """Decorator to handle async operations in endpoints"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)

    return wrapper


def generate_unique_filename(base_name: str, extension: str) -> str:
    """Generate a unique filename with timestamp and UUID"""
    if not base_name:
        base_name = "file"

    # Sanitize filename
    base_name = "".join(
        c for c in base_name if c.isalnum() or c in (" ", "-", "_")
    ).rstrip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]

    return f"{base_name}_{timestamp}_{unique_id}.{extension}"


def upload_to_supabase(file_path: str, filename: str) -> Dict[str, Any]:
    """Upload file to Supabase storage"""
    try:
        with open(file_path, "rb") as file:
            file_data = file.read()

        # Upload file
        result = supabase.storage.from_(SUPABASE_BUCKET_NAME).upload(
            filename, file_data, file_options={"cache-control": "3600"}
        )

        if result.status_code not in [200, 201]:
            raise Exception(f"Upload failed with status {result.status_code}")

        # Get public URL
        public_url = supabase.storage.from_(SUPABASE_BUCKET_NAME).get_public_url(
            filename
        )

        return {"success": True, "public_url": public_url, "file_size": len(file_data)}

    except Exception as e:
        logger.error(f"Supabase upload error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload to storage: {str(e)}",
        )


def markdown_to_pdf_sync(markdown_text: str, output_path: str) -> None:
    """Convert markdown to PDF synchronously using markdown-pdf library"""
    try:
        # Create MarkdownPdf instance with custom styling
        pdf = MarkdownPdf(toc_level=2)

        # Set custom CSS for better styling
        pdf.meta[
            "css"
        ] = """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
            margin-top: 2em;
            margin-bottom: 1em;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        pre {
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        blockquote {
            border-left: 4px solid #ddd;
            margin: 1em 0;
            padding-left: 1em;
            color: #666;
        }
        a {
            color: #3498db;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        """

        # Create temporary markdown file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as temp_md:
            temp_md.write(markdown_text)
            temp_md_path = temp_md.name

        try:
            # Convert markdown to PDF
            pdf.convert(temp_md_path, output_path)
        finally:
            # Clean up temporary markdown file
            if os.path.exists(temp_md_path):
                os.unlink(temp_md_path)

    except Exception as e:
        logger.error(f"PDF generation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate PDF: {str(e)}",
        )


def text_to_audio_sync(
    text: str, output_path: str, language: str = "en", slow: bool = False
) -> None:
    """Convert text to audio synchronously"""
    try:
        # Validate text length (gTTS has limits)
        if len(text) > 100000:  # 100KB limit
            raise ValueError("Text is too long for audio conversion")

        if not text.strip():
            raise ValueError("Text cannot be empty")

        # Generate audio
        tts = gTTS(text=text, lang=language, slow=slow)
        tts.save(output_path)

    except Exception as e:
        logger.error(f"Audio generation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate audio: {str(e)}",
        )


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Document & Audio Processing API",
        "version": "1.0.0",
        "endpoints": {
            "/convert/markdown-to-pdf": "POST - Convert markdown to PDF",
            "/convert/text-to-audio": "POST - Convert text to audio",
            "/health": "GET - Health check",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Supabase connection
        supabase.storage.list_buckets()
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}",
        )


@app.post("/convert/markdown-to-pdf", response_model=ProcessingResponse)
async def convert_markdown_to_pdf(
    request: MarkdownRequest, background_tasks: BackgroundTasks
):
    """Convert markdown text to PDF and upload to Supabase"""

    if not request.markdown_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Markdown text cannot be empty",
        )

    # Generate unique filename
    filename = generate_unique_filename(request.filename, "pdf")

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_path = temp_file.name

    try:
        # Convert markdown to PDF
        await asyncio.get_event_loop().run_in_executor(
            None, markdown_to_pdf_sync, request.markdown_text, temp_path
        )

        # Upload to Supabase
        upload_result = upload_to_supabase(temp_path, filename)

        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, temp_path)

        return ProcessingResponse(
            success=True,
            file_url=upload_result["public_url"],
            filename=filename,
            message="PDF generated and uploaded successfully",
            file_size=upload_result["file_size"],
        )

    except Exception as e:
        # Cleanup on error
        cleanup_temp_file(temp_path)
        raise e


@app.post("/convert/text-to-audio", response_model=ProcessingResponse)
async def convert_text_to_audio(
    request: AudioRequest, background_tasks: BackgroundTasks
):
    """Convert text script to audio and upload to Supabase"""

    if not request.script_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Script text cannot be empty",
        )

    # Validate language code
    try:
        # Test if language is supported by creating a small test
        test_tts = gTTS(text="test", lang=request.language)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported language code: {request.language}",
        )

    # Generate unique filename
    filename = generate_unique_filename(request.filename, "mp3")

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_path = temp_file.name

    try:
        # Convert text to audio
        await asyncio.get_event_loop().run_in_executor(
            None,
            text_to_audio_sync,
            request.script_text,
            temp_path,
            request.language,
            request.slow,
        )

        # Upload to Supabase
        upload_result = upload_to_supabase(temp_path, filename)

        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, temp_path)

        return ProcessingResponse(
            success=True,
            file_url=upload_result["public_url"],
            filename=filename,
            message="Audio generated and uploaded successfully",
            file_size=upload_result["file_size"],
        )

    except Exception as e:
        # Cleanup on error
        cleanup_temp_file(temp_path)
        raise e


def cleanup_temp_file(file_path: str):
    """Clean up temporary files"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {file_path}: {str(e)}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "An unexpected error occurred",
            "status_code": 500,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
