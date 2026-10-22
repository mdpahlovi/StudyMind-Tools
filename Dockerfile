# syntax=docker/dockerfile:1

FROM python:3.14-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

# Prefetch Docling's models and the ASR checkpoint so containers start without downloading.
RUN /app/.venv/bin/docling-tools models download --output-dir /opt/docling-models \
 && /app/.venv/bin/docling-tools models download whisper_base --output-dir /opt/docling-models


FROM python:3.14-slim AS runtime

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg libreoffice-writer-nogui \
 && rm -rf /var/lib/apt/lists/*

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DOCLING_ARTIFACTS_PATH=/opt/docling-models

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /opt/docling-models /opt/docling-models
COPY app ./app
COPY static ./static

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
