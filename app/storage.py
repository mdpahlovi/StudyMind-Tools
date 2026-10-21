"""Supabase Storage upload of the files produced by the Docling pipeline.

After conversion the consumer uploads the Markdown rendering, the
structured Docling JSON document and any picture images next to the
original upload, under ``<itemUid>/processed.{md,json}`` and
``<itemUid>/picture/<name>.png``, so everything about a library item
lives in one folder that can be deleted wholesale. The returned URLs
are included in the ``library.processed`` payload.

Configuration comes from SUPABASE_URL, SUPABASE_KEY, and BUCKET_NAME
in Tools/.env. When Supabase is not configured the upload is skipped
and no file URLs are reported.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any

from app.config import load_environment

LOGGER = logging.getLogger(__name__)

_CLIENT_LOCK = threading.Lock()
_CLIENT: Any = None

PICTURE_CONTENT_TYPE = "image/png"


def settings() -> dict[str, str] | None:
    """Reads SUPABASE_* variables; returns None when storage is not configured."""
    load_environment()
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    bucket = os.getenv("BUCKET_NAME", "studymind")
    if not (supabase_url and supabase_key):
        return None
    return {
        "url": supabase_url,
        "key": supabase_key,
        "bucket": bucket,
    }


def get_client() -> Any:
    """Returns a cached Supabase client, creating it on first use."""
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            cfg = settings()
            if cfg is None:
                raise RuntimeError("Supabase credentials not configured in environment")
            from supabase import create_client

            _CLIENT = create_client(cfg["url"], cfg["key"])
        return _CLIENT


def upload_result_files(
    text: str,
    structured: dict[str, Any],
    item_uid: str,
    pictures: list[tuple[str, bytes]] | None = None,
) -> dict[str, str]:
    """Uploads the converted outputs and returns kind -> public URL map."""
    cfg = settings()
    if cfg is None:
        LOGGER.info("SUPABASE_URL or SUPABASE_KEY is not set; converted files will not be uploaded")
        return {}

    client = get_client()
    bucket = cfg["bucket"]
    uploads: dict[str, tuple[bytes, str, str]] = {
        "markdown": (
            text.encode("utf-8"),
            "text/markdown; charset=utf-8",
            "processed.md",
        ),
        "json": (
            json.dumps(structured).encode("utf-8"),
            "application/json",
            "processed.json",
        ),
    }
    for name, data in pictures or []:
        uploads[f"picture/{name}"] = (
            data,
            PICTURE_CONTENT_TYPE,
            f"picture/{name}",
        )

    urls: dict[str, str] = {}
    for kind, (data, content_type, extension) in uploads.items():
        object_name = f"{item_uid}/{extension}"
        try:
            client.storage.from_(bucket).upload(
                path=object_name,
                file=data,
                file_options={"content-type": content_type, "upsert": "true"},
            )
            public_url = client.storage.from_(bucket).get_public_url(object_name)
            urls[kind] = public_url
            LOGGER.info("Uploaded %s result to %s", kind, public_url)
        except Exception as exc:
            LOGGER.error("Failed to upload %s (%s) to Supabase: %s", kind, object_name, exc)

    return urls


def absolute_url(item_uid: str, path: str) -> str:
    """Resolves an item-folder-relative artifact path to its public URL."""
    cfg = settings()
    if cfg is None:
        return path
    client = get_client()
    object_name = f"{item_uid}/{path}"
    return client.storage.from_(cfg["bucket"]).get_public_url(object_name)


def reset_client() -> None:
    """Drops the cached client (used by tests and config changes)."""
    global _CLIENT
    with _CLIENT_LOCK:
        _CLIENT = None
