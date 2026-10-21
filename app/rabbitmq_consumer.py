"""RabbitMQ consumer that feeds library items into the Docling conversion service.

Messages are published by the NestJS API on the ``library.processing`` queue with:
{
    "event": "created" | "updated",
    "item": { ...library item row... },
    "fileUrl": "https://...",
    "occurredAt": "2026-10-21T10:00:00.000Z"
}

For every library item with a fileUrl, the consumer downloads the file,
converts it into structured AI-ready data with Docling and publishes the
result on the ``library.processed`` queue with:
{
    "event": "processed",
    "itemUid": "...",
    "occurredAt": "...",
    "status": "success" | "failed",
    "fileType": "DOCUMENT" | "IMAGE" | "AUDIO" | "VIDEO",
    "text": "... Markdown rendering with absolute picture links ...",
    "structured": { ... lossless Docling JSON document ... },
    "meta": { "engine": "docling", ... },
    "files": {
        "markdown": "https://.../<itemUid>/processed.md",
        "json": "https://.../<itemUid>/processed.json",
        "picture/picture-1.png": "https://.../<itemUid>/picture/picture-1.png"
    },
    "error": null | "... failure description ..."
}
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import pika

from app import converter, storage
from app.config import load_environment

LOGGER = logging.getLogger(__name__)

PROCESSING_QUEUE = "library.processing"
PROCESSED_QUEUE = "library.processed"

DEFAULT_DOWNLOAD_TIMEOUT = 120  # seconds


def rabbitmq_url() -> str | None:
    """AMQP connection URL; configured via RABBITMQ_URL in the environment."""
    load_environment()
    return os.getenv("RABBITMQ_URL")


def processing_queue() -> str:
    """Queue name to consume incoming library events; configured via RABBITMQ_PROCESSING_QUEUE."""
    load_environment()
    return os.getenv("RABBITMQ_PROCESSING_QUEUE", PROCESSING_QUEUE)


def processed_queue() -> str:
    """Queue name to publish processed results; configured via RABBITMQ_PROCESSED_QUEUE."""
    load_environment()
    return os.getenv("RABBITMQ_PROCESSED_QUEUE", PROCESSED_QUEUE)


def parse_message(body: bytes) -> dict[str, Any]:
    """Decodes a queue message payload; raises on malformed input."""
    event = json.loads(body)
    if not isinstance(event, dict):
        raise ValueError("message payload must be a JSON object")
    return event


def download_resource_file(file_url: str, item_uid: str) -> Path:
    timeout = float(os.getenv("DOCLING_DOWNLOAD_TIMEOUT", DEFAULT_DOWNLOAD_TIMEOUT))
    suffix = Path(urlparse(file_url).path).suffix or ""
    target = Path(tempfile.gettempdir()) / f"studymind-tool-{item_uid}{suffix}"
    LOGGER.info("Downloading file for %s from %s", item_uid, file_url)
    request = urllib.request.Request(file_url, headers={"User-Agent": "Mozilla/5.0"})
    with (
        urllib.request.urlopen(request, timeout=timeout) as response,
        target.open("wb") as output,
    ):
        output.write(response.read())
    return target


def build_result_payload(
    event: dict[str, Any],
    conversion: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    item = event.get("item", {})
    status = "success" if error is None else "failed"
    return {
        "event": "processed",
        "itemUid": item.get("uid"),
        "occurredAt": datetime.now(UTC).isoformat(),
        "status": status,
        "fileType": item.get("type"),
        "text": conversion.get("text") if conversion else None,
        "structured": conversion.get("structured") if conversion else None,
        "meta": conversion.get("meta") if conversion else None,
        "files": conversion.get("files") if conversion else None,
        "error": error,
    }


def publish_result(payload: dict[str, Any], url: str | None = None, queue: str | None = None) -> None:
    resolved_url = url or rabbitmq_url()
    resolved_queue = queue or processed_queue()
    if not resolved_url:
        LOGGER.error("Cannot publish result: RABBITMQ_URL is not configured")
        return

    connection = pika.BlockingConnection(pika.URLParameters(resolved_url))
    try:
        channel = connection.channel()
        channel.queue_declare(queue=resolved_queue, durable=True)
        channel.basic_publish(
            exchange="",
            routing_key=resolved_queue,
            body=json.dumps(payload).encode("utf-8"),
            properties=pika.BasicProperties(
                delivery_mode=2,
                content_type="application/json",
            ),
        )
    finally:
        connection.close()


def resolve_picture_refs(markdown: str, item_uid: str) -> str:
    return re.sub(
        r"\]\((picture/[^)\s]+)\)",
        lambda match: f"]({storage.absolute_url(item_uid, match.group(1))})",
        markdown,
    )


def process_library_item(event: dict[str, Any]) -> dict[str, Any]:
    item = event.get("item", {})
    item_uid = item.get("uid")
    file_type = item.get("type")
    metadata = item.get("metadata") or {}
    file_url = event.get("fileUrl") or metadata.get("fileUrl")

    LOGGER.info(
        "Processing library item uid=%s type=%s event=%s",
        item_uid,
        file_type,
        event.get("event"),
    )

    if not isinstance(item_uid, str):
        LOGGER.warning("Item event has no valid uid; reporting failure")
        return build_result_payload(event, error="item has no uid")

    if not file_url:
        LOGGER.info("Item %s has no file to convert; skipping", item_uid)
        return build_result_payload(event, error="item has no fileUrl")

    temp_path: Path | None = None
    try:
        temp_path = download_resource_file(file_url, item_uid)
        conversion = converter.convert_file(temp_path, file_type)
        conversion["text"] = resolve_picture_refs(conversion["text"], item_uid)
        conversion["files"] = storage.upload_result_files(
            conversion["text"],
            conversion["structured"],
            item_uid,
            pictures=conversion["pictures"],
        )
        payload = build_result_payload(event, conversion=conversion)
    except Exception as exc:
        LOGGER.exception("Conversion failed for library item %s", item_uid)
        payload = build_result_payload(event, error=str(exc))
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    publish_result(payload)
    return payload


def handle_message(
    channel: Any,
    method: SimpleNamespace,
    properties: Any,
    body: bytes,
) -> None:
    """pika consumer callback: parse, process, then ack/reject."""
    try:
        event = parse_message(body)
    except (ValueError, json.JSONDecodeError) as exc:
        LOGGER.error("Discarding malformed message: %s", exc)
        channel.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
        return

    try:
        process_library_item(event)
    except Exception:
        LOGGER.exception(
            "Processing failed for item %s",
            event.get("item", {}).get("uid"),
        )
        channel.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
        return

    channel.basic_ack(delivery_tag=method.delivery_tag)


def run_consumer(
    url: str | None = None,
    queue: str | None = None,
    on_message: Callable[..., None] = handle_message,
) -> None:
    """Blocks forever consuming library item events from the RabbitMQ queue."""
    resolved_url = url or rabbitmq_url()
    if not resolved_url:
        raise RuntimeError("RABBITMQ_URL is not configured; set it in Tools/.env")
    resolved_queue = queue or processing_queue()
    parameters = pika.URLParameters(resolved_url)
    parameters.heartbeat = 0
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    channel.queue_declare(queue=resolved_queue, durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(
        queue=resolved_queue,
        on_message_callback=on_message,
        auto_ack=False,
    )
    LOGGER.info("Library consumer listening on queue '%s'", resolved_queue)
    try:
        channel.start_consuming()
    finally:
        with suppress(pika.exceptions.ConnectionWrongStateError):
            connection.close()


class RabbitMQConsumerThread:
    """Runs the consumer on a daemon thread so FastAPI stays responsive."""

    def __init__(self, url: str | None = None, queue: str | None = None) -> None:
        self._url = url
        self._queue = queue
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=run_consumer,
            args=(self._url, self._queue),
            name="library-rabbitmq-consumer",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._thread = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_consumer()
