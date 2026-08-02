"""Reload helper for file reads that may transiently fail.

Most useful when the experiment/segmentation folders live on a cloud-synced
drive."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from SECQUOIA.config import IORELOAD

LOG = logging.getLogger(__name__)


def _default_on_reload(fp: str, attempt: int, exc: Exception) -> None:
    LOG.warning(
        "Read failed (attempt %d): %s -> %r. Reloading...", attempt, fp, exc
    )


def read_with_reload(
    read_fn: Callable[[str], Any],
    fp: str,
    *,
    max_wait_s: float = IORELOAD.MAX_WAIT_S,
    base_delay_s: float = IORELOAD.BASE_DELAY_S,
    max_delay_s: float = IORELOAD.MAX_DELAY_S,
    on_reload: (
        Callable[[str, int, Exception], None] | None
    ) = _default_on_reload,
) -> Any:
    """Reloading with capped exponential backoff on failure."""
    start = time.monotonic()
    delay = base_delay_s
    attempt = 0
    while True:
        try:
            return read_fn(fp)
        except (OSError, ValueError, RuntimeError) as exc:
            attempt += 1
            remaining = max_wait_s - (time.monotonic() - start)
            if remaining <= 0:
                raise
            if on_reload is not None:
                on_reload(fp, attempt, exc)
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, max_delay_s)


def to_csv_with_reload(df: Any, path: str, **to_csv_kwargs: Any) -> None:
    """Write a DataFrame to CSV, reloading with backoff on transient I/O errors."""
    read_with_reload(lambda p: df.to_csv(p, **to_csv_kwargs), path)


def read_text_with_reload(path: str) -> str:
    """Read a text file, reloading with backoff if the read fails transiently."""

    def _read(p: str) -> str:
        with open(p, encoding="utf-8", errors="replace") as f:
            return f.read()

    return read_with_reload(_read, path)


def write_text_with_reload(path: str, text: str) -> None:
    """Write a text file, reloading with backoff if the write fails transiently."""

    def _write(p: str) -> None:
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)

    read_with_reload(_write, path)


def dump_json_with_reload(obj: Any, path: str, **json_kwargs: Any) -> None:
    """Write an object to a JSON file, reloading with backoff on transient I/O errors."""

    def _write(p: str) -> None:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, **json_kwargs)

    read_with_reload(_write, path)
