"""Reads the image and mask crops off the loaded stacks."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from SECQUOIA.gui.cell_inspector.geometry import CropWindow, crop_window

LOG = logging.getLogger(__name__)

DEFAULT_CACHE_ENTRIES = 256


@dataclass(frozen=True)
class CropRequest:
    """Everything needed to make one tile."""

    channel_id: str
    time_index: int
    center: tuple[float, float]
    width: int
    height: int
    mask_numbers: tuple[int, ...] = field(default=())


@dataclass(frozen=True)
class CropResult:
    """A finished tile: the image, one mask crop per mask, and the geometry."""

    request: CropRequest
    image: np.ndarray
    labels: tuple[tuple[int, np.ndarray], ...]
    window: CropWindow


def _plane(stack, time_index: int) -> np.ndarray | None:
    """Return one frame of a stack, or None if it is not there.

    Takes a 3D array or a list of 2D frames, since the loaders produce both.
    """
    if stack is None:
        return None
    try:
        length = len(stack)
    except TypeError:
        return None
    if not 0 <= time_index < length:
        return None

    try:
        frame = np.asarray(stack[time_index])
    except (IndexError, ValueError, TypeError) as exc:
        LOG.debug("cell inspector: unreadable frame %s: %s", time_index, exc)
        return None
    return frame if frame.ndim == 2 else None


class FrameSource:
    """Reads image and mask crops out of the loaded stacks."""

    def __init__(
        self, main_window, cache_entries: int = DEFAULT_CACHE_ENTRIES
    ):
        self._main_window = main_window
        self._cache_entries = max(1, int(cache_entries))
        self._cache: OrderedDict[CropRequest, CropResult] = OrderedDict()
        # read() runs on a background thread while invalidate() is called
        # from the GUI thread, so the cache is locked whenever it is used.
        self._lock = threading.Lock()

    def channel_ids(self) -> list[str]:
        """The channels that have images loaded."""
        images = getattr(self._main_window, "images", None)
        if not isinstance(images, dict):
            return []
        return [key for key, stack in images.items() if stack is not None]

    def mask_stack(self, mask_number: int):
        """Return the mask stack for a 1-based mask number, or None."""
        labels = getattr(self._main_window, "labels", None)
        if labels is None:
            return None
        try:
            index = int(mask_number) - 1
        except (TypeError, ValueError):
            return None
        if index < 0:
            return None

        if isinstance(labels, (list | tuple)):
            return labels[index] if index < len(labels) else None
        if isinstance(labels, np.ndarray):
            if labels.ndim == 4:
                return labels[index] if index < labels.shape[0] else None
            if labels.ndim == 3:
                return labels if index == 0 else None
        return None

    def mask_count(self) -> int:
        """How many masks are loaded."""
        labels = getattr(self._main_window, "labels", None)
        if isinstance(labels, (list | tuple)):
            return len(labels)
        if isinstance(labels, np.ndarray):
            if labels.ndim == 4:
                return int(labels.shape[0])
            if labels.ndim == 3:
                return 1
        return 0

    def read(self, request: CropRequest) -> CropResult | None:
        """Return the tile for a request, or None if that frame is missing."""
        cached = self._cache_get(request)
        if cached is not None:
            return cached

        images = getattr(self._main_window, "images", None)
        stack = (
            images.get(request.channel_id)
            if isinstance(images, dict)
            else None
        )
        frame = _plane(stack, request.time_index)
        if frame is None:
            return None

        height, width = frame.shape
        window = crop_window(
            request.center[0],
            request.center[1],
            request.width,
            request.height,
            width,
            height,
        )

        image = self._extract(frame, window, dtype=frame.dtype)

        labels: list[tuple[int, np.ndarray]] = []
        for mask_number in request.mask_numbers:
            label_frame = _plane(
                self.mask_stack(mask_number), request.time_index
            )
            if label_frame is not None and label_frame.shape == frame.shape:
                labels.append(
                    (mask_number, self._extract(label_frame, window, np.int32))
                )

        result = CropResult(
            request=request,
            image=image,
            labels=tuple(labels),
            window=window,
        )
        self._cache_put(request, result)
        return result

    @staticmethod
    def _extract(frame: np.ndarray, window: CropWindow, dtype) -> np.ndarray:
        """Cut the crop out of a frame and pad anything outside with zeros."""
        rows, cols = window.valid.slices()
        patch = np.asarray(frame[rows, cols], dtype=dtype)

        if not window.needs_padding:
            return np.array(patch, dtype=dtype, copy=True)

        tile = np.zeros(
            (window.requested.height, window.requested.width), dtype=dtype
        )
        if patch.size:
            row0, col0 = window.inset
            tile[
                row0 : row0 + patch.shape[0], col0 : col0 + patch.shape[1]
            ] = patch
        return tile

    def _cache_get(self, request: CropRequest) -> CropResult | None:
        """Get a tile from the cache and mark it as just used."""
        with self._lock:
            result = self._cache.get(request)
            if result is not None:
                self._cache.move_to_end(request)
            return result

    def _cache_put(self, request: CropRequest, result: CropResult) -> None:
        """Put a tile in the cache, dropping the oldest one when it is full."""
        with self._lock:
            self._cache[request] = result
            self._cache.move_to_end(request)
            while len(self._cache) > self._cache_entries:
                self._cache.popitem(last=False)

    def invalidate(self) -> None:
        """Throw away every cached tile."""
        with self._lock:
            self._cache.clear()
