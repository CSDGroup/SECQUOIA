"""Reads crops on a background thread."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from qtpy.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal

from SECQUOIA.gui.cell_inspector.frame_source import (
    CropRequest,
    CropResult,
    FrameSource,
)

LOG = logging.getLogger(__name__)
WORKER_THREADS = 2
FOREGROUND_PRIORITY = 1
PREFETCH_PRIORITY = 0


class _ReadTask(QRunnable):
    """A single crop read, run on the thread pool."""

    def __init__(
        self,
        dispatcher: CropDispatcher,
        source: FrameSource,
        request: CropRequest,
        generation: int,
        deliver: bool,
    ):
        super().__init__()
        self._dispatcher = dispatcher
        self._source = source
        self._request = request
        self._generation = generation
        self._deliver = deliver

    def run(self) -> None:
        """Read the crop and hand it back to the GUI thread."""
        result = None
        if not self._dispatcher.is_stale(self._generation):
            try:
                result = self._source.read(self._request)
            except (
                RuntimeError,
                AttributeError,
                TypeError,
                ValueError,
                IndexError,
                OSError,
            ) as exc:
                LOG.warning(
                    "cell inspector: crop read failed for %s: %s",
                    self._request,
                    exc,
                )

        if not self._deliver:
            return
        self._dispatcher.completed.emit(
            self._request, result, self._generation
        )


class CropDispatcher(QObject):
    """Hands out crop reads so the newest request always wins."""

    ready = Signal(object)
    unavailable = Signal(object)
    completed = Signal(object, object, int)

    def __init__(
        self,
        source: FrameSource,
        parent: QObject | None = None,
        pool: QThreadPool | None = None,
    ):
        super().__init__(parent)
        self._source = source
        self._owns_pool = pool is None
        self._pool = pool if pool is not None else QThreadPool(self)
        if self._owns_pool:
            self._pool.setMaxThreadCount(WORKER_THREADS)

        self._generation = 0
        self._pending: CropRequest | None = None
        self._prefetch: tuple[CropRequest, ...] = ()
        self._busy = False
        self._running = True

        self.completed.connect(self._on_completed, Qt.QueuedConnection)

    def request(
        self,
        request: CropRequest,
        prefetch: Iterable[CropRequest] = (),
    ) -> None:
        """Show this request next and drop any that has not started yet."""
        if not self._running:
            return
        self._generation += 1
        self._pending = request
        self._prefetch = tuple(prefetch)
        if not self._busy:
            self._start_pending()

    def is_stale(self, generation: int) -> bool:
        """True if this round is out of date, or we are shutting down."""
        return not self._running or generation != self._generation

    def _start_pending(self) -> None:
        """Start the waiting request, if there is one."""
        if self._pending is None or not self._running:
            self._busy = False
            return
        request, self._pending = self._pending, None
        self._busy = True
        self._pool.start(
            _ReadTask(self, self._source, request, self._generation, True),
            FOREGROUND_PRIORITY,
        )

    def _on_completed(
        self, request: CropRequest, result: CropResult | None, generation: int
    ) -> None:
        """Show a finished read, then start whatever came in meanwhile."""
        if not self._running:
            return
        if generation == self._generation:
            if result is None:
                self.unavailable.emit(request)
            else:
                self.ready.emit(result)
            self._start_prefetch(generation)

        self._busy = False
        self._start_pending()

    def _start_prefetch(self, generation: int) -> None:
        """Read the neighbouring time points into the cache in the background."""
        for request in self._prefetch:
            self._pool.start(
                _ReadTask(self, self._source, request, generation, False),
                PREFETCH_PRIORITY,
            )
        self._prefetch = ()

    def shutdown(self, timeout_ms: int = 2000) -> None:
        """Stop handing out work and wait for the running reads to finish."""
        self._running = False
        self._pending = None
        self._prefetch = ()
        if not self._owns_pool:
            return
        self._pool.clear()
        if not self._pool.waitForDone(timeout_ms):
            LOG.warning(
                "cell inspector: crop reads still running after %d ms",
                timeout_ms,
            )
