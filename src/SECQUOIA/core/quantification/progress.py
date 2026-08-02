"""Progress reporting for the quantify pipeline."""

import time

from qtpy.QtWidgets import QApplication


class ProgressReporter:
    """Thin wrapper around the optional `progress_cb` of `quantify`."""

    def __init__(
        self, callback, total_steps: int, *, min_interval: float = 0.05
    ) -> None:
        self._callback = callback
        self._total = max(1, int(total_steps))
        self._min_interval = float(min_interval)
        self._last_emit = 0.0
        self.done = 0

    def tick(
        self,
        message: str | None = None,
        steps: int = 1,
        *,
        force: bool = False,
    ) -> None:
        """Advance the progress by `steps`; report it if it is time to."""
        self.done += steps
        if not callable(self._callback):
            return

        now = time.monotonic()
        if not force and (now - self._last_emit) < self._min_interval:
            return
        self._last_emit = now

        fraction = min(1.0, self.done / self._total)
        self._callback(
            fraction, message or f"Measuring {self.done}/{self._total}"
        )
        QApplication.processEvents()

    def stage(self, message: str) -> None:
        """Advance one step and always report it (used between pipeline stages)."""
        self.tick(message, force=True)

    def finish(self, message: str = "Measurements done") -> None:
        """Report 100% progress, ignoring the min interval throttle."""
        if callable(self._callback):
            self._callback(1.0, message)
