"""Preload one position in the background."""

from __future__ import annotations

import logging
import time

from qtpy.QtCore import QSize, Qt, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QMessageBox,
    QProgressBar,
    QWidget,
)

from SECQUOIA.gui.loading_pipeline import (
    _PREFETCH_RAM_RESERVE_BYTES,
    _free_resources,
    _NextPositionPrefetch,
    _prefetch_needs,
    _prefetch_plan,
)
from SECQUOIA.gui.position_navigation import open_position_window, tree_counts
from SECQUOIA.utils.positions import position_number_from_folder
from SECQUOIA.utils.timing import current_t_range

LOG = logging.getLogger(__name__)

__all__ = ["PositionPreloader", "PreloadStatusBar", "ask_to_preload"]

_BAR_COLOURS = {
    "loading": "rgba(10, 132, 255, 0.55)",
    "ready": "rgba(48, 209, 88, 0.55)",
    "warn": "rgba(255, 159, 10, 0.6)",
}

_RETRY_SHOWN_S = 35.0


def _gb(n_bytes: float) -> str:
    return f"{n_bytes / 1024**3:.1f} GB"


def _loading_window_open() -> bool:
    return any(
        w.isVisible() and w.windowModality() != Qt.NonModal
        for w in QApplication.topLevelWidgets()
    )


def ask_to_preload(main_window, preloader=None) -> None:
    """Ask once per loaded experiment, after its loading windows are closed."""
    current = getattr(main_window, "position_preloader", None)
    preloader = preloader or current
    shown = isinstance(main_window, QWidget) and main_window.isVisible()
    if preloader is None or preloader is not current or not shown:
        return
    if _loading_window_open():
        QTimer.singleShot(500, lambda: ask_to_preload(main_window, preloader))
        return
    target = preloader._choose()
    if target is None:
        return
    if _prefetch_plan(main_window) is None:
        LOG.info("[preload] not offered: not enough free RAM or disk.")
        return
    raw_disk, basic_disk = _prefetch_needs(main_window)
    ram_free, disk_free = _free_resources()

    box = QMessageBox(main_window)
    box.setWindowTitle("Preload the next position?")
    box.setText(
        f"p{main_window.current_position_number:04d} is ready. SECQUOIA can "
        f"prepare p{target:04d} in the background while you work."
    )
    box.setInformativeText(
        f"RAM: {_gb(ram_free)} free, preloading keeps at least "
        f"{_gb(_PREFETCH_RAM_RESERVE_BYTES)} free\n"
        f"Disk: needs {_gb(raw_disk + basic_disk)}, {_gb(disk_free)} free"
    )
    box.addButton("No", QMessageBox.RejectRole)
    select = box.addButton("Select positions", QMessageBox.ActionRole)
    preload = box.addButton(f"Preload p{target:04d}", QMessageBox.AcceptRole)
    box.setDefaultButton(preload)
    box.exec_()

    clicked = box.clickedButton()
    if clicked is preload:
        preloader.set_enabled(True)
    elif clicked is select:
        preloader.set_enabled(True, start=False)
        dialog = open_position_window(main_window)
        if dialog is not None:
            dialog.finished.connect(lambda _result: preloader.schedule())


class PositionPreloader:
    """Chooses which position to preload and hands it over on a switch."""

    start_delay_ms = 2500

    def __init__(self, main_window) -> None:
        self._mw = main_window
        t_min, t_max, _, _ = current_t_range(main_window)
        self._prefetch = _NextPositionPrefetch(
            main_window, t_min, t_max, workers=2, read_measurements=True
        )
        self.enabled = False
        self.queue: list[int] = []
        self.target: int | None = None
        self.refused = False
        self._step = 1
        self._last = getattr(main_window, "current_position_number", None)
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._start)

    def set_enabled(self, enabled: bool, start: bool = True) -> None:
        self.enabled = bool(enabled)
        if self.enabled:
            if start:
                self.schedule()
        else:
            self._timer.stop()
            self._drop()

    def preload_next(self, pnum: int) -> None:
        self.queue = [pnum, *(p for p in self.queue if p != pnum)]
        self.schedule(0)

    def add(self, pnum: int) -> None:
        if self.target is not None and self.target not in self.queue:
            self.queue.insert(0, self.target)
        if pnum not in self.queue:
            self.queue.append(pnum)
        self.schedule()

    def remove(self, pnum: int) -> None:
        if pnum in self.queue:
            self.queue.remove(pnum)
        self.schedule(0)

    def clear(self) -> None:
        self.queue.clear()
        self.schedule(0)

    def schedule(self, delay_ms: int | None = None) -> None:
        """Pick and start the next preload."""
        if self.enabled:
            self._timer.start(
                self.start_delay_ms if delay_ms is None else delay_ms
            )

    def take(self, pnum: int):
        self._timer.stop()
        self._note_move(pnum)
        if pnum in self.queue:
            self.queue.remove(pnum)
        if self.target is None:
            return None
        if self.target != pnum:
            if self.target not in self.queue:
                self._drop()
            return None
        self.target = None
        return self._prefetch.take(pnum)

    def state(self) -> str:
        """off, idle, refused, loading, paused, ready or failed."""
        if not self.enabled:
            return "off"
        pending = self._prefetch.pending()
        if pending is None:
            return "refused" if self.refused else "idle"
        _pnum, future, gate = pending
        if future.done():
            return "failed" if future.exception() else "ready"
        return "paused" if gate.paused else "loading"

    def close(self) -> None:
        """Stop, the background files are removed once it ends."""
        self._timer.stop()
        self.target = None
        self._prefetch.close(wait=False)

    def _start(self) -> None:
        target = self._choose()
        if target == self.target and self._prefetch.pending() is not None:
            return
        self._drop()
        folder = self._folders().get(target)
        if folder is None:
            return
        plan = self._prefetch.start(target, folder)
        self.refused = plan is None
        self.target = target if plan else None

    def _drop(self) -> None:
        self._prefetch._discard()
        self.target = None
        self.refused = False

    def _choose(self) -> int | None:
        current = getattr(self._mw, "current_position_number", None)
        for pnum in self.queue:
            if pnum != current:
                return pnum
        numbers = list(self._folders())
        if current not in numbers:
            return None
        counts = tree_counts(self._mw)
        start = numbers.index(current)
        for k in range(1, len(numbers)):
            pnum = numbers[(start + k * self._step) % len(numbers)]
            tracked = counts is None or counts.get(pnum, 0) > 0
            if tracked:
                return pnum
        return None

    def _note_move(self, pnum: int) -> None:
        """Remember whether the user moves forwards or backwards."""
        numbers = list(self._folders())
        if self._last in numbers and pnum in numbers and pnum != self._last:
            moved = numbers.index(pnum) - numbers.index(self._last)
            self._step = 1 if moved > 0 else -1
        self._last = pnum

    def _folders(self) -> dict[int, str]:
        """Position number -> folder, in the order of the position list."""
        folders = {}
        for folder in getattr(self._mw, "position_folders", None) or []:
            pnum = position_number_from_folder(folder)
            if pnum is not None:
                folders[pnum] = folder
        return folders


class PreloadStatusBar(QProgressBar):
    """Preload status in the menu bar corner; a click opens the position window."""

    def __init__(self, main_window) -> None:
        super().__init__()
        self._mw = main_window
        self._colour = ""
        self.setFixedSize(160, 18)
        self.setRange(0, 100)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(500)
        self.refresh()

    def sizeHint(self) -> QSize:
        return QSize(160, 18)

    @property
    def _preloader(self):
        return getattr(self._mw, "position_preloader", None)

    def status(self) -> tuple[str, int, str | None]:
        """(text, percent, colour) to show right now."""
        preloader = self._preloader
        state = preloader.state() if preloader is not None else "off"
        if state == "off":
            return "Preload off", 0, None
        if state == "refused":
            return "Preload: low RAM/disk", 0, "warn"
        if state == "idle":
            return "Preload waiting", 0, None
        _pnum, _future, gate = preloader._prefetch.pending()
        name = f"p{preloader.target:04d}"
        percent = int(round(gate.progress * 100))
        if state == "ready":
            return f"{name} · ready", 100, "ready"
        if state == "failed":
            if getattr(gate, "stop_reason", None) == "disk":
                return "Stopped · low disk", percent, "warn"
            return f"{name} · failed", percent, "warn"
        if state == "paused":
            return "Paused · low RAM", percent, "warn"
        if time.monotonic() - gate.last_retry < _RETRY_SHOWN_S:
            return "Waiting for server", percent, "warn"
        return f"{name} · {percent}%", percent, "loading"

    def refresh(self) -> None:
        text, percent, colour = self.status()
        self.setValue(percent)
        self.setFormat(text)
        tooltip = text
        try:
            ram, disk = _free_resources()
            tooltip += f"\nRAM {_gb(ram)} · Disk {_gb(disk)} free"
        except (ImportError, OSError):
            pass
        self.setToolTip(tooltip)
        if colour != self._colour:
            self._colour = colour
            self.setStyleSheet(
                "QProgressBar { border: 1px solid #555; border-radius: 3px;"
                " font-size: 11px; }"
                "QProgressBar::chunk { background: "
                f"{_BAR_COLOURS.get(colour, 'transparent')}; }}"
            )

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            open_position_window(self._mw)
        else:
            super().mousePressEvent(event)
