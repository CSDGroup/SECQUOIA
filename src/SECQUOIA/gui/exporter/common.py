"""Small stateless UI/formatting helpers shared across the exporter."""

import logging
import os
import subprocess
import sys
from contextlib import contextmanager, suppress

import numpy as np
import qtawesome as qta
from matplotlib import font_manager
from PIL import ImageFont
from qtpy.QtCore import (
    QUrl,
)
from qtpy.QtGui import (
    QDesktopServices,
)
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from SECQUOIA.config import EXPORT
from SECQUOIA.gui.common.ui_utils import is_widget_alive

LOG = logging.getLogger(__name__)


class _Common:
    """Small stateless formatting helpers shared across the exporter."""

    @staticmethod
    def _message_host(parent):
        host = getattr(parent, "gif_window", parent)
        return host if is_widget_alive(host) else None

    @staticmethod
    def _warn(parent, title, text):
        """Show a warning message box."""
        QMessageBox.warning(_Common._message_host(parent), title, text)

    @staticmethod
    def _info(parent, title, text):
        """Show an informational message box."""
        QMessageBox.information(_Common._message_host(parent), title, text)

    @staticmethod
    def _error(parent, title, text):
        """Show a critical error message box."""
        QMessageBox.critical(_Common._message_host(parent), title, text)

    @staticmethod
    def _get_combo_text(cb):
        """Get text from a QComboBox."""
        with suppress(Exception):
            return (cb.currentText() or "").strip()
        return ""

    @staticmethod
    def _get_combo_data(cb):
        """Get current item data from a QComboBox."""
        with suppress(Exception):
            return cb.currentData()
        return None

    @staticmethod
    def _btn_color(btn: QPushButton, rgb: tuple[int, int, int]):
        """Style a small square color button with RGB."""
        btn.setText("")
        btn.setFixedWidth(28)
        btn.setStyleSheet(
            f"QPushButton {{ background-color: rgb({rgb[0]},{rgb[1]},{rgb[2]}); border: 1px solid #444; }}"
        )

    @staticmethod
    def _compact_combo(cb: QComboBox, minlen=8, fixed_w=130):
        """Make a compact, fixed-width combo box."""
        cb.setMinimumContentsLength(minlen)
        cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        cb.setFixedWidth(fixed_w)

    @staticmethod
    def _compact_spin(sp: QSpinBox | QDoubleSpinBox, fixed_w=72):
        """Make a compact, fixed-width spin box."""
        sp.setFixedWidth(fixed_w)

    @staticmethod
    def _make_section(title: str, layout_cls=QGridLayout):
        """Titled, bordered section that renders the same on macOS and Windows."""
        wrapper = QWidget()
        wrap_v = QVBoxLayout(wrapper)
        wrap_v.setContentsMargins(0, 0, 0, 0)
        wrap_v.setSpacing(4)

        lbl = QLabel(title)
        lbl.setObjectName("section_title")
        wrap_v.addWidget(lbl)

        frame = QFrame()
        frame.setObjectName("section_frame")
        frame.setFrameShape(QFrame.NoFrame)
        inner = layout_cls(frame)
        inner.setContentsMargins(10, 10, 10, 10)
        wrap_v.addWidget(frame)

        return wrapper, frame, inner

    @staticmethod
    def _safe_name(s: str) -> str:
        s = str(s)
        return (
            "".join(
                c if c.isalnum() or c in ("-", "_") else "_" for c in s
            ).strip("_")
        ) or "unnamed"

    @contextmanager
    def blocked(self, *widgets):
        """Temporarily block signals for given widgets."""
        try:
            for w in widgets:
                with suppress(Exception):
                    w.blockSignals(True)
            yield
        finally:
            for w in widgets:
                with suppress(Exception):
                    w.blockSignals(False)

    @staticmethod
    def _fmt_hhmmss(seconds: float) -> str:
        """Format seconds as HH:MM:SS string."""
        s = int(round(max(0.0, seconds)))
        return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"

    def _get_font(self, pt_size: int, font_name: str | None = None):
        """Load font, matching the requested family name via matplotlib."""
        size = max(8, int(pt_size))
        name = str(font_name) if font_name else EXPORT.DEFAULT_FONT_NAME

        with suppress(Exception):
            path = font_manager.findfont(
                font_manager.FontProperties(family=name),
                fallback_to_default=True,
            )
            return ImageFont.truetype(path, size)

        LOG.warning(
            "Could not resolve font '%s' via matplotlib; falling back to "
            "Pillow's bundled default font.",
            name,
        )
        return ImageFont.load_default(size=size)

    def _binary_dilate(
        self, mask_bool: np.ndarray, iterations: int
    ) -> np.ndarray:
        """Perform binary dilation in NumPy."""
        if iterations <= 0:
            return mask_bool
        m = mask_bool.astype(bool, copy=False)
        for _ in range(int(iterations)):
            p = np.pad(m, 1, mode="constant", constant_values=False)
            m = (
                p[1:-1, 1:-1]
                | p[:-2, 1:-1]
                | p[2:, 1:-1]
                | p[1:-1, :-2]
                | p[1:-1, 2:]
                | p[:-2, :-2]
                | p[:-2, 2:]
                | p[2:, :-2]
                | p[2:, 2:]
            )
        return m

    def _open_folder(self, path: str):
        """Open a folder."""
        with suppress(Exception):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            return
        with suppress(Exception):
            if sys.platform.startswith("win"):
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])

    def _show_done_with_open(
        self, parent, title: str, text: str, open_path: str
    ):
        """Show ‘done’ dialog with optional button to open output folder."""
        host = _Common._message_host(parent)

        m = QMessageBox(host)
        m.setIcon(QMessageBox.Information)
        m.setWindowTitle(title)
        m.setText(text)
        open_btn = m.addButton("Open", QMessageBox.ActionRole)
        m.addButton(QMessageBox.Ok)
        m.exec_()
        with suppress(RuntimeError, AttributeError):
            host.raise_()
            host.activateWindow()

        if m.clickedButton() is open_btn:
            self._open_folder(open_path)

    def _qta_icon(self, *names, **kwargs):
        """Return qtawesome icon by name."""
        for nm in names:
            with suppress(Exception):
                return qta.icon(nm, **kwargs)
        return None
