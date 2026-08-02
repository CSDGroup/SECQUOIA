"""The black/white level histogram inspector dialog."""

import logging
from contextlib import suppress

import numpy as np
from PIL import Image
from qtpy.QtCore import (
    QRect,
    Qt,
)
from qtpy.QtGui import (
    QColor,
    QPainter,
    QPen,
)
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QVBoxLayout,
    QWidget,
)

from SECQUOIA.config import EXPORT

LOG = logging.getLogger(__name__)


class _BWInspector:
    """The black/white level histogram inspector dialog."""

    def _bw_prepare_crop(self, main_window, panel):
        """Return the grayscale crop for `panel` at the current time, for the histogram."""
        t = int(main_window.start_t_input.value())
        st = self._get_panel_state_for_render(panel)
        channel_id = main_window.ids_channels[st["chan_abs_idx"]]
        raw = Image.fromarray(main_window.images[channel_id][t])
        row = self._get_track_row_for_time(
            main_window,
            st["ident"],
            t,
            lineage_path=st.get("lineage_path", []),
        )
        if row is None:
            raise ValueError("No track data for current time.")
        x, y = self._extract_centroid(row)
        crop_w = int(main_window.crop_full_x_input.value())
        crop_h = int(main_window.crop_full_y_input.value())
        left, top, right, bottom = self._compute_crop_box(
            x, y, crop_w, crop_h, raw.width, raw.height
        )
        return raw.crop((left, top, right, bottom)).convert("L")

    def _bw_inspector_title(self, panel) -> str:
        """Window title naming which panel the inspector currently targets."""
        ident = panel.get("ident", "")
        chan = panel.get("chan_label_text", "")
        return (
            f"B/W Inspector — {ident} / {chan}" if ident else "B/W Inspector"
        )

    def _retarget_bw_inspector(self, main_window):
        """If the B/W Inspector is open, re-point it at the now-active panel.

        Called whenever panel selection changes so the inspector always edits
        whichever panel is currently selected, without needing to be reopened.
        """
        dlg = getattr(main_window, "_bw_inspector_dialog", None)
        histw = getattr(main_window, "_bw_inspector_hist", None)
        if dlg is None or histw is None:
            return
        panel = getattr(main_window, "active_panel", None)
        if panel is None:
            return
        try:
            if not dlg.isVisible():
                return
            base_crop = self._bw_prepare_crop(main_window, panel)
            histw.set_hist_from_image(base_crop)
            histw.set_bw(
                int(panel.get("black_point", EXPORT.BW_DEFAULT_BLACK)),
                int(panel.get("white_point", EXPORT.BW_DEFAULT_WHITE)),
            )
            dlg.setWindowTitle(self._bw_inspector_title(panel))
        except (ValueError, TypeError, IndexError, KeyError, RuntimeError):
            # RuntimeError covers a dialog whose underlying Qt widget was
            # already deleted; the other errors mean this panel has no
            # renderable frame yet (e.g. missing track data) — leave the
            # histogram showing the previous panel rather than erroring out.
            LOG.debug("B/W Inspector retarget skipped", exc_info=True)

    def _open_bw_inspector(self, main_window):
        """Open (or retarget/raise) the interactive B/W level histogram tool."""
        p = getattr(main_window, "active_panel", None)
        if not p:
            self._info(
                main_window, "B/W Inspector", "No active tile selected."
            )
            return

        existing = getattr(main_window, "_bw_inspector_dialog", None)
        if existing is not None:
            try:
                self._retarget_bw_inspector(main_window)
                existing.raise_()
                existing.activateWindow()
                return
            except RuntimeError:
                main_window._bw_inspector_dialog = None
                main_window._bw_inspector_hist = None

        try:
            base_crop = self._bw_prepare_crop(main_window, p)
        except (ValueError, TypeError, IndexError, KeyError) as e:
            self._warn(
                main_window,
                "B/W Inspector",
                f"Could not prepare histogram:\n{e}",
            )
            return

        ctrl = self

        class HistWidget(QWidget):
            """Interactive histogram widget for adjusting black and white intensity levels."""

            def __init__(self, parent=None):
                """Build a log scaled normalized 256-bin histogram from image."""
                super().__init__(parent)
                self.setMinimumSize(420, 180)
                self.margin = 18
                self._hist = np.zeros(256, dtype=np.float64)
                self._b = int(p.get("black_point", EXPORT.BW_DEFAULT_BLACK))
                self._w = int(p.get("white_point", EXPORT.BW_DEFAULT_WHITE))
                self._drag = None

            def set_hist_from_image(self, imgL: Image.Image):
                """Set black/white thresholds and refresh plot."""
                arr = np.array(imgL, dtype=np.uint8).ravel()
                hist, _ = np.histogram(arr, bins=256, range=(0, 255))
                h = np.log1p(hist.astype(np.float64))
                denom = max(1e-6, np.percentile(h, 99.0))
                self._hist = np.clip(h / denom, 0.0, 1.0)
                self.update()

            def set_bw(self, b, w):
                """Set black and white threshold values and refresh the histogram."""
                self._b = int(max(EXPORT.BW_MIN, min(self._w - 1, b)))
                self._w = int(min(EXPORT.BW_MAX, max(self._b + 1, w)))
                self.update()

            def bw(self):
                """Return the current ``(black_point, white_point)`` pair."""
                return self._b, self._w

            def _plot_rect(self) -> QRect:
                """Return the inner plotting rectangle for the histogram."""
                return QRect(
                    self.margin,
                    self.margin,
                    self.width() - 2 * self.margin,
                    self.height() - 2 * self.margin,
                )

            def _x_for_value(self, v: int) -> int:
                """Map pixel value [0..255] to x-coordinate in plot."""
                pr = self._plot_rect()
                return int(pr.left() + (v / 255.0) * pr.width())

            def _value_for_x(self, x: int) -> int:
                """Map x-coordinate in plot back to pixel value."""
                pr = self._plot_rect()
                if pr.width() <= 0:
                    return 0
                t = (x - pr.left()) / float(pr.width())
                return int(round(np.clip(t, 0.0, 1.0) * 255))

            def paintEvent(self, e):
                """Draw the histogram plot and its black/white drag handles."""
                qp = QPainter(self)
                qp.fillRect(self.rect(), QColor(255, 255, 255))
                pr = self._plot_rect()
                qp.setPen(QPen(QColor(180, 180, 180), 1))
                qp.drawRect(pr)
                if self._hist.size == 256 and pr.width() > 0:
                    bar_w = max(1, int(pr.width() / 256.0))
                    for i in range(256):
                        h = int(self._hist[i] * (pr.height() - 2))
                        x0 = pr.left() + int(i * (pr.width() / 256.0))
                        y0 = pr.bottom() - 1
                        qp.fillRect(
                            QRect(x0, y0 - h, bar_w, h), QColor(80, 80, 80)
                        )
                qp.fillRect(
                    QRect(
                        self._x_for_value(self._b),
                        pr.top(),
                        max(
                            1,
                            self._x_for_value(self._w)
                            - self._x_for_value(self._b),
                        ),
                        pr.height(),
                    ),
                    QColor(100, 180, 255, 60),
                )
                for v, col in (
                    (self._b, QColor(30, 120, 255)),
                    (self._w, QColor(255, 80, 60)),
                ):
                    x = self._x_for_value(v)
                    qp.setPen(QPen(col, 2))
                    qp.drawLine(x, pr.top(), x, pr.bottom())
                qp.setPen(QPen(QColor(30, 30, 30), 1))
                qp.drawText(
                    pr.adjusted(2, 2, -2, -2),
                    Qt.AlignLeft | Qt.AlignTop,
                    f"B:{self._b}",
                )
                qp.drawText(
                    pr.adjusted(2, 2, -2, -2),
                    Qt.AlignRight | Qt.AlignTop,
                    f"W:{self._w}",
                )
                qp.end()

            def mousePressEvent(self, ev):
                """Begin dragging nearest B/W handle on left-click."""
                if ev.button() != Qt.LeftButton:
                    return
                x = ev.pos().x()
                xb, xw = self._x_for_value(self._b), self._x_for_value(self._w)
                self._drag = "b" if abs(x - xb) <= abs(x - xw) else "w"
                self.mouseMoveEvent(ev)

            def mouseMoveEvent(self, ev):
                """Update B/W threshold while dragging a handle."""
                if self._drag is None:
                    return
                v = self._value_for_x(ev.pos().x())
                if self._drag == "b":
                    self._b = max(EXPORT.BW_MIN, min(self._w - 1, v))
                else:
                    self._w = min(EXPORT.BW_MAX, max(self._b + 1, v))
                self.update()
                on_change(self._b, self._w)

            def mouseReleaseEvent(self, ev):
                """End the B/W handle drag started by `mousePressEvent`."""
                self._drag = None

        dlg = QDialog(main_window.gif_window)
        dlg.setAttribute(Qt.WA_DeleteOnClose, True)
        dlg.setWindowTitle(self._bw_inspector_title(p))
        lay = QVBoxLayout(dlg)
        histw = HistWidget()
        histw.set_hist_from_image(base_crop)
        histw.set_bw(
            int(p.get("black_point", EXPORT.BW_DEFAULT_BLACK)),
            int(p.get("white_point", EXPORT.BW_DEFAULT_WHITE)),
        )
        lay.addWidget(histw)
        btns = QDialogButtonBox(QDialogButtonBox.Close)
        lay.addWidget(btns)

        def on_change(b, w):
            if getattr(main_window, "active_panel", None):
                main_window.active_panel["black_point"] = int(b)
                main_window.active_panel["white_point"] = int(w)
            with self.blocked(
                main_window.sel_black_spin, main_window.sel_white_spin
            ):
                main_window.sel_black_spin.setValue(int(b))
                main_window.sel_white_spin.setValue(int(w))
            ctrl._update_single_panel_preview(
                main_window, main_window.active_panel
            )

        btns.rejected.connect(dlg.close)

        def _on_closed():
            with suppress(RuntimeError, AttributeError):
                main_window._bw_inspector_dialog = None
                main_window._bw_inspector_hist = None

        dlg.destroyed.connect(_on_closed)
        main_window._bw_inspector_dialog = dlg
        main_window._bw_inspector_hist = histw

        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
