"""Dialog for editing lineage tree and heatmap styling."""

from __future__ import annotations

import contextlib
import re

from qtpy.QtCore import Qt
from qtpy.QtGui import QCloseEvent, QColor, QFont
from qtpy.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import SECQUOIA.gui.lineage_tree.lineage_draw as ld
import SECQUOIA.gui.lineage_tree.lineage_tree as lt
from SECQUOIA.config import TOOLTIPSTEXT
from SECQUOIA.gui.common.ui_utils import use_fusion_combos
from SECQUOIA.gui.dialogs.plot_params_dialog import ensure_plot_params
from SECQUOIA.utils.plotting import update_plot

__all__ = ["LineageStyleDialog", "open_lineage_style_dialog"]


def open_lineage_style_dialog(main_window: QWidget) -> None:
    """Open the Lineage/HeatTree style dialog."""
    with contextlib.suppress(Exception):
        dlg = getattr(main_window, "_lineage_style_dialog", None)
    if "dlg" not in locals():
        dlg = None

    reuse = False
    if dlg is not None:
        with contextlib.suppress(Exception):
            if dlg.isVisible():
                reuse = True

    if reuse:
        with contextlib.suppress(Exception):
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
            return

    dlg = LineageStyleDialog(parent=main_window)
    with contextlib.suppress(Exception):
        dlg.destroyed.connect(
            lambda *_: setattr(main_window, "_lineage_style_dialog", None)
        )

    main_window._lineage_style_dialog = dlg
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()


class LineageStyleDialog(QDialog):
    """Dialog for editing lineageTree and heatmap styling."""

    def __init__(self, parent=None):
        """Initialize the lineage-style dialog and populate controls from current settings."""
        super().__init__(parent)
        self.setWindowTitle("Lineage/Heat Tree")
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        self.mode_combo = QComboBox()
        self.mode_combo.setToolTip(TOOLTIPSTEXT.COMBO_MODE)
        self.mode_combo.addItems(["T (Tree)", "H (Heatmap)"])
        self.mode_combo.setCurrentIndex(
            0 if self._get_current_mode() == "T" else 1
        )

        self.line_width_sb = QDoubleSpinBox()
        self.line_width_sb.setToolTip(TOOLTIPSTEXT.LINE_WIDTH)
        self.line_width_sb.setRange(0.1, 50.0)
        self.line_width_sb.setSingleStep(0.1)
        self.line_width_sb.setDecimals(2)
        self.line_width_sb.setValue(float(ld.LINEWIDTH))

        self.connector_width_sb = QDoubleSpinBox()
        self.connector_width_sb.setToolTip(TOOLTIPSTEXT.CONNECTOR_WIDTH_SB)
        self.connector_width_sb.setRange(0.1, 50.0)
        self.connector_width_sb.setSingleStep(0.1)
        self.connector_width_sb.setDecimals(2)
        self.connector_width_sb.setValue(float(ld.CONNECTOR_WIDTH))

        tree_form = QFormLayout()
        tree_form.addRow("Lineage line width", self.line_width_sb)
        tree_form.addRow("Connector line width", self.connector_width_sb)
        self.tree_grp = QGroupBox("Tree (T) parameters")
        self.tree_grp.setLayout(tree_form)
        self.tree_grp.setToolTip(TOOLTIPSTEXT.T_MODE)

        self.heat_width_sb = QDoubleSpinBox()
        self.heat_width_sb.setToolTip(TOOLTIPSTEXT.HEAT_WIDTH_SB)
        self.heat_width_sb.setRange(0.1, 100.0)
        self.heat_width_sb.setSingleStep(0.5)
        self.heat_width_sb.setDecimals(1)
        self.heat_width_sb.setValue(float(ld.HEAT_LINE_WIDTH))

        self.heat_width_multi_sb = QDoubleSpinBox()
        self.heat_width_multi_sb.setToolTip(TOOLTIPSTEXT.HEAT_WIDTH_MULTI_SB)
        self.heat_width_multi_sb.setRange(0.1, 100.0)
        self.heat_width_multi_sb.setSingleStep(0.5)
        self.heat_width_multi_sb.setDecimals(1)
        self.heat_width_multi_sb.setValue(float(ld.HEAT_LINE_WIDTH_MULTI))

        self.low_btn = QPushButton()
        self.low_btn.setToolTip(TOOLTIPSTEXT.LOW_BTN)
        self.high_btn = QPushButton()
        self.high_btn.setToolTip(TOOLTIPSTEXT.HIGH_BTN)
        self._init_color_button(self.low_btn, ld.HEAT_LOW_COLOR)
        self._init_color_button(self.high_btn, ld.HEAT_HIGH_COLOR)

        self.multi_btns = [QPushButton() for _ in range(5)]
        for i, btn in enumerate(self.multi_btns):
            self._init_color_button(
                btn,
                ld.HEAT_MULTI_LOW_COLORS[i % len(ld.HEAT_MULTI_LOW_COLORS)],
            )

        heat_form = QFormLayout()
        heat_form.addRow("Heat lane width (single-ch)", self.heat_width_sb)
        heat_form.addRow(
            "Heat lane width (multi-ch)", self.heat_width_multi_sb
        )
        heat_form.addRow("Single-ch LOW color", self.low_btn)
        heat_form.addRow("Single-ch HIGH color", self.high_btn)

        lanes_box = QHBoxLayout()
        lanes_box.addWidget(QLabel("Multi-ch lane lows:"))
        for btn in self.multi_btns:
            lanes_box.addWidget(btn)
        lanes_wrap = QWidget()
        lanes_wrap.setLayout(lanes_box)

        self.heat_grp = QGroupBox("Heatmap (H) parameters")
        heat_v = QVBoxLayout()
        heat_v.addLayout(heat_form)
        heat_v.addWidget(lanes_wrap)
        self.heat_grp.setLayout(heat_v)
        self.heat_grp.setToolTip(TOOLTIPSTEXT.HEAT_MODE)

        self.show_labels_cb = QCheckBox("Show labels")
        self.show_labels_cb.setToolTip(TOOLTIPSTEXT.SHOW_LABEL_CB)
        self.show_labels_cb.setChecked(bool(ld.SHOW_TRACK_LABELS))

        self.label_mode_combo = QComboBox()
        self.label_mode_combo.setToolTip(TOOLTIPSTEXT.T_G_COMBO)
        self.label_mode_combo.addItems(["Track number", "Generation"])
        self.label_mode_combo.setCurrentIndex(
            0 if ld.LABEL_MODE == "track" else 1
        )

        self.gen_font_sb = QSpinBox()
        self.gen_font_sb.setToolTip(TOOLTIPSTEXT.FONT_SB)
        self.gen_font_sb.setRange(6, 48)
        with contextlib.suppress(Exception):
            self.gen_font_sb.setValue(int(ld.GEN_TEXT_FONT.pointSize()))
        if self.gen_font_sb.value() == 0:
            self.gen_font_sb.setValue(10)

        labels_form = QFormLayout()
        labels_form.addRow(self.show_labels_cb)
        labels_form.addRow("Label content", self.label_mode_combo)
        labels_form.addRow("Label font size", self.gen_font_sb)
        self.labels_grp = QGroupBox("Labels")
        self.labels_grp.setLayout(labels_form)
        self.labels_grp.setToolTip(TOOLTIPSTEXT.LABEL_MODE)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setToolTip(TOOLTIPSTEXT.APPLY_BTN_L)
        self.ok_btn = QPushButton("OK")
        self.ok_btn.setToolTip(TOOLTIPSTEXT.OK_BTN_L)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setToolTip(TOOLTIPSTEXT.CANCEL_BTN_L)

        btns = QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(self.apply_btn)
        btns.addWidget(self.ok_btn)
        btns.addWidget(self.cancel_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Mode"))
        layout.addWidget(self.mode_combo)
        layout.addWidget(self.tree_grp)
        layout.addWidget(self.heat_grp)
        layout.addWidget(self.labels_grp)
        layout.addSpacing(8)
        layout.addLayout(btns)

        self._apply_mode_visibility(self._get_current_mode())

        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.apply_btn.clicked.connect(self.apply_changes)
        self.ok_btn.clicked.connect(self.accept_and_apply)
        self.cancel_btn.clicked.connect(self.reject)

        self.low_btn.clicked.connect(lambda: self._pick_single_color("low"))
        self.high_btn.clicked.connect(lambda: self._pick_single_color("high"))
        for i, btn in enumerate(self.multi_btns):
            btn.clicked.connect(lambda _, idx=i: self._pick_multi_color(idx))

        use_fusion_combos(self)

    def _get_current_mode(self) -> str:
        """Return the currently selected lineage styling mode."""
        with contextlib.suppress(Exception):
            tools = getattr(self.parent(), "lineage_tools", None)
            if tools and "F" in tools:
                txt = tools["F"].currentText()
                return "H" if (str(txt).upper().startswith("H")) else "T"
        return "T"

    def _set_current_mode(self, mode: str):
        """Select the requested lineage styling mode in the UI."""
        mode = "H" if str(mode).upper().startswith("H") else "T"
        with contextlib.suppress(Exception):
            tools = getattr(self.parent(), "lineage_tools", None)
            if tools and "F" in tools:
                w = tools["F"]
                prev = w.blockSignals(True)
                try:
                    w.setCurrentText(mode)
                finally:
                    w.blockSignals(prev)

    def _apply_mode_visibility(self, mode: str):
        """Show the parameter group for the active mode, hide the other."""
        is_tree = mode == "T"
        self.tree_grp.setVisible(is_tree)
        self.heat_grp.setVisible(not is_tree)
        self._force_dialog_relayout()

    def _on_mode_changed(self, _i):
        """Handle lineage style mode changes and refresh control visibility.

        Only toggles which parameter group is shown; the mode itself is not
        pushed to the main window until Apply/OK is clicked.
        """
        mode = "T" if self.mode_combo.currentIndex() == 0 else "H"
        self._apply_mode_visibility(mode)

    def _init_color_button(self, btn: QPushButton, qcolor: QColor):
        """Initialize a color button."""
        btn.setFixedWidth(60)
        btn.setFixedHeight(22)
        self._paint_button_swatch(btn, qcolor)

    def _paint_button_swatch(self, btn: QPushButton, qcolor: QColor):
        """Paint a button."""
        css = f"background-color: rgb({qcolor.red()}, {qcolor.green()}, {qcolor.blue()});"
        btn.setStyleSheet(css)

    def _pick_single_color(self, which: str):
        """Open a color picker for a single color setting."""
        cur = ld.HEAT_LOW_COLOR if which == "low" else ld.HEAT_HIGH_COLOR
        qcol = QColorDialog.getColor(
            cur, self, f"Pick {'LOW' if which=='low' else 'HIGH'} color"
        )
        if not qcol.isValid():
            return
        if which == "low":
            self._paint_button_swatch(self.low_btn, qcol)
        else:
            self._paint_button_swatch(self.high_btn, qcol)

    def _pick_multi_color(self, idx: int):
        """Open a color picker for one value in a multi-color setting."""
        cur = ld.HEAT_MULTI_LOW_COLORS[idx % len(ld.HEAT_MULTI_LOW_COLORS)]
        qcol = QColorDialog.getColor(cur, self, f"Pick lane {idx+1} LOW color")
        if not qcol.isValid():
            return
        self._paint_button_swatch(self.multi_btns[idx], qcol)

    def apply_changes(self):
        """Write the controls to the lineage_draw globals and redraw."""
        ld.LINEWIDTH = float(self.line_width_sb.value())
        ld.CONNECTOR_WIDTH = float(self.connector_width_sb.value())

        ld.HEAT_LINE_WIDTH = float(self.heat_width_sb.value())
        ld.HEAT_LINE_WIDTH_MULTI = float(self.heat_width_multi_sb.value())

        font_size = int(self.gen_font_sb.value())
        ld.GEN_TEXT_FONT = QFont("Arial", font_size)
        ld.SHOW_TRACK_LABELS = bool(self.show_labels_cb.isChecked())
        ld.LABEL_MODE = (
            "gen" if self.label_mode_combo.currentIndex() == 1 else "track"
        )

        ld.HEAT_LOW_COLOR = self._qcolor_from_btn(
            self.low_btn, fallback=ld.HEAT_LOW_COLOR
        )
        ld.HEAT_HIGH_COLOR = self._qcolor_from_btn(
            self.high_btn, fallback=ld.HEAT_HIGH_COLOR
        )

        new_multi = []
        for i, btn in enumerate(self.multi_btns):
            new_multi.append(
                self._qcolor_from_btn(
                    btn, fallback=ld.HEAT_MULTI_LOW_COLORS[i]
                )
            )
        ld.HEAT_MULTI_LOW_COLORS = new_multi

        self._set_current_mode(
            "T" if self.mode_combo.currentIndex() == 0 else "H"
        )
        mw = self.parent()
        if mw is not None:
            ensure_plot_params(mw)["axis_font_size"] = font_size
            lt.lineage_tree(mw)
            update_plot(mw)

    def _qcolor_from_btn(self, btn: QPushButton, fallback: QColor) -> QColor:
        """Return the colour parsed back out of a button's stylesheet."""
        with contextlib.suppress(Exception):
            css = btn.styleSheet()
            m = re.search(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", css)
            if m:
                r, g, b = map(int, m.groups())
                return QColor(r, g, b)
        return QColor(fallback)

    def accept_and_apply(self):
        """Apply changes and accept the dialog."""
        self.apply_changes()
        self.accept()

    def closeEvent(self, ev: QCloseEvent):
        """Handle dialog close events without automatically applying changes."""
        with contextlib.suppress(Exception):
            mw = self.parent()
            if (
                mw is not None
                and getattr(mw, "_lineage_style_dialog", None) is self
            ):
                mw._lineage_style_dialog = None
        super().closeEvent(ev)

    def _force_dialog_relayout(self):
        """Force the dialog to recompute layout and fit its contents."""
        with contextlib.suppress(Exception):
            lay = self.layout()
            if lay is not None:
                lay.invalidate()
                lay.activate()
        with contextlib.suppress(Exception):
            self.adjustSize()
            self.resize(self.sizeHint())
