"""Dialog for editing marker, line and axis appearance in the plots."""

from __future__ import annotations

import logging

from qtpy.QtCore import Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import SECQUOIA.gui.lineage_tree.lineage_tree as lt
from SECQUOIA.config import PLOTPARAMETERS, TOOLTIPSTEXT
from SECQUOIA.gui.common.messages import show_folder_warning
from SECQUOIA.gui.common.ui_utils import spinbox_arrow_pngs
from SECQUOIA.utils.plotting import update_plot

LOG = logging.getLogger(__name__)

__all__ = [
    "PlotParamsDialog",
    "create_plot_params_dialog",
    "ensure_plot_params",
    "open_plot_params_dialog",
]

_DEFAULT_AXIS_FONT_SIZE = PLOTPARAMETERS.DEFAULT_LABEL_FONT_SIZE
_SYMBOL_SIZE_RANGE = (0, 64)
_LINE_WIDTH_RANGE = (0.5, 20.0)
_HL_LINE_WIDTH_RANGE = (0.5, 30.0)
_AXIS_FONT_RANGE = (6, 48)
_SPIN_MIN_WIDTH = 110

# Palette
_TEXT_COLOR = "#e6e6e6"
_DIALOG_BG = "#1c1c1c"
_PANEL_BG = "#2b2b2b"
_BORDER = "#444444"
_HOVER = "#3a3a3a"
_PRESSED = "#1f1f1f"

# Built once and reused; see ``_build_stylesheet``.
_STYLESHEET_CACHE: str | None = None


def _build_stylesheet() -> str:
    """Return the dialog stylesheet, generating the spinbox arrow images."""
    global _STYLESHEET_CACHE
    if _STYLESHEET_CACHE is not None:
        return _STYLESHEET_CACHE

    arrows = spinbox_arrow_pngs(_TEXT_COLOR)
    up_png = str(arrows["up"]).replace("\\", "/")
    down_png = str(arrows["down"]).replace("\\", "/")

    _STYLESHEET_CACHE = f"""
QDialog {{ background: {_DIALOG_BG}; }}
QLabel {{ color: {_TEXT_COLOR}; }}
QGroupBox {{
    color: {_TEXT_COLOR};
    border: 1px solid {_BORDER};
    border-radius: 6px;
    margin-top: 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0px 4px;
}}

QComboBox {{
    background: {_PANEL_BG};
    color: {_TEXT_COLOR};
    border: 1px solid {_BORDER};
    border-radius: 6px;
    padding: 4px;
}}
QComboBox QAbstractItemView {{
    background: {_PANEL_BG};
    color: {_TEXT_COLOR};
    selection-background-color: {_HOVER};
    border: 1px solid {_BORDER};
}}

QSpinBox, QDoubleSpinBox {{
    background: {_PANEL_BG};
    color: {_TEXT_COLOR};
    border: 1px solid {_BORDER};
    border-radius: 6px;
    padding: 4px 22px 4px 6px;   /* right room for the buttons */
    min-height: 20px;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 18px;
    background-color: {_PANEL_BG};
    border-left: 1px solid {_BORDER};
    border-bottom: 1px solid {_BORDER};
    border-top-right-radius: 6px;
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 18px;
    background-color: {_PANEL_BG};
    border-left: 1px solid {_BORDER};
    border-bottom-right-radius: 6px;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {_HOVER};
}}
QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {{
    background-color: {_PRESSED};
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url("{up_png}");
    width: 8px; height: 8px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url("{down_png}");
    width: 8px; height: 8px;
}}
QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled,
QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{
    image: none;
}}

QPushButton {{
    padding: 4px 12px;
    border-radius: 6px;
    background: {_PANEL_BG};
    color: {_TEXT_COLOR};
    border: 1px solid {_BORDER};
}}
QPushButton:hover {{ background: {_HOVER}; }}
QPushButton:pressed {{ background: {_PRESSED}; }}
"""
    return _STYLESHEET_CACHE


def ensure_plot_params(main_window) -> dict:
    """Make sure ``main_window._plot_params`` exists and is fully populated."""
    if not isinstance(getattr(main_window, "_plot_params", None), dict):
        main_window._plot_params = {}
    params = main_window._plot_params

    # Symbol size follows the app style when it defines one.
    default_symbol_size = getattr(
        getattr(main_window, "STYLE", object()),
        "SYMBOL_SIZE",
        PLOTPARAMETERS.DEFAULT_SYMBOL_SIZE,
    )

    params.setdefault("symbol", "o")
    params.setdefault("symbol_size", int(default_symbol_size))
    params.setdefault("line_width", 2.0)
    params.setdefault("hl_line_width", 3.0)
    params.setdefault("axis_font_size", _DEFAULT_AXIS_FONT_SIZE)
    return params


class PlotParamsDialog(QDialog):
    """Non-modal editor for marker, line and axis appearance."""

    def __init__(self, main_window, parent: QWidget | None = None):
        """Build the dialog and populate it from the current plot parameters."""
        super().__init__(parent if parent is not None else main_window)
        self.main_window = main_window
        self.params = ensure_plot_params(main_window)

        self.setWindowTitle("Plot parameters")
        self.setModal(False)
        self.setStyleSheet(_build_stylesheet())

        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        """Lay out the three settings groups and the button row."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        layout.addWidget(self._build_markers_group())
        layout.addWidget(self._build_lines_group())
        layout.addWidget(self._build_axes_group())
        layout.addLayout(self._build_button_row())

    def _build_markers_group(self) -> QGroupBox:
        """Build the marker symbol and size controls."""
        self.symbol_combo = QComboBox(self)
        for label, symbol_key in PLOTPARAMETERS.PG_SYMBOLS:
            self.symbol_combo.addItem(label, symbol_key)
        self.symbol_combo.setCurrentIndex(self._current_symbol_index())
        self.symbol_combo.setToolTip(TOOLTIPSTEXT.SYMBOL_COMBO)
        self.symbol_combo.setMinimumWidth(_SPIN_MIN_WIDTH)

        self.symbol_size_spin = QSpinBox(self)
        self.symbol_size_spin.setRange(*_SYMBOL_SIZE_RANGE)
        self.symbol_size_spin.setValue(int(self.params["symbol_size"]))
        self.symbol_size_spin.setToolTip(TOOLTIPSTEXT.SIZE_SPIN)
        self.symbol_size_spin.setMinimumWidth(_SPIN_MIN_WIDTH)

        return self._group(
            "Markers",
            [
                ("Symbol", self.symbol_combo),
                ("Symbol size", self.symbol_size_spin),
            ],
        )

    def _build_lines_group(self) -> QGroupBox:
        """Build the normal and highlighted line width controls."""
        self.line_width_spin = self._make_width_spin(
            _LINE_WIDTH_RANGE,
            float(self.params["line_width"]),
            TOOLTIPSTEXT.LINE_WIDTH_PLOT,
        )
        self.hl_line_width_spin = self._make_width_spin(
            _HL_LINE_WIDTH_RANGE,
            float(self.params["hl_line_width"]),
            TOOLTIPSTEXT.LINE_WIDTH_PLOT_HIGHLIGHT,
        )

        return self._group(
            "Lines",
            [
                ("Line width", self.line_width_spin),
                ("Highlight line width", self.hl_line_width_spin),
            ],
        )

    def _build_axes_group(self) -> QGroupBox:
        """Build the axis font size control."""
        self.axis_font_spin = QSpinBox(self)
        self.axis_font_spin.setRange(*_AXIS_FONT_RANGE)
        self.axis_font_spin.setValue(int(self.params["axis_font_size"]))
        self.axis_font_spin.setToolTip(TOOLTIPSTEXT.AXIS_SPIN_PLOT)
        self.axis_font_spin.setMinimumWidth(_SPIN_MIN_WIDTH)

        return self._group("Axes", [("Font size", self.axis_font_spin)])

    def _build_button_row(self) -> QHBoxLayout:
        """Build the Apply / OK / Cancel row."""
        self.apply_btn = QPushButton("Apply", self)
        self.ok_btn = QPushButton("OK", self)
        self.cancel_btn = QPushButton("Cancel", self)

        self.apply_btn.setToolTip(TOOLTIPSTEXT.APPLY_BTN_PLOT)
        self.ok_btn.setToolTip(TOOLTIPSTEXT.OK_BTN_PLOT)
        self.cancel_btn.setToolTip(TOOLTIPSTEXT.CANCEL_BTN_PLOT)

        row = QHBoxLayout()
        row.addStretch(1)
        for button in (self.apply_btn, self.ok_btn, self.cancel_btn):
            row.addWidget(button)
        return row

    def _connect_signals(self) -> None:
        """Wire the three buttons to their handlers."""
        self.apply_btn.clicked.connect(self.apply_changes)
        self.ok_btn.clicked.connect(self._on_ok)
        self.cancel_btn.clicked.connect(self.reject)

    def _group(self, title: str, rows: list[tuple[str, QWidget]]) -> QGroupBox:
        """Wrap labelled widgets in a titled group box with a form layout."""
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        for label, widget in rows:
            form.addRow(label, widget)

        group = QGroupBox(title)
        group.setLayout(form)
        return group

    def _make_width_spin(
        self, value_range: tuple[float, float], value: float, tooltip: str
    ) -> QDoubleSpinBox:
        """Create a line width spin box with the usual step and precision."""
        spin = QDoubleSpinBox(self)
        spin.setDecimals(1)
        spin.setRange(*value_range)
        spin.setSingleStep(0.5)
        spin.setValue(value)
        spin.setToolTip(tooltip)
        spin.setMinimumWidth(_SPIN_MIN_WIDTH)
        return spin

    def _current_symbol_index(self) -> int:
        """Find the combo index matching the stored symbol."""
        current = (
            self.params.get("symbol", PLOTPARAMETERS.DEFAULT_SYMBOL) or ""
        )
        keys = [key for _, key in PLOTPARAMETERS.PG_SYMBOLS]
        try:
            return keys.index(current)
        except ValueError:
            return 0

    def apply_changes(self) -> None:
        """Write the controls back to the plot parameters and redraw."""
        symbol = self.symbol_combo.currentData()
        self.params["symbol"] = (
            None if (symbol is None or str(symbol) == "") else str(symbol)
        )
        self.params["symbol_size"] = int(self.symbol_size_spin.value())
        self.params["line_width"] = float(self.line_width_spin.value())
        self.params["hl_line_width"] = float(self.hl_line_width_spin.value())
        axis_font_size = int(self.axis_font_spin.value())
        self.params["axis_font_size"] = axis_font_size

        lt.GEN_TEXT_FONT = QFont("Arial", axis_font_size)

        update_plot(self.main_window)
        lt.lineage_tree(self.main_window)

    def _on_ok(self) -> None:
        """Apply the changes and close the dialog."""
        self.apply_changes()
        self.accept()


def create_plot_params_dialog(main_window: QWidget) -> PlotParamsDialog:
    """Build the dynamics plot parameters dialog without showing it."""
    return PlotParamsDialog(main_window)


def open_plot_params_dialog(main_window: QWidget) -> PlotParamsDialog | None:
    """Open the dynamics plot parameters dialog, if an experiment is loaded."""
    if not getattr(main_window, "folder_list", None):
        LOG.warning("Please first load CSV file and select folder")
        show_folder_warning(main_window)
        return None

    dialog = PlotParamsDialog(main_window)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog
