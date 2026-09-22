"""Read-only views of the currently saved outlier-detection parameters."""

from __future__ import annotations

import contextlib
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from qtpy.QtCore import QSize, Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from SECQUOIA.config import STYLE, TOOLTIPSTEXT
from SECQUOIA.core.outlier_detection import resolve_feature_columns
from SECQUOIA.utils.intervals import (
    compute_bins,
    intersect_intervals,
    resolve_multi,
    threshold_regions,
    union_intervals,
)

__all__ = [
    "OutlierParametersDialog",
    "OutlierPlotsDialog",
    "collect_values",
    "find_feature_columns",
    "show_current_outlier_parameters",
]

# Threshold 1 region, threshold 2 region, and the combined outlier region.
_COLOR_RULE1 = "#1976d2"
_COLOR_RULE2 = "#642349"
_COLOR_COMBINED = "#FFA500"

_LINE_ALPHA = 0.9
_REGION_ALPHA = 0.12
_COMBINED_ALPHA = 0.15

# Size of each subplot in the exported composite, in inches.
_COMPOSITE_SUBPLOT_SIZE = (4.5, 3.6)
_COMPOSITE_DPI = 150

_PARAMS_DIALOG_SIZE = (900, 420)
_PLOTS_DIALOG_SIZE = (1200, 860)

_RULE_COLUMNS = [
    "#",
    "Feature",
    "Masks",
    "Channels",
    "Op1",
    "Val1",
    "Op2",
    "Val2",
    "Combine",
]

_SLIDING_WINDOW_COLUMNS = [
    "#",
    "Feature",
    "Mask",
    "Channel",
    "t_min",
    "t_max",
    "± SD factor",
]

_CLOSE_MASK_COLUMNS = ["#", "Distance ≤ (px)", "Masks"]

_FONT_STYLESHEET = f"""
QWidget {{
    font-size: {STYLE.FONT_SIZE_outlier}pt;
    font-family: "{STYLE.FONT_outlier}";
}}
"""


def format_selection(raw_list, resolved_list, total_count: int) -> str:
    """Describe a multi-selection of masks or channels for display."""
    if raw_list == [] and total_count > 0:
        return f"ALL (1..{total_count})"
    return ", ".join(str(x) for x in (resolved_list or [])) or "—"


def format_single(val, total_count: int) -> str:
    """Describe a single mask or channel selection for display."""
    if val is None and total_count > 0:
        return f"ALL (1..{total_count})"
    if val is None:
        return "—"
    try:
        return str(int(val))
    except (TypeError, ValueError):
        return str(val)


def find_feature_columns(
    df: pd.DataFrame, feat: str, masks: list, channels: list
) -> list:
    """Find the numeric columns matching a feature/mask/channel selection."""
    return resolve_feature_columns(df, feat, masks or None, channels or None)


def collect_values(
    df: pd.DataFrame, feat: str, masks: list, channels: list
) -> np.ndarray:
    """Gather every finite value for a feature/mask/channel selection."""
    cols = find_feature_columns(df, feat, masks, channels)
    if not cols:
        return np.array([], dtype=float)

    vals = []
    for c in cols:
        vc = pd.to_numeric(df[c], errors="coerce").to_numpy()
        if vc.size:
            vals.append(vc)

    x = np.concatenate(vals) if vals else np.array([], dtype=float)
    return x[np.isfinite(x)]


def unpack_rule(rule: dict, m_n: int, channel_ids) -> dict:
    """Normalise one saved rule into the fields the plotting code needs."""
    return {
        "feat": str(rule.get("feat", "")),
        "masks": resolve_multi(rule.get("masks", []), range(1, m_n + 1)),
        "channels": resolve_multi(
            rule.get("channels", []), [ch[1:] for ch in channel_ids]
        ),
        "op1": rule.get("op1"),
        "val1": rule.get("val1"),
        "op2": rule.get("op2"),
        "val2": rule.get("val2"),
        "combine": str(rule.get("combine", "OR")).upper(),
    }


def draw_rule_histogram(ax, x: np.ndarray, rule: dict) -> None:
    """Draw one rule's histogram with its threshold lines and shaded regions."""
    if x.size == 0:
        ax.text(
            0.5,
            0.5,
            "No finite data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        return

    _, edges, _ = ax.hist(x, bins=compute_bins(x))

    reg1, reg2 = [], []
    for op, val, color, target in (
        (rule["op1"], rule["val1"], _COLOR_RULE1, 1),
        (rule["op2"], rule["val2"], _COLOR_RULE2, 2),
    ):
        if op is None or val is None:
            continue
        regions = threshold_regions(op, float(val), edges)
        ax.axvline(float(val), linestyle="--", color=color, alpha=_LINE_ALPHA)
        for a, b in regions:
            ax.axvspan(float(a), float(b), alpha=_REGION_ALPHA, color=color)
        if target == 1:
            reg1 = regions
        else:
            reg2 = regions

    # The region that actually flags an outlier.
    if reg1 and reg2 and rule["combine"] == "AND":
        combined = intersect_intervals(reg1, reg2)
    elif not reg2:
        combined = reg1
    else:
        combined = union_intervals(reg1 + reg2)

    for a, b in combined:
        ax.axvspan(
            float(a), float(b), alpha=_COMBINED_ALPHA, color=_COLOR_COMBINED
        )

    ax.set_xlabel("Value")
    ax.set_ylabel("Count")


def _read_only_table(
    columns: list[str], row_count: int, parent: QWidget
) -> QTableWidget:
    """Create a stretched, non-editable, non-selectable table."""
    table = QTableWidget(row_count, len(columns), parent)
    table.setHorizontalHeaderLabels(columns)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setSelectionMode(QAbstractItemView.NoSelection)
    return table


def _fill_row(table: QTableWidget, row: int, values: list[str]) -> None:
    """Write one row of non-editable cells into a table."""
    for col, text in enumerate(values):
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        table.setItem(row, col, item)


class OutlierPlotsDialog(QDialog):
    """Grid of per rule histograms, with export to a composite image."""

    def __init__(
        self,
        main_window,
        rules: list[dict],
        m_n: int,
        ch_n: int,
        *,
        parent: QWidget | None = None,
        max_cols: int = 3,
    ):
        """Store the rules to plot and build the dialog."""
        super().__init__(parent or main_window)
        self.main_window = main_window
        self.rules = list(rules)
        self.m_n = m_n
        self.ch_n = ch_n
        self.max_cols = max_cols
        self.df = main_window.filtered_df

        self.setWindowTitle("Outlier Plots (read-only)")
        self.setWindowModality(Qt.ApplicationModal)
        self.resize(*_PLOTS_DIALOG_SIZE)
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            self.setStyleSheet(_FONT_STYLESHEET)

        self._build_ui()

    def _build_ui(self) -> None:
        """Lay out the scrollable plot grid and the button row."""
        outer = QVBoxLayout(self)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        outer.addWidget(scroll)

        inner = QWidget()
        inner.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scroll.setWidget(inner)

        grid = QGridLayout(inner)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)

        for i, rule in enumerate(self.rules):
            row, col = divmod(i, self.max_cols)
            grid.addWidget(self._build_rule_group(rule), row, col)

        outer.addLayout(self._build_button_row())

    def _build_rule_group(self, rule: dict) -> QGroupBox:
        """Build one titled histogram panel with its navigation toolbar."""
        cfg = unpack_rule(rule, self.m_n, self.main_window.ids_channels)

        group = QGroupBox()
        group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        group.setTitle(self._group_title(cfg))

        box = QVBoxLayout(group)
        box.setContentsMargins(6, 6, 6, 6)
        box.setSpacing(6)

        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        toolbar = NavigationToolbar(canvas, group)
        toolbar.setIconSize(QSize(16, 16))
        toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        if toolbar.layout():
            toolbar.layout().setContentsMargins(0, 0, 0, 0)
            toolbar.layout().setSpacing(0)

        box.addWidget(toolbar)
        box.addWidget(canvas)

        ax.clear()
        draw_rule_histogram(
            ax,
            collect_values(
                self.df, cfg["feat"], cfg["masks"], cfg["channels"]
            ),
            cfg,
        )
        fig.tight_layout(pad=0.35)
        canvas.draw_idle()
        return group

    @staticmethod
    def _group_title(cfg: dict) -> str:
        """Describe a rule in one line, for the panel caption."""
        title = (
            f"{cfg['feat']} • masks={cfg['masks'] or 'ALL'}, "
            f"ch={cfg['channels'] or 'ALL'}"
        )
        if cfg["op1"] is not None and cfg["val1"] is not None:
            title += f"  —  ({cfg['op1']} {cfg['val1']}"
            if cfg["op2"] is not None and cfg["val2"] is not None:
                title += f"  {cfg['combine']}  {cfg['op2']} {cfg['val2']}"
            title += ")"
        return title

    def _build_button_row(self) -> QHBoxLayout:
        """Build the Save / Close row, centred."""
        save_btn = QPushButton("Save Composite…", self)
        close_btn = QPushButton("Close", self)
        save_btn.setToolTip(TOOLTIPSTEXT.SAVE_COMPOSITE_BTN)
        close_btn.setToolTip(TOOLTIPSTEXT.CLOSE_PLOTS_BTN)
        save_btn.clicked.connect(self._on_save_composite)
        close_btn.clicked.connect(self.reject)

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(save_btn)
        row.addSpacing(10)
        row.addWidget(close_btn)
        row.addStretch(1)
        return row

    def _on_save_composite(self) -> None:
        """Export every rule's histogram into one image file."""
        fname = self._ask_composite_filename()
        if not fname:
            return

        if not self.rules:
            QMessageBox.information(
                self, "Nothing to Save", "There are no plots to save."
            )
            return

        cols = min(self.max_cols, len(self.rules))
        rows = int(math.ceil(len(self.rules) / cols))
        w_per, h_per = _COMPOSITE_SUBPLOT_SIZE
        fig = plt.figure(
            figsize=(cols * w_per, rows * h_per), dpi=_COMPOSITE_DPI
        )

        for i, rule in enumerate(self.rules):
            ax = fig.add_subplot(rows, cols, i + 1)
            cfg = unpack_rule(rule, self.m_n, self.main_window.ids_channels)
            ax.set_title(cfg["feat"])
            draw_rule_histogram(
                ax,
                collect_values(
                    self.df, cfg["feat"], cfg["masks"], cfg["channels"]
                ),
                cfg,
            )

        fig.tight_layout()
        try:
            fig.savefig(fname, bbox_inches="tight")
            QMessageBox.information(
                self, "Saved", f"Composite plots saved to:\n{fname}"
            )
        except (RuntimeError, AttributeError, TypeError) as exc:
            QMessageBox.warning(
                self, "Save Failed", f"Could not save composite image:\n{exc}"
            )

    def _ask_composite_filename(self) -> str:
        """Ask where to save, appending an extension matching the filter."""
        try:
            fname, selected = QFileDialog.getSaveFileName(
                self,
                "Save Composite Plots",
                "outlier_plots.png",
                "PNG Image (*.png);;PDF Document (*.pdf);;SVG Vector (*.svg)",
            )
        except (RuntimeError, AttributeError, TypeError):
            return ""

        if not fname:
            return ""

        if "." not in os.path.basename(fname):
            if "PDF" in (selected or ""):
                fname += ".pdf"
            elif "SVG" in (selected or ""):
                fname += ".svg"
            else:
                fname += ".png"
        return fname


class OutlierParametersDialog(QDialog):
    """Tables listing the saved outlier rules, sliding-window and close-mask settings."""

    def __init__(self, main_window, parent: QWidget | None = None):
        """Read the saved rules pack and build the parameter tables."""
        super().__init__(parent or main_window)
        self.main_window = main_window

        pack = getattr(main_window, "_last_outlier_rules", None) or {}
        self.rules = list(pack.get("rules") or [])
        self.m_n = int(pack.get("m_n", 0) or 0)
        self.ch_n = int(pack.get("ch_n", 0) or 0)
        self.sliding_windows = self._read_sliding_windows(pack)
        close_masks = pack.get("close_masks")
        self.close_masks = (
            close_masks if isinstance(close_masks, dict) else None
        )

        self.setModal(True)
        self.setWindowTitle("Outlier Detection – Current Parameters")
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            self.setStyleSheet(_FONT_STYLESHEET)

        self._build_ui()
        self.resize(*_PARAMS_DIALOG_SIZE)

    @staticmethod
    def _read_sliding_windows(pack: dict) -> list[dict]:
        """Read the sliding-window configs, tolerating the old single-config key."""
        sw_list = list(pack.get("sliding_windows") or [])
        if not sw_list:
            legacy = pack.get("sliding_window")
            if legacy:
                sw_list = [legacy]
        return sw_list

    def _build_ui(self) -> None:
        """Lay out the header, the two tables and the button row."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        if not (self.rules or self.sliding_windows or self.close_masks):
            layout.addWidget(self._header("No parameters have been selected."))

        if self.rules:
            layout.addWidget(
                self._header(
                    f"{len(self.rules)} rule(s) currently configured:"
                )
            )
            layout.addWidget(self._build_rules_table())

        if self.sliding_windows:
            layout.addWidget(
                self._header(
                    f"{len(self.sliding_windows)} sliding window config(s):"
                )
            )
            layout.addWidget(self._build_sliding_window_table())

        if self.close_masks:
            layout.addWidget(self._header("Close-mask detection:"))
            layout.addWidget(self._build_close_masks_table())

        layout.addLayout(self._build_button_row())

    @staticmethod
    def _header(text: str) -> QLabel:
        """Build a non-wrapping caption label."""
        label = QLabel(text)
        label.setWordWrap(False)
        label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        return label

    def _build_rules_table(self) -> QTableWidget:
        """Build the threshold-rules table."""
        table = _read_only_table(_RULE_COLUMNS, len(self.rules), self)

        for i, rule in enumerate(self.rules, start=1):
            op2 = rule.get("op2", None)
            val2 = rule.get("val2", "")
            op1 = rule.get("op1", "")
            val1 = rule.get("val1", "")

            _fill_row(
                table,
                i - 1,
                [
                    str(i),
                    str(rule.get("feat", "")),
                    format_selection(
                        rule.get("masks_raw", []),
                        rule.get("masks", []),
                        self.m_n,
                    ),
                    format_selection(
                        rule.get("channels_raw", []),
                        rule.get("channels", []),
                        self.ch_n,
                    ),
                    "" if op1 is None else str(op1),
                    "" if val1 is None else str(val1),
                    "None" if op2 is None else str(op2),
                    "" if (op2 is None or val2 is None) else str(val2),
                    str(rule.get("combine", "OR")).upper(),
                ],
            )
        return table

    def _build_sliding_window_table(self) -> QTableWidget:
        """Build the sliding-window configuration table."""
        table = _read_only_table(
            _SLIDING_WINDOW_COLUMNS, len(self.sliding_windows), self
        )

        for i, cfg in enumerate(self.sliding_windows, start=1):
            _fill_row(
                table,
                i - 1,
                [
                    str(i),
                    str(cfg.get("feature", "")),
                    format_single(cfg.get("mask", None), self.m_n),
                    format_single(cfg.get("channel", None), self.ch_n),
                    str(int(cfg.get("t_min", 0))),
                    str(int(cfg.get("t_max", 0))),
                    str(cfg.get("sd_factor", "")),
                ],
            )
        return table

    def _build_close_masks_table(self) -> QTableWidget:
        """Build the one-row close-mask detection table."""
        table = _read_only_table(_CLOSE_MASK_COLUMNS, 1, self)
        masks = [int(m) for m in self.close_masks.get("masks") or []]

        _fill_row(
            table,
            0,
            [
                "1",
                f"{float(self.close_masks.get('distance', 0)):g}",
                format_selection(masks, masks, self.m_n),
            ],
        )
        table.setMaximumHeight(
            table.horizontalHeader().sizeHint().height()
            + table.verticalHeader().length()
            + 2 * table.frameWidth()
            + 2
        )
        return table

    def _build_button_row(self) -> QHBoxLayout:
        """Build the Show Plots / OK row, centred."""
        show_btn = QPushButton("Show Plots", self)
        show_btn.setMinimumWidth(140)
        show_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        show_btn.setToolTip(TOOLTIPSTEXT.SHOW_PLOT_BTN)
        show_btn.setEnabled(bool(self.rules) or bool(self.sliding_windows))
        show_btn.clicked.connect(self._on_show_plots)

        ok_btn = QPushButton("OK", self)
        ok_btn.setMinimumWidth(120)
        ok_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ok_btn.setToolTip(TOOLTIPSTEXT.OK_BTN_OUT)
        ok_btn.clicked.connect(self.accept)

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(show_btn)
        row.addWidget(ok_btn)
        row.addStretch(1)
        return row

    def _on_show_plots(self) -> None:
        """Open the histogram view for the current rules."""
        df = getattr(self.main_window, "filtered_df", None)
        if df is None or df.empty:
            QMessageBox.information(self, "No Data", "filtered_df is empty.")
            return

        OutlierPlotsDialog(
            self.main_window,
            self.rules,
            self.m_n,
            self.ch_n,
            parent=self,
        ).exec_()


def show_current_outlier_parameters(main_window) -> None:
    """Show the most recently saved outlier parameters."""
    OutlierParametersDialog(main_window).exec_()
