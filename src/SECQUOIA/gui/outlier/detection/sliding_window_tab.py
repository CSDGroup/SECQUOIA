"""The Sliding window tab: local outlier detection over a time window."""

from __future__ import annotations

import contextlib

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from SECQUOIA.config import TOOLTIPSTEXT
from SECQUOIA.gui.common.grid_rows import (
    remove_grid_row,
    reposition_grid_rows,
    update_row_remove_buttons,
    use_qt_drawn_popup,
    wire_row_buttons,
)
from SECQUOIA.gui.common.ui_utils import (
    compact_combo,
    compact_spin,
    make_tab_scaffold,
    make_tab_title,
)

__all__ = ["SlidingWindowTab"]


class SlidingWindowTab:
    """Sliding-window rule builder: one row per feature/mask/channel window config."""

    _FIELD_KEYS = ("feat", "mask", "ch", "tmin", "tmax", "factor")

    def __init__(self, main_window, feature_keys, m_n, ch_n):
        self.main_window = main_window
        self.feature_keys = feature_keys
        self.m_n = m_n
        self.ch_n = ch_n

        self.rows_widgets = []
        main_window._sliding_rows_widgets = self.rows_widgets

        self.page = QWidget()
        content_v, side_v = make_tab_scaffold(self.page)

        content_v.addWidget(
            make_tab_title("Define one or more sliding windows:")
        )
        self.rows_grid = QGridLayout()
        self.rows_grid.setHorizontalSpacing(6)
        self.rows_grid.setVerticalSpacing(6)

        sw_header = [
            ("Feature", ""),
            ("Mask", ""),
            ("Channel", ""),
            ("Δt before", TOOLTIPSTEXT.TMIN_SP),
            ("Δt after", TOOLTIPSTEXT.TMAX_SP),
            ("± SD", TOOLTIPSTEXT.FACTOR_SD),
            ("", ""),
        ]

        for col, (text, tip) in enumerate(sw_header):
            lab = QLabel(text + ":" if text else "")
            lab.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
            lab.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            if tip:
                lab.setToolTip(tip)
            self.rows_grid.addWidget(lab, 0, col)

        content_v.addLayout(self.rows_grid)

        self.add_row()

        self.next_btn = QPushButton("Next →")
        self.next_btn.setToolTip(TOOLTIPSTEXT.NEXT_OUT)
        self.next_btn.setMinimumHeight(28)
        self.next_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        side_v.addWidget(self.next_btn)

    def summary_lines(self):
        """Build summary lines for sliding-window rules."""
        cfgs = self._rows_to_list()
        lines = []
        if not cfgs:
            lines.append("  (no sliding windows)")
            return lines
        for i, c in enumerate(cfgs, 1):
            m = "ALL" if (c.get("mask") is None) else str(int(c["mask"]))
            ch_val = c.get("channel")
            ch = "ALL" if ch_val is None else str(ch_val)
            lines.append(
                f"  {i}. feature={c['feature']} | mask={m} | channel={ch} | "
                f"t_min={int(c['t_min'])} | t_max={int(c['t_max'])} | SD factor={c['sd_factor']}"
            )
        return lines

    def _rows_to_list(self):
        """Serialize sliding-window UI rows into dictionaries."""
        out = []
        sw_rows = getattr(self.main_window, "_sliding_rows_widgets", None)
        if sw_rows:
            for _i, w in enumerate(sw_rows, 1):
                out.append(
                    {
                        "feature": w["feat"].currentText(),
                        "mask": w["mask"].currentData(),
                        "channel": w["ch"].currentData(),
                        "t_min": w["tmin"].value(),
                        "t_max": w["tmax"].value(),
                        "sd_factor": w["factor"].value(),
                    }
                )
        else:
            swp = getattr(self.main_window, "sliding_window_params", {})
            if swp:
                out.append(
                    {
                        "feature": swp["feature_combo"].currentText(),
                        "mask": (
                            swp["mask_combo"].currentData()
                            if "mask_combo" in swp
                            else None
                        ),
                        "channel": (
                            swp["channel_combo"].currentData()
                            if "channel_combo" in swp
                            else None
                        ),
                        "t_min": swp["t_min_spin"].value(),
                        "t_max": swp["t_max_spin"].value(),
                        "sd_factor": swp["factor_spin"].value(),
                    }
                )
        return out

    def _make_row_widgets(self):
        """Create widgets for one sliding-window outlier rule row."""
        feat_cb = compact_combo(
            use_qt_drawn_popup(QComboBox()), min_chars=10, max_w=170
        )
        for k in self.feature_keys:
            feat_cb.addItem(str(k), k)

        feat_cb.setToolTip(TOOLTIPSTEXT.FEAT_CB)

        mask_cb = compact_combo(
            use_qt_drawn_popup(QComboBox()), min_chars=4, max_w=90
        )
        if self.m_n > 0:
            for i in range(1, self.m_n + 1):
                mask_cb.addItem(str(i), i)
        else:
            mask_cb.addItem("—", None)
            mask_cb.setEnabled(False)

        mask_cb.setToolTip(TOOLTIPSTEXT.MASK_CB)

        ch_cb = compact_combo(
            use_qt_drawn_popup(QComboBox()), min_chars=6, max_w=110
        )
        if self.ch_n > 0:
            for channel in self.main_window.ids_channels:
                ch_cb.addItem(channel[1:], channel[1:])
        else:
            ch_cb.addItem("—", None)
            ch_cb.setEnabled(False)

        ch_cb.setToolTip(TOOLTIPSTEXT.CH_CB)

        tmin_sp = QDoubleSpinBox()
        tmin_sp.setDecimals(0)
        tmin_sp.setRange(0, 1e6)
        tmin_sp.setSingleStep(1)
        tmin_sp.setValue(0.0)
        tmin_sp.setToolTip(TOOLTIPSTEXT.TMIN_SP)
        compact_spin(tmin_sp, max_w=90)

        tmax_sp = QDoubleSpinBox()
        tmax_sp.setDecimals(0)
        tmax_sp.setRange(0, 1e6)
        tmax_sp.setSingleStep(1)
        tmax_sp.setValue(0.0)
        tmax_sp.setToolTip(TOOLTIPSTEXT.TMAX_SP)
        compact_spin(tmax_sp, max_w=90)

        factor_sp = QDoubleSpinBox()
        factor_sp.setDecimals(2)
        factor_sp.setRange(0, 1e6)
        factor_sp.setSingleStep(0.1)
        factor_sp.setValue(0.0)
        factor_sp.setToolTip(TOOLTIPSTEXT.FACTOR_SD)
        compact_spin(factor_sp, max_w=90)

        btn_bar = QWidget()
        btn_bar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        hb = QHBoxLayout(btn_bar)
        hb.setContentsMargins(0, 0, 0, 0)
        hb.setSpacing(4)
        add_btn = QPushButton("＋")
        add_btn.setToolTip(TOOLTIPSTEXT.ADD_BTN)
        add_btn.setMinimumWidth(28)
        rm_btn = QPushButton("✕")
        rm_btn.setToolTip(TOOLTIPSTEXT.RM_PLOT_BTN)
        rm_btn.setMinimumWidth(28)
        add_btn.setProperty("row_index", -1)
        rm_btn.setProperty("row_index", -1)
        hb.addWidget(add_btn)
        hb.addWidget(rm_btn)
        hb.addStretch(1)

        return {
            "feat": feat_cb,
            "mask": mask_cb,
            "ch": ch_cb,
            "tmin": tmin_sp,
            "tmax": tmax_sp,
            "factor": factor_sp,
            "btn_bar": btn_bar,
            "btn_add": add_btn,
            "btn_rm": rm_btn,
        }

    def _rebuild_grid_positions(self):
        """Reposition sliding window rule rows in the grid layout."""
        reposition_grid_rows(
            self.rows_grid, self.rows_widgets, self._FIELD_KEYS
        )

    def _update_remove_buttons(self):
        """Enable or disable sliding window remove buttons based on row count."""
        update_row_remove_buttons(self.rows_widgets)

    def add_row(self):
        """Append an empty sliding-window row at the end of the grid."""
        w = self._make_row_widgets()
        self.rows_widgets.append(w)
        wire_row_buttons(w, self._on_add_after, self._on_remove)

        self._rebuild_grid_positions()
        self._update_remove_buttons()

    def append_row_with_values(
        self, *, feature, mask, channel, t_min, t_max, sd_factor
    ):
        """Append a sliding-window row initialized with saved values."""
        w = self._make_row_widgets()
        self.rows_widgets.append(w)

        idx = w["feat"].findText(str(feature))
        if idx >= 0:
            w["feat"].setCurrentIndex(idx)

        if mask is not None:
            m_idx = w["mask"].findData(int(mask))
            if m_idx >= 0:
                w["mask"].setCurrentIndex(m_idx)
        if channel is not None:
            c_idx = w["ch"].findData(channel)
            if c_idx >= 0:
                w["ch"].setCurrentIndex(c_idx)

        w["tmin"].setValue(float(t_min))
        w["tmax"].setValue(float(t_max))
        w["factor"].setValue(float(sd_factor))

        wire_row_buttons(w, self._on_add_after, self._on_remove)

        self._rebuild_grid_positions()
        self._update_remove_buttons()

    def _on_remove(self, idx):
        """Remove a sliding window rule row and refresh the layout."""
        if remove_grid_row(
            self.rows_grid, self.rows_widgets, self._FIELD_KEYS, idx
        ):
            self._rebuild_grid_positions()
            self._update_remove_buttons()

    def _on_add_after(self, idx):
        """Insert a sliding-window rule row after the selected row."""
        if idx is None or idx < 0 or idx >= len(self.rows_widgets):
            return
        src = self.rows_widgets[int(idx)]
        w = self._make_row_widgets()
        w["feat"].setCurrentIndex(src["feat"].currentIndex())
        w["mask"].setCurrentIndex(src["mask"].currentIndex())
        w["ch"].setCurrentIndex(src["ch"].currentIndex())
        w["tmin"].setValue(src["tmin"].value())
        w["tmax"].setValue(src["tmax"].value())
        w["factor"].setValue(src["factor"].value())

        wire_row_buttons(w, self._on_add_after, self._on_remove)

        self.rows_widgets.insert(int(idx) + 1, w)
        self._rebuild_grid_positions()
        self._update_remove_buttons()

    def clear_rows(self):
        """Remove all sliding-window rows from the UI."""
        with contextlib.suppress(Exception):
            while self.rows_widgets:
                row = self.rows_widgets.pop(0)
                for w in [row[k] for k in self._FIELD_KEYS] + [row["btn_bar"]]:
                    with contextlib.suppress(Exception):
                        self.rows_grid.removeWidget(w)
                        w.setParent(None)
                        w.deleteLater()
