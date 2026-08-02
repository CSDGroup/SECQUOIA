"""The Threshold Rules tab: static per feature comparison rules."""

from __future__ import annotations

import contextlib

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
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
from SECQUOIA.gui.outlier.preview_dialog import _open_outlier_preview_dialog
from SECQUOIA.gui.outlier.widgets import CheckableComboBox, set_combo_checks

__all__ = ["ThresholdRulesTab"]


class ThresholdRulesTab:
    """Static threshold-rule builder: one row per feature/mask/channel comparison."""

    _FIELD_KEYS = (
        "enabled",
        "feat",
        "m_multi",
        "ch_multi",
        "op1",
        "val1",
        "op2",
        "val2",
        "combine",
    )

    def __init__(self, main_window, win, feature_keys, m_n, ch_n, channel_ids):
        self.main_window = main_window
        self.win = win
        self.feature_keys = feature_keys
        self.m_n = m_n
        self.ch_n = ch_n
        self.channel_ids = channel_ids

        self.rows_widgets = []
        main_window._rows_widgets = self.rows_widgets

        self.page = QWidget()
        content_v, side_v = make_tab_scaffold(self.page)

        content_v.addWidget(make_tab_title("Define outlier rule(s):"))

        self.rows_grid = QGridLayout()
        self.rows_grid.setHorizontalSpacing(6)
        self.rows_grid.setVerticalSpacing(4)
        self.rows_grid.setContentsMargins(0, 0, 0, 0)

        header_labels = [
            ("On", TOOLTIPSTEXT.RULE_ENABLED_CB),
            ("Feature", ""),
            ("Mask(s)", ""),
            ("Channel(s)", ""),
            ("Op1", ""),
            ("Val1", ""),
            ("Op2", ""),
            ("Val2", ""),
            ("Combine", ""),
            ("", ""),
        ]
        for col, (text, tip) in enumerate(header_labels):
            lab = QLabel(text + ":" if text else "")
            lab.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
            lab.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            if tip:
                lab.setToolTip(tip)
            self.rows_grid.addWidget(lab, 0, col)

        content_v.addLayout(self.rows_grid)

        self.add_row()

        preview_btn = QPushButton("Preview")
        preview_btn.setToolTip(TOOLTIPSTEXT.PREVIEW_BTN)
        preview_btn.setMinimumHeight(28)
        preview_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.next_btn = QPushButton("Next →")
        self.next_btn.setToolTip(TOOLTIPSTEXT.NEXT_BTN)
        self.next_btn.setMinimumHeight(28)
        self.next_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        side_v.addWidget(preview_btn)
        side_v.addWidget(self.next_btn)

        preview_btn.clicked.connect(self._on_preview)

    def summary_lines(self):
        """Build summary lines for threshold outlier rules."""
        lines = []
        if not self.rows_widgets:
            lines.append("  (no static threshold rules)")
            return lines
        for i, w in enumerate(self.rows_widgets, 1):
            status = "" if w["enabled"].isChecked() else "[DISABLED] "
            feat = w["feat"].currentText()
            masks = w["m_multi"].selected_values()
            chans = w["ch_multi"].selected_values()
            op1 = w["op1"].currentText()
            v1 = w["val1"].value()
            op2_data = w["op2"].currentData()
            if op2_data is None:
                cond = f"{op1} {v1}"
            else:
                op2 = w["op2"].currentText()
                v2 = w["val2"].value()
                comb = w["combine"].currentText()
                cond = f"{op1} {v1} {comb} {op2} {v2}"
            mtxt = "ALL" if not masks else ",".join(map(str, masks))
            ctxt = "ALL" if not chans else ",".join(map(str, chans))
            lines.append(
                f"  {i}. {status}feature={feat} | masks={mtxt} | channels={ctxt} | rule=({cond})"
            )
        return lines

    def _make_row_widgets(self):
        """Create widgets for one threshold-based outlier rule row."""
        enabled_cb = QCheckBox()
        enabled_cb.setChecked(True)
        enabled_cb.setToolTip(TOOLTIPSTEXT.RULE_ENABLED_CB)

        feat_combo = compact_combo(
            use_qt_drawn_popup(QComboBox()), min_chars=10, max_w=170
        )
        feat_combo.setToolTip(TOOLTIPSTEXT.FEAT_OUT)
        for k in self.feature_keys:
            feat_combo.addItem(str(k), k)

        m_combo = CheckableComboBox(placeholder="All")
        m_combo.setToolTip(TOOLTIPSTEXT.M_OUT)
        compact_combo(m_combo, min_chars=7, max_w=110)
        for i in range(1, self.m_n + 1):
            m_combo.add_check_item(str(i), i)

        ch_combo = CheckableComboBox(placeholder="All")
        compact_combo(ch_combo, min_chars=7, max_w=110)
        ch_combo.setToolTip(TOOLTIPSTEXT.CH_OUT)
        for channel in self.channel_ids:
            ch_combo.add_check_item(channel, channel)

        # Op1 / Val1
        op1_combo = compact_combo(
            use_qt_drawn_popup(QComboBox()), min_chars=3, max_w=70
        )
        op1_combo.setToolTip(TOOLTIPSTEXT.CMP_OUT)
        for sym in ("<", "<=", "=", ">=", ">"):
            op1_combo.addItem(sym, sym)

        val1_spin = QDoubleSpinBox()
        val1_spin.setToolTip(TOOLTIPSTEXT.VAL_SPIN)
        val1_spin.setRange(-1e12, 1e12)
        val1_spin.setDecimals(2)
        val1_spin.setSingleStep(10)
        val1_spin.setValue(0.0)
        val1_spin.setMinimumWidth(70)
        compact_spin(val1_spin, max_w=90)

        # Op2 (w/ None) / Val2
        op2_combo = compact_combo(
            use_qt_drawn_popup(QComboBox()), min_chars=4, max_w=72
        )
        op2_combo.addItem("None", None)
        for sym in ("<", "<=", "=", ">=", ">"):
            op2_combo.addItem(sym, sym)
        op2_combo.setToolTip(TOOLTIPSTEXT.OP2_SP)

        val2_spin = QDoubleSpinBox()
        val2_spin.setRange(-1e12, 1e12)
        val2_spin.setDecimals(2)
        val2_spin.setSingleStep(10)
        val2_spin.setValue(5.0)
        val2_spin.setMinimumWidth(70)
        compact_spin(val2_spin, max_w=90)
        val2_spin.setEnabled(False)
        val2_spin.setToolTip(TOOLTIPSTEXT.VAL2_SP)

        def _sync_val2_enabled():
            """Enable the second threshold controls only when a second operator is selected."""
            val2_spin.setEnabled(op2_combo.currentData() is not None)

        op2_combo.currentIndexChanged.connect(_sync_val2_enabled)
        _sync_val2_enabled()

        combine_combo = compact_combo(
            use_qt_drawn_popup(QComboBox()), min_chars=4, max_w=72
        )
        combine_combo.addItem("OR", "OR")
        combine_combo.addItem("AND", "AND")
        combine_combo.setToolTip(TOOLTIPSTEXT.COMBINE_COMBO)

        btn_bar = QWidget()
        btn_bar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        hb = QHBoxLayout(btn_bar)
        hb.setContentsMargins(0, 0, 0, 0)
        hb.setSpacing(4)

        add_btn = QPushButton("＋")
        add_btn.setMinimumWidth(28)
        add_btn.setToolTip(TOOLTIPSTEXT.ADD_BTN)

        rm_btn = QPushButton("✕")
        rm_btn.setToolTip(TOOLTIPSTEXT.RM_BTN)
        rm_btn.setMinimumWidth(28)

        add_btn.setProperty("row_index", -1)
        rm_btn.setProperty("row_index", -1)

        hb.addWidget(add_btn)
        hb.addWidget(rm_btn)
        hb.addStretch(1)

        row = {
            "enabled": enabled_cb,
            "feat": feat_combo,
            "m_multi": m_combo,
            "ch_multi": ch_combo,
            "op1": op1_combo,
            "val1": val1_spin,
            "op2": op2_combo,
            "val2": val2_spin,
            "combine": combine_combo,
            "btn_bar": btn_bar,
            "btn_add": add_btn,
            "btn_rm": rm_btn,
        }
        wire_row_buttons(row, self._on_add_after, self._on_remove)
        return row

    def _rebuild_grid_positions(self):
        """Reposition threshold rule rows in the grid layout."""
        reposition_grid_rows(
            self.rows_grid, self.rows_widgets, self._FIELD_KEYS
        )

    def _update_remove_buttons(self):
        """Enable or disable rule remove buttons based on row count."""
        update_row_remove_buttons(self.rows_widgets)

    def append_row_with_values(
        self,
        *,
        feat,
        op1,
        val1,
        op2,
        val2,
        combine,
        masks=None,
        channels=None,
        enabled=True,
    ):
        """Append a threshold-rule row initialized with saved values."""
        widgets = self._make_row_widgets()
        self.rows_widgets.append(widgets)

        widgets["enabled"].setChecked(bool(enabled))

        # Set feature
        idx = widgets["feat"].findText(str(feat))
        if idx >= 0:
            widgets["feat"].setCurrentIndex(idx)

        # Set masks/channels checks if provided
        if masks is not None:
            set_combo_checks(widgets["m_multi"], masks)
        if channels is not None:
            set_combo_checks(widgets["ch_multi"], channels)

        # Set ops/values
        widgets["op1"].setCurrentText(str(op1))
        widgets["val1"].setValue(float(val1))
        if op2 is None:
            widgets["op2"].setCurrentIndex(0)
        else:
            widgets["op2"].setCurrentText(str(op2))
            widgets["val2"].setValue(float(val2 if val2 is not None else 0.0))

        widgets["combine"].setCurrentText(
            str(combine).upper() if combine else "OR"
        )

        self._rebuild_grid_positions()
        self._update_remove_buttons()

    def add_row(self):
        """Add a new threshold-rule row after the requested position."""
        widgets = self._make_row_widgets()
        self.rows_widgets.append(widgets)
        self._rebuild_grid_positions()
        self._update_remove_buttons()

    def _on_remove(self, idx):
        """Remove a threshold-rule row and refresh the layout."""
        if remove_grid_row(
            self.rows_grid, self.rows_widgets, self._FIELD_KEYS, idx
        ):
            self._rebuild_grid_positions()
            self._update_remove_buttons()

    def _on_add_after(self, idx):
        """Insert a threshold-rule row after the selected row."""
        if idx is None or idx < 0 or idx >= len(self.rows_widgets):
            return
        src = self.rows_widgets[int(idx)]
        widgets = self._make_row_widgets()
        widgets["enabled"].setChecked(src["enabled"].isChecked())
        widgets["feat"].setCurrentIndex(src["feat"].currentIndex())
        with contextlib.suppress(Exception):
            set_combo_checks(
                widgets["m_multi"],
                src["m_multi"].selected_values(),
            )
        with contextlib.suppress(Exception):
            set_combo_checks(
                widgets["ch_multi"],
                src["ch_multi"].selected_values(),
            )
        # Clone ops/vals/comb
        widgets["op1"].setCurrentText(src["op1"].currentText())
        widgets["val1"].setValue(src["val1"].value())
        if src["op2"].currentData() is None:
            widgets["op2"].setCurrentIndex(0)
            widgets["val2"].setValue(0.0)
        else:
            widgets["op2"].setCurrentText(src["op2"].currentText())
            widgets["val2"].setValue(src["val2"].value())
        widgets["combine"].setCurrentText(src["combine"].currentText())

        self.rows_widgets.insert(int(idx) + 1, widgets)
        self._rebuild_grid_positions()
        self._update_remove_buttons()

    def clear_all_rows(self):
        """Remove all threshold rule rows and restore a single empty row."""
        while self.rows_widgets:
            row = self.rows_widgets.pop(0)
            for w in [row[k] for k in self._FIELD_KEYS] + [row["btn_bar"]]:
                self.rows_grid.removeWidget(w)
                with contextlib.suppress(Exception):
                    w.setParent(None)
                    w.deleteLater()
        self._rebuild_grid_positions()
        self._update_remove_buttons()

    def _on_preview(self):
        """Open a preview dialog for the currently configured outlier rules."""
        _open_outlier_preview_dialog(
            self.main_window,
            self.win,
            self.rows_widgets,
            self.m_n,
            self.ch_n,
            self.feature_keys,
            add_row_callback=self.append_row_with_values,
        )
