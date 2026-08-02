"""Preview of what a set of outlier rules would flag, before applying them.

Opened from the detection dialog. Draws one histogram per rule with the
threshold lines and the resulting outlier region shaded, so the user can
judge the thresholds against the actual distribution without committing.
"""

from __future__ import annotations

import contextlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from SECQUOIA.config import TOOLTIPSTEXT
from SECQUOIA.core.outlier_detection import resolve_feature_columns
from SECQUOIA.gui.common.ui_utils import (
    open_at_screen_frac,
    use_readable_combo_popup_on_macos,
)
from SECQUOIA.gui.outlier.summary_dialog import draw_rule_histogram
from SECQUOIA.gui.outlier.widgets import CheckableComboBox, set_combo_checks
from SECQUOIA.utils.intervals import resolve_multi

__all__ = ["_open_outlier_preview_dialog"]


def _open_outlier_preview_dialog(
    main_window,
    parent_win,
    rows_widgets,
    m_n,
    ch_n,
    feature_keys,
    *,
    max_cols=3,
    add_row_callback=None,
):
    """Open a modal dialog with a live histogram preview panel per outlier rule row.

    Panels reflow into a `max_cols`-wide grid and redraw as the user edits a
    row's feature/mask/channel/threshold widgets in `rows_widgets`, so
    thresholds can be tuned against the actual data before being applied.
    """
    channel_ids = getattr(main_window, "ids_channels", [])

    df_all = getattr(main_window, "filtered_df", None)
    if df_all is None or df_all.empty:
        QMessageBox.information(parent_win, "No Data", "filtered_df is empty.")
        return

    dlg = QDialog(parent_win)
    dlg.setWindowTitle("Outlier Preview")
    dlg.setWindowModality(Qt.ApplicationModal)
    dlg.setWindowFlags(
        dlg.windowFlags()
        | Qt.WindowMaximizeButtonHint
        | Qt.WindowMinimizeButtonHint
    )
    dlg.setSizeGripEnabled(True)

    outer = QVBoxLayout(dlg)

    scroll = QScrollArea(dlg)
    scroll.setWidgetResizable(True)
    outer.addWidget(scroll)
    inner = QWidget()
    scroll.setWidget(inner)
    inner.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    grid = QGridLayout(inner)
    grid.setContentsMargins(8, 8, 8, 8)
    grid.setHorizontalSpacing(14)
    grid.setVerticalSpacing(14)
    for i in range(max_cols):
        grid.setColumnStretch(i, 1)

    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    CANVAS_H = 220
    PANEL_MIN_H = CANVAS_H + 90

    panels = []
    _grid_rows_applied = {"count": 0}

    def _remove_by_group(group_widget):
        """Remove all preview panels belonging to a row group."""
        for i, p in enumerate(panels):
            if p["group"] is group_widget:
                panels.pop(i)
                group_widget.setParent(None)
                _reflow_grid()
                break

    def _add_after_group(
        *,
        after_group,
        initial_feat,
        masks,
        channels,
        op1,
        val1,
        op2,
        val2,
        combine,
    ):
        """Add a preview panel after the given row group."""
        newp = _make_panel(
            row_widget=None,
            initial_feat=initial_feat,
            masks=masks,
            channels=channels,
            op1=op1,
            val1=val1,
            op2=op2,
            val2=val2,
            combine=combine,
        )
        idx = next(
            (i for i, _p in enumerate(panels) if _p["group"] is after_group),
            len(panels) - 1,
        )
        panels.insert(idx + 1, newp)
        _reflow_grid()

    def _make_panel(
        *,
        row_widget=None,
        initial_feat=None,
        masks=None,
        channels=None,
        op1="<",
        val1=0,
        op2=None,
        val2=0,
        combine="OR",
    ):
        """Create one interactive outlier-preview panel and its plot."""
        feat0 = str(
            initial_feat
            or (
                row_widget["feat"].currentData()
                or row_widget["feat"].currentText()
            )
            if row_widget
            else feature_keys[0]
        )
        masks0 = (
            list(
                resolve_multi(
                    row_widget["m_multi"].selected_values(), range(1, m_n + 1)
                )
            )
            if (row_widget and masks is None)
            else (list(masks or []))
        )
        channels0 = (
            list(
                resolve_multi(
                    row_widget["ch_multi"].selected_values(), channel_ids
                )
            )
            if (row_widget and channels is None)
            else (list(channels or []))
        )

        gb = QGroupBox()
        gb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vb = QVBoxLayout(gb)
        vb.setContentsMargins(6, 6, 6, 6)
        vb.setSpacing(6)

        # line 1: Feature + Mask(s) + Channel(s)
        controls_line1 = QHBoxLayout()
        controls_line1.setContentsMargins(0, 0, 0, 0)
        controls_line1.setSpacing(6)

        feat_cb = QComboBox()
        use_readable_combo_popup_on_macos(feat_cb)
        feat_cb.setToolTip(TOOLTIPSTEXT.FEAT_CB)
        for k in feature_keys:
            feat_cb.addItem(str(k), k)
        idxf = feat_cb.findText(str(feat0))
        if idxf >= 0:
            feat_cb.setCurrentIndex(idxf)
        subf = QVBoxLayout()
        subf.addWidget(QLabel("Feature"))
        subf.addWidget(feat_cb)
        controls_line1.addLayout(subf)

        m_combo = CheckableComboBox("Select masks")
        m_combo.setMinimumWidth(180)
        m_combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        m_combo.setToolTip(TOOLTIPSTEXT.MASK_CB)
        for i in range(1, m_n + 1):
            m_combo.add_check_item(str(i), i)
        m_combo.set_checked_values(masks0)
        subm = QVBoxLayout()
        subm.addWidget(QLabel("Mask(s)"))
        subm.addWidget(m_combo)
        controls_line1.addLayout(subm)

        ch_combo = CheckableComboBox("Select channels")
        ch_combo.setMinimumWidth(180)
        ch_combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        ch_combo.setToolTip(TOOLTIPSTEXT.CH_CB)
        for channel in main_window.ids_channels:
            ch_combo.add_check_item(channel[1:], channel[1:])
        ch_combo.set_checked_values(channels0)
        subc = QVBoxLayout()
        subc.addWidget(QLabel("Channel(s)"))
        subc.addWidget(ch_combo)
        controls_line1.addLayout(subc)

        controls_line1.addStretch(1)
        vb.addLayout(controls_line1)

        rm_btn = QPushButton("✕")
        rm_btn.setToolTip(TOOLTIPSTEXT.RM_BTN)
        rm_btn.setMinimumHeight(24)
        rm_btn.setFlat(True)
        rm_btn.clicked.connect(lambda *_: _remove_by_group(gb))
        add_btn = QPushButton("+")
        add_btn.setToolTip(TOOLTIPSTEXT.ADD_PLOT_BTN)
        add_btn.setMinimumHeight(24)
        add_btn.setFlat(True)

        def _on_add_like():
            """Add a new preview panel using values copied from the current panel."""
            feat_cur = feat_cb.currentData() or feat_cb.currentText()
            masks_cur = resolve_multi(
                m_combo.selected_values(), range(1, m_n + 1)
            )
            ch_cur = resolve_multi(ch_combo.selected_values(), channel_ids)
            op1_cur = op1_cb.currentData() or op1_cb.currentText()
            val1_cur = float(val1_sp.value())
            op2_cur = op2_cb.currentData()
            val2_cur = float(val2_sp.value()) if op2_cur is not None else 0.0
            comb_cur = (
                comb_cb.currentData() or comb_cb.currentText() or "OR"
            ).upper()
            _add_after_group(
                after_group=gb,
                initial_feat=str(feat_cur),
                masks=list(masks_cur),
                channels=list(ch_cur),
                op1=str(op1_cur),
                val1=val1_cur,
                op2=(None if op2_cur is None else str(op2_cur)),
                val2=val2_cur,
                combine=str(comb_cur),
            )

        add_btn.clicked.connect(_on_add_like)
        controls_line1.addStretch(1)
        controls_line1.addWidget(add_btn)
        controls_line1.addWidget(rm_btn)

        # Line 2: ops/values/combine
        controls_line2 = QHBoxLayout()
        controls_line2.setContentsMargins(0, 0, 0, 0)
        controls_line2.setSpacing(6)
        op1_cb = QComboBox()
        use_readable_combo_popup_on_macos(op1_cb)
        [op1_cb.addItem(s, s) for s in ("<", "<=", "=", ">=", ">")]
        op1_cb.setCurrentText(str(op1))
        op1_cb.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        op1_cb.setToolTip(TOOLTIPSTEXT.CMP_OUT)
        val1_sp = QDoubleSpinBox()
        val1_sp.setRange(-1e12, 1e12)
        val1_sp.setDecimals(2)
        val1_sp.setValue(float(val1))
        val1_sp.setToolTip(TOOLTIPSTEXT.VAL_SPIN)
        op2_cb = QComboBox()
        use_readable_combo_popup_on_macos(op2_cb)
        op2_cb.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        op2_cb.addItem("None", None)
        [op2_cb.addItem(s, s) for s in ("<", "<=", "=", ">=", ">")]
        op2_cb.setToolTip(TOOLTIPSTEXT.OP2_SP)
        if op2 is None:
            op2_cb.setCurrentIndex(0)
        else:
            op2_cb.setCurrentText(str(op2))
        val2_sp = QDoubleSpinBox()
        val2_sp.setRange(-1e12, 1e12)
        val2_sp.setDecimals(2)
        val2_sp.setValue(float(val2 or 0.0))
        val2_sp.setEnabled(op2 is not None)
        val2_sp.setToolTip(TOOLTIPSTEXT.VAL2_SP)

        def _sync_val2():
            """Synchronize visibility and enabled state of the second threshold controls."""
            val2_sp.setEnabled(op2_cb.currentData() is not None)

        op2_cb.currentIndexChanged.connect(_sync_val2)

        comb_cb = QComboBox()
        use_readable_combo_popup_on_macos(comb_cb)
        comb_cb.addItem("OR", "OR")
        comb_cb.addItem("AND", "AND")
        comb_cb.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        comb_cb.setCurrentText(str(combine).upper())
        comb_cb.setToolTip(TOOLTIPSTEXT.COMBINE_COMBO)

        for wid, lab in (
            (op1_cb, "Op1"),
            (val1_sp, "Val1"),
            (op2_cb, "Op2"),
            (val2_sp, "Val2"),
            (comb_cb, "Combine"),
        ):
            sub = QVBoxLayout()
            sub.addWidget(QLabel(lab))
            sub.addWidget(wid)
            controls_line2.addLayout(sub)
        controls_line2.addStretch(1)
        vb.addLayout(controls_line2)

        def _current_feat_masks_channels():
            """Return the current feature, mask, and channel selections."""
            feat = feat_cb.currentData() or feat_cb.currentText()
            masks = resolve_multi(m_combo.selected_values(), range(1, m_n + 1))
            channels = resolve_multi(ch_combo.selected_values(), channel_ids)
            return str(feat), masks, channels

        def _title():
            """Build a display title for the current preview panel."""
            feat, masks, chans = _current_feat_masks_channels()
            return f"{feat} • masks={masks or 'ALL'}, ch={chans or 'ALL'}"

        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(CANVAS_H)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar = NavigationToolbar(canvas, gb)
        vb.addWidget(toolbar)
        vb.addWidget(canvas)

        x_cache = {"x": np.array([], dtype=float)}

        def _recompute_data():
            """Recompute data values for the current preview-panel selections."""
            feat, masks, chans = _current_feat_masks_channels()
            cols = resolve_feature_columns(df_all, feat, masks, chans)
            if not cols:
                arr = np.array([], dtype=float)
            else:
                vals = []
                for c in cols:
                    vc = pd.to_numeric(df_all[c], errors="coerce").to_numpy()
                    if vc.size:
                        vals.append(vc)
                arr = (
                    np.concatenate(vals) if vals else np.array([], dtype=float)
                )
            x_cache["x"] = arr[np.isfinite(arr)]

        def _redraw():
            """Redraw the preview histogram and threshold overlays."""
            ax.clear()
            _recompute_data()

            op1v = op1_cb.currentData() or op1_cb.currentText()
            op2v = op2_cb.currentData()
            combv = (
                comb_cb.currentData() or comb_cb.currentText() or "OR"
            ).upper()

            draw_rule_histogram(
                ax,
                x_cache["x"],
                {
                    "op1": op1v,
                    "val1": float(val1_sp.value()),
                    "op2": op2v,
                    "val2": (
                        float(val2_sp.value()) if op2v is not None else None
                    ),
                    "combine": combv,
                },
            )

            gb.setTitle(_title())
            fig.tight_layout()
            canvas.draw_idle()

        _recompute_data()
        _redraw()

        # Dragging
        drag_state = {"dragging": None}
        tol_px = 6

        def _near(event, xval):
            """Check whether a mouse x-position is close to a draggable threshold."""
            if event.inaxes != ax:
                return False
            xpix, _ = ax.transData.transform((float(xval), 0))
            return abs(event.x - xpix) <= tol_px

        def on_press(event):
            """Handle mouse press events for dragging threshold markers."""
            if event.inaxes != ax:
                return
            t1 = float(val1_sp.value())
            t2 = float(val2_sp.value())
            has2 = op2_cb.currentData() is not None
            if _near(event, t1):
                drag_state["dragging"] = "t1"
            elif has2 and _near(event, t2):
                drag_state["dragging"] = "t2"

        def on_move(event):
            """Handle mouse movement while dragging threshold markers."""
            if event.inaxes != ax or event.xdata is None:
                return
            if drag_state["dragging"] == "t1":
                val1_sp.blockSignals(True)
                val1_sp.setValue(float(event.xdata))
                val1_sp.blockSignals(False)
                _redraw()
            elif drag_state["dragging"] == "t2":
                val2_sp.blockSignals(True)
                val2_sp.setValue(float(event.xdata))
                val2_sp.blockSignals(False)
                _redraw()

        def on_release(event):
            """Handle mouse release events and finish dragging."""
            drag_state["dragging"] = None

        canvas.mpl_connect("button_press_event", on_press)
        canvas.mpl_connect("motion_notify_event", on_move)
        canvas.mpl_connect("button_release_event", on_release)

        for wid in (
            feat_cb,
            m_combo,
            ch_combo,
            op1_cb,
            val1_sp,
            op2_cb,
            val2_sp,
            comb_cb,
        ):
            if hasattr(wid, "currentIndexChanged"):
                wid.currentIndexChanged.connect(_redraw)
            if hasattr(wid, "valueChanged"):
                wid.valueChanged.connect(_redraw)
        if hasattr(m_combo.view(), "pressed"):
            m_combo.view().pressed.connect(lambda *_: _redraw())
        if hasattr(ch_combo.view(), "pressed"):
            ch_combo.view().pressed.connect(lambda *_: _redraw())

        return {
            "row": row_widget,
            "feat_cb": feat_cb,
            "m_combo": m_combo,
            "ch_combo": ch_combo,
            "op1_cb": op1_cb,
            "val1_sp": val1_sp,
            "op2_cb": op2_cb,
            "val2_sp": val2_sp,
            "comb_cb": comb_cb,
            "group": gb,
        }

    for w in rows_widgets:
        p = _make_panel(
            row_widget=w,
            initial_feat=w["feat"].currentData() or w["feat"].currentText(),
            op1=w["op1"].currentData() or w["op1"].currentText(),
            val1=float(w["val1"].value()),
            op2=w["op2"].currentData(),
            val2=(
                float(w["val2"].value())
                if w["op2"].currentData() is not None
                else None
            ),
            combine=w["combine"].currentData()
            or w["combine"].currentText()
            or "OR",
        )
        panels.append(p)

    def _reflow_grid():
        """Reflow preview panels into the configured grid layout."""
        while grid.count():
            it = grid.takeAt(0)
            w = it.widget()
            if w is not None:
                w.setParent(None)

        for i, p in enumerate(panels):
            r, c = divmod(i, max_cols)
            grid.addWidget(p["group"], r, c)

        prev = _grid_rows_applied["count"]
        for r in range(prev):
            grid.setRowStretch(r, 0)
            grid.setRowMinimumHeight(r, 0)

        rows = (len(panels) + max_cols - 1) // max_cols
        for r in range(rows):
            grid.setRowStretch(r, 1)
            grid.setRowMinimumHeight(r, PANEL_MIN_H + 12)

        _grid_rows_applied["count"] = rows

        grid.invalidate()
        inner.updateGeometry()
        scroll.updateGeometry()

    _reflow_grid()

    btns = QHBoxLayout()
    btns.addStretch(1)
    btns.setContentsMargins(0, 0, 0, 0)
    btns.setSpacing(6)
    submit_btn = QPushButton("Submit parameters to GUI")
    submit_btn.setMinimumHeight(28)
    submit_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
    submit_btn.setToolTip(TOOLTIPSTEXT.SUBMIT_BTN)

    close_btn = QPushButton("Close")
    close_btn.setMinimumHeight(28)
    close_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
    close_btn.setToolTip(TOOLTIPSTEXT.CLOSE_BTN)
    btns.addWidget(submit_btn)
    btns.addSpacing(10)
    btns.addWidget(close_btn)
    btns.addStretch(1)
    outer.addLayout(btns)

    def _on_submit():
        """Apply preview-panel rule changes and close the preview dialog."""
        for p in panels:
            row = p["row"]
            feat_txt = p["feat_cb"].currentText()
            masks_sel = p["m_combo"].selected_values()
            ch_sel = p["ch_combo"].selected_values()
            op1_txt = p["op1_cb"].currentText()
            val1_v = float(p["val1_sp"].value())
            op2_data = p["op2_cb"].currentData()
            op2_txt = None if op2_data is None else p["op2_cb"].currentText()
            val2_v = float(p["val2_sp"].value())
            comb_txt = p["comb_cb"].currentText()

            if row is None:
                if add_row_callback is not None:
                    add_row_callback(
                        feat=feat_txt,
                        op1=op1_txt,
                        val1=val1_v,
                        op2=op2_txt,
                        val2=(val2_v if op2_txt is not None else 0),
                        combine=comb_txt,
                        masks=masks_sel,
                        channels=ch_sel,
                    )
            else:
                idxf = row["feat"].findText(str(feat_txt))
                if idxf >= 0:
                    row["feat"].setCurrentIndex(idxf)
                # Masks/channels to main GUI
                with contextlib.suppress(
                    RuntimeError, AttributeError, TypeError
                ):
                    set_combo_checks(row["m_multi"], masks_sel)
                with contextlib.suppress(
                    RuntimeError, AttributeError, TypeError
                ):
                    set_combo_checks(row["ch_multi"], ch_sel)
                # ops/vals/comb
                row["op1"].setCurrentText(op1_txt)
                row["val1"].setValue(val1_v)
                if op2_txt is None:
                    row["op2"].setCurrentIndex(0)
                else:
                    row["op2"].setCurrentText(op2_txt)
                    row["val2"].setValue(val2_v)
                row["combine"].setCurrentText(comb_txt)
        dlg.accept()

    def _on_close():
        """Close the preview dialog without applying changes."""
        dlg.reject()

    submit_btn.clicked.connect(_on_submit)
    close_btn.clicked.connect(_on_close)

    open_at_screen_frac(dlg)
    dlg.exec_()
