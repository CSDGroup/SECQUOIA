"""Building and wiring the widgets inside a single Dynamics plot row."""

from functools import partial

import pyqtgraph as pg
import qtawesome as qta
from qtpy.QtCore import QSize, Qt
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from SECQUOIA.config import TOOLTIPSTEXT, ColorStyle
from SECQUOIA.gui.dialogs.metric_dialog import open_metric_dialog
from SECQUOIA.gui.main_window.mouse_bindings import on_plot_single_click
from SECQUOIA.utils.plotting import update_plot


class DynamicsPlotRowWidgets:
    """Build the Feature/M/CH top bar, buttons, and plot widget for one plot row."""

    def _build_row_feature_combo(self, row: int) -> tuple[QLabel, QComboBox]:
        """Build the Feature label/combo for a plot row, with its context menu wired."""
        feat_label = QLabel("Feature:")
        feat_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        feat_label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        feat_combo = QComboBox()
        feat_combo.setToolTip(TOOLTIPSTEXT.FEATURE)
        feat_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        feat_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        feat_combo.setMinimumContentsLength(12)
        feat_combo.setContextMenuPolicy(Qt.CustomContextMenu)
        feat_combo.customContextMenuRequested.connect(
            lambda pos, r=row, cb=feat_combo: self._feature_combo_context_menu(
                r, cb, pos
            )
        )
        return feat_label, feat_combo

    def _build_row_mask_combo(
        self, row: int, m_count: int, row_default_m: int | None
    ) -> tuple[QLabel, QComboBox]:
        """Build the M (mask) label/combo for a plot row and restore its prior selection."""
        m_label = QLabel("M:")
        m_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        m_label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        m_combo = QComboBox()
        m_combo.setToolTip(TOOLTIPSTEXT.MPLOT)
        m_combo.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        m_combo.setMinimumContentsLength(2)
        m_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        for i in range(1, m_count + 1):
            m_combo.addItem(str(i), i)

        if m_count > 0:
            prev_m = self.selected_m_by_channel.get(row, row_default_m or 1)
            if not isinstance(prev_m, int) or not (1 <= prev_m <= m_count):
                prev_m = row_default_m or 1
            self.selected_m_by_channel[row] = prev_m
            m_combo.setCurrentIndex(prev_m - 1)
        else:
            self.selected_m_by_channel[row] = None
            m_combo.setEnabled(False)

        return m_label, m_combo

    def _build_row_channel_combo(
        self, row: int, ch_count: int, row_default_ch: str | None
    ) -> tuple[QLabel, QComboBox]:
        """Build the CH (channel) label/combo for a plot row and restore its prior selection."""
        ch_label = QLabel("CH:")
        ch_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        ch_label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        ch_combo = QComboBox()
        ch_combo.setToolTip(TOOLTIPSTEXT.CPLOT)
        ch_combo.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        ch_combo.setMinimumContentsLength(4)
        ch_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        for channel in self.ids_channels:
            ch_combo.addItem(channel[1:], channel[1:])

        if ch_count > 0:
            prev_c = self.selected_ch_by_channel.get(row, row_default_ch)
            if (
                not isinstance(prev_c, str)
                or f"w{prev_c}" not in self.ids_channels
            ):
                prev_c = row_default_ch
            self.selected_ch_by_channel[row] = prev_c
            ch_combo.setCurrentIndex(self.ids_channels.index(f"w{prev_c}"))
        else:
            self.selected_ch_by_channel[row] = None
            ch_combo.setEnabled(False)

        return ch_label, ch_combo

    def _build_row_metric_buttons(self) -> tuple[QPushButton, QPushButton]:
        """Build the metrics and arithmetic buttons for a plot row, sized to their text."""
        btn_r = QPushButton()
        btn_r.setToolTip(TOOLTIPSTEXT.BTN_R)
        btn_r.setCursor(Qt.PointingHandCursor)
        btn_r.setIcon(qta.icon("fa5s.chart-bar", color="white"))
        btn_r.setIconSize(QSize(14, 14))

        btn_arith = QPushButton()
        btn_arith.setToolTip(TOOLTIPSTEXT.BTN_ARITH)
        btn_arith.setCursor(Qt.PointingHandCursor)
        btn_arith.setIcon(qta.icon("fa5s.layer-group", color="white"))
        btn_arith.setIconSize(QSize(14, 14))

        for b in (btn_r, btn_arith):
            fm = b.fontMetrics()
            text_w = fm.horizontalAdvance(b.text())
            pad = 2 * b.style().pixelMetric(
                QStyle.PM_ButtonMargin
            ) + 2 * b.style().pixelMetric(QStyle.PM_DefaultFrameWidth)
            icon_pad = b.iconSize().width() + 8
            b.setMinimumWidth(text_w + pad + icon_pad)

        return btn_r, btn_arith

    def _build_row_eye_button(self) -> QToolButton:
        """Build the show/hide toggle button for a plot row."""
        eye_btn = QToolButton()
        eye_btn.setCheckable(True)
        eye_btn.setChecked(True)
        eye_btn.setAutoRaise(True)
        eye_btn.setIcon(self._eye_icon(True))
        eye_btn.setIconSize(QSize(14, 14))
        eye_btn.setToolTip(TOOLTIPSTEXT.EYEP)
        eye_btn.setCursor(Qt.PointingHandCursor)
        return eye_btn

    def _build_row_summary_labels(self) -> tuple[QLabel, QLabel]:
        """Build the selection-summary and labels shown in a plot row's top bar."""
        summary_lbl = QLabel("")
        summary_lbl.setTextFormat(Qt.RichText)
        summary_lbl.setStyleSheet(ColorStyle.font_css())
        summary_lbl.setMinimumWidth(120)
        summary_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        summary_lbl.setToolTip(TOOLTIPSTEXT.SELECTION)

        igt_lbl = QLabel("")
        igt_lbl.setTextFormat(Qt.RichText)
        igt_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        igt_lbl.setMinimumWidth(120)
        igt_lbl.setToolTip(TOOLTIPSTEXT.ID)
        igt_lbl.setStyleSheet(
            f"color: white; "
            f"font-family:{ColorStyle.FONT_FAMILY}; "
            f"font-size:{ColorStyle.FONT_SIZE}px; "
            f"font-weight:{ColorStyle.FONT_WEIGHT};"
        )

        return summary_lbl, igt_lbl

    def _open_arithmetic_dialog(self) -> None:
        """Open the mask arithmetic dialog for this window.

        The import is deferred because ``arithmetic_dialog`` imports from
        ``SECQUOIA.gui.main_window``, so importing it at module level makes
        the two modules circular whenever the dialog is imported first.
        """
        from SECQUOIA.gui.dialogs.arithmetic_dialog import (
            open_arithmetic_dialog,
        )

        open_arithmetic_dialog(self)

    def _wire_row_signals(
        self,
        row: int,
        feat_combo: QComboBox,
        m_combo: QComboBox,
        ch_combo: QComboBox,
        btn_r: QPushButton,
        btn_arith: QPushButton,
        eye_btn: QToolButton,
        plot_widget: pg.PlotWidget,
    ) -> None:
        """Connect a plot row's combos, buttons, and plot widget to their callbacks."""
        btn_r.clicked.connect(
            lambda checked=False, r=row: open_metric_dialog(
                self,
                r,
            )
        )
        btn_arith.clicked.connect(
            lambda checked=False: self._open_arithmetic_dialog()
        )

        eye_btn.toggled.connect(
            lambda checked, r=row: (
                self._toggle_row_plot_visibility(r, checked),
                self._update_row_summary_label(r),
                self._update_row_igt_label(r),
            )
        )

        def _mark_r_black_and_update(plot_row: int) -> None:
            """Redraw plots and refresh a row's summary/IGT labels after a combo change."""
            update_plot(self)
            self.fit_plot_row(plot_row)
            self._update_row_summary_label(plot_row)
            self._update_row_igt_label(plot_row)

        feat_combo.currentIndexChanged.connect(
            lambda _ix, r=row: _mark_r_black_and_update(r)
        )
        m_combo.currentIndexChanged.connect(
            lambda _ix, r=row: _mark_r_black_and_update(r)
        )
        ch_combo.currentIndexChanged.connect(
            lambda _ix, r=row: _mark_r_black_and_update(r)
        )

        plot_widget.scene().sigMouseClicked.connect(
            partial(on_plot_single_click, self, plot_widget, row)
        )

    def _build_plot_row_widgets(
        self, row: int, ch_count: int, m_count: int
    ) -> None:
        """Build one plot row: its Feature/M/CH top bar, buttons, and plot widget."""
        top_bar = QWidget()
        h = QHBoxLayout(top_bar)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)

        row_default_ch, row_default_m = self._defaults_for_row(
            row, ch_count, m_count
        )
        feat_label, feat_combo = self._build_row_feature_combo(row)
        m_label, m_combo = self._build_row_mask_combo(
            row, m_count, row_default_m
        )
        ch_label, ch_combo = self._build_row_channel_combo(
            row, ch_count, row_default_ch
        )
        btn_r, btn_arith = self._build_row_metric_buttons()
        eye_btn = self._build_row_eye_button()

        target_h = max(
            feat_combo.sizeHint().height(),
            m_combo.sizeHint().height(),
            ch_combo.sizeHint().height(),
        )

        for w in (feat_combo, m_combo, ch_combo, btn_r, btn_arith, eye_btn):
            w.setFixedHeight(target_h)

        eye_btn.setIconSize(
            QSize(max(12, target_h - 8), max(12, target_h - 8))
        )

        for b in (btn_r, btn_arith):
            b.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
            b.setMinimumWidth(max(b.minimumWidth(), b.sizeHint().width()))

        summary_lbl, igt_lbl = self._build_row_summary_labels()

        # Assemble the top bar
        h.addWidget(feat_label)
        h.addWidget(feat_combo)
        h.addSpacing(8)
        h.addWidget(m_label)
        h.addWidget(m_combo)
        h.addSpacing(8)
        h.addWidget(ch_label)
        h.addWidget(ch_combo)
        h.addWidget(btn_r)
        h.addWidget(btn_arith)
        h.addSpacing(6)
        h.addWidget(summary_lbl)
        h.addSpacing(10)
        h.addWidget(igt_lbl)
        h.addStretch(1)
        h.addSpacing(8)
        h.addWidget(eye_btn)

        # Plot
        plot_widget = pg.PlotWidget(name=f"PlotWidget_{row}")
        plot_widget.setMouseEnabled(False, False)
        setattr(self, f"plot_widget_{row}", plot_widget)

        row_container = QWidget()
        v = QVBoxLayout(row_container)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        v.addWidget(top_bar)
        v.addWidget(plot_widget)

        self._plot_grid_layout.addWidget(row_container, row - 1, 0)
        self._plot_grid_layout.setRowStretch(row - 1, 1)

        self.row_tools[row] = {
            "R": btn_r,
            "arith": btn_arith,
            "feat": feat_combo,
            "feat_label": feat_label,
            "m": m_combo,
            "m_label": m_label,
            "ch": ch_combo,
            "ch_label": ch_label,
            "eye": eye_btn,
            "summary": summary_lbl,
            "igt": igt_lbl,
        }

        self._wire_row_signals(
            row,
            feat_combo,
            m_combo,
            ch_combo,
            btn_r,
            btn_arith,
            eye_btn,
            plot_widget,
        )

    def _toggle_row_plot_visibility(
        self, row: int, visible: bool | None = None
    ) -> None:
        """Show/hide the plot in a given row, update the eye icon, and rebalance row space."""
        pw = getattr(self, f"plot_widget_{row}", None)
        if pw is None:
            return

        if visible is None:
            visible = not pw.isVisible()

        pw.setVisible(visible)

        # Keep button state and icon/tooltip in sync
        tools = self.row_tools.get(row, {})
        eye_btn = tools.get("eye")
        if eye_btn:
            eye_btn.setChecked(visible)
            eye_btn.setIcon(self._eye_icon(visible))
            eye_btn.setToolTip("Hide plot" if visible else "Show plot")

        self._reflow_row_stretches()

    def _reflow_row_stretches(self) -> None:
        """Give equal stretch to rows with visible plots; minimized rows get no stretch."""
        gl = getattr(self, "_plot_grid_layout", None)
        if gl is None:
            return

        for idx in range(self._max_plot_rows):
            gl.setRowStretch(idx, 0)

        visible_any = False
        for r in range(1, self._max_plot_rows + 1):
            pw = getattr(self, f"plot_widget_{r}", None)
            if pw and pw.isVisible():
                gl.setRowStretch(r - 1, 1)
                visible_any = True

        if not visible_any and self._max_plot_rows > 0:
            gl.setRowStretch(self._max_plot_rows - 1, 1)
