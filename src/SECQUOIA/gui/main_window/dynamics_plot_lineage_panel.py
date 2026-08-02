"""The Dynamics plot lineage sub-panel: top bar, highlight picker, and collapse/expand."""

import contextlib

import pyqtgraph as pg
import qtawesome as qta
from qtpy.QtCore import QSize, Qt, QTimer
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from SECQUOIA.config import TOOLTIPSTEXT
from SECQUOIA.gui.lineage_tree.lineage_selection import (
    _clear_track_selection,
    _select_all_tracks,
)
from SECQUOIA.gui.lineage_tree.lineage_tree import lineage_tree
from SECQUOIA.utils.helpers import clear_highlight_paints
from SECQUOIA.utils.plotting import update_plot


class DynamicsPlotLineagePanel:
    """Build and control the lineage sub-panel next to the Dynamics plot grid."""

    def _build_lineage_graph_placeholder(self) -> None:
        """Create the lineage plot's container widget and its placeholder view."""
        self.graph3_layout = QVBoxLayout()
        self.graph3_layout.setContentsMargins(0, 0, 0, 0)
        self.graph3_widget = QWidget()
        self.graph3_widget.setLayout(self.graph3_layout)

        self._lineage_placeholder = pg.PlotWidget(background="k")
        self._lineage_placeholder.setMouseEnabled(False, False)
        self.graph3_layout.addWidget(self._lineage_placeholder)

    def _build_lineage_highlight_button(self) -> QToolButton:
        """Build the lineage highlight toggle button and its color-picker menu."""
        btn_hl = QToolButton()
        btn_hl.setCheckable(True)
        btn_hl.setChecked(False)
        btn_hl.setAutoRaise(True)
        btn_hl.setCursor(Qt.PointingHandCursor)
        btn_hl.setIcon(qta.icon("fa5s.highlighter", color="white"))
        btn_hl.setIconSize(QSize(14, 14))
        btn_hl.setToolTip(TOOLTIPSTEXT.PAINT_LINEAGE)

        def _apply_hl_btn_style(checked: bool) -> None:
            """Apply the active or inactive visual style to the lineage highlight button."""
            if checked:
                c = self._hl_active_color
                btn_hl.setStyleSheet(
                    f"QToolButton{{background-color: rgb({c.red()},{c.green()},{c.blue()}); "
                    f"border-radius:4px; padding:2px 6px;}}"
                )
            else:
                btn_hl.setStyleSheet("QToolButton{background: transparent;}")

        _apply_hl_btn_style(False)
        self._hl_sync_button_style = lambda: _apply_hl_btn_style(
            btn_hl.isChecked()
        )

        def _set_active_hl_color(qcol: QColor) -> None:
            """Set the active lineage highlight color and refresh the highlight button style."""
            self._hl_active_color = qcol
            _apply_hl_btn_style(btn_hl.isChecked())

        menu = QMenu(btn_hl)

        def _mk_color_action(
            parent_menu: QMenu, name: str, qcol: QColor
        ) -> QWidgetAction:
            """Create a QWidgetAction with a color swatch + colored text, and close menu on pick."""
            act = QWidgetAction(parent_menu)
            w = QWidget(parent_menu)
            lay = QHBoxLayout(w)
            lay.setContentsMargins(8, 4, 8, 4)
            lay.setSpacing(8)

            sw = QLabel()
            sw.setFixedSize(14, 14)
            sw.setStyleSheet(
                f"background-color: rgb({qcol.red()},{qcol.green()},{qcol.blue()});"
                "border:1px solid rgba(255,255,255,120); border-radius:3px;"
            )

            lbl = QLabel(name)
            lbl.setStyleSheet(
                f"color: rgb({qcol.red()},{qcol.green()},{qcol.blue()});"
            )

            lay.addWidget(sw)
            lay.addWidget(lbl, 1)

            act.setDefaultWidget(w)

            def _on_pick() -> None:
                """Handle selecting a highlight color from the color menu."""
                _set_active_hl_color(qcol)
                with contextlib.suppress(
                    RuntimeError, AttributeError, TypeError, ValueError
                ):
                    parent_menu.close()

            w.mousePressEvent = lambda _ev: _on_pick()
            act.triggered.connect(_on_pick)

            return act

        # Build colored entries
        for name, qcol in self._hl_palette.items():
            menu.addAction(_mk_color_action(menu, name, qcol))

        menu.addSeparator()

        # Clear painted highlights
        act_clear = menu.addAction("Clear painted highlights")
        act_clear.triggered.connect(
            lambda: (clear_highlight_paints(self), menu.close())
        )

        btn_hl.setContextMenuPolicy(Qt.CustomContextMenu)
        btn_hl.customContextMenuRequested.connect(
            lambda pos: menu.exec_(btn_hl.mapToGlobal(pos))
        )

        def _on_hl_toggled(checked: bool) -> None:
            """Enable or disable lineage highlight mode and redraw the lineage tree."""
            self._hl_active = bool(checked)
            _apply_hl_btn_style(checked)
            lineage_tree(self)

        btn_hl.toggled.connect(_on_hl_toggled)

        return btn_hl

    def _build_lineage_top_bar(self) -> QWidget:
        """Build the lineage panel's top bar: tree-type combo, highlight, and view controls."""
        lin_top = QWidget()
        lin_h = QHBoxLayout(lin_top)
        lin_h.setContentsMargins(0, 0, 0, 0)
        lin_h.setSpacing(6)

        f_label = QLabel("Tree type:")
        f_label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        f_combo = QComboBox()
        f_combo.setMinimumWidth(52)
        f_combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        f_combo.setToolTip(TOOLTIPSTEXT.TREE)
        f_combo.setSizeAdjustPolicy(QComboBox.AdjustToContentsOnFirstShow)
        f_combo.addItems(["T", "H"])

        btn_hl = self._build_lineage_highlight_button()

        btn_all = QToolButton()
        btn_all.setIcon(qta.icon("fa5s.sitemap", color="white"))
        btn_all.setIconSize(QSize(14, 14))
        btn_all.setToolTip(TOOLTIPSTEXT.HIGHLIGHT_ALL)
        btn_all.setAutoRaise(True)
        btn_all.setCursor(Qt.PointingHandCursor)
        btn_all.clicked.connect(lambda: _select_all_tracks(self))

        btn_e = QToolButton()
        btn_e.setIcon(qta.icon("fa5s.undo", color="white"))
        btn_e.setIconSize(QSize(14, 14))
        btn_e.setToolTip(TOOLTIPSTEXT.E_BTN)
        btn_e.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        btn_e.setAutoRaise(True)
        btn_e.clicked.connect(lambda: _clear_track_selection(self))

        # Eye button for lineage
        lin_eye_btn = QToolButton()
        lin_eye_btn.setCheckable(True)
        lin_eye_btn.setChecked(True)
        lin_eye_btn.setAutoRaise(True)
        lin_eye_btn.setIcon(self._eye_icon(True))
        lin_eye_btn.setIconSize(QSize(14, 14))
        lin_eye_btn.setToolTip(TOOLTIPSTEXT.EYEL)
        lin_eye_btn.setFixedSize(24, 22)
        lin_eye_btn.setCursor(Qt.PointingHandCursor)

        target_h = f_combo.sizeHint().height()

        for w in (f_combo, btn_e, lin_eye_btn, btn_all, btn_hl):
            w.setFixedHeight(target_h)

        btn_hl.setIconSize(QSize(max(12, target_h - 8), max(12, target_h - 8)))
        lin_eye_btn.setIconSize(
            QSize(max(12, target_h - 8), max(12, target_h - 8))
        )
        btn_all.setIconSize(
            QSize(max(12, target_h - 8), max(12, target_h - 8))
        )
        btn_e.setIconSize(QSize(max(12, target_h - 8), max(12, target_h - 8)))

        lin_h.addWidget(f_label)
        lin_h.addWidget(f_combo)

        self.lineage_dynamic_container = QWidget()
        self.lineage_dynamic_layout = QHBoxLayout(
            self.lineage_dynamic_container
        )
        self.lineage_dynamic_layout.setContentsMargins(0, 0, 0, 0)
        self.lineage_dynamic_layout.setSpacing(10)
        lin_h.addSpacing(4)
        lin_h.addWidget(self.lineage_dynamic_container)

        lin_h.addWidget(btn_hl)
        lin_h.addSpacing(4)
        lin_h.addWidget(btn_all)
        lin_h.addSpacing(4)
        lin_h.addWidget(btn_e)
        lin_h.addStretch(1)
        lin_h.addSpacing(8)
        lin_h.addWidget(lin_eye_btn)

        prev_mode = getattr(self, "_lineage_mode", "T")
        if prev_mode in ("T", "H"):
            f_combo.setCurrentText(prev_mode)

        self.lineage_tools = {
            "F": f_combo,
            "ALL": btn_all,
            "E": btn_e,
            "eye": lin_eye_btn,
            "HL": btn_hl,
        }

        lin_eye_btn.toggled.connect(
            lambda checked: self._toggle_lineage_visibility(checked)
        )

        self._lineage_collapsed = not lin_eye_btn.isChecked()

        def _on_mode_changed(txt: str) -> None:
            """Switch the lineage tree mode and preserve collapsed-state."""
            was_collapsed = bool(getattr(self, "_lineage_collapsed", False))
            self._lineage_mode = txt or "T"
            lineage_tree(self)
            if was_collapsed:
                self._toggle_lineage_visibility(False)

        f_combo.currentTextChanged.connect(_on_mode_changed)

        return lin_top

    def _build_lineage_container(self) -> QWidget:
        """Assemble the lineage panel: top bar plus the lineage graph area."""
        self.lineage_container = QWidget()
        lc_grid = QGridLayout(self.lineage_container)
        lc_grid.setContentsMargins(0, 0, 0, 0)
        lc_grid.setHorizontalSpacing(8)
        lc_grid.setVerticalSpacing(4)
        lc_grid.setColumnStretch(0, 1)

        lin_top = self._build_lineage_top_bar()
        self._build_lineage_graph_placeholder()

        lc_grid.addWidget(lin_top, 0, 0)
        lc_grid.addWidget(self.graph3_widget, 1, 0)

        return self.lineage_container

    def _toggle_lineage_visibility(self, visible: bool | None = None) -> None:
        """Collapse/expand lineage pane without removing it.
        Collapsed = only the top bar stays as a thin strip at the bottom."""
        if not hasattr(self, "plots_and_lineage_splitter"):
            return

        sp = self.plots_and_lineage_splitter
        lin_area = getattr(self, "lineage_container", None)
        lin_top = None
        with contextlib.suppress(
            RuntimeError, AttributeError, TypeError, ValueError
        ):
            lin_top = (
                self.lineage_container.layout().itemAtPosition(0, 0).widget()
            )
        graph = getattr(self, "graph3_widget", None)

        if lin_area is None or graph is None or lin_top is None:
            return

        if visible is None:
            visible = not bool(getattr(self, "_lineage_collapsed", False))
        self._lineage_collapsed = not visible

        eye_btn = (
            self.lineage_tools.get("eye")
            if hasattr(self, "lineage_tools")
            else None
        )
        if eye_btn:
            eye_btn.setChecked(visible)
            eye_btn.setIcon(self._eye_icon(visible))
            eye_btn.setToolTip(
                "Hide lineage view" if visible else "Show lineage view"
            )

        try:
            if visible:

                graph.setVisible(True)
                graph.setMaximumHeight(16777215)
                lin_area.setMaximumHeight(16777215)
                sizes = getattr(self, "_splitter_prev_sizes", None)
                if sizes and sum(sizes) > 0:
                    sp.setSizes(sizes)
                else:
                    sp.setSizes([500, 260])
                sp.setStretchFactor(0, 3)
                sp.setStretchFactor(1, 1)
            else:
                self._splitter_prev_sizes = sp.sizes()
                graph.setVisible(False)
                graph.setMaximumHeight(0)
                collapsed_h = max(lin_top.sizeHint().height(), 22)
                lin_area.setMaximumHeight(collapsed_h)
                sp.setStretchFactor(0, 1)
                sp.setStretchFactor(1, 0)
                sp.setSizes([1, collapsed_h])
        except (RuntimeError, AttributeError, TypeError, ValueError):
            pass

    def _keep_lineage_collapsed_after_update(self) -> None:
        """If lineage is collapsed, keep it collapsed even after we rebuild its contents."""
        if not getattr(self, "_lineage_collapsed", False):
            return

        sp = getattr(self, "plots_and_lineage_splitter", None)
        lin_area = getattr(self, "lineage_container", None)
        graph = getattr(self, "graph3_widget", None)
        if not (sp and lin_area and graph):
            return

        try:
            lin_top = lin_area.layout().itemAtPosition(0, 0).widget()
            collapsed_h = max(lin_top.sizeHint().height(), 22)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            collapsed_h = 24

        try:
            graph.setVisible(False)
            graph.setMaximumHeight(0)

            lin_area.setMaximumHeight(collapsed_h)

            sp.setStretchFactor(0, 1)
            sp.setStretchFactor(1, 0)
            sp.setSizes([1, collapsed_h])
        except (RuntimeError, AttributeError, TypeError, ValueError):
            pass

    def _redraw_lineage_and_plots(self) -> None:
        """Redraw lineage tree + plots together so their x-axes align."""
        with contextlib.suppress(
            RuntimeError, AttributeError, TypeError, ValueError
        ):
            lineage_tree(self)
            update_plot(self)

    def _autorange_lineage(self) -> None:
        """Mimic clicking pyqtgraph's default 'A' (auto-range) button on the lineage plot."""

        def _do():
            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                pw = getattr(self, "graph3_plot", None)
                if pw is not None:
                    pw.getPlotItem().autoBtnClicked()

        QTimer.singleShot(0, _do)
