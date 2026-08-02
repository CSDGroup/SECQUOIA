"""``LineageTreeView``: one lineage plot widget, plus everything that
redraws it and handles clicks on it."""

from __future__ import annotations

import contextlib
import numbers

import numpy as np
import pandas as pd
import pyqtgraph as pg
from qtpy import QtWidgets
from qtpy.QtCore import Qt

from SECQUOIA.gui.common.ui_utils import clear_layout
from SECQUOIA.gui.lineage_tree.lineage_controls import HeatmapControls
from SECQUOIA.gui.lineage_tree.lineage_data import (
    _lineage_data,
    _time_mapper_for_ident,
)
from SECQUOIA.gui.lineage_tree.lineage_draw import (
    _EVENT_MARKERS,
    BG_COLOR,
    _draw_lineage_heatmap,
    _draw_lineage_heatmap_multi,
    _draw_lineage_pg,
    _scatter_events_pg,
)
from SECQUOIA.gui.lineage_tree.lineage_geometry import (
    FeatureCatalog,
    LineageGeometry,
    default_xmap,
)
from SECQUOIA.gui.lineage_tree.lineage_interaction import (
    install_shift_pan,
    strip_default_menu_actions,
)
from SECQUOIA.gui.lineage_tree.lineage_render import (
    QT_DRAW_ERRORS,
    draw_division_connector,
)
from SECQUOIA.gui.lineage_tree.lineage_selection import (
    _assign_highlight_color,
    _sync_viewers_to_time,
    _toggle_track,
)
from SECQUOIA.gui.lineage_tree.lineage_zoom import _register_redraw


class LineageTreeView:
    """One lineage plot widget, plus everything that redraws it."""

    def __init__(
        self,
        df: pd.DataFrame,
        ident: str,
        *,
        xmax_override: float | None = None,
        track_colors: dict[int, pg.QtGui.QColor] | None = None,
        render_mode: str = "T",
        main_window=None,
    ) -> None:
        """Build the widget and draw the tree for ``ident``."""
        self.df = df
        self.ident = ident
        self.xmax_override = xmax_override
        self.main_window = main_window
        self.render_mode = (render_mode or "T").upper()

        self.plot_widget = pg.PlotWidget(background=BG_COLOR)
        self.plot = self.plot_widget.getPlotItem()
        self.widget: QtWidgets.QWidget = self.plot_widget
        self.controls: HeatmapControls | None = None
        self.state: dict = {}
        self._orig_vb_click = self.plot.vb.mouseClickEvent

        if not self._has_rows():
            return

        self._setup_view()
        self.state = _lineage_data(df, ident, main_window, track_colors)
        self._install_click_handlers()

        if self.render_mode.startswith("H"):
            self._build_heatmap_mode()
        else:
            self._build_plain_mode()

        self.plot_widget.setMouseEnabled(True, True)
        self.plot.vb.setMouseMode(pg.ViewBox.RectMode)

    def _has_rows(self) -> bool:
        """True when there is something to draw for this identification."""
        if not isinstance(self.df, pd.DataFrame) or self.df.empty:
            return False
        if self.ident is None or str(self.ident).lower() == "nan":
            return False
        sub = self.df[self.df["Identification"].astype(str) == str(self.ident)]
        return not sub.empty

    def _setup_view(self) -> None:
        """Configure mouse behaviour and the context menu."""
        self.plot_widget.setMouseEnabled(True, True)
        self.plot.vb.setLimits(xMin=0, xMax=None, yMin=None, yMax=None)
        install_shift_pan(self.plot)
        self.plot_widget.setContextMenuPolicy(Qt.DefaultContextMenu)
        strip_default_menu_actions(self.plot, self.main_window)

    def _build_plain_mode(self) -> None:
        """Draw the plain tree with event markers."""
        topbar = getattr(self.main_window, "lineage_dynamic_layout", None)
        if isinstance(topbar, QtWidgets.QLayout):
            clear_layout(topbar)
        self.draw_plain(hl_white=True)
        self.draw_events()
        _register_redraw(self.main_window, self.redraw_live)

    def _build_heatmap_mode(self) -> None:
        """Draw the heatmap tree and wire up its Feature / M / CH bar."""
        sub = self.df[self.df["Identification"] == self.ident]
        catalog = FeatureCatalog.from_columns(
            list(sub.columns),
            getattr(self.main_window, "_derived_features", None),
        )

        if not catalog.heat_columns_present or not catalog:
            self.draw_plain(hl_white=False)
            _register_redraw(self.main_window, self.redraw_live)
            return

        self.controls = HeatmapControls(catalog, self._default_mask())
        self.controls.selectionChanged.connect(self.redraw)
        self._place_controls()
        self.redraw()
        _register_redraw(self.main_window, self.redraw_live)

    def _default_mask(self) -> int | None:
        """The mask index the first viewer is showing, when that is known."""
        try:
            return int(
                getattr(
                    self.main_window, "active_mask_index_by_viewer", {}
                ).get(0, None)
            )
        except (TypeError, ValueError):
            return None

    def _place_controls(self) -> None:
        """Dock the control bar in the shared top bar, or in our own column."""
        topbar = getattr(self.main_window, "lineage_dynamic_layout", None)
        if isinstance(topbar, QtWidgets.QLayout):
            clear_layout(topbar)
            topbar.addWidget(self.controls)
            return

        container = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(container)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        column.addWidget(self.controls)
        column.addWidget(self.plot_widget)
        self.widget = container

    def refresh_state(self) -> None:
        """Re-read the current dataframe and recompute the lineage topology."""
        current = getattr(self.main_window, "filtered_df", None)
        if (
            not isinstance(current, pd.DataFrame)
            or current.empty
            or "Identification" not in current.columns
            or current[
                current["Identification"].astype(str) == str(self.ident)
            ].empty
        ):
            current = self.df
        self.state.update(
            _lineage_data(current, self.ident, self.main_window, None)
        )

    def _xmax(self) -> float | None:
        """The mapped x maximum, falling back to the caller's override."""
        mapped = self.state.get("xmax_mapped")
        return (
            mapped if isinstance(mapped, numbers.Real) else self.xmax_override
        )

    def _highlight_active(self) -> bool:
        """True while the user is painting tracks rather than selecting them."""
        return bool(getattr(self.main_window, "_hl_active", False))

    def _xmap(self):
        """The active time -> x mapping."""
        return self.state.get("xmap") or default_xmap

    def redraw(self) -> None:
        """Repaint the tree for the current mode and control selection."""
        if self.controls is not None:
            self._draw_heatmap()
        elif self.render_mode.startswith("H"):
            self.draw_plain(hl_white=False)
        else:
            self.draw_plain(hl_white=True)
            self.draw_events()

    def redraw_live(self) -> None:
        """Pick up new measurement values, then repaint in place."""
        self.refresh_state()
        self.redraw()

    def set_identification(
        self,
        ident: str,
        df: pd.DataFrame,
        *,
        xmax_override: float | None = None,
        track_colors: dict[int, pg.QtGui.QColor] | None = None,
    ) -> None:
        """Point this view at a different identification and repaint in place.

        Reuses the existing ``plot_widget``/``ViewBox``.
        """
        self.df = df
        self.ident = str(ident)
        self.xmax_override = xmax_override
        if not self._has_rows():
            self.plot.clear()
            return
        self.state = _lineage_data(
            df, self.ident, self.main_window, track_colors
        )
        self.redraw()

    def _draw_heatmap(self) -> None:
        """Draw whichever heatmap the control selection resolves to."""
        columns = self.controls.resolve()
        if not columns:
            self.draw_plain(hl_white=False)
            return

        target = (
            self.plot,
            self.state["tracks"],
            self.state["edges"],
            self.state["y_map"],
            self.state["sub_ident"],
        )
        options = {
            "xmax_override": self._xmax(),
            "valid_times": self.state["valid_times"],
            "xmap": (
                None if self.state["use_fast_heatmap"] else self.state["xmap"]
            ),
            "main_window": self.main_window,
        }
        if len(columns) == 1:
            _draw_lineage_heatmap(*target, channel_col=columns[0], **options)
        else:
            _draw_lineage_heatmap_multi(
                *target, channel_cols=columns, **options
            )
        self.overlay_highlight()

    def draw_plain(self, hl_white: bool) -> None:
        """Draw the plain tree plus the highlight overlay."""
        colors = self.state["track_colors"]
        painted = getattr(self.main_window, "_track_highlight_colors", {})
        if hl_white and self._highlight_active() and painted:
            colors = {
                int(t): pg.mkColor("w")
                for t in self.state["tracks"]["TrackNumber"].unique()
            }

        _draw_lineage_pg(
            self.plot,
            self.state["tracks"],
            self.state["edges"],
            self.state["y_map"],
            xmax_override=self._xmax(),
            track_colors=colors,
            valid_times=self.state["valid_times"],
            xmap=self.state["xmap"],
            main_window=self.main_window,
        )
        self.overlay_highlight()

    def overlay_highlight(self, pen=None) -> None:
        """Redraw selected or painted tracks on top, in their own colours."""
        edges = self.state["edges"]
        y_map = self.state["y_map"]
        xmap = self._xmap()
        geom = LineageGeometry.from_tracks(self.state["tracks"], edges)

        painted = (
            getattr(self.main_window, "_track_highlight_colors", {}) or {}
        )
        selected = getattr(self.main_window, "_selected_tracks", set())
        painting = self._highlight_active() and bool(painted)
        if not painting and not selected:
            return

        if pen is None:
            pen = pg.mkPen((255, 230, 0), width=4)

        def pen_for(track: int):
            """The pen one highlighted track is drawn with."""
            if painting:
                return pg.mkPen(painted[int(track)], width=4)
            return pen

        def draw_span(track: int) -> None:
            """Draw one highlighted track span."""
            y = y_map.get(int(track))
            bounds = geom.division_span(track)
            if y is None or bounds is None:
                return
            x0, x1 = bounds
            if not (np.isfinite(x0) and np.isfinite(x1) and x1 > x0):
                return
            item = self.plot.plot(
                [xmap(x0), xmap(x1)], [y, y], pen=pen_for(track)
            )
            with contextlib.suppress(*QT_DRAW_ERRORS):
                item.setZValue(100)

        if painting:
            chosen = painted
            targets = [int(tn) for tn in painted]
        else:
            chosen = selected
            targets = [tn for tn in geom.t_start if tn in selected]

        for track in targets:
            draw_span(track)

        for _, e in edges.iterrows():
            p_tn, c_tn, tdiv = int(e.parent), int(e.child), float(e.t_div)
            if p_tn not in chosen or c_tn not in chosen:
                continue
            yp, yc = y_map.get(p_tn), y_map.get(c_tn)
            if yp is None or yc is None:
                continue
            draw_division_connector(
                self.plot, xmap(tdiv), yp, yc, pen=pen_for(c_tn), z=100
            )

    def draw_events(self) -> None:
        """Add the cell-fate event markers (T mode only)."""
        sub = self.state["sub_ident"]

        if "Cellfate" in sub.columns:
            fates = sub["Cellfate"]
            try:
                fates = fates.astype("string")
            except TypeError:
                fates = fates.astype(str)
            fates = fates.fillna("")
        else:
            fates = pd.Series([""] * len(sub), index=sub.index, dtype="string")

        for name, symbol, color, filled in _EVENT_MARKERS:
            rows = sub[fates.str.contains(rf"^{name}$", case=False, na=False)]
            _scatter_events_pg(
                self.plot,
                rows,
                self.state["y_map"],
                symbol=symbol,
                size=12,
                pen=None if filled else pg.mkPen(color, width=1.8),
                brush=pg.mkBrush(color, width=1.8) if filled else None,
                xmap=self.state["xmap"],
            )

    def _install_click_handlers(self) -> None:
        """Route clicks on the plot and the widget through our handlers."""
        self.plot.vb.mouseClickEvent = self._on_plot_click
        self.plot_widget.mousePressEvent = self._on_widget_press

    def _y_map_for_hit_test(self) -> dict:
        """The y-map to hit-test against, preferring the last one drawn."""
        return (
            getattr(
                self.main_window, "_last_lineage_y_map", self.state["y_map"]
            )
            or {}
        )

    def _nearest_track(self, y_click: float, y_map: dict) -> int | None:
        """The track whose row is closest to ``y_click``."""
        if not y_map:
            return None
        return min(y_map, key=lambda tn: abs(y_map[tn] - y_click))

    def _select_or_paint(self, track: int, y_map: dict) -> None:
        """Ctrl-click behaviour: paint when highlighting, else toggle select."""
        with contextlib.suppress(*QT_DRAW_ERRORS, ValueError):
            self.main_window._lineage_y_map = dict(y_map)
        if self._highlight_active():
            _assign_highlight_color(self.main_window, int(track))
        else:
            _toggle_track(self.main_window, int(track))

    def _frame_at(self, x_clicked: float) -> int:
        """The frame index whose mapped x is nearest to ``x_clicked``."""
        mw = self.main_window
        try:
            df = getattr(mw, "filtered_df", None)
            ident = getattr(mw, "ident", None)
            if df is None or ident is None:
                raise RuntimeError("No dataframe/ident")
            sub = df[df["Identification"].astype(str) == str(ident)]
            mapper, _, _, _ = _time_mapper_for_ident(sub, mw)

            times = pd.to_numeric(sub.get("t"), errors="coerce").to_numpy(
                dtype=float
            )
            times = times[np.isfinite(times)]
            if not times.size:
                return 0
            xs = np.array([float(mapper(t)) for t in times], dtype=float)
            return int(times[int(np.nanargmin(np.abs(xs - x_clicked)))])
        except (*QT_DRAW_ERRORS, ValueError):
            return int(np.floor(x_clicked))

    def _clamp_frame(self, frame: int) -> int:
        """Keep ``frame`` inside the loaded image stack, or the dataframe."""
        try:
            n_frames = self.main_window.images[0].shape[0]
        except (AttributeError, IndexError, KeyError, TypeError):
            n_frames = 0

        if n_frames > 0:
            return max(0, min(frame, n_frames - 1))

        t_max = int(self.state.get("t_max_df", 0) or 0)
        if frame < 0:
            return 0
        return min(frame, t_max) if t_max > 0 else frame

    def _on_plot_click(self, event):
        """Handle tree clicks: time jumps, track selection and highlighting."""
        mw = self.main_window
        if hasattr(event, "button") and event.button() != Qt.LeftButton:
            return self._orig_vb_click(event)
        if hasattr(event, "double") and event.double():
            return self._orig_vb_click(event)

        try:
            scene_point = (
                event.scenePos()
                if hasattr(event, "scenePos")
                else self.plot_widget.mapToScene(event.pos())
            )
            point = self.plot.vb.mapSceneToView(scene_point)
        except QT_DRAW_ERRORS:
            return self._orig_vb_click(event)

        y_map = self._y_map_for_hit_test()
        nearest = self._nearest_track(float(point.y()), y_map)

        modifiers = (
            event.modifiers() if hasattr(event, "modifiers") else Qt.NoModifier
        )
        if modifiers & Qt.ControlModifier:
            if nearest is not None:
                self._select_or_paint(nearest, y_map)
            event.accept()
            return None

        if nearest is not None:
            mw.current_TrackNumber_plot = int(nearest)

        mw.current_time_index = self._clamp_frame(
            self._frame_at(float(point.x()))
        )
        _sync_viewers_to_time(mw)
        event.accept()
        return None

    def _on_widget_press(self, event):
        """Handle Ctrl-click selection at the widget level."""
        try:
            if event.button() != Qt.LeftButton:
                return pg.PlotWidget.mousePressEvent(self.plot_widget, event)

            modifiers = (
                event.modifiers()
                if hasattr(event, "modifiers")
                else Qt.NoModifier
            )
            if not (modifiers & Qt.ControlModifier):
                return pg.PlotWidget.mousePressEvent(self.plot_widget, event)

            local = (
                event.position().toPoint()
                if hasattr(event, "position")
                else event.pos()
            )
            point = self.plot.vb.mapSceneToView(
                self.plot_widget.mapToScene(local)
            )

            y_map = self._y_map_for_hit_test()
            if not y_map:
                return None
            nearest = self._nearest_track(float(point.y()), y_map)
            if nearest is not None:
                self._select_or_paint(nearest, y_map)
            event.accept()
            return None
        except (*QT_DRAW_ERRORS, ValueError):
            with contextlib.suppress(*QT_DRAW_ERRORS, ValueError):
                return pg.PlotWidget.mousePressEvent(self.plot_widget, event)
            return None
