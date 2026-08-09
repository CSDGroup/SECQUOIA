"""Paint routines for the lineage tree."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets

from SECQUOIA.config import PLOTPARAMETERS
from SECQUOIA.gui.lineage_tree.lineage_geometry import (
    LineageGeometry,
    build_value_map,
    default_xmap,
    generation,
    segments_to_polyline,
)
from SECQUOIA.gui.lineage_tree.lineage_render import (
    QT_DRAW_ERRORS,
    ChannelScale,
    add_track_label,
    apply_lineage_axis,
    clear_colorbar,
    clear_inline_multi_legend,
    draw_division_connector,
    draw_heat_segments,
    resolve_xmax,
    y_bounds,
)

pg.setConfigOptions(antialias=True)

# Appearance
BG_COLOR = "k"
FG_COLOR = "w"
LINE_COLOR = "w"
LINEWIDTH = 1.2
EVENT_LOST = (255, 176, 0)
EVENT_DEAD = (255, 176, 0)
EVENT_OUTLIER = (255, 176, 0)
EVENT_OOF = (255, 176, 0)
LABEL_COLOR = (255, 255, 255)
LABEL_DY = 0.2

# Cellfate value, scatter symbol, colour, filled
_EVENT_MARKERS = (
    ("Dead", "x", EVENT_DEAD, False),
    ("Lost", "t3", EVENT_LOST, True),
    ("Oof", "o", EVENT_OOF, False),
    ("Outlier", "star", EVENT_OUTLIER, False),
)


HEAT_LINE_WIDTH = 20
CONNECT_Z = 8
HEAT_SEG_PAD = 0.12
GEN_TEXT_DY = 0.18
HEAT_LINE_WIDTH_MULTI = 20
CONNECTOR_WIDTH = 8
GEN_TEXT_FONT = QtGui.QFont("Arial", PLOTPARAMETERS.DEFAULT_LABEL_FONT_SIZE)
SHOW_TRACK_LABELS = True
LABEL_MODE = "track"

HEAT_LOW_COLOR = QtGui.QColor(128, 128, 128)
HEAT_HIGH_COLOR = QtGui.QColor(255, 255, 255)


HEAT_MULTI_LOW_COLORS = [
    QtGui.QColor(128, 128, 128),
    QtGui.QColor(255, 255, 0),
    QtGui.QColor(0, 0, 255),
    QtGui.QColor(0, 255, 0),
    QtGui.QColor(255, 0, 255),
]


LEGEND_CELL_MAX_W = 240.0


def _fmt_tick(value: float) -> str:
    """Compact tick text so wide feature values do not blow out a cell."""
    v = float(value)
    if v and (abs(v) >= 1e4 or abs(v) < 1e-2):
        return f"{v:.2g}"
    return f"{v:.2f}"


def _channel_label(col: str) -> str:
    """Channel digits from a feature column, else the column name itself."""
    m = re.search(r"Ch(\d+)M\d+", str(col), re.IGNORECASE)
    return m.group(1) if m else str(col)


def _label_text_for(tn: int) -> str:
    """Return label text per global LABEL_MODE."""
    try:
        if LABEL_MODE == "gen":
            return str(generation(int(tn)))
        return str(int(tn))
    except (TypeError, ValueError):
        return str(tn)


def _draw_lineage_pg(
    plot: pg.PlotItem,
    tracks: pd.DataFrame,
    edges: pd.DataFrame,
    y_map: dict[int, float],
    xmax_override: float | None = None,
    track_colors: dict[int, pg.QtGui.QColor] | None = None,
    valid_times: dict[int, set[float]] | None = None,
    xmap=None,
    main_window=None,
) -> None:
    """Draw the plain lineage tree into a pyqtgraph PlotItem."""
    plot.clear()
    clear_inline_multi_legend(plot)
    clear_colorbar(plot)
    if xmap is None:
        xmap = default_xmap

    geom = LineageGeometry.from_tracks(tracks, edges)

    def pen_for(tn: int):
        """Return the drawing pen for a track, using custom colors when available."""
        col = (
            track_colors.get(int(tn), LINE_COLOR)
            if track_colors
            else LINE_COLOR
        )
        return pg.mkPen(col, width=LINEWIDTH)

    def span_item(tn: int, t0: float, t1: float, y: float, *, strict: bool):
        """Build the polyline for one track span over ``[t0, t1]``."""
        tn = int(tn)
        has_key = valid_times is not None and tn in valid_times
        vt = valid_times.get(tn) if has_key else None
        use_vt = has_key if strict else bool(vt)

        if use_vt:
            xs = segments_to_polyline(
                [t for t in (vt or ()) if t0 <= float(t) < t1], xmap
            )
            if xs.size == 0:
                return None
            ys = np.full_like(xs, float(y), dtype=float)
            return pg.PlotDataItem(xs, ys, pen=pen_for(tn), connect="finite")

        if t1 <= t0:
            return None
        return pg.PlotDataItem(
            [xmap(t0), xmap(t1)], [float(y), float(y)], pen=pen_for(tn)
        )

    parent_span: dict[int, tuple[float, float]] = {}
    for tn in geom.track_numbers():
        y = y_map.get(tn)
        span = geom.division_span(tn)
        if y is None or span is None:
            continue
        t0, t1 = span
        parent_span[tn] = (xmap(t0), xmap(t1))

        item = span_item(tn, t0, t1, y, strict=False)
        if item is not None:
            item.setZValue(1)
            plot.addItem(item)

    conn_x: list[float] = []
    conn_y: list[float] = []
    for _, e in edges.iterrows():
        p, c, tdiv = int(e.parent), int(e.child), float(e.t_div)
        yp, yc = y_map.get(p), y_map.get(c)
        if yp is None or yc is None:
            continue
        if conn_x:
            conn_x.append(np.nan)
            conn_y.append(np.nan)
        x_div = xmap(tdiv)
        conn_x.extend([x_div, x_div])
        conn_y.extend([float(yp), float(yc)])

        child = span_item(c, *geom.child_span(c, tdiv), yc, strict=True)
        if child is not None:
            child.setZValue(1)
            plot.addItem(child)

    if conn_x:
        conn_item = pg.PlotDataItem(
            np.asarray(conn_x, float),
            np.asarray(conn_y, float),
            pen=pg.mkPen(LINE_COLOR, width=CONNECTOR_WIDTH),
            connect="finite",
        )
        conn_item.setZValue(2)
        plot.addItem(conn_item)

    try:
        _, (vy0, vy1) = plot.vb.viewRange()
    except QT_DRAW_ERRORS:
        vy0, vy1 = -1e9, 1e9
    visible = {tn: y for tn, y in y_map.items() if vy0 <= float(y) <= vy1}

    if SHOW_TRACK_LABELS and len(visible) <= 80:
        for tn, (x0, x1) in parent_span.items():
            add_track_label(
                plot,
                _label_text_for(int(tn)),
                x0,
                x1,
                y_map.get(int(tn)),
                color=LABEL_COLOR,
                font=GEN_TEXT_FONT,
                dy=LABEL_DY,
            )

    y_min, y_max = y_bounds(y_map)
    apply_lineage_axis(
        plot,
        resolve_xmax(xmax_override, [xmap(v) for v in geom.t_end.values()]),
        y_min,
        y_max,
        main_window,
    )

    plot.plot(
        [xmap(0.0), xmap(0.0)],
        [y_min, y_max],
        pen=pg.mkPen(LINE_COLOR, width=LINEWIDTH),
    )


def _draw_lineage_heatmap(
    plot: pg.PlotItem,
    tracks: pd.DataFrame,
    edges: pd.DataFrame,
    y_map: dict[int, float],
    df_ident: pd.DataFrame,
    channel_col: str = "AreaMorphologyM1",
    xmax_override: float | None = None,
    valid_times: dict[int, set[float]] | None = None,
    xmap=None,
    main_window=None,
) -> list:
    """Draw a lineage tree with one feature value encoded as line color."""
    if channel_col not in df_ident.columns:
        _draw_lineage_pg(
            plot,
            tracks,
            edges,
            y_map,
            xmax_override=xmax_override,
            valid_times=valid_times,
            xmap=xmap,
            main_window=main_window,
        )
        return []

    plot.clear()
    clear_inline_multi_legend(plot)
    clear_colorbar(plot)
    if xmap is None:
        xmap = default_xmap

    geom = LineageGeometry.from_tracks(tracks, edges)
    scale = ChannelScale.from_values(
        df_ident[channel_col],
        HEAT_LOW_COLOR,
        HEAT_HIGH_COLOR,
        HEAT_LINE_WIDTH,
    )
    values = build_value_map(df_ident, channel_col, valid_times)

    def strip(tn: int, t0: float, t1: float, y: float) -> None:
        """Draw one track's heat strip over ``[t0, t1]``, plus its label."""
        draw_heat_segments(
            plot,
            values.get(int(tn), {}),
            scale,
            t0,
            t1,
            y,
            xmap,
            HEAT_SEG_PAD,
        )
        if SHOW_TRACK_LABELS:
            add_track_label(
                plot,
                _label_text_for(int(tn)),
                xmap(t0),
                xmap(t1),
                y,
                color=FG_COLOR,
                font=GEN_TEXT_FONT,
                dy=GEN_TEXT_DY,
            )

    for tn in geom.track_numbers():
        y = y_map.get(tn)
        span = geom.division_span(tn)
        if y is None or span is None:
            continue
        strip(tn, span[0], span[1], y)

    for _, e in edges.iterrows():
        p, c, tdiv = int(e.parent), int(e.child), float(e.t_div)
        yp, yc = y_map.get(p), y_map.get(c)
        if yp is None or yc is None:
            continue
        draw_division_connector(
            plot,
            xmap(tdiv),
            yp,
            yc,
            pen=pg.mkPen("w", width=max(2, CONNECTOR_WIDTH)),
            z=CONNECT_Z,
        )
        strip(c, *geom.child_span(c, tdiv), yc)

    y_min, y_max = y_bounds(y_map)
    apply_lineage_axis(
        plot,
        resolve_xmax(xmax_override, [xmap(v) for v in geom.t_end.values()]),
        y_min,
        y_max,
        main_window,
    )

    plot.plot(
        [xmap(0.0), xmap(0.0)],
        [y_min, y_max],
        pen=pg.mkPen(LINE_COLOR, width=LINEWIDTH),
    )
    return [
        (
            _channel_label(channel_col),
            scale.vmin,
            scale.vmax,
            HEAT_LOW_COLOR,
        )
    ]


def _scatter_events_pg(
    plot: pg.PlotItem,
    df_events: pd.DataFrame,
    y_map: dict[int, float],
    symbol,
    size: int,
    pen=None,
    brush=None,
    xmap=None,
) -> None:
    """Single ScatterPlotItem for all points of one event type."""
    if df_events.empty:
        return
    x_all, y_all = [], []
    for tn, sub in df_events.groupby("TrackNumber"):
        yy = y_map.get(int(tn))
        if yy is None:
            a = int(tn)
            while a and a not in y_map:
                a //= 2
            yy = y_map.get(a)
        if yy is None:
            continue
        tx = pd.to_numeric(sub["t"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(tx)
        if not mask.any():
            continue
        x_all.append(tx[mask])
        y_all.append(np.full(mask.sum(), float(yy), dtype=float))
    if not x_all:
        return
    if xmap is None:
        xmap = default_xmap
    xs = np.concatenate(
        [np.array([xmap(float(v)) for v in arr], dtype=float) for arr in x_all]
    )
    ys = np.concatenate(y_all)
    sp = pg.ScatterPlotItem(
        x=xs, y=ys, symbol=symbol, size=size, pen=pen, brush=brush
    )
    sp.setZValue(9)
    plot.addItem(sp)


LANE_ROW_FRACTION = 0.42
LEGEND_CLEARANCE_PX = 6


def _view_px_per_y(plot: pg.PlotItem) -> float:
    """Data units per vertical pixel, guarded against an unrealised view."""
    try:
        px_per_y = float(plot.vb.viewPixelSize()[1])
    except QT_DRAW_ERRORS:
        px_per_y = 1.0 / 80.0
    if not np.isfinite(px_per_y) or px_per_y <= 0.0:
        px_per_y = 1.0 / 80.0
    return px_per_y


def _multi_lane_offsets(
    plot: pg.PlotItem, n_lanes: int
) -> tuple[list[float], float, float, int]:
    """Lane offsets in row units, plus spacing, thickness and pen width."""
    px_per_y = _view_px_per_y(plot)
    center_sep_y = LANE_ROW_FRACTION / max(n_lanes, 1)
    line_thick_y = center_sep_y * 0.9
    offsets = [(i - (n_lanes - 1) / 2) * center_sep_y for i in range(n_lanes)]
    max_px = max(1, int(round(float(HEAT_LINE_WIDTH_MULTI))))
    lane_width_px = max(1, min(max_px, int(round(line_thick_y / px_per_y))))
    return offsets, center_sep_y, line_thick_y, lane_width_px


class ChannelLegendBar(QtWidgets.QWidget):
    """Per channel colour key, as a fixed height row above the tree."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Build an empty, hidden legend row."""
        super().__init__(parent)
        self._cells: list[tuple[str, float, float, QtGui.QColor]] = []
        self.setFixedHeight(46)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        self.setVisible(False)

    def set_cells(self, cells) -> None:
        """Show one gradient cell per ``(label, vmin, vmax, low_color)``."""
        self._cells = list(cells)
        self.setVisible(bool(self._cells))
        self.update()

    def paintEvent(self, _event) -> None:
        """Paint label, gradient and the low / high value per channel."""
        if not self._cells:
            return

        painter = QtGui.QPainter(self)
        n_cells = len(self._cells)
        cell_w = min(self.width() / n_cells, LEGEND_CELL_MAX_W)
        x_off = 0.5 * (self.width() - n_cells * cell_w)
        white = QtGui.QColor(255, 255, 255)

        for i, (label, vmin, vmax, low) in enumerate(self._cells):
            x = x_off + i * cell_w
            bar_x = x + 0.10 * cell_w
            bar_w = 0.80 * cell_w

            painter.setPen(white)
            painter.setFont(QtGui.QFont("Arial", 10))
            painter.drawText(
                QtCore.QRectF(x, 0.0, cell_w, 17.0),
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter,
                str(label),
            )

            gradient = QtGui.QLinearGradient(bar_x, 0.0, bar_x + bar_w, 0.0)
            gradient.setColorAt(0.0, low)
            gradient.setColorAt(1.0, HEAT_HIGH_COLOR)
            painter.fillRect(
                QtCore.QRectF(bar_x, 18.0, bar_w, 9.0),
                QtGui.QBrush(gradient),
            )

            ticks = QtCore.QRectF(bar_x, 29.0, bar_w, 16.0)
            painter.setFont(QtGui.QFont("Arial", 9))
            painter.drawText(
                ticks,
                QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                _fmt_tick(vmin),
            )
            painter.drawText(
                ticks,
                QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
                _fmt_tick(vmax),
            )
        painter.end()


def _draw_lineage_heatmap_multi(
    plot: pg.PlotItem,
    tracks: pd.DataFrame,
    edges: pd.DataFrame,
    y_map: dict[int, float],
    df_ident: pd.DataFrame,
    channel_cols: list[str],
    xmax_override: float | None = None,
    valid_times: dict[int, set[float]] | None = None,
    xmap=None,
    main_window=None,
) -> list:
    """Draw a lineage tree with multiple feature channels stacked per track."""
    plot.clear()
    clear_inline_multi_legend(plot)
    clear_colorbar(plot)
    if xmap is None:
        xmap = default_xmap

    geom = LineageGeometry.from_tracks(tracks, edges)
    n_lanes = max(1, len(channel_cols))
    lane_offsets, center_sep_y, line_thick_y, lane_width_px = (
        _multi_lane_offsets(plot, n_lanes)
    )

    scales: list[ChannelScale] = []
    lane_values: list[dict[int, dict[float, float]]] = []
    for i, col in enumerate(channel_cols):
        low = HEAT_MULTI_LOW_COLORS[i % len(HEAT_MULTI_LOW_COLORS)]
        scales.append(
            ChannelScale.from_values(
                df_ident.get(col),
                low,
                HEAT_HIGH_COLOR,
                lane_width_px,
            )
        )
        lane_values.append(build_value_map(df_ident, col, valid_times))

    def stack(tn: int, t0: float, t1: float, y: float) -> None:
        """Draw every channel lane for one track span, plus its label."""
        for offset, scale, values in zip(
            lane_offsets, scales, lane_values, strict=False
        ):
            draw_heat_segments(
                plot,
                values.get(int(tn), {}),
                scale,
                t0,
                t1,
                y + offset,
                xmap,
                HEAT_SEG_PAD,
            )
        if SHOW_TRACK_LABELS:
            add_track_label(
                plot,
                _label_text_for(int(tn)),
                xmap(t0),
                xmap(t1),
                y,
                color=FG_COLOR,
                font=GEN_TEXT_FONT,
                dy=GEN_TEXT_DY,
            )

    for tn in geom.track_numbers():
        y = y_map.get(tn)
        span = geom.division_span(tn)
        if y is None or span is None:
            continue
        stack(tn, span[0], span[1], y)

    for _, e in edges.iterrows():
        p, c, tdiv = int(e.parent), int(e.child), float(e.t_div)
        yp, yc = y_map.get(p), y_map.get(c)
        if yp is None or yc is None:
            continue
        draw_division_connector(
            plot,
            xmap(tdiv),
            yp,
            yc,
            pen=pg.mkPen("w", width=max(2, CONNECTOR_WIDTH)),
            z=CONNECT_Z,
        )
        stack(c, *geom.child_span(c, tdiv), yc)

    total_stack_y = (n_lanes - 1) * center_sep_y + line_thick_y
    clearance = LEGEND_CLEARANCE_PX * _view_px_per_y(plot)
    y_min, y_max = y_bounds(y_map, pad=0.5 * total_stack_y + clearance)
    xmax = resolve_xmax(xmax_override, [xmap(v) for v in geom.t_end.values()])
    apply_lineage_axis(plot, xmax, y_min, y_max, main_window)

    labels = [_channel_label(col) for col in channel_cols]
    return [
        (
            label,
            scale.vmin,
            scale.vmax,
            HEAT_MULTI_LOW_COLORS[i % len(HEAT_MULTI_LOW_COLORS)],
        )
        for i, (label, scale) in enumerate(zip(labels, scales, strict=False))
    ]
