"""Per row rendering engine for the dynamics plot grid.

Draws each row's curves, styles its axes, keeps its Feature/Mask/Channel
combo boxes in sync with the data, and computes the scales shared across all
rows so they line up with one another.
"""

from __future__ import annotations

import contextlib
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyqtgraph as pg

from SECQUOIA.utils.plotting.axis_ticks import (
    _PLOT_LEFT_MARGIN_PX,
    _shared_left_axis_width,
    _y_view_range,
    apply_y_range_and_ticks,
    axis_font,
)
from SECQUOIA.utils.plotting.feature_discovery import (
    DEFAULT_FEATURE,
    TIME_MODE_T,
    _finite_max,
    _finite_min,
    _format_feature_column,
    x_column_for,
)

# Fallback curve color when a track has no assigned color.
DEFAULT_TRACK_COLOR = "#4FC3F7"
# Z-order for the white base curve drawn under highlighted tracks.
Z_HIGHLIGHT_BASE = 5
# Z-order for the colored highlight curve drawn on top.
Z_HIGHLIGHT_TOP = 100


class _CurveStyle(NamedTuple):
    """How one track's curve should be drawn."""

    pen: object
    symbol_brush: object
    z: int | None


def _track_xy(
    df_subset: pd.DataFrame, track, xcol: str, ycol: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """Finite ``(x, y)`` arrays for one track, or ``None`` if nothing to draw."""
    rows = df_subset[df_subset["TrackNumber"] == track]
    if rows.empty:
        return None
    x = rows[xcol].to_numpy(dtype=float)
    y = rows[ycol].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return None
    return x[finite], y[finite]


def _row_curve_plan(
    tracks,
    *,
    highlight_on: bool,
    color_map: dict,
    track_colors: dict,
    line_w: float,
    hl_line_w: float,
):
    """Yield ``(track, style)`` pairs in the order they must be drawn.

    Without highlighting each track is drawn once in its own colour. With
    highlighting every track is first drawn in white as a background layer,
    then the highlighted ones are drawn again on top in their colour - which
    is why highlighted tracks appear twice and the order matters.
    """
    if not highlight_on:
        for track in tracks:
            color = track_colors.get(track, pg.mkColor(DEFAULT_TRACK_COLOR))
            yield track, _CurveStyle(
                pg.mkPen(color=color, width=line_w), color, None
            )
        return

    base_pen = pg.mkPen(pg.mkColor("w"), width=line_w)
    base_brush = pg.mkBrush(pg.mkColor("w"))
    for track in tracks:
        yield track, _CurveStyle(base_pen, base_brush, Z_HIGHLIGHT_BASE)

    for track in tracks:
        highlight = color_map.get(int(track))
        if highlight is None:
            continue
        yield track, _CurveStyle(
            pg.mkPen(highlight, width=hl_line_w),
            pg.mkBrush(highlight),
            Z_HIGHLIGHT_TOP,
        )


def _draw_row_curves(
    plot_widget,
    df_subset: pd.DataFrame,
    plan,
    xcol: str,
    ycol: str,
    *,
    symbol,
    sym_size: float,
) -> tuple[float, float]:
    """Draw one row's curves and report the y-extent of what was drawn."""
    min_y, max_y = np.inf, -np.inf

    for track, style in plan:
        finite = _track_xy(df_subset, track, xcol, ycol)
        if finite is None:
            continue
        x, y = finite

        with contextlib.suppress(
            RuntimeError, AttributeError, TypeError, ValueError
        ):
            max_y = max(max_y, float(np.nanmax(y)))
            min_y = min(min_y, float(np.nanmin(y)))

        item = plot_widget.plot(
            x,
            y,
            pen=style.pen,
            symbol=(symbol if sym_size > 0 else None),
            symbolSize=sym_size,
            symbolBrush=style.symbol_brush,
            name=f"Track {track}",
        )

        if style.z is not None:
            with contextlib.suppress(
                RuntimeError, AttributeError, TypeError, ValueError
            ):
                item.setZValue(style.z)

    return min_y, max_y


def _row_axis_metrics(main_window) -> tuple[int, int, int]:
    """``(tick font size, shared left-axis width, bottom-axis height)``.

    The bottom axis has to grow with the font or the tick labels are clipped.
    """
    axis_fs = getattr(main_window, "_plot_params", {}).get(
        "axis_font_size", 10
    )
    left_w = getattr(main_window, "_shared_left_axis_width", 30)
    return axis_fs, left_w, max(24, int(axis_fs * 2.6))


def _size_row_axes(
    plot_item,
    xmax,
    *,
    left_margin_px: int = _PLOT_LEFT_MARGIN_PX,
    right_pad: float = 0.5,
    left_axis_width_px: int = 28,
    bottom_axis_height_px: int = 24,
) -> None:
    """Set a row's x-range, plot margins and axis box sizes."""
    with contextlib.suppress(
        RuntimeError, AttributeError, TypeError, ValueError
    ):
        plot_item.vb.setXRange(0, xmax + right_pad, padding=0.0)

    plot_item.layout.setContentsMargins(left_margin_px, 6, 6, 6)

    ax_left = plot_item.getAxis("left")
    ax_bottom = plot_item.getAxis("bottom")
    if ax_left:
        ax_left.setWidth(left_axis_width_px)
    if ax_bottom:
        ax_bottom.setHeight(bottom_axis_height_px)


def _apply_row_time_axis(
    plot_item, main_window, xmax, min_left_width=0
) -> int:
    """Size a row's axes for the current font, and report that font size."""
    axis_fs, left_w, bottom_h = _row_axis_metrics(main_window)
    _size_row_axes(
        plot_item,
        xmax,
        left_margin_px=_PLOT_LEFT_MARGIN_PX,
        right_pad=0.5,
        left_axis_width_px=max(int(left_w), int(min_left_width or 0)),
        bottom_axis_height_px=bottom_h,
    )
    return axis_fs


def _style_row_axes(plot_item, axis_fs: int) -> None:
    """White axis lines with no tick marks, in the chosen tick font."""
    ax_left = plot_item.getAxis("left")
    if ax_left is not None:
        ax_left.setStyle(showValues=True)
        ax_left.setPen(pg.mkPen("w", width=3))
        ax_left.setTextPen(pg.mkPen("w"))
        ax_left.setTickPen(pg.mkPen(None))

    ax_bottom = plot_item.getAxis("bottom")
    if ax_bottom is not None:
        ax_bottom.setPen(pg.mkPen("w", width=3))
        ax_bottom.setTextPen(pg.mkPen("w"))
        ax_bottom.setTickPen(pg.mkPen(None))

    font = axis_font(axis_fs)
    if ax_left is not None:
        ax_left.setStyle(tickFont=font)
    if ax_bottom is not None:
        ax_bottom.setStyle(tickFont=font)


def _row_xmax(df_subset: pd.DataFrame, xcol: str, default):
    """Largest finite x value in this row's x column, or ``default``."""
    try:
        xmax = _finite_max(df_subset[xcol], default=default)
    except (KeyError, TypeError, ValueError):
        return default
    if xmax is None or not np.isfinite(xmax):
        return default
    return xmax


def _feature_combo_needs_reload(feat_combo, keys: list[str]) -> bool:
    """True when the combo's contents no longer match the available features."""
    if feat_combo.count() != len(keys):
        return True
    return any(feat_combo.itemData(i) != key for i, key in enumerate(keys))


def _initial_feature_index(
    keys: list[str], previous, features_defaulted: bool
) -> int:
    """Which feature to select after rebuilding the list."""
    if previous in keys:
        return keys.index(previous)
    if not features_defaulted and DEFAULT_FEATURE in keys:
        return keys.index(DEFAULT_FEATURE)
    return 0


def _sync_feature_combo(
    main_window, row_idx: int, feat_combo, feature_defs: dict
) -> None:
    """Rebuild one row's Features list if the available features changed."""
    keys = sorted(feature_defs.keys(), key=lambda k: k.replace("_", " "))
    if not _feature_combo_needs_reload(feat_combo, keys):
        return

    previous = getattr(main_window, "selected_feature_by_row", {}).get(row_idx)

    feat_combo.blockSignals(True)
    feat_combo.clear()
    for key in keys:
        feat_combo.addItem(key, key)
    feat_combo.setCurrentIndex(
        _initial_feature_index(keys, previous, main_window._features_defaulted)
    )
    main_window.selected_feature_by_row[row_idx] = feat_combo.currentData()
    feat_combo.blockSignals(False)


def _sync_mask_combo(main_window, row_idx: int, m_combo, m_count: int) -> None:
    """Rebuild one row's mask picker if the mask count changed."""
    if not (m_combo.isEnabled() and m_combo.count() != m_count):
        return

    m_combo.blockSignals(True)
    m_combo.clear()
    for mask in range(1, m_count + 1):
        m_combo.addItem(str(mask), mask)

    selected = getattr(main_window, "selected_m_by_channel", {}).get(
        row_idx, 1
    )
    selected = 1 if selected is None else int(selected)
    selected = max(1, min(max(m_count, 1), selected)) if m_count > 0 else 1

    m_combo.setCurrentIndex(selected - 1)
    main_window.selected_m_by_channel[row_idx] = selected
    m_combo.blockSignals(False)


def _sync_channel_combo(
    main_window, row_idx: int, ch_combo, ch_count: int
) -> None:
    """Rebuild one row's channel picker if the channel count changed."""
    if not (ch_combo.isEnabled() and ch_combo.count() != ch_count):
        return

    ch_combo.blockSignals(True)
    ch_combo.clear()
    for channel in range(ch_count):
        label = f"{channel:02d}"
        ch_combo.addItem(label, label)

    selected = getattr(main_window, "selected_ch_by_channel", {}).get(
        row_idx, "00"
    )
    available = [ch_combo.itemData(i) for i in range(ch_combo.count())]
    if selected is None or selected not in available:
        selected = "00"

    ch_combo.setCurrentIndex(
        available.index(selected) if selected in available else 0
    )
    main_window.selected_ch_by_channel[row_idx] = selected
    ch_combo.blockSignals(False)


def _sync_row_combos(
    main_window,
    row_idx: int,
    feature_defs: dict,
    ch_count: int,
    m_count: int,
) -> None:
    """Bring one row's Feature / M / CH pickers in line with the data.

    Order matters: the feature list is rebuilt first because the channel and
    mask pickers are shown or hidden according to whichever feature ends up
    selected.
    """
    tools = getattr(main_window, "row_tools", {}).get(row_idx, None)
    if not tools:
        return

    feat_combo = tools["feat"]
    m_combo = tools["m"]
    ch_combo = tools["ch"]

    _sync_feature_combo(main_window, row_idx, feat_combo, feature_defs)
    _sync_mask_combo(main_window, row_idx, m_combo, m_count)

    definition = feature_defs.get(feat_combo.currentData(), {})
    has_ch = definition.get("has_ch", False)
    has_m = bool(definition.get("has_m", True))

    ch_label = tools.get("ch_label")
    if ch_label is not None:
        ch_label.setVisible(has_ch)
    ch_combo.setVisible(has_ch)

    m_label = tools.get("m_label")
    if m_label is not None:
        m_label.setVisible(has_m)
    m_combo.setVisible(has_m)

    if has_ch:
        _sync_channel_combo(main_window, row_idx, ch_combo, ch_count)


def _sync_all_row_combos(
    main_window,
    max_plots: int,
    feature_defs: dict,
    ch_count: int,
    m_count: int,
) -> None:
    """Populate every row's Features/M/CH dropdowns to match current data."""
    for row in range(1, max_plots + 1):
        _sync_row_combos(main_window, row, feature_defs, ch_count, m_count)

    if not main_window._features_defaulted:
        main_window._features_defaulted = True


_FALLBACK_PALETTE = (
    "#4FC3F7",
    "#81C784",
    "#FFB74D",
    "#E57373",
    "#BA68C8",
    "#64B5F6",
    "#A1887F",
    "#90A4AE",
)


def _visible_tracks(all_tracks, selected, zoom) -> list:
    """Tracks to draw: the selection, then narrowed further by the zoom."""
    tracks = (
        [t for t in all_tracks if int(t) in selected]
        if selected
        else list(all_tracks)
    )
    if zoom:
        tracks = [t for t in tracks if int(t) in zoom]
    return tracks


def _track_color_map(all_tracks: list) -> dict:
    """One color per track, spread across tab20, or a fixed palette if that fails."""
    try:
        cmap = plt.get_cmap("tab20")
    except (RuntimeError, AttributeError, TypeError, ValueError):
        cmap = None

    if cmap is None:
        palette = [pg.mkColor(c) for c in _FALLBACK_PALETTE]
        return {
            track: palette[i % len(palette)]
            for i, track in enumerate(all_tracks)
        }

    span = max(len(all_tracks), 1)
    return {
        track: pg.mkColor(tuple(int(c * 255) for c in cmap(i / span)[:3]))
        for i, track in enumerate(all_tracks)
    }


def _highlight_colors(main_window) -> dict:
    """Painted highlight colors, keyed by integer TrackNumber."""
    color_map = getattr(main_window, "_track_highlight_colors", {}) or {}
    with contextlib.suppress(AttributeError, TypeError, ValueError):
        return {int(k): v for k, v in color_map.items()}
    return color_map


class _RowColumn(NamedTuple):
    """One row's combo selections, resolved down to an actual data column."""

    feat_key: str
    col_name: str
    xcol: str
    m_idx: int | None
    channel: str | None


class _RowRenderContext(NamedTuple):
    """Inputs shared by every plot row on one redraw."""

    unique_tracks: list
    global_xmax: float
    hl_on: bool
    color_map: dict
    track_colors: dict
    symbol: object
    sym_size: float
    line_w: float
    hl_line_w: float


def _reset_plot_widget(pw: pg.PlotWidget):
    """Blank one plot widget and strip its unused context-menu entries.

    Returns the widget's ``PlotItem`` so callers don't need a second lookup.
    """
    pw.clear()
    pw.setBackground("k")
    pw.setMouseEnabled(False, False)
    plot_item = pw.getPlotItem()

    try:
        plot_item.vb.setMouseMode(pg.ViewBox.RectMode)
        plot_item.ctrlMenu = None
        vb_menu = plot_item.vb.menu
        for action in list(vb_menu.actions()):
            if action.text() in ("View All", "Mouse Mode"):
                vb_menu.removeAction(action)
    except (RuntimeError, AttributeError, TypeError, ValueError):
        pass

    return plot_item


def _resolve_row_mask_index(m_combo, m_count: int) -> int:
    """The row's mask index from its combo box, clamped into ``[1, m_count]``."""
    try:
        m_idx = int(m_combo.currentData())
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return 1
    if not isinstance(m_idx, int) or not (1 <= m_idx <= m_count):
        return 1
    return m_idx


def _resolve_row_channel(ch_combo) -> str:
    """The row's zero-padded channel label (e.g. ``"03"``) from its combo box."""
    if ch_combo is None:
        return "00"
    channel = ch_combo.currentData()
    if not isinstance(channel, str):
        return "00"
    try:
        return f"{int(channel):02d}"
    except (ValueError, TypeError):
        return channel if channel else "00"


def _resolve_row_column(
    main_window,
    tools: dict,
    feature_defs: dict,
    m_count: int,
    df_subset: pd.DataFrame,
) -> _RowColumn | None:
    """Turn one row's Feature/M/CH combo selections into a plottable column."""
    feat_combo = tools.get("feat")
    m_combo = tools.get("m")
    ch_combo = tools.get("ch")

    feat_key = feat_combo.currentData() if feat_combo is not None else None
    if feat_key is None or feat_key not in feature_defs:
        return None

    definition = feature_defs[feat_key]
    has_ch = bool(definition.get("has_ch", False))
    has_m = bool(definition.get("has_m", True))

    m_idx = _resolve_row_mask_index(m_combo, m_count) if has_m else None
    channel = _resolve_row_channel(ch_combo) if has_ch else None

    col_name = _format_feature_column(
        definition.get("template"),
        has_ch=has_ch,
        has_m=has_m,
        channel=channel,
        m_idx=m_idx,
    )
    if not col_name or col_name not in df_subset.columns:
        return None

    try:
        ch_idx = int(channel)
    except (TypeError, ValueError):
        ch_idx = 1

    mode = getattr(main_window, "_time_mode", TIME_MODE_T)
    xcol = x_column_for(mode, ch_idx=ch_idx, df_cols=list(df_subset.columns))
    return _RowColumn(feat_key, col_name, xcol, m_idx, channel)


def _render_resolved_row(
    main_window,
    pw: pg.PlotWidget,
    plot_item,
    df_subset: pd.DataFrame,
    resolved: _RowColumn,
    ctx: _RowRenderContext,
) -> int:
    """Draw one row's curves and style its axes, now that its column is known."""
    min_y_row, max_y_row = _draw_row_curves(
        pw,
        df_subset,
        _row_curve_plan(
            ctx.unique_tracks,
            highlight_on=ctx.hl_on,
            color_map=ctx.color_map,
            track_colors=ctx.track_colors,
            line_w=ctx.line_w,
            hl_line_w=ctx.hl_line_w,
        ),
        resolved.xcol,
        resolved.col_name,
        symbol=ctx.symbol,
        sym_size=ctx.sym_size,
    )

    axis_fs = _row_axis_metrics(main_window)[0]
    y_lo, y_hi = _y_view_range(min_y_row, max_y_row)
    needed_left_w = apply_y_range_and_ticks(plot_item, y_lo, y_hi, axis_fs)

    _apply_row_time_axis(
        plot_item,
        main_window,
        _row_xmax(df_subset, resolved.xcol, ctx.global_xmax),
        min_left_width=needed_left_w,
    )
    _style_row_axes(plot_item, axis_fs)

    # Circular-import: gui.main_window imports update_plot from this package
    from SECQUOIA.gui.main_window.dynamics_plot_time_menu import (
        install_time_menu,
    )

    install_time_menu(plot_item, main_window)
    return needed_left_w


def _global_x_max(main_window, df_subset: pd.DataFrame) -> float:
    """Largest x value in the current Identification, used to align the rows.

    Channel 1 is the reference: in real-time mode each channel has its own
    timestamps, and rows would otherwise end at slightly different points.
    """
    try:
        xcol = x_column_for(
            getattr(main_window, "_time_mode", TIME_MODE_T),
            ch_idx=1,
            df_cols=list(df_subset.columns),
        )
        global_xmax = _finite_max(df_subset[xcol], default=1.0)
    except (KeyError, TypeError, ValueError):
        return 1.0
    if not np.isfinite(global_xmax) or global_xmax <= 0:
        return 1.0
    return global_xmax


def _global_y_range(
    main_window, df_subset: pd.DataFrame, feature_defs: dict
) -> tuple[float, float]:
    """``(min, max)`` across every currently selected feature column.

    Only used to size the shared left axis, so all rows line up even when
    their own values differ in magnitude.
    """
    try:
        y_cols = [
            feature_defs[key]["template"].format(
                ch=main_window.selected_ch_by_channel.get(row, 1),
                m=main_window.selected_m_by_channel.get(row, 1),
            )
            for row, key in main_window.selected_feature_by_row.items()
            if key in feature_defs
        ]
        present = [c for c in y_cols if c in df_subset.columns]
        highs = [
            v
            for v in (_finite_max(df_subset[c]) for c in present)
            if v is not None and np.isfinite(v)
        ]
        lows = [
            v
            for v in (_finite_min(df_subset[c]) for c in present)
            if v is not None and np.isfinite(v)
        ]
        global_ymax = max(highs, default=1.0)
        global_ymin = min(lows, default=0.0)
    except (KeyError, TypeError, ValueError):
        global_ymax, global_ymin = 1.0, 0.0

    if not np.isfinite(global_ymax):
        global_ymax = 1.0
    if not np.isfinite(global_ymin):
        global_ymin = 0.0
    if global_ymax <= global_ymin:
        global_ymax = global_ymin + 1.0
    return global_ymin, global_ymax


def _compute_shared_scales(
    main_window, df_subset: pd.DataFrame, feature_defs: dict
) -> float:
    """Compute and cache the x/y scale limits shared by every plot row.

    Returns the shared x-axis max, so every row's time axis lines up.
    """
    global_xmax = _global_x_max(main_window, df_subset)
    main_window._shared_time_xmax = global_xmax

    axis_fs = getattr(main_window, "_plot_params", {}).get(
        "axis_font_size", 10
    )
    global_ymin, global_ymax = _global_y_range(
        main_window, df_subset, feature_defs
    )
    main_window._shared_left_axis_width = _shared_left_axis_width(
        global_ymin, global_ymax, axis_fs
    )
    return global_xmax


def get_plot_params(main_window) -> tuple[str | None, int, float, float]:
    """Return (symbol, symbol_size, line_width, hl_line_width) with defaults."""
    p = getattr(main_window, "_plot_params", {}) or {}
    symbol = p.get("symbol", "o")
    if symbol in ("", None):
        symbol = None
    try:
        sym_size = int(
            p.get(
                "symbol_size",
                getattr(
                    getattr(main_window, "STYLE", object()), "SYMBOL_SIZE", 6
                ),
            )
        )
    except (TypeError, ValueError):
        sym_size = 6
    try:
        lw = float(p.get("line_width", 2.0))
    except (TypeError, ValueError):
        lw = 2.0
    try:
        hlw = float(p.get("hl_line_width", 3.0))
    except (TypeError, ValueError):
        hlw = max(3.0, lw)
    return symbol, sym_size, lw, hlw


def _build_row_render_context(
    main_window, df_subset: pd.DataFrame, ident: str, global_xmax: float
) -> _RowRenderContext:
    """Build the per row render context (colors, sizes, highlight state)
    shared by every plot row, and cache this identification's track colors.
    """
    all_tracks = list(df_subset["TrackNumber"].unique())
    unique_tracks = _visible_tracks(
        all_tracks,
        getattr(main_window, "_selected_tracks", set()),
        getattr(main_window, "_zoom_visible_tracks", set()),
    )

    symbol, sym_size, line_w, hl_line_w = get_plot_params(main_window)
    track_colors = _track_color_map(all_tracks)
    if not hasattr(main_window, "_track_colors"):
        main_window._track_colors = {}
    main_window._track_colors[str(ident)] = track_colors

    return _RowRenderContext(
        unique_tracks=unique_tracks,
        global_xmax=global_xmax,
        hl_on=bool(getattr(main_window, "_hl_active", False)),
        color_map=_highlight_colors(main_window),
        track_colors=track_colors,
        symbol=symbol,
        sym_size=sym_size,
        line_w=line_w,
        hl_line_w=hl_line_w,
    )


def _reset_row_y_axis(main_window, plot_item) -> int:
    """Put an empty row's Y axis back to the default range and ticks."""
    axis_fs = _row_axis_metrics(main_window)[0]
    y_lo, y_hi = _y_view_range(None, None)
    return apply_y_range_and_ticks(plot_item, y_lo, y_hi, axis_fs)


def _apply_shared_left_width(
    main_window, plot_items: list, widths: list[int]
) -> None:
    """Give every row the same left axis, wide enough for the widest label."""
    estimate = int(getattr(main_window, "_shared_left_axis_width", 0) or 0)
    shared = max([estimate, *widths], default=estimate)
    if shared <= 0:
        return

    main_window._shared_left_axis_width = shared
    for plot_item in plot_items:
        with contextlib.suppress(
            RuntimeError, AttributeError, TypeError, ValueError
        ):
            ax_left = plot_item.getAxis("left")
            if ax_left is not None:
                ax_left.setWidth(shared)


def _render_all_rows(
    main_window,
    max_plots: int,
    feature_defs: dict,
    m_count: int,
    df_subset: pd.DataFrame,
    render_ctx: _RowRenderContext,
    global_xmax: float,
) -> None:
    """Render each plot row's curves and axes for the resolved column."""
    rendered: list = []
    needed_widths: list[int] = []

    for row in range(1, max_plots + 1):
        pw: pg.PlotWidget = getattr(main_window, f"plot_widget_{row}", None)
        if pw is None:
            continue

        plot_item = _reset_plot_widget(pw)
        rendered.append(plot_item)
        tools = getattr(main_window, "row_tools", {}).get(row, {})

        resolved = _resolve_row_column(
            main_window, tools, feature_defs, m_count, df_subset
        )
        if resolved is None:
            needed_widths.append(_reset_row_y_axis(main_window, plot_item))
            _apply_row_time_axis(plot_item, main_window, global_xmax)
            continue

        needed_widths.append(
            _render_resolved_row(
                main_window, pw, plot_item, df_subset, resolved, render_ctx
            )
        )

        main_window.selected_feature_by_row[row] = resolved.feat_key
        main_window.selected_m_by_channel[row] = resolved.m_idx
        main_window.selected_ch_by_channel[row] = resolved.channel

    _apply_shared_left_width(main_window, rendered, needed_widths)
