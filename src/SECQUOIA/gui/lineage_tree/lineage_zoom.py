"""Zoom capture/restore, cross-plot sync, and view lifecycle for the tree.

Preserves pyqtgraph zoom state across redraws and mirrors the lineage tree's
x range onto the row plots.
"""

from __future__ import annotations

import contextlib
import logging
import numbers
from collections.abc import Callable

import numpy as np
import pandas as pd
import pyqtgraph as pg
from qtpy import QtWidgets

from SECQUOIA.gui.common.ui_utils import clear_layout
from SECQUOIA.gui.lineage_tree.lineage_draw import BG_COLOR
from SECQUOIA.utils.plotting import sync_row_y_ticks, update_plot

LOG = logging.getLogger(__name__)


def _register_redraw(main_window, fn: Callable[[], None] | None) -> None:
    """Publish (or clear) the in-place redraw entry point for this tree."""
    with contextlib.suppress(RuntimeError, AttributeError, TypeError):
        main_window._lineage_redraw = fn


def refresh_lineage_values(main_window) -> bool:
    """Repaint the existing lineage tree in place, keeping the widget alive."""
    fn = getattr(main_window, "_lineage_redraw", None)
    if not callable(fn):
        return False

    pw = getattr(main_window, "graph3_plot", None)
    if not isinstance(pw, pg.PlotWidget):
        return False

    try:
        vb = pw.getPlotItem().vb
    except (RuntimeError, AttributeError, TypeError):
        return False

    rng = None
    auto_on = True
    with contextlib.suppress(
        RuntimeError, AttributeError, TypeError, KeyError
    ):
        rng = vb.viewRange()
        auto_on = bool(any(vb.state.get("autoRange", [True, True])))

    try:
        fn()
    except (
        RuntimeError,
        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
    ) as e:
        LOG.warning(
            "[lineage] in-place refresh failed (%s); rebuilding instead", e
        )
        return False

    if rng is not None and not auto_on:
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            vb.setRange(xRange=rng[0], yRange=rng[1], padding=0)

    return True


def _capture_view_range(pw) -> dict | None:
    """Per axis view range for one ``PlotWidget``, or ``None``.

    ``None`` means the widget is fully auto-ranging: it already shows the whole
    data extent, so there is nothing worth restoring.
    """
    if not isinstance(pw, pg.PlotWidget):
        return None
    try:
        vb = pw.getPlotItem().vb
        (x0, x1), (y0, y1) = vb.viewRange()
        auto = vb.state.get("autoRange", [True, True])
        x_auto, y_auto = bool(auto[0]), bool(auto[1])
    except (RuntimeError, AttributeError, TypeError, KeyError, IndexError):
        return None

    if x_auto and y_auto:
        return None
    return {
        "x": None if x_auto else (x0, x1),
        "y": None if y_auto else (y0, y1),
    }


def _restore_view_range(pw, saved: dict | None) -> None:
    """Reapply a range captured by :func:`_capture_view_range`."""
    if not saved or not isinstance(pw, pg.PlotWidget):
        return
    with contextlib.suppress(RuntimeError, AttributeError, TypeError):
        pw.getPlotItem().vb.setRange(
            xRange=saved.get("x"), yRange=saved.get("y"), padding=0
        )


def _dynamics_plot_widgets(main_window) -> list[tuple[str, pg.PlotWidget]]:
    """Every plot widget whose zoom should survive a mask-edit redraw:
    the lineage tree first, then each row's feature-vs-time plot."""
    widgets: list[tuple[str, pg.PlotWidget]] = []
    graph3 = getattr(main_window, "graph3_plot", None)
    if isinstance(graph3, pg.PlotWidget):
        widgets.append(("graph3_plot", graph3))

    max_rows = int(getattr(main_window, "_max_plot_rows", 0)) or 4
    for row in range(1, max_rows + 1):
        pw = getattr(main_window, f"plot_widget_{row}", None)
        if isinstance(pw, pg.PlotWidget):
            widgets.append((f"plot_widget_{row}", pw))
    return widgets


def capture_dynamics_zoom(main_window) -> dict:
    """Snapshot the current zoom of the lineage tree and every row plot."""
    return {
        name: _capture_view_range(pw)
        for name, pw in _dynamics_plot_widgets(main_window)
    }


def restore_dynamics_zoom(main_window, saved: dict | None) -> None:
    """Reapply a snapshot captured by :func:`capture_dynamics_zoom`."""
    if not saved:
        return
    for name, pw in _dynamics_plot_widgets(main_window):
        _restore_view_range(pw, saved.get(name))
        if name == "graph3_plot":
            continue
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            sync_row_y_ticks(
                pw.getPlotItem(),
                getattr(main_window, "_plot_params", {}).get(
                    "axis_font_size", 10
                ),
                min_width=getattr(
                    main_window, "_shared_left_axis_width", None
                ),
            )


def _attach_lineage_zoom_sync(main_window):
    """Keep the row plots and the visible-track set in step with the tree."""
    try:
        graph = getattr(main_window, "graph3_plot", None)
        if not isinstance(graph, pg.PlotWidget):
            return
        vb = graph.getPlotItem().vb
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return

    y_map = getattr(main_window, "_last_lineage_y_map", None)
    main_window._lineage_y_map = dict(y_map) if y_map else {}
    main_window._zoom_visible_tracks = None

    def _on_range_changed(*_):
        """Sync lineage zoom ranges to row plots and update visible-track filtering."""
        try:
            (x0, x1), (y0, y1) = vb.viewRange()
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return

        max_plots = int(getattr(main_window, "_max_plot_rows", 0)) or 4
        for row in range(1, max_plots + 1):
            pw = getattr(main_window, f"plot_widget_{row}", None)
            if pw is None:
                continue
            pvb = pw.getPlotItem().vb
            pvb.setLimits(xMin=0)
            pvb.setXRange(float(x0), float(x1), padding=0.0)
            with contextlib.suppress(
                RuntimeError, AttributeError, TypeError, ValueError
            ):
                pw.getPlotItem().vb.setXRange(
                    float(x0), float(x1), padding=0.0
                )

        ymap = getattr(main_window, "_lineage_y_map", {}) or {}
        vis = {int(tn) for tn, yy in ymap.items() if y0 <= float(yy) <= y1}
        main_window._zoom_visible_tracks = vis if vis else None

        with contextlib.suppress(
            RuntimeError, AttributeError, TypeError, ValueError
        ):
            update_plot(main_window)

        for row in range(1, max_plots + 1):
            pw = getattr(main_window, f"plot_widget_{row}", None)
            if pw is None:
                continue
            with contextlib.suppress(
                RuntimeError, AttributeError, TypeError, ValueError
            ):
                pw.getPlotItem().vb.setXRange(
                    float(x0), float(x1), padding=0.0
                )

    try:
        prev = getattr(vb, "_zoom_sync_slot", None)
        if prev is not None:
            vb.sigRangeChanged.disconnect(prev)
    except (RuntimeError, AttributeError, TypeError, ValueError):
        pass

    vb._zoom_sync_slot = _on_range_changed
    vb.sigRangeChanged.connect(_on_range_changed)

    with contextlib.suppress(
        RuntimeError, AttributeError, TypeError, ValueError
    ):
        vb.setMouseMode(pg.ViewBox.RectMode)


def _reset_lineage_view(main_window, layout: QtWidgets.QLayout) -> None:
    """Tear down any existing lineage view/widget and leave an empty plot."""
    _register_redraw(main_window, None)
    main_window.graph3_view = None
    clear_layout(layout)
    layout.addWidget(pg.PlotWidget(background=BG_COLOR))


def _lineage_view_reusable(main_window, render_mode: str) -> bool:
    """True when the existing lineage view can be repainted in place instead
    of being torn down and rebuilt."""
    from SECQUOIA.gui.lineage_tree.lineage_view import LineageTreeView

    view = getattr(main_window, "graph3_view", None)
    if not isinstance(view, LineageTreeView):
        return False
    if view.render_mode != (render_mode or "T").upper():
        return False
    try:
        view.plot_widget.isVisible()
    except RuntimeError:
        return False
    return True


def reset_lineage_zoom(main_window) -> None:
    """Reset the lineage tree zoom to show the full extent for the current ident."""
    try:
        pw = getattr(main_window, "graph3_plot", None)
        if pw is None:
            gw = getattr(main_window, "graph3_widget", None)
            if gw is not None:
                kids = gw.findChildren(pg.PlotWidget)
                if kids:
                    pw = kids[0]
        if not isinstance(pw, pg.PlotWidget):
            return
        plot = pw.getPlotItem()
        vb = plot.vb
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return

    df = getattr(main_window, "filtered_df", None)
    if (
        not isinstance(df, pd.DataFrame)
        or df.empty
        or "Identification" not in df.columns
    ):
        return

    ids_in_df = set(df["Identification"].astype(str).unique())
    ident = getattr(main_window, "ident", None)
    if not ident or ident not in ids_in_df:
        uids = getattr(main_window, "unique_ids", None)
        idx = getattr(main_window, "current_ident_index", None)
        if (
            isinstance(uids, (list | np.ndarray))
            and isinstance(idx, int)
            and 0 <= idx < len(uids)
        ):
            ident = str(uids[idx])
        else:
            ident = str(df["Identification"].iloc[0])
    main_window.ident = ident

    sub = df[df["Identification"].astype(str) == ident]
    if sub.empty:
        return

    t_series = pd.to_numeric(sub["t"], errors="coerce")
    xmax_override = getattr(main_window, "_shared_time_xmax", None)
    if isinstance(xmax_override, numbers.Real) and np.isfinite(xmax_override):
        xmax = float(xmax_override)
    else:
        tmax = float(np.nanmax(t_series)) if len(t_series) else 1.0
        xmax = max(1.0, tmax)

    y_map = getattr(main_window, "_last_lineage_y_map", None) or {}
    if y_map:
        ys = list(y_map.values())
        y_min = float(min(ys)) - 0.6
        y_max = float(max(ys)) + 0.6
    else:
        y_min, y_max = -1.0, 1.0

    try:
        vb.setLimits(xMin=0)
        vb.enableAutoRange(False, False)
        vb.setXRange(0.0, float(xmax), padding=0.0)
        vb.setYRange(float(y_min), float(y_max), padding=0.0)
        vb.setMouseMode(pg.ViewBox.RectMode)
        if hasattr(vb, "_tmp_shift_pan"):
            vb._tmp_shift_pan = False
    except (RuntimeError, AttributeError, TypeError, ValueError):
        pass

    try:
        main_window._zoom_visible_tracks = None
        update_plot(main_window)
        vb.setXRange(0.0, float(xmax), padding=0.0)
        vb.setYRange(float(y_min), float(y_max), padding=0.0)
    except (RuntimeError, AttributeError, TypeError, ValueError):
        pass
