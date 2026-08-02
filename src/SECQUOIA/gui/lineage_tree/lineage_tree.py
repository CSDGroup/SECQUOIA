"""Interactive lineage-tree plotting utilities for SECQUOIA.

This module builds and renders cell lineage trees from tracking data stored in
a pandas DataFrame. It supports plain lineage views, single-channel heatmap
overlays, multi-channel stacked heatmaps, cell-fate event markers, track
selection, track highlighting, and synchronization between the lineage tree and
row-level time-series plots."""

from __future__ import annotations

import contextlib
import logging
import numbers

import numpy as np
import pandas as pd
import pyqtgraph as pg
from qtpy import QtWidgets

from SECQUOIA.gui.common.ui_utils import clear_layout
from SECQUOIA.gui.lineage_tree.lineage_geometry import (
    FeatureCatalog,
    TrackStats,
    assign_y_tidy,
    build_edges,
    build_track_table,
    generation,
    left,
    parent,
    right,
)
from SECQUOIA.gui.lineage_tree.lineage_view import LineageTreeView
from SECQUOIA.gui.lineage_tree.lineage_zoom import (
    _attach_lineage_zoom_sync,
    _lineage_view_reusable,
    _register_redraw,
    _reset_lineage_view,
)
from SECQUOIA.utils.plotting import TIME_MODE_T, update_plot

LOG = logging.getLogger(__name__)

__all__ = [
    "FeatureCatalog",
    "TrackStats",
    "assign_y_tidy",
    "build_edges",
    "build_track_table",
    "generation",
    "left",
    "lineage_tree",
    "parent",
    "right",
]


def lineage_tree(main_window) -> None:
    """Show the lineage tree for the current identification.

    Repaints the existing plot widget in place when possible; only tears
    down and rebuilds it when there is no reusable view yet, or the render
    mode (plain vs. heatmap) changed.
    """
    df = getattr(main_window, "filtered_df", None)
    if not hasattr(main_window, "_time_mode"):
        main_window._time_mode = TIME_MODE_T
    layout = getattr(main_window, "graph3_layout", None)
    if layout is None or not isinstance(layout, QtWidgets.QLayout):
        raise ValueError(
            "lineage_tree: main_window.graph3_layout must be a QLayout"
        )

    if (
        not isinstance(df, pd.DataFrame)
        or df.empty
        or "Identification" not in df.columns
    ):
        _reset_lineage_view(main_window, layout)
        return

    ids_in_df = set(df["Identification"].astype(str).dropna().unique())
    ident = getattr(main_window, "ident", None)
    if not ident or str(ident) not in ids_in_df:
        nonnull = df["Identification"].dropna()
        if nonnull.empty:
            _reset_lineage_view(main_window, layout)
            return
        ident = str(nonnull.iloc[0])
    main_window.ident = ident

    xmax_override = getattr(main_window, "_shared_time_xmax", None)
    if not (
        isinstance(xmax_override, numbers.Real) and np.isfinite(xmax_override)
    ):
        xmax_override = None

    render_mode = "T"
    tools = getattr(main_window, "lineage_tools", None)
    if tools and "F" in tools:
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            txt = tools["F"].currentText()
            if txt:
                render_mode = txt

    colors_for_ident = getattr(main_window, "_track_colors", {}).get(
        str(ident)
    )

    if _lineage_view_reusable(main_window, render_mode):
        main_window.graph3_view.set_identification(
            ident,
            df,
            xmax_override=xmax_override,
            track_colors=colors_for_ident,
        )
    else:
        _register_redraw(main_window, None)
        clear_layout(layout)

        view = LineageTreeView(
            df,
            ident,
            xmax_override=xmax_override,
            track_colors=colors_for_ident,
            render_mode=render_mode,
            main_window=main_window,
        )
        main_window.graph3_view = view
        plot_widget = view.widget

        layout.addWidget(plot_widget)
        main_window.graph3_widget = plot_widget

        try:
            if isinstance(plot_widget, pg.PlotWidget):
                main_window.graph3_plot = plot_widget
            else:
                kids = plot_widget.findChildren(pg.PlotWidget)
                if kids:
                    main_window.graph3_plot = kids[0]
        except (RuntimeError, AttributeError, TypeError, ValueError):
            pass

        _attach_lineage_zoom_sync(main_window)

    with contextlib.suppress(RuntimeError, AttributeError, TypeError):
        vb = main_window.graph3_plot.getPlotItem().vb
        vb.setMouseMode(pg.ViewBox.RectMode)
        main_window.lineage_plot_widget = getattr(
            main_window, "graph3_plot", None
        )
        main_window._zoom_visible_tracks = None
        update_plot(main_window)

    with contextlib.suppress(RuntimeError, AttributeError, TypeError):
        pw = getattr(main_window, "graph3_plot", None)
        if isinstance(pw, pg.PlotWidget):
            vb = pw.getPlotItem().vb
            vb.enableAutoRange(pg.ViewBox.XYAxes, True)
            vb.autoRange()
