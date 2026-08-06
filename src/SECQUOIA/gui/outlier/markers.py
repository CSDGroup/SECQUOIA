"""Outlier star markers drawn on top of the dynamics plots."""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
import pyqtgraph as pg

from SECQUOIA.config import PLOTPARAMETERS
from SECQUOIA.utils.plotting import (
    TIME_MODE_T,
    _current_ident,
    _format_feature_column,
    _plot_counts,
    x_column_for,
)

__all__ = ["update_outlier_marker"]


def _column_for_row(main_window, feature_defs: dict, row_idx: int):
    """The dataframe column a plot row is currently showing, and its feature."""
    feat_key = getattr(main_window, "selected_feature_by_row", {}).get(row_idx)
    if not feat_key or feat_key not in feature_defs:
        return None, None

    definition = feature_defs[feat_key]
    m_idx = getattr(main_window, "selected_m_by_channel", {}).get(row_idx, 1)
    channel = getattr(main_window, "selected_ch_by_channel", {}).get(
        row_idx, 1
    )

    col = _format_feature_column(
        definition.get("template"),
        has_ch=bool(definition.get("has_ch", False)),
        has_m=bool(definition.get("has_m", True)),
        channel=channel,
        m_idx=m_idx,
    )
    return col, feat_key


def _clear_outlier_markers(main_window, rows=None) -> None:
    """Remove star markers from `rows`, or from every row when `rows` is None."""
    stored = getattr(main_window, "_outlier_marker_items", {})
    targets = list(stored) if rows is None else list(rows)
    for row in targets:
        marker = stored.get(row)
        if not marker:
            continue
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            plot_widget = marker.get("plot_widget")
            item = marker.get("item")
            if plot_widget is not None and item is not None:
                plot_widget.removeItem(item)
        stored.pop(row, None)


def _new_star_symbol() -> str:
    """Return "star" if pyqtgraph supports that symbol, else the round fallback."""
    with contextlib.suppress(
        RuntimeError, AttributeError, TypeError, ValueError
    ):
        pg.ScatterPlotItem([0], [0], symbol="star")
        return "star"
    return "o"


def _draw_outlier_row(
    main_window,
    plot_widget,
    df_out: pd.DataFrame,
    df_subset: pd.DataFrame,
    feature_defs: dict,
    row: int,
    star_symbol: str,
):
    """Draw one row's outlier markers, or return ``None`` if there are none."""
    col_name, feat_key = _column_for_row(main_window, feature_defs, row)
    if not col_name or col_name not in df_subset.columns:
        return None

    ch_idx = getattr(main_window, "selected_ch_by_channel", {}).get(row, 1)
    xcol = x_column_for(
        getattr(main_window, "_time_mode", TIME_MODE_T),
        ch_idx=int(ch_idx or 1),
        df_cols=list(df_subset.columns),
    )
    x_series = df_out.get(xcol, None)
    if x_series is None:
        return None

    t_vals = np.asarray(x_series.to_numpy(), dtype=float)
    y_vals = np.asarray(df_out[col_name].to_numpy(), dtype=float)
    finite = np.isfinite(t_vals) & np.isfinite(y_vals)
    if not finite.any():
        return None

    star_item = pg.ScatterPlotItem(
        t_vals[finite],
        y_vals[finite],
        symbol=star_symbol,
        size=PLOTPARAMETERS.OUTLIERSIZE,
        pen=pg.mkPen(PLOTPARAMETERS.ORANGE, width=2),
        brush=pg.mkBrush(PLOTPARAMETERS.ORANGE),
        hoverable=False,
    )
    with contextlib.suppress(RuntimeError, AttributeError, TypeError):
        star_item.setZValue(10_000)
    plot_widget.addItem(star_item)

    return {
        "plot_widget": plot_widget,
        "item": star_item,
        "column": col_name,
        "feature": feat_key,
    }


def update_outlier_marker(main_window) -> None:
    """Draw orange star markers wherever the current ident is flagged as an outlier.

    A star sits on every row at each time point whose `Outlier_detection` is
    "Outlier", using that row's own selected feature column.
    """
    df_all = getattr(main_window, "filtered_df", None)
    if df_all is None or len(df_all) == 0:
        _, _, max_plots = _plot_counts(main_window)
        for row in range(1, max_plots + 1):
            plot_widget = getattr(main_window, f"plot_widget_{row}", None)
            if isinstance(plot_widget, pg.PlotWidget):
                plot_widget.clear()
        return

    ident = _current_ident(main_window, df_all)
    df_subset = df_all[df_all["Identification"] == ident]
    if df_subset.empty:
        return

    outcol = "Outlier_detection"
    if outcol not in df_subset.columns:
        return
    df_out = df_subset[df_subset[outcol] == "Outlier"]

    if not hasattr(main_window, "_outlier_marker_items"):
        main_window._outlier_marker_items = {}

    _, _, max_plots = _plot_counts(main_window)
    if max_plots < 1:
        return

    if df_out.empty:
        _clear_outlier_markers(main_window)
        main_window._outlier_marker_items = {}
        return

    feature_defs = getattr(main_window, "_feature_defs", None)
    if not feature_defs:
        return

    _clear_outlier_markers(main_window, range(1, max_plots + 1))

    star_symbol = _new_star_symbol()
    for row in range(1, max_plots + 1):
        plot_widget = getattr(main_window, f"plot_widget_{row}", None)
        if plot_widget is None:
            continue
        marker = _draw_outlier_row(
            main_window,
            plot_widget,
            df_out,
            df_subset,
            feature_defs,
            row,
            star_symbol,
        )
        if marker is not None:
            main_window._outlier_marker_items[row] = marker
