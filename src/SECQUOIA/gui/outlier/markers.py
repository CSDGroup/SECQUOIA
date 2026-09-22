"""Star markers drawn on top of the dynamics plots.

Orange stars mark outlier time points, purple stars mark time points whose
tracking point has a close second mask (``Close_mask_flag == "Flagged"``).
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

import numpy as np
import pandas as pd
import pyqtgraph as pg

from SECQUOIA.config import PLOTPARAMETERS
from SECQUOIA.core.outlier_detection.close_masks import (
    CLOSE_MASK_FLAG,
    FLAG_FLAGGED,
)
from SECQUOIA.utils.plotting import (
    TIME_MODE_T,
    _current_ident,
    _format_feature_column,
    _plot_counts,
    x_column_for,
)

__all__ = ["update_close_mask_marker", "update_outlier_marker"]

_OUTLIER_STORE = "_outlier_marker_items"
_CLOSE_MASK_STORE = "_close_mask_marker_items"

# Extra size of a purple star that sits on an orange one: the purple star
# then frames the orange star instead of hiding it.
_CLOSE_MASK_OVERLAP_GROWTH = 8


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


def _clear_markers(main_window, store_attr: str, rows=None) -> None:
    """Remove star markers from `rows`, or from every row when `rows` is None."""
    stored = getattr(main_window, store_attr, {})
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


def _clear_close_mask_markers(main_window, rows=None) -> None:
    """Remove purple star markers from `rows`, or from every row when None."""
    _clear_markers(main_window, _CLOSE_MASK_STORE, rows)


def _new_star_symbol() -> str:
    """Return "star" if pyqtgraph supports that symbol, else the round fallback."""
    with contextlib.suppress(
        RuntimeError, AttributeError, TypeError, ValueError
    ):
        pg.ScatterPlotItem([0], [0], symbol="star")
        return "star"
    return "o"


def _draw_star_row(
    main_window,
    plot_widget,
    df_flagged: pd.DataFrame,
    df_subset: pd.DataFrame,
    feature_defs: dict,
    row: int,
    star_symbol: str,
    *,
    color=PLOTPARAMETERS.ORANGE,
    size: float | pd.Series = PLOTPARAMETERS.OUTLIERSIZE,
    z_value: int = 10_000,
):
    """Draw one row's star markers, or return ``None`` if there are none.

    `size` is one size for every star, or a series aligned with
    `df_flagged` giving each star its own.
    """
    col_name, feat_key = _column_for_row(main_window, feature_defs, row)
    if not col_name or col_name not in df_subset.columns:
        return None

    ch_idx = getattr(main_window, "selected_ch_by_channel", {}).get(row, 1)
    xcol = x_column_for(
        getattr(main_window, "_time_mode", TIME_MODE_T),
        ch_idx=int(ch_idx or 1),
        df_cols=list(df_subset.columns),
    )
    x_series = df_flagged.get(xcol, None)
    if x_series is None:
        return None

    t_vals = np.asarray(x_series.to_numpy(), dtype=float)
    y_vals = np.asarray(df_flagged[col_name].to_numpy(), dtype=float)
    finite = np.isfinite(t_vals) & np.isfinite(y_vals)
    if not finite.any():
        return None

    sizes = (
        np.asarray(size.to_numpy(), dtype=float)[finite]
        if isinstance(size, pd.Series)
        else size
    )
    star_item = pg.ScatterPlotItem(
        t_vals[finite],
        y_vals[finite],
        symbol=star_symbol,
        size=sizes,
        pen=pg.mkPen(color, width=2),
        brush=pg.mkBrush(color),
        hoverable=False,
    )
    with contextlib.suppress(RuntimeError, AttributeError, TypeError):
        star_item.setZValue(z_value)
    plot_widget.addItem(star_item)

    return {
        "plot_widget": plot_widget,
        "item": star_item,
        "column": col_name,
        "feature": feat_key,
    }


def _draw_flagged_stars(
    main_window,
    df_subset: pd.DataFrame,
    df_flagged: pd.DataFrame,
    *,
    store_attr: str,
    color,
    size: float | Callable[[pd.DataFrame], pd.Series],
    z_value: int,
) -> None:
    """Redraw one kind of star on every plot row from the flagged rows."""
    if not hasattr(main_window, store_attr):
        setattr(main_window, store_attr, {})

    _, _, max_plots = _plot_counts(main_window)
    if max_plots < 1:
        return

    if df_flagged.empty:
        _clear_markers(main_window, store_attr)
        setattr(main_window, store_attr, {})
        return

    feature_defs = getattr(main_window, "_feature_defs", None)
    if not feature_defs:
        return

    _clear_markers(main_window, store_attr, range(1, max_plots + 1))

    star_symbol = _new_star_symbol()
    star_size = size(df_flagged) if callable(size) else size
    for row in range(1, max_plots + 1):
        plot_widget = getattr(main_window, f"plot_widget_{row}", None)
        if plot_widget is None:
            continue
        marker = _draw_star_row(
            main_window,
            plot_widget,
            df_flagged,
            df_subset,
            feature_defs,
            row,
            star_symbol,
            color=color,
            size=star_size,
            z_value=z_value,
        )
        if marker is not None:
            getattr(main_window, store_attr)[row] = marker


def _update_orange_markers(main_window) -> None:
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

    _draw_flagged_stars(
        main_window,
        df_subset,
        df_out,
        store_attr=_OUTLIER_STORE,
        color=PLOTPARAMETERS.ORANGE,
        size=PLOTPARAMETERS.OUTLIERSIZE,
        z_value=10_000,
    )


def _close_mask_star_sizes(df_flagged: pd.DataFrame) -> pd.Series:
    """Star size per flagged row: larger where the point is also an outlier."""
    sizes = pd.Series(
        float(PLOTPARAMETERS.OUTLIERSIZE), index=df_flagged.index
    )
    if "Outlier_detection" in df_flagged.columns:
        is_outlier = df_flagged["Outlier_detection"] == "Outlier"
        sizes[is_outlier] += _CLOSE_MASK_OVERLAP_GROWTH
    return sizes


def update_close_mask_marker(main_window) -> None:
    """Draw purple stars wherever the current ident has a flagged close mask.

    A star sits on every row at each time point whose `Close_mask_flag` is
    "Flagged". Where the point is also an outlier the purple star is drawn
    larger and just below the orange one, so both stay visible.
    """
    df_all = getattr(main_window, "filtered_df", None)
    if df_all is None or len(df_all) == 0:
        _clear_close_mask_markers(main_window)
        return

    ident = _current_ident(main_window, df_all)
    df_subset = df_all[df_all["Identification"] == ident]
    if df_subset.empty:
        return

    if CLOSE_MASK_FLAG not in df_subset.columns:
        _clear_close_mask_markers(main_window)
        return
    df_flagged = df_subset[df_subset[CLOSE_MASK_FLAG] == FLAG_FLAGGED]

    _draw_flagged_stars(
        main_window,
        df_subset,
        df_flagged,
        store_attr=_CLOSE_MASK_STORE,
        color=PLOTPARAMETERS.PURPLE,
        size=_close_mask_star_sizes,
        z_value=9_999,
    )


def update_outlier_marker(main_window) -> None:
    """Redraw the orange outlier stars and the purple close-mask stars.

    Every caller that used to refresh only the outlier stars goes through
    here, so the two kinds are always redrawn together.
    """
    _update_orange_markers(main_window)
    update_close_mask_marker(main_window)
