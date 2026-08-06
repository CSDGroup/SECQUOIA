"""Plotting utilities for the SECQUOIA graphical interface."""

from __future__ import annotations

import contextlib

import pandas as pd

from SECQUOIA.utils.plotting.axis_ticks import (
    apply_time_axis,
    fit_row_y_axis,
    sync_row_y_ticks,
)
from SECQUOIA.utils.plotting.feature_discovery import (
    DEFAULT_FEATURE,
    EXCLUDED_FEATURES,
    TIME_MODE_CALC,
    TIME_MODE_REAL,
    TIME_MODE_T,
    _discover_features,
    _ensure_time_mode,
    _finite_max,
    _finite_min,
    _format_feature_column,
    x_column_for,
)
from SECQUOIA.utils.plotting.row_render import (
    DEFAULT_TRACK_COLOR,
    Z_HIGHLIGHT_BASE,
    Z_HIGHLIGHT_TOP,
    _build_row_render_context,
    _compute_shared_scales,
    _feature_combo_needs_reload,
    _global_y_range,
    _initial_feature_index,
    _render_all_rows,
    _row_axis_metrics,
    _row_curve_plan,
    _row_xmax,
    _style_row_axes,
    _sync_all_row_combos,
    _track_color_map,
    _track_xy,
    _visible_tracks,
)

__all__ = [
    "DEFAULT_FEATURE",
    "DEFAULT_TRACK_COLOR",
    "EXCLUDED_FEATURES",
    "TIME_MODE_CALC",
    "TIME_MODE_REAL",
    "TIME_MODE_T",
    "Z_HIGHLIGHT_BASE",
    "Z_HIGHLIGHT_TOP",
    "_current_ident",
    "_discover_features",
    "_feature_combo_needs_reload",
    "_finite_max",
    "_finite_min",
    "_format_feature_column",
    "_global_y_range",
    "_initial_feature_index",
    "_plot_counts",
    "_row_axis_metrics",
    "_row_curve_plan",
    "_row_xmax",
    "_style_row_axes",
    "_track_color_map",
    "_track_xy",
    "_visible_tracks",
    "apply_time_axis",
    "fit_row_y_axis",
    "sync_row_y_ticks",
    "update_plot",
    "x_column_for",
]


def _plot_counts(main_window) -> tuple[int, int, int]:
    """``(channel count, mask count, number of plot rows)``.
    The row count falls back to counting the plot widgets that actually exist,
    for windows built before ``_max_plot_rows`` was set.
    """
    try:
        ch_count = int(getattr(main_window, "n_channels", 0))
    except (RuntimeError, AttributeError, TypeError, ValueError):
        ch_count = 0
    try:
        m_count = getattr(main_window, "n_masks", 0)
    except (RuntimeError, AttributeError, TypeError, ValueError):
        m_count = 0

    max_plots = getattr(main_window, "_max_plot_rows", None)
    if not isinstance(max_plots, int) or max_plots < 1:
        max_plots = sum(
            1
            for r in range(1, 5)
            if getattr(main_window, f"plot_widget_{r}", None) is not None
        )
    return ch_count, m_count, max_plots


def _current_ident(main_window, df_all: pd.DataFrame) -> str:
    """The Identification being plotted, falling back to the first in the data."""
    with contextlib.suppress(
        RuntimeError, AttributeError, TypeError, ValueError
    ):
        if main_window.current_ident_index < len(main_window.unique_ids):
            ident = main_window.unique_ids[main_window.current_ident_index]
            if ident is not None:
                return ident
    return str(df_all["Identification"].iloc[0])


def _ensure_selection_dicts(main_window) -> None:
    """Create the per row selection dicts and the feature default flag on first use."""
    if not hasattr(main_window, "_features_defaulted"):
        main_window._features_defaulted = False
    for attr in (
        "selected_feature_by_row",
        "selected_m_by_channel",
        "selected_ch_by_channel",
    ):
        if not hasattr(main_window, attr):
            setattr(main_window, attr, {})


def _clear_all_rows(main_window, max_plots: int) -> None:
    """Blank every plot row, used when there is nothing to draw."""
    for row in range(1, max_plots + 1):
        plot_widget = getattr(main_window, f"plot_widget_{row}", None)
        if plot_widget:
            plot_widget.clear()


def update_plot(main_window) -> None:
    """Update dynamics plots for the current position.

    Uses 3 dropdowns per row (Features, M, CH). The Features list is
    populated dynamically from dataframe columns.
    """
    _ensure_time_mode(main_window)

    df_all = getattr(main_window, "filtered_df", None)
    if df_all is None or len(df_all) == 0:
        return

    ch_count, m_count, max_plots = _plot_counts(main_window)

    ident = _current_ident(main_window, df_all)
    df_subset = df_all[df_all["Identification"] == ident]
    main_window.df_subset = df_subset
    if df_subset.empty:
        _clear_all_rows(main_window, max_plots)
        return

    feature_defs = _discover_features(
        list(df_all.columns),
        getattr(main_window, "_derived_features", {}),
    )
    if not feature_defs:
        _clear_all_rows(main_window, max_plots)
        return

    main_window._feature_defs = feature_defs
    _ensure_selection_dicts(main_window)

    # Shared scales, so every row's axes line up with the others.
    global_xmax = _compute_shared_scales(main_window, df_subset, feature_defs)
    render_ctx = _build_row_render_context(
        main_window, df_subset, ident, global_xmax
    )

    _sync_all_row_combos(
        main_window, max_plots, feature_defs, ch_count, m_count
    )
    _render_all_rows(
        main_window,
        max_plots,
        feature_defs,
        m_count,
        df_subset,
        render_ctx,
        global_xmax,
    )

    # Circular import
    from SECQUOIA.gui.outlier.markers import update_outlier_marker

    update_outlier_marker(main_window)

    # Circular import
    from SECQUOIA.utils.helpers import update_time_marker

    update_time_marker(main_window)

    main_window._refresh_all_row_summaries()
