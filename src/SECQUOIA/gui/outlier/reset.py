"""Clearing outlier rules, sliding-window UI state, flags (outliers and close masks), and markers."""

from __future__ import annotations

import contextlib
import logging

import pandas as pd

from SECQUOIA.core.outlier_detection import _normalize_unique_ids
from SECQUOIA.core.outlier_detection.close_masks import reset_close_mask_state
from SECQUOIA.core.segmentation.close_mask_view import clear_close_mask_view
from SECQUOIA.core.segmentation.mask_selection import ensure_current_df_subset
from SECQUOIA.gui.curation_tree import update_list
from SECQUOIA.gui.outlier.markers import _clear_close_mask_markers
from SECQUOIA.utils.plotting import update_plot

LOG = logging.getLogger(__name__)

__all__ = ["clear_close_mask_flags", "reset_outlier_state"]


def _clear_saved_outlier_rules(main_window) -> None:
    """Delete the saved outlier rules attribute, if any."""
    if hasattr(main_window, "_last_outlier_rules"):
        try:
            delattr(main_window, "_last_outlier_rules")
        except (RuntimeError, AttributeError, TypeError):
            main_window._last_outlier_rules = {"rules": []}


def _teardown_sliding_window_rows(main_window) -> None:
    """Destroy and clear the sliding-window list UI row widgets."""
    rows = getattr(main_window, "_sliding_rows_widgets", None)
    if not rows:
        return
    try:
        while rows:
            r = rows.pop(0)
            for key in (
                "feat",
                "mask",
                "ch",
                "tmin",
                "tmax",
                "factor",
                "btn_bar",
                "btn_add",
                "btn_rm",
            ):
                w = r.get(key)
                if w is not None:
                    with contextlib.suppress(Exception):
                        w.setParent(None)
                        w.deleteLater()
    except (RuntimeError, AttributeError, TypeError):
        pass
    main_window._sliding_rows_widgets = []


def _reset_sliding_window_form(main_window) -> None:
    """Reset the single sliding-window UI form fields to their defaults."""
    swp = getattr(main_window, "sliding_window_params", None)
    if not (isinstance(swp, dict) and swp):
        return
    with contextlib.suppress(Exception):
        if "feature_combo" in swp:
            swp["feature_combo"].setCurrentIndex(0)
        if "mask_combo" in swp and swp["mask_combo"].isEnabled():
            swp["mask_combo"].setCurrentIndex(0)
        if "channel_combo" in swp and swp["channel_combo"].isEnabled():
            swp["channel_combo"].setCurrentIndex(0)
        if "t_min_spin" in swp:
            swp["t_min_spin"].setValue(0.0)
        if "t_max_spin" in swp:
            swp["t_max_spin"].setValue(0.0)
        if "factor_spin" in swp:
            swp["factor_spin"].setValue(0.0)


def _uncheck_outliers_toggle(main_window) -> None:
    """Uncheck the Outliers view toggle button, if present."""
    with contextlib.suppress(Exception):
        if hasattr(main_window, "Outliers"):
            main_window.Outliers.setChecked(False)


def _reset_outlier_id_bookkeeping(main_window) -> None:
    """Clear the outlier id list and clamp the current ident index into range."""
    main_window.unique_outliers_ids = []
    main_window.current_outlier_index = 0

    ids = _normalize_unique_ids(main_window)
    if not ids:
        main_window.current_ident_index = 0
        return

    try:
        cur = int(getattr(main_window, "current_ident_index", 0) or 0)
    except (TypeError, ValueError):
        cur = 0
    main_window.current_ident_index = max(0, min(cur, len(ids) - 1))
    main_window.ident = str(ids[main_window.current_ident_index])


def _reset_outcol(
    df: pd.DataFrame, outcol: str, drop_columns: bool
) -> pd.DataFrame:
    """Reset `outcol` to 'OK' in `df`, or drop it entirely if `drop_columns`."""
    if outcol in df.columns:
        if drop_columns:
            with contextlib.suppress(Exception):
                df.drop(columns=[outcol], inplace=True)
        else:
            if df[outcol].dtype != object:
                with contextlib.suppress(Exception):
                    df[outcol] = df[outcol].astype(object)
            df[outcol] = "OK"
    elif not drop_columns:
        df[outcol] = "OK"
    return df


def _clear_outlier_markers(main_window) -> None:
    """Remove outlier star markers from all plots."""
    if not hasattr(main_window, "_outlier_marker_items"):
        return
    for rec in list(main_window._outlier_marker_items.values()):
        if not rec:
            continue
        pw = rec.get("plot_widget")
        item = rec.get("item")
        with contextlib.suppress(Exception):
            if pw is not None and item is not None:
                pw.removeItem(item)
    main_window._outlier_marker_items = {}


def clear_close_mask_flags(main_window) -> None:
    """Remove the close-mask flags, purple stars and napari highlight.

    Also forgets the close-mask settings kept with the saved outlier rules, so
    a later position switch does not detect them again.
    """
    reset_close_mask_state(main_window)
    _clear_close_mask_markers(main_window)
    with contextlib.suppress(RuntimeError, AttributeError, TypeError):
        clear_close_mask_view(main_window)

    rules = getattr(main_window, "_last_outlier_rules", None)
    if isinstance(rules, dict):
        rules.pop("close_masks", None)


def _refresh_views_after_reset(main_window, refresh_plots: bool) -> None:
    """Refresh the derived subset, list, and plot after an outlier-state reset."""
    with contextlib.suppress(Exception):
        ensure_current_df_subset(main_window)
    with contextlib.suppress(Exception):
        all_cb = getattr(main_window, "ALL", None)
        if all_cb is not None:
            all_cb.setChecked(True)
        update_list(main_window)

    if refresh_plots:
        with contextlib.suppress(Exception):
            update_plot(main_window)


def reset_outlier_state(
    main_window,
    *,
    outcol: str = "Outlier_detection",
    drop_columns: bool = False,
    refresh_plots: bool = True,
    show_message: bool = True,
) -> None:
    """Clear every trace of a previous outlier or close-mask run, in the UI and in the data."""
    _clear_saved_outlier_rules(main_window)
    _teardown_sliding_window_rows(main_window)
    _reset_sliding_window_form(main_window)
    _uncheck_outliers_toggle(main_window)
    _reset_outlier_id_bookkeeping(main_window)

    for attr in ("filtered_df", "track_df"):
        df = getattr(main_window, attr, None)
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            setattr(main_window, attr, _reset_outcol(df, outcol, drop_columns))

    _clear_outlier_markers(main_window)
    clear_close_mask_flags(main_window)
    _refresh_views_after_reset(main_window, refresh_plots)

    if show_message:
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            LOG.info(
                "[Reset] Cleared outlier rules, sliding-window settings, flags, markers, and refreshed views."
            )
