"""Track selection and highlight state for the lineage tree."""

from __future__ import annotations

import contextlib
import logging

import numpy as np
import pandas as pd
from qtpy import QtGui

from SECQUOIA.gui.lineage_tree.lineage_render import QT_DRAW_ERRORS
from SECQUOIA.utils.plotting import update_plot

LOG = logging.getLogger(__name__)


def _require_loaded_project(main_window) -> bool:
    """Warn and return False when no experiment folder has been loaded yet."""
    if getattr(main_window, "folder_list", None):
        return True
    LOG.warning("Please first load CSV file and select folder")
    from SECQUOIA.gui.common.messages import show_folder_warning

    show_folder_warning(main_window)
    return False


def _refresh_dependent_views(main_window) -> None:
    """Rebuild the lineage tree and the row plots after a selection change."""
    from SECQUOIA.gui.lineage_tree.lineage_tree import lineage_tree

    try:
        lineage_tree(main_window)
    except (*QT_DRAW_ERRORS, ValueError) as exc:
        LOG.warning("lineage tree rebuild failed: %s", exc)

    try:
        update_plot(main_window)
        main_window._refresh_all_row_igt()
        main_window._refresh_all_row_summaries()
    except (*QT_DRAW_ERRORS, ValueError) as exc:
        LOG.warning("row plot refresh failed: %s", exc)


def _toggle_track(main_window, tn: int) -> None:
    """Toggle a TrackNumber in the selection set, then refresh UI + plots."""
    selected = getattr(main_window, "_selected_tracks", None)
    if not isinstance(selected, set):
        selected = main_window._selected_tracks = set()
    selected.symmetric_difference_update({tn})
    _refresh_dependent_views(main_window)


def _assign_highlight_color(main_window, tn: int) -> None:
    """Toggle a track's highlight color in the highlight-color map, then refresh."""
    painted = getattr(main_window, "_track_highlight_colors", None)
    if not isinstance(painted, dict):
        painted = main_window._track_highlight_colors = {}

    try:
        track = int(tn)
    except (TypeError, ValueError) as exc:
        LOG.warning("cannot highlight track %r: %s", tn, exc)
        return

    if track in painted:
        del painted[track]
    else:
        color = getattr(main_window, "_hl_active_color", QtGui.QColor("lime"))
        painted[track] = QtGui.QColor(color)

    _refresh_dependent_views(main_window)


def _clear_track_selection(main_window) -> None:
    """Clear selected tracks and refresh lineage and row plots."""
    if not _require_loaded_project(main_window):
        return

    main_window._selected_tracks = set()
    _refresh_dependent_views(main_window)


def _select_all_tracks(main_window) -> None:
    """Select all tracks for the current identification and refresh views."""
    if not _require_loaded_project(main_window):
        return

    df = next(
        (
            d
            for a in ("filtered_df", "df_subset", "df")
            if isinstance((d := getattr(main_window, a, None)), pd.DataFrame)
            and not d.empty
        ),
        None,
    )
    if df is None:
        return

    ids = set(df["Identification"].astype(str).unique())
    ident = getattr(main_window, "ident", None)
    if not ident or ident not in ids:
        uids = getattr(main_window, "unique_ids", None)
        idx = getattr(main_window, "current_ident_index", None)
        ident = (
            str(uids[idx])
            if isinstance(uids, (list | np.ndarray))
            and isinstance(idx, int)
            and 0 <= idx < len(uids)
            else str(df["Identification"].iloc[0])
        )
    main_window.ident = ident

    sub = df[df["Identification"].astype(str) == ident]
    main_window._selected_tracks = set(
        pd.to_numeric(sub["TrackNumber"], errors="coerce").dropna().astype(int)
    )

    _refresh_dependent_views(main_window)


def _sync_viewers_to_time(main_window) -> None:
    """Push ``current_time_index`` out to the viewers and the row plots."""
    try:
        from SECQUOIA.utils.helpers import (
            change_time_point,
            synchronize_viewers_tracking,
            update_time_marker,
        )
    except ImportError:
        return

    try:
        change_time_point(main_window, 0)
    except (AttributeError, TypeError):
        with contextlib.suppress(ImportError, AttributeError, TypeError):
            from SECQUOIA.gui.outlier.markers import update_outlier_marker

            update_plot(main_window)
            synchronize_viewers_tracking(main_window)
            update_outlier_marker(main_window)
            update_time_marker(main_window)
            main_window.zoom_in()
            main_window._refresh_all_row_igt()
            main_window._refresh_all_row_summaries()
