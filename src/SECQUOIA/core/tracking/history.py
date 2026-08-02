"""Undo/redo for tracking edits, and the shared write back/refresh routine."""

from __future__ import annotations

import contextlib
import logging
from collections import deque

import pandas as pd

from SECQUOIA.core.tracking.context import _get_position_number
from SECQUOIA.gui.curation_tree import update_list
from SECQUOIA.gui.lineage_tree.lineage_tree import lineage_tree
from SECQUOIA.utils.helpers import ensure_current_df_subset
from SECQUOIA.utils.plotting import update_plot
from SECQUOIA.utils.timing import apply_realtime, build_realtime_lookup

LOG = logging.getLogger(__name__)

_TRACKING_HISTORY_LIMIT = 20


def _refill_realtime(main_window, df: pd.DataFrame | None):
    """Re-apply the imported real-time columns to ``df``."""
    with contextlib.suppress(AttributeError, KeyError, ValueError, TypeError):
        wide = build_realtime_lookup(main_window)
        if wide is not None:
            return apply_realtime(df, wide)
    return df


def _tracking_history_stacks(main_window) -> tuple[deque, deque]:
    """Lazily attach and return (undo_stack, redo_stack) for tracking edits."""
    if getattr(main_window, "_tracking_undo_stack", None) is None:
        main_window._tracking_undo_stack = deque(
            maxlen=_TRACKING_HISTORY_LIMIT
        )
    if getattr(main_window, "_tracking_redo_stack", None) is None:
        main_window._tracking_redo_stack = deque(
            maxlen=_TRACKING_HISTORY_LIMIT
        )
    return main_window._tracking_undo_stack, main_window._tracking_redo_stack


def _snapshot_for_undo(main_window) -> None:
    """Record the current track_df so a tracking edit can be undone."""
    undo_stack, redo_stack = _tracking_history_stacks(main_window)
    current = getattr(main_window, "track_df", None)
    if not isinstance(current, pd.DataFrame):
        return
    undo_stack.append(current.copy())
    redo_stack.clear()


def on_undo_tracking_clicked(main_window) -> None:
    """Undo the most recent tracking edit (new ID, division, split or fuse)."""
    undo_stack, redo_stack = _tracking_history_stacks(main_window)
    if not undo_stack:
        LOG.info("[Tracking] Nothing to undo.")
        return
    current = getattr(main_window, "track_df", None)
    prev_df = undo_stack.pop()
    if isinstance(current, pd.DataFrame):
        redo_stack.append(current.copy())
    _write_back_and_refresh(main_window, prev_df, log_msg="[Tracking] Undo")


def on_redo_tracking_clicked(main_window) -> None:
    """Redo the most recently undone tracking edit."""
    undo_stack, redo_stack = _tracking_history_stacks(main_window)
    if not redo_stack:
        LOG.info("[Tracking] Nothing to redo.")
        return
    current = getattr(main_window, "track_df", None)
    next_df = redo_stack.pop()
    if isinstance(current, pd.DataFrame):
        undo_stack.append(current.copy())
    _write_back_and_refresh(main_window, next_df, log_msg="[Tracking] Redo")


def _write_back_and_refresh(
    main_window, out_df: pd.DataFrame, log_msg: str = ""
) -> None:
    """Standardized write-back and UI refresh."""
    sort_cols = [
        c
        for c in ("Identification", "t", "TrackNumber")
        if c in out_df.columns
    ]
    if sort_cols:
        out_df = out_df.sort_values(by=sort_cols, kind="stable").reset_index(
            drop=True
        )

    out_df = _refill_realtime(main_window, out_df)

    main_window.track_df = out_df

    if "Position" in out_df.columns:
        pos = _get_position_number(main_window)
        main_window.filtered_df = out_df[out_df["Position"] == int(pos)].copy()
    else:
        main_window.filtered_df = out_df.copy()

    if sort_cols:
        main_window.filtered_df = main_window.filtered_df.sort_values(
            by=sort_cols, kind="stable"
        ).reset_index(drop=True)

    main_window.unique_ids = (
        main_window.filtered_df["Identification"].dropna().unique().tolist()
    )

    with contextlib.suppress(AttributeError, ValueError, KeyError):
        ensure_current_df_subset(main_window)
        update_list(main_window)
        update_plot(main_window)
        lineage_tree(main_window)
        main_window.update_display_track_id()

    if log_msg:
        LOG.info("%s", log_msg)
