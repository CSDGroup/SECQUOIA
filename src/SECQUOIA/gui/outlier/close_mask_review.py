"""Reviewing close-mask cases: mark one as checked and move on to the next."""

from __future__ import annotations

import contextlib
import logging

from SECQUOIA.core.outlier_detection.close_masks import (
    CLOSE_MASK_FLAG,
    FLAG_FLAGGED,
    FLAG_REVIEWED,
    flagged_times,
    is_pinned_close_mask_row,
    next_flagged_ident,
    next_flagged_time,
    set_close_mask_flag,
    unpin_close_mask_row,
    update_unique_close_mask_ids,
)
from SECQUOIA.core.outlier_detection.recheck import _refresh_close_mask_views
from SECQUOIA.core.segmentation.close_mask_view import clear_close_mask_view
from SECQUOIA.core.segmentation.mask_selection import current_selection_row
from SECQUOIA.gui.outlier.navigation import go_to_ident
from SECQUOIA.gui.outlier.outlier_list import _show_info_dialog
from SECQUOIA.utils.helpers import (
    _on_time_index_changed,
    _set_time_on_viewer,
    select_ident_in_tree,
)

LOG = logging.getLogger(__name__)

__all__ = ["mark_close_mask_checked"]


def _jump_to_time(main_window, t: int) -> None:
    """Show time point `t` in both viewers and refresh what depends on it."""
    for attr in ("viewer_1", "viewer_2"):
        viewer = getattr(main_window, attr, None)
        if viewer is not None:
            _set_time_on_viewer(viewer, t)

    with contextlib.suppress(RuntimeError, AttributeError, TypeError):
        _on_time_index_changed(main_window, t)


def _all_checked(main_window) -> None:
    """Tell the user nothing is left to review and go back to the normal view."""
    clear_close_mask_view(main_window)
    LOG.info("[CloseMasks] All close-mask cases checked.")
    with contextlib.suppress(RuntimeError, AttributeError, TypeError):
        _show_info_dialog(
            main_window, "Close masks", "All close-mask cases checked"
        )


def mark_close_mask_checked(main_window) -> None:
    """Mark the flagged time point on screen as checked and go to the next case.

    1. The row becomes ``Reviewed``, which removes its purple star and its
       napari highlight. The flag is saved with the CSV.
    2. The view jumps to the next flagged time point of the same
       identification, wrapping around.
    3. With none left there, the identification drops out of the list and the
       view jumps to the next flagged identification.
    4. With nothing left at all, the normal view is restored.

    A row that is not flagged is left alone. A row that an edit just marked
    reviewed counts as checked, so the key moves on from it.
    """
    df = getattr(main_window, "filtered_df", None)
    if df is None or len(df) == 0 or CLOSE_MASK_FLAG not in df.columns:
        LOG.warning(
            "[CloseMasks] No flags yet: press Find on the Close masks tab of "
            "the Outlier Detection window first."
        )
        return

    row = current_selection_row(main_window)
    if row is None:
        return

    flagged = row.get(CLOSE_MASK_FLAG) == FLAG_FLAGGED
    if not flagged and not is_pinned_close_mask_row(main_window, row):
        LOG.info("[CloseMasks] The current time point is not flagged.")
        return

    ident = str(row["Identification"])
    update_unique_close_mask_ids(main_window)
    old_ids = list(main_window.unique_close_mask_ids)

    if flagged:
        set_close_mask_flag(main_window, [row.name], FLAG_REVIEWED)
    unpin_close_mask_row(main_window)
    _refresh_close_mask_views(main_window)

    current_t = int(getattr(main_window, "current_time_index", row["t"]))
    next_t = next_flagged_time(flagged_times(df, ident), current_t)
    if next_t is not None:
        select_ident_in_tree(main_window)
        _jump_to_time(main_window, next_t)
        return

    next_ident = next_flagged_ident(
        old_ids, ident, main_window.unique_close_mask_ids
    )
    if next_ident is not None and go_to_ident(main_window, next_ident):
        first_t = int(flagged_times(df, next_ident)[0])
        _jump_to_time(main_window, first_t)
        return

    _all_checked(main_window)
