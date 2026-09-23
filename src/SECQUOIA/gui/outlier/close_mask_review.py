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
from SECQUOIA.core.outlier_detection.recheck import (
    _refresh_close_mask_views,
    _refresh_outlier_views,
)
from SECQUOIA.core.outlier_detection.track_sync import (
    outlier_times,
    update_outlier_detection_in_track_df_fast,
    update_unique_outliers_ids,
)
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

__all__ = [
    "mark_close_mask_checked",
    "mark_outlier_reviewed",
    "review_current_point",
]


def _jump_to_time(main_window, t: int) -> None:
    """Show time point `t` in both viewers and refresh what depends on it."""
    for attr in ("viewer_1", "viewer_2"):
        viewer = getattr(main_window, attr, None)
        if viewer is not None:
            _set_time_on_viewer(viewer, t)

    with contextlib.suppress(RuntimeError, AttributeError, TypeError):
        _on_time_index_changed(main_window, t)


def _all_close_masks_checked(main_window) -> None:
    """Tell the user nothing is left to review and go back to the normal view."""
    clear_close_mask_view(main_window)
    LOG.info("[CloseMasks] All close-mask cases checked.")
    with contextlib.suppress(RuntimeError, AttributeError, TypeError):
        _show_info_dialog(
            main_window, "Close masks", "All close-mask cases checked"
        )


def _all_outliers_reviewed(main_window) -> None:
    """Tell the user every outlier has been reviewed."""
    LOG.info("[Outliers] All outliers reviewed.")
    with contextlib.suppress(RuntimeError, AttributeError, TypeError):
        _show_info_dialog(main_window, "Outliers", "All outliers reviewed")


def _advance_to_next_review(
    main_window,
    *,
    ident: str,
    old_ids: list,
    times_for_ident,
    ids_attr: str,
    on_finished,
) -> None:
    """Jump to `ident`'s next still-to-review time point, else its next
    identification, else call `on_finished`."""
    df = getattr(main_window, "filtered_df", None)
    current_t = int(getattr(main_window, "current_time_index", 0))
    next_t = next_flagged_time(times_for_ident(df, ident), current_t)
    if next_t is not None:
        select_ident_in_tree(main_window)
        _jump_to_time(main_window, next_t)
        return

    next_ident = next_flagged_ident(
        old_ids, ident, getattr(main_window, ids_attr)
    )
    if next_ident is not None and go_to_ident(main_window, next_ident):
        first_t = int(times_for_ident(df, next_ident)[0])
        _jump_to_time(main_window, first_t)
        return

    on_finished()


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

    _advance_to_next_review(
        main_window,
        ident=ident,
        old_ids=old_ids,
        times_for_ident=flagged_times,
        ids_attr="unique_close_mask_ids",
        on_finished=lambda: _all_close_masks_checked(main_window),
    )


def mark_outlier_reviewed(main_window) -> None:
    """Mark the outlier time point on screen as reviewed and go to the next one."""
    df = getattr(main_window, "filtered_df", None)
    if df is None or len(df) == 0 or "Outlier_detection" not in df.columns:
        LOG.warning("[Outliers] No outliers yet: run outlier detection first.")
        return

    row = current_selection_row(main_window)
    if row is None:
        return

    if row.get("Outlier_detection") != "Outlier":
        LOG.info("[Outliers] The current time point is not an outlier.")
        return

    ident = str(row["Identification"])
    update_unique_outliers_ids(main_window)
    old_ids = list(main_window.unique_outliers_ids)

    df.loc[[row.name], "Outlier_detection"] = "Reviewed"
    update_outlier_detection_in_track_df_fast(main_window, [row.name])
    _refresh_outlier_views(main_window)

    _advance_to_next_review(
        main_window,
        ident=ident,
        old_ids=old_ids,
        times_for_ident=outlier_times,
        ids_attr="unique_outliers_ids",
        on_finished=lambda: _all_outliers_reviewed(main_window),
    )


def review_current_point(main_window) -> None:
    """The "C" hotkey: review whatever is flagged at the current time point."""
    row = current_selection_row(main_window)
    if row is not None and (
        row.get(CLOSE_MASK_FLAG) == FLAG_FLAGGED
        or is_pinned_close_mask_row(main_window, row)
    ):
        mark_close_mask_checked(main_window)
        return

    if row is not None and row.get("Outlier_detection") == "Outlier":
        mark_outlier_reviewed(main_window)
        return

    LOG.info("[Outliers] The current time point is not flagged.")
