"""Stepping between outliers and between outlier time points."""

from __future__ import annotations

import contextlib
import logging

import numpy as np
from qtpy.QtWidgets import QWidget

from SECQUOIA.core.outlier_detection import resolve_current_ident
from SECQUOIA.core.outlier_detection.close_masks import (
    CLOSE_MASK_FLAG,
    flagged_times,
    unique_review_ids,
    update_unique_close_mask_ids,
)
from SECQUOIA.core.segmentation.mask_selection import (
    ensure_current_df_subset,
    set_active_layers_from_header_buttons,
)
from SECQUOIA.gui.lineage_tree.lineage_tree import lineage_tree
from SECQUOIA.utils.helpers import (
    _on_time_index_changed,
    _set_time_on_viewer,
    select_ident_in_tree,
)
from SECQUOIA.utils.plotting import update_plot

LOG = logging.getLogger(__name__)

__all__ = ["change_outlier", "change_to_next_outlier", "go_to_ident"]


def change_outlier(main_window: QWidget, direction: str) -> None:
    """Step to the next or previous ident of the Out list and refresh the whole view.

    The list holds the outlier identifications followed by the ones that only
    have a flagged close mask. ''direction'' is "next" or "previous"; the
    index wraps around at both ends.
    """
    columns = main_window.track_df.columns
    if "Outlier_detection" not in columns and CLOSE_MASK_FLAG not in columns:
        LOG.warning("Please perform outlier detection first")
        return

    assert direction in ["next", "previous"], "Invalid direction"

    update_unique_close_mask_ids(main_window)
    idents = unique_review_ids(main_window)
    if len(idents) == 0:
        LOG.warning("No outliers available to navigate.")
        return

    # Move within the list
    increment = 1 if direction == "next" else -1
    main_window.current_outlier_index = (
        main_window.current_outlier_index + increment
    ) % len(idents)

    go_to_ident(main_window, idents[main_window.current_outlier_index])


def go_to_ident(main_window: QWidget, ident) -> bool:
    """Show `ident` from its first time point and refresh the whole view.

    Returns False, leaving the view alone, when `ident` is not a known
    identification.
    """
    # Normalize types to strings to avoid int/str mismatches
    ids = np.asarray(main_window.unique_ids)
    ids_str = ids.astype(str)
    ident_str = str(ident)

    matches = np.flatnonzero(ids_str == ident_str)
    if matches.size == 0:
        LOG.warning("Ident %r not found in unique_ids", ident_str)
        return False

    main_window.current_ident_index = int(matches[0])
    main_window.ident = ident_str
    ensure_current_df_subset(main_window)
    main_window.current_time_index = 0
    main_window.current_TrackNumber_plot = 1
    update_plot(main_window)
    main_window.zoom_in()
    lineage_tree(main_window)
    main_window._keep_lineage_collapsed_after_update()
    main_window.fit_all_plots()
    set_active_layers_from_header_buttons(main_window)
    main_window._refresh_all_row_igt()
    main_window._refresh_all_row_summaries()
    select_ident_in_tree(main_window)
    return True


def _review_times(df, ident) -> np.ndarray:
    """Sorted time points at which `ident` has an outlier or a flagged close mask."""
    outlier_times = np.array([], dtype=int)
    if "Outlier_detection" in df.columns:
        mask = (df["Identification"].astype(str) == ident) & (
            df["Outlier_detection"].astype(str).str.lower() == "outlier"
        )
        with contextlib.suppress(ValueError, TypeError):
            outlier_times = df.loc[mask, "t"].to_numpy(dtype=int)
    return np.unique(np.concatenate([outlier_times, flagged_times(df, ident)]))


def change_to_next_outlier(main_window, direction: int = +1) -> None:
    """Jump to the next or previous outlier or close-mask time point of the current ident.

    `direction` is +1 for forwards and -1 for backwards. The search wraps
    around to the first (or last) such time point when none is left in that
    direction.
    """
    df = getattr(main_window, "filtered_df", None)
    if df is None or df.empty:
        return

    ident = resolve_current_ident(main_window, df)
    if ident is None:
        return

    try:
        current_t = int(getattr(main_window, "current_time_index", 0))
    except (TypeError, ValueError):
        current_t = 0

    times = _review_times(df, ident)
    if times.size == 0:
        return

    if direction >= 0:
        later = times[times > current_t]
        new_t = int(later[0] if later.size else times[0])
    else:
        earlier = times[times < current_t]
        new_t = int(earlier[-1] if earlier.size else times[-1])

    # Update both viewers
    for attr in ("viewer_1", "viewer_2"):
        v = getattr(main_window, attr, None)
        if v is not None:
            _set_time_on_viewer(v, new_t)

    with contextlib.suppress(RuntimeError, AttributeError, TypeError):
        _on_time_index_changed(main_window, new_t)
