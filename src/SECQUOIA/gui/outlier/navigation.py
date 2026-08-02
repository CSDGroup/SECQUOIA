"""Stepping between outliers and between outlier time points."""

from __future__ import annotations

import contextlib
import logging

import numpy as np
from qtpy.QtWidgets import QWidget

from SECQUOIA.core.outlier_detection import resolve_current_ident
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

__all__ = ["change_outlier", "change_to_next_outlier"]


def change_outlier(main_window: QWidget, direction: str) -> None:
    """Change the current outlier index based on the direction `next` or `previous`."""
    if "Outlier_detection" not in main_window.track_df.columns:
        LOG.warning("Please perform outlier detection first")
        return

    assert direction in ["next", "previous"], "Invalid direction"

    if (
        not hasattr(main_window, "unique_outliers_ids")
        or len(main_window.unique_outliers_ids) == 0
    ):
        LOG.warning("No outliers available to navigate.")
        return

    # Move within outliers list
    increment = 1 if direction == "next" else -1
    main_window.current_outlier_index = (
        main_window.current_outlier_index + increment
    ) % len(main_window.unique_outliers_ids)

    ident = main_window.unique_outliers_ids[main_window.current_outlier_index]

    # Normalize types to strings to avoid int/str mismatches
    ids = np.asarray(main_window.unique_ids)
    ids_str = ids.astype(str)
    ident_str = str(ident)

    matches = np.flatnonzero(ids_str == ident_str)
    if matches.size == 0:
        LOG.warning("Outlier ident %r not found in unique_ids", ident_str)
        return

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


def change_to_next_outlier(main_window, direction: int = +1) -> None:
    """Jump to next (direction=+1) / previous (direction=-1) time point `t` where Outlier_detection == 'Outlier' for the current Identification."""
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

    try:
        mask = (df["Identification"].astype(str) == ident) & (
            df["Outlier_detection"].astype(str).str.lower() == "outlier"
        )
        times = df.loc[mask, "t"].to_numpy(dtype=int)
    except (KeyError, ValueError, TypeError):
        return

    if times.size == 0:
        return

    times = np.unique(times)

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
