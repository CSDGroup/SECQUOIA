"""Curation tree item clicks: selecting an identification/track loads its
row into the viewers, lineage tree, and plots.
"""

from __future__ import annotations

import contextlib
import logging
import re

import numpy as np
import pandas as pd

from SECQUOIA.config import NAPARIPARAMETERS
from SECQUOIA.core.segmentation.mask_selection import (
    ensure_current_df_subset,
    set_active_layers_from_header_buttons,
)
from SECQUOIA.gui.lineage_tree.lineage_tree import lineage_tree
from SECQUOIA.utils.helpers import synchronize_viewers_tracking
from SECQUOIA.utils.plotting import update_plot

LOG = logging.getLogger(__name__)

__all__ = ["handle_item_click"]


def _mask_indices_from_df(df: pd.DataFrame) -> list[int]:
    """Return the sorted list of mask indices present as label_id_m<N> columns."""
    out = []
    with contextlib.suppress(RuntimeError, AttributeError, TypeError):
        patt = re.compile(r"^label_id_m(\d+)$")
        for c in df.columns:
            m = patt.match(str(c))
            if m:
                out.append(int(m.group(1)))
    return sorted(set(out))


def _apply_labels_all_masks_to_viewer(
    viewer, row: pd.Series, mask_idxs: list[int]
) -> None:
    """Select and highlight the row's label on each SegmentationN layer.

    A label of 0 or NaN turns `show_selected_label` off for that layer
    instead of selecting anything.
    """
    if viewer is None or not hasattr(viewer, "layers"):
        return
    for m in mask_idxs:
        layer_name = f"Segmentation{m}"
        if layer_name not in viewer.layers:
            continue
        layer = viewer.layers[layer_name]
        lab = row.get(f"label_id_m{m}", 0)
        try:
            lab = 0 if pd.isna(lab) else int(lab)
        except (RuntimeError, AttributeError, TypeError):
            lab = 0
        try:
            if lab > 0:
                layer.selected_label = lab
                layer.show_selected_label = True
            else:
                layer.show_selected_label = False
        except (RuntimeError, AttributeError, TypeError):
            pass


def _center_from_row_for_mask(
    row: pd.Series, m_idx: int
) -> tuple[float, float, bool]:
    """Return (cy, cx, ok) center coordinates for the given mask index."""
    try:
        cx = float(
            row.get(f"XMorphologyM{m_idx}", row.get("XMorphology", np.nan))
        )
        cy = float(
            row.get(f"YMorphologyM{m_idx}", row.get("YMorphology", np.nan))
        )
    except (RuntimeError, AttributeError, TypeError):
        cx, cy = np.nan, np.nan

    eps = 0.0
    ok = (
        np.isfinite(cx)
        and np.isfinite(cy)
        and not (abs(cx) <= eps and abs(cy) <= eps)
    )
    return cy, cx, ok


def _resolve_ident_from_suffix(main_window, suffix_text: str) -> str | None:
    """Resolve a tree item's numeric suffix back to a full Identification."""
    if (
        getattr(main_window, "filtered_df", None) is None
        or main_window.filtered_df.empty
    ):
        return None
    try:
        last_part = (
            main_window.filtered_df["Identification"]
            .astype(str)
            .str.split("-")
            .str[-1]
        )
        matches = (
            main_window.filtered_df.loc[
                last_part == str(suffix_text), "Identification"
            ]
            .dropna()
            .unique()
        )
        return str(matches[0]) if len(matches) else None
    except (RuntimeError, AttributeError, TypeError):
        return None


def _set_current_ident_index(win, ident: str) -> None:
    """Set win.current_ident_index to ident's position in unique_ids.

    Falls back to 0 when ident is not in the list.
    """
    uids = list(getattr(win, "unique_ids", []))
    try:
        win.current_ident_index = int(uids.index(ident))
    except (RuntimeError, AttributeError, TypeError):
        win.current_ident_index = 0


def _checked_mask_for_viewer(main_window, vi: int) -> int:
    """Return the active mask index for viewer 'vi'."""
    try:
        return int(
            getattr(main_window, "active_mask_index_by_viewer", {}).get(vi, 1)
        )
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return 1


def _apply_camera_and_labels(main_window, row: pd.Series, df_ident) -> None:
    """Apply row's segmentation labels and center the camera on both viewers."""
    mask_idxs = _mask_indices_from_df(df_ident)
    _apply_labels_all_masks_to_viewer(
        getattr(main_window, "viewer_1", None), row, mask_idxs
    )
    _apply_labels_all_masks_to_viewer(
        getattr(main_window, "viewer_2", None), row, mask_idxs
    )

    for vi, v in enumerate(
        [
            getattr(main_window, "viewer_1", None),
            getattr(main_window, "viewer_2", None),
        ]
    ):
        if v is None:
            continue
        m_idx_vi = _checked_mask_for_viewer(main_window, vi)
        cy, cx, ok = _center_from_row_for_mask(row, m_idx_vi)
        if ok:
            try:
                v.camera.center = (cy, cx)
                v.camera.zoom = NAPARIPARAMETERS.ZOOMFACTOR
            except (RuntimeError, AttributeError, TypeError):
                pass


def _sync_outlier_index(main_window, ident: str) -> None:
    """Update current_outlier_index if ident is flagged as an outlier."""
    try:
        if "Outlier_detection" in main_window.track_df.columns and np.any(
            main_window.unique_outliers_ids == ident
        ):
            main_window.current_outlier_index = int(
                np.where(main_window.unique_outliers_ids == ident)[0][0]
            )
    except (RuntimeError, AttributeError, TypeError):
        pass


def _finish_selection(
    main_window, row: pd.Series, df_ident, ident: str
) -> None:
    """Shared tail of both handle_item_click branches: apply the selected
    row/identification to the viewers, lineage tree, and row summaries."""
    _apply_camera_and_labels(main_window, row, df_ident)
    ensure_current_df_subset(main_window)
    lineage_tree(main_window)
    main_window._keep_lineage_collapsed_after_update()
    _sync_outlier_index(main_window, ident)
    set_active_layers_from_header_buttons(main_window)
    main_window._refresh_all_row_igt()
    main_window._refresh_all_row_summaries()


def _select_identification(main_window, item) -> None:
    """Handle a click on a top-level (Identification) tree item."""
    selected_ident = _resolve_ident_from_suffix(main_window, item.text(0))
    if not selected_ident:
        return

    main_window.ident = selected_ident
    ensure_current_df_subset(main_window)

    df_ident = main_window.filtered_df[
        main_window.filtered_df["Identification"].astype(str) == selected_ident
    ]
    if df_ident.empty:
        return

    _set_current_ident_index(main_window, selected_ident)

    try:
        main_window.current_time_index = int(df_ident["t"].min())
    except (RuntimeError, AttributeError, TypeError):
        main_window.current_time_index = 0

    main_window.current_TrackNumber_plot = 1

    try:
        main_window.df_subset = df_ident.copy()
    except (RuntimeError, AttributeError, TypeError):
        main_window.df_subset = df_ident

    update_plot(main_window)
    main_window.fit_all_plots()
    synchronize_viewers_tracking(main_window)

    try:
        row = df_ident[
            (df_ident["TrackNumber"] == main_window.current_TrackNumber_plot)
            & (df_ident["t"] == main_window.current_time_index)
        ].iloc[0]
    except (RuntimeError, AttributeError, TypeError):
        row = df_ident.iloc[0]

    _finish_selection(main_window, row, df_ident, selected_ident)


def _select_track(main_window, item) -> None:
    """Handle a click on a child (track) tree item."""
    parent = item.parent()
    unique_id = _resolve_ident_from_suffix(main_window, parent.text(0))
    if not unique_id:
        return

    main_window.ident = unique_id
    ensure_current_df_subset(main_window)
    txt = item.text(0)
    m = re.search(r"(\d+)$", str(txt))
    try:
        track_number = int(m.group(1)) if m else 1
    except (RuntimeError, AttributeError, TypeError):
        track_number = 1

    LOG.debug("Unique ID: %s, TrackNumber: %s", unique_id, track_number)

    df_ident = main_window.filtered_df[
        main_window.filtered_df["Identification"].astype(str) == unique_id
    ]
    if df_ident.empty:
        return

    _set_current_ident_index(main_window, unique_id)
    main_window.current_TrackNumber_plot = track_number

    # Choose first time point of that track
    df_track = df_ident[
        df_ident["TrackNumber"] == main_window.current_TrackNumber_plot
    ]
    if df_track.empty:
        return
    try:
        main_window.current_time_index = int(df_track["t"].min())
    except (RuntimeError, AttributeError, TypeError):
        main_window.current_time_index = 0

    try:
        main_window.df_subset = df_ident.copy()
    except (RuntimeError, AttributeError, TypeError):
        main_window.df_subset = df_ident

    update_plot(main_window)
    synchronize_viewers_tracking(main_window)

    try:
        row = df_track[df_track["t"] == main_window.current_time_index].iloc[0]
    except (RuntimeError, AttributeError, TypeError):
        row = df_track.iloc[0]

    _finish_selection(main_window, row, df_ident, unique_id)


def handle_item_click(main_window, item, column: int = 0) -> None:
    """Handle clicks on tree widget items."""
    if item.parent() is None:
        _select_identification(main_window, item)
    else:
        _select_track(main_window, item)

    main_window._autorange_lineage()
