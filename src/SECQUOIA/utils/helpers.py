"""This module contains utility functions for time navigation, cell navigation,
curation-tree updates, plot markers, viewer synchronization, and highlighting.
"""

from __future__ import annotations

import contextlib
import logging

import numpy as np
import pandas as pd
import pyqtgraph as pg
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget,
)

from SECQUOIA.core.segmentation.mask_selection import (
    ensure_current_df_subset,
    set_active_layers_from_header_buttons,
)
from SECQUOIA.gui.curation_tree import update_list
from SECQUOIA.gui.lineage_tree.lineage_data import _time_mapper_for_ident
from SECQUOIA.gui.lineage_tree.lineage_tree import lineage_tree
from SECQUOIA.utils.plotting import (
    TIME_MODE_T,
    update_plot,
    x_column_for,
)

LOG = logging.getLogger(__name__)


def extract_unique_tracknumbers(main_window) -> dict:
    """Map each Identification to its sorted unique TrackNumbers, based on
    main_window.filtered_df."""
    data = {}
    try:
        df = getattr(main_window, "filtered_df", None)
        if ("Identification" not in df.columns) or (
            "TrackNumber" not in df.columns
        ):
            return data

        for unique_id, sub in df.groupby("Identification", dropna=False):
            ser = sub["TrackNumber"].dropna()
            try:
                vals = ser.astype(int, errors="ignore").unique().tolist()
            except TypeError:
                try:
                    vals = (
                        pd.to_numeric(ser, errors="coerce")
                        .dropna()
                        .astype(int)
                        .unique()
                        .tolist()
                    )
                except (TypeError, ValueError):
                    vals = ser.unique().tolist()

            with contextlib.suppress(TypeError, ValueError):
                vals = sorted(vals)

            data[unique_id] = vals

        return data
    except (ImportError, TypeError, ValueError, AttributeError):
        return {}


def _is_image_layer(layer) -> bool:
    """True for a napari Image layer, and False for a Labels layer."""
    t = getattr(layer, "_type_string", "").lower()
    if t == "image":
        return True
    return hasattr(layer, "data") and not hasattr(layer, "selected_label")


def _infer_n_frames_from_viewers(mw) -> int:
    """Prefer real viewer layers to infer T; fall back to stored stacks."""
    for viewer_attr in ("viewer_1", "viewer_2"):
        v = getattr(mw, viewer_attr, None)
        if v is None:
            continue
        for layer in getattr(v, "layers", []):
            if not _is_image_layer(layer):
                continue
            data = getattr(layer, "data", None)
            if data is None:
                continue
            try:
                shape = data.shape
            except (RuntimeError, AttributeError, TypeError):
                shape = np.asarray(data).shape
            if len(shape) >= 3:
                return int(shape[0])
            if len(shape) == 2:
                return 1

    for ch in range(mw.n_channels):
        arr = getattr(mw, "images", None)
        arr = arr[ch] if arr is not None else None
        if arr is None:
            continue
        shape = np.asarray(arr).shape
        if len(shape) >= 3:
            return int(shape[0])
        if len(shape) == 2:
            return 1
    return 0


def _set_time_on_viewer(viewer, t: int) -> None:
    """Set a viewer to the requested time index."""
    try:
        viewer.dims.set_current_step(0, int(t))
    except (RuntimeError, AttributeError, TypeError):
        try:
            steps = list(viewer.dims.current_step)
            steps[0] = int(t)
            viewer.dims.current_step = tuple(steps)
        except (RuntimeError, AttributeError, TypeError):
            pass


def _set_time_on_all_viewers(main_window, t: int) -> None:
    """Set both viewers to the requested time index."""
    for attr in ("viewer_1", "viewer_2"):
        v = getattr(main_window, attr, None)
        if v is not None:
            _set_time_on_viewer(v, t)


def change_time_point(main_window, increment: int) -> None:
    """Move the current time point forward or backward, wrapping at the ends."""
    if not hasattr(main_window, "folder_list") or not main_window.folder_list:
        LOG.warning("Please first load CSV file and select folder")
        from SECQUOIA.gui.common.messages import show_folder_warning

        show_folder_warning(main_window)
        return

    n_frames = _infer_n_frames_from_viewers(main_window)
    if n_frames <= 1:
        new_t = 0
    else:
        new_t = (
            int(getattr(main_window, "current_time_index", 0)) + int(increment)
        ) % n_frames

    _set_time_on_all_viewers(main_window, new_t)

    _on_time_index_changed(main_window, new_t)


def _on_time_index_changed(main_window, t: int) -> None:
    """Refresh dependent GUI state after the current time index changes."""
    if getattr(main_window, "_in_time_change_cb", False):
        return
    main_window._in_time_change_cb = True
    try:
        main_window.current_time_index = int(t)

        df = getattr(main_window, "df_subset", None)
        if (
            not isinstance(df, pd.DataFrame)
            or df.empty
            or "t" not in df.columns
            or "TrackNumber" not in df.columns
        ):
            return

        _update_current_track_number_plot(main_window)
        # circular-import
        from SECQUOIA.gui.outlier.markers import update_outlier_marker

        update_outlier_marker(main_window)
        update_time_marker(main_window)
        main_window.zoom_in()
        set_active_layers_from_header_buttons(main_window)
        main_window._refresh_all_row_igt()
        main_window._refresh_all_row_summaries()
    finally:
        main_window._in_time_change_cb = False


def set_jump_channel(main_window, channel: str) -> None:
    """Select the channel used for jump-channel navigation."""
    if (
        not hasattr(main_window, "image_present")
        or channel not in main_window.image_present
    ):
        LOG.warning("[jump-channel] channel not available: %s", channel)
        return
    main_window.jump_channel = channel
    LOG.debug("[jump-channel] selected: %s", channel)


def change_time_point_in_jump_channel(main_window, increment: int) -> None:
    """Jump to the next or previous valid time point in the selected channel."""
    channel = getattr(main_window, "jump_channel", None)
    if not channel:
        LOG.warning("[jump-channel] no jump channel selected")
        return

    present_map = getattr(main_window, "image_present", None)
    if not isinstance(present_map, dict) or channel not in present_map:
        LOG.warning("[jump-channel] no presence map for channel: %s", channel)
        return

    present = np.asarray(present_map[channel], dtype=bool)
    if present.size == 0 or not present.any():
        LOG.warning("[jump-channel] no images present in channel: %s", channel)
        return

    current_t = int(getattr(main_window, "current_time_index", 0))
    valid_t = np.flatnonzero(present)

    if increment > 0:
        candidates = valid_t[valid_t > current_t]
        new_t = int(candidates[0]) if candidates.size else int(valid_t[0])
    else:
        candidates = valid_t[valid_t < current_t]
        new_t = int(candidates[-1]) if candidates.size else int(valid_t[-1])

    _set_time_on_all_viewers(main_window, new_t)

    _on_time_index_changed(main_window, new_t)


def _compute_unique_ids(main_window) -> list:
    """Recompute the sorted by appearance list of unique Identifications."""
    return (
        main_window.filtered_df["Identification"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


def select_ident_in_tree(main_window) -> None:
    """Move the blue tree selection to the current Identification and scroll it into view, without re-triggering the item-click handler."""
    tree = getattr(main_window, "tree_widget", None)
    if tree is None:
        return

    ident_text = str(getattr(main_window, "ident", "") or "").split("-")[-1]
    if not ident_text:
        return

    items = tree.findItems(ident_text, Qt.MatchExactly | Qt.MatchRecursive)
    if not items:
        return

    item = items[0]
    tree.blockSignals(True)
    tree.setCurrentItem(item)
    item.setSelected(True)
    tree.blockSignals(False)
    tree.scrollToItem(item)


def change_cell(main_window: QWidget, direction: str) -> None:
    """Change the current cell based on the direction."""
    if not hasattr(main_window, "folder_list") or not main_window.folder_list:
        LOG.warning("Please first load CSV file and select folder")
        from SECQUOIA.gui.common.messages import show_folder_warning

        show_folder_warning(main_window)
        return

    if (
        main_window.filtered_df is None
        or "Identification" not in main_window.filtered_df.columns
    ):
        return

    if direction not in ("next", "previous"):
        raise ValueError(f"invalid direction: {direction!r}")

    main_window.unique_ids = _compute_unique_ids(main_window)

    if not main_window.unique_ids:
        return

    current_ident = str(getattr(main_window, "ident", "") or "")

    try:
        current_idx = next(
            i
            for i, v in enumerate(main_window.unique_ids)
            if str(v) == current_ident
        )
    except StopIteration:
        current_idx = int(getattr(main_window, "current_ident_index", 0))

    if direction == "next":
        new_idx = min(current_idx + 1, len(main_window.unique_ids) - 1)
    else:
        new_idx = max(current_idx - 1, 0)

    if new_idx == current_idx:
        return

    main_window.current_ident_index = new_idx
    main_window.ident = str(main_window.unique_ids[new_idx])

    ensure_current_df_subset(main_window)
    main_window.current_time_index = 0
    main_window.current_TrackNumber_plot = 1
    synchronize_viewers_tracking(main_window)
    update_plot(main_window)
    main_window.zoom_in()
    lineage_tree(main_window)
    main_window._keep_lineage_collapsed_after_update()
    main_window.fit_all_plots()
    set_active_layers_from_header_buttons(main_window)
    main_window._refresh_all_row_igt()
    main_window._refresh_all_row_summaries()

    select_ident_in_tree(main_window)
    main_window._autorange_lineage()


def jump_to_identification(
    main_window,
    ident: str,
    t: int | None = None,
    tracknumber: int | None = None,
) -> None:
    """Select `ident` and refresh UI. If `t` is None, jump to the first time point."""
    if not ident or not hasattr(main_window, "filtered_df"):
        LOG.warning("[Jump] Missing ident or filtered_df.")
        return
    if "Identification" not in main_window.filtered_df.columns:
        LOG.warning("[Jump] 'Identification' column missing.")
        return

    ids = getattr(main_window, "unique_ids", None)
    if not ids:
        main_window.unique_ids = _compute_unique_ids(main_window)
        ids = main_window.unique_ids

    target_ident = str(ident)
    try:
        idx = next(i for i, v in enumerate(ids) if str(v) == target_ident)
    except StopIteration:
        main_window.unique_ids = _compute_unique_ids(main_window)
        try:
            idx = next(
                i
                for i, v in enumerate(main_window.unique_ids)
                if str(v) == target_ident
            )
        except StopIteration:
            LOG.warning("[Jump] Identification '%s' not found.", target_ident)
            return

    main_window.current_ident_index = int(idx)
    main_window.ident = target_ident

    if t is None:
        sub = main_window.filtered_df[
            main_window.filtered_df["Identification"].astype(str)
            == target_ident
        ]
        t_min = (
            int(sub["t"].min())
            if (
                "t" in sub.columns
                and not sub.empty
                and pd.notna(sub["t"].min())
            )
            else 0
        )
        main_window.current_time_index = t_min
    else:
        main_window.current_time_index = int(t)

    main_window.current_TrackNumber_plot = int(tracknumber or 1)

    with contextlib.suppress(
        RuntimeError, AttributeError, TypeError, ValueError
    ):
        ensure_current_df_subset(main_window)
        update_list(main_window)
        update_plot(main_window)
        main_window.zoom_in()
        lineage_tree(main_window)
        main_window._keep_lineage_collapsed_after_update()
        main_window.fit_all_plots()
        set_active_layers_from_header_buttons(main_window)
        main_window._refresh_all_row_igt()
        main_window._refresh_all_row_summaries()

    LOG.debug(
        "[Jump] -> %r, t=%s, TN=%s",
        target_ident,
        main_window.current_time_index,
        main_window.current_TrackNumber_plot,
    )


def synchronize_viewers_tracking(main_window) -> None:
    """Synchronize viewers in main window during manual tracking."""
    idx = int(getattr(main_window, "current_time_index", 0))
    _set_time_on_all_viewers(main_window, idx)


def _update_current_track_number_plot(main_window) -> None:
    """Update the active track number for the current time point."""
    df = getattr(main_window, "df_subset", None)
    if (
        not isinstance(df, pd.DataFrame)
        or df.empty
        or "t" not in df.columns
        or "TrackNumber" not in df.columns
    ):
        main_window.current_TrackNumber_plot = None
        return

    t = int(getattr(main_window, "current_time_index", 0))
    subset_at_time = df[df["t"] == t]
    if subset_at_time.empty:
        main_window.current_TrackNumber_plot = None
        return

    cur = getattr(main_window, "current_TrackNumber_plot", None)
    if pd.notna(cur) and (subset_at_time["TrackNumber"] == cur).any():
        return

    candidates = pd.to_numeric(
        subset_at_time["TrackNumber"], errors="coerce"
    ).dropna()
    if candidates.empty:
        main_window.current_TrackNumber_plot = None
        return

    main_window.current_TrackNumber_plot = int(candidates.iloc[0])


def _clear_time_markers_fallback(main_window) -> None:
    """Remove all per row and lineage time markers (used when the marker is toggled off or before a redraw)."""
    if hasattr(main_window, "_clear_time_markers"):
        main_window._clear_time_markers()
        return

    _clear_row_markers(main_window)

    old_line = getattr(main_window, "_lineage_time_line", None)
    if isinstance(old_line, pg.InfiniteLine):
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            for attr in ("graph3_plot", "graph3_widget"):
                pw = getattr(main_window, attr, None)
                if isinstance(pw, pg.PlotWidget):
                    pw.getPlotItem().removeItem(old_line)
                    break
    main_window._lineage_time_line = None


def _clear_row_markers(main_window) -> None:
    """Remove existing per row time marker items (does not touch the lineage marker)."""
    if hasattr(main_window, "current_time_markers"):
        for marker in list(main_window.current_time_markers.values()):
            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                pw = marker.get("plot_widget", None)
                item = marker.get("item", None)
                if isinstance(pw, pg.PlotWidget) and item is not None:
                    pw.removeItem(item)
    main_window.current_time_markers = {}


def _current_ident_df(main_window):
    """Resolve the currently selected Identification and its row subset, or None if unavailable."""
    df_filt = getattr(main_window, "filtered_df", None)
    if (
        not hasattr(main_window, "unique_ids")
        or len(getattr(main_window, "unique_ids", [])) == 0
    ):
        return None
    try:
        if 0 <= main_window.current_ident_index < len(main_window.unique_ids):
            ident = main_window.unique_ids[main_window.current_ident_index]
        else:
            ident = df_filt["Identification"].iloc[0]
    except (RuntimeError, AttributeError, TypeError):
        ident = df_filt["Identification"].iloc[0]

    df_id = df_filt[df_filt["Identification"] == ident]
    if df_id.empty:
        return None
    return ident, df_id


def _x_position_for_row(
    main_window, df_id, row, current_time, mode, feature_defs
):
    """Compute the x-position at which to draw the time marker for one plot row."""
    feat_key = getattr(main_window, "selected_feature_by_row", {}).get(row)
    has_ch = bool(feature_defs.get(feat_key, {}).get("has_ch", False))
    ch_idx = getattr(main_window, "selected_ch_by_channel", {}).get(row, 1)
    xcol = x_column_for(
        mode,
        has_ch=has_ch,
        ch_idx=int(ch_idx or 1),
        df_cols=list(df_id.columns),
    )

    row_at_t = df_id[df_id["t"] == current_time]
    if not row_at_t.empty and xcol in row_at_t.columns:
        xv = pd.to_numeric(row_at_t[xcol], errors="coerce").dropna()
        if not xv.empty:
            return float(xv.iloc[0])

    tt = pd.to_numeric(df_id["t"], errors="coerce").to_numpy(dtype=float)
    xx = pd.to_numeric(df_id.get(xcol), errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(tt) & np.isfinite(xx)
    if m.any():
        tt, xx = tt[m], xx[m]
        k = int(np.nanargmin(np.abs(tt - float(current_time))))
        return float(xx[k])

    return float(current_time)


def _draw_row_marker(main_window, pw, row, x_line) -> None:
    """Draw (or replace) the vertical time-marker line on a single plot row."""
    try:
        line = pg.InfiniteLine(
            pos=float(x_line),
            angle=90,
            movable=False,
            pen=pg.mkPen("g", width=2),
        )
        line.setZValue(100)
        if line.scene() is None:
            pw.addItem(line)
        main_window.current_time_markers[row] = {
            "item": line,
            "plot_widget": pw,
        }
    except (RuntimeError, AttributeError, TypeError):
        pass


def _find_lineage_plot_widget(main_window):
    """Locate the lineage tree's PlotWidget, trying the known attributes and falling back to a child search."""
    lineage_pw = getattr(main_window, "graph3_plot", None)

    if not isinstance(lineage_pw, pg.PlotWidget):
        lineage_pw = getattr(main_window, "graph3_widget", None)

    if not isinstance(lineage_pw, pg.PlotWidget):
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            container = getattr(main_window, "graph3_widget", None)
            if container is not None:
                kids = container.findChildren(pg.PlotWidget)
                if kids:
                    lineage_pw = kids[0]

    return lineage_pw if isinstance(lineage_pw, pg.PlotWidget) else None


def _draw_lineage_marker(main_window, df_id, current_time) -> None:
    """Draw (or replace) the vertical time-marker line on the lineage tree plot."""
    lineage_pw = _find_lineage_plot_widget(main_window)
    if lineage_pw is None:
        return

    try:
        plot_item = lineage_pw.getPlotItem()

        old_line = getattr(main_window, "_lineage_time_line", None)
        if isinstance(old_line, pg.InfiniteLine):
            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                plot_item.removeItem(old_line)
        main_window._lineage_time_line = None

        try:
            xmap, _, _, _ = _time_mapper_for_ident(df_id, main_window)
            xpos = float(xmap(float(current_time)))
        except (RuntimeError, AttributeError, TypeError, ValueError):
            xpos = float(current_time)

        new_line = pg.InfiniteLine(
            pos=xpos, angle=90, movable=False, pen=pg.mkPen("g", width=2)
        )
        new_line.setZValue(100)
        if new_line.scene() is None:
            plot_item.addItem(new_line)
        main_window._lineage_time_line = new_line

    except (RuntimeError, AttributeError, TypeError):
        pass


def update_time_marker(main_window) -> None:
    """Draw a green vertical line at the current time on ALL plot rows,
    independent of which metric (C/M) is shown. Also update the lineage
    tree.
    """
    if not getattr(main_window, "show_time_marker", True):
        _clear_time_markers_fallback(main_window)
        return

    resolved = _current_ident_df(main_window)
    if resolved is None:
        return
    _ident, df_id = resolved

    current_time = int(main_window.current_time_index)
    _clear_row_markers(main_window)

    max_plots = getattr(main_window, "_max_plot_rows", None)
    if not isinstance(max_plots, int) or max_plots < 1:
        max_plots = sum(
            1
            for r in range(1, 5)
            if getattr(main_window, f"plot_widget_{r}", None) is not None
        )

    mode = getattr(main_window, "_time_mode", TIME_MODE_T)
    feature_defs = getattr(main_window, "_feature_defs", {}) or {}

    for row in range(1, max_plots + 1):
        pw = getattr(main_window, f"plot_widget_{row}", None)
        if not isinstance(pw, pg.PlotWidget):
            continue

        x_line = _x_position_for_row(
            main_window, df_id, row, current_time, mode, feature_defs
        )
        _draw_row_marker(main_window, pw, row, x_line)

    _draw_lineage_marker(main_window, df_id, current_time)


def toggle_highlight_mode(main_window) -> None:
    """Toggle coloring mode."""
    try:
        btn = getattr(main_window, "lineage_tools", {}).get("HL")
        if hasattr(btn, "toggle") and callable(btn.toggle):
            btn.toggle()
            return
    except (RuntimeError, AttributeError, TypeError, ValueError):
        pass

    try:
        main_window._hl_active = not bool(
            getattr(main_window, "_hl_active", False)
        )
    except (RuntimeError, AttributeError, TypeError, ValueError):
        main_window._hl_active = True

    lineage_tree(main_window)
    fn = getattr(main_window, "_hl_sync_button_style", None)
    if callable(fn):
        with contextlib.suppress(
            RuntimeError, AttributeError, TypeError, ValueError
        ):
            fn()


def cycle_highlight_color(main_window, forward: bool = True) -> None:
    """Cycle the active highlight color (used by Ctrl+Shift+P)."""
    try:
        order = getattr(main_window, "_hl_palette_order", None)
        if not order:
            pal = getattr(main_window, "_hl_palette", {}) or {}
            order = list(pal.values())
            main_window._hl_palette_order = order
        if not order:
            return

        idx = int(getattr(main_window, "_hl_palette_index", 0))
        idx = (idx + (1 if forward else -1)) % len(order)
        main_window._hl_palette_index = idx
        main_window._hl_active_color = order[idx]

        fn = getattr(main_window, "_hl_sync_button_style", None)
        if callable(fn):
            fn()
    except (RuntimeError, AttributeError, TypeError, ValueError):
        pass


def clear_highlight_paints(main_window) -> None:
    """Remove ALL painted lineage/plot colors and refresh the UI."""
    try:
        cmap = getattr(main_window, "_track_highlight_colors", None)
        if isinstance(cmap, dict):
            cmap.clear()
    except (RuntimeError, AttributeError, TypeError, ValueError):
        pass

    lineage_tree(main_window)
    update_plot(main_window)
    for fn_name in ("_refresh_all_row_igt", "_refresh_all_row_summaries"):
        fn = getattr(main_window, fn_name, None)
        if callable(fn):
            with contextlib.suppress(
                RuntimeError, AttributeError, TypeError, ValueError
            ):
                fn()
