"""napari layer state for segmentation masks.

Label selection driven by the current dataframe row, camera centring, and the
per viewer map from mask index to Labels layer name.
"""

import contextlib
import logging
import re
import weakref

import numpy as np
import pandas as pd
from napari.layers import Labels

from SECQUOIA.config import NAPARIPARAMETERS

LOG = logging.getLogger(__name__)


def mask_index_from_layer_name(name) -> int | None:
    """Extract the mask index from a segmentation layer name."""
    text = str(name or "")

    match = re.search(
        r"seg(?:mentation)?[_ ]*(\d+)", text, flags=re.IGNORECASE
    )
    if match:
        return int(match.group(1))

    match = re.search(r"(\d+)\s*$", text)
    return int(match.group(1)) if match else None


def get_mask_indices(main_window) -> list[int]:
    """Return sorted mask indices."""
    try:
        mc = getattr(main_window, "n_masks", 0)
        if mc > 0:
            return list(range(1, mc + 1))
    except (RuntimeError, AttributeError, TypeError):
        pass

    try:
        df = getattr(main_window, "filtered_df", None)
        if df is not None and hasattr(df, "columns"):
            idxs = []
            for c in df.columns:
                m = re.match(r"^label_id_m(\d+)$", str(c))
                if m:
                    with contextlib.suppress(
                        RuntimeError, AttributeError, TypeError
                    ):
                        idxs.append(int(m.group(1)))
            idxs = sorted(set(idxs))
            if idxs:
                return idxs
    except (RuntimeError, AttributeError, TypeError):
        pass

    return [1]


def _resolve_zoom_center(
    row: pd.Series, mask_idxs: list[int]
) -> tuple[float, float]:
    """Return (cy, cx) from row['YMorphology','XMorphology'],
    falling back to the first available XMorphologyM*/YMorphologyM* pair.
    """
    try:
        cy = float(row.get("YMorphology", np.nan))
        cx = float(row.get("XMorphology", np.nan))
        if not np.isfinite(cy) or not np.isfinite(cx):
            cy, cx = np.nan, np.nan
            for m in mask_idxs:
                x_col, y_col = f"XMorphologyM{m}", f"YMorphologyM{m}"
                if x_col in row and y_col in row:
                    try:
                        cand_x = float(row[x_col])
                        cand_y = float(row[y_col])
                        if np.isfinite(cand_x) and np.isfinite(cand_y):
                            cx, cy = cand_x, cand_y
                            break
                    except (
                        RuntimeError,
                        AttributeError,
                        TypeError,
                        ValueError,
                    ):
                        pass
    except (RuntimeError, AttributeError, TypeError, ValueError):
        cy, cx = np.nan, np.nan

    return cy, cx


def _is_valid_center(cy: float, cx: float, eps: float = 0.0) -> bool:
    """Return True if the center point is finite and not sitting at the origin."""
    return (
        np.isfinite(cy)
        and np.isfinite(cx)
        and not (abs(cy) <= eps and abs(cx) <= eps)
    )


def _label_value_from_row(row: pd.Series, mask_idx: int) -> int:
    """Read label_id_m{mask_idx} from row, defaulting to 0 if missing/invalid."""
    lab_val = 0
    lab_col = f"label_id_m{mask_idx}"
    if lab_col in row:
        try:
            v = row[lab_col]
            if pd.notna(v):
                lab_val = int(v)
        except (RuntimeError, AttributeError, TypeError):
            lab_val = 0
    return lab_val


def _resolve_t_index(main_window, row: pd.Series) -> int:
    """Resolve the time index from row['t'], falling back to current_time_index."""
    try:
        return int(row.get("t", getattr(main_window, "current_time_index", 0)))
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return int(getattr(main_window, "current_time_index", 0))


# Avoids scanning the label array on every revisit.
_SLICE_MAX_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_CACHE_INVALIDATION_WIRED: "weakref.WeakSet" = weakref.WeakSet()


def _invalidate_layer_cache(layer) -> None:
    """Drop all cached slice maxima for `layer` (called when it's painted)."""
    _SLICE_MAX_CACHE.pop(layer, None)


def _ensure_cache_invalidation_wired(layer) -> None:
    """Clear this layer's cache on paint, so a correction can never leave a stale max behind."""
    try:
        if layer in _CACHE_INVALIDATION_WIRED:
            return
    except TypeError:
        return

    paint_event = getattr(getattr(layer, "events", None), "paint", None)
    if paint_event is None:
        return
    try:
        paint_event.connect(
            lambda _event, _layer=layer: _invalidate_layer_cache(_layer)
        )
        _CACHE_INVALIDATION_WIRED.add(layer)
    except (RuntimeError, AttributeError, TypeError):
        pass


def _next_label_for_layer(layer, t_idx: int) -> int:
    """Return the next free label id for `layer`: max label in the slice + 1."""
    _ensure_cache_invalidation_wired(layer)

    try:
        cache = _SLICE_MAX_CACHE.setdefault(layer, {})
    except TypeError:
        cache = None

    data_id = id(getattr(layer, "data", None))
    if cache is not None:
        cached = cache.get(t_idx)
        if cached is not None and cached[0] == data_id:
            return cached[1]

    try:
        data = np.asarray(layer.data)
    except (RuntimeError, AttributeError, TypeError, ValueError):
        data = None

    max_lbl = 0
    if data is not None and data.size:
        try:
            if data.ndim == 3:
                if 0 <= t_idx < data.shape[0]:
                    slice_ = data[t_idx]
                else:
                    slice_ = data.max(axis=0)
                max_lbl = int(np.nanmax(slice_)) if slice_.size else 0
            elif data.ndim == 2:
                max_lbl = int(np.nanmax(data)) if data.size else 0
            else:
                max_lbl = int(np.nanmax(data)) if data.size else 0
        except (
            RuntimeError,
            AttributeError,
            TypeError,
            ValueError,
        ):
            max_lbl = 0

    new_lbl = int(max_lbl + 1)
    if cache is not None:
        cache[t_idx] = (data_id, new_lbl)
    return new_lbl


def _apply_mask_selection_to_layer(
    main_window,
    row: pd.Series,
    layer,
    layer_name: str,
    lab_val: int,
    created_label_for_layer: dict[str, int],
) -> None:
    """Select `lab_val` on `layer`, or create+select a fresh label if there isn't one."""
    try:
        if lab_val > 0:
            layer.show_selected_label = True
            layer.selected_label = lab_val
        else:
            layer.show_selected_label = False

            if layer_name in created_label_for_layer:
                with contextlib.suppress(
                    RuntimeError, AttributeError, TypeError, ValueError
                ):
                    layer.selected_label = int(
                        created_label_for_layer[layer_name]
                    )

            t_idx = _resolve_t_index(main_window, row)
            new_lbl = _next_label_for_layer(layer, t_idx)

            try:
                layer.selected_label = new_lbl
                created_label_for_layer[layer_name] = new_lbl
            except (
                RuntimeError,
                AttributeError,
                TypeError,
                ValueError,
            ):
                pass
    except (RuntimeError, AttributeError, TypeError):
        pass


def _center_viewer_camera(
    viewer, cy: float, cx: float, zoom_level: int, is_new_ident: bool
) -> None:
    """Center the viewer's camera on (cy, cx), resetting zoom only for a new identification."""
    try:
        viewer.camera.center = (cy, cx)
        if is_new_ident:
            viewer.camera.zoom = zoom_level
    except (RuntimeError, AttributeError, TypeError):
        pass


def apply_all_mask_selections(
    main_window,
    row: pd.Series,
    *,
    center_camera: bool = True,
    zoom_level: int = NAPARIPARAMETERS.ZOOMFACTOR,
) -> None:
    """Center camera once (per viewer) using row['YMorphology','XMorphology']
    or the first available XMorphologyM*/YMorphologyM* pair.
    """
    mask_idxs = get_mask_indices(main_window)

    try:
        current_ident = str(row.get("Identification", ""))
    except (RuntimeError, AttributeError, TypeError, ValueError):
        current_ident = ""

    cy, cx = _resolve_zoom_center(row, mask_idxs)
    is_center_valid = _is_valid_center(cy, cx)
    is_new_ident = current_ident != getattr(
        main_window, "_last_zoom_ident", None
    )

    for viewer_attr in ("viewer_1", "viewer_2"):
        viewer = getattr(main_window, viewer_attr, None)
        if viewer is None or not hasattr(viewer, "layers"):
            continue

        created_label_for_layer: dict[str, int] = {}

        for m in mask_idxs:
            layer_name = f"Segmentation{m}"
            if layer_name not in viewer.layers:
                continue

            lab_val = _label_value_from_row(row, m)
            layer = viewer.layers[layer_name]
            _apply_mask_selection_to_layer(
                main_window,
                row,
                layer,
                layer_name,
                lab_val,
                created_label_for_layer,
            )

        if center_camera and is_center_valid:
            _center_viewer_camera(viewer, cy, cx, zoom_level, is_new_ident)

    main_window._last_zoom_ident = current_ident
    _refresh_close_mask_view(main_window, row)


def _refresh_close_mask_view(main_window, row: pd.Series) -> None:
    """Show, or clear, the close-mask pair for `row`; never let it break navigation."""
    from SECQUOIA.core.segmentation.close_mask_view import (
        refresh_close_mask_view,
    )

    try:
        refresh_close_mask_view(main_window, row)
    except (
        RuntimeError,
        AttributeError,
        TypeError,
        ValueError,
        KeyError,
    ) as e:
        LOG.warning("[close-mask view] refresh failed: %s", e)


def current_selection_row(main_window) -> pd.Series | None:
    """The dataframe row the UI is currently showing, or None."""
    df_all = getattr(main_window, "filtered_df", None)
    if df_all is None or df_all.empty:
        return None

    try:
        if 0 <= main_window.current_ident_index < len(main_window.unique_ids):
            ident = main_window.unique_ids[main_window.current_ident_index]
        else:
            ident = df_all["Identification"].iloc[0]
    except (RuntimeError, AttributeError, TypeError):
        ident = df_all["Identification"].iloc[0]

    df_id = df_all[df_all["Identification"] == ident]
    if df_id.empty:
        return None

    try:
        tr = main_window.current_TrackNumber_plot
    except (RuntimeError, AttributeError, TypeError):
        tr = None
    try:
        tt = main_window.current_time_index
    except (RuntimeError, AttributeError, TypeError):
        tt = None

    try:
        if tr is not None and tt is not None:
            sub = df_id[(df_id["TrackNumber"] == tr) & (df_id["t"] == tt)]
            if not sub.empty:
                return sub.iloc[0]
    except (RuntimeError, AttributeError, TypeError):
        pass

    try:
        return df_id.iloc[0]
    except (RuntimeError, AttributeError, TypeError):
        return None


def refresh_all_mask_selections_at_current(main_window) -> None:
    """Reapply the mask selections for the row the UI is currently showing."""
    row = current_selection_row(main_window)
    if row is None:
        return

    apply_all_mask_selections(
        main_window,
        row,
        center_camera=True,
        zoom_level=NAPARIPARAMETERS.ZOOMFACTOR,
    )


def ensure_current_df_subset(main_window) -> bool:
    """Ensure main_window.df_subset exists and matches the current Identification."""
    df_all = getattr(main_window, "filtered_df", None)
    if df_all is None or df_all.empty:
        main_window.df_subset = None
        return False

    try:
        if 0 <= main_window.current_ident_index < len(main_window.unique_ids):
            ident = main_window.unique_ids[main_window.current_ident_index]
        else:
            ident = df_all["Identification"].iloc[0]
    except (RuntimeError, AttributeError, TypeError):
        ident = df_all["Identification"].iloc[0]

    sub = df_all[df_all["Identification"] == ident]
    main_window.df_subset = sub
    return not sub.empty


def rebuild_segmentation_layer_index(main_window) -> None:
    """Build main_window.seg_layers_by_viewer = {viewer_idx: {mask_idx: layer_name}}."""
    viewers = [
        getattr(main_window, "viewer_1", None),
        getattr(main_window, "viewer_2", None),
    ]
    main_window.seg_layers_by_viewer = {}

    for vi, v in enumerate(viewers):
        mapping = {}
        if v is None or not hasattr(v, "layers"):
            main_window.seg_layers_by_viewer[vi] = mapping
            continue

        labels_names = []
        for nm in list(v.layers):
            try:
                if isinstance(v.layers[nm], Labels):
                    labels_names.append(nm)
            except (RuntimeError, AttributeError, TypeError):
                pass

        for nm in labels_names:
            idx = mask_index_from_layer_name(nm)
            if idx is not None:
                mapping[idx] = nm

        next_idx = 1
        for nm in labels_names:
            while next_idx in mapping:
                next_idx += 1
            if nm in mapping.values():
                continue
            mapping[next_idx] = nm
            next_idx += 1

        main_window.seg_layers_by_viewer[vi] = mapping


def set_active_layers_from_header_buttons(main_window) -> None:
    """For each viewer, restore its active layer from the last remembered mask index.

    Reads `main_window.active_mask_index_by_viewer` for that viewer
    (defaulting to 1), then sets its active layer to the mapped
    segmentation layer.
    """
    if not hasattr(main_window, "seg_layers_by_viewer"):
        rebuild_segmentation_layer_index(main_window)

    if not hasattr(main_window, "active_mask_index_by_viewer"):
        main_window.active_mask_index_by_viewer = {}

    viewers = [
        getattr(main_window, "viewer_1", None),
        getattr(main_window, "viewer_2", None),
    ]

    for vi, v in enumerate(viewers):
        if v is None or not hasattr(v, "layers"):
            continue

        try:
            m_sel = int(main_window.active_mask_index_by_viewer.get(vi, 1))
        except (RuntimeError, AttributeError, TypeError, ValueError):
            m_sel = 1
        if m_sel < 1:
            m_sel = 1

        # Remember per viewer
        main_window.active_mask_index_by_viewer[vi] = m_sel

        # Use the mapping to find the exact layer name
        name = None
        try:
            name = main_window.seg_layers_by_viewer.get(vi, {}).get(
                m_sel, None
            )
        except (RuntimeError, AttributeError, TypeError):
            name = None

        if (not name) and hasattr(v, "layers"):
            rebuild_segmentation_layer_index(main_window)
            try:
                name = main_window.seg_layers_by_viewer.get(vi, {}).get(
                    m_sel, None
                )
            except (RuntimeError, AttributeError, TypeError):
                name = None

        try:
            if name and name in v.layers:
                v.layers.selection.active = v.layers[name]
        except (RuntimeError, AttributeError, TypeError):
            pass


def show_all_masks(main_window) -> None:
    """Disable 'show_selected_label' on every Segmentation{m} layer in both viewers."""
    try:
        mask_idxs = get_mask_indices(main_window)
    except (RuntimeError, AttributeError, TypeError):
        return

    for viewer_attr in ("viewer_1", "viewer_2"):
        viewer = getattr(main_window, viewer_attr, None)
        if viewer is None or not hasattr(viewer, "layers"):
            continue

        for m in mask_idxs:
            layer_name = f"Segmentation{m}"
            if layer_name not in viewer.layers:
                continue
            layer = viewer.layers[layer_name]
            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                layer.show_selected_label = False

    from SECQUOIA.core.segmentation.close_mask_view import (
        clear_close_mask_view,
    )

    clear_close_mask_view(main_window)


def show_current_mask(main_window) -> None:
    """Show only the currently selected track's label in each mask layer.

    The inverse of `show_all_masks`: for every Segmentation{m} layer in both
    viewers, select the label this row carries for that mask and switch
    isolation on. A mask with no label for this row keeps showing everything.
    """
    try:
        mask_idxs = get_mask_indices(main_window)
    except (RuntimeError, AttributeError, TypeError):
        return

    df = getattr(main_window, "filtered_df", None)
    if df is None or df.empty:
        return

    try:
        ident = main_window._current_zoom_identification(df)
        df_ident = df[df["Identification"] == ident]
        row_df = main_window._filtered_zoom_subset(df_ident)
        if row_df.empty:
            row_df = df_ident
        if row_df.empty:
            return
        row = row_df.iloc[0]
    except (RuntimeError, AttributeError, TypeError, KeyError, IndexError):
        return

    for viewer_attr in ("viewer_1", "viewer_2"):
        viewer = getattr(main_window, viewer_attr, None)
        if viewer is None or not hasattr(viewer, "layers"):
            continue

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
            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                if lab > 0:
                    layer.selected_label = lab
                    layer.show_selected_label = True
                else:
                    layer.show_selected_label = False

    _refresh_close_mask_view(main_window, row)
