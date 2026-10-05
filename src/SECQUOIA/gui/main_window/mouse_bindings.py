"""Mouse triggered interaction for MainWindow: napari canvas drag callbacks
(paint/erase/pick) and dynamics plot clicks.
"""

import contextlib
import logging
from types import MethodType

import napari
import napari.layers.labels._labels_mouse_bindings as _lb
import numpy as np
import pandas as pd
from napari.layers import Labels
from napari.layers.labels._labels_mouse_bindings import (
    draw as _NAPARI_LABELS_DRAW,
)
from napari.layers.labels.labels import Labels as NapariLabels
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication
from skimage.draw import polygon2mask

from SECQUOIA.core.segmentation.mask_selection import (
    _next_label_for_layer,
    apply_all_mask_selections,
    current_selection_row,
    ensure_current_df_subset,
    refresh_all_mask_selections_at_current,
)
from SECQUOIA.gui.cell_inspector.integration import notify_cell_inspector
from SECQUOIA.utils.helpers import change_time_point
from SECQUOIA.utils.plotting import (
    TIME_MODE_T,
    _discover_features,
    x_column_for,
)

LOG = logging.getLogger(__name__)


class MouseBindings:
    """Mouse drag callbacks registered on the napari viewers' canvases."""

    def _restrict_labels_draw_to_left_click(
        self, labels_layer: Labels | None = None
    ) -> None:
        """Make labels drawing respond to left-click only."""

        def _draw_left_only(layer, event):
            """Block right and middle mouse drawing in labels layers."""
            btn = getattr(event, "button", None)
            if btn in (2, 3, "right", "Right", "middle", "Middle"):
                yield
                while event.type == "mouse_move":
                    yield
                return

            yield from _NAPARI_LABELS_DRAW(layer, event)

        drag_modes = getattr(NapariLabels, "_drag_modes", None)
        if isinstance(drag_modes, dict):
            for mode_key, func in list(drag_modes.items()):
                mode_name = getattr(mode_key, "name", str(mode_key))
                if (
                    mode_name in ("PAINT", "FILL", "ERASE")
                    and func is _NAPARI_LABELS_DRAW
                ):
                    drag_modes[mode_key] = _draw_left_only

        if getattr(_lb, "draw", None) is _NAPARI_LABELS_DRAW:
            _lb.draw = _draw_left_only

    def ctrl_right_click_viewer(self, viewer, event):
        """On Ctrl+Right-click, report the picked cell.

        Picks the segmentation label under the cursor, resolves it to a
        row of ``self.filtered_df`` at the current time point, and shows
        the row's Identification/TrackNumber in the viewer status bar
        (forwarding the identification to the fuse dialog if it is open).
        """
        if not self._is_ctrl_right_click(event):
            return

        pick = self._pick_segmentation_at_cursor(viewer, event)
        if pick:
            self._report_pick_identification(viewer, pick)

        yield
        while event.type == "mouse_move":
            yield

    def _is_ctrl_right_click(self, event) -> bool:
        """True if `event` is a Ctrl + Right mouse button press."""
        return (
            event.type == "mouse_press"
            and getattr(event, "button", None) == 2
            and bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
        )

    def _is_right_click_suppressed(self, event) -> bool:
        """True if this event must not be handled as a segmentation pick.

        Ctrl+right-click is handled separately by ``ctrl_right_click_viewer``,
        and the callback fires for the whole drag, so only the initial press
        counts.
        """
        if QApplication.keyboardModifiers() & Qt.ControlModifier:
            return True
        if event.type != "mouse_press":
            return True
        return getattr(event, "button", None) != 2

    def right_mouse_click_viewer(self, viewer, event):
        """Right-click a viewer to set the morphology anchor of the current track.

        Also triggers measurement recomputation for the affected mask(s) and
        refreshes the lineage tree to reflect the updated anchor.
        """
        if self._is_right_click_suppressed(event):
            return

        v_name = self._viewer_display_name(viewer)
        pick = self._pick_segmentation_at_cursor(
            viewer, event, include_background=True
        )

        picked_layer = None
        if pick is None:
            LOG.debug(
                "[right-click] viewer=%s (no Segmentation layer under cursor)",
                v_name,
            )
        else:
            LOG.debug(
                "[right-click] viewer=%s layer=%s T=%s -> mask=%s at (y=%s, x=%s)",
                v_name,
                pick["tag"],
                pick["t"],
                pick["label"],
                pick["YMorphology"],
                pick["XMorphology"],
            )
            if self._write_morphology_anchor(pick):
                picked_layer = pick["layer"]

        self._recompute_measurements_after_pick(viewer, picked_layer)
        self._refresh_lineage_after_edit()

        yield
        while event.type == "mouse_move":
            yield

    def erase(self, layer, event):
        """Handle labels erase edits and refresh measurements."""
        if getattr(layer, "mode", None) != "erase":
            return
        self._edit_begin(layer)
        yield
        while event.type == "mouse_move":
            yield
        self._finish_labels_edit(layer, "erase")

    def paint(self, layer, event):
        """Handle labels paint edits and refresh measurements."""
        if getattr(layer, "mode", None) != "paint":
            return
        self._edit_begin(layer)
        yield
        while event.type == "mouse_move":
            yield
        self._finish_labels_edit(layer, "paint")

    def _viewer_of_layer(self, layer):
        """Return the napari viewer that holds `layer`, or None."""
        for v in vars(self).values():
            for cand in v if isinstance(v, (list | tuple)) else (v,):
                if isinstance(cand, napari.Viewer) and layer in cand.layers:
                    return cand
        return None

    def _lasso_line_for(self, viewer):
        """Return the viewer's reusable preview line (a vispy visual, not a layer)."""
        lines = self.__dict__.setdefault("_lasso_lines", {})
        line = lines.get(id(viewer))
        if line is None:
            from vispy.scene.visuals import Line

            scene = viewer.window._qt_viewer.canvas.view.scene
            line = Line(
                np.zeros((2, 2), np.float32),
                color="yellow",
                width=2,
                parent=scene,
            )
            line.order = 10_000
            line.set_gl_state("translucent", depth_test=False)
            line.visible = False
            lines[id(viewer)] = line
        return line

    def lasso(self, layer, event):
        """Replace whatever is under a freehand closed shape with one new mask in the same layer (Shift erases)."""
        if not getattr(layer, "_lasso_on", False) or event.button != 1:
            return
        viewer = self._viewer_of_layer(layer)
        if viewer is None:
            return
        self._edit_begin(layer)
        line = self._lasso_line_for(viewer)
        disp = list(event.dims_displayed)
        erase = "Shift" in event.modifiers
        pts = [np.asarray(event.position, float)]
        min_step = 2.0 / max(float(viewer.camera.zoom), 1e-6)
        pan = viewer.camera.mouse_pan
        viewer.camera.mouse_pan = False
        try:
            yield
            while event.type == "mouse_move":
                p = np.asarray(event.position, float)
                if np.hypot(*(p[disp] - pts[-1][disp])) >= min_step:
                    pts.append(p)
                    xy = np.array([q[disp][::-1] for q in pts], np.float32)
                    line.set_data(xy)
                    line.visible = True
                yield
        finally:
            viewer.camera.mouse_pan = pan
            line.visible = False
        if len(pts) < 3:
            self._discard_edit_cache_entry(layer)
            return
        data_pts = np.array([layer.world_to_data(q) for q in pts])
        new_label = None
        if erase:
            layer.paint_polygon(data_pts, 0)
        else:
            new_label = self._lasso_replace(layer, data_pts)
        self._finish_labels_edit(layer, "lasso")
        self._lasso_refresh_views()
        if new_label is not None:
            layer.selected_label = new_label

    def _lasso_refresh_views(self) -> None:
        """Reselect the cell's (possibly new) mask in every viewer and redraw the inspector."""
        row = current_selection_row(self)
        if row is not None:
            apply_all_mask_selections(self, row, center_camera=False)
        notify_cell_inspector(self, "_apply_label_edit")

    @staticmethod
    def _lasso_replace(layer, data_pts) -> int:
        """Delete every mask the shape touches, then fill it with a fresh label."""
        t = int(round(data_pts[0][0])) if layer.ndim == 3 else None
        sl = layer.data[t] if t is not None else layer.data
        inside = polygon2mask(sl.shape, data_pts[:, -2:].astype(int))
        old = np.unique(sl[inside])
        old = old[old != 0]
        new_label = _next_label_for_layer(layer, t if t is not None else 0)
        with layer.block_history():
            if old.size:
                yy, xx = np.nonzero(np.isin(sl, old))
                idx = (yy, xx) if t is None else (np.full(yy.size, t), yy, xx)
                layer.data_setitem(idx, 0)
            layer.paint_polygon(data_pts, new_label)
        layer.selected_label = int(new_label)
        return int(new_label)


def add_mouse_drag_to_segmentation_layer(
    main_window, base_name: str = "Segmentation"
) -> None:
    """Attach paint, erase, and optional right-click callbacks to segmentation layers."""

    def _is_seg_label(name: str) -> bool:
        """True for a segmentation layer name: base, base_<key>, or base<n>."""
        if name == base_name:
            return True
        if name.startswith(base_name + "_"):
            return True
        suffix = name[len(base_name) :] if name.startswith(base_name) else ""
        return suffix.isdigit() if suffix else False

    def _iter_viewers():
        """Yield all napari viewers attached to main_window."""
        for attr in ("viewer_1", "viewer_2"):
            v = getattr(main_window, attr, None)
            if isinstance(v, napari.Viewer):
                yield v
        vf = getattr(main_window, "viewer_fluorescence", None)
        if isinstance(vf, (list | tuple)):
            for v in vf:
                if isinstance(v, napari.Viewer):
                    yield v
        for k, v in vars(main_window).items():
            if k.startswith("viewer_") and isinstance(v, napari.Viewer):
                yield v

    def _unique(seq):
        """Yield items in order, skipping ones already seen by identity."""
        seen = set()
        for x in seq:
            xid = id(x)
            if xid not in seen:
                seen.add(xid)
                yield x

    def _ensure_attached_mouse_provider(obj, cb):
        """Append `cb` to obj.mouse_drag_callbacks if not already present."""
        callbacks = getattr(obj, "mouse_drag_callbacks", None)
        if callbacks is None:
            return

        cb_func = getattr(cb, "__func__", cb)
        cb_self = getattr(cb, "__self__", None)

        for c in callbacks:
            c_func = getattr(c, "__func__", c)
            c_self = getattr(c, "__self__", None)
            if c_func is cb_func and c_self is cb_self:
                return

        callbacks.append(cb)

    attached_layers = 0
    viewers = list(_unique(_iter_viewers()))
    if not viewers:
        LOG.warning("No napari viewers found on main_window.")
        return

    has_erase = isinstance(getattr(main_window, "erase", None), MethodType)
    has_paint = isinstance(getattr(main_window, "paint", None), MethodType)
    has_rclick = isinstance(
        getattr(main_window, "right_mouse_click_viewer", None), MethodType
    )

    if not (has_erase and has_paint):
        LOG.warning(
            "main_window.erase/paint not found as bound methods; nothing attached."
        )
        return

    # Attach layer level paint/erase
    for v in viewers:
        for layer in list(v.layers):
            if isinstance(layer, napari.layers.Labels) and _is_seg_label(
                layer.name
            ):
                _ensure_attached_mouse_provider(layer, main_window.erase)
                _ensure_attached_mouse_provider(layer, main_window.paint)
                _ensure_attached_mouse_provider(layer, main_window.lasso)
                attached_layers += 1

    if has_rclick:
        for v in viewers:
            _ensure_attached_mouse_provider(
                v, main_window.right_mouse_click_viewer
            )

    has_ctrl_rclick = isinstance(
        getattr(main_window, "ctrl_right_click_viewer", None), MethodType
    )
    if has_ctrl_rclick:
        for v in viewers:
            _ensure_attached_mouse_provider(
                v, main_window.ctrl_right_click_viewer
            )


def _map_click_to_data_coords(plot_widget, event):
    """Convert a scene click event into (x, y) data coordinates, or None."""
    try:
        mouse_point = plot_widget.plotItem.vb.mapSceneToView(event.scenePos())
        return float(mouse_point.x()), float(mouse_point.y())
    except (RuntimeError, AttributeError, TypeError):
        return None


def _resolve_feature_selection(main_window, row):
    """Resolve the (feat_key, m_idx, ch_idx, definition) selected for a row."""
    tools = getattr(main_window, "row_tools", {}).get(row, {})
    feat_combo = tools.get("feat")
    m_combo = tools.get("m")
    ch_combo = tools.get("ch")

    # Feature key
    feat_key = getattr(main_window, "selected_feature_by_row", {}).get(row)
    if feat_key is None and feat_combo is not None:
        feat_key = (
            feat_combo.currentData()
            if feat_combo.currentData() is not None
            else feat_combo.currentText()
        )
    if not feat_key:
        return None

    # Mask index
    try:
        m_idx = int(
            getattr(main_window, "selected_m_by_channel", {}).get(row, None)
        )
    except (RuntimeError, AttributeError, TypeError):
        m_idx = None
    if (
        (m_idx is None or m_idx < 1)
        and m_combo is not None
        and m_combo.currentData() is not None
    ):
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            m_idx = int(m_combo.currentData())
    if m_idx is None or m_idx < 1:
        m_idx = 1

    feature_defs = getattr(main_window, "_feature_defs", {}) or {}
    if feat_key not in feature_defs:
        df_all = getattr(main_window, "filtered_df", None)
        if df_all is not None:
            feature_defs = _discover_features(
                list(df_all.columns),
                getattr(main_window, "_derived_features", {}),
            )

    definition = feature_defs.get(feat_key)
    if not definition:
        return None
    template = definition.get("template")
    if template is None:
        return None
    has_ch = bool(definition.get("has_ch", False))

    # Channel index
    ch_idx = None
    if has_ch:
        ch_idx = getattr(main_window, "selected_ch_by_channel", {}).get(
            row, None
        )
        if ch_idx is None and ch_combo is not None:
            ch_idx = ch_combo.currentData()

        try:
            ch_idx = f"{int(ch_idx):02d}"
        except (ValueError, TypeError):
            ch_idx = "00"

    return feat_key, m_idx, ch_idx, definition


def _build_intensity_column(definition, ch_idx, m_idx):
    """Build the dataframe column name for a feature definition, or None."""
    template = definition.get("template")
    has_ch = bool(definition.get("has_ch", False))
    try:
        return (
            template.format(ch=ch_idx, m=m_idx)
            if has_ch
            else template.format(m=m_idx)
        )
    except (RuntimeError, AttributeError, TypeError):
        return None


def _infer_time_len(main_window) -> int:
    """Infer the number of time points from the dataframe or image stack."""
    df_all = getattr(main_window, "filtered_df", None)
    if df_all is not None and not df_all.empty:
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            return int(pd.to_numeric(df_all["t"], errors="coerce").max()) + 1

    imgs = getattr(main_window, "images", None)
    if isinstance(imgs, (list | tuple)) and imgs:
        first = imgs[0]
        shape = getattr(first, "shape", None)
        if isinstance(shape, tuple) and len(shape) >= 1:
            try:
                return int(shape[0])
            except (TypeError, ValueError):
                pass
    return 0


def _current_ident_df(main_window, df_all):
    """Slice df_all down to the currently active identification."""
    try:
        if main_window.current_ident_index < len(main_window.unique_ids):
            ident = main_window.unique_ids[main_window.current_ident_index]
        else:
            ident = df_all["Identification"].iloc[0]
    except (RuntimeError, AttributeError, TypeError, ValueError):
        ident = df_all["Identification"].iloc[0]
    return df_all[df_all["Identification"] == ident]


def _finite_rows_at_t(df_ident, intensity_column, tt_int: int):
    """Return the rows of df_ident at time tt_int with a finite value."""
    d = df_ident[df_ident["t"] == tt_int]
    if d.empty:
        return d
    v = pd.to_numeric(d[intensity_column], errors="coerce")
    mask = v.notna() & np.isfinite(v.astype(float))
    return d[mask]


def _nearest_time_with_data(df_ident, intensity_column, t_target, T):
    """Find the nearest time (searching outward from t_target) with finite
    data, returning (t_target, candidates) or (None, empty) if none exists.
    """
    candidates = _finite_rows_at_t(df_ident, intensity_column, t_target)
    if not candidates.empty:
        return t_target, candidates

    for delta in range(1, T):
        for tt_i in (t_target - delta, t_target + delta):
            if 0 <= tt_i < T:
                cand = _finite_rows_at_t(df_ident, intensity_column, tt_i)
                if not cand.empty:
                    return tt_i, cand
    return None, candidates


def _closest_track_at(candidates, intensity_column, y_value):
    """Return the TrackNumber whose value is closest to y_value, or None."""
    try:
        vals = pd.to_numeric(
            candidates[intensity_column], errors="coerce"
        ).astype(float)
        finite_mask = np.isfinite(vals)
        if not finite_mask.any():
            return None
        sub = candidates.loc[finite_mask]
        vals = vals.loc[finite_mask]
        closest_index = (vals - y_value).abs().idxmin()
        return sub.loc[closest_index, "TrackNumber"]
    except (RuntimeError, AttributeError, TypeError, ValueError) as e:
        LOG.warning(
            "[on_plot_single_click] Could not determine closest track: %s", e
        )
        return None


def on_plot_single_click(main_window, plot_widget, row, event) -> None:
    """Handle a single left-click on a plot row.

    Jumps to the clicked time point and selects the track whose value at
    that time is closest to the clicked y-position, for the metric currently
    displayed in that row.
    """
    btn = getattr(event, "button", lambda: None)()
    if btn != Qt.LeftButton:
        return

    coords = _map_click_to_data_coords(plot_widget, event)
    if coords is None:
        return
    x_clicked, y_value = coords

    selection = _resolve_feature_selection(main_window, row)
    if selection is None:
        return
    _feat_key, m_idx, ch_idx, definition = selection

    intensity_column = _build_intensity_column(definition, ch_idx, m_idx)
    if intensity_column is None:
        return

    df_all = getattr(main_window, "filtered_df", None)
    if df_all is None or df_all.empty:
        return

    T = _infer_time_len(main_window)
    if T <= 0:
        return

    df_ident = _current_ident_df(main_window, df_all)
    if df_ident.empty or intensity_column not in df_ident.columns:
        return

    mode = getattr(main_window, "_time_mode", TIME_MODE_T)
    xcol = x_column_for(
        mode,
        ch_idx=int(ch_idx or 1),
        df_cols=list(df_ident.columns),
    )

    tt = pd.to_numeric(df_ident["t"], errors="coerce").to_numpy(dtype=float)
    xx = pd.to_numeric(df_ident.get(xcol), errors="coerce").to_numpy(
        dtype=float
    )
    mask = np.isfinite(tt) & np.isfinite(xx)
    if not mask.any():
        return
    tt, xx = tt[mask], xx[mask]
    t_target = int(tt[np.nanargmin(np.abs(xx - x_clicked))])
    t_target = max(0, min(T - 1, t_target))

    t_target, candidates = _nearest_time_with_data(
        df_ident, intensity_column, t_target, T
    )
    if t_target is None:
        return
    main_window.current_time_index = t_target

    track_number = _closest_track_at(candidates, intensity_column, y_value)
    if track_number is None:
        return
    main_window.current_TrackNumber_plot = track_number

    ensure_current_df_subset(main_window)
    change_time_point(main_window, 0)
    refresh_all_mask_selections_at_current(main_window)
    main_window._refresh_all_row_igt()
    main_window._refresh_all_row_summaries()
