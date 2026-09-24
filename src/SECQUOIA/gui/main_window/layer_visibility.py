"""Napari layer visibility for MainWindow: channel/segmentation show/hide, zoom-to-track,
and populating the viewers with image/labels layers.
"""

import contextlib
import logging

import napari
import numpy as np
import qtawesome as qta
from napari.layers import Labels
from qtpy.QtCore import QTimer
from qtpy.QtGui import QIcon

from SECQUOIA.config import NAPARIPARAMETERS
from SECQUOIA.core.segmentation.mask_selection import apply_all_mask_selections
from SECQUOIA.core.tracking.track_data import update_track_df
from SECQUOIA.gui.cell_inspector.integration import notify_cell_inspector
from SECQUOIA.gui.lineage_tree.lineage_tree import lineage_tree
from SECQUOIA.gui.main_window.mouse_bindings import (
    add_mouse_drag_to_segmentation_layer,
)
from SECQUOIA.utils.helpers import _on_time_index_changed, _set_time_on_viewer
from SECQUOIA.utils.plotting import update_plot

LOG = logging.getLogger(__name__)


class LayerVisibility:
    """Channel/segmentation layer show/hide, zoom to track, and viewer population."""

    def _get_seg_layer(self, viewer, m_idx: int):
        """Find the segmentation layer for mask id m_idx in this viewer."""
        if viewer is None:
            return None

        try:
            if hasattr(self, "_seg_layer_name_for"):
                name = self._seg_layer_name_for(int(m_idx))
                lyr = self._find_layer_by_name(viewer, name)
                if lyr is not None:
                    return lyr
        except (RuntimeError, AttributeError, TypeError):
            pass

        try:
            guess = f"Segmentation{int(m_idx)}"
            lyr = self._find_layer_by_name(viewer, guess)
            if lyr is not None:
                return lyr
        except (RuntimeError, AttributeError, TypeError):
            pass

        try:

            labels_layers = [
                ly for ly in viewer.layers if isinstance(ly, Labels)
            ]
            if 1 <= int(m_idx) <= len(labels_layers):
                return labels_layers[int(m_idx) - 1]
        except (RuntimeError, AttributeError, TypeError):
            pass

        return None

    def select_channel_one_all(self) -> None:
        """Restore each viewer's last selected channel, auto-pick mask, and sync contrast UI."""
        combos = getattr(self, "channel_combos", [])
        if not combos:
            return
        default_channel = self.ids_channels[0][1:]
        viewer_channel = getattr(self, "_viewer_channel", {})
        for vi, combo in enumerate(combos):
            channel = viewer_channel.get(vi, default_channel)
            idx = combo.findData(channel)
            if idx == -1:
                channel = default_channel
                idx = combo.findData(channel)
            if idx == -1:
                continue
            combo.setCurrentIndex(idx)

            v = self._viewer_for_row_index(vi)
            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                self._set_active_channel_only(v, f"w{channel}")

            if hasattr(self, "mask_combos") and vi < len(self.mask_combos):
                mcombo = self.mask_combos[vi]
                midx = mcombo.findData(0)
                if midx != -1:
                    mcombo.blockSignals(True)
                    mcombo.setCurrentIndex(midx)
                    mcombo.blockSignals(False)
                    with contextlib.suppress(
                        RuntimeError, AttributeError, TypeError
                    ):
                        self._set_all_segs_visible(v)

            if hasattr(self, "contrast_widgets") and vi < len(
                self.contrast_widgets
            ):
                with contextlib.suppress(
                    RuntimeError, AttributeError, TypeError
                ):
                    self._sync_contrast_ui_from_layer(
                        v, self.contrast_widgets[vi]
                    )

    def zoom_in(self) -> None:
        """Zoom into the current time/track and select labels for ALL masks/layers,
        independent of which M is chosen in the UI."""
        if (
            not self._has_df_subset()
            and not self._init_df_subset_from_identification()
        ):
            return

        df_sub = self.df_subset
        if df_sub is None or df_sub.empty:
            LOG.debug("zoom_in: df_subset empty.")
            return

        df_sub = self._filtered_zoom_subset(df_sub)
        if df_sub.empty:
            df_sub = self._fallback_zoom_subset_by_identification()

        if df_sub is None or df_sub.empty:
            LOG.debug("zoom_in: no row to zoom.")
            return

        row = df_sub.iloc[0]
        apply_all_mask_selections(
            self,
            row,
            center_camera=True,
            zoom_level=NAPARIPARAMETERS.ZOOMFACTOR,
        )

    def _has_df_subset(self) -> bool:
        """True if `self.df_subset` is already set and non-empty."""
        return (
            hasattr(self, "df_subset")
            and self.df_subset is not None
            and not self.df_subset.empty
        )

    def _init_df_subset_from_identification(self) -> bool:
        """Set `self.df_subset` from the current Identification. False if no data."""
        df_all = getattr(self, "filtered_df", None)
        if df_all is None or df_all.empty:
            LOG.debug("zoom_in: no data.")
            return False
        ident = self._current_zoom_identification(df_all)
        self.df_subset = df_all[df_all["Identification"] == ident]
        return True

    def _current_zoom_identification(self, df_all):
        """Identification at `current_ident_index`, or the first row's as fallback."""
        try:
            if 0 <= self.current_ident_index < len(self.unique_ids):
                return self.unique_ids[self.current_ident_index]
        except (RuntimeError, AttributeError):
            pass
        return df_all["Identification"].iloc[0]

    def _filtered_zoom_subset(self, df_sub):
        """Narrow `df_sub` to the current track and time index, best effort."""
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            df_sub = df_sub[
                df_sub["TrackNumber"] == self.current_TrackNumber_plot
            ]
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            df_sub = df_sub[df_sub["t"] == self.current_time_index]
        return df_sub

    def _fallback_zoom_subset_by_identification(self):
        """Rederive a zoom subset from Identification alone, ignoring track/time."""
        try:
            df_all = getattr(self, "filtered_df", None)
            if df_all is not None and not df_all.empty:
                ident = self._current_zoom_identification(df_all)
                return df_all[df_all["Identification"] == ident]
        except (RuntimeError, AttributeError):
            pass
        return None

    def _eye_icon(self, visible: bool) -> QIcon:
        """White eye/eye slash icon for all states."""
        name = "fa5s.eye" if visible else "fa5s.eye-slash"
        return qta.icon(
            name,
            color="white",
            color_active="white",
            color_selected="white",
            color_disabled="white",
        )

    def _ch_layer_name_for(self, channel: str) -> str:
        """Return the napari layer name used for the given channel."""
        return f"Channel {channel}"

    def _is_channel_layer(self, layer) -> bool:
        """Recognize channel image layers by name."""
        return isinstance(getattr(layer, "__class__", None), type) and str(
            getattr(layer, "name", "")
        ).startswith("Channel ")

    def _set_active_channel_only(self, viewer, channel: str):
        """Show only the selected channel layer in the given viewer and make it active."""
        if viewer is None:
            return

        target_name = self._ch_layer_name_for(channel)
        target_layer = self._find_layer_by_name(viewer, target_name)

        for layer in list(viewer.layers):
            if self._is_channel_layer(layer):
                layer.visible = layer is target_layer

        if target_layer is not None:
            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                viewer.layers.selection = {target_layer}
                viewer.layers.selection.active = target_layer
            if (
                getattr(viewer.layers.selection, "active", None)
                is not target_layer
            ):
                with contextlib.suppress(RuntimeError, AttributeError):
                    sel = viewer.layers.selection
                    sel.clear()
                    sel.add(target_layer)
                    sel.active = target_layer

    def _seg_layer_name_for(self, idx: int) -> str:
        """Return the segmentation layer name for an index."""
        return f"Segmentation{idx}"

    def _is_seg_layer(self, layer) -> bool:
        """Return whether a layer is a segmentation layer."""
        return str(getattr(layer, "name", "")).startswith("Segmentation")

    def _set_active_seg_only(self, viewer, idx: int):
        """Show only Segmentation{idx} and make it active."""
        if viewer is None:
            return
        target_name = self._seg_layer_name_for(idx)
        target_layer = self._find_layer_by_name(viewer, target_name)
        for layer in list(viewer.layers):
            if self._is_seg_layer(layer):
                layer.visible = layer is target_layer
        if target_layer is not None:
            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                viewer.layers.selection = {target_layer}
                viewer.layers.selection.active = target_layer
            if (
                getattr(viewer.layers.selection, "active", None)
                is not target_layer
            ):
                with contextlib.suppress(
                    RuntimeError, AttributeError, TypeError
                ):
                    sel = viewer.layers.selection
                    sel.clear()
                    sel.add(target_layer)
                    sel.active = target_layer

    def _set_all_segs_visible(self, viewer):
        """Make every segmentation layer visible."""
        if viewer is None:
            return
        first_seg = None
        for layer in list(viewer.layers):
            if self._is_seg_layer(layer):
                layer.visible = True
                if first_seg is None:
                    first_seg = layer
        active = getattr(getattr(viewer, "layers", None), "selection", None)
        current_active = (
            getattr(active, "active", None) if active is not None else None
        )
        target = (
            current_active
            if (
                current_active is not None
                and self._is_seg_layer(current_active)
            )
            else first_seg
        )
        if target is not None:
            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                viewer.layers.selection = {target}
                viewer.layers.selection.active = target
            if getattr(viewer.layers.selection, "active", None) is not target:
                with contextlib.suppress(
                    RuntimeError, AttributeError, TypeError
                ):
                    sel = viewer.layers.selection
                    sel.clear()
                    sel.add(target)
                    sel.active = target

    def _find_layer_by_name(self, viewer, name: str):
        """Find a viewer layer by name."""
        if viewer is None:
            return None
        with contextlib.suppress(
            RuntimeError, AttributeError, TypeError, KeyError
        ):
            return viewer.layers[name]
        for lyr in viewer.layers:
            if getattr(lyr, "name", None) == name:
                return lyr
        return None

    def _viewer_for_row_index(self, idx: int):
        """Return the napari viewer mapped to the given plot row index."""
        if idx == 0:
            return getattr(self, "viewer_1", None)
        if idx == 1:
            return getattr(self, "viewer_2", None)
        return None

    def update_napari_viewer(self) -> None:
        """Update the napari viewer with current image stacks + segmentation,
        and live-sync time sliders across viewer_1 and viewer_2."""
        self.current_time_index = 0
        self.update_fluorescence_viewers()

        def _as_label_stack_array(stack) -> np.ndarray | None:
            if stack is None:
                return None
            if isinstance(stack, np.ndarray):
                arr = stack
                if arr.ndim == 2:
                    arr = arr[None, ...]
            else:
                try:
                    frames = [np.asarray(f) for f in stack if f is not None]
                except (RuntimeError, AttributeError, TypeError):
                    return None
                if not frames:
                    return None
                if not all(fr.ndim == 2 for fr in frames):
                    try:
                        arr = np.asarray(stack)
                    except (RuntimeError, AttributeError, TypeError):
                        return None
                else:
                    if not all(fr.shape == frames[0].shape for fr in frames):
                        return None
                    arr = np.stack(frames, axis=0)

            if not np.issubdtype(arr.dtype, np.integer):
                arr = arr.astype(np.int32, copy=False)
            return arr

        def _add_labels_flex(viewer, source, base_name, opacity) -> list:
            """Add labels layer(s) from an ndarray, list/tuple or dict source.

            Returns a list of (layer, desired_name) pairs.
            """
            created = []
            if isinstance(source, dict):
                for key, stk in source.items():
                    arr = _as_label_stack_array(stk)
                    if arr is not None:
                        safe_key = str(key).replace(" ", "")
                        desired = f"{base_name}_{safe_key}"
                        lyr = viewer.add_labels(
                            arr, name=desired, opacity=opacity
                        )

                        with contextlib.suppress(
                            RuntimeError, AttributeError, TypeError
                        ):
                            self._restrict_labels_draw_to_left_click(lyr)

                        lyr.name = desired
                        created.append((lyr, desired))

            elif isinstance(source, (list | tuple)):
                for i, stk in enumerate(source):
                    arr = _as_label_stack_array(stk)
                    if arr is not None:
                        desired = f"{base_name}{i+1}"
                        lyr = viewer.add_labels(
                            arr, name=desired, opacity=opacity
                        )

                        with contextlib.suppress(
                            RuntimeError, AttributeError, TypeError
                        ):
                            self._restrict_labels_draw_to_left_click(lyr)

                        lyr.name = desired
                        created.append((lyr, desired))

            else:
                arr = _as_label_stack_array(source)
                if arr is not None:
                    desired = base_name
                    lyr = viewer.add_labels(arr, name=desired, opacity=opacity)
                    with contextlib.suppress(
                        RuntimeError, AttributeError, TypeError
                    ):
                        self._restrict_labels_draw_to_left_click(lyr)
                    lyr.name = desired
                    created.append((lyr, desired))

            return created

        def _apply_drag_helper_to(viewer, lyr, desired_name):
            """Make the layer active, call the mouse drag helper, then restore name."""
            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                viewer.layers.selection.select_only(lyr)
            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                viewer.layers.active = lyr

            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                add_mouse_drag_to_segmentation_layer(self)

            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                lyr.name = desired_name

        def _disconnect_if_connected(viewer, cb):
            """Disconnect a time step callback from the viewer, ignoring if not connected."""
            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                viewer.dims.events.current_step.disconnect(cb)

        def _link_time_axes(v1, v2):
            """Link the time axes of two viewers so stepping one mirrors the other."""
            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                v1.dims.axis_labels = ("t", "YMorphology", "XMorphology")
                v2.dims.axis_labels = ("t", "YMorphology", "XMorphology")

            v1_id, v2_id = id(v1), id(v2)
            prev1 = self._time_link_registry.get(v1_id)
            prev2 = self._time_link_registry.get(v2_id)
            if prev1:
                _disconnect_if_connected(v1, prev1)
            if prev2:
                _disconnect_if_connected(v2, prev2)

            syncing = {"on": False}

            def _max_t(v) -> int | None:
                """Return the maximum time index for a viewer."""
                with contextlib.suppress(
                    RuntimeError, AttributeError, TypeError
                ):
                    return int(v.dims.nsteps[0]) - 1
                with contextlib.suppress(
                    RuntimeError, AttributeError, TypeError
                ):
                    for lyr in v.layers:
                        if isinstance(lyr, napari.layers.Image):
                            return int(lyr.data.shape[0]) - 1
                return None

            def _clamped_set(dst, t: int) -> int:
                mt = _max_t(dst)
                if mt is not None:
                    t = max(0, min(int(t), mt))
                _set_time_on_viewer(dst, int(t))
                return int(t)

            def _mirror_and_fire(src, dst):
                """Mirror time changes between viewers and trigger callbacks."""
                if syncing["on"]:
                    return
                syncing["on"] = True
                try:
                    t = int(src.dims.current_step[0])
                    t = _clamped_set(dst, t)
                    with contextlib.suppress(
                        RuntimeError, AttributeError, TypeError
                    ):
                        _on_time_index_changed(self, t)
                    self.current_time_index = t
                finally:
                    syncing["on"] = False

            def on_v1_step(event):
                """Handle step changes in viewer 1."""
                _mirror_and_fire(v1, v2)

            def on_v2_step(event):
                """Handle step changes in viewer 2."""
                _mirror_and_fire(v2, v1)

            v1.dims.events.current_step.connect(on_v1_step)
            v2.dims.events.current_step.connect(on_v2_step)
            self._time_link_registry[v1_id] = on_v1_step
            self._time_link_registry[v2_id] = on_v2_step

            try:
                t0 = int(getattr(self, "current_time_index", 0))
            except (RuntimeError, AttributeError, TypeError, ValueError):
                t0 = 0
            _clamped_set(v1, t0)
            _clamped_set(v2, t0)
            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                _on_time_index_changed(self, t0)
            self.current_time_index = t0

        for lyr in list(self.viewer_1.layers):
            if isinstance(lyr, napari.layers.Labels):
                self.viewer_1.layers.remove(lyr)
        for lyr in list(self.viewer_2.layers):
            if isinstance(lyr, napari.layers.Labels):
                self.viewer_2.layers.remove(lyr)

        # Add segmentation labels to both viewers
        base_seg_name = "Segmentation"
        created_1 = _add_labels_flex(
            self.viewer_1,
            self.labels,
            base_seg_name,
            opacity=NAPARIPARAMETERS.OPACITY,
        )
        created_2 = _add_labels_flex(
            self.viewer_2,
            self.labels,
            base_seg_name,
            opacity=NAPARIPARAMETERS.OPACITY,
        )

        for lyr, desired in created_1:
            _apply_drag_helper_to(self.viewer_1, lyr, desired)
        for lyr, desired in created_2:
            _apply_drag_helper_to(self.viewer_2, lyr, desired)

        if "Cellfate" not in self.track_df.columns:
            self.track_df["Cellfate"] = "Healthy"
            self.track_df["active"] = int("1")
            self.track_df["inspected"] = int("0")

        update_track_df(self)
        update_plot(self)
        lineage_tree(self)

        self.select_channel_one_all()

        try:
            _link_time_axes(self.viewer_1, self.viewer_2)
        except (RuntimeError, AttributeError, TypeError) as e:
            LOG.warning("Could not link time axes: %r", e)

        self._apply_mask_defaults_now()

        for vi in range(2):
            oc = (
                self.opacity_widgets[vi]
                if vi < len(self.opacity_widgets)
                else None
            )
            cc = (
                self.contrast_widgets[vi]
                if vi < len(self.contrast_widgets)
                else None
            )
            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                self._restore_viewer_display_settings(vi, oc, cc)

        notify_cell_inspector(self, "refresh_sources")

        self._viewers_dirty = True

    def _apply_mask_defaults_now(self) -> None:
        """Restore each viewer's last selected mask, falling back to the default."""
        m_n = max(0, int(getattr(self, "n_masks", 0)))
        viewer_mask = getattr(self, "_viewer_mask", {})
        for vi, v in enumerate((self.viewer_1, self.viewer_2)):
            default = 2 if (vi == 1 and m_n >= 2) else (1 if m_n >= 1 else 0)
            desired = viewer_mask.get(vi, default)
            if desired > m_n:
                desired = default
            try:
                if desired > 0:
                    self._set_active_seg_only(v, desired)
                else:
                    self._set_all_segs_visible(v)
            except (TypeError, ValueError):
                pass
            try:
                combo = self.mask_combos[vi]
                for idx in range(combo.count()):
                    if combo.itemData(idx) == desired:
                        combo.blockSignals(True)
                        combo.setCurrentIndex(idx)
                        combo.blockSignals(False)
                        break
            except (TypeError, ValueError):
                pass

    def update_fluorescence_viewers(self) -> None:
        """Replace both viewers' layers with one image layer per channel."""

        def _as_stacked(arr_or_list) -> np.ndarray:
            """Convert image data to a stacked array."""
            if isinstance(arr_or_list, np.ndarray):
                return arr_or_list
            return np.stack(arr_or_list, axis=0)

        image_stacks = []
        for channel in self.ids_channels:
            arr = getattr(self, "images", None)
            arr = arr[channel] if arr is not None else None
            if arr is None:
                LOG.warning(
                    "Image stack for channel %s is not initialized.", channel
                )
                continue
            stacked = _as_stacked(arr)
            self.images[channel] = stacked
            image_stacks.append((channel, stacked))

        for viewer_idx in (1, 2):
            viewer = getattr(self, f"viewer_{viewer_idx}", None)
            if viewer is None:
                LOG.warning(
                    "viewer_%s not initialized. Call create_napari_viewers() first.",
                    viewer_idx,
                )
                continue

            viewer.layers.clear()

            for ch, stack in image_stacks:
                viewer.add_image(stack, name=f"Channel {ch}")

        QTimer.singleShot(
            0,
            lambda: self.action_tracks_visible.isChecked()
            and self.action_tracks_visible.trigger(),
        )
