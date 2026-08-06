"""Labels layer edit history for MainWindow.

Picks the undo/redo target layer, snapshots each edit, writes the edited slice
back into ``self.labels``, and runs the shared post-edit refresh. The paint and
erase mouse callbacks that feed this engine live in ``mouse_bindings``.
"""

import contextlib
import logging
import weakref

import numpy as np
from napari.layers import Labels

from SECQUOIA.gui.lineage_tree.lineage_tree import lineage_tree
from SECQUOIA.gui.lineage_tree.lineage_zoom import (
    capture_dynamics_zoom,
    refresh_lineage_values,
    restore_dynamics_zoom,
)

LOG = logging.getLogger(__name__)


class LabelsEditing:
    """Undo/redo target resolution and the shared labels-edit write-back engine."""

    def _labels_target_for_history(self):
        """Pick a Labels layer to apply undo/redo."""
        viewer = self._history_viewer()
        strategies = (
            lambda: self._active_selected_labels_layer(viewer),
            self._last_used_labels_layer,
            lambda: self._named_labels_layer(viewer),
            lambda: self._fallback_segmentation_layer(viewer),
        )
        for strategy in strategies:
            with contextlib.suppress(
                RuntimeError, AttributeError, TypeError, ValueError
            ):
                lyr = strategy()
                if lyr is not None:
                    return lyr
        return None

    def _history_viewer(self):
        """Return the napari viewer used for undo/redo history, if any."""
        return getattr(self, "viewer", None) or getattr(self, "_viewer", None)

    def _active_selected_labels_layer(self, viewer):
        """Return the viewer's actively selected layer if it is a Labels layer."""
        if viewer is None:
            return None
        lyr = viewer.layers.selection.active
        return lyr if isinstance(lyr, Labels) else None

    def _last_used_labels_layer(self):
        """Return the most recently edited Labels layer, if still tracked."""
        getter = getattr(self, "_last_labels_layer_ref", None)
        return getter() if getter else None

    def _named_labels_layer(self, viewer):
        """Return the Labels layer named `self._last_labels_layer_name`."""
        name = getattr(self, "_last_labels_layer_name", None)
        if viewer is None or not name:
            return None
        for lyr in viewer.layers:
            if isinstance(lyr, Labels) and getattr(lyr, "name", None) == name:
                return lyr
        return None

    def _fallback_segmentation_layer(self, viewer):
        """Return the most recent Labels layer with a segmentation like name."""
        if viewer is None:
            return None
        for lyr in reversed(viewer.layers):
            name = getattr(lyr, "name", "") or ""
            if isinstance(lyr, Labels) and (
                name.startswith("Segmentation")
                or name.lower().startswith("labels")
            ):
                return lyr
        return None

    def _goto_time_for_history(self, t: int) -> None:
        """Move the active viewer to time index t (for undo/redo history)."""
        try:
            t = int(t)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return
        try:
            self.current_time_index = t
            viewer = getattr(self, "viewer", None) or getattr(
                self, "_viewer", None
            )
            if viewer is not None and hasattr(viewer, "dims"):
                step = list(viewer.dims.current_step)
                if step:
                    step[0] = t
                    viewer.dims.current_step = tuple(step)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            pass

    def _binding_from_layer_name(self, layer_name: str) -> dict | None:
        """Map a segmentation layer name to its label source."""
        base = "Segmentation"

        if layer_name == base:
            return {"kind": "ndarray", "key": None, "tag": base}

        if layer_name.startswith(base + "_"):
            key = layer_name[len(base) + 1 :]
            return {"kind": "dict", "key": key, "tag": layer_name}

        if layer_name.startswith(base):
            suffix = layer_name[len(base) :]
            if suffix.isdigit():
                return {
                    "kind": "seq",
                    "key": int(suffix) - 1,
                    "tag": layer_name,
                }

        return None

    def _coerce_tyx(self, arr: np.ndarray) -> np.ndarray:
        """Ensure array is [T, Y, X] for indexing by time."""
        return arr[None, ...] if arr.ndim == 2 else arr

    def _ensure_time_capacity(
        self, arr: np.ndarray, t: int, fill_value=0
    ) -> np.ndarray:
        """Grow an array along the time axis so index t is valid."""
        if arr.ndim == 2:
            return arr
        T = arr.shape[0]
        if t < T:
            return arr
        pad = t + 1 - T
        pad_width = [(0, pad), (0, 0), (0, 0)]
        fv = int(fill_value)
        if not np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.int32, copy=False)
        return np.pad(arr, pad_width, mode="constant", constant_values=fv)

    def _writeback_current_slice(self, layer):
        """Write the edited slice for time t back to self.labels, growing time axis if needed."""
        bind = self._binding_from_layer_name(layer.name)
        if not bind:
            return

        t = self.current_time_index
        slice_2d = np.asarray(self._coerce_tyx(layer.data)[t])

        if self._write_slice_to_labels(bind, t, slice_2d):
            self._mark_slice_corrected(bind, t)

    def _write_slice_to_labels(self, bind: dict, t: int, slice_2d) -> bool:
        """Write `slice_2d` into `self.labels` per `bind`. False if not applicable."""
        kind = bind["kind"]
        if kind == "ndarray":
            self.labels = self._place_slice(
                np.asarray(self.labels), t, slice_2d
            )
            return True
        if kind == "seq":
            return self._write_slice_into_seq(bind["key"], t, slice_2d)
        if kind == "dict":
            return self._write_slice_into_dict(bind["key"], t, slice_2d)
        return False

    def _write_slice_into_seq(self, idx: int, t: int, slice_2d) -> bool:
        """Write `slice_2d` at time `t` into the `idx`-th sequence element."""
        seq = list(self.labels)
        if not (0 <= idx < len(seq)):
            return False
        seq[idx] = self._place_slice(np.asarray(seq[idx]), t, slice_2d)
        self.labels = seq
        return True

    def _write_slice_into_dict(self, key, t: int, slice_2d) -> bool:
        """Write `slice_2d` at time `t` into `self.labels[key]`."""
        src = self.labels
        if key not in src:
            return False
        src[key] = self._place_slice(np.asarray(src[key]), t, slice_2d)
        self.labels = src
        return True

    def _place_slice(self, arr: np.ndarray, t: int, slice_2d):
        """Return `arr` with time index `t` replaced by `slice_2d`, growing T if needed."""
        if arr.ndim == 2:
            return slice_2d
        arr = self._ensure_time_capacity(arr, t, 0)
        arr[t] = slice_2d
        return arr

    def _mark_slice_corrected(self, bind: dict, t: int) -> None:
        """Record (mask_idx, t) as a corrected slice for save-state tracking."""
        try:
            if (
                not hasattr(self, "_corrected_slices")
                or self._corrected_slices is None
            ):
                self._corrected_slices = set()

            mask_idx = self._mask_index_from_binding(bind)
            if mask_idx is not None:
                self._corrected_slices.add((int(mask_idx), int(t)))
                LOG.debug(
                    "[save] Marked corrected: mask=%s, t=%d", mask_idx, int(t)
                )
        except (RuntimeError, AttributeError, TypeError) as e:
            LOG.warning("[save] Could not mark corrected slice: %s", e)

    def _edit_begin(self, layer):
        """Capture the preedit slice at current T."""
        if getattr(self, "_edit_cache", None) is None:
            self._edit_cache = {}

        t = int(getattr(self, "current_time_index", 0))

        try:
            self._last_labels_layer_ref = weakref.ref(layer)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            self._last_labels_layer_ref = None
        self._last_labels_layer_name = getattr(layer, "name", None)
        self._last_labels_layer_t = t

        tyx = self._coerce_tyx(layer.data)
        if t >= tyx.shape[0]:
            tyx = self._ensure_time_capacity(tyx, t, 0)

        prev = np.asarray(tyx[t]).copy()
        self._edit_cache[(layer.name, t)] = prev

    def _compute_edit_info(self, layer, mode: str) -> dict:
        """Compute, store, and log details about a labels edit."""
        t = int(getattr(self, "current_time_index", 0))
        key = (layer.name, t)
        cache = getattr(self, "_edit_cache", {})
        pre = cache.get(key, None)
        post = self._labels_slice_at(layer, t)

        diff = self._edit_diff_mask(pre, post)
        info = self._build_edit_info(layer, t, mode, pre, post, diff)

        self.last_edit_info = info
        self._log_edit_info(mode, t, info)

        if key in cache:
            del cache[key]
        self._edit_cache = cache
        return info

    def _labels_slice_at(self, layer, t: int) -> np.ndarray:
        """Return the 2D label slice at time `t` for `layer`, growing T if needed."""
        tyx = self._coerce_tyx(layer.data)
        if t >= tyx.shape[0]:
            tyx = self._ensure_time_capacity(tyx, t, 0)
        return np.asarray(tyx[t])

    def _edit_diff_mask(self, pre, post) -> np.ndarray:
        """Boolean mask of pixels changed between `pre` and `post`.

        No prior slice, or a shape mismatch, is treated as "no change"
        (an all False mask) since a pixel-wise comparison isn't possible.
        """
        if pre is not None and pre.shape == post.shape:
            return pre != post
        return post != post

    def _build_edit_info(
        self, layer, t: int, mode: str, pre, post, diff
    ) -> dict:
        """Assemble the edit info dict: changed pixel count, bbox, label deltas."""
        info = {
            "layer_name": layer.name,
            "t": t,
            "binding": self._binding_from_layer_name(layer.name),
            "changed_pixels": 0,
            "bbox": None,
            "labels_added": [],
            "labels_removed": [],
            "mode": mode,
        }

        n = int(np.count_nonzero(diff))
        info["changed_pixels"] = n
        if n:
            info["bbox"] = self._edit_bbox(diff)
            info["labels_added"], info["labels_removed"] = (
                self._edit_label_deltas(pre, post, diff)
            )
        return info

    def _edit_bbox(self, diff: np.ndarray) -> tuple[int, int, int, int]:
        """Bounding box (y0, x0, y1, x1) enclosing the changed pixels in `diff`."""
        ys, xs = np.where(diff)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        return (y0, x0, y1, x1)

    def _edit_label_deltas(
        self, pre, post, diff
    ) -> tuple[list[int], list[int]]:
        """Return (labels_added, labels_removed) among the changed pixels."""
        pre_lbls = (
            np.unique(pre[diff])
            if pre is not None
            else np.array([], dtype=post.dtype)
        )
        post_lbls = np.unique(post[diff])
        labels_added = [
            int(v) for v in post_lbls if v not in pre_lbls and v != 0
        ]
        labels_removed = [
            int(v) for v in pre_lbls if v not in post_lbls and v != 0
        ]
        return labels_added, labels_removed

    def _log_edit_info(self, mode: str, t: int, info: dict) -> None:
        """Log a one line summary of an applied labels edit."""
        tag = info["binding"]["tag"] if info["binding"] else info["layer_name"]
        LOG.info(
            "[%s] change @T=%s on %s: pixels=%s bbox=%s added=%s removed=%s",
            mode,
            t,
            tag,
            info["changed_pixels"],
            info["bbox"],
            info["labels_added"],
            info["labels_removed"],
        )

    def _refresh_lineage_after_edit(self) -> None:
        """Repaint the lineage tree after a mask edit.

        Measurement only changes (paint/erase/undo/redo/right-click) never
        change which Identification is shown, so the existing plot widget is
        redrawn in place. The user's current zoom on the lineage tree and on
        each row plot is captured up front and reapplied afterward, since
        both the in-place redraw and the full rebuild fallback below reset
        every plot's view range to the full data extent.
        """
        saved_zoom = capture_dynamics_zoom(self)
        if not refresh_lineage_values(self):
            lineage_tree(self)
        restore_dynamics_zoom(self, saved_zoom)

    def _refresh_sibling_labels_layers(self, layer) -> None:
        """Force every other viewer's copy of this mask layer to redraw."""
        name = getattr(layer, "name", None)
        if not name:
            return
        for viewer in getattr(self, "viewer_fluorescence", []):
            if viewer is None or not hasattr(viewer, "layers"):
                continue
            for other in viewer.layers:
                if other is layer or getattr(other, "name", None) != name:
                    continue
                try:
                    other.refresh()
                except (
                    IndexError,
                    KeyError,
                    RuntimeError,
                    AttributeError,
                    TypeError,
                    ValueError,
                ) as e:
                    LOG.warning(
                        "[sibling-sync] refresh failed for '%s': %s", name, e
                    )

    def _finish_labels_edit(self, layer, mode: str) -> None:
        """Shared tail for paint, erase, undo and redo.

        Computes the edit diff, writes the edited slice back into self.labels,
        remeasures the current track, then refreshes the lineage tree and any
        copy of the layer held by the other viewer.
        """
        LOG.debug(
            "[%s] '%s' updated at T=%s",
            mode,
            layer.name,
            self.current_time_index,
        )
        self._compute_edit_info(layer, mode=mode)
        self._writeback_current_slice(layer)

        try:
            self.update_mask_measurements_for_current_track(layer, source=mode)
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            LOG.warning("[update][%s] error: %s", mode, e)
        self._refresh_lineage_after_edit()
        self._refresh_sibling_labels_layers(layer)

    def _discard_edit_cache_entry(self, layer) -> None:
        """Drop this layer's cached preedit state at the current time point."""
        cache = getattr(self, "_edit_cache", {})
        key = (
            getattr(layer, "name", ""),
            int(getattr(self, "current_time_index", 0)),
        )
        cache.pop(key, None)
        self._edit_cache = cache

    def undo_labels_edit(self, layer):
        """Undo the last Labels edit via napari's own layer history.

        Runs the same post-edit tail as a paint or erase drag, so measurements
        and the lineage tree stay in sync with the reverted mask.
        """
        if layer is None or not hasattr(layer, "undo"):
            LOG.warning("[undo] layer has no undo()")
            return

        self._edit_begin(layer)
        try:
            layer.undo()
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            LOG.warning("[undo] error: %s", e)
            self._discard_edit_cache_entry(layer)
            return

        self._finish_labels_edit(layer, "undo")

    def redo_labels_edit(self, layer):
        """Redo the last undone Labels edit via napari's own layer history.

        Runs the same post-edit tail as a paint or erase drag, so measurements
        and the lineage tree stay in sync with the restored mask.
        """
        if layer is None or not hasattr(layer, "redo"):
            LOG.warning("[redo] layer has no redo()")
            return

        self._edit_begin(layer)
        try:
            layer.redo()
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            LOG.warning("[redo] error: %s", e)
            self._discard_edit_cache_entry(layer)
            return

        self._finish_labels_edit(layer, "redo")
