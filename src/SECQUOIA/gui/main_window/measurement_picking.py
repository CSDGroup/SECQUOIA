"""Click to identify and right-click measurement picking for MainWindow.

Resolves a click to a filtered_df row, forwards it to the fuse dialog, writes
the morphology anchor, and recomputes mask measurements for the current track.
"""

import contextlib
import logging
import re

import numpy as np
import pandas as pd
from napari.layers import Labels

from SECQUOIA.config import CURATIONSTATUS
from SECQUOIA.core.outlier_detection import (
    apply_outlier_selection_for_current_point,
    resolve_current_ident,
)
from SECQUOIA.core.outlier_detection.close_masks import update_flags_after_edit
from SECQUOIA.core.quantification import compute_step_distance_for_row
from SECQUOIA.core.quantification.candidates import (
    refresh_distances_and_candidates,
)
from SECQUOIA.core.quantification.remeasure_object import (
    changed_columns,
    current_position,
    log_measurement_update,
    measure_edit,
    measurement_columns,
    resolve_edit_target,
    snapshot_columns,
    write_measurements,
)
from SECQUOIA.core.segmentation.close_mask_view import (
    refresh_close_mask_view_at_current,
)
from SECQUOIA.core.segmentation.mask_selection import ensure_current_df_subset
from SECQUOIA.gui.curation_tree import apply_curation_status
from SECQUOIA.gui.dialogs.track_fuse_dialog import open_track_fuse_window
from SECQUOIA.gui.lineage_tree.lineage_zoom import (
    capture_dynamics_zoom,
    restore_dynamics_zoom,
)
from SECQUOIA.utils.plotting import update_plot
from SECQUOIA.utils.profiling import mask_edit_measurement

LOG = logging.getLogger(__name__)


class MeasurementPicking:
    """Resolve viewer clicks to filtered_df rows and recompute measurements."""

    def _report_pick_identification(self, viewer, pick: dict) -> None:
        """Look up `pick` in filtered_df and report it on the viewer."""
        label = int(pick["label"])
        t = pick["t"]
        m_idx = pick.get("m_idx")
        if not m_idx:
            LOG.warning(
                "[ctrl+right-click] Could not infer mask index (m); abort."
            )
            return

        df_t = self._filtered_rows_at_time(t)
        if df_t is None:
            return

        col = f"label_id_m{int(m_idx)}"
        if col not in df_t.columns:
            LOG.warning(
                "[ctrl+right-click] column '%s' not found at t=%s", col, t
            )
            return

        target = self._cast_label_to_column_dtype(df_t[col], label)
        hits = df_t[df_t[col] == target]
        if hits.empty:
            msg = f"[ctrl+right-click] No Identification found for {col}={label}, t={t}"
            LOG.info(msg)
            self._set_viewer_status(viewer, msg)
            return

        ident, cell_num = self._identification_from_row(hits.iloc[0])
        cell_part = f", Cell {cell_num}" if cell_num is not None else ""
        msg = f"[ctrl+right-click] {col}={label}, t={t} -> Identification {ident}{cell_part}"
        LOG.info(msg)
        self._set_viewer_status(viewer, msg)

        if ident:
            self._forward_identification_to_fuse_dialog(ident, t, cell_num)

    def _filtered_rows_at_time(self, t):
        """Rows of `self.filtered_df` at time `t`, or None if unavailable."""
        df = getattr(self, "filtered_df", None)
        if not isinstance(df, pd.DataFrame) or df.empty:
            LOG.warning("[ctrl+right-click] filtered_df missing or empty")
            return None
        return df[df["t"] == t]

    def _cast_label_to_column_dtype(self, column, label):
        """Cast `label` to `column`'s dtype so equality lookups work."""
        try:
            return column.dtype.type(label)
        except (RuntimeError, AttributeError, TypeError):
            return label

    def _identification_from_row(self, row) -> tuple[str, int | None]:
        """Extract (Identification, TrackNumber) from a filtered_df row."""
        ident_val = row.get("Identification", "")
        ident = "" if pd.isna(ident_val) else str(ident_val)

        cell_num = None
        tv = row.get("TrackNumber", None)
        if tv is not None and not (isinstance(tv, float) and pd.isna(tv)):
            try:
                cell_num = int(round(float(tv)))
            except (RuntimeError, AttributeError, TypeError, ValueError):
                with contextlib.suppress(
                    RuntimeError, AttributeError, TypeError, ValueError
                ):
                    cell_num = int(tv)
        return ident, cell_num

    def _set_viewer_status(self, viewer, msg: str) -> None:
        """Set `viewer.status`, ignoring viewers that don't support it."""
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            viewer.status = msg

    def _forward_identification_to_fuse_dialog(
        self, ident: str, t, cell_num: int | None
    ) -> None:
        """Send a picked identification to the fuse dialog, if available."""
        if not hasattr(self, "_add_ident_to_fuse_dialog"):
            return
        with contextlib.suppress(
            RuntimeError, AttributeError, TypeError, ValueError
        ):
            self._add_ident_to_fuse_dialog(ident, t=t, cell_num=cell_num)

    def _ensure_fuse_dialog(self):
        """Ensure the fuse dialog is open."""
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            if getattr(self, "fuse_dialog", None) is None:
                open_track_fuse_window(self)
        return getattr(self, "fuse_dialog", None)

    def _add_ident_to_fuse_dialog(
        self,
        ident_str: str,
        *,
        t: int | None = None,
        cell_num: int | None = None,
    ) -> None:
        """Fill the next free Identification (and cell/time) slot in the fuse dialog."""
        dlg = self._ensure_fuse_dialog()
        if dlg is None:
            return

        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            f1 = self.fuse_tree_id_field_1
            f2 = self.fuse_tree_id_field_2
            c1 = getattr(self, "fuse_cell_field_1", None)
            c2 = getattr(self, "fuse_cell_field_2", None)

            slot = int(getattr(self, "_fuse_next_slot", 1))
            ident_str = "" if ident_str is None else str(ident_str)

            if slot == 1:
                f1.setText(ident_str)
                if c1 is not None:
                    c1.setText("" if cell_num is None else str(int(cell_num)))
                self._fuse_next_slot = 2
            else:
                f2.setText(ident_str)
                if c2 is not None:
                    c2.setText("" if cell_num is None else str(int(cell_num)))
                self._fuse_next_slot = 1

            if t is not None:
                self.fuse_time_spin.setValue(int(t))

    def _is_labels_layer(self, layer) -> bool:
        """True if `layer` is a napari Labels layer, or quacks like one.

        The duck-typed fallback exists because ``isinstance`` can raise if the
        layer has been deleted on the C++ side while a callback is in flight.
        """
        try:
            return isinstance(layer, Labels)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return hasattr(layer, "get_value") and hasattr(layer, "data")

    def _viewer_display_name(self, viewer) -> str:
        """Human-readable name for a viewer."""
        name = getattr(viewer, "title", None)
        if name:
            return str(name)
        return next(
            (k for k, v in vars(self).items() if v is viewer),
            f"Viewer@{id(viewer):x}",
        )

    def _frame_shape(self, layer, t: int):
        """Return (H, W) of the frame shown at time `t`, or (None, None)."""
        try:
            arr = np.asarray(layer.data)
            if arr.ndim == 2:
                return arr.shape[-2], arr.shape[-1]
            if arr.ndim >= 3 and t < arr.shape[0]:
                return arr[t].shape[-2], arr[t].shape[-1]
        except (RuntimeError, AttributeError, TypeError, ValueError):
            pass
        return None, None

    def _mask_index_from_binding(self, bind: dict):
        """Map a layer binding to its 1-based mask index, or None."""
        try:
            kind = bind.get("kind")
            if kind == "ndarray":
                return 1
            if kind == "seq":
                return int(bind["key"]) + 1
            if kind == "dict":
                m = re.search(
                    r"Segmentation[_ ]*?(\d+)",
                    bind.get("tag", ""),
                    flags=re.IGNORECASE,
                )
                if m:
                    return int(m.group(1))
        except (RuntimeError, AttributeError, TypeError, ValueError):
            pass
        return None

    def _pick_segmentation_at_cursor(
        self, viewer, event, *, include_background: bool = False
    ):
        """Return info about the Segmentation label under the cursor, or None.

        By default, background pixels (label <= 0) are skipped; pass
        `include_background=True` to also report a hit on background.
        """
        t = int(getattr(self, "current_time_index", 0))

        for layer in reversed(viewer.layers):
            if not getattr(layer, "visible", True):
                continue
            if not self._is_labels_layer(layer):
                continue

            bind = self._binding_from_layer_name(getattr(layer, "name", ""))
            if not bind:
                continue

            try:
                label_val = layer.get_value(
                    event.position,
                    view_direction=getattr(event, "view_direction", None),
                    dims_displayed=getattr(event, "dims_displayed", None),
                    world=True,
                )
            except (RuntimeError, AttributeError, TypeError, ValueError):
                label_val = None

            if label_val is None:
                continue
            if not include_background and int(label_val) <= 0:
                continue

            try:
                pos_data = layer.world_to_data(np.asarray(event.position))
                y = int(round(float(pos_data[-2])))
                x = int(round(float(pos_data[-1])))
            except (RuntimeError, AttributeError, TypeError, ValueError):
                continue

            height, width = self._frame_shape(layer, t)
            if height is not None and width is not None:
                y = max(0, min(height - 1, y))
                x = max(0, min(width - 1, x))

            return {
                "layer": layer,
                "label": int(label_val),
                "XMorphology": x,
                "YMorphology": y,
                "t": t,
                "tag": bind.get("tag", getattr(layer, "name", "")),
                "m_idx": self._mask_index_from_binding(bind),
            }

        return None

    def _resolve_current_ident_str(self) -> str:
        """Identification of the track currently under inspection, as str."""
        ident = getattr(self, "ident", None)
        if ident is not None:
            return str(ident)
        try:
            return str(self.unique_ids[self.current_ident_index])
        except (
            RuntimeError,
            AttributeError,
            TypeError,
            ValueError,
            IndexError,
            KeyError,
        ):
            return ""

    def _infer_track_number(self, df, ident: str, t: int):
        """Derive the TrackNumber for (ident, t) when the plot has none."""
        try:
            candidates = df[
                (df["Identification"].astype(str) == ident) & (df["t"] == t)
            ]
            unique = candidates["TrackNumber"].dropna().unique()
            if unique.size == 1:
                return int(unique[0])
        except (RuntimeError, AttributeError, TypeError, ValueError, KeyError):
            return None
        return None

    def _resolve_track_row(self, df, ident: str, t: int):
        """Return (row_index, track_number) for the current selection, or None."""
        track_no = getattr(self, "current_TrackNumber_plot", None)
        if track_no is None:
            track_no = self._infer_track_number(df, ident, t)

        if track_no is None:
            LOG.warning(
                "[right-click] TrackNumber unknown for Identification=%r, t=%s.",
                ident,
                t,
            )
            return None

        try:
            row_mask = (
                (df["Identification"].astype(str) == ident)
                & (df["TrackNumber"] == track_no)
                & (df["t"] == t)
            )
            idxs = df.index[row_mask].to_list()
        except (
            RuntimeError,
            AttributeError,
            TypeError,
            ValueError,
            KeyError,
        ) as e:
            LOG.warning("[right-click] Row lookup failed: %s", e)
            return None

        if not idxs:
            LOG.warning(
                "[right-click] No row for Identification=%r, TrackNumber=%s, t=%s.",
                ident,
                track_no,
                t,
            )
            return None

        return idxs[0], track_no

    def _ensure_float_column(self, df, column: str) -> None:
        """Make sure `column` exists on `df` and can hold floats."""
        if column not in df.columns:
            df[column] = np.nan
        elif pd.api.types.is_integer_dtype(df[column].dtype):
            df[column] = df[column].astype(float)

    def _write_morphology_anchor(self, pick: dict) -> bool:
        """Store the picked pixel as the morphology anchor for the current row.

        Returns True when the dataframe was updated. A False return means the
        pick is unusable and no measurement refresh should be attributed to it.
        """
        df = getattr(self, "track_df", None)
        if not (isinstance(df, pd.DataFrame) and not df.empty):
            LOG.warning(
                "[right-click] track_df missing or empty; aborting update."
            )
            return False

        t = int(pick["t"])
        ident = self._resolve_current_ident_str()

        located = self._resolve_track_row(df, ident, t)
        if located is None:
            return False
        row_idx, _track_no = located

        self._ensure_float_column(df, "XMorphology")
        self._ensure_float_column(df, "YMorphology")

        x = int(pick["XMorphology"])
        y = int(pick["YMorphology"])
        df.at[row_idx, "XMorphology"] = float(x)
        df.at[row_idx, "YMorphology"] = float(y)
        LOG.debug(
            "[right-click] Set base anchors -> x=%s, y=%s (row=%s)",
            x,
            y,
            row_idx,
        )

        self.track_df = df
        return True

    def _remeasure_all_masks(self, viewer) -> None:
        """Recompute measurements for every configured mask in this viewer."""
        m_n = max(0, int(getattr(self, "n_masks", 0)))
        if m_n == 0:
            LOG.warning("[right-click][ALL] No masks configured (n_masks=0).")
            return

        for m_idx in range(1, m_n + 1):
            layer = self._get_seg_layer(viewer, m_idx)
            if layer is None:
                continue
            try:
                self.update_mask_measurements_for_current_track(
                    layer, source="right_click_all"
                )
            except (
                RuntimeError,
                AttributeError,
                TypeError,
                ValueError,
            ) as e:
                LOG.warning(
                    "[update][right_click_all] m%s error: %s", m_idx, e
                )

    def _remeasure_single_layer(self, layer) -> None:
        """Recompute measurements for the one layer the user clicked."""
        try:
            self.update_mask_measurements_for_current_track(
                layer, source="right_click"
            )
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            LOG.warning("[update][right_click] error: %s", e)

    def _recompute_measurements_after_pick(self, viewer, picked_layer) -> None:
        """Dispatch the measurement refresh for the viewer's mask selection.

        Mask selection 0 means "ALL" in the per viewer dropdown, in which case
        every mask is refreshed regardless of which layer was hit.
        """
        try:
            m_sel = self._mask_selection_for_viewer(viewer)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            m_sel = None

        if m_sel == 0:
            self._remeasure_all_masks(viewer)
        elif picked_layer is not None:
            self._remeasure_single_layer(picked_layer)

    def update_mask_measurements_for_current_track(
        self, layer, *, source: str = ""
    ) -> None:
        """Recompute mask-based measurements for the currently selected Tree-ID."""
        t = int(getattr(self, "current_time_index", 0))
        ident = resolve_current_ident(self) or ""

        target = resolve_edit_target(self, layer, ident=ident, t=t)
        if target is None:
            return

        with mask_edit_measurement(self, current_position(self)):
            columns, lab_col = measurement_columns(self, target.mask_idx)
            touched = columns + [lab_col]
            before = snapshot_columns(self.track_df, target.row_idx, touched)

            measured = measure_edit(self, target)
            write_measurements(self, target, measured)

            try:
                rows = refresh_distances_and_candidates(self, target)
                update_flags_after_edit(self, rows, target.row_idx)
            except (
                KeyError,
                TypeError,
                ValueError,
                RuntimeError,
                AttributeError,
            ) as err:
                LOG.warning("[update][candidates] refresh failed: %s", err)

            compute_step_distance_for_row(
                self.track_df,
                ident=target.ident,
                track_no=target.track_no,
                t=target.t,
                mask_idx=target.mask_idx,
            )

            try:
                self._recompute_derived_for_row(
                    target.row_idx, mask_idx=target.mask_idx
                )
            except (
                KeyError,
                TypeError,
                ValueError,
                RuntimeError,
                AttributeError,
            ) as err:
                LOG.warning("[update][derived] recompute failed: %s", err)

            after = snapshot_columns(self.track_df, target.row_idx, touched)
            log_measurement_update(
                target, changed_columns(before, after, touched), source=source
            )

        self._refresh_after_measurement_update(target)

    def _refresh_after_measurement_update(self, target) -> None:
        """Refresh the derived frames, plots and tree state after a write."""
        df = self.track_df

        pos = current_position(self)
        if pos is not None and "Position" in df.columns:
            try:
                self.filtered_df = df[df["Position"] == int(pos)].copy()
                LOG.debug(
                    "[update] filtered_df refreshed for Position %s (n=%d)",
                    pos,
                    len(self.filtered_df),
                )
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                LOG.warning("[update] filtered_df refresh failed: %s", e)
        else:
            LOG.debug(
                "[update] Skipped filtered_df refresh (Position unknown or column missing)."
            )

        try:
            ok = ensure_current_df_subset(self)
            n = (
                0
                if getattr(self, "df_subset", None) is None
                else len(self.df_subset)
            )
            LOG.debug(
                "[update] df_subset %s for Identification=%r (n=%d)",
                "ready" if ok else "empty",
                target.ident,
                n,
            )
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            LOG.warning("[update] ensure_current_df_subset failed: %s", e)

        apply_outlier_selection_for_current_point(self)

        try:
            refresh_close_mask_view_at_current(self)
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            LOG.warning("[update] close-mask view refresh skipped: %s", e)

        try:
            changed_track_id = int(
                self.track_df.at[target.row_idx, "track_id"]
            )
            self.update_tracks_for_ids([changed_track_id])
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            LOG.warning("[update] fast tracks refresh skipped: %s", e)

        saved_row_zoom = capture_dynamics_zoom(self)
        update_plot(self)
        restore_dynamics_zoom(self, saved_row_zoom)
        self._set_inspected_for_ident(target.ident, 2)

        item = self._find_tree_item_for(target.ident, None)
        if item:
            apply_curation_status(item, CURATIONSTATUS.CURATION_CHECKED)
            self._sync_children_visuals_from_df(item)
