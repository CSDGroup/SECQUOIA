"""The napari 'Tracks' layer for MainWindow: the tracking toolbar, the tracks/track-ID
menu toggles, and syncing the Tracks + TrackLabels layers from filtered_df.
"""

import contextlib
from functools import partial

import numpy as np
import pandas as pd
import qtawesome as qta
from qtpy.QtCore import QSignalBlocker, Qt
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QSizePolicy, QWidget

from SECQUOIA.config import STYLE, TOOLTIPSTEXT, TracksViewConfig
from SECQUOIA.core.tracking import (
    on_division_clicked,
    on_new_id_clicked,
    # on_redo_tracking_clicked, # TODO: tracking undo/redo hidden until further testing
    on_remove_division_clicked,
    on_split_tree,
    # on_undo_tracking_clicked, # TODO: tracking undo/redo hidden until further testing
)
from SECQUOIA.gui.dialogs.track_fuse_dialog import open_track_fuse_window


class TracksLayer:
    """Tracking toolbar and the napari Tracks/TrackLabels layer sync."""

    def _toggle_tracking(self, checked: bool | None = None) -> None:
        """Show or hide the tracking bar, building it on first use."""
        if not hasattr(self, "tracking_bar_container"):
            self.tracking_bar_container = self._build_tracking_bar()
            try:
                self.layout2.insertWidget(1, self.tracking_bar_container)
            except (RuntimeError, AttributeError, TypeError):
                self.layout2.addWidget(self.tracking_bar_container)

        if checked is None:
            desired = bool(self.tracking_bar_container.isVisible())
        else:
            desired = bool(checked)

        self.tracking_bar_container.setVisible(desired)

        if hasattr(self, "action_tracking_bar"):
            with QSignalBlocker(self.action_tracking_bar):
                self.action_tracking_bar.setChecked(desired)

    def _build_tracking_bar(self) -> QWidget:
        """Create the tracking toolbar."""
        container = QWidget()
        container.setObjectName("tracking_bar")
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        h = QHBoxLayout(container)
        h.setContentsMargins(4, 4, 4, 4)
        h.setSpacing(6)

        def _mk_btn(text: str) -> QPushButton:
            """Create a tracking toolbar button."""
            b = QPushButton(text)
            b.setFocusPolicy(Qt.NoFocus)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.setMinimumHeight(24)
            b.setMinimumWidth(40)
            b.setStyleSheet(f"""
                QPushButton {{
                    padding: 2px 10px;
                    border-radius: 6px;
                    background: #2b2b2b;
                    color: #e6e6e6;
                    border: 1px solid #444;
                    font-size: {STYLE.FONT_SIZE}px;
                }}
                QPushButton:hover {{ background: #3a3a3a; }}
                QPushButton:pressed {{ background: #1f1f1f; }}
                """)
            return b

        def _mk_icon_btn(icon_name: str) -> QPushButton:
            """Create a compact, icon-only tracking toolbar button."""
            b = QPushButton()
            b.setIcon(qta.icon(icon_name, color="white"))
            b.setFocusPolicy(Qt.NoFocus)
            b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            b.setFixedSize(32, 24)
            b.setStyleSheet("""
                QPushButton {
                    padding: 2px;
                    border-radius: 6px;
                    background: #2b2b2b;
                    border: 1px solid #444;
                }
                QPushButton:hover { background: #3a3a3a; }
                QPushButton:pressed { background: #1f1f1f; }
                """)
            return b

        btn_new_id = _mk_btn("New ID")
        btn_div = _mk_btn("Division")
        btn_remove = _mk_btn("Remove Division")
        btn_split = _mk_btn("Split Tree")
        btn_fuse = _mk_btn("Fuse Tree")
        # TODO: Undo/Redo for tracking edits — hidden for now, will be tested
        # further before it is added back to the tracking bar.
        # btn_undo_track = _mk_icon_btn("fa5s.undo")
        # btn_redo_track = _mk_icon_btn("fa5s.redo")

        # Wire signals
        btn_new_id.clicked.connect(partial(on_new_id_clicked, self))
        btn_div.clicked.connect(partial(on_division_clicked, self))
        btn_remove.clicked.connect(partial(on_remove_division_clicked, self))
        btn_split.clicked.connect(partial(on_split_tree, self))
        btn_fuse.clicked.connect(partial(open_track_fuse_window, self))
        # btn_undo_track.clicked.connect(partial(on_undo_tracking_clicked, self))
        # btn_redo_track.clicked.connect(partial(on_redo_tracking_clicked, self))

        btn_new_id.setToolTip(TOOLTIPSTEXT.NEW_ID)
        btn_div.setToolTip(TOOLTIPSTEXT.DIVISION)
        btn_remove.setToolTip(TOOLTIPSTEXT.REMOVE_DIVISION)
        btn_split.setToolTip(TOOLTIPSTEXT.SPLIT_TREE)
        btn_fuse.setToolTip(TOOLTIPSTEXT.FUSE_TREE)
        # btn_undo_track.setToolTip(TOOLTIPSTEXT.UNDO_TRACKING)
        # btn_redo_track.setToolTip(TOOLTIPSTEXT.REDO_TRACKING)

        h.addWidget(btn_new_id, 1)
        h.addWidget(btn_div, 1)
        h.addWidget(btn_remove, 1)
        h.addWidget(btn_split, 1)
        h.addWidget(btn_fuse, 1)
        # h.addWidget(btn_undo_track, 0)
        # h.addWidget(btn_redo_track, 0)
        container.setVisible(False)

        self._tracking_edit_buttons = (
            btn_div,
            btn_remove,
            btn_split,
            btn_fuse,
        )
        self._tracking_edit_button_tooltips = {
            btn_div: TOOLTIPSTEXT.DIVISION,
            btn_remove: TOOLTIPSTEXT.REMOVE_DIVISION,
            btn_split: TOOLTIPSTEXT.SPLIT_TREE,
            btn_fuse: TOOLTIPSTEXT.FUSE_TREE,
        }
        self.refresh_tracking_button_states()

        return container

    def refresh_tracking_button_states(self) -> None:
        """Grey out lineage-editing buttons when the current position has no tracking data yet."""
        buttons = getattr(self, "_tracking_edit_buttons", None)
        if not buttons:
            return

        df = getattr(self, "filtered_df", None)
        has_idents = (
            isinstance(df, pd.DataFrame)
            and not df.empty
            and "Identification" in df.columns
            and not df["Identification"].dropna().empty
        )
        disabled_tip = (
            "This position has no tracking data yet — create one first "
            "via Tracking bar → New ID."
        )
        tooltips = getattr(self, "_tracking_edit_button_tooltips", {})

        for btn in buttons:
            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                btn.setEnabled(has_idents)
                btn.setToolTip(
                    tooltips.get(btn, "") if has_idents else disabled_tip
                )

    def _foreach_tracks_layer(self):
        """Yield the tracks layer for each fluorescence viewer."""
        for viewer in getattr(self, "viewer_fluorescence", []):
            layer = self._ensure_tracks_layer(viewer)
            if layer is not None:
                yield layer

    def _sync_tracks_menu_state(self) -> None:
        """Synchronize the 'View → Tracks' menu checkboxes with the actual layer state."""
        if not hasattr(self, "action_tracks_visible") or not hasattr(
            self, "action_tracks_ids"
        ):
            return

        layers = list(self._foreach_tracks_layer())
        if not layers:
            with QSignalBlocker(self.action_tracks_visible):
                self.action_tracks_visible.setChecked(False)
            with QSignalBlocker(self.action_tracks_ids):
                self.action_tracks_ids.setChecked(False)
            return

        any_lines = any(
            (
                getattr(layer, "display_tail", False)
                and getattr(layer, "tail_length", 0) > 0
            )
            or (getattr(layer, "head_length", 0) > 0)
            for layer in layers
        )

        any_ids = any(
            "TrackLabels" in viewer.layers
            and viewer.layers["TrackLabels"].visible
            for viewer in getattr(self, "viewer_fluorescence", [])
        )

        with QSignalBlocker(self.action_tracks_visible):
            self.action_tracks_visible.setChecked(bool(any_lines))
        with QSignalBlocker(self.action_tracks_ids):
            self.action_tracks_ids.setChecked(bool(any_ids))
        if hasattr(self, "action_tracking_bar") and hasattr(
            self, "tracking_bar_container"
        ):
            with QSignalBlocker(self.action_tracking_bar):
                self.action_tracking_bar.setChecked(
                    bool(self.tracking_bar_container.isVisible())
                )

    def _backup_line_render_state(self, layer):
        """Store the layer's tail/head render settings in its metadata for later restore."""
        meta = getattr(layer, "metadata", None)
        if meta is None:
            layer.metadata = {}
            meta = layer.metadata
        meta["__lines_backup__"] = {
            "display_tail": bool(getattr(layer, "display_tail", True)),
            "tail_length": int(getattr(layer, "tail_length", 0)),
            "head_length": int(getattr(layer, "head_length", 0)),
        }

    def _restore_line_render_state(self, layer, cfg):
        """Restore a tracks layer's tail/head render settings, or fall back to cfg."""
        meta = getattr(layer, "metadata", {})
        backup = meta.get("__lines_backup__")
        if backup:
            layer.display_tail = bool(backup.get("display_tail", True))
            layer.tail_length = int(backup.get("tail_length", 0))
            layer.head_length = int(backup.get("head_length", 0))
        else:
            layer.display_tail = bool(cfg.show_track_tail)
            layer.tail_length = int(cfg.tail_length)
            layer.head_length = int(cfg.head_length)

    def on_toggle_tracks_visible(self, checked: bool) -> None:
        """Toggle track line/segment rendering; also makes the layer visible when enabling."""

        cfg = getattr(self, "tracks_cfg", None) or TracksViewConfig()

        for layer in self._foreach_tracks_layer():
            if checked:
                self._restore_line_render_state(layer, cfg)
                if not layer.visible:
                    layer.visible = True
                with contextlib.suppress(
                    RuntimeError, AttributeError, TypeError
                ):
                    layer.data = layer.data
            else:
                self._backup_line_render_state(layer)
                layer.display_tail = False
                layer.head_length = 0
                if int(getattr(layer, "tail_length", 0)) <= 0:
                    layer.tail_length = 1

        if hasattr(self, "tracks_cfg"):
            self.tracks_cfg.render_track_lines = bool(checked)

        self._sync_tracks_menu_state()

    def on_toggle_tracks_ids(self, checked: bool) -> None:
        """Toggle custom track labels instead of napari's native numeric track IDs."""

        for viewer in getattr(self, "viewer_fluorescence", []):
            track_layer = self._ensure_tracks_layer(viewer)

            track_layer.display_id = False

            label_layer = self._ensure_track_label_layer(viewer)
            label_layer.visible = bool(checked)

            if checked and not track_layer.visible:
                track_layer.visible = True

        if hasattr(self, "tracks_cfg"):
            self.tracks_cfg.show_track_ids = bool(checked)

        self._sync_tracks_menu_state()

    def _ensure_tracks_layer(self, viewer):
        """Ensure a 'Tracks' layer exists on a viewer and apply initial settings."""

        cfg = getattr(self, "tracks_cfg", None) or TracksViewConfig()

        if "Tracks" not in viewer.layers:
            data = (
                np.array([[0, 0, 0, 0]], dtype=int)
                if cfg.use_placeholder_row
                else np.empty((0, 4), dtype=np.int64)
            )

            layer = viewer.add_tracks(
                data,
                head_length=int(cfg.head_length),
                tail_length=int(cfg.tail_length),
                name="Tracks",
            )

            layer.display_id = False
            layer.display_tail = bool(cfg.show_track_tail)
            layer.opacity = float(cfg.opacity)

            # Start hidden if configured
            layer.visible = not cfg.init_hidden_until_data

            if cfg.use_placeholder_row:
                layer.metadata["is_placeholder"] = True

        return viewer.layers["Tracks"]

    def _sort_tracks_array_inplace(self, arr) -> np.ndarray:
        """Return the tracks array sorted by the configured primary and secondary columns."""
        if arr.size == 0:
            return arr
        cfg = getattr(self, "tracks_cfg", None) or TracksViewConfig()
        col = {"track_id": 0, "t": 1}
        secondary = col.get(cfg.sort_secondary, 1)
        primary = col.get(cfg.sort_primary, 0)
        order = np.lexsort((arr[:, secondary], arr[:, primary]))
        return arr[order]

    def update_tracks_for_ids(self, track_ids) -> None:
        """Refresh the 'Tracks' layer for a subset of track IDs."""
        if (
            not self._has_filtered_tracks()
            or track_ids is None
            or len(track_ids) == 0
        ):
            return

        self._ensure_track_label_column()
        ids = np.asarray(track_ids)
        new_block = self._track_rows_as_array(ids)
        cfg = getattr(self, "tracks_cfg", None) or TracksViewConfig()

        for viewer in getattr(self, "viewer_fluorescence", []):
            self._update_tracks_layer_for_viewer(viewer, ids, new_block, cfg)

    def _has_filtered_tracks(self) -> bool:
        """True if `self.filtered_df` exists and has rows."""
        df = getattr(self, "filtered_df", None)
        return df is not None and len(df) > 0

    def _track_rows_as_array(self, ids) -> np.ndarray:
        """Rows [track_id, t, YMorphology, XMorphology] for `ids`, as a float array."""
        return (
            self.filtered_df.loc[
                self.filtered_df["track_id"].isin(ids),
                ["track_id", "t", "YMorphology", "XMorphology"],
            ]
            .to_numpy()
            .astype(float, copy=False)
        )

    def _update_tracks_layer_for_viewer(
        self, viewer, ids, new_block, cfg
    ) -> None:
        """Merge `new_block` into one viewer's Tracks layer, replacing rows in `ids`."""
        layer = self._ensure_tracks_layer(viewer)
        old = self._current_tracks_data(layer)

        if old.size == 0 and new_block.size == 0:
            return

        updated = self._merge_track_rows(old, new_block, ids)
        if updated is None:
            return

        if updated.size:
            updated[:, 0] = updated[:, 0].astype(np.int64, copy=False)
            layer.data = updated
            self._update_track_label_layer(viewer, self.filtered_df)

        if updated.size > 0 and layer.metadata.get("is_placeholder", False):
            self._clear_placeholder_flag(layer, cfg)

        self._sync_tracks_menu_state()

    def _current_tracks_data(self, layer) -> np.ndarray:
        """Current track rows for `layer`, or an empty (0, 4) array."""
        if getattr(layer, "metadata", {}).get("is_placeholder", False):
            return np.empty((0, 4), dtype=float)
        return (
            layer.data
            if layer.data is not None
            else np.empty((0, 4), dtype=float)
        )

    def _merge_track_rows(self, old: np.ndarray, new_block: np.ndarray, ids):
        """Replace rows for `ids` in `old` with `new_block`, sorted; None if unchanged."""
        if old.size == 0:
            return self._sort_tracks_array_inplace(new_block)

        keep_mask = ~np.isin(old[:, 0], ids)
        if keep_mask.sum() == old.shape[0] and new_block.size == 0:
            return None

        updated = old[keep_mask]
        if new_block.size:
            updated = np.concatenate([updated, new_block], axis=0)
        return self._sort_tracks_array_inplace(updated)

    def _clear_placeholder_flag(self, layer, cfg) -> None:
        """Reveal a placeholder Tracks layer now that it holds real data."""
        if cfg.auto_show_on_first_data:
            layer.visible = True
        try:
            del layer.metadata["is_placeholder"]
        except (RuntimeError, AttributeError, TypeError):
            layer.metadata["is_placeholder"] = False

    def update_display_track_id(
        self, track_ids: list[int] | None = None
    ) -> None:
        """Refresh the 'Tracks' layer data for all tracks or a selected subset of IDs."""
        if getattr(self, "filtered_df", None) is not None:
            self._ensure_track_label_column()

        if track_ids is None:
            track_data = (
                self.filtered_df[
                    ["track_id", "t", "YMorphology", "XMorphology"]
                ]
                .to_numpy()
                .astype(float, copy=False)
                if getattr(self, "filtered_df", None) is not None
                else np.empty((0, 4), float)
            )
            for viewer in self.viewer_fluorescence:
                layer = self._ensure_tracks_layer(viewer)
                if track_data.size == 0:
                    continue

                arr = self._sort_tracks_array_inplace(track_data)
                arr[:, 0] = arr[:, 0].astype(np.int64, copy=False)

                layer.data = arr
                self._update_track_label_layer(viewer, self.filtered_df)
                if layer.metadata.get("is_placeholder", False):
                    if (
                        getattr(self, "tracks_cfg", None) or TracksViewConfig()
                    ).auto_show_on_first_data:
                        layer.visible = True
                    try:
                        del layer.metadata["is_placeholder"]
                    except (RuntimeError, AttributeError, TypeError):
                        layer.metadata["is_placeholder"] = False
            self._sync_tracks_menu_state()
            return

        self.update_tracks_for_ids(track_ids)
        self._sync_tracks_menu_state()

    def _ensure_track_label_column(self) -> None:
        """Ensure the filtered DataFrame has a track label column."""
        if (
            getattr(self, "filtered_df", None) is None
            or len(self.filtered_df) == 0
        ):
            return

        self.filtered_df["track_label"] = (
            self.filtered_df["track_id"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            + "_"
            + self.filtered_df["TrackNumber"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
        )

    def _ensure_track_label_layer(self, viewer):
        """Ensure a Points layer exists for custom track labels."""
        if "TrackLabels" not in viewer.layers:
            layer = viewer.add_points(
                np.empty((0, 3), dtype=float),
                name="TrackLabels",
                size=1,
                properties={"track_label": np.asarray([], dtype=str)},
                text={
                    "string": "{track_label}",
                    "size": 10,
                    "color": "white",
                },
            )

            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                layer.face_color = [0, 0, 0, 0]

            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                layer.edge_color = [0, 0, 0, 0]

            layer.editable = False

            cfg = getattr(self, "tracks_cfg", None) or TracksViewConfig()
            layer.visible = bool(cfg.show_track_ids)

        return viewer.layers["TrackLabels"]

    def _update_track_label_layer(self, viewer, df):
        """Refresh the TrackLabels points/text from df, clearing it when df is empty."""
        label_layer = self._ensure_track_label_layer(viewer)

        if df is None or len(df) == 0:
            label_layer.data = np.empty((0, 3), dtype=float)
            label_layer.properties = {"track_label": np.asarray([], dtype=str)}
            return

        label_data = (
            df[["t", "YMorphology", "XMorphology"]]
            .to_numpy()
            .astype(float, copy=False)
        )

        labels = df["track_label"].to_numpy(dtype=str)

        label_layer.data = label_data
        label_layer.properties = {"track_label": labels}
        label_layer.text = {
            "string": "{track_label}",
            "size": 10,
            "color": "white",
        }
