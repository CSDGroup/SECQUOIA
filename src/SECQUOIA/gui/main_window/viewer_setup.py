"""Napari viewer creation, layout assembly, camera linking, and minimize state."""

import contextlib
import logging
from typing import Any

import napari
import numpy as np
from napari.layers import Image
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QDockWidget,
    QGridLayout,
    QHBoxLayout,
    QSizePolicy,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from SECQUOIA.config import (
    NAPARIPARAMETERS,
    TOOLTIPSTEXT,
    TracksViewConfig,
)
from SECQUOIA.gui.main_window.key_bindings import setup_key_bindings_curation

QWIDGETSIZE_MAX = (1 << 24) - 1

LOG = logging.getLogger(__name__)


class ViewerSetup:
    """Napari viewer creation, layout, camera linking, and minimize/restore."""

    def create_napari_viewers(self) -> None:
        """Create two Napari viewers and reuse them if they already exist."""
        self.viewer_fluorescence = []
        self.viewer_fluorescence_windows = []

        # Viewer 1
        if getattr(self, "viewer_1", None) is None:
            v1 = napari.Viewer(show=False)
            w1 = v1.window._qt_viewer
            self.viewer_1 = v1
            self.viewer_fluorescence_window_1 = w1
            self.add_empty_segmentation_image_layer(v1)
            self._hide_napari_left_panel(v1)

        # Viewer 2
        if getattr(self, "viewer_2", None) is None:
            v2 = napari.Viewer(show=False)
            w2 = v2.window._qt_viewer
            self.viewer_2 = v2
            self.viewer_fluorescence_window_2 = w2
            self.add_empty_segmentation_image_layer(v2)
            self._hide_napari_left_panel(v2)

        # Rebuild lists
        self.viewer_fluorescence = [
            self.viewer_1,
            self.viewer_2,
        ]
        self.viewer_fluorescence_windows = [
            self.viewer_fluorescence_window_1,
            self.viewer_fluorescence_window_2,
        ]

        self._canvas_native_to_viewer = {}
        for v in self.viewer_fluorescence:
            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                self._canvas_native_to_viewer[
                    v.window._qt_viewer.canvas.native
                ] = v

        for v in self.viewer_fluorescence:
            if (
                getattr(self, "tracks_cfg", None) or TracksViewConfig()
            ).enable_tracks_layer:
                self._ensure_tracks_layer(v)
            self._disable_playback(v)

        setup_key_bindings_curation(self)
        self._sync_tracks_menu_state()
        self.install_global_hotkeys()
        self._ensure_minimize_state()

    def _disable_playback(self, viewer) -> None:
        """Turn off napari's play button on a viewer's time slider."""
        try:
            qt_dims = viewer.window._qt_viewer.dims
        except (RuntimeError, AttributeError, TypeError) as exc:
            LOG.warning("[playback] no time slider to disable: %s", exc)
            return

        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            qt_dims.stop()

        for slider_widget in list(getattr(qt_dims, "slider_widgets", [])):
            button = getattr(slider_widget, "play_button", None)
            if button is None:
                continue
            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                button.setEnabled(False)
                button.setToolTip(TOOLTIPSTEXT.PLAYBACK_DISABLED)

        if viewer in getattr(self, "_playback_guarded_viewers", []):
            return
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            viewer.dims.events.ndim.connect(
                lambda _event=None, v=viewer: self._disable_playback(v)
            )
            self._playback_guarded_viewers = [
                *getattr(self, "_playback_guarded_viewers", []),
                viewer,
            ]

    def add_viewers_to_layout2(self) -> None:
        """Rebuild the Napari area inside self.layout2 with splitter."""
        if not hasattr(self, "_napari_container"):
            self._napari_container = QWidget()
            self._napari_grid = QGridLayout(self._napari_container)
            self._napari_grid.setContentsMargins(0, 0, 0, 0)
            self._napari_grid.setHorizontalSpacing(12)
            self._napari_grid.setVerticalSpacing(12)
            self.layout2.insertWidget(0, self._napari_container)
        else:
            if hasattr(self, "viewer_wrappers"):
                for w in self.viewer_wrappers:
                    with contextlib.suppress(
                        RuntimeError, AttributeError, TypeError
                    ):
                        self._napari_grid.removeWidget(w)

        windows = []
        if (
            hasattr(self, "viewer_fluorescence_windows")
            and self.viewer_fluorescence_windows
        ):
            windows = list(self.viewer_fluorescence_windows)
        else:
            self.create_napari_viewers()
            windows = list(self.viewer_fluorescence_windows)

        if not hasattr(self, "viewer_wrappers") or not self.viewer_wrappers:
            self.viewer_wrappers = []
            for i, win in enumerate(windows):
                if win is None:
                    continue

                wrapper = QWidget()
                wrapper.setObjectName(f"napari_wrapper_{i}")
                outer_vbox = QVBoxLayout(wrapper)
                outer_vbox.setContentsMargins(0, 0, 0, 0)
                outer_vbox.setSpacing(2)
                wrapper.setSizePolicy(
                    QSizePolicy.Preferred, QSizePolicy.Expanding
                )

                header_row = QWidget()
                header_row.setObjectName("header_row")
                header_row.setSizePolicy(
                    QSizePolicy.Expanding, QSizePolicy.Fixed
                )
                hbox = QHBoxLayout(header_row)
                hbox.setContentsMargins(0, 0, 0, 0)
                hbox.setSpacing(2)
                hbox.addStretch(1)
                outer_vbox.addWidget(header_row, 0)

                wrapper._napari_win = win
                wrapper._minimized = False
                outer_vbox.addWidget(win)
                outer_vbox.setStretch(0, 0)
                outer_vbox.setStretch(1, 1)

                try:
                    viewer = self.viewer_fluorescence[i]
                    self._hide_napari_left_panel(viewer)
                    QTimer.singleShot(
                        0, lambda v=viewer: self._hide_napari_left_panel(v)
                    )
                    QTimer.singleShot(
                        50, lambda v=viewer: self._hide_napari_left_panel(v)
                    )
                except (RuntimeError, AttributeError, TypeError):
                    pass

                self.viewer_wrappers.append(wrapper)

        self.channel_combos = []
        self.mask_combos = []
        self.contrast_widgets = []

        try:
            self.update_channel_mask_dropdowns()
        except (RuntimeError, AttributeError, TypeError) as e:
            LOG.warning(
                "[add_viewers_to_layout2] update_channel_mask_dropdowns error: %s",
                e,
            )

        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            self.link_viewers_2d()

        self._napari_grid.setColumnStretch(0, 1)

        for idx, wrapper in enumerate(self.viewer_wrappers):
            self._napari_grid.addWidget(wrapper, idx, 0)
            self._napari_grid.setRowStretch(idx, 1)

        if hasattr(self, "fate_bar") and hasattr(self, "fate_bar_container"):
            self._move_fate_buttons_to_bottom_bar()

    def add_empty_segmentation_image_layer(self, viewer) -> Image:
        """Add an empty Labels layer and an empty channel Image layer to *viewer*, and return the image layer."""
        empty_labels = np.zeros((512, 512), dtype=int)

        viewer.add_labels(
            empty_labels,
            name="Segmentation1",
            opacity=NAPARIPARAMETERS.OPACITY,
        )
        image_layer = viewer.add_image(empty_labels, name="Channel w00")
        self._restrict_labels_draw_to_left_click(image_layer)
        return image_layer

    def _hide_napari_left_panel(self, obj) -> None:
        """Hide napari's left sidebar (Layers + Controls)."""
        viewer = None
        qt_main = None

        if hasattr(obj, "window") and hasattr(obj.window, "_qt_window"):
            viewer = obj
            qt_main = obj.window._qt_window

        elif hasattr(obj, "_qt_window"):
            qt_main = obj._qt_window
        elif hasattr(obj, "findChildren"):  # already a QWidget
            qt_main = obj

        if viewer is not None:
            try:
                qt_viewer = viewer.window._qt_viewer
                for name in ("dockLayerList", "dockLayerControls"):
                    dock = getattr(qt_viewer, name, None)
                    if dock and hasattr(dock, "hide"):
                        dock.hide()
            except (RuntimeError, AttributeError, TypeError):
                pass

        if qt_main is not None:
            try:
                for dock in qt_main.findChildren(QDockWidget):
                    title = (dock.windowTitle() or "").lower()
                    if any(
                        k in title for k in ("layer", "layers", "controls")
                    ):
                        dock.hide()
            except (RuntimeError, AttributeError, TypeError):
                pass

    def _hide_napari_chrome(
        self, viewer, *, menu=True, toolbars=True, status=True
    ):
        """Hide top UI chrome of a napari viewer (menu bar, toolbars, status)."""
        if viewer is None:
            return
        try:
            qt_main = viewer.window._qt_window
        except (RuntimeError, AttributeError, TypeError):
            return

        try:
            if menu and qt_main.menuBar():
                qt_main.menuBar().setVisible(False)
        except (RuntimeError, AttributeError, TypeError):
            pass

        try:
            if toolbars:
                for tb in qt_main.findChildren(QToolBar):
                    tb.setVisible(False)
        except (RuntimeError, AttributeError, TypeError):
            pass

        try:
            if status and qt_main.statusBar():
                qt_main.statusBar().setVisible(False)
        except (RuntimeError, AttributeError, TypeError):
            pass

    def unlink_viewers_2d(self) -> None:
        """Disconnect previous camera links."""
        for emitter, cb in getattr(self, "_cam_link_cbs", []) or []:
            try:
                cbs: Any = getattr(emitter, "callbacks", None)
                if cbs is not None and cb not in list(cbs):
                    continue

                emitter.disconnect(cb)
            except (TypeError, ValueError, RuntimeError):
                pass
        self._cam_link_cbs = []

    def link_viewers_2d(self):
        """Link center/zoom and rotation for two 2D napari viewers."""
        v1 = self.viewer_1
        v2 = self.viewer_2
        if v1 is None or v2 is None:
            return

        try:
            v1.dims.ndisplay = 2
            v2.dims.ndisplay = 2
        except (AttributeError, RuntimeError):
            pass

        self.unlink_viewers_2d()

        self._syncing_cam = False

        def _copy_cam(src, dst):
            """Copy camera center, zoom, and rotation from one viewer to another."""
            if self._syncing_cam:
                return
            self._syncing_cam = True
            try:
                # center/zoom
                dst.camera.center = tuple(src.camera.center)
                dst.camera.zoom = float(src.camera.zoom)
                if hasattr(src.camera, "angle") and hasattr(
                    dst.camera, "angle"
                ):
                    dst.camera.angle = float(src.camera.angle)
                elif hasattr(src.camera, "angles") and hasattr(
                    dst.camera, "angles"
                ):
                    dst.camera.angles = tuple(src.camera.angles)
            finally:
                self._syncing_cam = False

        def _on_v1(_=None):
            """Handle camera changes in viewer 1 by synchronizing viewer 2."""
            _copy_cam(v1, v2)

        def _on_v2(_=None):
            """Handle camera changes in viewer 2 by synchronizing viewer 1."""
            _copy_cam(v2, v1)

        cbs = []
        cbs += [
            (v1.camera.events.center, _on_v1),
            (v1.camera.events.zoom, _on_v1),
        ]
        if hasattr(v1.camera.events, "angle"):
            cbs.append((v1.camera.events.angle, _on_v1))
        elif hasattr(v1.camera.events, "angles"):
            cbs.append((v1.camera.events.angles, _on_v1))

        cbs += [
            (v2.camera.events.center, _on_v2),
            (v2.camera.events.zoom, _on_v2),
        ]
        if hasattr(v2.camera.events, "angle"):
            cbs.append((v2.camera.events.angle, _on_v2))
        elif hasattr(v2.camera.events, "angles"):
            cbs.append((v2.camera.events.angles, _on_v2))

        for emitter, cb in cbs:
            emitter.connect(cb)
        self._cam_link_cbs = cbs

        # Initial alignment
        _copy_cam(v1, v2)

    def _ensure_minimize_state(self):
        """Initialize the per viewer minimized-state list if it is missing or invalid."""
        if (
            not hasattr(self, "_viewer_minimized")
            or len(getattr(self, "_viewer_minimized", [])) != 2
        ):
            self._viewer_minimized = [False, False]

    def _set_viewer_minimized(self, vi: int, minimized: bool) -> None:
        """Show or hide viewer and update its wrapper height."""
        self._ensure_minimize_state()
        self._viewer_minimized[vi] = bool(minimized)

        if not hasattr(self, "viewer_wrappers") or vi >= len(
            self.viewer_wrappers
        ):
            return
        wrapper = self.viewer_wrappers[vi]

        win = getattr(wrapper, "_napari_win", None)
        header = None
        if wrapper.layout() and wrapper.layout().count():
            header = wrapper.layout().itemAt(0).widget()

        if win is not None:
            win.setVisible(not minimized)

        if header is not None:
            header_h = header.sizeHint().height() + 6
            if minimized:
                wrapper.setMaximumHeight(header_h)
                wrapper.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            else:
                wrapper.setMaximumHeight(QWIDGETSIZE_MAX)
                wrapper.setSizePolicy(
                    QSizePolicy.Preferred, QSizePolicy.Preferred
                )

        try:
            btn = getattr(wrapper, "_eye_btn", None)
            if btn is not None:
                btn.setIcon(self._eye_icon(visible=not minimized))
                btn.setChecked(not minimized)
        except (RuntimeError, AttributeError, TypeError):
            pass

        self._apply_minimize_layout_effects()

    def _apply_minimize_layout_effects(self):
        """Adjust layout after minimizing/restoring viewers."""
        if not hasattr(self, "_napari_grid") or not hasattr(
            self, "viewer_wrappers"
        ):
            return

        try:
            self._napari_grid.invalidate()
            self._napari_grid.update()
        except (RuntimeError, AttributeError, TypeError):
            pass
