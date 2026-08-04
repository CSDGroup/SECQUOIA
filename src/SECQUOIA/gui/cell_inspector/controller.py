"""Connects the main window to the inspector window."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field

from qtpy.QtCore import QObject, QThreadPool, QTimer

from SECQUOIA.gui.cell_inspector.camera_link import NapariCameraLink
from SECQUOIA.gui.cell_inspector.frame_source import (
    CropRequest,
    CropResult,
    FrameSource,
)
from SECQUOIA.gui.cell_inspector.overlay import (
    composite_overlay,
    labels_colormap,
)
from SECQUOIA.gui.cell_inspector.pane import InspectorPane
from SECQUOIA.gui.cell_inspector.panel import CellInspectorPanel
from SECQUOIA.gui.cell_inspector.render_worker import CropDispatcher

LOG = logging.getLogger(__name__)

PREFETCH_RADIUS = 3
MAX_CROP_SIZE = 2048
WORKER_THREADS = 3
EDIT_COALESCE_MS = 60

NO_IMAGE_MESSAGE = "No image for this channel\nat this time point"
NO_CHANNEL_MESSAGE = "No channel loaded"
NO_VIEW_MESSAGE = "Waiting for the viewer"


@dataclass
class PaneState:
    """What the controller remembers for one view."""

    pane: InspectorPane
    dispatcher: CropDispatcher
    levels: dict[str, tuple[float, float]] = field(default_factory=dict)
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)


class CellInspectorController(QObject):
    """Keeps every view showing what the Napari viewer is showing."""

    def __init__(
        self,
        main_window,
        panel: CellInspectorPanel,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self._main_window = main_window
        self._panel = panel

        self._source = FrameSource(main_window)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(WORKER_THREADS)
        self._camera = NapariCameraLink(self)
        self._states: dict[InspectorPane, PaneState] = {}
        self._time_connection = None
        self._label_connections: list[tuple[object, object]] = []

        self._edit_timer = QTimer(self)
        self._edit_timer.setSingleShot(True)
        self._edit_timer.setInterval(EDIT_COALESCE_MS)
        self._edit_timer.timeout.connect(self._apply_label_edit)

        self._camera.camera_changed.connect(self.render)
        self._panel.closed.connect(self._on_panel_closed)
        self._panel.pane_added.connect(self._adopt_pane)
        self._panel.pane_removed.connect(self._release_pane)

        for pane in self._panel.panes:
            self._adopt_pane(pane)

    def _adopt_pane(self, pane: InspectorPane) -> None:
        """Set a new view up."""
        dispatcher = CropDispatcher(self._source, self, pool=self._pool)
        state = PaneState(pane=pane, dispatcher=dispatcher)
        self._states[pane] = state

        dispatcher.ready.connect(
            lambda result, p=pane: self._on_tile_ready(p, result)
        )
        dispatcher.unavailable.connect(
            lambda _request, p=pane: self._on_tile_unavailable(p)
        )

        pane.channel_changed.connect(lambda _c, p=pane: self.render_pane(p))
        pane.masks_changed.connect(lambda _m, p=pane: self.render_pane(p))
        pane.opacity_changed.connect(pane.view.set_overlay_opacity)

        pane.view.set_overlay_opacity(pane.current_opacity())
        self.render_pane(pane)

    def _release_pane(self, pane: InspectorPane) -> None:
        state = self._states.pop(pane, None)
        if state is not None:
            state.dispatcher.shutdown()

    def attach_viewers(self) -> None:
        """Start following the viewers."""
        self.detach_viewers()
        viewer = self._viewer()
        if viewer is not None:
            self._camera.attach(viewer)
            try:
                emitter = viewer.dims.events.current_step
                emitter.connect(self._on_time_changed)
                self._time_connection = emitter
            except (RuntimeError, AttributeError, TypeError) as exc:
                LOG.warning(
                    "cell inspector: cannot follow the time slider: %s", exc
                )

        for each in self._all_viewers():
            self._watch_label_edits(each)

    def _watch_label_edits(self, viewer) -> None:
        """Redraw when a mask is painted in the viewer."""
        try:
            layers = list(viewer.layers)
        except (RuntimeError, AttributeError, TypeError):
            return

        for layer in layers:
            if type(layer).__name__ != "Labels":
                continue
            for name in ("paint", "data"):
                emitter = getattr(getattr(layer, "events", None), name, None)
                if emitter is None:
                    continue
                with contextlib.suppress(
                    RuntimeError, AttributeError, TypeError
                ):
                    emitter.connect(self._on_labels_edited)
                    self._label_connections.append(
                        (emitter, self._on_labels_edited)
                    )

    def detach_viewers(self) -> None:
        """Stop following the camera."""
        self._camera.detach()
        if self._time_connection is not None:
            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                self._time_connection.disconnect(self._on_time_changed)
            self._time_connection = None

        for emitter, callback in self._label_connections:
            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                emitter.disconnect(callback)
        self._label_connections = []

    def _viewer(self):
        return getattr(self._main_window, "viewer_1", None)

    def _all_viewers(self) -> list:
        """Every viewer to watch for mask edits."""
        viewers = getattr(self._main_window, "viewer_fluorescence", None)
        if not viewers:
            viewers = [
                getattr(self._main_window, "viewer_1", None),
                getattr(self._main_window, "viewer_2", None),
            ]
        return [v for v in viewers if v is not None]

    def _current_time(self) -> int:
        """The time point the viewer is on."""
        viewer = self._viewer()
        try:
            return int(viewer.dims.current_step[0])
        except (RuntimeError, AttributeError, TypeError, IndexError):
            return int(
                getattr(self._main_window, "current_time_index", 0) or 0
            )

    def render(self) -> None:
        """Redraw every view."""
        for pane in list(self._states):
            self.render_pane(pane)

    def render_pane(self, pane: InspectorPane) -> None:
        """Ask for the crop the viewer is showing.

        The crop is square, sized by the larger of the camera's visible
        width and height and capped at ``MAX_CROP_SIZE``.
        """
        state = self._states.get(pane)
        if state is None or not self._panel.isVisible():
            return

        channel_id = pane.current_channel()
        if channel_id is None:
            pane.view.show_placeholder(NO_CHANNEL_MESSAGE)
            return

        camera = self._camera.state()
        if camera is None:
            pane.view.show_placeholder(NO_VIEW_MESSAGE)
            return

        size = int(
            min(
                max(camera.visible_width, camera.visible_height, 1.0),
                MAX_CROP_SIZE,
            )
        )
        request = CropRequest(
            channel_id=channel_id,
            time_index=self._current_time(),
            center=(camera.center_x, camera.center_y),
            width=size,
            height=size,
            mask_numbers=pane.current_masks(),
        )
        state.dispatcher.request(request, self._prefetch_requests(request))

    @staticmethod
    def _prefetch_requests(request: CropRequest) -> list[CropRequest]:
        """The same crop at the neighbouring time points, nearest first."""
        ahead: list[CropRequest] = []
        for offset in range(1, PREFETCH_RADIUS + 1):
            for step in (offset, -offset):
                neighbour = int(request.time_index) + step
                if neighbour < 0:
                    continue
                ahead.append(
                    CropRequest(
                        channel_id=request.channel_id,
                        time_index=neighbour,
                        center=request.center,
                        width=request.width,
                        height=request.height,
                        mask_numbers=request.mask_numbers,
                    )
                )
        return ahead

    def _on_tile_ready(self, pane: InspectorPane, result: CropResult) -> None:
        """Show a finished tile and line it up with the viewer."""
        state = self._states.get(pane)
        if state is None:
            return

        overlay = composite_overlay(
            result.labels,
            lambda number: labels_colormap(self._viewer(), number),
        )
        pane.view.set_tile(
            result.image,
            result.window.requested,
            overlay,
        )
        pane.view.set_overlay_opacity(pane.current_opacity())
        self._sync_levels_ui(state, result.request.channel_id)

        camera = self._camera.state()
        if camera is not None:
            pane.view.set_view_region(
                camera.center_x,
                camera.center_y,
                camera.visible_width,
                camera.visible_height,
            )

    def _on_tile_unavailable(self, pane: InspectorPane) -> None:
        """Show the placeholder when the frame could not be read."""
        pane.view.show_placeholder(NO_IMAGE_MESSAGE)

    def _channel_range(
        self, state: PaneState, channel_id: str
    ) -> tuple[float, float] | None:
        """The intensity range the sliders cover for a channel."""
        channel_id = str(channel_id)
        known = state.ranges.get(channel_id)
        if known is not None:
            return known

        span = self._napari_contrast_range(channel_id)
        if span is None:
            span = state.pane.view.tile_range()
        if span is None:
            return None

        low, high = float(span[0]), float(span[1])
        if high <= low:
            high = low + 1.0
        state.ranges[channel_id] = (low, high)
        return state.ranges[channel_id]

    def _napari_contrast_range(
        self, channel_id: str
    ) -> tuple[float, float] | None:
        """Read the full intensity range off the viewer's layer for this channel."""
        viewer = self._viewer()
        main_window = self._main_window
        if viewer is None:
            return None
        try:
            name = main_window._ch_layer_name_for(channel_id)
            layer = main_window._find_layer_by_name(viewer, name)
            if layer is None:
                return None
            low, high = (float(v) for v in layer.contrast_limits_range)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return None
        return (low, high) if high > low else None

    def _sync_levels_ui(self, state: PaneState, channel_id: str) -> None:
        """Set the sliders to this channel's range."""
        channel_id = str(channel_id)
        span = self._channel_range(state, channel_id)
        if span is None:
            return

        bar = state.pane.levels_bar
        bar.set_range(*span)
        bar.set_sliders_enabled(True)
        self._watch_levels(state)

        remembered = state.levels.get(channel_id)
        if remembered is None:
            remembered = self._napari_contrast_limits(channel_id) or span
            state.levels[channel_id] = remembered

        state.pane.view.set_levels(*remembered)
        bar.set_levels(*remembered)

    def _napari_contrast_limits(
        self, channel_id: str
    ) -> tuple[float, float] | None:
        """The black and white points the viewer is using for this channel."""
        viewer = self._viewer()
        if viewer is None:
            return None
        try:
            name = self._main_window._ch_layer_name_for(channel_id)
            layer = self._main_window._find_layer_by_name(viewer, name)
            if layer is None:
                return None
            low, high = (float(v) for v in layer.contrast_limits)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return None
        return (low, high) if high > low else None

    def _watch_levels(self, state: PaneState) -> None:
        if getattr(state, "_watching", False):
            return
        state.pane.levels_bar.levels_changed.connect(
            lambda low, high, s=state: self._on_levels_changed(s, low, high)
        )
        state._watching = True

    @staticmethod
    def _on_levels_changed(state: PaneState, low: float, high: float) -> None:
        """Remember new black and white points for the channel they were set on."""
        channel_id = state.pane.current_channel()
        if channel_id is not None:
            state.levels[str(channel_id)] = (float(low), float(high))

    def _on_time_changed(self, _event=None) -> None:
        """The viewer went to another time point."""
        self.render()

    def _on_labels_edited(self, _event=None) -> None:
        """A mask was painted."""
        if not self._edit_timer.isActive():
            self._edit_timer.start()

    def _apply_label_edit(self) -> None:
        """Read the masks again after an edit."""
        self._source.invalidate()
        self.render()

    def _on_panel_closed(self) -> None:
        """Stop following the viewers while the window is closed."""
        self.detach_viewers()

    def show(self) -> None:
        """Bring the window up and get it up to date."""
        self._panel.show()
        self._panel.raise_()
        self._panel.activateWindow()
        self.refresh_sources()

    def refresh_sources(self) -> None:
        """Rebuild the controls after the channels or masks changed."""
        self._source.invalidate()
        self._panel.rebuild_all_controls()
        self.attach_viewers()
        self.render()

    def shutdown(self) -> None:
        """Stop all background work."""
        self.detach_viewers()
        for state in self._states.values():
            state.dispatcher.shutdown()
        self._states.clear()
        self._pool.clear()
        self._pool.waitForDone(2000)
