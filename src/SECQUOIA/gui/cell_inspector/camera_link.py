"""Follows a Napari viewer's camera."""

from __future__ import annotations

import contextlib
import logging

from qtpy.QtCore import QObject, QTimer, Signal

LOG = logging.getLogger(__name__)

COALESCE_MS = 16


class CameraState:
    """Where a Napari camera is looking, in image pixels."""

    __slots__ = (
        "center_x",
        "center_y",
        "zoom",
        "canvas_width",
        "canvas_height",
    )

    def __init__(
        self,
        center_x: float,
        center_y: float,
        zoom: float,
        canvas_width: float,
        canvas_height: float,
    ):
        self.center_x = center_x
        self.center_y = center_y
        self.zoom = zoom
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

    @property
    def visible_width(self) -> float:
        """How many image pixels wide the visible area is."""
        return self.canvas_width / self.zoom if self.zoom > 0 else 0.0

    @property
    def visible_height(self) -> float:
        """How many image pixels tall the visible area is."""
        return self.canvas_height / self.zoom if self.zoom > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"CameraState(center=({self.center_x:.1f}, {self.center_y:.1f}), "
            f"zoom={self.zoom:.3f})"
        )


class NapariCameraLink(QObject):
    """Watches one viewer's camera and reports when it moved."""

    camera_changed = Signal()

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._viewer = None
        self._connections: list[tuple[object, object]] = []

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(COALESCE_MS)
        self._timer.timeout.connect(self.camera_changed)

    def attach(self, viewer) -> None:
        """Start watching a viewer's camera, dropping any earlier one."""
        self.detach()
        if viewer is None:
            return
        self._viewer = viewer

        try:
            events = viewer.camera.events
        except (
            RuntimeError,
            AttributeError,
            TypeError,
            ValueError,
            IndexError,
        ) as exc:
            LOG.warning("cell inspector: no camera to follow: %s", exc)
            self._viewer = None
            return

        for name in ("center", "zoom"):
            emitter = getattr(events, name, None)
            if emitter is None:
                continue
            with contextlib.suppress(
                RuntimeError, AttributeError, TypeError, ValueError, IndexError
            ):
                emitter.connect(self._on_camera_event)
                self._connections.append((emitter, self._on_camera_event))

        if not self._connections:
            LOG.warning("cell inspector: camera exposes no center/zoom events")

    def detach(self) -> None:
        """Stop watching and disconnect everything this put in place."""
        self._timer.stop()
        for emitter, callback in self._connections:
            with contextlib.suppress(
                RuntimeError, AttributeError, TypeError, ValueError, IndexError
            ):
                emitter.disconnect(callback)
        self._connections = []
        self._viewer = None

    @property
    def is_attached(self) -> bool:
        """True if a camera is being watched."""
        return bool(self._connections)

    def _on_camera_event(self, _event=None) -> None:
        """Remember that the camera moved. The timer decides when to report it."""
        if not self._timer.isActive():
            self._timer.start()

    def state(self) -> CameraState | None:
        """Read where the camera is now, or None if it cannot be read."""
        viewer = self._viewer
        if viewer is None:
            return None
        try:
            center = tuple(viewer.camera.center)
            zoom = float(viewer.camera.zoom)
            canvas = viewer.window._qt_viewer.canvas.native
            canvas_width = float(canvas.width())
            canvas_height = float(canvas.height())
        except (
            RuntimeError,
            AttributeError,
            TypeError,
            ValueError,
            IndexError,
        ) as exc:
            LOG.debug("cell inspector: camera unreadable: %s", exc)
            return None

        if len(center) < 2 or zoom <= 0:
            return None
        return CameraState(
            center_x=float(center[-1]),
            center_y=float(center[-2]),
            zoom=zoom,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )
