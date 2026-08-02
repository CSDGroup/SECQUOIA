"""Draws the tile and the mask on top of it.

Two images share one view: the grey image below and the mask above. Hiding the
mask or changing its opacity is then just a setting on the top image, with no
pixel work and no re-read. Mixing the mask into the image data instead would
mean redrawing everything each time.

Tiles are placed where they really are in the frame, not at the origin, so the
view works in the same pixel coordinates as the Napari canvas. That is what
makes following the camera a direct copy instead of an offset calculation.

One thing to watch: `axisOrder` is set on each image, not globally. The global
`pg.setConfigOption('imageAxisOrder', ...)` would reach every other pyqtgraph
widget in the program, including the lineage tree and the dynamics plots, and
quietly change how they read coordinates.
"""

from __future__ import annotations

import logging

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import QRectF, Qt, Signal
from qtpy.QtWidgets import QVBoxLayout, QWidget

from SECQUOIA.gui.cell_inspector.geometry import Rect

LOG = logging.getLogger(__name__)

PLACEHOLDER_COLOR = "#8a8a8a"


class InspectorView(QWidget):
    """Shows one tile, with the mask drawn over it.

    The black and white sliders are not here but in `LevelsBar`, so the
    inspector can use the same pair of sliders as the Napari viewers.
    """

    #: Right-click on the tile, carrying the position in global coordinates.
    context_menu_requested = Signal(object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._graphics = pg.GraphicsLayoutWidget()
        self._view_box = self._graphics.addViewBox()
        self._view_box.setAspectLocked(True)
        self._view_box.invertY(True)  # row 0 at the top, as in the image
        self._view_box.setMenuEnabled(False)
        self._view_box.setMouseEnabled(False, False)

        # Axis order per image, not globally. See the note at the top.
        self._image_item = pg.ImageItem(axisOrder="row-major")
        self._overlay_item = pg.ImageItem(axisOrder="row-major")
        self._overlay_item.setZValue(1)
        self._view_box.addItem(self._image_item)
        self._view_box.addItem(self._overlay_item)

        self._placeholder = pg.TextItem(
            color=PLACEHOLDER_COLOR, anchor=(0.5, 0.5)
        )
        self._placeholder.setZValue(2)
        self._view_box.addItem(self._placeholder)
        self._placeholder.hide()

        # pyqtgraph's own right-click menu is switched off above, so the
        # click is free to open the inspector's add-a-view menu.
        self._graphics.setContextMenuPolicy(Qt.CustomContextMenu)
        self._graphics.customContextMenuRequested.connect(
            lambda pos: self.context_menu_requested.emit(
                self._graphics.mapToGlobal(pos)
            )
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._graphics, 1)

        self._has_tile = False
        self._tile_range: tuple[float, float] | None = None

    # -- accessors ---------------------------------------------------------

    @property
    def view_box(self) -> pg.ViewBox:
        """The view the camera moves."""
        return self._view_box

    @property
    def image_item(self) -> pg.ImageItem:
        """The grey image the black and white points apply to."""
        return self._image_item

    # -- content -----------------------------------------------------------

    def set_tile(
        self,
        image: np.ndarray,
        placement: Rect,
        overlay: np.ndarray | None,
        auto_levels: bool,
    ) -> None:
        """Show a tile where it really sits in the frame, with its mask.

        `auto_levels` refits the black and white points to this tile. It is
        normally off: refitting every frame means brightness cannot be
        compared between time points, which is worse than a dark tile.
        """
        self._placeholder.hide()
        self._image_item.show()
        self._image_item.setImage(image, autoLevels=False)
        self._image_item.setRect(
            QRectF(
                float(placement.left),
                float(placement.top),
                float(placement.width),
                float(placement.height),
            )
        )

        self._tile_range = self._range_of(image)
        if auto_levels or not self._has_tile:
            self._apply_auto_levels(image)
        self._has_tile = True

        self.set_overlay(overlay, placement)

    def set_overlay(self, overlay: np.ndarray | None, placement: Rect) -> None:
        """Replace the mask on top, or clear it when there is none."""
        if overlay is None:
            self._overlay_item.clear()
            return
        self._overlay_item.setImage(overlay, autoLevels=False)
        self._overlay_item.setRect(
            QRectF(
                float(placement.left),
                float(placement.top),
                float(placement.width),
                float(placement.height),
            )
        )

    def show_placeholder(self, message: str) -> None:
        """Hide the tile and say why there is nothing to show."""
        self._image_item.clear()
        self._overlay_item.clear()
        self._placeholder.setText(message)
        self._placeholder.show()
        centre = self._view_box.viewRect().center()
        self._placeholder.setPos(centre.x(), centre.y())

    # -- overlay appearance ------------------------------------------------

    def set_overlay_visible(self, visible: bool) -> None:
        """Show or hide the mask without touching the image."""
        self._overlay_item.setVisible(bool(visible))

    def set_overlay_opacity(self, opacity: float) -> None:
        """Set how see-through the mask is. Costs no pixel work."""
        self._overlay_item.setOpacity(float(np.clip(opacity, 0.0, 1.0)))

    # -- levels ------------------------------------------------------------

    def levels(self) -> tuple[float, float] | None:
        """The black and white points, as real intensity values."""
        values = self._image_item.getLevels()
        if values is None:
            return None
        return float(values[0]), float(values[1])

    def set_levels(self, low: float, high: float) -> None:
        """Apply new black and white points without re-reading the tile."""
        if high <= low:
            high = low + 1.0
        self._image_item.setLevels((float(low), float(high)))

    def apply_levels_from_bar(self, low: float, high: float) -> None:
        """Apply values that came from the sliders."""
        self.set_levels(low, high)

    @staticmethod
    def _range_of(image: np.ndarray) -> tuple[float, float] | None:
        """The lowest and highest value in an image, ignoring NaNs."""
        finite = (
            image[np.isfinite(image)] if image.dtype.kind == "f" else image
        )
        if finite.size == 0:
            return None
        low, high = float(np.min(finite)), float(np.max(finite))
        return low, high if high > low else low + 1.0

    def _apply_auto_levels(self, image: np.ndarray) -> None:
        """Fit the black and white points to the tile on screen."""
        span = self._range_of(image)
        if span is not None:
            self.set_levels(*span)

    def tile_range(self) -> tuple[float, float] | None:
        """The value range of the tile on screen, used to size the sliders."""
        return self._tile_range

    # -- camera ------------------------------------------------------------

    def set_interactive(self, interactive: bool) -> None:
        """Allow or block panning and zooming with the mouse."""
        self._view_box.setMouseEnabled(bool(interactive), bool(interactive))

    def set_view_region(
        self, center_x: float, center_y: float, width: float, height: float
    ) -> None:
        """Look at a region of the frame, given in image pixels."""
        half_w, half_h = max(width, 1.0) / 2.0, max(height, 1.0) / 2.0
        self._view_box.setRange(
            xRange=(center_x - half_w, center_x + half_w),
            yRange=(center_y - half_h, center_y + half_h),
            padding=0,
            update=True,
        )
