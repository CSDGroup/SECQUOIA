"""The control row are located above ."""

from __future__ import annotations

import contextlib
import logging

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from SECQUOIA.gui.cell_inspector.levels_bar import LevelsBar
from SECQUOIA.gui.cell_inspector.view import InspectorView
from SECQUOIA.gui.main_window.viewer_contrast import ViewerContrast
from SECQUOIA.gui.main_window.viewer_toolbar import ViewerToolbar

LOG = logging.getLogger(__name__)
ALL_MASKS = 0


class InspectorPane(QWidget):
    """One view with its own channel, mask and black/white settings."""

    channel_changed = Signal(str)
    masks_changed = Signal(object)
    opacity_changed = Signal(float)
    context_menu_requested = Signal(object)

    def __init__(self, widget_source, parent: QWidget | None = None):
        """`widget_source` is the main window, used only to build the widgets."""
        super().__init__(parent)
        self._widget_source = widget_source

        self.view = InspectorView()
        self.view.context_menu_requested.connect(self.context_menu_requested)

        self.levels_bar = LevelsBar()
        self.levels_bar.levels_changed.connect(self.view.apply_levels_from_bar)

        self._controls = QWidget()
        self._controls_layout = QHBoxLayout(self._controls)
        self._controls_layout.setContentsMargins(0, 0, 0, 0)
        self._controls_layout.setSpacing(4)

        self.channel_combo = None
        self.mask_combo = None
        self.opacity_slider = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(3)
        layout.addWidget(self._controls)
        layout.addWidget(self.view, 1)

        self.rebuild_controls()

    def rebuild_controls(self) -> None:
        """Rebuild the control row from the channels and masks."""
        keep_channel = self.current_channel()
        keep_mask = self.current_mask_selection()
        keep_opacity = self.current_opacity()

        while self._controls_layout.count():
            item = self._controls_layout.takeAt(0)
            widget = item.widget()
            if widget is None:
                continue
            if widget is self.levels_bar:
                continue
            widget.setParent(None)
            widget.deleteLater()

        mask_container, self.mask_combo = self._build_mask_widgets()
        opacity_container, self.opacity_slider = self._build_opacity_widgets()
        channel_container, self.channel_combo = self._build_channel_widgets()

        self._controls_layout.addWidget(mask_container)
        self._controls_layout.addSpacing(1)
        self._controls_layout.addWidget(opacity_container)
        self._controls_layout.addSpacing(1)
        self._controls_layout.addWidget(channel_container)
        self._controls_layout.addSpacing(1)
        self._controls_layout.addWidget(self.levels_bar)
        self._controls_layout.addStretch(1)

        self._restore(keep_channel, keep_mask, keep_opacity)
        self._connect_controls()

    def _build_mask_widgets(self):
        """Build the M: dropdown with the viewer's own builder."""
        mask_count = self._mask_count()
        try:
            return ViewerToolbar._build_mask_combo(
                self._widget_source, mask_count
            )
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            LOG.warning("cell inspector: mask dropdown unavailable: %s", exc)
            return QWidget(), None

    def _build_channel_widgets(self):
        """Build the CH: dropdown with the viewer's own builder."""
        try:
            return ViewerToolbar._build_channel_combo(self._widget_source)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            LOG.warning(
                "cell inspector: channel dropdown unavailable: %s", exc
            )
            return QWidget(), None

    def _build_opacity_widgets(self):
        """Build the opacity slider."""
        try:
            container = ViewerContrast._build_opacity_control(
                self._widget_source, None
            )
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            LOG.warning("cell inspector: opacity slider unavailable: %s", exc)
            return QWidget(), None

        slider = getattr(container, "_slider", None)
        if slider is not None:
            with contextlib.suppress(TypeError, RuntimeError):
                slider.valueChanged.disconnect()
        return container, slider

    def _connect_controls(self) -> None:
        """Connect the rebuilt widgets to this view's signals."""
        if self.channel_combo is not None:
            self.channel_combo.currentIndexChanged.connect(self._emit_channel)
        if self.mask_combo is not None:
            self.mask_combo.currentIndexChanged.connect(self._emit_masks)
        if self.opacity_slider is not None:
            self.opacity_slider.valueChanged.connect(
                lambda value: self.opacity_changed.emit(value / 100.0)
            )

    def _restore(
        self, channel: str | None, mask: int | None, opacity: float
    ) -> None:
        """Put back the settings a rebuild would otherwise have thrown away."""
        if self.channel_combo is not None and channel is not None:
            index = self.channel_combo.findData(channel[1:])
            if index >= 0:
                self.channel_combo.setCurrentIndex(index)
        if self.mask_combo is not None and mask is not None:
            index = self.mask_combo.findData(mask)
            if index >= 0:
                self.mask_combo.setCurrentIndex(index)
        if self.opacity_slider is not None:
            self.opacity_slider.setValue(int(round(opacity * 100)))

    def _emit_channel(self, _index: int) -> None:
        """Report the chosen channel as a full channel id."""
        channel = self.current_channel()
        if channel is not None:
            self.channel_changed.emit(channel)

    def _emit_masks(self, _index: int) -> None:
        """Report the chosen mask as the 1-based numbers to draw."""
        self.masks_changed.emit(self.current_masks())

    def _mask_count(self) -> int:
        """Loaded masks."""
        try:
            return max(0, int(getattr(self._widget_source, "n_masks", 0) or 0))
        except (TypeError, ValueError):
            return 0

    def current_channel(self) -> str | None:
        """The chosen channel id, with the w in front, or None."""
        if self.channel_combo is None:
            return None
        data = self.channel_combo.currentData()
        return None if data is None else f"w{data}"

    def current_mask_selection(self) -> int | None:
        """What the mask dropdown is set to: 0 for ALL, else the mask number."""
        if self.mask_combo is None:
            return None
        data = self.mask_combo.currentData()
        if data is None:
            return None
        try:
            return int(data)
        except (TypeError, ValueError):
            return None

    def current_masks(self) -> tuple[int, ...]:
        """The 1-based mask numbers to draw."""
        number = self.current_mask_selection()
        if number is None:
            return ()
        if number == ALL_MASKS:
            return tuple(range(1, self._mask_count() + 1))
        return (number,)

    def current_opacity(self) -> float:
        """Current opacity from 0 to 1."""
        if self.opacity_slider is None:
            return 1.0
        return float(self.opacity_slider.value()) / 100.0
