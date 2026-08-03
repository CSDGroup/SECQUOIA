"""The black and white point sliders."""

from __future__ import annotations

import logging

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QWidget,
)

from SECQUOIA.config import TOOLTIPSTEXT
from SECQUOIA.gui.main_window.viewer_contrast import (
    _SLIDER_STEPS,
    ViewerContrast,
    _to_real,
    _to_slider,
)

LOG = logging.getLogger(__name__)

SLIDER_WIDTH = 56
RAIL_MIN_HEIGHT = 14
DECIMALS_BELOW_SPAN = 20.0


def format_level(value: float, span: float) -> str:
    """Format one value for the number shown next to the sliders."""
    return f"{value:.1f}" if span < DECIMALS_BELOW_SPAN else f"{value:.0f}"


class LevelsBar(QWidget):
    """Two half sliders over one intensity range: black point and white point."""

    levels_changed = Signal(float, float)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        rail = QWidget(self)
        rail_layout = QHBoxLayout(rail)
        rail_layout.setContentsMargins(0, 0, 0, 0)
        rail_layout.setSpacing(0)
        rail.setMinimumHeight(RAIL_MIN_HEIGHT)

        self.black_slider = QSlider(Qt.Horizontal, rail)
        self.black_slider.setToolTip(TOOLTIPSTEXT.SLEFT)
        self.white_slider = QSlider(Qt.Horizontal, rail)
        self.white_slider.setToolTip(TOOLTIPSTEXT.RRIGHT)

        for slider in (self.black_slider, self.white_slider):
            ViewerContrast._style_half_slider(
                None, slider, groove_h=2, handle_w=8
            )
            slider.setMinimumWidth(SLIDER_WIDTH)
            slider.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
            rail_layout.addWidget(slider, 0)

        outer.addWidget(rail, 0)

        self.readout = QLabel()
        self.readout.setToolTip(TOOLTIPSTEXT.INSPECTOR_LEVEL_READOUT)
        self.readout.setStyleSheet("QLabel{font-size:9pt;}")
        self.readout.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        outer.addWidget(self.readout, 0)
        outer.addStretch(1)

        self._range = (0.0, 1.0)
        self._updating = False

        self.black_slider.valueChanged.connect(
            lambda _value: self._on_slider_moved("black")
        )
        self.white_slider.valueChanged.connect(
            lambda _value: self._on_slider_moved("white")
        )

        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)

    def set_range(self, low: float, high: float) -> None:
        """Set the intensity range the sliders cover, keeping the current values."""
        if high <= low:
            high = low + 1.0
        current = self.levels()
        self._range = (float(low), float(high))
        if current is not None:
            self.set_levels(*current)
        else:
            self._update_readout()

    @property
    def range(self) -> tuple[float, float]:
        """The intensity range."""
        return self._range

    def levels(self) -> tuple[float, float] | None:
        """The black and white points values."""
        low, high = self._range
        return (
            float(_to_real(self.black_slider.value(), low, high)),
            float(_to_real(self.white_slider.value(), low, high)),
        )

    def set_levels(self, low: float, high: float) -> None:
        range_low, range_high = self._range
        black = _to_slider(low, range_low, range_high)
        white = _to_slider(high, range_low, range_high)
        if white <= black:
            white = min(_SLIDER_STEPS, black + 1)
            black = max(0, white - 1)

        self._updating = True
        try:
            self.black_slider.setValue(black)
            self.white_slider.setValue(white)
        finally:
            self._updating = False
        self._update_readout()

    def _on_slider_moved(self, which: str) -> None:
        """Keep black points value below white point values."""
        if self._updating:
            return

        black = self.black_slider.value()
        white = self.white_slider.value()
        if black >= white:
            self._updating = True
            try:
                if which == "black":
                    white = min(_SLIDER_STEPS, black + 1)
                    self.white_slider.setValue(white)
                    if white <= black:
                        black = max(0, white - 1)
                        self.black_slider.setValue(black)
                else:
                    black = max(0, white - 1)
                    self.black_slider.setValue(black)
                    if white <= black:
                        white = min(_SLIDER_STEPS, black + 1)
                        self.white_slider.setValue(white)
            finally:
                self._updating = False

        self._update_readout()
        levels = self.levels()
        if levels is not None:
            self.levels_changed.emit(*levels)

    def _update_readout(self) -> None:
        """Update the numbers shown next to the sliders."""
        low, high = self.levels()
        span = self._range[1] - self._range[0]
        self.readout.setText(
            f"{format_level(low, span)} – {format_level(high, span)}"
        )

    def set_sliders_enabled(self, enabled: bool) -> None:
        """Grey the sliders out when there is no image to use them on."""
        self.black_slider.setEnabled(bool(enabled))
        self.white_slider.setEnabled(bool(enabled))
