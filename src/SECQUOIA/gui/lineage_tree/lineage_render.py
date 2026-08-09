"""Drawing primitives shared by the three lineage tree renderers."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from qtpy import QtGui

from SECQUOIA.gui.lineage_tree.lineage_geometry import (
    normalize,
    plan_heat_segments,
    value_range,
)
from SECQUOIA.utils.plotting import apply_time_axis

__all__ = [
    "QT_DRAW_ERRORS",
    "ChannelScale",
    "add_track_label",
    "apply_lineage_axis",
    "clear_colorbar",
    "clear_inline_multi_legend",
    "draw_division_connector",
    "draw_heat_segments",
]

QT_DRAW_ERRORS = (RuntimeError, AttributeError, TypeError)


def _rgba(color: QtGui.QColor) -> tuple[int, int, int, int]:
    """``QColor`` -> opaque RGBA tuple."""
    return (color.red(), color.green(), color.blue(), 255)


@dataclass(frozen=True)
class ChannelScale:
    """Maps one feature channel's values onto a low -> high colour ramp."""

    vmin: float
    vmax: float
    cmap: object
    width: int

    @classmethod
    def from_values(
        cls,
        values,
        low_color: QtGui.QColor,
        high_color: QtGui.QColor,
        width: int,
    ) -> ChannelScale:
        """Build a scale spanning the finite range of ``values``."""
        vmin, vmax = value_range(values)
        cmap = pg.ColorMap(
            pos=[0.0, 1.0], color=[_rgba(low_color), _rgba(high_color)]
        )
        return cls(vmin=vmin, vmax=vmax, cmap=cmap, width=width)

    def pen_for(self, value: float):
        """Return the pen that encodes ``value`` on this scale."""
        qcol = self.cmap.map(normalize(value, self.vmin, self.vmax), "qcolor")
        return pg.mkPen(qcol, width=self.width, cap="flat", join="miter")


def draw_heat_segments(
    plot: pg.PlotItem,
    value_map: Mapping[float, float],
    scale: ChannelScale,
    t0_raw: float,
    t1_raw: float,
    y: float | None,
    xmap: Callable[[float], float] | None = None,
    pad: float = 0.0,
) -> int:
    """Draw one track's heat strip over ``[t0_raw, t1_raw]`` at height ``y``."""
    if y is None:
        return 0
    segments = plan_heat_segments(value_map, t0_raw, t1_raw, xmap, pad)
    for seg in segments:
        plot.plot([seg.x0, seg.x1], [y, y], pen=scale.pen_for(seg.value))
    return len(segments)


def draw_division_connector(
    plot: pg.PlotItem,
    x: float,
    y_parent: float,
    y_child: float,
    pen,
    z: int | None = None,
):
    """Draw the vertical line joining a parent to one of its children."""
    item = plot.plot([x, x], [y_parent, y_child], pen=pen)
    if z is not None:
        with contextlib.suppress(*QT_DRAW_ERRORS):
            item.setZValue(z)
    return item


def add_track_label(
    plot: pg.PlotItem,
    text: str,
    x_left: float,
    x_right: float,
    y: float | None,
    *,
    color,
    font: QtGui.QFont | None = None,
    dy: float = 0.0,
    z: int = 10,
):
    """Centre a text label above the span ``[x_left, x_right]``."""
    if (
        y is None
        or not np.isfinite(x_left)
        or not np.isfinite(x_right)
        or x_right <= x_left
    ):
        return None

    txt = pg.TextItem(text=text, color=color, anchor=(0.5, 0.0))
    if font is not None:
        with contextlib.suppress(*QT_DRAW_ERRORS):
            txt.setFont(font)
    txt.setPos(0.5 * (float(x_left) + float(x_right)), float(y) + dy)
    txt.setZValue(z)
    plot.addItem(txt)
    return txt


def apply_lineage_axis(
    plot: pg.PlotItem,
    xmax: float,
    y_min: float,
    y_max: float,
    main_window=None,
) -> None:
    """Apply the shared lineage time axis."""
    axis_font_size = getattr(main_window, "_plot_params", {}).get(
        "axis_font_size", 10
    )
    apply_time_axis(
        plot,
        xmax,
        baseline_width=3,
        left_margin_px=1,
        right_pad=0.5,
        y_min=y_min,
        y_max=y_max,
        left_axis_width_px=getattr(main_window, "_shared_left_axis_width", 28),
        bottom_axis_height_px=max(24, int(axis_font_size * 2.6)),
        show_left_axis=False,
        tick_font=QtGui.QFont("Arial", axis_font_size),
    )


def resolve_xmax(xmax_override, mapped_ends) -> float:
    """Pick the axis maximum: an explicit override, else the last mapped end."""
    import numbers

    if isinstance(xmax_override, numbers.Real) and np.isfinite(xmax_override):
        return float(xmax_override)
    return max(list(mapped_ends) or [1.0])


def y_bounds(
    y_map: Mapping[int, float], pad: float = 0.6
) -> tuple[float, float]:
    """Padded ``(y_min, y_max)`` covering every track row."""
    ys = list(y_map.values()) or [0.0]
    return min(ys) - pad, max(ys) + pad


def clear_inline_multi_legend(plot: pg.PlotItem) -> None:
    """Remove the inline legend used by multi-channel heatmap mode."""
    items = getattr(plot, "_multi_legend_items", None)
    if items:
        for item in items:
            with contextlib.suppress(*QT_DRAW_ERRORS, ValueError):
                plot.removeItem(item)
    plot._multi_legend_items = []


def clear_colorbar(plot: pg.PlotItem) -> None:
    """Remove any ``pg.ColorBarItem`` previously added by heatmap mode."""
    old_bar = getattr(plot, "_heatbar", None)
    if old_bar is not None:
        with contextlib.suppress(*QT_DRAW_ERRORS, ValueError):
            plot.layout.removeItem(old_bar)
        with contextlib.suppress(*QT_DRAW_ERRORS, ValueError):
            if old_bar.scene() is not None:
                old_bar.scene().removeItem(old_bar)
    plot._heatbar = None
