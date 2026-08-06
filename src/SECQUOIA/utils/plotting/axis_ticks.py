"""Generic axis/tick formatting math, shared by every pyqtgraph plot."""

from __future__ import annotations

import contextlib
import math
import sys

import numpy as np
import pyqtgraph as pg
from qtpy.QtGui import QFont, QFontMetrics

_IS_MACOS = sys.platform == "darwin"
_AXIS_LABEL_EXTRA_PX = 8 if _IS_MACOS else 0
_PLOT_LEFT_MARGIN_PX = 6 if _IS_MACOS else 1
_AXIS_FONT_FAMILY = "Arial"
_TICK_TEXT_OFFSET_PX = 5
_MIN_TICK_STEP = 1e-6
_MAX_TICK_DECIMALS = 6
_MAX_LABEL_DECIMALS = 15
_MAX_TICKS = 12
_MIN_TICKS = 3
_Y_PAD_TOP_FRAC = 0.26
_Y_PAD_BOTTOM_FRAC = 0.06


def _choose_tick_step(
    xmax: float, nticks_target: int = 8, min_step: float = _MIN_TICK_STEP
) -> float:
    """Pick a "nice" tick step (1/2/5 x 10^k) covering a span of ``xmax``."""
    try:
        xmax = float(xmax)
    except (TypeError, ValueError):
        xmax = 0.0
    if not np.isfinite(xmax):
        xmax = 0.0
    span = max(xmax, 0.0)

    nticks_target = max(int(nticks_target), 1)
    if span <= 0:
        return min_step

    # Ideal step, snapped up to the next 1/2/5 x 10^k.
    raw = span / nticks_target
    exp = math.floor(math.log10(raw)) if raw > 0 else 0
    for mult in (1.0, 2.0, 5.0, 10.0):
        step = mult * (10.0**exp)
        if step >= raw:
            break

    return max(step, min_step)


def _decimals_for_step(step: float) -> int:
    """Digits after the point needed to write multiples of ``step`` exactly."""
    try:
        step = abs(float(step))
    except (TypeError, ValueError):
        return _MAX_TICK_DECIMALS
    if not np.isfinite(step) or step <= 0:
        return _MAX_TICK_DECIMALS
    return max(0, min(-math.floor(math.log10(step)), _MAX_LABEL_DECIMALS))


def axis_font(font_size: int) -> QFont:
    """The font the plot rows draw their tick labels with."""
    return QFont(_AXIS_FONT_FAMILY, int(font_size))


def _axis_font_metrics(font_size: int) -> QFontMetrics:
    """QFontMetrics for the axis font."""
    return QFontMetrics(axis_font(font_size))


def _left_axis_width_for_label(
    label: str, font_size: int, padding: int = _TICK_TEXT_OFFSET_PX + 3
) -> int:
    """Pixel width needed to render a specific tick label string without clipping."""
    fm = _axis_font_metrics(font_size)
    return max(
        22, fm.horizontalAdvance(label) + padding + _AXIS_LABEL_EXTRA_PX
    )


def _left_axis_width_for_ticks(ticks, font_size: int) -> int:
    """Width the left axis needs for the widest label in ``ticks``."""
    return max(
        (_left_axis_width_for_label(label, font_size) for _, label in ticks),
        default=_left_axis_width_for_label("0", font_size),
    )


_AXIS_UNITS = ((1_000_000.0, "M"), (1_000.0, "K"), (1.0, ""))


def _axis_unit(scale_hint: float) -> tuple[float, str]:
    """The ``(divisor, suffix)`` a value of this magnitude should be written in."""
    try:
        hint = abs(float(scale_hint))
    except (TypeError, ValueError):
        return 1.0, ""
    if not np.isfinite(hint):
        return 1.0, ""
    for divisor, suffix in _AXIS_UNITS:
        if hint >= divisor:
            return divisor, suffix
    return 1.0, ""


def _format_axis_value(
    v: float, step: float | None = None, unit: tuple[float, str] | None = None
) -> str:
    """Compact number formatting: 9000 -> '9K', 1500000 -> '1.5M', 0.05 -> '0.05'."""
    if v == 0:
        return "0"

    divisor, suffix = _axis_unit(abs(v)) if unit is None else unit
    scaled = v / divisor

    if step is not None and np.isfinite(step) and step > 0:
        decimals = _decimals_for_step(step / divisor)
        return f"{scaled:.{decimals}f}{suffix}"

    return f"{scaled:g}{suffix}"


def _y_view_range(
    data_min: float | None, data_max: float | None
) -> tuple[float, float]:
    """Padded (y_low, y_high) view range for the observed data extent."""

    def _f(v, fallback):
        """Coerce ``v`` to a finite float, falling back when it is not."""
        try:
            v = float(v)
        except (TypeError, ValueError):
            return fallback
        return v if np.isfinite(v) else fallback

    lo = _f(data_min, None)
    hi = _f(data_max, None)

    if lo is None and hi is None:
        return -_Y_PAD_BOTTOM_FRAC, 1.0 + _Y_PAD_TOP_FRAC
    if lo is None:
        lo = hi
    if hi is None:
        hi = lo
    if lo > hi:
        lo, hi = hi, lo

    lo = min(lo, 0.0)
    hi = max(hi, 0.0)

    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        span = max(abs(hi), 1.0)

    return lo - _Y_PAD_BOTTOM_FRAC * span, hi + _Y_PAD_TOP_FRAC * span


def _round_to_step(value: float, step: float) -> float:
    """Drop the floating point noise in ``value`` without flattening it."""
    return round(value, _decimals_for_step(step) + 1)


def _labelled_ticks(values, step: float) -> list[tuple[float, str]]:
    """Label tick ``values``, in one unit, keeping only distinct positions."""
    unit = _axis_unit(max((abs(v) for v in values), default=0.0))

    ticks: list[tuple[float, str]] = []
    seen: set[float] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ticks.append((float(value), _format_axis_value(value, step, unit)))

    return ticks or [(0.0, "0")]


def _choose_y_ticks(
    y_max: float, nticks_target: int = 5, y_min: float = 0.0
) -> list[tuple[float, str]]:
    """Evenly spaced y-tick positions with compact labels.

    Ticks are placed on a grid aligned to multiples of the chosen step, so the
    range may start below zero and 0 is always included when it is in range.
    """
    try:
        y_max = float(y_max)
    except (TypeError, ValueError):
        y_max = 1.0
    try:
        y_min = float(y_min)
    except (TypeError, ValueError):
        y_min = 0.0

    if not np.isfinite(y_max):
        y_max = 1.0
    if not np.isfinite(y_min):
        y_min = 0.0
    if y_min > y_max:
        y_min, y_max = y_max, y_min

    span = y_max - y_min
    if span <= 0:
        pad = max(abs(y_max) * 0.5, 0.5)
        y_min, y_max = y_max - pad, y_max + pad
        span = y_max - y_min

    min_step = min(_MIN_TICK_STEP, span / 100.0)
    step = _choose_tick_step(span, nticks_target, min_step=min_step)
    if not np.isfinite(step) or step <= 0:
        step = min_step

    def _grid(step_v):
        """First/last grid index of ``step_v`` fully inside [y_min, y_max]."""
        i0 = int(math.ceil(y_min / step_v - 1e-9))
        i1 = int(math.floor(y_max / step_v + 1e-9))
        return i0, max(i1, i0)

    i0, i1 = _grid(step)

    if (i1 - i0) > _MAX_TICKS:
        factor = int(math.ceil((i1 - i0) / float(_MAX_TICKS)))
        step = max(step * max(factor, 1), min_step)
        i0, i1 = _grid(step)

    values = [_round_to_step(i * step, step) for i in range(i0, i1 + 1)]
    if len(values) < _MIN_TICKS:
        n = _MIN_TICKS - 1
        step = span / n
        values = [_round_to_step(y_min + i * step, step) for i in range(n + 1)]

    values = [min(max(v, y_min), y_max) for v in values]

    return _labelled_ticks(values, step)


def _shared_left_axis_width(
    global_ymin: float, global_ymax: float, axis_fs: int
) -> int:
    """Left-axis width for the widest label the global y-range produces."""
    lo, hi = _y_view_range(global_ymin, global_ymax)
    return _left_axis_width_for_ticks(_choose_y_ticks(hi, y_min=lo), axis_fs)


def _set_left_axis_ticks(plot_item, ticks, axis_fs: int, min_width) -> None:
    """Apply ``ticks`` to the left axis, widening it so none get dropped."""
    ax_left = plot_item.getAxis("left")
    if ax_left is None:
        return
    ax_left.setTicks([ticks])
    if min_width is None:
        return
    ax_left.setWidth(
        max(int(min_width), _left_axis_width_for_ticks(ticks, axis_fs))
    )


def apply_y_range_and_ticks(
    plot_item, y_lo: float, y_hi: float, axis_fs: int, min_width=None
) -> int:
    """Set a plot's Y range and ticks; return the left-axis width they need."""
    ticks = _choose_y_ticks(y_hi, y_min=y_lo)
    width = _left_axis_width_for_ticks(ticks, axis_fs)

    with contextlib.suppress(
        RuntimeError, AttributeError, TypeError, ValueError
    ):
        plot_item.vb.setYRange(y_lo, y_hi, padding=0.0)
    with contextlib.suppress(
        RuntimeError, AttributeError, TypeError, ValueError
    ):
        _set_left_axis_ticks(plot_item, ticks, axis_fs, min_width)

    return width


def sync_row_y_ticks(plot_item, axis_fs: int = 10, min_width=None) -> None:
    """Recompute and reapply left-axis ticks for the plot's current Y range."""
    with contextlib.suppress(
        RuntimeError, AttributeError, TypeError, ValueError
    ):
        y_lo, y_hi = plot_item.vb.viewRange()[1]
        _set_left_axis_ticks(
            plot_item, _choose_y_ticks(y_hi, y_min=y_lo), axis_fs, min_width
        )


def _plotted_y_extent(plot_item) -> tuple[float | None, float | None]:
    """``(min, max)`` finite y across the plot's data items, or ``(None, None)``."""
    lo, hi = np.inf, -np.inf
    try:
        items = list(plot_item.listDataItems())
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return None, None

    for item in items:
        with contextlib.suppress(
            RuntimeError, AttributeError, TypeError, ValueError, IndexError
        ):
            y = np.asarray(item.getData()[1], dtype=float)
            y = y[np.isfinite(y)]
            if y.size == 0:
                continue
            lo = min(lo, float(y.min()))
            hi = max(hi, float(y.max()))

    if not np.isfinite(lo) or not np.isfinite(hi):
        return None, None
    return lo, hi


def fit_row_y_axis(plot_item, axis_fs: int = 10, min_width=None) -> int:
    """Rescale one row's Y axis to the data it currently shows."""
    data_min, data_max = _plotted_y_extent(plot_item)
    y_lo, y_hi = _y_view_range(data_min, data_max)
    return apply_y_range_and_ticks(plot_item, y_lo, y_hi, axis_fs, min_width)


def apply_time_axis(
    plot_item: pg.PlotItem,
    xmax: float,
    *,
    baseline_width: int = 3,
    left_margin_px: int = _PLOT_LEFT_MARGIN_PX,
    right_pad: float = 0.2,
    y_min: float | None = None,
    y_max: float | None = None,
    left_axis_width_px: int = 28,
    bottom_axis_height_px: int = 18,
    show_left_axis: bool = True,
    tick_font: QFont | None = None,
):
    """Uniform time axis: thick baseline, majors only,
    start at 0 with small right padding, fixed left/bottom axis sizes so
    different widgets align horizontally/vertically.
    """
    plot_item.layout.setContentsMargins(left_margin_px, 6, 6, 6)
    ax_l = plot_item.getAxis("left")
    ax_b = plot_item.getAxis("bottom")

    if ax_l is not None:
        ax_l.setWidth(left_axis_width_px)
        if show_left_axis:
            ax_l.setStyle(showValues=True)
            ax_l.setPen(pg.mkPen("w", width=3))
            ax_l.setTextPen(pg.mkPen("w"))
            ax_l.setTickPen(pg.mkPen(None))
            if tick_font is not None:
                ax_l.setStyle(tickFont=tick_font)
        else:
            ax_l.setStyle(showValues=False)
            ax_l.setPen(None)

    if ax_b is not None:
        ax_b.setHeight(bottom_axis_height_px)
        ax_b.setPen(pg.mkPen("w", width=baseline_width))
        ax_b.setTextPen(pg.mkPen("w"))
        ax_b.setStyle(tickLength=-6)
        if tick_font is not None:
            ax_b.setStyle(tickFont=tick_font)

        step = _choose_tick_step(xmax)
        majors = [
            (v, f"{v:g}") for v in np.arange(0, float(xmax) + 1e-9, step)
        ]
        ax_b.setTicks([majors])

    vb = plot_item.getViewBox()
    x_left, x_right = 0.0, float(xmax) + right_pad
    if y_min is None or y_max is None:
        vb.setXRange(x_left, x_right, padding=0)
    else:
        vb.setRange(xRange=(x_left, x_right), yRange=(y_min, y_max), padding=0)
