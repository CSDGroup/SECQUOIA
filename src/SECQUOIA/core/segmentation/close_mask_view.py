"""Show a close-mask case in napari.

- the assigned mask in red and the candidate in blue, every other label
  hidden.
- a yellow line between the two centres. napari cannot draw a line without a
  layer, so each viewer gets one ``CloseMaskLink`` Shapes layer. It is created
  the first time it is needed and from then on only has its contents replaced:
  adding and removing layers is what made this slow. The line is only drawn
  for masks that are highlighted;
"""

from __future__ import annotations

import contextlib
import logging
import weakref
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from napari.utils.colormaps import DirectLabelColormap

from SECQUOIA.config import PLOTPARAMETERS
from SECQUOIA.core.outlier_detection.close_masks import (
    CLOSE_MASK_FLAG,
    FLAG_FLAGGED,
    close_mask_row_key,
    is_pinned_close_mask_row,
    unpin_close_mask_row,
)
from SECQUOIA.core.quantification.naming import (
    alt_distance_column,
    alt_label_column,
    centroid_columns,
    label_column,
)
from SECQUOIA.core.quantification.remeasure_object import labels_2d
from SECQUOIA.core.segmentation.mask_selection import (
    _resolve_t_index,
    current_selection_row,
    get_mask_indices,
)

LOG = logging.getLogger(__name__)

__all__ = [
    "ASSIGNED_TOOLTIP",
    "LINK_LAYER",
    "MaskPair",
    "assigned_mask_tooltip",
    "clear_close_mask_view",
    "close_mask_pairs",
    "label_centroid",
    "refresh_close_mask_view",
    "refresh_close_mask_view_at_current",
]

LINK_LAYER = "CloseMaskLink"
ASSIGNED_TOOLTIP = "Assigned mask"
_VIEWER_ATTRS = ("viewer_1", "viewer_2")

_SAVED_ATTR = (
    "_close_mask_view_saved"  # (viewer, layer name) -> (layer, colormap)
)
_PAIRS_ATTR = "_close_mask_view_pairs"  # (viewer, layer name) -> MaskPair
_TRANSPARENT = (0.0, 0.0, 0.0, 0.0)
_SEARCH_MARGIN_PX = 128

_WIRED_LAYERS: weakref.WeakSet = weakref.WeakSet()
_VISIBILITY_WIRED: weakref.WeakSet = weakref.WeakSet()
_RECOLORING: set[int] = set()


@dataclass(frozen=True)
class MaskPair:
    """The assigned and the candidate label of one mask at one time point."""

    mask_idx: int
    assigned: int
    candidate: int
    distance: float


@dataclass(frozen=True)
class PairGeometry:
    """The two ends of the line of a pair, as ``(y, x)`` or None if unknown.

    The line starts at the assigned mask (at the tracking point when there is
    none) and ends at the candidate.
    """

    pair: MaskPair
    start_yx: tuple[float, float] | None
    candidate_yx: tuple[float, float] | None


def _rgba(color: Iterable[int]) -> tuple[float, ...]:
    """A 0-255 colour tuple as the 0-1 floats napari expects."""
    return tuple(float(c) / 255.0 for c in color)


def _finite(value) -> float | None:
    """`value` as a float, or None if it is missing or not finite."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _int_label(value) -> int:
    """A label id read from a row, with 0 for missing values."""
    number = _finite(value)
    return int(number) if number is not None else 0


def close_mask_pairs(
    row: pd.Series,
    masks: Iterable[int],
    threshold_px: float,
) -> list[MaskPair]:
    """The masks of `row` whose second candidate is within `threshold_px`."""
    pairs: list[MaskPair] = []
    for mask_idx in masks:
        candidate = _int_label(row.get(alt_label_column(mask_idx)))
        distance = _finite(row.get(alt_distance_column(mask_idx)))
        if candidate <= 0 or distance is None or distance > threshold_px:
            continue
        pairs.append(
            MaskPair(
                mask_idx=int(mask_idx),
                assigned=_int_label(row.get(label_column(mask_idx))),
                candidate=candidate,
                distance=distance,
            )
        )
    return pairs


def _centroid_of(
    ys: np.ndarray, xs: np.ndarray, y0: int = 0, x0: int = 0
) -> tuple[float, float] | None:
    """Rounded ``(y, x)`` centre of pixel coordinates, offset by the window origin."""
    if ys.size == 0:
        return None
    return float(np.rint(ys.mean() + y0)), float(np.rint(xs.mean() + x0))


def _centroid_in_window(
    labels2d: np.ndarray, label: int, near: tuple[float, float], radius: float
) -> tuple[float, float] | None:
    """Centre of `label` from a window around `near`, or None if that is not enough."""
    height, width = labels2d.shape[-2:]
    y0 = max(0, int(near[0] - radius))
    y1 = min(height, int(near[0] + radius) + 1)
    x0 = max(0, int(near[1] - radius))
    x1 = min(width, int(near[1] + radius) + 1)
    window = labels2d[y0:y1, x0:x1]
    ys, xs = np.nonzero(window == label)
    if ys.size == 0:
        return None

    cut_off = (
        (y0 > 0 and ys.min() == 0)
        or (y1 < height and ys.max() == window.shape[0] - 1)
        or (x0 > 0 and xs.min() == 0)
        or (x1 < width and xs.max() == window.shape[1] - 1)
    )
    return None if cut_off else _centroid_of(ys, xs, y0, x0)


def label_centroid(
    labels2d: np.ndarray | None,
    label: int,
    *,
    near: tuple[float, float] | None = None,
    radius: float | None = None,
) -> tuple[float, float] | None:
    """The ``(y, x)`` centre of `label`, rounded like quantification, or None.

    With `near` and `radius` only that window is read first, which is much
    faster on a large frame; the result is the same as reading all of it.
    """
    if labels2d is None or label <= 0:
        return None
    if near is not None and radius is not None:
        found = _centroid_in_window(labels2d, label, near, radius)
        if found is not None:
            return found
    return _centroid_of(*np.nonzero(labels2d == label))


def _flagged_pairs(main_window, row: pd.Series) -> list[MaskPair]:
    """The pairs to show for `row`: for a flagged row, or the pinned reviewed one."""
    if row is None:
        return []
    if row.get(
        CLOSE_MASK_FLAG
    ) != FLAG_FLAGGED and not is_pinned_close_mask_row(main_window, row):
        return []

    threshold = getattr(main_window, "close_mask_threshold", None)
    if threshold is None:
        threshold = getattr(main_window, "threshold", 0)
    masks = getattr(main_window, "close_mask_masks", None)
    if not masks:
        masks = get_mask_indices(main_window)
    return close_mask_pairs(row, masks, float(threshold))


def _store(main_window, attr: str) -> dict:
    """The named bookkeeping dict of `main_window`, created on first use."""
    if not isinstance(getattr(main_window, attr, None), dict):
        setattr(main_window, attr, {})
    return getattr(main_window, attr)


def _layer_named(viewer, name: str):
    """The layer called `name` in `viewer`, or None."""
    with contextlib.suppress(
        RuntimeError, AttributeError, TypeError, KeyError
    ):
        if name in viewer.layers:
            return viewer.layers[name]
    return None


def _recolor(layer, pair: MaskPair) -> None:
    """Colour the pair, hide every other label, and keep a brush label visible."""
    colors: dict = {None: _TRANSPARENT}
    if pair.assigned > 0:
        colors[pair.assigned] = _rgba(PLOTPARAMETERS.RED)
    colors[pair.candidate] = _rgba(PLOTPARAMETERS.BLUE)

    selected = _int_label(getattr(layer, "selected_label", 0))
    if selected > 0 and selected not in colors:
        colors[selected] = _rgba(PLOTPARAMETERS.YELLOW)

    if id(layer) in _RECOLORING:
        return
    _RECOLORING.add(id(layer))
    try:
        layer.colormap = DirectLabelColormap(color_dict=colors)
    finally:
        _RECOLORING.discard(id(layer))


def _wire_selected_label(main_window, viewer_attr: str, layer) -> None:
    """Recolour `layer` when another label is picked, while its highlight is on."""
    if layer in _WIRED_LAYERS:
        return
    key = (viewer_attr, layer.name)

    def on_selected_label(_event=None) -> None:
        if layer.show_selected_label:
            return
        pair = _store(main_window, _PAIRS_ATTR).get(key)
        if pair is not None:
            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                _recolor(layer, pair)

    with contextlib.suppress(RuntimeError, AttributeError, TypeError):
        layer.events.selected_label.connect(on_selected_label)
        _WIRED_LAYERS.add(layer)


def _highlight_layer(
    main_window, viewer_attr: str, layer, pair: MaskPair
) -> None:
    """Colour `pair` on `layer`, remembering its original colormap once."""
    key = (viewer_attr, layer.name)
    saved = _store(main_window, _SAVED_ATTR)
    if key not in saved or saved[key][0] is not layer:
        saved[key] = (layer, layer.colormap)
    _store(main_window, _PAIRS_ATTR)[key] = pair
    layer.show_selected_label = False
    _recolor(layer, pair)
    _wire_selected_label(main_window, viewer_attr, layer)


def _restore_layer(main_window, viewer, viewer_attr: str, name: str) -> None:
    """Put the original colormap back on `name`, if it was highlighted."""
    key = (viewer_attr, name)
    _store(main_window, _PAIRS_ATTR).pop(key, None)
    entry = _store(main_window, _SAVED_ATTR).pop(key, None)
    if entry is None:
        return
    layer, colormap = entry
    if _layer_named(viewer, name) is layer:
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            layer.colormap = colormap


def _is_shown(viewer, mask_idx: int) -> bool:
    """True if `viewer` is showing its ``Segmentation{mask_idx}`` layer."""
    layer = _layer_named(viewer, f"Segmentation{mask_idx}")
    return layer is not None and bool(layer.visible)


def _wire_visibility(main_window, layer) -> None:
    """Refresh the view when `layer` is shown or hidden, e.g. by the mask selector."""
    if layer in _VISIBILITY_WIRED:
        return

    def on_visible(_event=None) -> None:
        try:
            refresh_close_mask_view_at_current(main_window)
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            LOG.warning(
                "[close-mask view] refresh after a layer change failed: %s", e
            )

    with contextlib.suppress(RuntimeError, AttributeError, TypeError):
        layer.events.visible.connect(on_visible)
        _VISIBILITY_WIRED.add(layer)


def _tracking_point(row: pd.Series) -> tuple[float, float] | None:
    """The ``(y, x)`` tracking point of `row`, or None."""
    y, x = _finite(row.get("YMorphology")), _finite(row.get("XMorphology"))
    return (y, x) if y is not None and x is not None else None


def _assigned_centre(row: pd.Series, pair: MaskPair) -> tuple | None:
    """The ``(y, x)`` centre of the assigned mask as measured into `row`, or None."""
    if pair.assigned <= 0:
        return None
    x_col, y_col = centroid_columns(pair.mask_idx)
    x, y = _finite(row.get(x_col)), _finite(row.get(y_col))
    if x is None or y is None or (x == 0 and y == 0):
        return None
    return y, x


def _pair_geometry(
    main_window, row: pd.Series, pairs: list[MaskPair], t: int
) -> list[PairGeometry]:
    """Locate every pair once; both viewers show the same frame."""
    tracking_point = _tracking_point(row)
    geometry = []
    for pair in pairs:
        frame = None
        for attr in _VIEWER_ATTRS:
            layer = _layer_named(
                getattr(main_window, attr, None),
                f"Segmentation{pair.mask_idx}",
            )
            if layer is not None:
                frame = labels_2d(layer, t)
                break

        assigned = _assigned_centre(row, pair)
        candidate = label_centroid(
            frame,
            pair.candidate,
            near=tracking_point or assigned,
            radius=pair.distance + _SEARCH_MARGIN_PX,
        )
        geometry.append(
            PairGeometry(pair, assigned or tracking_point, candidate)
        )
    return geometry


def _add_keeping_active(viewer, add):
    """Add a layer with `add()`, leaving the previously active layer active."""
    active = getattr(viewer.layers.selection, "active", None)
    layer = add()
    if active is not None:
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            viewer.layers.selection.active = active
    return layer


def _link_lines(t: int, geometry: list[PairGeometry]) -> list[np.ndarray]:
    """One line per pair, between the centre of the assigned mask and the candidate."""
    return [
        np.array([[t, *item.start_yx], [t, *item.candidate_yx]], dtype=float)
        for item in geometry
        if item.start_yx is not None and item.candidate_yx is not None
    ]


def _show_link(viewer, t: int, geometry: list[PairGeometry]) -> None:
    """Put the line of every pair on the viewer's link layer."""
    lines = _link_lines(t, geometry)
    if not lines:
        _clear_link(viewer)
        return

    color = [_rgba(PLOTPARAMETERS.YELLOW)] * len(lines)
    layer = _layer_named(viewer, LINK_LAYER)
    if layer is None:
        _add_keeping_active(
            viewer,
            lambda: viewer.add_shapes(
                lines,
                shape_type="line",
                edge_color=color,
                edge_width=1.5,
                name=LINK_LAYER,
            ),
        )
        return

    layer.data = lines
    layer.shape_type = ["line"] * len(lines)
    layer.edge_color = color


def _clear_link(viewer) -> None:
    """Empty the link layer, keeping the layer itself."""
    layer = _layer_named(viewer, LINK_LAYER)
    if layer is not None and len(layer.data):
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            layer.data = []


def _show_in_viewer(
    main_window,
    viewer,
    viewer_attr: str,
    geometry: list[PairGeometry],
    t: int,
) -> None:
    """Highlight the pairs whose mask `viewer` shows, and draw their link."""
    shown = [
        item for item in geometry if _is_shown(viewer, item.pair.mask_idx)
    ]
    by_mask = {item.pair.mask_idx: item.pair for item in shown}

    for mask_idx in get_mask_indices(main_window):
        name = f"Segmentation{mask_idx}"
        layer = _layer_named(viewer, name)
        if layer is None:
            continue
        pair = by_mask.get(mask_idx)
        if pair is None:
            _restore_layer(main_window, viewer, viewer_attr, name)
        else:
            _highlight_layer(main_window, viewer_attr, layer, pair)

    _show_link(viewer, t, shown)


def assigned_mask_tooltip(main_window, viewer) -> str:
    """The tooltip text for the cursor of `viewer`: over the assigned mask."""
    viewer_attr = next(
        (a for a in _VIEWER_ATTRS if getattr(main_window, a, None) is viewer),
        None,
    )
    if viewer_attr is None:
        return ""

    with contextlib.suppress(RuntimeError, AttributeError, TypeError):
        cursor = viewer.cursor
        for (attr, name), pair in list(
            _store(main_window, _PAIRS_ATTR).items()
        ):
            layer = _layer_named(viewer, name)
            if attr != viewer_attr or pair.assigned <= 0 or layer is None:
                continue
            if not layer.visible:
                continue
            value = layer.get_value(
                np.asarray(cursor.position),
                view_direction=getattr(cursor, "_view_direction", None),
                dims_displayed=list(viewer.dims.displayed),
                world=True,
            )
            if value == pair.assigned:
                return ASSIGNED_TOOLTIP
    return ""


def clear_close_mask_view(main_window) -> None:
    """Restore the normal colormaps and empty the link layers."""
    for viewer_attr in _VIEWER_ATTRS:
        viewer = getattr(main_window, viewer_attr, None)
        if viewer is None or not hasattr(viewer, "layers"):
            continue
        for key in [
            key
            for key in _store(main_window, _SAVED_ATTR)
            if key[0] == viewer_attr
        ]:
            _restore_layer(main_window, viewer, viewer_attr, key[1])
        _clear_link(viewer)


def refresh_close_mask_view(main_window, row: pd.Series | None) -> None:
    """Show the close-mask pair of `row` in both viewers, or restore the normal view."""
    pinned = getattr(main_window, "close_mask_pinned_row", None)
    if pinned is not None and (
        row is None or close_mask_row_key(row) != pinned
    ):
        unpin_close_mask_row(main_window)
    viewers = [
        (attr, viewer)
        for attr in _VIEWER_ATTRS
        if (viewer := getattr(main_window, attr, None)) is not None
        and hasattr(viewer, "layers")
    ]
    flagged = _flagged_pairs(main_window, row)
    if flagged:
        for _, viewer in viewers:
            for mask_idx in get_mask_indices(main_window):
                layer = _layer_named(viewer, f"Segmentation{mask_idx}")
                if layer is not None:
                    _wire_visibility(main_window, layer)

    pairs = [
        pair
        for pair in flagged
        if any(_is_shown(viewer, pair.mask_idx) for _, viewer in viewers)
    ]
    if not pairs:
        clear_close_mask_view(main_window)
        return

    t = _resolve_t_index(main_window, row)
    geometry = _pair_geometry(main_window, row, pairs, t)
    for viewer_attr, viewer in viewers:
        _show_in_viewer(main_window, viewer, viewer_attr, geometry, t)


def refresh_close_mask_view_at_current(main_window) -> None:
    """`refresh_close_mask_view` for the row the UI is currently showing."""
    refresh_close_mask_view(main_window, current_selection_row(main_window))
