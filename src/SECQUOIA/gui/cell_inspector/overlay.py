"""Colours a mask crop the same way Napari does."""

from __future__ import annotations

import logging

import numpy as np
from skimage.segmentation import find_boundaries

LOG = logging.getLogger(__name__)

FILL = "Fill"
CONTOUR = "Contour"
OVERLAY_MODES = (FILL, CONTOUR)
FALLBACK_RGBA = (255, 215, 0, 255)


def _colors_for(colormap, label_ids: np.ndarray) -> np.ndarray:
    """Ask a Napari colormap for the colour of each label."""
    if colormap is not None:
        try:
            mapped = np.asarray(colormap.map(label_ids), dtype=np.float64)
            if mapped.ndim == 2 and mapped.shape[1] == 4:
                scale = 255.0 if mapped.max() <= 1.0 else 1.0
                return np.clip(mapped * scale, 0, 255).astype(np.uint8)
            LOG.debug(
                "cell inspector: unexpected colormap shape %s", mapped.shape
            )
        except (AttributeError, TypeError, ValueError, IndexError) as exc:
            LOG.warning("cell inspector: label colormap unusable: %s", exc)

    return np.tile(
        np.array(FALLBACK_RGBA, dtype=np.uint8), (len(label_ids), 1)
    )


def labels_to_rgba(
    label_crop: np.ndarray | None,
    colormap=None,
    mode: str = FILL,
) -> np.ndarray | None:
    """Colour one mask crop, or None if there is nothing to draw."""
    if label_crop is None:
        return None
    labels = np.asarray(label_crop)
    if labels.ndim != 2 or labels.size == 0:
        return None

    present = np.unique(labels)
    present = present[present != 0]
    if present.size == 0:
        return None

    if mode == CONTOUR:
        labels = np.where(find_boundaries(labels, mode="inner"), labels, 0)
        present = np.unique(labels)
        present = present[present != 0]
        if present.size == 0:
            return None

    lut = _colors_for(colormap, present)

    index = np.searchsorted(present, labels)
    np.clip(index, 0, present.size - 1, out=index)
    is_label = present[index] == labels

    rgba = np.zeros((*labels.shape, 4), dtype=np.uint8)
    rgba[is_label] = lut[index[is_label]]
    return rgba


def composite_overlay(
    label_crops: tuple[tuple[int, np.ndarray], ...],
    colormap_for,
    mode: str = FILL,
) -> np.ndarray | None:
    """Draw several masks into one overlay."""
    combined: np.ndarray | None = None
    for mask_number, crop in label_crops:
        rgba = labels_to_rgba(crop, colormap_for(mask_number), mode)
        if rgba is None:
            continue
        if combined is None:
            combined = rgba
            continue
        if combined.shape != rgba.shape:
            LOG.debug(
                "cell inspector: mask %s has a different crop shape",
                mask_number,
            )
            continue
        painted = rgba[..., 3] > 0
        combined[painted] = rgba[painted]
    return combined


def labels_colormap(viewer, mask_number: int):
    """Find the Napari colormap for a 1-based mask number."""
    if viewer is None:
        return None
    try:
        number = int(mask_number)
        layers = list(viewer.layers)
    except (AttributeError, TypeError, RuntimeError, ValueError):
        return None

    wanted = f"Segmentation{number}"
    named = next(
        (ly for ly in layers if getattr(ly, "name", "") == wanted), None
    )
    if named is None:
        label_layers = [
            ly
            for ly in layers
            if type(ly).__name__ == "Labels" and hasattr(ly, "colormap")
        ]
        if 1 <= number <= len(label_layers):
            named = label_layers[number - 1]

    return getattr(named, "colormap", None) if named is not None else None
