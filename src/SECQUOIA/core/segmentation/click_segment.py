"""Segment the single cell under a click (brightfield or fluorescence)."""

import logging
import os
import threading
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gaussian, sobel
from skimage.measure import label as cc_label
from skimage.morphology import disk
from skimage.segmentation import watershed

LOG = logging.getLogger(__name__)

_WINDOW_RADII = (24, 48, 96, 192)
_CELLPOSE_RADIUS = 96
_MIN_AREA = 20
SMOOTHING_RANGE = (0.0, 5.0)
EDGE_SENSITIVITY_RANGE = (0.1, 2.0)
CELLPROB_RANGE = (-6.0, 6.0)
DIAMETER_RANGE = (5.0, 300.0)
MASK_SIZE_RANGE = (-10, 10)
_lock = threading.RLock()
_model = None
_unavailable = False
_warm = False


def _clamped(value, bounds, default):
    """`value` as a float inside `bounds`, or `default` if it is not a number."""
    try:
        return min(max(float(value), bounds[0]), bounds[1])
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class ClickSettings:
    """Tunable values of the click tool."""

    smoothing: float = 1.0
    edge_sensitivity: float = 0.5
    cellprob_threshold: float = 0.0
    diameter: float | None = None
    mask_size: int = 0

    @classmethod
    def from_dict(cls, data) -> "ClickSettings":
        """Build settings from saved metadata. Missing or invalid values become defaults."""
        d = cls()
        if not isinstance(data, dict):
            return d
        diameter = data.get("diameter")
        return cls(
            smoothing=_clamped(
                data.get("smoothing"), SMOOTHING_RANGE, d.smoothing
            ),
            edge_sensitivity=_clamped(
                data.get("edge_sensitivity"),
                EDGE_SENSITIVITY_RANGE,
                d.edge_sensitivity,
            ),
            cellprob_threshold=_clamped(
                data.get("cellprob_threshold"),
                CELLPROB_RANGE,
                d.cellprob_threshold,
            ),
            diameter=(
                None
                if diameter is None
                else _clamped(diameter, DIAMETER_RANGE, None)
            ),
            mask_size=round(
                _clamped(data.get("mask_size"), MASK_SIZE_RANGE, d.mask_size)
            ),
        )

    def to_dict(self) -> dict:
        """Plain dict for the project metadata JSON."""
        return asdict(self)


@dataclass(frozen=True)
class ClickSettingsSet:
    """Settings for all masks, with optional overrides for individual masks."""

    default: ClickSettings = field(default_factory=ClickSettings)
    overrides: dict = field(default_factory=dict)

    def for_mask(self, mask_idx: int | None) -> ClickSettings:
        """The values to use for `mask_idx`: its own, else the shared default."""
        return self.overrides.get(mask_idx, self.default)

    @classmethod
    def from_dict(cls, data) -> "ClickSettingsSet":
        """Load saved metadata."""
        if not isinstance(data, dict):
            return cls()
        if "default" not in data and "masks" not in data:
            return cls(default=ClickSettings.from_dict(data))
        overrides = {}
        masks = data.get("masks")
        for key, values in masks.items() if isinstance(masks, dict) else ():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            if idx >= 1:
                overrides[idx] = ClickSettings.from_dict(values)
        return cls(
            default=ClickSettings.from_dict(data.get("default")),
            overrides=overrides,
        )

    def to_dict(self) -> dict:
        """Plain dict for the project metadata JSON."""
        return {
            "default": self.default.to_dict(),
            "masks": {
                str(idx): cfg.to_dict()
                for idx, cfg in sorted(self.overrides.items())
            },
        }


def resize_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Grow (pixels > 0) or shrink (pixels < 0) a mask."""
    if pixels == 0 or not mask.any():
        return mask
    ys, xs = np.nonzero(mask)
    pad = abs(pixels) + 1
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, mask.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, mask.shape[1])
    sub = mask[y0:y1, x0:x1]
    step = disk(1)
    if pixels > 0:
        sub = ndi.binary_dilation(sub, structure=step, iterations=pixels)
    else:
        for _ in range(-pixels):
            smaller = ndi.binary_erosion(sub, structure=step)
            if not smaller.any():
                break
            sub = smaller
    out = mask.copy()
    out[y0:y1, x0:x1] = sub
    return out


def _normalize(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, np.float32)
    lo, hi = np.percentile(img, (1, 99.5))
    return np.clip((img - lo) / (hi - lo), 0, 1) if hi > lo else img * 0


def _crop(img: np.ndarray, y: int, x: int, r: int):
    """Square crop of radius r clipped to the image, with its (y0, x0) origin."""
    y0, x0 = max(0, y - r), max(0, x - r)
    return img[y0 : y + r + 1, x0 : x + r + 1], y0, x0


def _window_mask(
    crop: np.ndarray, cy: int, cx: int, cfg: ClickSettings
) -> np.ndarray:
    """Seeded watershed in one crop; returns the clicked cell's mask."""
    g = gaussian(crop, sigma=cfg.smoothing) if cfg.smoothing > 0 else crop
    edges = sobel(g)
    yy, xx = np.ogrid[: crop.shape[0], : crop.shape[1]]
    seed = (yy - cy) ** 2 + (xx - cx) ** 2 <= 4
    similar = np.abs(g - np.median(g[seed])) < cfg.edge_sensitivity * max(
        float(g.std()), 1e-3
    )
    seed |= similar & ndi.binary_dilation(seed, iterations=3)

    markers = np.zeros(crop.shape, np.int32)
    markers[seed] = 1
    frame = np.zeros(crop.shape, bool)
    frame[[0, -1], :] = frame[:, [0, -1]] = True
    far = ndi.distance_transform_edt(~seed) > 0.3 * min(crop.shape)
    flat = edges < np.percentile(edges, 30)
    markers[(frame | (flat & far)) & ~seed] = 2

    mask = ndi.binary_fill_holes(watershed(edges, markers) == 1)
    mask = ndi.binary_opening(mask, structure=disk(1))
    lab = cc_label(mask)
    return lab == lab[cy, cx] if lab[cy, cx] else np.zeros_like(mask)


def _watershed(
    image: np.ndarray, y: int, x: int, cfg: ClickSettings
) -> np.ndarray:
    """Grow the window until the cell no longer touches its border."""
    img = _normalize(image)
    out = np.zeros(img.shape, bool)
    for r in _WINDOW_RADII:
        crop, y0, x0 = _crop(img, y, x, r)
        m = _window_mask(crop, y - y0, x - x0, cfg)
        if m.sum() < _MIN_AREA:
            continue
        h, w = m.shape
        open_edges = (
            (y0 > 0 and m[0].any())
            or (y0 + h < img.shape[0] and m[-1].any())
            or (x0 > 0 and m[:, 0].any())
            or (x0 + w < img.shape[1] and m[:, -1].any())
        )
        if not open_edges:
            out[y0 : y0 + h, x0 : x0 + w] = m
            break
    return out


def _load_model():
    """Load Cellpose once (None if not installed)."""
    global _model, _unavailable
    with _lock:
        if _model is None and not _unavailable:
            try:
                import torch
                from cellpose import models

                threads = max(
                    1, (os.cpu_count() or 2) - 1
                )  # leave one core for the UI
                torch.set_num_threads(threads)
                gpu = (
                    torch.backends.mps.is_available()
                    or torch.cuda.is_available()
                )
                _model = models.CellposeModel(
                    gpu=gpu,
                    use_bfloat16=False,
                )
                LOG.info(
                    "[click-segment] Cellpose loaded on %s with %d CPU threads",
                    _model.device,
                    threads,
                )
            except (
                ImportError,
                AttributeError,
                OSError,
                RuntimeError,
                ValueError,
            ) as exc:
                LOG.info("[click-segment] Cellpose unavailable: %s", exc)
                _unavailable = True
        return _model


def preload_cellpose() -> None:
    """Load Cellpose and run one dummy crop so the first real click is fast."""
    global _warm
    with _lock:
        if _warm or _load_model() is None:
            return
        side = 2 * _CELLPOSE_RADIUS + 1
        _model.eval(np.random.default_rng(0).random((side, side), np.float32))
        _warm = True


def _cellpose(image: np.ndarray, y: int, x: int, cfg: ClickSettings):
    """Cellpose mask of the clicked cell, or None."""
    with _lock:
        if _load_model() is None:
            return None
        crop, y0, x0 = _crop(_normalize(image), y, x, _CELLPOSE_RADIUS)
        try:
            masks = _model.eval(
                crop,
                cellprob_threshold=cfg.cellprob_threshold,
                diameter=cfg.diameter,
            )[0]
        except (RuntimeError, ValueError) as exc:
            LOG.warning("[click-segment] Cellpose failed: %s", exc)
            return None
    lab = masks[y - y0, x - x0]
    if not lab:
        return None
    out = np.zeros(image.shape, bool)
    out[y0 : y0 + masks.shape[0], x0 : x0 + masks.shape[1]] = masks == lab
    return out


def segment_at(
    image,
    y: int,
    x: int,
    engine: str = "watershed",
    settings: ClickSettings | None = None,
):
    """Return (bool mask of the cell under (y, x), engine used) for a 2D image."""
    cfg = settings or ClickSettings()
    if engine == "cellpose":
        mask = _cellpose(image, y, x, cfg)
        if mask is not None and mask.sum() >= _MIN_AREA:
            return resize_mask(mask, cfg.mask_size), "cellpose"
    return (
        resize_mask(_watershed(image, y, x, cfg), cfg.mask_size),
        "watershed",
    )
