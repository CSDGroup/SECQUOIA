"""Per (mask, channel, frame) regionprops measurement engine."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import find_objects
from skimage.measure import regionprops_table

from SECQUOIA.config import FEATURES
from SECQUOIA.core.quantification.image_layers import (
    ImageLayer,
    _ensure_int_labels,
    _image_layers,
    _is_frame_present,
    _presence_flags,
)
from SECQUOIA.core.quantification.naming import (
    FeatureNaming,
    centroid_columns,
    label_column,
)
from SECQUOIA.core.quantification.progress import ProgressReporter

# regionprops properties measured once per mask (channel independent).
SHAPE_PROPERTIES = (
    "centroid",
    "area",
    "perimeter",
    "orientation",
    "eccentricity",
    "label",
    "axis_major_length",
    "axis_minor_length",
)

# regionprops properties measured for every (mask, channel, frame).
INTENSITY_PROPERTIES = (
    "mean_intensity",
    "min_intensity",
    "max_intensity",
    "intensity_std",
)

# metrics written per image layer; "Sum"/"CV" are derived, not measured.
INTENSITY_METRICS = tuple(FEATURES.METRIC_PREFIXES)

# regionprops key -> SECQUOIA column prefix (see :class:`FeatureNaming`).
SHAPE_PROP_TO_PREFIX = {
    "area": "AreaMorphology",
    "perimeter": "PerimeterMorphology",
    "orientation": "Orientation",
    "eccentricity": "Eccentricity",
    "axis_major_length": "AxisMajorLength",
    "axis_minor_length": "AxisMinorLength",
}


def sum_intensity(region_mask, intensity_image) -> float:
    """Return the per-pixel intensity sum over a boolean region mask."""
    return float(np.sum(intensity_image[region_mask], dtype=np.float64))


@dataclass
class FrameMeasurements:
    """Regionprops results of a single (mask, frame), as arrays."""

    label_ids: np.ndarray
    x: np.ndarray
    y: np.ndarray
    shape: dict[str, np.ndarray | None]
    intensity: list[dict[str, np.ndarray | None]]

    def select(self, keep: np.ndarray) -> "FrameMeasurements":
        """Return a copy restricted to the objects flagged in `keep`."""

        def _sub(values):
            return None if values is None else values[keep]

        return FrameMeasurements(
            label_ids=self.label_ids[keep],
            x=self.x[keep],
            y=self.y[keep],
            shape={k: _sub(v) for k, v in self.shape.items()},
            intensity=[
                {k: _sub(v) for k, v in layer.items()}
                for layer in self.intensity
            ],
        )


def _as_2d_image(image):
    """Return a 2-D view of `image`, or None if it cannot be interpreted."""
    if image is None:
        return None
    if image.ndim > 2:
        image = image[..., 0]
    return image if image.ndim == 2 else None


def _stack_intensity_images(images: list[np.ndarray]) -> np.ndarray | None:
    """Stack per layer 2-D frames into one ``(H, W, K)`` multichannel image."""
    if not images:
        return None
    return np.stack(images, axis=-1)


def _regionprops_for_frame(labels, intensity_stack):
    """Run regionprops **once** for one (mask, frame)."""
    properties = list(SHAPE_PROPERTIES)
    if intensity_stack is None:
        props = regionprops_table(labels, properties=properties)
    else:
        props = regionprops_table(
            labels,
            intensity_image=intensity_stack,
            properties=properties + list(INTENSITY_PROPERTIES),
        )

    if len(props["label"]) == 0:
        return None
    return props


def _float_array(props: dict, key: str):
    if key not in props:
        return None
    return np.asarray(props[key], dtype=float)


def _coefficient_of_variation(std_values, mean_values):
    """CV in percent; NaN wherever the mean is zero."""
    if std_values is None or mean_values is None:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = std_values / mean_values * 100.0
    return np.where(mean_values == 0, np.nan, cv)


def _layer_intensity(props: dict, layer: int, areas):
    """Pull the metrics of one layer out of a multichannel regionprops dict."""
    mean_values = _float_array(props, f"mean_intensity-{layer}")
    std_values = _float_array(props, f"intensity_std-{layer}")
    sum_values = (
        None if (mean_values is None or areas is None) else mean_values * areas
    )
    return {
        "Mean": mean_values,
        "Min": _float_array(props, f"min_intensity-{layer}"),
        "Max": _float_array(props, f"max_intensity-{layer}"),
        "Sum": sum_values,
        "Std": std_values,
        "CV": _coefficient_of_variation(std_values, mean_values),
    }


def _measurements_from_props(props: dict, n_layers: int) -> FrameMeasurements:
    """Convert one multichannel regionprops dict into a `FrameMeasurements`."""
    label_ids = np.asarray(props["label"], dtype=int)
    zeros = np.zeros_like(label_ids)
    ys = (
        np.rint(props["centroid-0"]).astype(int)
        if "centroid-0" in props
        else zeros
    )
    xs = (
        np.rint(props["centroid-1"]).astype(int)
        if "centroid-1" in props
        else zeros
    )

    shape = {
        prefix: _float_array(props, key)
        for key, prefix in SHAPE_PROP_TO_PREFIX.items()
    }
    areas = shape.get("AreaMorphology")

    return FrameMeasurements(
        label_ids=label_ids,
        x=xs,
        y=ys,
        shape=shape,
        intensity=[
            _layer_intensity(props, layer, areas) for layer in range(n_layers)
        ],
    )


def measure_frame(labels2d, layer_images: list) -> FrameMeasurements | None:
    """Measure every object of one frame against every image layer."""
    props = _regionprops_for_frame(
        labels2d, _stack_intensity_images(layer_images)
    )
    if props is None:
        return None
    return _measurements_from_props(props, len(layer_images))


def measure_single_object(
    labels2d, layer_images: list, label_id: int
) -> FrameMeasurements | None:
    """Measure exactly one label, without touching the rest of the frame."""
    label_id = int(label_id)
    if label_id <= 0 or labels2d is None:
        return None

    boxes = find_objects(labels2d, max_label=label_id)
    if len(boxes) < label_id:
        return None
    box = boxes[label_id - 1]
    if box is None:
        return None

    rows, cols = box
    labels_crop = np.where(
        labels2d[box] == label_id, np.int32(label_id), np.int32(0)
    )
    measurements = measure_frame(
        labels_crop, [img[box] for img in layer_images]
    )
    if measurements is None:
        return None

    # Centroids are relative to the crop: shift them back onto the full frame.
    measurements.y = measurements.y + rows.start
    measurements.x = measurements.x + cols.start
    return measurements


def _drop_small_objects(
    measurements: FrameMeasurements, min_area_pixels: float
) -> FrameMeasurements | None:
    """Remove objects below `min_area_pixels`; None if nothing survives."""
    areas = measurements.shape.get("AreaMorphology")
    if areas is not None and areas.size:
        keep = areas >= float(min_area_pixels)
    else:
        keep = np.ones_like(measurements.label_ids, dtype=bool)
    if not np.any(keep):
        return None
    if keep.all():
        return measurements
    return measurements.select(keep)


def _rows_from_measurements(
    measurements: FrameMeasurements,
    *,
    mask_idx: int,
    t: int,
    position,
    shape_columns: dict[str, str],
    intensity_columns: list[dict[str, str]],
) -> list[dict]:
    """Turn one frame's measurements into one row dict per object."""
    x_col, y_col = centroid_columns(mask_idx)
    lab_col = label_column(mask_idx)

    shape_writes = [
        (shape_columns[prefix], values)
        for prefix, values in measurements.shape.items()
        if values is not None
    ]
    intensity_writes = [
        (columns[metric], values)
        for columns, layer in zip(
            intensity_columns, measurements.intensity, strict=False
        )
        for metric, values in layer.items()
        if values is not None and np.isfinite(values).all()
    ]

    rows = []
    for k, label_id in enumerate(measurements.label_ids):
        row = {
            "Position": position,
            "t": t,
            "XMorphology": int(measurements.x[k]),
            "YMorphology": int(measurements.y[k]),
            x_col: float(measurements.x[k]),
            y_col: float(measurements.y[k]),
            lab_col: int(label_id),
            "__mask_idx__": mask_idx,
        }
        for column, values in shape_writes:
            row[column] = float(values[k])
        for column, values in intensity_writes:
            row[column] = float(values[k])
        rows.append(row)
    return rows


def _active_layers_at(
    layers: list[ImageLayer],
    *,
    t: int,
    layer_frames: list[int],
    layer_flags: list,
) -> tuple[list[int], list[np.ndarray]]:
    """Which layers have a usable frame at `t`, and their 2-D images."""
    active_idx: list[int] = []
    images: list[np.ndarray] = []

    for i, layer in enumerate(layers):
        if t >= layer_frames[i]:
            continue
        if not _is_frame_present(layer_flags[i], t, layer_frames[i]):
            continue
        image = _as_2d_image(layer.stack[t])
        if image is None:
            continue
        active_idx.append(i)
        images.append(image)

    return active_idx, images


def measure_objects(
    main_window,
    label_entries,
    naming: FeatureNaming,
    *,
    position,
    min_area_pixels: float,
    progress: ProgressReporter,
) -> list[dict]:
    """Measure every mask and frame, returning one row per segmented object."""
    layers = _image_layers(main_window)
    if not layers:
        return []

    layer_flags = [
        _presence_flags(main_window, layer.channel) for layer in layers
    ]
    rows: list[dict] = []

    for mask_idx, raw_stack in label_entries:
        label_stack = _ensure_int_labels(raw_stack)

        shape_columns = {
            prefix: naming.shape_column(prefix, mask_idx)
            for prefix in SHAPE_PROP_TO_PREFIX.values()
        }
        layer_columns = [
            {
                metric: naming.intensity_column(
                    f"{metric}{layer.variant}", layer.channel, mask_idx
                )
                for metric in INTENSITY_METRICS
            }
            for layer in layers
        ]
        layer_frames = [
            min(len(layer.stack), len(label_stack)) for layer in layers
        ]
        n_frames = max(layer_frames, default=0)

        for t in range(n_frames):
            try:
                labels = label_stack[t]
                if labels is None or labels.ndim != 2:
                    continue

                active_idx, images = _active_layers_at(
                    layers,
                    t=t,
                    layer_frames=layer_frames,
                    layer_flags=layer_flags,
                )

                measurements = measure_frame(labels, images)
                if measurements is None:
                    continue

                measurements = _drop_small_objects(
                    measurements, min_area_pixels
                )
                if measurements is None:
                    continue

                rows.extend(
                    _rows_from_measurements(
                        measurements,
                        mask_idx=mask_idx,
                        t=t,
                        position=position,
                        shape_columns=shape_columns,
                        intensity_columns=[
                            layer_columns[i] for i in active_idx
                        ],
                    )
                )
            finally:
                progress.tick("Measuring frames…")

    return rows
