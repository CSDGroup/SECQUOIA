"""Remeasure a single object after an interactive mask edit.

This is the interactive counterpart to ``measure_objects`` in
:mod:`SECQUOIA.core.quantification`. Where that function measures every object
of every frame during ``quantify``, this module measures exactly the one object
the user just painted, erased, undone or redone, and writes it back into the
matching ``track_df`` row.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from SECQUOIA.config import FEATURES
from SECQUOIA.core.quantification import (
    FeatureNaming,
    label_column,
    measure_single_object,
)
from SECQUOIA.core.segmentation.mask_selection import (
    mask_index_from_layer_name,
)
from SECQUOIA.utils.positions import position_number_at_current_index

LOG = logging.getLogger(__name__)

VARIANT_ATTRIBUTES = {
    "NoBgCorrected": "images",
    "BaSiCBgCorrectedRatioFlat": "corrected_images_ratioflat",
    "BaSiCBgCorrectedNoRatioFlat": "corrected_images_noratioflat",
}


def basic_variants(main_window) -> tuple[str, ...]:
    """Image variants to measure, depending on whether BaSiC is enabled."""
    return (
        FEATURES.BASIC_VARIANT
        if main_window.basic.flag
        else FEATURES.NO_BASIC_VARIANT
    )


def labels_2d(layer, t: int) -> np.ndarray | None:
    """Return the 2D labels slice of `layer` at time `t`, or None."""
    data = getattr(layer, "data", None)
    if data is None:
        return None

    arr = np.asarray(data)
    if arr.ndim == 2:
        return arr
    if arr.ndim >= 3:
        return None if t < 0 or t >= arr.shape[0] else arr[t]
    return None


def image_2d(main_window, variant: str, channel: str, t: int):
    """Return the 2D image of (`variant`, `channel`) at time `t`, or None."""
    attribute = VARIANT_ATTRIBUTES.get(variant)
    if attribute is None:
        return None

    images = getattr(main_window, attribute, None)
    try:
        stack = images[channel]
    except (KeyError, TypeError):
        LOG.warning(
            "images[%r] missing for variant %s; skipping.", channel, variant
        )
        return None

    if stack is None or not hasattr(stack, "__len__"):
        return None

    n_frames = stack.shape[0] if isinstance(stack, np.ndarray) else len(stack)
    if t < 0 or t >= n_frames:
        return None

    image = np.asarray(stack[t])
    if image.ndim > 2:
        image = image[..., 0]
    return image if image.ndim == 2 else None


def frame_missing(main_window, channel: str, t: int) -> bool:
    """Return True when `channel` has no usable frame at `t`."""
    present = getattr(main_window, "image_present", None)
    if (
        isinstance(present, dict)
        and channel in present
        and present[channel] is not None
    ):
        flags = present[channel]
        try:
            if t < 0 or t >= len(flags):
                return True
            return not bool(flags[t])
        except (RuntimeError, AttributeError, TypeError):
            pass

    image = image_2d(main_window, "NoBgCorrected", channel, t)
    if image is None:
        return True
    try:
        return bool(np.all(np.asarray(image) == 0))
    except (RuntimeError, AttributeError, TypeError):
        return False


def current_position(main_window) -> int | None:
    """Return the 1-based number of the currently selected position."""
    return position_number_at_current_index(main_window)


@dataclass(frozen=True)
class EditTarget:
    """The single object an interactive edit applies to."""

    ident: str
    track_no: int
    t: int
    mask_idx: int
    row_idx: object
    label_id: int
    labels2d: np.ndarray
    layer_name: str


def mask_index_for_layer(main_window, layer) -> int | None:
    """Resolve the mask index of `layer`, preferring its binding tag."""
    name = getattr(layer, "name", "")
    tag = ""
    if hasattr(main_window, "_binding_from_layer_name"):
        try:
            binding = main_window._binding_from_layer_name(name) or {}
            tag = binding.get("tag", "")
        except (RuntimeError, AttributeError, TypeError, ValueError):
            tag = ""
    return mask_index_from_layer_name(tag or (name or ""))


def _row_coord(df, row_idx, col_base: str, fallback_col: str | None = None):
    """Return a rounded int coordinate from the row, trying `fallback_col` second."""
    for column in (col_base, fallback_col):
        if not column or column not in df.columns:
            continue
        if not pd.notna(df.at[row_idx, column]):
            continue
        try:
            return int(round(float(df.at[row_idx, column])))
        except (RuntimeError, AttributeError, TypeError, ValueError):
            continue
    return None


def _resolve_track_number(df, main_window, ident: str, t: int):
    """Return the TrackNumber being edited, inferred from the row when unambiguous."""
    track_no = getattr(main_window, "current_TrackNumber_plot", None)
    if track_no is not None:
        return track_no

    candidates = df[
        (df["Identification"].astype(str) == ident) & (df["t"] == t)
    ]
    unique = candidates["TrackNumber"].dropna().unique()
    return int(unique[0]) if unique.size == 1 else None


def _resolve_row_index(df, ident: str, track_no, t: int):
    """Return the single track_df row for (ident, track_no, t), or None."""
    row_mask = (
        (df["Identification"].astype(str) == ident)
        & (df["TrackNumber"] == track_no)
        & (df["t"] == t)
    )
    idxs = df.index[row_mask].to_list()
    if not idxs:
        LOG.warning(
            "No track_df row for Identification=%r, TrackNumber=%s, t=%s.",
            ident,
            track_no,
            t,
        )
        return None
    if len(idxs) > 1:
        LOG.warning(
            "Multiple rows match; updating the first one. Matches=%d",
            len(idxs),
        )
    return idxs[0]


def resolve_edit_target(
    main_window, layer, *, ident: str, t: int
) -> EditTarget | None:
    """Identify the object an edit applies to, or None with a logged reason."""
    df = getattr(main_window, "track_df", None)
    if not isinstance(df, pd.DataFrame) or df.empty:
        LOG.warning("track_df missing or empty; nothing to update.")
        return None

    track_no = _resolve_track_number(df, main_window, ident, t)
    if track_no is None:
        LOG.warning(
            "TrackNumber not set and could not infer uniquely for "
            "Identification=%r, t=%s. Aborting.",
            ident,
            t,
        )
        return None

    mask_idx = mask_index_for_layer(main_window, layer)
    if mask_idx is None:
        LOG.warning("Could not infer mask index from layer name; aborting.")
        return None

    layer_name = getattr(layer, "name", "")
    labels = labels_2d(layer, t)
    if labels is None:
        LOG.warning("No 2D labels for t=%s on layer '%s'.", t, layer_name)
        return None

    row_idx = _resolve_row_index(df, ident, track_no, t)
    if row_idx is None:
        return None

    x_anchor = _row_coord(
        df, row_idx, "XMorphology", f"XMorphologyM{mask_idx}"
    )
    y_anchor = _row_coord(
        df, row_idx, "YMorphology", f"YMorphologyM{mask_idx}"
    )
    height, width = labels.shape[-2], labels.shape[-1]
    if (
        x_anchor is None
        or y_anchor is None
        or not (0 <= y_anchor < height and 0 <= x_anchor < width)
    ):
        LOG.warning(
            "Invalid anchor (x=%s, y=%s) for labels bounds W=%s, H=%s.",
            x_anchor,
            y_anchor,
            width,
            height,
        )
        return None

    return EditTarget(
        ident=ident,
        track_no=track_no,
        t=t,
        mask_idx=mask_idx,
        row_idx=row_idx,
        label_id=int(labels[y_anchor, x_anchor]),
        labels2d=labels,
        layer_name=layer_name,
    )


@dataclass(frozen=True)
class MeasuredEdit:
    """Result of measuring one object, keyed by column-name components."""

    shape: dict[str, float]
    intensity: dict[tuple[str, str, str], float]
    missing_channel: dict[str, bool]
    measured: bool


def active_images(main_window, t: int, missing_channel: dict[str, bool]):
    """The (variant, channel) pairs usable at `t`, and their images.

    Channels flagged in `missing_channel` are skipped.
    """
    specs: list[tuple[str, str]] = []
    images: list[np.ndarray] = []

    for channel in main_window.ids_channels:
        if missing_channel[channel]:
            continue
        for variant in basic_variants(main_window):
            image = image_2d(main_window, variant, channel, t)
            if image is not None:
                specs.append((variant, channel))
                images.append(image)

    return specs, images


def measure_edit(main_window, target: EditTarget) -> MeasuredEdit:
    """Measure the edited object against every image that has a frame at t."""
    missing_channel = {
        channel: frame_missing(main_window, channel, target.t)
        for channel in main_window.ids_channels
    }
    specs, images = active_images(main_window, target.t, missing_channel)

    measurements = measure_single_object(
        target.labels2d.astype(np.int32, copy=False), images, target.label_id
    )

    if measurements is None:
        return MeasuredEdit(
            shape={},
            intensity={},
            missing_channel=missing_channel,
            measured=False,
        )

    shape = {
        prefix: float(values[0])
        for prefix, values in measurements.shape.items()
        if values is not None
    }

    if measurements.x is not None and len(measurements.x) > 0:
        shape["XMorphology"] = float(measurements.x[0])
    if measurements.y is not None and len(measurements.y) > 0:
        shape["YMorphology"] = float(measurements.y[0])

    intensity: dict[tuple[str, str, str], float] = {}
    for i, (variant, channel) in enumerate(specs):
        for prefix, values in measurements.intensity[i].items():

            if values is not None and np.isfinite(values[0]):
                intensity[(variant, channel, prefix)] = float(values[0])

    return MeasuredEdit(
        shape=shape,
        intensity=intensity,
        missing_channel=missing_channel,
        measured=True,
    )


def measurement_columns(main_window, mask_idx: int) -> tuple[list[str], str]:
    """Columns an edit of `mask_idx` touches, plus its label column.

    Ordered channel -> variant -> metric, then morphology.
    `changed_columns` walks this list, so the order fixes how the
    change log reads.
    """
    naming = FeatureNaming.from_main_window(main_window)
    variants = basic_variants(main_window)

    columns = [
        naming.intensity_column(f"{metric}{variant}", channel, mask_idx)
        for channel in main_window.ids_channels
        for variant in variants
        for metric in FEATURES.METRIC_PREFIXES
    ]
    columns += [
        naming.shape_column(prefix, mask_idx)
        for prefix in FEATURES.MORPH_PREFIXES
    ]
    return columns, label_column(mask_idx)


def snapshot_columns(df: pd.DataFrame, row_idx, columns) -> dict:
    """Return the current value of each column on `row_idx`; None when the column is absent."""
    return {
        column: df.at[row_idx, column] if column in df.columns else None
        for column in columns
    }


def ensure_columns(main_window, columns: list[str]) -> pd.DataFrame:
    """Create any missing measurement columns and force them to float."""
    df = main_window.track_df

    new_columns = [c for c in columns if c not in df.columns]
    if new_columns:
        df = pd.concat(
            [df, pd.DataFrame(np.nan, index=df.index, columns=new_columns)],
            axis=1,
        )
        main_window.track_df = df

    wrong_dtype = [c for c in columns if df[c].dtype.kind not in "fc"]
    if wrong_dtype:
        df[wrong_dtype] = df[wrong_dtype].astype(float)

    return df


def _write_label_id(df: pd.DataFrame, row_idx, lab_col: str, label_id: int):
    """Write the matched label id, widening the column only if forced to."""
    if lab_col not in df.columns:
        df[lab_col] = pd.Series([pd.NA] * len(df), dtype="Int64")
    try:
        df.at[row_idx, lab_col] = int(label_id)
    except (RuntimeError, AttributeError, TypeError, ValueError):
        df[lab_col] = df[lab_col].astype(object)
        df.at[row_idx, lab_col] = int(label_id)


def _intensity_value(measured: MeasuredEdit, key, channel: str):
    """Return the value to write for one intensity column."""
    if measured.missing_channel[channel] or not measured.measured:
        return np.nan
    return measured.intensity.get(key, 0.0)


def write_measurements(
    main_window, target: EditTarget, measured: MeasuredEdit
) -> pd.DataFrame:
    """Write one object's measurements into its ``track_df`` row."""
    columns, lab_col = measurement_columns(main_window, target.mask_idx)
    df = ensure_columns(main_window, columns)
    naming = FeatureNaming.from_main_window(main_window)

    _write_label_id(df, target.row_idx, lab_col, target.label_id)

    for prefix in FEATURES.MORPH_PREFIXES:
        df.at[target.row_idx, naming.shape_column(prefix, target.mask_idx)] = (
            measured.shape.get(prefix, np.nan)
        )

    for channel in main_window.ids_channels:
        for variant in basic_variants(main_window):
            for metric in FEATURES.METRIC_PREFIXES:
                column = naming.intensity_column(
                    f"{metric}{variant}", channel, target.mask_idx
                )
                df.at[target.row_idx, column] = _intensity_value(
                    measured, (variant, channel, metric), channel
                )

    return df


def changed_columns(before: dict, after: dict, columns) -> list[tuple]:
    """Columns whose value changed, as ``(column, old, new)``."""
    changed = []
    for column in columns:
        old, new = before.get(column), after.get(column)
        same = (pd.isna(old) and pd.isna(new)) or (old == new)
        if not same:
            changed.append((column, old, new))
    return changed


def log_measurement_update(
    target: EditTarget, changed: list[tuple], *, source: str = ""
) -> None:
    """Log what the edit was, and which columns it moved."""
    LOG.info(
        "%sIdentification=%r, TrackNumber=%s, t=%s, layer=%r, m=%s, label=%s",
        f"({source}) " if source else "",
        target.ident,
        target.track_no,
        target.t,
        target.layer_name,
        target.mask_idx,
        target.label_id,
    )
    if not changed:
        LOG.info("No column changes (values identical).")
        return
    for column, old, new in changed:
        LOG.info("  - %s: %s -> %s", column, old, new)
