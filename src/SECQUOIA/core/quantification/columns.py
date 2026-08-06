"""Measurement dataframe column setup, ordering and pruning."""

import logging
import re

import numpy as np
import pandas as pd

from SECQUOIA.config import FEATURES
from SECQUOIA.core.quantification.naming import (
    FeatureNaming,
    centroid_columns,
    distance_column,
    label_column,
    step_distance_column,
)

LOG = logging.getLogger(__name__)

_ALL_INTENSITY_FEATURES = frozenset(
    f"{metric}{variant}"
    for metric in FEATURES.METRIC_PREFIXES
    for variant in FEATURES.BASIC_VARIANT
)

# Shape features written once per mask, without a channel token.
_SHAPE_FEATURES = frozenset(FEATURES.MORPH_PREFIXES)

# ``<feature>Ch<nn>M<n>`` - an intensity metric of one channel and mask.
_INTENSITY_COL_RE = re.compile(r"^(?P<feat>[A-Za-z0-9_]+)Ch(\d+)M(?P<m>\d+)$")
# ``<feature>M<n>`` - a per mask measurement with no channel.
_SHAPE_COL_RE = re.compile(r"^(?P<feat>[A-Za-z0-9_]+)M(?P<m>\d+)$")
# Per mask bookkeeping columns written by matching/lineage.
_MASK_INDEXED_COL_RE = re.compile(
    r"^(?P<feat>label_id_m|nn_dist_px_m|step_disp_px_m)(?P<m>\d+)$"
)


def init_mask_channel_columns(
    df: pd.DataFrame, mask_indices, naming: FeatureNaming
) -> pd.DataFrame:
    """Create the per mask intensity, shape, label and distance columns.

    Existing columns are kept and only cast to their expected dtype.
    Lineage displacement (``step_disp_px_m*``) is not created here -
    `add_lineage_step_distance` writes it after matching.
    """
    to_add: dict = {}
    fill_zero_cols: list[str] = []
    float_cols: list[str] = []
    label_cols: list[str] = []

    for mask_idx in mask_indices:
        for channel in naming.channels:
            for column in naming.intensity_columns(channel, mask_idx):
                if column in df.columns:
                    float_cols.append(column)
                else:
                    to_add[column] = np.nan

        for prefix in naming.shape_prefixes:
            column = naming.shape_column(prefix, mask_idx)
            if column in df.columns:
                fill_zero_cols.append(column)
            else:
                to_add[column] = 0.0

        lab_col = label_column(mask_idx)
        if lab_col in df.columns:
            label_cols.append(lab_col)
        else:
            to_add[lab_col] = pd.Series(0, index=df.index, dtype="Int64")

        dist_col = distance_column(mask_idx)
        if dist_col in df.columns:
            float_cols.append(dist_col)
        else:
            to_add[dist_col] = np.nan

    if to_add:
        df = pd.concat([df, pd.DataFrame(to_add, index=df.index)], axis=1)

    if fill_zero_cols:
        df.loc[:, fill_zero_cols] = df[fill_zero_cols].fillna(0.0)
    if float_cols:
        df.loc[:, float_cols] = df[float_cols].astype(float)
    for column in label_cols:
        df[column] = (
            pd.to_numeric(df[column], errors="coerce")
            .fillna(0)
            .astype("Int64")
        )

    return df.copy()


def expected_measurement_columns(
    naming: FeatureNaming, mask_indices
) -> set[str]:
    """Every measurement column the current configuration produces."""
    columns: set[str] = set()
    for mask_idx in mask_indices:
        for channel in naming.channels:
            columns.update(naming.intensity_columns(channel, mask_idx))
        for prefix in naming.shape_prefixes:
            columns.add(naming.shape_column(prefix, mask_idx))
        columns.update(centroid_columns(mask_idx))
        columns.add(label_column(mask_idx))
        columns.add(distance_column(mask_idx))
        columns.add(step_distance_column(mask_idx))
    return columns


def stale_measurement_columns(
    columns, naming: FeatureNaming, mask_indices
) -> list[str]:
    """Columns of a generated family the current config no longer writes."""
    keep = expected_measurement_columns(naming, mask_indices)
    masks = {int(mask_idx) for mask_idx in mask_indices}

    stale: list[str] = []
    for column in columns:
        name = str(column)
        if name in keep:
            continue

        match = _INTENSITY_COL_RE.match(name)
        if match and match.group("feat") in _ALL_INTENSITY_FEATURES:
            stale.append(name)
            continue

        match = _MASK_INDEXED_COL_RE.match(name)
        if match and int(match.group("m")) not in masks:
            stale.append(name)
            continue

        match = _SHAPE_COL_RE.match(name)
        if (
            match
            and match.group("feat") in _SHAPE_FEATURES
            and int(match.group("m")) not in masks
        ):
            stale.append(name)

    return stale


def drop_stale_measurement_columns(
    df: pd.DataFrame,
    naming: FeatureNaming,
    mask_indices,
    *,
    log_prefix: str = "columns",
) -> pd.DataFrame:
    """Return `df` without the measurement columns the current config no longer writes."""
    if not isinstance(df, pd.DataFrame):
        return df

    stale = stale_measurement_columns(df.columns, naming, mask_indices)
    if not stale:
        return df

    LOG.info(
        "[%s] Dropping %d stale measurement column(s): %s",
        log_prefix,
        len(stale),
        ", ".join(
            f"{c} ({_value_count(df, c)} values)" for c in sorted(stale)
        ),
    )
    return df.drop(columns=stale)


def _value_count(df: pd.DataFrame, column: str) -> int:
    """Number of non-null values in `column`, 0 if it cannot be counted."""
    try:
        values = df[column]
        if isinstance(values, pd.DataFrame):
            values = values.iloc[:, 0]
        return int(values.notna().sum())
    except (KeyError, TypeError, ValueError):
        return 0


def sort_measurement_columns(
    df: pd.DataFrame, mask_indices, naming: FeatureNaming, id_col: str
) -> pd.DataFrame:
    """Reorder columns: identifiers, intensities, shapes, labels, distances."""
    base = [
        c
        for c in ("Position", "t", "XMorphology", "YMorphology", id_col)
        if c in df.columns
    ]

    intensity_cols = [
        naming.intensity_column(prefix, channel, mask_idx)
        for prefix in naming.metric_prefixes
        for channel in naming.channels
        for mask_idx in mask_indices
    ]
    shape_cols = [
        naming.shape_column(prefix, mask_idx)
        for prefix in naming.shape_prefixes
        for mask_idx in mask_indices
    ]
    label_cols = [label_column(m) for m in mask_indices]
    dist_cols = [distance_column(m) for m in mask_indices]

    if df.columns.has_duplicates:
        dupes = df.columns[df.columns.duplicated()].unique().tolist()
        LOG.warning("Dropping duplicate columns: %s", dupes)
        df = df.loc[:, ~df.columns.duplicated()]

    seen: set = set()
    preferred = []
    for c in base + intensity_cols + shape_cols + label_cols + dist_cols:
        if c in df.columns and c not in seen:
            seen.add(c)
            preferred.append(c)

    remaining = []
    for c in df.columns:
        if c not in seen:
            seen.add(c)
            remaining.append(c)

    return df.reindex(columns=preferred + remaining)
