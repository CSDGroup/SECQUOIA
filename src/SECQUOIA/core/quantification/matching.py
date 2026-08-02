"""Nearest neighbour matching of measured objects onto existing tracks."""

import numpy as np
import pandas as pd

from SECQUOIA.core.quantification.naming import (
    FeatureNaming,
    distance_column,
    label_column,
)

# per mask geometry columns copied from the objects onto the tracks.
GEOMETRY_PREFIXES = (
    "XMorphology",
    "YMorphology",
    "AreaMorphology",
    "PerimeterMorphology",
    "Orientation",
    "Eccentricity",
)


def greedy_nearest_assign(
    tracks_xy: np.ndarray,
    objects_xy: np.ndarray,
    max_dist: float,
    *,
    one_to_one: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each track to its nearest object within `max_dist`.

    Returns ``(object_index_per_track, distance_per_track)`` where an index of
    -1 / a distance of inf means "unmatched". With `one_to_one`, pairs are
    resolved shortest-distance-first and each object is used at most once.
    """
    n_tracks, n_objects = tracks_xy.shape[0], objects_xy.shape[0]
    if n_tracks == 0 or n_objects == 0:
        return np.full(n_tracks, -1, dtype=int), np.full(n_tracks, np.inf)

    distances = np.hypot(
        tracks_xy[:, [0]] - objects_xy[:, 0][None, :],
        tracks_xy[:, [1]] - objects_xy[:, 1][None, :],
    )
    nearest_idx = np.argmin(distances, axis=1)
    nearest_dist = distances[np.arange(n_tracks), nearest_idx]

    if not one_to_one:
        too_far = nearest_dist > max_dist
        nearest_idx[too_far] = -1
        nearest_dist[too_far] = np.inf
        return nearest_idx, nearest_dist

    candidates = sorted(
        (
            (nearest_dist[i], i, int(nearest_idx[i]))
            for i in range(n_tracks)
            if nearest_dist[i] <= max_dist
        ),
    )

    assigned = np.full(n_tracks, -1, dtype=int)
    assigned_dist = np.full(n_tracks, np.inf)
    used: set[int] = set()

    for dist, track_i, object_j in candidates:
        if object_j not in used:
            used.add(object_j)
            assigned[track_i] = object_j
            assigned_dist[track_i] = dist
            continue

        # Preferred object already taken: fall back to the next free one.
        row = distances[track_i].copy()
        row[list(used)] = np.inf
        fallback_j = int(np.argmin(row))
        if np.isfinite(row[fallback_j]) and row[fallback_j] <= max_dist:
            used.add(fallback_j)
            assigned[track_i] = fallback_j
            assigned_dist[track_i] = row[fallback_j]

    return assigned, assigned_dist


def _mask_measurement_columns(
    objects_df: pd.DataFrame, mask_idx: int, naming: FeatureNaming
) -> list[str]:
    """Object columns belonging to `mask_idx` that are copied onto a track."""
    shape_cols = {
        naming.shape_column(prefix, mask_idx)
        for prefix in naming.shape_prefixes
    }
    return [
        c
        for c in objects_df.columns
        if (
            c.startswith(naming.metric_prefixes) and c.endswith(f"M{mask_idx}")
        )
        or c in shape_cols
    ]


def _prepare_track_columns(
    merged_df: pd.DataFrame,
    objects_df: pd.DataFrame,
    mask_indices,
    naming: FeatureNaming,
) -> pd.DataFrame:
    """Create/normalise the columns that the matching step writes into."""
    measurement_cols = [
        c for c in objects_df.columns if c.startswith(naming.metric_prefixes)
    ]
    geometry_cols = [
        f"{prefix}M{mask_idx}"
        for mask_idx in mask_indices
        for prefix in GEOMETRY_PREFIXES
        if f"{prefix}M{mask_idx}" in objects_df.columns
    ]
    label_cols = [c for c in objects_df.columns if c.startswith("label_id_m")]

    for column in measurement_cols:
        if column not in merged_df.columns:
            merged_df[column] = np.nan
    if measurement_cols:
        merged_df[measurement_cols] = merged_df[measurement_cols].astype(float)

    for column in geometry_cols:
        if column not in merged_df.columns:
            merged_df[column] = 0.0
    if geometry_cols:
        merged_df[geometry_cols] = merged_df[geometry_cols].astype(float)

    for column in label_cols:
        if column not in merged_df.columns:
            merged_df[column] = 0
        # NaN => 0 so that "no link" is always encoded as 0.
        merged_df[column] = (
            pd.to_numeric(merged_df[column], errors="coerce")
            .fillna(0)
            .astype("Int64")
        )

    for mask_idx in mask_indices:
        dist_col = distance_column(mask_idx)
        if dist_col not in merged_df.columns:
            merged_df[dist_col] = np.nan
        merged_df[dist_col] = merged_df[dist_col].astype(float)

    return merged_df


def assign_objects_to_tracks(
    merged_df: pd.DataFrame,
    objects_df: pd.DataFrame,
    mask_indices,
    naming: FeatureNaming,
    *,
    max_pixel_distance: float,
    one_to_one: bool,
) -> pd.DataFrame:
    """Copy object measurements onto their nearest track, per time point and mask."""
    pairs: dict[int, tuple[list, list, list]] = {}

    for t_val in sorted(objects_df["t"].unique()):
        tracks_t = merged_df[merged_df["t"] == t_val]
        if tracks_t.empty:
            continue
        track_indices = tracks_t.index.to_numpy()
        tracks_xy = tracks_t[["XMorphology", "YMorphology"]].to_numpy(
            dtype=float
        )

        for mask_idx in mask_indices:
            objects_tm = objects_df[
                (objects_df["t"] == t_val)
                & (objects_df["__mask_idx__"] == mask_idx)
            ]
            if objects_tm.empty:
                continue

            objects_xy = objects_tm[["XMorphology", "YMorphology"]].to_numpy(
                dtype=float
            )
            object_indices = objects_tm.index.to_numpy()

            assigned, distances = greedy_nearest_assign(
                tracks_xy,
                objects_xy,
                max_pixel_distance,
                one_to_one=one_to_one,
            )

            assigned = np.asarray(assigned)
            matched = assigned >= 0
            if not matched.any():
                continue

            bucket = pairs.setdefault(mask_idx, ([], [], []))
            bucket[0].append(track_indices[matched])
            bucket[1].append(object_indices[assigned[matched]])
            bucket[2].append(np.asarray(distances, dtype=float)[matched])

    for mask_idx, (tracks, objects, dists) in pairs.items():
        track_rows = np.concatenate(tracks)
        object_rows = np.concatenate(objects)

        columns = _mask_measurement_columns(objects_df, mask_idx, naming)
        lab_col = label_column(mask_idx)
        if lab_col in objects_df.columns:
            columns = [*columns, lab_col]

        for column in columns:
            merged_df.loc[track_rows, column] = objects_df.loc[
                object_rows, column
            ].to_numpy()
        merged_df.loc[track_rows, distance_column(mask_idx)] = np.concatenate(
            dists
        )

    return merged_df


def zero_fill_present_frames(
    main_window,
    merged_df: pd.DataFrame,
    mask_indices,
    naming: FeatureNaming,
) -> pd.DataFrame:
    """On frames that were acquired, an unmatched track means 0, not NaN."""
    present_by_channel = getattr(main_window, "image_present", None)
    if not isinstance(present_by_channel, dict) or len(
        present_by_channel
    ) < int(main_window.n_channels):
        return merged_df

    t_series = pd.to_numeric(merged_df["t"], errors="coerce").astype("Int64")

    for channel in naming.channels:
        present_flags = present_by_channel.get(channel)
        if present_flags is None or len(present_flags) == 0:
            continue

        for mask_idx in mask_indices:
            columns = [
                c
                for c in naming.intensity_columns(channel, mask_idx)
                if c in merged_df.columns
            ]
            if not columns:
                continue

            for t_val, is_present in enumerate(present_flags):
                if not is_present:
                    continue
                rows_t = t_series == t_val
                if not rows_t.any():
                    continue
                for column in columns:
                    na_rows = rows_t & merged_df[column].isna()
                    if na_rows.any():
                        merged_df.loc[na_rows, column] = 0.0

    return merged_df
