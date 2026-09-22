"""Nearest neighbour matching of measured objects onto existing tracks."""

import logging
import os

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from SECQUOIA.core.quantification.naming import (
    FeatureNaming,
    alt_distance_column,
    alt_label_column,
    distance_column,
    label_column,
)

LOG = logging.getLogger(__name__)


MATCH_STRATEGIES = ("legacy", "sorted_pairs", "optimal")
DEFAULT_MATCH_STRATEGY = "sorted_pairs"
MATCH_STRATEGY_ENV = "SECQUOIA_MATCH_STRATEGY"
COMPARE_MATCHING_ENV = "SECQUOIA_COMPARE_MATCHING"


def resolve_match_strategy(requested: str | None = None) -> str:
    """The strategy to use: `requested`, else the environment variable, else the default."""
    strategy = (
        requested
        or os.environ.get(MATCH_STRATEGY_ENV, "").strip().lower()
        or DEFAULT_MATCH_STRATEGY
    )
    if strategy not in MATCH_STRATEGIES:
        raise ValueError(
            f"Unknown match strategy {strategy!r}; "
            f"expected one of {MATCH_STRATEGIES}."
        )
    return strategy


def compare_matching_enabled() -> bool:
    """True when the developer switch for comparing the strategies is on."""
    value = os.environ.get(COMPARE_MATCHING_ENV, "").strip().lower()
    return value in ("1", "true", "yes", "on")


# Per mask geometry columns precreated on the tracks before matching.
GEOMETRY_PREFIXES = (
    "XMorphology",
    "YMorphology",
    "AreaMorphology",
    "PerimeterMorphology",
    "Orientation",
    "Eccentricity",
)


def _distance_matrix(
    tracks_xy: np.ndarray, objects_xy: np.ndarray
) -> np.ndarray:
    """Euclidean distances, one row per track and one column per object."""
    return np.hypot(
        tracks_xy[:, [0]] - objects_xy[:, 0][None, :],
        tracks_xy[:, [1]] - objects_xy[:, 1][None, :],
    )


def _sorted_pairs_assign(
    distances: np.ndarray, max_dist: float
) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one assignment over every (track, object) pair within `max_dist`.

    Pairs are taken shortest first and skipped when their track or their
    object is already used. Ties break by track index, then object index.
    """
    n_tracks = distances.shape[0]
    assigned = np.full(n_tracks, -1, dtype=int)
    assigned_dist = np.full(n_tracks, np.inf)

    track_idx, object_idx = np.nonzero(distances <= max_dist)
    order = np.lexsort(
        (object_idx, track_idx, distances[track_idx, object_idx])
    )

    used_objects: set[int] = set()
    for k in order:
        track_i, object_j = int(track_idx[k]), int(object_idx[k])
        if assigned[track_i] >= 0 or object_j in used_objects:
            continue
        used_objects.add(object_j)
        assigned[track_i] = object_j
        assigned_dist[track_i] = distances[track_i, object_j]

    return assigned, assigned_dist


def _optimal_assign(
    distances: np.ndarray, max_dist: float
) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one assignment with the most matches, then the smallest total distance.

    Only pairs within `max_dist` may be matched. Every other pair is given a
    cost above the total that any set of allowed pairs can reach, so the
    solver never trades an allowed match for a forbidden one and drops
    forbidden pairs from the result afterwards.
    """
    n_tracks, n_objects = distances.shape
    allowed = distances <= max_dist
    forbidden = max_dist * min(n_tracks, n_objects) + 1.0
    rows, cols = linear_sum_assignment(np.where(allowed, distances, forbidden))
    keep = allowed[rows, cols]

    assigned = np.full(n_tracks, -1, dtype=int)
    assigned_dist = np.full(n_tracks, np.inf)
    assigned[rows[keep]] = cols[keep]
    assigned_dist[rows[keep]] = distances[rows[keep], cols[keep]]
    return assigned, assigned_dist


def greedy_nearest_assign(
    tracks_xy: np.ndarray,
    objects_xy: np.ndarray,
    max_dist: float,
    *,
    one_to_one: bool = True,
    strategy: str = DEFAULT_MATCH_STRATEGY,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each track to its nearest object within `max_dist`.

    Returns ``(object_index_per_track, distance_per_track)`` where an
    index of -1 and a distance of inf mean "unmatched". With
    `one_to_one`, each object is used at most once, resolved by `strategy`:

    - ``"legacy"``: each track proposes its nearest object; proposals are
      processed shortest first, and a track whose object is already taken
      falls back to the closest free one within `max_dist`. The fallback
      pair is not re-sorted against later proposals, so in a chain of
      conflicts an object can go to a farther track.
    - ``"sorted_pairs"``: every pair within `max_dist` is sorted by
      distance and assigned from the shortest, skipping used tracks and
      objects. Ties break by track index, then object index.
    - ``"optimal"``: the assignment with the most matches within `max_dist`
      and, among those, the smallest total distance (the Hungarian method).
      It can give a track a farther object so that another track is not left
      without one, which the two greedy strategies never do.

    """
    if strategy not in MATCH_STRATEGIES:
        raise ValueError(
            f"Unknown match strategy {strategy!r}; "
            f"expected one of {MATCH_STRATEGIES}."
        )
    n_tracks, n_objects = tracks_xy.shape[0], objects_xy.shape[0]
    if n_tracks == 0 or n_objects == 0:
        return np.full(n_tracks, -1, dtype=int), np.full(n_tracks, np.inf)

    distances = _distance_matrix(tracks_xy, objects_xy)
    nearest_idx = np.argmin(distances, axis=1)
    nearest_dist = distances[np.arange(n_tracks), nearest_idx]

    if not one_to_one:
        too_far = nearest_dist > max_dist
        nearest_idx[too_far] = -1
        nearest_dist[too_far] = np.inf
        return nearest_idx, nearest_dist

    if strategy == "sorted_pairs":
        return _sorted_pairs_assign(distances, max_dist)
    if strategy == "optimal":
        return _optimal_assign(distances, max_dist)

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


def second_nearest_within(
    tracks_xy: np.ndarray,
    objects_xy: np.ndarray,
    assigned_idx: np.ndarray,
    max_dist: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest object other than the assigned one, within `max_dist`.

    `assigned_idx` holds the assigned object per track (-1 when the track
    is unmatched, in which case nothing is excluded). Objects claimed by
    another track still count as candidates. Returns
    ``(object_index_per_track, distance_per_track)`` with -1 and inf where
    no other object lies within `max_dist`. Ties go to the lower object
    index, so the result is deterministic.
    """
    n_tracks, n_objects = tracks_xy.shape[0], objects_xy.shape[0]
    if n_tracks == 0 or n_objects == 0:
        return np.full(n_tracks, -1, dtype=int), np.full(n_tracks, np.inf)

    distances = _distance_matrix(tracks_xy, objects_xy)
    assigned_idx = np.asarray(assigned_idx, dtype=int)
    has_assigned = assigned_idx >= 0
    distances[np.flatnonzero(has_assigned), assigned_idx[has_assigned]] = (
        np.inf
    )

    candidate_idx = np.argmin(distances, axis=1)
    candidate_dist = distances[np.arange(n_tracks), candidate_idx]

    too_far = ~(candidate_dist <= max_dist)
    candidate_idx[too_far] = -1
    candidate_dist[too_far] = np.inf
    return candidate_idx, candidate_dist


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


def _reset_alt_columns(merged_df: pd.DataFrame, mask_indices) -> None:
    """Clear the second candidate columns so stale values never survive a rematch."""
    for mask_idx in mask_indices:
        merged_df[alt_label_column(mask_idx)] = pd.Series(
            0, index=merged_df.index, dtype="Int64"
        )
        merged_df[alt_distance_column(mask_idx)] = np.nan


def assign_objects_to_tracks(
    merged_df: pd.DataFrame,
    objects_df: pd.DataFrame,
    mask_indices,
    naming: FeatureNaming,
    *,
    max_pixel_distance: float,
    one_to_one: bool,
    strategy: str = DEFAULT_MATCH_STRATEGY,
) -> pd.DataFrame:
    """Copy object measurements onto their nearest track, per time point and mask."""
    pairs: dict[int, tuple[list, list, list]] = {}
    alt_pairs: dict[int, tuple[list, list, list]] = {}
    _reset_alt_columns(merged_df, mask_indices)
    other_strategies = (
        [name for name in MATCH_STRATEGIES if name != strategy]
        if one_to_one and compare_matching_enabled()
        else []
    )
    n_differ = dict.fromkeys(other_strategies, 0)

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
                strategy=strategy,
            )

            assigned = np.asarray(assigned)
            for name in other_strategies:
                other, _ = greedy_nearest_assign(
                    tracks_xy,
                    objects_xy,
                    max_pixel_distance,
                    one_to_one=True,
                    strategy=name,
                )
                n_differ[name] += int(
                    np.count_nonzero(np.asarray(other) != assigned)
                )

            alt_assigned, alt_distances = second_nearest_within(
                tracks_xy, objects_xy, assigned, max_pixel_distance
            )
            has_alt = alt_assigned >= 0
            if has_alt.any():
                alt_bucket = alt_pairs.setdefault(mask_idx, ([], [], []))
                alt_bucket[0].append(track_indices[has_alt])
                alt_bucket[1].append(object_indices[alt_assigned[has_alt]])
                alt_bucket[2].append(alt_distances[has_alt])

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

    for mask_idx, (tracks, objects, dists) in alt_pairs.items():
        lab_col = label_column(mask_idx)
        if lab_col not in objects_df.columns:
            continue
        track_rows = np.concatenate(tracks)
        object_rows = np.concatenate(objects)
        merged_df.loc[track_rows, alt_label_column(mask_idx)] = objects_df.loc[
            object_rows, lab_col
        ].to_numpy()
        merged_df.loc[track_rows, alt_distance_column(mask_idx)] = (
            np.concatenate(dists)
        )

    if other_strategies:
        LOG.info(
            "[matching] strategy=%s: track assignments that would differ: %s",
            strategy,
            ", ".join(f"{n} under {name}" for name, n in n_differ.items()),
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
