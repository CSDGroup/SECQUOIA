"""Keep distances and second candidate columns current after an interactive edit.

`quantify` writes ``nn_dist_px_m*`` and ``alt_*_m*`` for a whole position.
An edit (right-click, paint, erase, undo, redo) changes the labels of one
frame, so the columns of that frame and mask are recomputed here with the
same rules: integer-rounded regionprops centroids, objects below
``min_mask_size`` are not candidates, and the search radius is the matching
threshold. Rows other than the edited one keep their assigned labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

from SECQUOIA.core.quantification.matching import second_nearest_within
from SECQUOIA.core.quantification.naming import (
    alt_distance_column,
    alt_label_column,
    distance_column,
    label_column,
)


def frame_centroids(
    labels2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label ids, integer-rounded ``(x, y)`` centroids and areas of one frame."""
    props = regionprops_table(
        np.asarray(labels2d).astype(np.int32, copy=False),
        properties=("label", "area", "centroid"),
    )
    ids = np.asarray(props["label"], dtype=int)
    xy = np.column_stack(
        [np.rint(props["centroid-1"]), np.rint(props["centroid-0"])]
    ).astype(float)
    return ids, xy, np.asarray(props["area"], dtype=float)


def _ensure_columns(df: pd.DataFrame, mask_idx: int) -> None:
    """Create any distance or candidate column an older table lacks."""
    if alt_label_column(mask_idx) not in df.columns:
        df[alt_label_column(mask_idx)] = pd.Series(
            0, index=df.index, dtype="Int64"
        )
    for column in (distance_column(mask_idx), alt_distance_column(mask_idx)):
        if column not in df.columns:
            df[column] = np.nan
        elif df[column].dtype.kind != "f":
            df[column] = df[column].astype(float)


def _rows_at_time(df: pd.DataFrame, t: int, row_idx) -> pd.Index:
    """Index of the rows at time `t` in the position of the edited row."""
    keep = pd.to_numeric(df["t"], errors="coerce") == t
    if "Position" in df.columns:
        position = pd.to_numeric(df["Position"], errors="coerce")
        keep &= position == position.loc[row_idx]
    return df.index[keep]


def _assigned_labels(df: pd.DataFrame, rows: pd.Index, mask_idx: int):
    """Assigned label per row, with 0 for "none"."""
    return (
        pd.to_numeric(df.loc[rows, label_column(mask_idx)], errors="coerce")
        .fillna(0)
        .astype(int)
        .to_numpy()
    )


def _refresh_assigned_distances(
    df: pd.DataFrame,
    rows: pd.Index,
    mask_idx: int,
    ids: np.ndarray,
    xy: np.ndarray,
) -> None:
    """Recompute ``nn_dist_px_m*`` for `rows` from their assigned label."""
    labels = _assigned_labels(df, rows, mask_idx)
    points = df.loc[rows, ["XMorphology", "YMorphology"]].to_numpy(float)
    position = {int(label): i for i, label in enumerate(ids)}

    distances = np.full(len(rows), np.nan)
    for i, label in enumerate(labels):
        j = position.get(int(label))
        if label > 0 and j is not None:
            distances[i] = np.hypot(
                points[i, 0] - xy[j, 0], points[i, 1] - xy[j, 1]
            )
    df.loc[rows, distance_column(mask_idx)] = distances


def _refresh_candidates(
    df: pd.DataFrame,
    rows: pd.Index,
    mask_idx: int,
    ids: np.ndarray,
    xy: np.ndarray,
    areas: np.ndarray,
    *,
    threshold: float,
    min_area: float,
) -> None:
    """Recompute ``alt_*_m*`` for every row in `rows`."""
    big = areas >= float(min_area)
    cand_ids, cand_xy = ids[big], xy[big]

    labels = _assigned_labels(df, rows, mask_idx)
    points = df.loc[rows, ["XMorphology", "YMorphology"]].to_numpy(float)
    position = {int(label): i for i, label in enumerate(cand_ids)}
    assigned_idx = np.array(
        [position.get(int(label), -1) for label in labels], dtype=int
    )

    alt_idx, alt_dist = second_nearest_within(
        points, cand_xy, assigned_idx, threshold
    )
    found = alt_idx >= 0
    alt_labels = np.zeros(len(rows), dtype=int)
    alt_labels[found] = cand_ids[alt_idx[found]]
    df.loc[rows, alt_label_column(mask_idx)] = alt_labels
    df.loc[rows, alt_distance_column(mask_idx)] = np.where(
        found, alt_dist, np.nan
    )


def refresh_distances_and_candidates(main_window, target) -> pd.Index:
    """Bring the distance and candidate columns up to date after an edit."""
    df = main_window.track_df
    mask_idx = target.mask_idx
    _ensure_columns(df, mask_idx)

    ids, xy, areas = frame_centroids(target.labels2d)

    rows = _rows_at_time(df, target.t, target.row_idx)
    if target.row_idx not in rows:
        rows = rows.append(pd.Index([target.row_idx]))

    same_label = rows[
        (_assigned_labels(df, rows, mask_idx) == target.label_id)
        & (target.label_id > 0)
    ]
    distance_rows = same_label.union(pd.Index([target.row_idx]))
    _refresh_assigned_distances(df, distance_rows, mask_idx, ids, xy)

    _refresh_candidates(
        df,
        rows,
        mask_idx,
        ids,
        xy,
        areas,
        threshold=float(getattr(main_window, "threshold", 0) or 0),
        min_area=float(getattr(main_window, "min_mask_size", 0) or 0),
    )
    return rows
