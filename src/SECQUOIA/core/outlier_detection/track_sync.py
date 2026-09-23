"""Syncing Outlier_detection flags between main_window.filtered_df and main_window.track_df."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

__all__ = [
    "outlier_times",
    "update_outlier_detection_in_track_df",
    "update_outlier_detection_in_track_df_fast",
    "update_unique_outliers_ids",
]


def _pick_if_both(
    left_df, right_df, left_name: str, right_name: str
) -> tuple[str | None, str | None]:
    """Return (left_col, right_col) only if both exist; otherwise (None, None)."""
    if left_name in left_df.columns and right_name in right_df.columns:
        return left_name, right_name
    return None, None


def _shared_preferred_keys(track_df, df_all):
    """Return (left_keys, right_keys, l_pos, r_pos) for the columns shared by track_df and filtered_df.

    The keys are the Identification/t/TrackNumber/Position columns present in
    both frames.
    """
    names = ("Identification", "t", "TrackNumber", "Position")
    pairs = [_pick_if_both(track_df, df_all, name, name) for name in names]

    left_keys = [L for L, R in pairs if L and R]
    right_keys = [R for L, R in pairs if L and R]
    l_pos, r_pos = pairs[-1]
    return left_keys, right_keys, l_pos, r_pos


def _position_reset_mask(track_df, df_all, l_pos, r_pos) -> pd.Series:
    """Boolean mask of track_df rows to reset to 'OK': every row if Position isn't a shared key, else only rows whose Position appears in filtered_df."""
    if not (l_pos and r_pos):
        return pd.Series(True, index=track_df.index)

    pos_values = df_all[r_pos].dropna().unique().tolist()
    if not pos_values:
        return pd.Series(True, index=track_df.index)

    return track_df[l_pos].isin(pos_values)


def _combo_series(
    df: pd.DataFrame, cols: list[str], sep: str = "\x1f"
) -> pd.Series:
    """Join `cols` per row into one separator-delimited string, for set-based row matching."""
    return df[cols].astype("string").fillna("").agg(sep.join, axis=1)


def update_outlier_detection_in_track_df(
    main_window, outcol: str = "Outlier_detection"
) -> None:
    """Sync the Outlier flags from main_window.filtered_df -> main_window.track_df."""
    df_all = getattr(main_window, "filtered_df", None)
    track_df = getattr(main_window, "track_df", None)
    if (
        df_all is None
        or track_df is None
        or len(df_all) == 0
        or len(track_df) == 0
    ):
        LOG.warning(
            "[UpdateTrackDF] Nothing to update: filtered_df or track_df is empty."
        )
        return

    left_keys, right_keys, l_pos, r_pos = _shared_preferred_keys(
        track_df, df_all
    )
    if not left_keys:
        LOG.warning(
            "[UpdateTrackDF] No common preferred key columns found between track_df and filtered_df."
        )
        return

    if outcol not in track_df.columns:
        track_df[outcol] = "OK"

    reset_mask = _position_reset_mask(track_df, df_all, l_pos, r_pos)
    track_df.loc[reset_mask, outcol] = "OK"

    if outcol not in df_all.columns:
        LOG.warning(
            "[UpdateTrackDF] Column '%s' not found in filtered_df.", outcol
        )
        main_window.track_df = track_df
        return

    df_out = df_all[df_all[outcol] == "Outlier"]
    if df_out.empty:
        LOG.info(
            "[UpdateTrackDF] No outliers in filtered_df; track_df reset to OK for target subset."
        )
        main_window.track_df = track_df
        return

    right_combo_set = set(_combo_series(df_out, right_keys).unique())
    full_left_combo = _combo_series(track_df, left_keys)

    mark_mask = reset_mask & full_left_combo.isin(right_combo_set)
    track_df.loc[mark_mask, outcol] = "Outlier"
    main_window.track_df = track_df

    n_marked = int(mark_mask.sum())
    n_reset = int(reset_mask.sum())
    LOG.info(
        "[UpdateTrackDF] Reset %d row(s) to 'OK'; marked %d row(s) as 'Outlier'.",
        n_reset,
        n_marked,
    )


def update_unique_outliers_ids(
    main_window, *, outcol: str = "Outlier_detection"
) -> None:
    """Build main_window.unique_outliers_ids from main_window.filtered_df by
    collecting Identification values that have at least one 'Outlier' in `outcol`.
    """
    df = getattr(main_window, "filtered_df", None)
    if df is None or len(df) == 0:
        main_window.unique_outliers_ids = []
        return

    if outcol not in df.columns:
        LOG.warning(
            "[update_unique_outliers_ids] Column '%s' not found in filtered_df.",
            outcol,
        )
        main_window.unique_outliers_ids = []
        return

    # Pick the ID column name
    id_candidates = ["Identification", "identification"]
    id_col = next((c for c in id_candidates if c in df.columns), None)
    if id_col is None:
        LOG.warning(
            "[update_unique_outliers_ids] No Identification column found."
        )
        main_window.unique_outliers_ids = []
        return

    mask = df[outcol] == "Outlier"
    if not mask.any():
        main_window.unique_outliers_ids = []
        return

    ids = (
        df.loc[mask, id_col]
        .dropna()
        .astype("object")
        .drop_duplicates()
        .tolist()
    )

    main_window.unique_outliers_ids = ids


def outlier_times(
    df: pd.DataFrame, ident, *, outcol: str = "Outlier_detection"
) -> np.ndarray:
    """Sorted time points at which `ident` has an ``Outlier`` row."""
    if outcol not in df.columns or "Identification" not in df.columns:
        return np.array([], dtype=int)
    keep = (df["Identification"].astype(str) == str(ident)) & (
        df[outcol] == "Outlier"
    )
    times = pd.to_numeric(df.loc[keep, "t"], errors="coerce").dropna()
    return np.unique(times.to_numpy(dtype=int))


def _shared_match_keys(df_all, track_df) -> list[str]:
    """Return the preferred id/time/track/position columns present in both dataframes."""
    return [
        k
        for k in ["Identification", "t", "TrackNumber", "Position"]
        if k in df_all.columns and k in track_df.columns
    ]


def _column_match_mask(value, series: pd.Series) -> pd.Series:
    """Boolean mask of `series` entries equal to `value`, coercing to `series`'s dtype."""
    if pd.isna(value):
        return series.isna()

    if pd.api.types.is_integer_dtype(series):
        try:
            coerced = int(value)
        except (TypeError, ValueError):
            coerced = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[
                0
            ]
        return series.eq(coerced)

    if pd.api.types.is_float_dtype(series):
        try:
            coerced = float(value)
        except (TypeError, ValueError):
            coerced = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[
                0
            ]
        return series.eq(coerced)

    try:
        return series.eq(value)
    except (TypeError, ValueError):
        return series.astype("string").eq(str(value))


def _row_match_mask(df_all, idx, track_df, keys: list[str]) -> pd.Series:
    """Boolean mask of `track_df` rows matching `df_all`'s row `idx` on every column in `keys`."""
    m = pd.Series(True, index=track_df.index)
    for k in keys:
        v = df_all.at[idx, k] if k in df_all.columns else None
        m &= _column_match_mask(v, track_df[k])
    return m


def update_outlier_detection_in_track_df_fast(
    main_window,
    changed_filtered_df_idxs,
    outcol: str = "Outlier_detection",
):
    """Update only the track_df rows matching the given filtered_df indices.

    The full sync in `update_outlier_detection_in_track_df` rewrites every
    row of the position; this touches just the points that changed.
    """
    df_all = getattr(main_window, "filtered_df", None)
    track_df = getattr(main_window, "track_df", None)

    keys = _shared_match_keys(df_all, track_df)

    if outcol not in track_df.columns:
        track_df[outcol] = "OK"

    if isinstance(changed_filtered_df_idxs, list | tuple):
        idxs = list(changed_filtered_df_idxs)
    else:
        idxs = [changed_filtered_df_idxs]

    touched = 0
    for idx in idxs:
        m = _row_match_mask(df_all, idx, track_df, keys)
        if m.any():
            track_df.loc[m, outcol] = df_all.at[idx, outcol]
            touched += int(m.sum())

    main_window.track_df = track_df
    LOG.info(
        "[UpdateTrackDF:fast] Updated %d point(s); touched %d track_df row(s).",
        len(idxs),
        touched,
    )
