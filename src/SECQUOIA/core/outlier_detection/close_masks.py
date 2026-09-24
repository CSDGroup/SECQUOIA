"""Flag tracking points whose second potential mask is close by.

Quantification records, per tracking point and mask, the nearest other mask
within the matching tolerance (``alt_label_id_m*`` / ``alt_dist_px_m*``).

- ``OK``: nothing to review.
- ``Flagged``: some selected mask has a second candidate within the distance.
- ``Reviewed``: the user checked the case; re-running the detection keeps it.

"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from SECQUOIA.core.outlier_detection.track_sync import (
    _combo_series,
    _position_reset_mask,
    _row_match_mask,
    _shared_match_keys,
    _shared_preferred_keys,
)
from SECQUOIA.core.quantification.naming import (
    CLOSE_MASK_FLAG,
    alt_distance_column,
)

LOG = logging.getLogger(__name__)

__all__ = [
    "CLOSE_MASK_FLAG",
    "FLAG_FLAGGED",
    "FLAG_OK",
    "FLAG_REVIEWED",
    "apply_close_mask_detection",
    "close_mask_hits",
    "close_mask_row_key",
    "close_mask_summary",
    "find_close_mask_cases",
    "flagged_times",
    "is_pinned_close_mask_row",
    "mask_indices_with_candidates",
    "next_flagged_ident",
    "next_flagged_time",
    "set_close_mask_flag",
    "replay_close_mask_detection",
    "reset_close_mask_state",
    "sync_close_mask_flag_to_track_df",
    "unique_review_ids",
    "unpin_close_mask_row",
    "update_flags_after_edit",
    "update_unique_close_mask_ids",
]

FLAG_OK = "OK"
FLAG_FLAGGED = "Flagged"
FLAG_REVIEWED = "Reviewed"
_VALID_FLAGS = (FLAG_OK, FLAG_FLAGGED, FLAG_REVIEWED)

_ALT_DIST_COL_RE = re.compile(r"^alt_dist_px_m(\d+)$")


def mask_indices_with_candidates(columns: Iterable[str]) -> list[int]:
    """Mask indices that have an ``alt_dist_px_m*`` column, ascending."""
    found = {
        int(match.group(1))
        for column in columns
        if (match := _ALT_DIST_COL_RE.match(str(column)))
    }
    return sorted(found)


def close_mask_hits(
    df: pd.DataFrame, threshold_px: float, masks: Iterable[int] | None = None
) -> pd.Series:
    """Rows where a selected mask has a second candidate within `threshold_px`."""
    wanted = (
        mask_indices_with_candidates(df.columns)
        if masks is None
        else [int(m) for m in masks]
    )
    hit = pd.Series(False, index=df.index)
    for mask_idx in wanted:
        dist_col = alt_distance_column(mask_idx)
        if dist_col not in df.columns:
            continue
        close = pd.to_numeric(df[dist_col], errors="coerce") <= threshold_px
        hit |= close.fillna(False)
    return hit


def _normalised_flags(df: pd.DataFrame) -> pd.Series:
    """The flag column with anything unknown or missing read as ``OK``."""
    if CLOSE_MASK_FLAG not in df.columns:
        return pd.Series(FLAG_OK, index=df.index, dtype=object)
    flags = df[CLOSE_MASK_FLAG].astype(object)
    return flags.where(flags.isin(_VALID_FLAGS), FLAG_OK)


def find_close_mask_cases(
    df: pd.DataFrame,
    threshold_px: float,
    masks: Iterable[int] | None = None,
    *,
    keep_reviewed: bool = True,
) -> None:
    """Flag the rows with a close second mask."""
    hits = close_mask_hits(df, threshold_px, masks)
    flags = _normalised_flags(df)
    if not keep_reviewed:
        flags = flags.where(flags != FLAG_REVIEWED, FLAG_OK)

    flags[hits & (flags != FLAG_REVIEWED)] = FLAG_FLAGGED
    flags[~hits & (flags == FLAG_FLAGGED)] = FLAG_OK
    df[CLOSE_MASK_FLAG] = flags


def close_mask_summary(df: pd.DataFrame | None) -> dict[str, int]:
    """Counts of flagged and reviewed rows, and of identifications with a flag."""
    empty = {"flagged": 0, "reviewed": 0, "identifications": 0}
    if df is None or len(df) == 0 or CLOSE_MASK_FLAG not in df.columns:
        return empty

    flags = _normalised_flags(df)
    flagged = flags == FLAG_FLAGGED
    n_ids = (
        int(df.loc[flagged, "Identification"].nunique())
        if "Identification" in df.columns
        else 0
    )
    return {
        "flagged": int(flagged.sum()),
        "reviewed": int((flags == FLAG_REVIEWED).sum()),
        "identifications": n_ids,
    }


def update_unique_close_mask_ids(main_window) -> None:
    """Set ``main_window.unique_close_mask_ids`` to the identifications with a flag."""
    df = getattr(main_window, "filtered_df", None)
    if (
        df is None
        or len(df) == 0
        or CLOSE_MASK_FLAG not in df.columns
        or "Identification" not in df.columns
    ):
        main_window.unique_close_mask_ids = []
        return

    flagged = _normalised_flags(df) == FLAG_FLAGGED
    main_window.unique_close_mask_ids = (
        df.loc[flagged, "Identification"]
        .dropna()
        .astype("object")
        .drop_duplicates()
        .tolist()
    )


def sync_close_mask_flag_to_track_df(main_window) -> None:
    """Copy ``Close_mask_flag`` from ``filtered_df`` onto the matching ``track_df`` rows.

    Rows are matched on the shared Identification / t / TrackNumber /
    Position keys. Rows of the same positions with no counterpart in
    ``filtered_df`` fall back to ``OK``.
    """
    df_all = getattr(main_window, "filtered_df", None)
    track_df = getattr(main_window, "track_df", None)
    if (
        df_all is None
        or track_df is None
        or len(df_all) == 0
        or len(track_df) == 0
        or CLOSE_MASK_FLAG not in df_all.columns
    ):
        return

    left_keys, right_keys, l_pos, r_pos = _shared_preferred_keys(
        track_df, df_all
    )
    if not left_keys:
        LOG.warning(
            "[CloseMasks] No common key columns between track_df and filtered_df."
        )
        return

    scope = _position_reset_mask(track_df, df_all, l_pos, r_pos)
    flag_by_key = pd.Series(
        _normalised_flags(df_all).to_numpy(),
        index=_combo_series(df_all, right_keys),
    )
    flag_by_key = flag_by_key[~flag_by_key.index.duplicated(keep="last")]

    keys = _combo_series(track_df.loc[scope], left_keys)
    synced = keys.map(flag_by_key).fillna(FLAG_OK)

    if CLOSE_MASK_FLAG not in track_df.columns:
        track_df[CLOSE_MASK_FLAG] = pd.Series(
            FLAG_OK, index=track_df.index, dtype=object
        )
    track_df[CLOSE_MASK_FLAG] = track_df[CLOSE_MASK_FLAG].astype(object)
    track_df.loc[scope, CLOSE_MASK_FLAG] = synced.to_numpy()
    main_window.track_df = track_df


def apply_close_mask_detection(
    main_window,
    threshold_px: float,
    masks: Iterable[int] | None = None,
) -> int:
    """Detect close-mask cases in ``filtered_df`` and update every dependant.

    Sets the flag column, mirrors it into ``track_df`` and rebuilds
    ``unique_close_mask_ids``; ``Reviewed`` rows stay so. The distance and
    masks are remembered on the main window for the napari view. Returns the
    number of ``Flagged`` rows.
    """
    df = getattr(main_window, "filtered_df", None)
    if df is None or len(df) == 0:
        main_window.unique_close_mask_ids = []
        return 0

    main_window.close_mask_threshold = float(threshold_px)
    main_window.close_mask_masks = (
        [int(m) for m in masks] if masks is not None else None
    )

    find_close_mask_cases(df, threshold_px, masks)

    sync_close_mask_flag_to_track_df(main_window)
    update_unique_close_mask_ids(main_window)
    return close_mask_summary(df)["flagged"]


def close_mask_row_key(row: pd.Series) -> tuple[str, int, int] | None:
    """``(Identification, TrackNumber, t)`` of `row`, the identity of one flag."""
    try:
        return (
            str(row["Identification"]),
            int(row["TrackNumber"]),
            int(row["t"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def is_pinned_close_mask_row(main_window, row: pd.Series) -> bool:
    """True for the reviewed row that an edit just dealt with."""
    pinned = getattr(main_window, "close_mask_pinned_row", None)
    return (
        pinned is not None
        and row.get(CLOSE_MASK_FLAG) == FLAG_REVIEWED
        and close_mask_row_key(row) == pinned
    )


def unpin_close_mask_row(main_window) -> None:
    """Forget the row kept on show after an edit."""
    main_window.close_mask_pinned_row = None


def update_flags_after_edit(main_window, rows: pd.Index, edited_row) -> None:
    """Re-evaluate the flags of `rows` in ``track_df`` after an interactive edit."""
    df = main_window.track_df
    if CLOSE_MASK_FLAG not in df.columns or len(rows) == 0:
        return

    threshold = getattr(main_window, "close_mask_threshold", None)
    masks = getattr(main_window, "close_mask_masks", None)
    subset = df.loc[rows]
    before = _normalised_flags(subset)
    after = before.copy()

    if threshold is not None:
        hits = close_mask_hits(subset, float(threshold), masks)
        after[hits & (before != FLAG_REVIEWED)] = FLAG_FLAGGED
    else:
        hits = close_mask_hits(subset, float("inf"), masks)
    after[~hits & (before == FLAG_FLAGGED)] = FLAG_OK

    if edited_row in after.index and before.loc[edited_row] == FLAG_FLAGGED:
        after.loc[edited_row] = FLAG_REVIEWED
        main_window.close_mask_pinned_row = close_mask_row_key(
            df.loc[edited_row]
        )

    df[CLOSE_MASK_FLAG] = df[CLOSE_MASK_FLAG].astype(object)
    df.loc[rows, CLOSE_MASK_FLAG] = after.to_numpy()


def set_close_mask_flag(main_window, filtered_idxs, value: str) -> None:
    """Set ``Close_mask_flag`` to `value` on ``filtered_df`` rows and their ``track_df`` twins."""
    df_all = getattr(main_window, "filtered_df", None)
    track_df = getattr(main_window, "track_df", None)
    if df_all is None or CLOSE_MASK_FLAG not in df_all.columns:
        return

    idxs = list(filtered_idxs)
    df_all[CLOSE_MASK_FLAG] = df_all[CLOSE_MASK_FLAG].astype(object)
    df_all.loc[idxs, CLOSE_MASK_FLAG] = value

    if track_df is None or len(track_df) == 0:
        return
    if CLOSE_MASK_FLAG not in track_df.columns:
        track_df[CLOSE_MASK_FLAG] = pd.Series(
            FLAG_OK, index=track_df.index, dtype=object
        )
    track_df[CLOSE_MASK_FLAG] = track_df[CLOSE_MASK_FLAG].astype(object)

    keys = _shared_match_keys(df_all, track_df)
    for idx in idxs:
        track_df.loc[
            _row_match_mask(df_all, idx, track_df, keys), CLOSE_MASK_FLAG
        ] = value
    main_window.track_df = track_df


def flagged_times(df: pd.DataFrame, ident) -> np.ndarray:
    """Sorted time points at which `ident` has a ``Flagged`` row."""
    if CLOSE_MASK_FLAG not in df.columns or "Identification" not in df.columns:
        return np.array([], dtype=int)
    keep = (df["Identification"].astype(str) == str(ident)) & (
        _normalised_flags(df) == FLAG_FLAGGED
    )
    times = pd.to_numeric(df.loc[keep, "t"], errors="coerce").dropna()
    return np.unique(times.to_numpy(dtype=int))


def next_flagged_time(times: np.ndarray, current: int) -> int | None:
    """The first of `times` after `current`, wrapping to the first overall."""
    if len(times) == 0:
        return None
    later = times[times > current]
    return int(later[0] if later.size else times[0])


def next_flagged_ident(old_ids: list, current, remaining_ids: list):
    """The identification to continue with once `current` has no flags left."""
    remaining = {str(ident) for ident in remaining_ids}
    if not remaining:
        return None

    order = [str(ident) for ident in old_ids]
    start = order.index(str(current)) + 1 if str(current) in order else 0
    for ident in order[start:] + order[:start]:
        if ident in remaining and ident != str(current):
            return next(i for i in remaining_ids if str(i) == ident)
    return None


def _as_list(values) -> list:
    """`values` (list, array, scalar or None) as a plain list."""
    if values is None:
        return []
    try:
        return list(values)
    except TypeError:
        return [values]


def unique_review_ids(main_window) -> list:
    """The identifications the Out list shows: outliers, then close-mask cases."""
    seen: set[str] = set()
    idents: list = []
    for ident in (
        *_as_list(getattr(main_window, "unique_outliers_ids", None)),
        *_as_list(getattr(main_window, "unique_close_mask_ids", None)),
    ):
        if str(ident) not in seen:
            seen.add(str(ident))
            idents.append(ident)
    return idents


def reset_close_mask_state(main_window) -> None:
    """Remove every close-mask flag, ``Reviewed`` ones included, and forget the run."""
    for attr in ("filtered_df", "track_df"):
        df = getattr(main_window, attr, None)
        if isinstance(df, pd.DataFrame) and CLOSE_MASK_FLAG in df.columns:
            df[CLOSE_MASK_FLAG] = pd.Series(
                FLAG_OK, index=df.index, dtype=object
            )

    main_window.unique_close_mask_ids = []
    main_window.close_mask_threshold = None
    main_window.close_mask_masks = None
    main_window.close_mask_pinned_row = None


def replay_close_mask_detection(main_window, pack) -> int:
    """Detect close masks again with the settings saved in the rules `pack`."""
    settings = getattr(pack, "close_masks", None)
    if settings is None:
        return 0
    return apply_close_mask_detection(
        main_window, settings.distance, settings.masks
    )
