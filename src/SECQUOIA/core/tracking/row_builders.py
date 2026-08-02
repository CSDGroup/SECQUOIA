"""Building and aligning the blank stub rows lineage edits insert."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from SECQUOIA.config import TRACKING
from SECQUOIA.core.tracking.numbering import _forward_scope_mask
from SECQUOIA.utils.timing import realtime_columns


def _ensure_columns(
    df: pd.DataFrame, required_cols: list[str]
) -> pd.DataFrame:
    """Add missing columns and return df."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = 0
    return df


def _align_like(df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
    """Return df with exactly target_cols, filling missing with 0."""
    df2 = df.copy()
    for c in target_cols:
        if c not in df2.columns:
            df2[c] = 0
    return df2.loc[:, target_cols]


def _add_stub_rows_for_t_range(
    base_cols: list[str],
    ident: str,
    t_range: Iterable[int],
    tracknumber: int,
    position_number: int,
    time_interval: float,
    has_tidx: bool,
    has_tfile: bool,
    track_id_val: int | None = None,
    t_file_value: int = 0,
) -> pd.DataFrame:
    """Create blank rows for a time range with the standard defaults set.

    Every measurement column starts at 0; only the bookkeeping columns
    (Identification, t, TrackNumber, Position, Cellfate, active, inspected,
    Calculated_Time, track_id) are filled in.
    """
    rows = []
    for t in t_range:
        row = dict.fromkeys(base_cols, 0)
        row["Identification"] = ident
        row["t"] = int(t)
        row["TrackNumber"] = int(tracknumber)
        if has_tidx and "t_idx" in base_cols:
            row["t_idx"] = int(t)
        if has_tfile and "t_file" in base_cols:
            row["t_file"] = int(t_file_value)
        if "Position" in base_cols:
            row["Position"] = int(position_number)
        if "Cellfate" in base_cols:
            row["Cellfate"] = "Healthy"
        if "active" in base_cols:
            row["active"] = 1
        if "inspected" in base_cols:
            row["inspected"] = 0
        if "Calculated_Time" in base_cols:
            row["Calculated_Time"] = (int(t) * float(time_interval)) / 60.0
        if track_id_val is not None and "track_id" in base_cols:
            row["track_id"] = int(track_id_val)
        rows.append(row)
    return pd.DataFrame(rows, columns=base_cols)


def _append_rows_to_track_df(main_window, new_df: pd.DataFrame) -> None:
    """Append ``new_df`` to ``main_window.track_df``, reconciling columns."""
    track_df = getattr(main_window, "track_df", None)
    if track_df is None or track_df.empty:
        main_window.track_df = new_df.copy()
        return
    track_df = _ensure_columns(track_df, list(new_df.columns))
    aligned = _align_like(new_df, list(track_df.columns))
    main_window.track_df = pd.concat([track_df, aligned], ignore_index=True)


def _zero_numeric_except(df: pd.DataFrame, mask: pd.Series) -> None:
    """Zero numeric columns."""
    safe_keep = set(TRACKING.SAFE_KEEP_NUMERIC) | set(
        realtime_columns(df.columns)
    )

    if not isinstance(mask, pd.Series):
        raise TypeError(
            f"_zero_numeric_except: mask must be a pandas Series, got {type(mask)!r}"
        )
    if mask.dtype != bool:
        raise TypeError(
            "_zero_numeric_except: mask must be boolean dtype, got "
            f"{mask.dtype!r}. Wrap the comparison in .fillna(False).astype(bool)."
        )
    if not mask.index.equals(df.index):
        raise ValueError(
            "_zero_numeric_except: mask index does not match df index "
            f"(mask={len(mask)} rows, df={len(df)} rows). The mask was built "
            "from a different DataFrame."
        )

    if not mask.any():
        return
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in safe_keep]
    if num_cols:
        df.loc[mask, num_cols] = 0


def _stub_rows_for_missing_slots(
    base_cols: list[str],
    ident: str,
    t_range: Iterable[int],
    tracknumbers: Iterable[int],
    existing: set[tuple[int, int]],
    *,
    position_number: int,
    time_interval: float,
    track_id_val=None,
) -> pd.DataFrame | None:
    """Blank rows for every ``(t, TrackNumber)`` slot not already present."""
    frames = []
    times = list(t_range)
    for tracknumber in tracknumbers:
        missing = [t for t in times if (t, tracknumber) not in existing]
        if not missing:
            continue
        frames.append(
            _add_stub_rows_for_t_range(
                base_cols,
                ident,
                missing,
                tracknumber,
                position_number,
                time_interval,
                "t_idx" in base_cols,
                "t_file" in base_cols,
                track_id_val,
            )
        )
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


def _leave_blank_continuation(
    df: pd.DataFrame,
    ident: str,
    t_from: int,
    t_max: int,
    tracknumber: int,
    *,
    position_number: int,
    time_interval: float,
    track_id_val=None,
    keep_active: bool,
) -> pd.DataFrame:
    """Give ``tracknumber`` a blank row for every frame from ``t_from`` on.

    Missing frames are created as stubs and every frame from ``t_from`` is
    then blanked, so the track still exists in the lineage but carries no
    measurements it did not earn.
    """
    base_cols = list(df.columns)
    stub = _add_stub_rows_for_t_range(
        base_cols,
        ident,
        range(int(t_from), int(t_max) + 1),
        tracknumber,
        position_number,
        time_interval,
        "t_idx" in base_cols,
        "t_file" in base_cols,
        track_id_val,
    )
    if not stub.empty:
        df = pd.concat([df, _align_like(stub, base_cols)], ignore_index=True)

    scope = _forward_scope_mask(df, ident, t_from, tracknumber)
    _zero_numeric_except(df, scope)
    if keep_active and "active" in df.columns:
        df.loc[scope, "active"] = 1
    return df


def _backfill_root_rows(
    df_move: pd.DataFrame,
    new_ident: str,
    t_min: int,
    t_split: int,
    *,
    position_number: int,
    time_interval: float,
    track_id_val=None,
) -> pd.DataFrame:
    """Prepend blank root rows so a split-off lineage starts at ``t_min``."""
    if t_min >= t_split:
        return df_move

    base_cols = list(df_move.columns)
    already_rooted = (df_move["Identification"].astype(str) == new_ident) & (
        df_move["TrackNumber"] == 1
    )
    covered = set(df_move.loc[already_rooted, "t"].tolist())
    if len(covered) == (t_split - t_min):
        return df_move

    pre_df = _add_stub_rows_for_t_range(
        base_cols,
        new_ident,
        range(int(t_min), int(t_split)),
        1,
        position_number,
        time_interval,
        "t_idx" in base_cols,
        "t_file" in base_cols,
        track_id_val,
    )
    if pre_df.empty:
        return df_move
    return pd.concat([pre_df, df_move], ignore_index=True)
