"""Lineage gap filling for the track DataFrame."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from SECQUOIA.core.tracking.context import _get_time_interval
from SECQUOIA.utils.timing import current_t_range as _current_t_range

__all__ = [
    "CoordinatePolicy",
    "GapFillContext",
    "add_missing_rows",
    "child_tracks",
    "ensure_track_schema",
    "fill_missing_frames",
    "sibling_group",
    "track_windows",
]

LOG = logging.getLogger(__name__)


class CoordinatePolicy(str, Enum):
    """Where a synthetic row's coordinates come from.

    SENTINEL leaves ``XMorphology``/``YMorphology`` at 0, marking the row as
    having no real position. MEDIAN borrows the median of the track's own
    measured coordinates, falling back to its sisters, then to the whole
    Identification.
    """

    SENTINEL = "sentinel"
    MEDIAN = "median"


# Natural key of a track row.
KEY_COLUMNS = ("Position", "Identification", "t", "TrackNumber")

# Columns that must exist before gap filling, with their default value.
_SCHEMA_DEFAULTS: dict[str, Any] = {
    "Position": 0,
    "Identification": 0,
    "t": 0,
    "XMorphology": 0,
    "YMorphology": 0,
    "active": 1,
    "Cellfate": "",
}

# Columns that need a specific dtype.
_SCHEMA_NULLABLE: dict[str, str] = {
    "track_id": "Int64",
    "Calculated_Time": "float",
}

# Value written into ``Cellfate``.
SYNTHETIC_CELLFATE = "Healthy"

# Matches an intensity/measurement column, e.g. ``MeanNoBgCorrectedCh00M1``.
_INTENSITY_COL_RE = re.compile(r"Ch(\d+)M(\d+)$")

# Matches the numeric tail of an Identification, e.g. ``..._p0002-3`` -> 3.
_IDENT_TAIL_RE = re.compile(r"-([0-9]+)$")


def _null_series(n: int, dtype: str, index=None) -> pd.Series:
    """Return an all missing Series of ``dtype`` (pandas rejects ``pd.NA`` for floats)."""
    if dtype == "Int64":
        values = pd.array([pd.NA] * n, dtype="Int64")
    else:
        values = np.full(n, np.nan, dtype=float)
    return pd.Series(values, index=index, dtype=dtype)


def ensure_track_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``df`` with every column the filler needs, added with typed defaults."""
    out = df.copy()
    for col, default in _SCHEMA_DEFAULTS.items():
        if col not in out.columns:
            out[col] = default
    for col, dtype in _SCHEMA_NULLABLE.items():
        if col not in out.columns:
            out[col] = _null_series(len(out), dtype, out.index)
    return out


@dataclass(frozen=True)
class GapFillContext:
    """Everything the filler needs from ``main_window``, with no Qt dependency."""

    position: int
    frames: range
    time_interval_s: float = 0.0
    channels: tuple[str, ...] = ()
    presence: dict[str, Any] = field(default_factory=dict)
    mask_absent_frames: bool = True
    coordinates: CoordinatePolicy = CoordinatePolicy.SENTINEL

    @classmethod
    def from_main_window(cls, main_window) -> GapFillContext:
        """Read position, frame range, frame interval and channel presence off the window."""
        try:
            t_file_min, t_file_max, _t_idx_min, _t_idx_max = _current_t_range(
                main_window
            )
            span = int(t_file_max) - int(t_file_min)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"current_t_range failed: {exc}") from exc

        try:
            interval = float(_get_time_interval(main_window) or 0.0)
        except (TypeError, ValueError):
            interval = 0.0

        presence = getattr(main_window, "image_present", None)
        if not isinstance(presence, dict):
            presence = {}

        return cls(
            position=int(main_window.current_position_number),
            frames=range(max(span, 0) + 1),
            time_interval_s=interval,
            channels=tuple(getattr(main_window, "ids_channels", ()) or ()),
            presence=presence,
        )

    def calculated_time(self, t: int) -> float:
        """Acquisition time of frame ``t`` in minutes."""
        return (int(t) * self.time_interval_s) / 60.0

    def presence_flags(self, channel_suffix: str):
        """Per frame acquisition flags for the channel behind ``Ch<suffix>`` columns."""
        for channel in self.channels:
            if str(channel)[1:] == channel_suffix:
                flags = self.presence.get(channel)
                return flags if flags is not None and len(flags) else None
        return None

    def frame_is_absent(self, channel_suffix: str, t: int) -> bool:
        """Return True only when we positively know the frame was not acquired."""
        flags = self.presence_flags(channel_suffix)
        if flags is None:
            return False
        if 0 <= t < len(flags):
            return not bool(flags[t])
        if 1 <= t <= len(flags):
            return not bool(flags[t - 1])
        return False


def _normalise_track(track_number) -> int | None:
    """Track number as a plain ``int``, or ``None`` when absent/unparsable."""
    if track_number is None:
        return None
    n = pd.to_numeric(track_number, errors="coerce")
    return None if pd.isna(n) else int(n)


def child_tracks(track_number) -> tuple[int, ...]:
    """Return the two tracks this one divides into (``2n`` and ``2n + 1``)."""
    n = _normalise_track(track_number)
    return () if n is None or n < 1 else (2 * n, 2 * n + 1)


def sibling_group(track_number) -> tuple:
    """Return the sister pair ``track_number`` belongs to: 1 alone, then (2,3), (4,5), ..."""
    n = _normalise_track(track_number)
    if n is None or n <= 1:
        return (n,)
    even = n if n % 2 == 0 else n - 1
    return (even, even + 1)


def track_windows(
    spans: dict[Any, tuple[int, int]], frames: range
) -> dict[Any, range]:
    """Frame range each track may be filled over."""
    if not frames or not spans:
        return {}

    range_min, range_max = frames.start, frames.stop - 1
    earliest = min(first for first, _last in spans.values())

    windows: dict[Any, range] = {}
    for track, (first_frame, _last_frame) in spans.items():
        birth = min(
            (spans[s][0] for s in sibling_group(track) if s in spans),
            default=first_frame,
        )
        start = range_min if birth == earliest else max(birth, range_min)

        daughters = [spans[c][0] for c in child_tracks(track) if c in spans]
        end = min(min(daughters) - 1, range_max) if daughters else range_max

        if start > end:
            continue
        windows[track] = range(start, end + 1)
    return windows


def representative_xy(rows: pd.DataFrame) -> tuple[float, float] | None:
    """Median of the non-zero coordinates in ``rows``, or ``None`` if there are none."""
    xs = pd.to_numeric(
        rows.get("XMorphology", pd.Series(dtype=float)), errors="coerce"
    )
    ys = pd.to_numeric(
        rows.get("YMorphology", pd.Series(dtype=float)), errors="coerce"
    )
    usable = xs.notna() & ys.notna() & (xs != 0) & (ys != 0)
    if not usable.any():
        return None
    return float(xs[usable].median()), float(ys[usable].median())


def extract_track_id(identification) -> int | Any:
    """Track id encoded in the trailing ``-<n>`` of an Identification."""
    if pd.isna(identification):
        return pd.NA
    m = _IDENT_TAIL_RE.search(str(identification))
    return int(m.group(1)) if m else pd.NA


def _integer_frames(rows: pd.DataFrame) -> pd.Series:
    """Return the ``t`` column as a clean integer Series, unparsable values dropped."""
    return pd.to_numeric(rows["t"], errors="coerce").dropna().astype(int)


def _track_spans(
    ident_rows: pd.DataFrame, has_track_number: bool
) -> dict[Any, tuple[int, int]]:
    """Measured ``(first, last)`` frame of every track, absent sisters included."""
    if not has_track_number:
        frames = _integer_frames(ident_rows)
        return (
            {}
            if frames.empty
            else {None: (int(frames.min()), int(frames.max()))}
        )

    spans: dict[Any, tuple[int, int]] = {}
    for track_number, group in ident_rows.groupby("TrackNumber", dropna=False):
        track = _normalise_track(track_number)
        if track is None:
            continue
        frames = _integer_frames(group)
        if frames.empty:
            continue
        spans[track] = (int(frames.min()), int(frames.max()))

    for track in list(spans):
        for sister in sibling_group(track):
            if sister is not None and sister not in spans:
                spans[sister] = spans[track]
    return spans


def _member_rows(
    ident_rows: pd.DataFrame, track, has_track_number: bool
) -> pd.DataFrame:
    """Rows of one track within an Identification."""
    if not has_track_number or track is None:
        return ident_rows
    mask = pd.to_numeric(ident_rows["TrackNumber"], errors="coerce").astype(
        "Int64"
    ) == int(track)
    return ident_rows.loc[mask.fillna(False)]


def sort_columns(has_track_number: bool) -> list[str]:
    """Canonical sort order of the track DataFrame."""
    cols = ["Position", "Identification", "t"]
    if has_track_number:
        cols.append("TrackNumber")
    return cols


class _RowCollector:
    """Accumulates synthetic rows, deduplicated on the natural key."""

    def __init__(self, columns: Iterable[str]) -> None:
        self._columns = list(columns)
        self._rows: dict[tuple, dict[str, Any]] = {}
        self._staged: dict[tuple, set[int]] = defaultdict(set)

    def __len__(self) -> int:
        return len(self._rows)

    @staticmethod
    def _key(position: int, identification, track_number, t: int) -> tuple:
        """Natural key of a row: ``(position, ident, track, frame)``."""
        return (
            int(position),
            identification,
            _normalise_track(track_number),
            int(t),
        )

    def staged_frames(
        self, position: int, identification, track_number
    ) -> set[int]:
        """Frames already synthesised for this exact track."""
        return self._staged[
            (int(position), identification, _normalise_track(track_number))
        ]

    def blank_row(self) -> dict[str, Any]:
        """Return a row template with every column set to 0."""
        return dict.fromkeys(self._columns, 0)

    def add(
        self,
        row: dict[str, Any],
        *,
        position: int,
        identification,
        track_number,
        t: int,
    ) -> bool:
        """Stage ``row``; returns ``False`` if an identical key already exists."""
        key = self._key(position, identification, track_number, t)
        if key in self._rows:
            return False
        self._rows[key] = row
        self._staged[key[:3]].add(key[3])
        return True

    def to_frame(self) -> pd.DataFrame:
        """All staged rows as a DataFrame with a clean index."""
        if not self._rows:
            return pd.DataFrame(columns=self._columns)
        return pd.DataFrame(list(self._rows.values())).reset_index(drop=True)


def _make_row(
    collector: _RowCollector,
    ctx: GapFillContext,
    *,
    identification,
    track_number,
    t: int,
    xy: tuple[float, float] | None,
    track_id,
    has_track_number: bool,
) -> dict[str, Any]:
    """Build one synthetic row for ``track_number`` at frame ``t``."""
    row = collector.blank_row()
    row["Position"] = ctx.position
    row["Identification"] = identification
    row["t"] = int(t)
    row["active"] = 1
    row["track_id"] = track_id
    row["Cellfate"] = SYNTHETIC_CELLFATE
    row["Calculated_Time"] = ctx.calculated_time(t)
    if has_track_number:
        row["TrackNumber"] = track_number
    if xy is not None:
        row["XMorphology"], row["YMorphology"] = xy
    return row


def mask_absent_measurements(
    add_df: pd.DataFrame, ctx: GapFillContext
) -> pd.DataFrame:
    """Set measurements to NaN on frames that were never acquired, so 0 still means 0."""
    if add_df.empty or not ctx.mask_absent_frames or not ctx.presence:
        return add_df

    out = add_df.reset_index(drop=True)
    frames = pd.to_numeric(out["t"], errors="coerce")

    for col in out.columns:
        m = _INTENSITY_COL_RE.search(str(col))
        if not m:
            continue
        suffix = m.group(1)
        if ctx.presence_flags(suffix) is None:
            continue
        absent = frames.map(
            lambda t, _s=suffix: (
                False if pd.isna(t) else ctx.frame_is_absent(_s, int(t))
            )
        ).to_numpy(dtype=bool)
        if absent.any():
            out.loc[absent, col] = np.nan
    return out


def _align_to_schema(add_df: pd.DataFrame, columns: pd.Index) -> pd.DataFrame:
    """Give ``add_df`` exactly ``columns``, filling anything it is missing."""
    out = add_df.copy()
    for col in columns:
        if col in out.columns:
            continue
        if col == "active":
            out[col] = 1
        elif col == "Cellfate":
            out[col] = SYNTHETIC_CELLFATE
        elif col in _SCHEMA_NULLABLE:
            out[col] = _null_series(len(out), _SCHEMA_NULLABLE[col], out.index)
        else:
            out[col] = 0
    return out[columns]


def fill_missing_frames(df: pd.DataFrame, ctx: GapFillContext) -> pd.DataFrame:
    """Return ``df`` with the gaps filled for ``ctx.position``."""
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()

    df = ensure_track_schema(df)

    position_mask = df["Position"] == ctx.position
    pos_df = df[position_mask].copy()
    if pos_df.empty:
        return df

    has_track_number = "TrackNumber" in pos_df.columns
    place_rows = ctx.coordinates is CoordinatePolicy.MEDIAN
    collector = _RowCollector(df.columns)

    for ident, ident_rows in pos_df.groupby("Identification", dropna=False):
        if pd.isna(ident):
            continue

        spans = _track_spans(ident_rows, has_track_number)
        windows = track_windows(spans, ctx.frames)
        if not windows:
            continue

        track_id = extract_track_id(ident)
        ident_xy = representative_xy(ident_rows) if place_rows else None

        for track, window in windows.items():
            track_rows = _member_rows(ident_rows, track, has_track_number)

            xy = None
            if place_rows:
                sisters = pd.concat(
                    [
                        _member_rows(ident_rows, s, has_track_number)
                        for s in sibling_group(track)
                        if s in spans
                    ]
                    or [ident_rows.iloc[0:0]]
                )
                xy = (
                    representative_xy(track_rows)
                    or representative_xy(sisters)
                    or ident_xy
                )

            _fill_track(
                collector,
                ctx,
                identification=ident,
                track_number=track,
                allowed=set(window),
                existing=set(_integer_frames(track_rows)),
                xy=xy,
                track_id=track_id,
                has_track_number=has_track_number,
            )

    if len(collector):
        add_df = collector.to_frame()
        add_df = _align_to_schema(add_df, df.columns)
        add_df = add_df.drop_duplicates(
            subset=[c for c in KEY_COLUMNS if c in add_df.columns],
            keep="first",
        )
        add_df = mask_absent_measurements(add_df, ctx)
        add_df = add_df.replace("", 0)
        pos_df = pd.concat([pos_df, add_df], ignore_index=True)

    updated = pd.concat(
        [df[~position_mask], pos_df], ignore_index=True
    ).sort_values(by=sort_columns(has_track_number), kind="mergesort")
    return updated.reset_index(drop=True)


def _fill_track(
    collector: _RowCollector,
    ctx: GapFillContext,
    *,
    identification,
    track_number,
    allowed: set[int],
    existing: set[int],
    xy: tuple[float, float] | None,
    track_id,
    has_track_number: bool,
) -> None:
    """Stage a synthetic row for every allowed frame this track is missing."""
    staged = collector.staged_frames(
        ctx.position, identification, track_number
    )
    for t in sorted(allowed - existing - staged):
        row = _make_row(
            collector,
            ctx,
            identification=identification,
            track_number=track_number,
            t=t,
            xy=xy,
            track_id=track_id,
            has_track_number=has_track_number,
        )
        collector.add(
            row,
            position=ctx.position,
            identification=identification,
            track_number=track_number,
            t=t,
        )


def add_missing_rows(main_window) -> None:
    """Fill gaps for the current Position, writing back ``track_df`` and ``filtered_df``."""
    df = getattr(main_window, "track_df", None)

    if df is None or df.empty:
        LOG.warning("track_df is empty; nothing to do.")
        main_window.filtered_df = pd.DataFrame(
            columns=df.columns if df is not None else []
        )
        return

    try:
        ctx = GapFillContext.from_main_window(main_window)
    except ValueError as exc:
        LOG.warning("%s; aborting.", exc)
        return

    updated = fill_missing_frames(df, ctx)

    main_window.track_df = updated
    has_track_number = "TrackNumber" in updated.columns
    main_window.filtered_df = (
        updated[updated["Position"] == ctx.position]
        .copy()
        .sort_values(by=sort_columns(has_track_number), kind="mergesort")
        .reset_index(drop=True)
    )

    if main_window.filtered_df.empty:
        LOG.warning("no rows for Position %s.", ctx.position)
