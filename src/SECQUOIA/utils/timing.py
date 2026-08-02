"""Utilities for working with time point indices in microscopy image filenames.

This module provides helpers to extract 1-based time point numbers from
filenames containing patterns like `_t00001_`, and to convert selected
time ranges from filename-based indexing to 0-based napari/track dataframe
indices. It also hosts the shared "imported real time" helpers.
"""

from __future__ import annotations

import os
import re

import pandas as pd

from SECQUOIA.config import TRACKING

_T_RE = re.compile(r"_t(\d{5})_")

#: Parses position / timepoint / wavelength out of an ``Image File`` entry.
RT_PREFIX = TRACKING.REALTIME_PREFIX
_RT_NAME_RE = re.compile(
    r"[\\/_]p(?P<p>\d+).*?[\\/_]t(?P<t>\d+).*?(?:[\\/_]z\d+.*?)?[\\/_]w(?P<w>\d+)",
    re.IGNORECASE,
)


def filename_t_file_number(path: str) -> int | None:
    """Extract 1-based t from a filename like _t00001_.

    Returns int or None if not found.
    """
    m = _T_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None


def current_t_range(main_window) -> tuple[int, int, int, int]:
    """Returns a 4-tuple of ints:
    t_file_min, t_file_max : 1-based (as in filenames `_t00001_`)
    t_idx_min,  t_idx_max  : 0-based (napari / track_df indices)"""
    t_file_min = int(getattr(main_window, "time_min_selected", 1))
    t_file_max = int(getattr(main_window, "time_max_selected", t_file_min))
    if t_file_max < t_file_min:
        t_file_max = t_file_min
    t_idx_min = t_file_min - 1
    t_idx_max = t_file_max - 1
    return t_file_min, t_file_max, t_idx_min, t_idx_max


def calculate_time(main_window) -> None:
    """Calculate the time based on the saved Δt (seconds) and the 't' column
    n track_df. 'Calculated_Time' is in minutes.
    """
    dt_sec = float(main_window.dt_seconds)
    main_window.time_interval = dt_sec

    df = main_window.track_df
    if "t" in df.columns:
        df = df.copy()
        df["Calculated_Time"] = (df["t"].astype(float) * dt_sec) / 60.0
        main_window.track_df = df


def realtime_columns(cols) -> list[str]:
    """Return the imported real-time columns present in ``cols``."""
    return [c for c in cols if isinstance(c, str) and c.startswith(RT_PREFIX)]


def build_realtime_lookup(
    main_window, force: bool = False
) -> pd.DataFrame | None:
    """Build (and cache) the wide ``(Position, t) -> per channel time`` table."""
    if not force:
        cached = getattr(main_window, "_rt_wide", None)
        if cached is not None:
            return cached

    rt_df = getattr(main_window, "import_rt_df", None)
    if not isinstance(rt_df, pd.DataFrame) or rt_df.empty:
        main_window._rt_wide = None
        return None
    if not {"Image File", "Measurement Time (ms)"}.issubset(rt_df.columns):
        main_window._rt_wide = None
        return None

    try:
        n_channels = int(getattr(main_window, "n_channels", 0) or 0)
    except (TypeError, ValueError):
        n_channels = 0

    rows = []
    for s, ms in zip(
        rt_df["Image File"].astype(str),
        rt_df["Measurement Time (ms)"],
        strict=False,
    ):
        m = _RT_NAME_RE.search(s)
        if not m:
            continue
        pos = int(m.group("p").lstrip("0") or "0")
        t_raw = int(m.group("t"))
        ch_orig = int(m.group("w"))
        t_adj = max(0, t_raw - 1)

        try:
            minutes = float(ms) / 60_000.0  # ms -> minutes
        except (TypeError, ValueError):
            continue

        rows.append((pos, t_adj, ch_orig, minutes))

    if not rows:
        main_window._rt_wide = None
        return None

    tidy = pd.DataFrame(
        rows, columns=["Position", "t", "_ch_orig", "_minutes"]
    ).drop_duplicates(subset=["Position", "t", "_ch_orig"])

    unique_ch = sorted(tidy["_ch_orig"].unique())
    ch_map = {orig: i + 1 for i, orig in enumerate(unique_ch)}
    tidy["_ch"] = tidy["_ch_orig"].map(ch_map)

    if n_channels > 0:
        tidy = tidy[tidy["_ch"] <= n_channels]
    if tidy.empty:
        main_window._rt_wide = None
        return None

    wide = tidy.pivot_table(
        index=["Position", "t"],
        columns="_ch",
        values="_minutes",
        aggfunc="first",
    ).reset_index()

    main_window._rt_wide = wide
    return wide


def apply_realtime(
    df: pd.DataFrame | None, wide: pd.DataFrame | None
) -> pd.DataFrame | None:
    """Left-join the real-time columns onto ``df`` by ``(Position, t)``."""
    if df is None or wide is None or getattr(df, "empty", True):
        return df
    if not {"Position", "t"}.issubset(df.columns):
        return df

    left = df.copy()
    left["_rt_pos"] = pd.to_numeric(left["Position"], errors="coerce")
    left["_rt_t"] = pd.to_numeric(left["t"], errors="coerce")

    right = wide.copy()
    right["_rt_pos"] = pd.to_numeric(right["Position"], errors="coerce")
    right["_rt_t"] = pd.to_numeric(right["t"], errors="coerce")
    right = right.drop(columns=["Position", "t"])

    merged = left.merge(
        right, how="left", on=["_rt_pos", "_rt_t"], suffixes=("", "_rt")
    )
    merged.drop(columns=["_rt_pos", "_rt_t"], inplace=True)

    for ch in [c for c in merged.columns if isinstance(c, int)]:
        merged[f"{RT_PREFIX}{ch}"] = merged[ch]
        merged.drop(columns=[ch], inplace=True)
    return merged
