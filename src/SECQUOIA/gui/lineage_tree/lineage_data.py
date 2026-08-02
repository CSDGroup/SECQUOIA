"""Data assembly for the lineage tree: everything a view needs to *draw*,
computed with no Qt widgets involved.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyqtgraph as pg

from SECQUOIA.gui.lineage_tree.lineage_geometry import (
    assign_y_tidy,
    build_edges,
    build_track_table,
    default_xmap,
    valid_times_by_track,
)
from SECQUOIA.utils.plotting import TIME_MODE_T, x_column_for


def _infer_active_mask(main_window, _=None) -> int:
    """Return the active mask index used to determine valid track frames."""
    n = int(getattr(main_window, "n_masks", 1) or 1)
    try:
        active = int(
            getattr(main_window, "active_mask_index_by_viewer", {}).get(0, 1)
        )
    except (TypeError, ValueError):
        active = 1
    return max(1, min(n, active))


def _time_mapper_for_ident(
    df_ident: pd.DataFrame, main_window
) -> tuple[Callable[[float], float], float, float, str]:
    """Return a time->x mapping function plus x-range and column name for an ident."""
    mode = getattr(main_window, "_time_mode", TIME_MODE_T)
    if mode == TIME_MODE_T:

        def f(t):
            """Map a frame index to itself for raw-frame time mode."""
            return float(t)

        tt = pd.to_numeric(df_ident.get("t"), errors="coerce").to_numpy(
            dtype=float
        )
        tt = tt[np.isfinite(tt)]
        if tt.size == 0:
            return f, 0.0, 1.0, "t"
        return f, float(np.nanmin(tt)), float(np.nanmax(tt)), "t"

    ch_idx = 1
    try:
        ch_idx = int(
            getattr(main_window, "selected_ch_by_channel", {}).get(1, 1) or 1
        )
    except (AttributeError, TypeError, ValueError):
        ch_idx = 1

    cols = list(df_ident.columns)
    xcol = x_column_for(mode, has_ch=True, ch_idx=ch_idx, df_cols=cols)

    if xcol not in df_ident.columns:

        def f(t):
            """Fall back to identity mapping when the x-axis column is missing."""
            return float(t)

        tt = pd.to_numeric(df_ident.get("t"), errors="coerce").to_numpy(
            dtype=float
        )
        tt = tt[np.isfinite(tt)]
        if tt.size == 0:
            return f, 0.0, 1.0, "t"
        return f, float(np.nanmin(tt)), float(np.nanmax(tt)), "t"

    tt = pd.to_numeric(df_ident.get("t"), errors="coerce").to_numpy(
        dtype=float
    )
    xx = pd.to_numeric(df_ident.get(xcol), errors="coerce").to_numpy(
        dtype=float
    )
    mask = np.isfinite(tt) & np.isfinite(xx)
    if not np.any(mask):

        def f(t):
            """Fall back to identity mapping when no finite (t, x) pairs exist."""
            return float(t)

        return f, 0.0, 1.0, "t"

    tt = tt[mask]
    xx = xx[mask]
    order = np.argsort(tt)
    tt = tt[order]
    xx = xx[order]
    uniq_t, idx = np.unique(tt, return_index=True)
    uniq_x = xx[idx]

    def f(t):
        """Interpolate the active x-axis value at frame `t` from the ident's known (t, x) points."""
        return float(
            np.interp(
                float(t), uniq_t, uniq_x, left=uniq_x[0], right=uniq_x[-1]
            )
        )

    return f, float(uniq_x.min()), float(uniq_x.max()), xcol


def _lineage_data(
    df: pd.DataFrame,
    ident: str,
    main_window=None,
    track_colors: dict[int, pg.QtGui.QColor] | None = None,
) -> dict:
    """Compute everything the lineage view needs to *draw*, with no Qt widgets."""
    sub_ident = df[df["Identification"] == ident].copy()
    mk_active = _infer_active_mask(main_window, sub_ident)

    try:
        xmap, _xmin_mapped, xmax_mapped, _xcol = _time_mapper_for_ident(
            sub_ident, main_window
        )
    except (RuntimeError, AttributeError, TypeError, ValueError, KeyError):
        xmap = default_xmap
        xmax_mapped = None

    use_fast_heatmap = (
        getattr(main_window, "_time_mode", TIME_MODE_T) == TIME_MODE_T
    )

    tracks = build_track_table(df, ident)
    valid_times = valid_times_by_track(sub_ident, mk_active)

    try:
        min_t_by_tn = (
            pd.to_numeric(sub_ident["t"], errors="coerce")
            .groupby(sub_ident["TrackNumber"])
            .min()
            .astype(float)
            .to_dict()
        )
    except (RuntimeError, AttributeError, TypeError, ValueError):
        min_t_by_tn = {}

    keep_rows = []
    for _, row in tracks.iterrows():
        tn = int(row["TrackNumber"])
        vt = valid_times.get(tn, set())
        keep_rows.append(bool(vt) or (tn in min_t_by_tn))
    tracks = tracks.loc[keep_rows].copy()

    for i, row in tracks.iterrows():
        tn = int(row["TrackNumber"])
        vt_sorted = sorted(valid_times.get(tn, []))
        if vt_sorted:
            tracks.at[i, "t_start"] = float(vt_sorted[0])
            tracks.at[i, "t_end"] = float(vt_sorted[-1])
        else:
            t0 = float(min_t_by_tn.get(tn, row["t_start"]))
            tracks.at[i, "t_start"] = t0
            tracks.at[i, "t_end"] = t0

    edges = build_edges(tracks)
    y_map = assign_y_tidy(tracks)

    if track_colors is None and main_window is not None:
        with contextlib.suppress(
            RuntimeError, AttributeError, TypeError, ValueError
        ):
            track_colors = getattr(main_window, "_track_colors", {}).get(
                str(ident)
            )

    if track_colors is None:
        try:
            cmap = plt.get_cmap("tab20")
            uniq = tracks["TrackNumber"].unique()
            track_colors = {
                int(t): pg.mkColor(
                    tuple(
                        int(c * 255) for c in cmap(i / max(len(uniq), 1))[:3]
                    )
                )
                for i, t in enumerate(uniq)
            }
        except (TypeError, ValueError):
            palette = [
                pg.mkColor(c)
                for c in (
                    "#4FC3F7",
                    "#81C784",
                    "#FFB74D",
                    "#E57373",
                    "#BA68C8",
                    "#64B5F6",
                    "#A1887F",
                    "#90A4AE",
                )
            ]
            uniq = tracks["TrackNumber"].unique()
            track_colors = {
                int(t): palette[i % len(palette)] for i, t in enumerate(uniq)
            }

    try:
        if main_window is not None:
            main_window._last_lineage_y_map = dict(y_map)
            main_window._lineage_y_map = dict(y_map)
    except (RuntimeError, AttributeError, TypeError, ValueError):
        pass

    try:
        s = pd.to_numeric(
            tracks.get("t_end", pd.Series([], dtype=float)), errors="coerce"
        )
        if s.size:
            m = np.nanmax(s.to_numpy())
            t_max_df = int(m) if np.isfinite(m) else 0
        else:
            t_max_df = 0
    except (RuntimeError, AttributeError, TypeError, ValueError):
        t_max_df = 0

    return {
        "df": df,
        "ident": str(ident),
        "sub_ident": sub_ident,
        "mk_active": mk_active,
        "xmap": xmap,
        "xmax_mapped": xmax_mapped,
        "use_fast_heatmap": use_fast_heatmap,
        "tracks": tracks,
        "edges": edges,
        "y_map": y_map,
        "valid_times": valid_times,
        "track_colors": track_colors,
        "t_max_df": t_max_df,
    }
