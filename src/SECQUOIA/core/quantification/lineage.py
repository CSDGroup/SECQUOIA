"""Per frame lineage displacement metrics."""

import logging

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


def _compute_lineage_step_distance(
    df,
    mask_idx,
    *,
    t_col="t",
    id_col="Identification",
    tr_col="TrackNumber",
    x_fallback="XMorphology",
    y_fallback="YMorphology",
) -> pd.Series:
    """Return a pandas Series with per frame displacement (pixels) for a given mask index, respecting lineage splits at TrackNumber changes."""
    xcol, ycol = f"XMorphologyM{mask_idx}", f"YMorphologyM{mask_idx}"
    if xcol not in df.columns or ycol not in df.columns:
        xcol, ycol = x_fallback, y_fallback

    out = pd.Series(0.0, index=df.index, dtype=float)
    required = {id_col, t_col, tr_col, xcol, ycol}
    if df.empty or not required.issubset(df.columns):
        return out

    dd = df[[id_col, t_col, tr_col, xcol, ycol]].copy()
    dd["__row__"] = np.arange(len(dd), dtype=np.int64)
    dd[t_col] = pd.to_numeric(dd[t_col], errors="coerce")
    dd[tr_col] = pd.to_numeric(dd[tr_col], errors="coerce")
    dd[xcol] = pd.to_numeric(dd[xcol], errors="coerce").astype(float)
    dd[ycol] = pd.to_numeric(dd[ycol], errors="coerce").astype(float)

    dd = dd[dd[tr_col].notna() & dd[t_col].notna()]
    if dd.empty:
        return out

    dd = dd.sort_values([id_col, t_col], kind="stable")
    dd["__k__"] = (
        dd.groupby(id_col, sort=False)[t_col]
        .rank(method="dense")
        .astype(np.int64)
    )

    pos = dd.drop_duplicates(subset=[id_col, "__k__", tr_col], keep="last")

    prev = pos[[id_col, "__k__", tr_col, xcol, ycol]].copy()
    prev["__k__"] = prev["__k__"] + 1
    prev = prev.rename(columns={xcol: "__px__", ycol: "__py__"})
    prev["__hasprev__"] = True

    cont = pos.merge(prev, on=[id_col, "__k__", tr_col], how="left")
    has_prev = cont["__hasprev__"].notna().to_numpy(dtype=bool)

    res = cont[[id_col, "__k__", tr_col]].copy()
    res["__d__"] = np.where(
        has_prev,
        np.hypot(
            cont[xcol].to_numpy() - cont["__px__"].to_numpy(),
            cont[ycol].to_numpy() - cont["__py__"].to_numpy(),
        ),
        np.nan,
    )

    new_rows = cont.loc[~has_prev, [id_col, "__k__", tr_col, xcol, ycol]]
    if not new_rows.empty:
        prev_any = prev[[id_col, "__k__", "__px__", "__py__"]]
        cand = new_rows.merge(prev_any, on=[id_col, "__k__"], how="inner")
        if not cand.empty:
            cand["__d__"] = np.hypot(
                cand[xcol].to_numpy() - cand["__px__"].to_numpy(),
                cand[ycol].to_numpy() - cand["__py__"].to_numpy(),
            )
            dmin = (
                cand.groupby([id_col, "__k__", tr_col], sort=False)["__d__"]
                .min()
                .rename("__dmin__")
                .reset_index()
            )
            res = res.merge(dmin, on=[id_col, "__k__", tr_col], how="left")
            res["__d__"] = res["__d__"].fillna(res["__dmin__"])
            res = res.drop(columns="__dmin__")

    res["__d__"] = res["__d__"].fillna(0.0)

    back = dd[[id_col, "__k__", tr_col, "__row__"]].merge(
        res, on=[id_col, "__k__", tr_col], how="left"
    )
    values = out.to_numpy(dtype=float, copy=True)
    values[back["__row__"].to_numpy()] = (
        back["__d__"].fillna(0.0).to_numpy(dtype=float)
    )
    return pd.Series(values, index=df.index, dtype=float)


def _set_columns_without_duplicating(
    df: pd.DataFrame, cols: dict
) -> pd.DataFrame:
    """Attach ``cols`` to ``df``, overwriting rather than appending duplicates."""
    if not cols:
        return df
    if df.columns.has_duplicates:
        dupes = df.columns[df.columns.duplicated()].unique().tolist()
        LOG.warning("Collapsing pre-existing duplicate columns: %s", dupes)
        df = df.loc[:, ~df.columns.duplicated()].copy()
    for name, series in cols.items():
        df[name] = series
    return df


def add_lineage_step_distance(
    main_window, mask_indices, *, id_col="Identification", tr_col="TrackNumber"
) -> None:
    """Add step_disp_px_m{m} (per frame displacement) to main_window.track_df and main_window.filtered_df."""
    tdf = getattr(main_window, "track_df", None)
    if tdf is not None and not tdf.empty:
        cols = {
            f"step_disp_px_m{m}": _compute_lineage_step_distance(
                tdf, m, id_col=id_col, tr_col=tr_col
            ).astype(float)
            for m in mask_indices
        }
        main_window.track_df = _set_columns_without_duplicating(tdf, cols)

    fdf = getattr(main_window, "filtered_df", None)
    if fdf is not None and hasattr(fdf, "empty") and not fdf.empty:
        cols = {
            f"step_disp_px_m{m}": _compute_lineage_step_distance(
                fdf, m, id_col=id_col, tr_col=tr_col
            ).astype(float)
            for m in mask_indices
        }
        main_window.filtered_df = _set_columns_without_duplicating(fdf, cols)


def compute_step_distance_for_row(
    df: pd.DataFrame,
    *,
    ident: str,
    track_no: int,
    t: int,
    mask_idx: int,
    id_col: str = "Identification",
    tr_col: str = "TrackNumber",
    t_col: str = "t",
    x_fallback: str = "XMorphology",
    y_fallback: str = "YMorphology",
) -> tuple[float, int | None]:
    """Compute the per frame step distance for the row (ident, track_no, t)."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return 0.0, None

    mask_row = (
        (df[id_col].astype(str) == str(ident))
        & (df[tr_col] == track_no)
        & (df[t_col] == t)
    )
    idxs = df.index[mask_row].to_list()
    if not idxs:
        return 0.0, None
    row_idx = idxs[0]

    ident_rows = df[df[id_col].astype(str) == str(ident)]
    distances = _compute_lineage_step_distance(
        ident_rows,
        mask_idx,
        t_col=t_col,
        id_col=id_col,
        tr_col=tr_col,
        x_fallback=x_fallback,
        y_fallback=y_fallback,
    )
    d = float(distances.at[row_idx])

    out_col = f"step_disp_px_m{mask_idx}"
    if out_col not in df.columns:
        df[out_col] = np.nan
    df.at[row_idx, out_col] = d
    return d, row_idx
