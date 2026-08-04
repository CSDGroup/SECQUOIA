"""Normalisation of metric columns."""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd

__all__ = ["apply_normalization"]

ID_COLUMN = "Identification"
TIME_COLUMN = "t"
METHOD_ZSCORE = "zscore"
METHOD_INVERSE = "inv"


def _window_mask(df: pd.DataFrame, tmin: int, tmax: int) -> pd.Series:
    """Build a boolean mask selecting rows inside a time point window."""
    if TIME_COLUMN not in df.columns:
        return pd.Series(True, index=df.index)

    tseries = pd.to_numeric(df[TIME_COLUMN], errors="coerce")
    return (tseries >= int(tmin)) & (tseries <= int(tmax))


def _zscore(df: pd.DataFrame, col: str, s: pd.Series, by_id: bool) -> None:
    """Replace a column with its z-score, in place."""
    if by_id:
        stats = (
            s.groupby(df[ID_COLUMN])
            .agg(["mean", "std"])
            .rename(columns={"mean": "_mu", "std": "_sigma"})
        )
        mu = df[ID_COLUMN].map(stats["_mu"])
        # A zero standard deviation would divide by zero; NaN then becomes 0.
        sigma = df[ID_COLUMN].map(stats["_sigma"]).replace(0, np.nan)
        df[col] = ((s - mu) / sigma).fillna(0.0)
        return

    mu = s.mean()
    sigma = s.std()
    if not np.isfinite(sigma) or sigma == 0:
        df[col] = 0.0
    else:
        df[col] = (s - mu) / sigma


def _inverse(
    df: pd.DataFrame,
    col: str,
    s: pd.Series,
    by_id: bool,
    mask: pd.Series,
) -> None:
    """Divide a column by its baseline, in place."""
    if by_id:
        per_id_baseline = s[mask].groupby(df.loc[mask, ID_COLUMN]).mean()
        baseline = df[ID_COLUMN].map(per_id_baseline)

        # IDs with no rows in the window fall back to the global baseline.
        baseline = baseline.fillna(s[mask].mean())

        df[col] = s / baseline.replace(0, np.nan)
        return

    baseline_all = s[mask].mean()
    if not np.isfinite(baseline_all) or baseline_all == 0:
        baseline_all = np.nan
    df[col] = s / baseline_all


def apply_normalization(
    df: pd.DataFrame,
    col: str,
    method: str,
    scope: str,
    tmin: int,
    tmax: int,
) -> None:
    """Normalise one column of a dataframe in place."""
    if not isinstance(df, pd.DataFrame) or col not in df.columns:
        return

    s = pd.to_numeric(df[col], errors="coerce")
    mask = _window_mask(df, tmin, tmax)
    by_id = (scope == "ids") and (ID_COLUMN in df.columns)

    if method == METHOD_ZSCORE:
        _zscore(df, col, s, by_id)
    elif method == METHOD_INVERSE:
        _inverse(df, col, s, by_id, mask)

    with contextlib.suppress(RuntimeError, AttributeError, TypeError):
        df[col] = pd.to_numeric(df[col], errors="coerce")
