"""Time-mode and feature column discovery for the dynamics plots.

Works out which dataframe columns are plottable features and which column to
use for the x-axis, given the current time mode. Pure numpy/regex logic, no
Qt dependency.
"""

from __future__ import annotations

import re

import numpy as np

from SECQUOIA.config import TRACKING
from SECQUOIA.utils.timing import realtime_columns

# Values of ``main_window._time_mode``; the comment is the menu label.
TIME_MODE_T = "t"  # "Time point"
TIME_MODE_CALC = "calc"  # "Time"
TIME_MODE_REAL = "real"  # "RealTime"

# Pre-selected feature
DEFAULT_FEATURE = "MeanNoBgCorrected"


def _finite_max(series, default: float | None = None) -> float | None:
    """Max of the finite values in ``series``."""
    try:
        arr = np.asarray(series, dtype=float)
    except (TypeError, ValueError):
        return default
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(arr.max())


def _finite_min(series, default: float | None = None) -> float | None:
    """Min of the finite values in ``series``."""
    try:
        arr = np.asarray(series, dtype=float)
    except (TypeError, ValueError):
        return default
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(arr.min())


def x_column_for(mode: str, ch_idx: int | None, df_cols: list[str]) -> str:
    """The x-axis column for the current time mode, falling back to ``"t"``.

    Real-time mode prefers this channel's column, then channel 1's, then any
    imported real-time column that exists.
    """
    if mode == TIME_MODE_CALC and "Calculated_Time" in df_cols:
        return "Calculated_Time"
    if mode == TIME_MODE_REAL:
        ch = int(ch_idx or 1)
        col = f"{TRACKING.REALTIME_PREFIX}{ch}"
        if col in df_cols:
            return col
        first = f"{TRACKING.REALTIME_PREFIX}1"
        if first in df_cols:
            return first
        present = sorted(realtime_columns(df_cols))
        if present:
            return present[0]
    return "t"


def _format_feature_column(
    template: str | None, *, has_ch: bool, has_m: bool, channel, m_idx
) -> str | None:
    """Fill in a feature's ``{ch}``/``{m}`` template, or ``None`` if it doesn't fit.

    Used both when a row's column is resolved live from its combo boxes and
    when it is reconstructed later from stored selections, so the two stay
    in agreement about how a feature name is built.
    """
    if not template:
        return None
    try:
        if has_ch and has_m:
            return template.format(ch=channel, m=int(m_idx))
        if has_m:
            return template.format(m=int(m_idx))
        if has_ch:
            return template.format(ch=channel)
        return template
    except (RuntimeError, AttributeError, TypeError, ValueError, KeyError):
        return None


# Columns that look like features but must never be offered in the dropdown.
EXCLUDED_FEATURES = frozenset(
    {
        "XMorphology",
        "YMorphology",
        "nn_dist_px_",
        "label_id_",
        "alt_dist_px_",
        "alt_label_id_",
    }
)

# ``<feature>Ch<n>M<n>`` - a per channel, per mask measurement.
_FEATURE_CH_M_RE = re.compile(
    r"^(?P<feat>[A-Za-z0-9_]+)Ch(?P<ch>\d+)M(?P<m>\d+)$", re.IGNORECASE
)
# ``<feature>M<n>`` - a per mask measurement with no channel.
_FEATURE_M_RE = re.compile(
    r"^(?P<feat>[A-Za-z0-9_]+)M(?P<m>\d+)$", re.IGNORECASE
)


def _discover_features(
    columns: list[str], derived_registry: dict | None = None
) -> dict[str, dict]:
    """Work out which plottable features the dataframe columns describe."""
    feature_defs: dict[str, dict] = {}

    for col in columns:
        channel_match = _FEATURE_CH_M_RE.match(col)
        if channel_match:
            feat = channel_match.group("feat")
            if feat in EXCLUDED_FEATURES:
                continue
            template = re.sub(r"Ch\d+M\d+$", "Ch{ch}M{m}", col, count=1)
            entry = feature_defs.get(feat, {"has_m": True})
            entry["has_ch"] = True
            entry["template"] = template
            entry["display"] = feat
            feature_defs[feat] = entry
            continue

        mask_match = _FEATURE_M_RE.match(col)
        if mask_match:
            feat = mask_match.group("feat")
            if feat in EXCLUDED_FEATURES:
                continue
            if feat not in feature_defs:
                feature_defs[feat] = {
                    "template": re.sub(r"M\d+$", "M{m}", col, count=1),
                    "has_ch": False,
                    "has_m": True,
                    "display": feat,
                }

    for name, desc in (derived_registry or {}).items():
        direct_col = desc.get("template")
        if isinstance(direct_col, str) and direct_col in columns:
            feature_defs[name] = {
                "template": direct_col,
                "has_ch": False,
                "has_m": False,
                "display": desc.get("display", name),
            }

    return feature_defs


def _ensure_time_mode(main_window) -> None:
    """Initialize the main window time plotting mode if it is not already set."""
    if not hasattr(main_window, "_time_mode"):
        main_window._time_mode = TIME_MODE_T
