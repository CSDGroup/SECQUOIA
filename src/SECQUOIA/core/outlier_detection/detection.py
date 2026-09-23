"""Outlier detection rules, feature column resolution, and the threshold/sliding-window detection algorithms."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, fields
from datetime import datetime

import numpy as np
import pandas as pd

from SECQUOIA.config import CloseMaskSettings, Rule, SlidingWindow
from SECQUOIA.utils.paths import project_analysis_dir

LOG = logging.getLogger(__name__)

__all__ = [
    "RulesPack",
    "_outlier_rules_dir",
    "compare_series_op",
    "load_outlier_rules_from_disk",
    "resolve_feature_columns",
    "run_outlier_pipeline",
    "run_sliding_windows",
    "run_threshold_rules",
    "save_outlier_rules_to_disk",
    "sliding_window_outlier_mask",
]


@dataclass
class RulesPack:
    """Container for threshold, sliding-window and close-mask detection settings."""

    version: int
    m_n: int
    ch_n: int
    rules: list[Rule]
    sliding_windows: list[SlidingWindow]
    close_masks: CloseMaskSettings | None = None

    def to_dict(self) -> dict:
        """Return the rules pack as a JSON-serializable dictionary."""
        return {
            "version": self.version,
            "m_n": self.m_n,
            "ch_n": self.ch_n,
            "rules": [asdict(r) for r in self.rules],
            "sliding_windows": [asdict(s) for s in self.sliding_windows],
            "sliding_window": (
                asdict(self.sliding_windows[0])
                if self.sliding_windows
                else None
            ),
            "close_masks": (
                asdict(self.close_masks) if self.close_masks else None
            ),
        }

    @staticmethod
    def from_dict(d: dict) -> RulesPack:
        """Create a RulesPack from a saved rules dictionary."""
        rule_fields = {f.name for f in fields(Rule)}
        rules = [
            Rule(**{k: v for k, v in r.items() if k in rule_fields})
            for r in d.get("rules", [])
            if r.get("enabled", True)
        ]
        return RulesPack(
            version=int(d.get("version", 1)),
            m_n=int(d.get("m_n", 0)),
            ch_n=int(d.get("ch_n", 0)),
            rules=[r for r in rules if rule_is_active(r)],
            sliding_windows=[
                SlidingWindow(**s)
                for s in d.get("sliding_windows", [])
                or (
                    []
                    if not d.get("sliding_window")
                    else [d["sliding_window"]]
                )
            ],
            close_masks=_close_mask_settings_from_dict(d.get("close_masks")),
        )


def _close_mask_settings_from_dict(raw) -> CloseMaskSettings | None:
    """Read the saved close-mask settings; None if absent or unusable.

    Rules files written before close-mask detection existed have no such
    entry, and a damaged one must not stop the rest of the rules loading.
    """
    if not isinstance(raw, dict):
        return None
    try:
        distance = float(raw["distance"])
        masks = raw.get("masks")
        masks = [int(m) for m in masks] if masks else None
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(distance) or distance < 0:
        return None
    return CloseMaskSettings(distance=distance, masks=masks)


def save_outlier_rules_to_disk(main_window, rules_payload: dict) -> str | None:
    """Save rules as JSON under a timestamped name, returning the path or None on failure."""
    folder = _outlier_rules_dir(main_window)
    if not folder:
        return None
    path = os.path.join(folder, _default_rules_filename())
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rules_payload, f, ensure_ascii=False, indent=2)
        LOG.info("[Rules] Saved outlier rules to: %s", path)
        return path
    except (OSError, RuntimeError, AttributeError, TypeError) as e:
        LOG.error("[Rules] Failed to save rules: %s", e)
        return None


def _default_rules_filename() -> str:
    """Return a timestamped default filename for saved outlier rules."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"rules_{stamp}.json"


def _outlier_rules_dir(main_window) -> str | None:
    """Return <experiment_root>/Analysis/SECQUOIA_files_<tracking_format>/Outlier_detection and create it if it doesn't exist."""
    experiment_root = getattr(main_window, "folder", None)
    if not experiment_root:
        LOG.warning(
            "[Rules] main_window.folder is not set; cannot resolve save path."
        )
        return None
    base = os.path.join(
        project_analysis_dir(main_window, experiment_root), "Outlier_detection"
    )
    os.makedirs(base, exist_ok=True)
    return base


def load_outlier_rules_from_disk(file_path: str) -> dict | None:
    """Load and return the JSON rules."""
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        return data
    except (OSError, json.JSONDecodeError) as e:
        LOG.error("[Rules] Failed to load rules file '%s': %s", file_path, e)
        return None


_MASK_TOKEN_RE = re.compile(r"M(\d+)")
_CHANNEL_TOKEN_RE = re.compile(r"Ch(\d+)")


def _column_mask_channel(col_name: str) -> tuple[int | None, int | None]:
    """Extract the (mask, channel) integer tokens from a column name, if present."""
    m = _MASK_TOKEN_RE.search(col_name)
    ch = _CHANNEL_TOKEN_RE.search(col_name)
    return (
        int(m.group(1)) if m else None,
        int(ch.group(1)) if ch else None,
    )


def _to_int_set(values) -> set[int]:
    """Return the values that parse as int, silently dropping the rest."""
    out = set()
    for v in values:
        try:
            out.add(int(v))
        except (TypeError, ValueError):
            continue
    return out


def _matches_mask_channel_tokens(
    col_name: str, masks: list[int] | None, channels: list[str] | None
) -> bool:
    """Return True if `col_name`'s mask/channel numbers match every requested filter.

    Compares the actual numeric tokens (e.g. M1 vs M10), not a plain
    substring check, so a request for mask 1 cannot also match mask 10.
    """
    col_mask, col_channel = _column_mask_channel(col_name)
    mask_ok = not masks or (
        col_mask is not None and col_mask in _to_int_set(masks)
    )
    channel_ok = not channels or (
        col_channel is not None and col_channel in _to_int_set(channels)
    )
    return mask_ok and channel_ok


def _generic_suffix_matches(
    df_cols, feat: str, masks: list[int] | None, channels: list[str] | None
) -> set:
    """Columns starting with `feat` whose mask/channel tokens match."""
    return {
        c
        for c in df_cols
        if str(c).startswith(feat)
        and _matches_mask_channel_tokens(str(c), masks, channels)
    }


def _exact_mask_only_names(df_cols, feat: str, masks: list[int]) -> set:
    """Exact 'feat+M+m' names, the columns the channel filter would otherwise drop."""
    found = set()
    for m in masks:
        name = f"{feat}M{m}"
        if name in df_cols:
            found.add(name)
    return found


def resolve_feature_columns(
    df, feat: str, masks: list[int] | None, channels: list[str] | None
) -> list[str]:
    """Return numeric dataframe columns matching a feature, mask selection, and channel selection."""
    feat = str(feat)
    df_cols = df.columns
    has_suffix = bool(re.search(r"(M\d+|Ch\d+)", feat))

    if has_suffix:
        feat_mask, feat_channel = _column_mask_channel(feat)
        if not masks and feat_mask is not None:
            masks = [feat_mask]
        if not channels and feat_channel is not None:
            channels = [feat_channel]
        cols = _generic_suffix_matches(df_cols, feat, masks, channels)
    else:
        cols = set()
        if masks:
            cols |= _exact_mask_only_names(df_cols, feat, masks)
        cols |= _generic_suffix_matches(df_cols, feat, masks, channels)
        if feat in df_cols:
            cols.add(feat)

    return sorted(
        [c for c in cols if str(df[c].dtype.kind) in ("i", "u", "f")]
    )


def compare_series_op(series, op: str, val: float) -> pd.Series:
    """Compare a numeric series against a scalar threshold using the selected operator."""
    s = pd.to_numeric(series, errors="coerce")
    if op == "<":
        return s < val
    if op == "<=":
        return s <= val
    if op == "=":
        return s == val
    if op == ">=":
        return s >= val
    if op == ">":
        return s > val
    return pd.Series(False, index=series.index)


def threshold_values_set(val1, op2, val2) -> bool:
    """True once a threshold row carries a non-zero limit."""
    if float(val1 or 0.0) != 0.0:
        return True
    return op2 is not None and float(val2 or 0.0) != 0.0


def rule_is_active(rule: Rule) -> bool:
    """True if a threshold rule has values and should be applied."""
    return threshold_values_set(rule.val1, rule.op2, rule.val2)


def run_threshold_rules(
    df, pack: RulesPack, outcol="Outlier_detection", on_rule_done=None
) -> pd.DataFrame:
    """Apply all static threshold rules in a RulesPack, writing 'Outlier' into `outcol` in place."""
    if outcol not in df.columns:
        df[outcol] = "OK"
    for r in pack.rules:
        cols = resolve_feature_columns(
            df, r.feat, (r.masks or None), (r.channels or None)
        )
        if cols:
            m1 = pd.Series(False, index=df.index)
            for c in cols:
                m1 |= compare_series_op(df[c], r.op1, r.val1)

            if r.op2 is None:
                mask = m1
            else:
                m2 = pd.Series(False, index=df.index)
                for c in cols:
                    m2 |= compare_series_op(
                        df[c], r.op2, (r.val2 if r.val2 is not None else 0.0)
                    )
                mask = (m1 & m2) if r.combine.upper() == "AND" else (m1 | m2)

            df.loc[mask & (df[outcol] != "Reviewed"), outcol] = "Outlier"
        if on_rule_done is not None:
            on_rule_done()
    return df


def _window_sd(
    t_curr: float,
    t_min: float,
    t_max: float,
    t_vals: np.ndarray,
    vals: np.ndarray,
    t_min_all: float,
    t_max_all: float,
    step: float,
) -> float | None:
    """Return the sample SD of finite values in a time window around `t_curr`, widening the window symmetrically until it holds at least 2 finite values."""
    lo, hi = t_curr - t_min, t_curr + t_max

    def finite_vals(a, b):
        in_range = (t_vals >= a) & (t_vals <= b)
        w = vals[in_range]
        return w[np.isfinite(w)]

    win = finite_vals(lo, hi)
    guard = 0
    while win.size < 2 and (lo > t_min_all or hi < t_max_all):
        if lo > t_min_all:
            lo -= step
        if hi < t_max_all:
            hi += step
        win = finite_vals(lo, hi)
        guard += 1
        if guard > 10000:
            break
    if win.size < 2:
        return None
    return float(np.std(win, ddof=1))


def _flag_next_time_outliers(
    vals: np.ndarray,
    idx: pd.Index,
    curr_rows: list,
    next_rows: list,
    sd: float,
    sd_factor: float,
) -> dict:
    """Return {row_index: True} for rows in `next_rows` whose value falls outside the SD band built around every row in `curr_rows`."""
    next_is_out = dict.fromkeys(next_rows, True)

    for i_curr in curr_rows:
        pos = np.where(idx == i_curr)[0]
        if pos.size != 1:
            continue
        v_curr = vals[pos[0]]
        if not np.isfinite(v_curr):
            continue
        lo = v_curr - sd_factor * sd
        hi = v_curr + sd_factor * sd
        for i_next in next_rows:
            if not next_is_out[i_next]:
                continue
            pos2 = np.where(idx == i_next)[0]
            if pos2.size != 1:
                continue
            v_next = vals[pos2[0]]
            if np.isfinite(v_next) and (lo <= v_next <= hi):
                next_is_out[i_next] = False

    return next_is_out


def _mark_group_outliers(
    t_vals: np.ndarray,
    vals: np.ndarray,
    idx: pd.Index,
    t_min: float,
    t_max: float,
    sd_factor: float,
) -> pd.Series:
    """Mark sliding-window outliers within a single track (one Identification and TrackNumber)."""
    mark_outlier = pd.Series(False, index=idx)

    times = np.unique(t_vals[np.isfinite(t_vals)])
    if times.size <= 1:
        return mark_outlier

    diffs = np.diff(times)
    step = float(np.median(diffs)) if diffs.size else 1.0
    if not np.isfinite(step) or step <= 0:
        step = 1.0

    t_min_all = float(times.min())
    t_max_all = float(times.max())
    # Map each time point to the dataframe indices observed at that time.
    rows_at_t = {t: list(idx[np.where(t_vals == t)[0]]) for t in times}

    # Compare point at t_next to band around value at t_curr
    for i in range(len(times) - 1):
        t_curr, t_next = times[i], times[i + 1]
        curr_rows = rows_at_t.get(t_curr, [])
        next_rows = rows_at_t.get(t_next, [])
        if not curr_rows or not next_rows:
            continue

        sd = _window_sd(
            t_curr, t_min, t_max, t_vals, vals, t_min_all, t_max_all, step
        )
        if sd is None or not np.isfinite(sd):
            continue

        next_is_out = _flag_next_time_outliers(
            vals, idx, curr_rows, next_rows, sd, sd_factor
        )
        for i_next, flag in next_is_out.items():
            if flag:
                mark_outlier.loc[i_next] = True

    return mark_outlier


def _mark_feature_column_outliers(
    df: pd.DataFrame,
    feat_col: str,
    id_col: str,
    t_col: str,
    t_min: float,
    t_max: float,
    sd_factor: float,
    track_col: str | None = None,
) -> pd.Series:
    """Mark sliding-window outliers in `feat_col`, one continuous branch at a time."""
    vals_all = pd.to_numeric(df[feat_col], errors="coerce")
    mark_outlier = pd.Series(False, index=df.index)

    group_cols = (
        [id_col, track_col]
        if track_col and track_col in df.columns
        else [id_col]
    )

    for _key, sub in df.groupby(group_cols, sort=False):
        idx = sub.index
        t_vals = (
            pd.to_numeric(sub[t_col], errors="coerce").astype(float).to_numpy()
        )
        vals = vals_all.loc[idx].to_numpy()

        mark_outlier.loc[idx] = _mark_group_outliers(
            t_vals, vals, idx, t_min, t_max, sd_factor
        )

    return mark_outlier


def _apply_sliding_window_config(
    df: pd.DataFrame,
    cfg: SlidingWindow,
    id_col: str,
    t_col: str,
    track_col: str | None = None,
) -> pd.Series:
    """Evaluate one sliding-window config across all its resolved feature columns."""
    feat = cfg.feature
    m = cfg.mask
    ch = cfg.channel
    t_min = float(cfg.t_min)
    t_max = float(cfg.t_max)
    sd_factor = float(cfg.sd_factor)

    mark_outlier_any = pd.Series(False, index=df.index)
    if sd_factor <= 0:
        return mark_outlier_any

    feat_cols = resolve_feature_columns(
        df,
        feat,
        ([int(m)] if m is not None else None),
        ([ch] if ch is not None else None),
    )
    if not feat_cols:
        LOG.warning(
            "[Sliding] Skip: no columns for feature=%r, mask=%s, channel=%s.",
            feat,
            m,
            ch,
        )
        return mark_outlier_any

    for feat_col in feat_cols:
        mark_outlier = _mark_feature_column_outliers(
            df, feat_col, id_col, t_col, t_min, t_max, sd_factor, track_col
        )
        mark_outlier_any |= mark_outlier
        LOG.debug(
            "[Sliding] Done: feature=%r, mask=%s, ch=%s, t_min=%s, t_max=%s, "
            "factor=%s, new_outliers=%d",
            feat_col,
            m,
            ch,
            t_min,
            t_max,
            sd_factor,
            int(mark_outlier.sum()),
        )

    return mark_outlier_any


def run_sliding_windows(
    df, pack: RulesPack, outcol="Outlier_detection", on_window_done=None
) -> pd.DataFrame:
    """Run sliding-window outlier detection rules and return the sorted result."""
    if df is None or len(df) == 0:
        return df

    ID_COL, T_COL, TR_COL = "Identification", "t", "TrackNumber"
    if ID_COL not in df.columns or T_COL not in df.columns:
        return df

    if outcol not in df.columns:
        df[outcol] = "OK"

    cfgs = pack.sliding_windows or []
    if not cfgs:
        return df

    sort_cols = [ID_COL, T_COL] + ([TR_COL] if TR_COL in df.columns else [])
    df = df.sort_values(sort_cols)

    any_out = pd.Series(False, index=df.index)
    for cfg in cfgs:
        any_out |= _apply_sliding_window_config(df, cfg, ID_COL, T_COL, TR_COL)
        if on_window_done is not None:
            on_window_done()

    if any_out.any():
        df.loc[any_out & (df[outcol] != "Reviewed"), outcol] = "Outlier"

    return df


def sliding_window_outlier_mask(
    df: pd.DataFrame,
    pack: RulesPack,
    id_col: str = "Identification",
    t_col: str = "t",
    track_col: str | None = "TrackNumber",
) -> pd.Series:
    """Return a boolean mask of sliding-window outliers in `df`, without writing any column."""
    mark_any = pd.Series(False, index=df.index)
    if df is None or len(df) == 0:
        return mark_any
    if id_col not in df.columns or t_col not in df.columns:
        return mark_any
    for cfg in pack.sliding_windows or []:
        mark_any |= _apply_sliding_window_config(
            df, cfg, id_col, t_col, track_col
        )
    return mark_any


def run_outlier_pipeline(
    df, pack: RulesPack, outcol="Outlier_detection"
) -> pd.DataFrame:
    """Run the full outlier detection pipeline, resetting `outcol` to 'OK' first."""
    reviewed = df[outcol] == "Reviewed" if outcol in df.columns else None
    df[outcol] = "OK"
    if reviewed is not None:
        df.loc[reviewed, outcol] = "Reviewed"
    df = run_threshold_rules(df, pack, outcol=outcol)
    df = run_sliding_windows(df, pack, outcol=outcol)
    return df
