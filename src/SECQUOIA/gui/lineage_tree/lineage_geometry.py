"""Pure geometry and data-shaping helpers for the lineage tree."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd

__all__ = [
    "parent",
    "left",
    "right",
    "generation",
    "default_xmap",
    "build_track_table",
    "build_edges",
    "assign_y_tidy",
    "valid_times_by_track",
    "segments_to_polyline",
    "LineageGeometry",
    "HeatSegment",
    "plan_heat_segments",
    "build_value_map",
    "FeatureCatalog",
    "EXCLUDED_TREE_FEATURES",
]


def parent(n: int) -> int:
    """Return the binary-tree parent track number for ``n``."""
    return n // 2


def left(n: int) -> int:
    """Return the left child track number for ``n``."""
    return n * 2


def right(n: int) -> int:
    """Return the right child track number for ``n``."""
    return n * 2 + 1


def generation(n: int) -> int:
    """Return the zero-based binary-tree generation for a track number."""
    return int(np.floor(np.log2(max(1, n))))


def default_xmap(t: float) -> float:
    """Default mapping for time -> x."""
    return float(t)


def _resolve_xmap(xmap: Callable[[float], float] | None):
    """Return ``xmap`` or the identity mapping when it is ``None``."""
    return default_xmap if xmap is None else xmap


def build_track_table(df: pd.DataFrame, ident: str) -> pd.DataFrame:
    """Build one row per track with start time, end time, and final fate."""
    sub = df[df["Identification"] == ident].copy()
    if sub.empty:
        raise ValueError(f"No rows for Identification={ident!r}")

    sub["t"] = pd.to_numeric(sub["t"], errors="coerce")
    grp = sub.groupby("TrackNumber", sort=True)

    def last_fate(s: pd.Series) -> str:
        """Return the last non-null cell fate, or Healthy if none is recorded."""
        s2 = s.dropna()
        return s2.iloc[-1] if not s2.empty else "Healthy"

    out = (
        grp["t"]
        .agg(t_start="min", t_end="max")
        .join(grp["Cellfate"].agg(last_fate))
        .reset_index()
    )

    out["t_start"] = pd.to_numeric(out["t_start"], errors="coerce").astype(
        float
    )
    out["t_end"] = pd.to_numeric(out["t_end"], errors="coerce").astype(float)
    out.rename(columns={"Cellfate": "fate"}, inplace=True)
    return out


def _nearest_present_ancestor(n: int, present: set[int]) -> int | None:
    """Closest ancestor of ``n`` still present, skipping missing levels."""
    a = parent(int(n))
    while a >= 1:
        if a in present:
            return a
        a = parent(a)
    return None


def build_edges(tracks: pd.DataFrame) -> pd.DataFrame:
    """Build parent-child division edges from available track numbers."""
    present = {int(n) for n in tracks["TrackNumber"]}
    t_start_map = {
        int(k): float(v)
        for k, v in zip(
            tracks["TrackNumber"],
            pd.to_numeric(tracks["t_start"], errors="coerce").astype(float),
            strict=False,
        )
    }
    rows: list[tuple[int, int, float]] = []
    for c in sorted(present):
        p = _nearest_present_ancestor(c, present)
        if p is not None:
            rows.append((p, c, t_start_map[c]))
    edges = pd.DataFrame(rows, columns=["parent", "child", "t_div"])

    edges["t_div"] = pd.to_numeric(edges["t_div"], errors="coerce").astype(
        float
    )
    edges = edges.dropna(subset=["t_div"])
    edges["t_div"] = edges["t_div"].round(6)
    return edges


def assign_y_tidy(tracks: pd.DataFrame) -> dict[int, float]:
    """Assign one y row per leaf track, parents centred above their children."""
    present = {int(n) for n in tracks["TrackNumber"]}
    children: dict[int, list[int]] = {n: [] for n in present}
    roots: list[int] = []
    for n in sorted(present):
        anchor = _nearest_present_ancestor(n, present)
        if anchor is None:
            roots.append(n)
        else:
            children[anchor].append(n)
    roots = roots or [1]

    y_map: dict[int, float] = {}
    next_row = 0

    def dfs(n: int) -> float:
        """Assign y-positions, centering parents above their children."""
        nonlocal next_row
        kids = sorted(children.get(n, []))
        if not kids:
            y = float(next_row)
            next_row += 1
            y_map[n] = y
            return y
        ys = [dfs(k) for k in kids]
        y = float(np.mean(ys))
        y_map[n] = y
        return y

    for r in sorted(roots):
        dfs(r)
    return y_map


def valid_times_by_track(
    df_ident: pd.DataFrame, mask_index: int
) -> dict[int, set[float]]:
    """Return valid time points per track based on nonzero label IDs."""
    lab_col = f"label_id_m{int(mask_index)}"
    vt: dict[int, set[float]] = {}

    if lab_col not in df_ident.columns:
        for tn, _ in df_ident.groupby("TrackNumber"):
            vt[int(tn)] = set()
        return vt

    for tn, g in df_ident.groupby("TrackNumber"):
        t = pd.to_numeric(g["t"], errors="coerce").to_numpy(dtype=float)
        lab = pd.to_numeric(g[lab_col], errors="coerce").to_numpy(dtype=float)
        keep = [
            float(tt)
            for tt, ll in zip(t, lab, strict=False)
            if np.isfinite(tt) and np.isfinite(ll) and ll != 0
        ]
        vt[int(tn)] = set(keep)
    return vt


def segments_to_polyline(
    valid_ts: Iterable[float], xmap: Callable[[float], float] | None = None
) -> np.ndarray:
    """Convert valid integer time points into a mapped polyline with NaN gaps."""
    xmap = _resolve_xmap(xmap)
    valid_ts = list(valid_ts)
    if not valid_ts:
        return np.array([], dtype=float)
    xs = np.asarray(sorted({int(np.floor(t)) for t in valid_ts}), dtype=float)
    if xs.size == 0:
        return np.array([], dtype=float)
    cuts = np.where(np.diff(xs) != 1)[0] + 1
    chunks = np.split(xs, cuts)
    pieces: list[np.ndarray] = []
    for i, ch in enumerate(chunks):
        if ch.size == 0:
            continue
        seg = np.column_stack(
            [[xmap(t) for t in ch], [xmap(t + 1) for t in ch]]
        ).ravel()
        if i > 0:
            seg = np.concatenate([np.array([np.nan]), seg])
        pieces.append(seg)
    return np.concatenate(pieces) if pieces else np.array([], dtype=float)


@dataclass(frozen=True)
class LineageGeometry:
    """Start/end/division times for every track in one lineage."""

    t_start: dict[int, float] = field(default_factory=dict)
    t_end: dict[int, float] = field(default_factory=dict)
    parent_div: dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_tracks(
        cls, tracks: pd.DataFrame, edges: pd.DataFrame | None
    ) -> LineageGeometry:
        """Build the span lookup tables from the track and edge tables."""
        t_start = {
            int(k): float(v)
            for k, v in zip(
                tracks["TrackNumber"], tracks["t_start"], strict=False
            )
        }
        t_end = {
            int(k): float(v)
            for k, v in zip(
                tracks["TrackNumber"], tracks["t_end"], strict=False
            )
        }
        parent_div: dict[int, float] = {}
        if edges is not None and len(edges):
            parent_div = {
                int(k): float(v)
                for k, v in edges.groupby("parent")["t_div"]
                .min()
                .to_dict()
                .items()
            }
        return cls(t_start=t_start, t_end=t_end, parent_div=parent_div)

    def track_numbers(self) -> list[int]:
        """Every track number, in ascending order."""
        return sorted(self.t_start)

    def division_span(self, tn: int) -> tuple[float, float] | None:
        """``(t0, t1)`` where ``t1`` is the division time, else the track end.

        The span every renderer draws a parent track over, and the one the
        highlight overlay follows.
        """
        tn = int(tn)
        t0 = self.t_start.get(tn)
        if t0 is None:
            return None
        t1 = self.parent_div.get(tn, self.t_end.get(tn))
        if t1 is None:
            return None
        return float(t0), float(t1)

    def clamped_span(self, tn: int) -> tuple[float, float] | None:
        """Like `division_span`, but capped so `t1` never exceeds the track's actual end."""
        span = self.division_span(tn)
        if span is None:
            return None
        t0, t1 = span
        end = self.t_end.get(int(tn))
        return (t0, t1) if end is None else (t0, min(t1, float(end)))

    def child_span(self, child: int, t_div: float) -> tuple[float, float]:
        """``(t_div, t_end)`` for the child half of a division edge."""
        return float(t_div), float(self.t_end[int(child)])


class HeatSegment(NamedTuple):
    """One coloured piece of a track's heat strip."""

    x0: float
    x1: float
    value: float


def plan_heat_segments(
    value_map: Mapping[float, float],
    t0_raw: float,
    t1_raw: float,
    xmap: Callable[[float], float] | None = None,
    pad: float = 0.0,
) -> list[HeatSegment]:
    """Plan the coloured segments covering ``[t0_raw, t1_raw]`` for one track."""
    xmap = _resolve_xmap(xmap)

    if (
        t0_raw is None
        or t1_raw is None
        or not np.isfinite(t0_raw)
        or not np.isfinite(t1_raw)
        or t1_raw <= t0_raw
    ):
        return []

    segments: list[HeatSegment] = []
    first_full = int(np.ceil(t0_raw))
    last_full = int(np.floor(t1_raw))

    def _emit(x0: float, x1: float, value: float) -> None:
        """Append a segment unless it would be empty or inverted."""
        if x1 > x0:
            segments.append(HeatSegment(float(x0), float(x1), float(value)))

    key0 = float(int(np.floor(t0_raw)))
    if key0 in value_map:
        _emit(
            xmap(float(t0_raw)),
            xmap(float(min(t1_raw, first_full))) + pad,
            value_map[key0],
        )

    for t in range(first_full, last_full):
        key = float(t)
        if key not in value_map:
            continue
        _emit(xmap(float(t)) - pad, xmap(float(t + 1)) + pad, value_map[key])

    key1 = float(last_full)
    if key1 in value_map:
        _emit(
            xmap(float(max(t0_raw, last_full))) - pad,
            xmap(float(t1_raw)),
            value_map[key1],
        )

    return segments


def build_value_map(
    df_ident: pd.DataFrame,
    column: str,
    valid_times: Mapping[int, set[float]] | None = None,
    *,
    track_col: str = "TrackNumber",
    time_col: str = "t",
) -> dict[int, dict[float, float]]:
    """Map ``{track: {time: value}}`` for one feature column."""
    out: dict[int, dict[float, float]] = {}
    if column not in df_ident.columns:
        return out

    for tn, g in df_ident.groupby(track_col):
        times = pd.to_numeric(g[time_col], errors="coerce").to_numpy(
            dtype=float
        )
        values = pd.to_numeric(g[column], errors="coerce").to_numpy(
            dtype=float
        )
        allowed = valid_times.get(int(tn)) if valid_times else None

        keep: dict[float, float] = {}
        for tt, vv in zip(times, values, strict=False):
            if not np.isfinite(tt) or not np.isfinite(vv):
                continue
            if allowed is not None and float(tt) not in allowed:
                continue
            keep[float(tt)] = float(vv)
        out[int(tn)] = keep
    return out


def value_range(
    values, default: tuple[float, float] = (0.0, 1.0)
) -> tuple[float, float]:
    """``(vmin, vmax)`` over ``values``, falling back to ``default``."""
    vals = pd.to_numeric(pd.Series(values), errors="coerce")
    vmin = float(vals.min(skipna=True)) if len(vals) else default[0]
    vmax = float(vals.max(skipna=True)) if len(vals) else default[1]
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return default
    return vmin, vmax


def normalize(value: float, vmin: float, vmax: float) -> float:
    """Scale ``value`` into ``[0, 1]``; non-finite input maps to 0."""
    if not np.isfinite(value):
        return 0.0
    a = (value - vmin) / (vmax - vmin)
    return 0.0 if a < 0 else (1.0 if a > 1 else a)


# Also used as the gate deciding whether heatmap mode is offered at all.
_FEATURE_CH_M_RE = re.compile(
    r"^(?P<feat>[A-Za-z0-9_]+)Ch(?P<ch>\d+)M(?P<m>\d+)$", re.IGNORECASE
)
_FEATURE_M_RE = re.compile(
    r"^(?P<feat>[A-Za-z0-9_]+)M(?P<m>\d+)$", re.IGNORECASE
)

EXCLUDED_TREE_FEATURES = frozenset(
    {
        "XMorphology",
        "YMorphology",
        "label_id",
        "label_id_",
        "nn_dist_px",
        "nn_dist_px_",
        "alt_dist_px_",
        "alt_label_id_",
    }
)


@dataclass
class FeatureCatalog:
    """Which ``(feature, channel, mask)`` combinations the columns provide."""

    features: set[str] = field(default_factory=set)
    has_channel: dict[str, bool] = field(default_factory=dict)
    has_mask: dict[str, bool] = field(default_factory=dict)
    masks_by_feat: dict[str, set[int]] = field(default_factory=dict)
    chs_by_feat: dict[str, set[int]] = field(default_factory=dict)
    col_by_key: dict[tuple[str, int | None, int | None], str] = field(
        default_factory=dict
    )
    heat_columns_present: bool = False

    @classmethod
    def from_columns(
        cls,
        columns: Sequence[str],
        derived_registry: Mapping[str, Mapping] | None = None,
    ) -> FeatureCatalog:
        """Parse dataframe column names into a catalog."""
        catalog = cls()

        claimed: dict[str, str] = {}
        for name, desc in (derived_registry or {}).items():
            column = desc.get("template")
            if isinstance(column, str) and column in columns:
                claimed[column] = name

        channels: set[int] = set()
        masks: set[int] = set()
        for col in columns:
            if col in claimed:
                continue
            gate = _FEATURE_CH_M_RE.match(col)
            if gate:
                channels.add(int(gate.group("ch")))
                masks.add(int(gate.group("m")))
        catalog.heat_columns_present = bool(channels and masks)

        for col in columns:
            if col in claimed:
                name = claimed[col]
                catalog.features.add(name)
                catalog.has_channel.setdefault(name, False)
                catalog.has_mask[name] = False
                catalog.col_by_key[(name, None, None)] = col
                continue

            match = _FEATURE_CH_M_RE.match(col)
            if match:
                feat = match.group("feat")
                ch = int(match.group("ch"))
                mask = int(match.group("m"))
                catalog.features.add(feat)
                catalog.has_channel[feat] = True
                catalog.has_mask[feat] = True
                catalog.masks_by_feat.setdefault(feat, set()).add(mask)
                catalog.chs_by_feat.setdefault(feat, set()).add(ch)
                catalog.col_by_key[(feat, ch, mask)] = col
                continue

            match = _FEATURE_M_RE.match(col)
            if match:
                feat = match.group("feat")
                mask = int(match.group("m"))
                catalog.features.add(feat)
                catalog.has_channel.setdefault(feat, False)
                catalog.has_mask[feat] = True
                catalog.masks_by_feat.setdefault(feat, set()).add(mask)
                catalog.col_by_key[(feat, None, mask)] = col

        return catalog

    def __bool__(self) -> bool:
        """True when at least one feature was discovered."""
        return bool(self.features)

    def selectable_features(
        self, excluded: Iterable[str] = EXCLUDED_TREE_FEATURES
    ) -> list[str]:
        """Sorted feature names suitable for the Feature dropdown."""
        excluded = set(excluded)
        return sorted(f for f in self.features if f not in excluded)

    def masks_for(self, feat: str) -> list[int]:
        """Sorted mask indices available for ``feat``, empty if it has none."""
        if not self.has_mask.get(feat, True):
            return []
        return sorted(self.masks_by_feat.get(feat, {1}))

    def channels_for(self, feat: str) -> list[int]:
        """Sorted channel indices available for ``feat``."""
        return sorted(self.chs_by_feat.get(feat, {1}))

    def resolve(
        self, feat: str, mask: int | None, channel: int | None
    ) -> list[str]:
        """Column(s) to draw for a dropdown selection."""
        if not self.has_mask.get(feat, True):
            column = self.col_by_key.get((feat, None, None))
            return [column] if column is not None else []

        if mask is None:
            return []
        has_ch = self.has_channel.get(feat, False)

        if has_ch and channel == -1:
            return [
                self.col_by_key[(feat, ch, mask)]
                for ch in self.channels_for(feat)
                if (feat, ch, mask) in self.col_by_key
            ]

        key = (
            (feat, int(channel), mask)
            if has_ch and isinstance(channel, int) and channel >= 0
            else (feat, None, mask)
        )
        col = self.col_by_key.get(key)
        return [col] if col is not None else []
