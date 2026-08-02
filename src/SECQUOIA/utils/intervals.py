"""Interval and histogram helpers shared by the outlier plotting dialogs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

__all__ = [
    "compute_bins",
    "intersect_intervals",
    "resolve_multi",
    "threshold_regions",
    "union_intervals",
]


def resolve_multi(selection: Sequence | None, all_values: Iterable) -> list:
    """Resolve an empty multi-selection as "all available values"."""
    if selection == [] and all_values:
        return list(all_values)
    return list(selection or [])


def compute_bins(arr: np.ndarray) -> int:
    """Compute a histogram bin count using the Freedman-Diaconis rule."""
    try:
        iqr = np.subtract(*np.percentile(arr, [75, 25]))
        bin_width = (2 * iqr) / np.cbrt(arr.size) if iqr > 0 else None
        bins = (
            int(np.ceil((arr.max() - arr.min()) / bin_width))
            if bin_width
            else 50
        )
        return max(5, min(200, bins))
    except (RuntimeError, AttributeError, TypeError, ValueError, IndexError):
        return 50


def threshold_regions(
    op_sym: str, thr: float, edges_local: Sequence[float]
) -> list[tuple[float, float]]:
    """Return the x-intervals satisfying a threshold comparison."""
    xmin, xmax = float(edges_local[0]), float(edges_local[-1])
    if op_sym in ("<", "<="):
        return [(xmin, float(thr))]
    if op_sym in (">", ">="):
        return [(float(thr), xmax)]
    if op_sym == "=":
        widths = np.diff(edges_local)
        bin_width = widths.min() if len(widths) else (xmax - xmin) / 50.0
        eps = max(bin_width, 1e-9)
        t = float(thr)
        return [(t - 0.5 * eps, t + 0.5 * eps)]
    return []


def union_intervals(
    iv: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Merge overlapping intervals into a sorted, compact union."""
    iv = list(iv)
    if not iv:
        return []
    iv = sorted((min(a, b), max(a, b)) for a, b in iv)
    out = [iv[0]]
    for a, b in iv[1:]:
        last_a, last_b = out[-1]
        if a <= last_b:
            out[-1] = (last_a, max(last_b, b))
        else:
            out.append((a, b))
    return out


def intersect_intervals(
    a: Iterable[tuple[float, float]], b: Iterable[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Return the intersection of two interval lists."""
    a, b = list(a), list(b)
    out = []
    for a1, a2 in a:
        for b1, b2 in b:
            lo, hi = max(a1, b1), min(a2, b2)
            if lo < hi:
                out.append((lo, hi))
    return union_intervals(out)
