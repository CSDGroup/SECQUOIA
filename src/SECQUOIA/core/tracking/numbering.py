"""Binary TrackNumber math: the tree-position scheme lineage edits rely on.

Every lineage editing tool (division, split, fuse, undo) assumes TrackNumbers
follow root=1, daughters=2*parent[+1].
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _binary_tracknumbers(
    roots: Iterable[int], children: dict[int, list[int]]
) -> dict[int, int]:
    """Assign binary-tree TrackNumbers (root=1, daughters=2*parent[+1]).

    ``children`` must map each node id to its list of child ids, already
    sorted in birth/appearance order. All lineage editing tools (division,
    split, fuse, undo) assume this exact numbering via ``_is_desc`` and
    ``_path_from_root``, so every loader (CTC, Ultrack, btrack) must use
    this same helper rather than its own numbering scheme.
    """
    tracknum: dict[int, int] = {}
    for r in roots:
        tracknum[r] = 1
        queue = [r]
        while queue:
            p = queue.pop(0)
            for i, c in enumerate(children.get(p, [])):
                tracknum[c] = 2 * tracknum[p] + i
                queue.append(c)
    return tracknum


def _is_desc(h: int, root: int) -> bool:
    """Return True if TrackNumber is in the binary subtree of root."""
    try:
        h = int(h)
        root = int(root)
    except (ValueError, TypeError):
        return False
    if h < root:
        return False
    while h > root:
        h //= 2
    return h == root


def _path_from_root(h: int, root: int) -> list[str]:
    """Return L/R path from root."""
    h = int(h)
    root = int(root)
    steps = []
    while h > root:
        steps.append("R" if h % 2 else "L")
        h //= 2
    steps.reverse()
    return steps


def _apply_path_from_one(steps: list[str]) -> int:
    """Apply steps starting from node 1 to get new track number."""
    x = 1
    for s in steps:
        x = 2 * x if s == "L" else 2 * x + 1
    return x


def _apply_path_from(start: int, steps: list[str]) -> int:
    x = int(start)
    for s in steps:
        x = 2 * x if s == "L" else 2 * x + 1
    return x


def _remap_tracknumber_from_root(h: int, root: int) -> int:
    """Map any descendant h of root to new numbering with root → 1."""
    if h == root:
        return 1
    return _apply_path_from_one(_path_from_root(h, root))


def _remap_from_to(h: int, src_root: int, dst_root: int) -> int:
    """Map any descendant h of src_root into a tree where src_root→dst_root."""
    if h == src_root:
        return int(dst_root)
    return _apply_path_from(
        int(dst_root), _path_from_root(int(h), int(src_root))
    )


def _tracknumbers(df: pd.DataFrame) -> pd.Series:
    """Return TrackNumber as numbers, or an all-missing column when it is absent."""
    return _safe_num(df.get("TrackNumber", pd.Series(index=df.index)))


def _remap_tracknumbers(numbers: pd.Series, mapper) -> list:
    """Apply ``mapper`` to every TrackNumber, leaving missing values alone."""
    return [
        np.nan if pd.isna(value) else mapper(int(value))
        for value in _safe_num(numbers)
    ]


def _subtree_members(
    df: pd.DataFrame, scope_mask: pd.Series, root: int
) -> list[int]:
    """Sorted TrackNumbers in scope that descend from ``root``, ``root`` included."""
    in_scope = _tracknumbers(df).loc[scope_mask].dropna().astype(int)
    return sorted(
        {number for number in in_scope.unique() if _is_desc(number, root)}
    )


def _subtree_row_index(df: pd.DataFrame, scope_mask: pd.Series, root: int):
    """Index of rows in scope whose TrackNumber descends from ``root``."""
    members = _subtree_members(df, scope_mask, root) or [root]
    return df.index[scope_mask & _tracknumbers(df).isin(members)]


def _forward_scope_mask(
    df: pd.DataFrame, ident: str, t_from: int, tracknumber: int
) -> pd.Series:
    """Rows of one TrackNumber of one Identification, from ``t_from`` onward."""
    return (
        (
            (df["Identification"].astype(str) == ident)
            & (_safe_num(df["t"]) >= int(t_from))
            & (_safe_num(df["TrackNumber"]) == int(tracknumber))
        )
        .fillna(False)
        .astype(bool)
    )


def _descendant_index(
    df: pd.DataFrame,
    ident: str,
    t_from: int,
    root: int,
    keep: Iterable[int] = (),
):
    """Index of rows at/after ``t_from`` that sit in ``root``'s subtree."""
    scope = (df["Identification"].astype(str) == ident) & (
        df["t"] >= int(t_from)
    )
    numbers = _safe_num(df["TrackNumber"]).loc[scope].astype(int)
    in_subtree = numbers.apply(lambda value: _is_desc(value, int(root)))
    keep = list(keep)
    if keep:
        in_subtree &= ~numbers.isin(keep)
    return numbers.index[in_subtree]
