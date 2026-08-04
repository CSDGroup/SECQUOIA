"""Generating new Identification / track_id values for lineage edits."""

from __future__ import annotations

import contextlib
import logging
import re

import pandas as pd

from SECQUOIA.core.tracking.context import _get_position_number
from SECQUOIA.core.tracking.numbering import _safe_num

LOG = logging.getLogger(__name__)


def _next_ident_like(s: str, all_ids: pd.Series) -> str:
    """Return the next free Identification sharing."""
    m = re.match(r"^(.*?-.*?-)(\d+)$", str(s).strip())
    if not m:
        base, width, start = s + "-", 3, 0
    else:
        base = m.group(1)
        width = len(m.group(2))
        start = 0

    max_num = start
    for val in all_ids.dropna().astype(str):
        mm = re.match(rf"^({re.escape(base)})(\d+)$", val.strip())
        if not mm:
            continue
        with contextlib.suppress(ValueError):
            n = int(mm.group(2))
            if n > max_num:
                max_num = n
    return f"{base}{max_num + 1:0{width}d}"


def _first_identification(main_window) -> str:
    """Build a position's first Identification, e.g. ``EXP-p0002-001``."""
    base = getattr(main_window, "experiment_name", "").strip().rstrip("-")
    position_number = _get_position_number(main_window)
    return f"{base}-p{int(position_number):04d}-001"


def _next_identification_after(
    ids_series: pd.Series | None,
) -> tuple[str, int] | None:
    """Return ``(next Identification, its numeric suffix)`` or None."""
    if ids_series is None or ids_series.dropna().empty:
        LOG.warning(
            "[New ID] 'Identification' column missing or empty; cannot infer base."
        )
        return None

    last_id = str(ids_series.dropna().iloc[-1]).strip()
    match = re.match(r"^(.*?-.*?-)(\d+)$", last_id)
    if not match:
        LOG.warning(
            "[New ID] Could not parse last Identification: %r", last_id
        )
        return None

    base_prefix = match.group(1)
    width = len(match.group(2))

    highest = 0
    for value in ids_series.dropna().astype(str):
        numbered = re.match(
            rf"^({re.escape(base_prefix)})(\d+)$", value.strip()
        )
        if not numbered:
            continue
        with contextlib.suppress(ValueError):
            highest = max(highest, int(numbered.group(2)))

    next_number = highest + 1
    return f"{base_prefix}{next_number:0{width}d}", next_number


def _track_id_for_ident(
    df: pd.DataFrame, ident: str, *, fallback_to_name: bool = True
):
    """Return the ``track_id`` recorded for ``ident``, or None."""
    if "track_id" not in df.columns:
        return None
    with contextlib.suppress(KeyError, IndexError):
        values = (
            df.loc[df["Identification"].astype(str) == ident, "track_id"]
            .dropna()
            .unique()
        )
        if values.size > 0:
            return values[0]
        if fallback_to_name:
            match = re.search(r"-(\d+)$", ident)
            if match:
                return int(match.group(1))
    return None


def _new_lineage_identity(
    df: pd.DataFrame, source_ident: str
) -> tuple[str, int]:
    """Next free Identification derived from ``source_ident``, and its track_id."""
    new_ident = _next_ident_like(source_ident, df["Identification"])

    new_track_id = None
    match = re.search(r"-(\d+)$", new_ident)
    if match:
        with contextlib.suppress(
            AttributeError, KeyError, RuntimeError, TypeError, ValueError
        ):
            new_track_id = int(match.group(1))
    if new_track_id is None:
        with contextlib.suppress(
            AttributeError, KeyError, RuntimeError, TypeError, ValueError
        ):
            highest = _safe_num(
                df.get("track_id", pd.Series([], dtype=float))
            ).max()
            if pd.notna(highest):
                new_track_id = int(highest) + 1
    return new_ident, int(1 if new_track_id is None else new_track_id)
