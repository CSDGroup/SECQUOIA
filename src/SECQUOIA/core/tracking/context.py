"""Reading the state an edit acts on.

Pulls the current position, Identification, time point and TrackNumber
off ``main_window``.
"""

from __future__ import annotations

import contextlib
import logging
from typing import NamedTuple

import pandas as pd

from SECQUOIA.core.tracking.numbering import _safe_num
from SECQUOIA.utils.positions import position_number_at_current_index
from SECQUOIA.utils.timing import current_t_range as _current_t_range

LOG = logging.getLogger(__name__)


def _get_time_interval(main_window) -> float:
    """Return the seconds between frames, from the cache or the Qt field."""
    cached = getattr(main_window, "time_interval", None)
    if cached is not None:
        with contextlib.suppress(ValueError, TypeError):
            return float(cached)
    with contextlib.suppress(
        ValueError, TypeError, AttributeError, RuntimeError
    ):
        return float(main_window.Time_input.text())
    return 0.0


def _get_position_number(main_window) -> int:
    """Return the position number currently shown, or 0 if unknown."""
    number = position_number_at_current_index(main_window)
    return number if number is not None else 0


def _get_current_ident(main_window) -> str:
    """Return current Identification."""
    with contextlib.suppress(AttributeError, IndexError, KeyError):
        return str(
            getattr(
                main_window,
                "ident",
                str(main_window.unique_ids[main_window.current_ident_index]),
            )
        )
    return str(getattr(main_window, "ident", ""))


def _get_current_time_index(main_window) -> int:
    """Return the 0-based frame index currently displayed."""
    with contextlib.suppress(AttributeError, ValueError, TypeError):
        return int(getattr(main_window, "current_time_index", 0))
    return 0


def _current_t_bounds(main_window) -> tuple[int, int, int]:
    """Return ``(t_file_min, t_idx_min, t_idx_max)`` for the current view."""
    t_file_min = t_idx_min = t_idx_max = None
    with contextlib.suppress(AttributeError, ValueError, KeyError):
        t_file_min, _, t_idx_min, t_idx_max = _current_t_range(main_window)
    return (
        int(t_file_min) if t_file_min is not None else 0,
        int(t_idx_min) if t_idx_min is not None else 0,
        int(t_idx_max) if t_idx_max is not None else 0,
    )


def _infer_tracknumber_at_t(
    df: pd.DataFrame, ident: str, t: int
) -> int | None:
    """Return TrackNumber."""
    try:
        cand = df[(df["Identification"].astype(str) == ident) & (df["t"] == t)]
        if "TrackNumber" not in cand.columns:
            return None
        u = _safe_num(cand["TrackNumber"]).dropna().unique()
        if u.size == 1:
            return int(u[0])
    except (KeyError, ValueError, TypeError):
        pass
    return None


class _EditContext(NamedTuple):
    df: pd.DataFrame
    ident: str
    t: int
    tracknumber: int


def _resolve_edit_context(main_window, log_prefix: str) -> _EditContext | None:
    """Resolve the frame, Identification, time point and TrackNumber to edit."""
    if not _require_data_loaded(main_window):
        return None

    df = getattr(main_window, "track_df", None)
    if not isinstance(df, pd.DataFrame) or df.empty:
        LOG.warning("[%s] track_df missing/empty.", log_prefix)
        return None

    filtered = getattr(main_window, "filtered_df", None)
    if not isinstance(filtered, pd.DataFrame) or filtered.empty:
        LOG.warning(
            "[%s] No tracking data for the current position.", log_prefix
        )
        return None

    position_idents = set(
        filtered.get("Identification", pd.Series(dtype=object))
        .dropna()
        .astype(str)
    )
    ident = _get_current_ident(main_window)
    if ident not in position_idents:
        LOG.warning(
            "[%s] Identification unknown for the current position; aborting.",
            log_prefix,
        )
        return None

    t = _get_current_time_index(main_window)

    tracknumber = getattr(main_window, "current_TrackNumber_plot", None)
    if tracknumber is None:
        tracknumber = _infer_tracknumber_at_t(filtered, ident, t)
    if tracknumber is None:
        LOG.warning("[%s] current TrackNumber unknown; aborting.", log_prefix)
        return None

    return _EditContext(df, ident, t, int(tracknumber))


def _ident_t_min(df: pd.DataFrame, ident: str, default: int) -> int:
    """First time point an Identification occupies, or ``default``."""
    with contextlib.suppress(AttributeError, KeyError, TypeError, ValueError):
        return int(
            df.loc[df["Identification"].astype(str) == ident, "t"].min()
        )
    return int(default)


def _ident_t_max(df: pd.DataFrame, ident: str, default: int) -> int:
    """Last time point an Identification occupies, or ``default``."""
    with contextlib.suppress(AttributeError, KeyError, TypeError, ValueError):
        return int(
            df.loc[df["Identification"].astype(str) == ident, "t"].max()
        )
    return int(default)


def _forward_t_max(
    main_window, df: pd.DataFrame, ident: str, default: int
) -> int:
    """Last time point an edit should extend to."""
    t_max = None
    with contextlib.suppress(AttributeError, ValueError, KeyError):
        _, _, _, t_max = _current_t_range(main_window)
    if t_max is None:
        return _ident_t_max(df, ident, default)
    return int(t_max)


def _existing_time_tracknumbers(
    df: pd.DataFrame, ident: str
) -> set[tuple[int, int]]:
    rows = df.loc[
        df["Identification"].astype(str) == ident, ["t", "TrackNumber"]
    ]
    times = _safe_num(rows["t"])
    numbers = _safe_num(rows["TrackNumber"])
    valid = times.notna() & numbers.notna()
    return set(
        zip(
            times[valid].astype(int),
            numbers[valid].astype(int),
            strict=False,
        )
    )


def _require_data_loaded(main_window) -> bool:
    """Return whether data is loaded, warning the user if it is not."""
    if not hasattr(main_window, "folder_list") or not main_window.folder_list:
        LOG.warning("Please first load CSV file and select folder")
        from SECQUOIA.gui.common.messages import show_folder_warning

        show_folder_warning(main_window)
        return False
    return True


def _widget_text(main_window, name: str) -> str:
    """Text of a named Qt field, or "" when it is absent or unreadable."""
    widget = getattr(main_window, name, None)
    with contextlib.suppress(AttributeError, KeyError, RuntimeError):
        if widget is not None:
            return str(widget.text()).strip()
    return ""


def _widget_int(main_window, name: str) -> int | None:
    """Return the rounded integer value of a named Qt field, or ``None``."""
    with contextlib.suppress(AttributeError, TypeError, ValueError):
        widget = getattr(main_window, name, None)
        if widget is None:
            return None
        text = str(widget.text()).strip()
        if not text:
            return None
        return int(round(float(text)))
    return None
