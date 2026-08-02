"""Position-range helpers shared by the GUI dialogs and the loading pipeline."""

from __future__ import annotations

import os
from collections.abc import Iterable

__all__ = [
    "coerce_int",
    "current_position_number",
    "detected_position_numbers",
    "position_index_from_number",
    "position_number_at_current_index",
    "position_number_from_folder",
    "resolve_position_range",
]


_POSITION_TOKEN = "_p"


def coerce_int(*candidates) -> int | None:
    """Return the first candidate that converts to ``int``, else ``None``."""
    for value in candidates:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def position_number_from_folder(path: str | os.PathLike) -> int | None:
    """Extract the trailing position number from a ``..._pXXXX`` folder path."""
    try:
        basename = os.path.basename(os.path.normpath(os.fspath(path)))
    except (TypeError, ValueError):
        return None
    if _POSITION_TOKEN not in basename:
        return None
    return coerce_int(basename.rsplit(_POSITION_TOKEN, 1)[-1])


def detected_position_numbers(
    main_window, pos_min: int | None = None, pos_max: int | None = None
) -> list[int]:
    """Return the sorted position numbers found in ``position_folders``."""
    numbers = sorted(
        n
        for n in (
            position_number_from_folder(p)
            for p in getattr(main_window, "position_folders", None) or []
        )
        if n is not None
    )
    if pos_min is not None:
        numbers = [n for n in numbers if n >= pos_min]
    if pos_max is not None:
        numbers = [n for n in numbers if n <= pos_max]
    return numbers


def resolve_position_range(main_window) -> tuple[int, int]:
    """Resolve the inclusive position range the user is working on."""
    detected = detected_position_numbers(main_window)

    pos_min = coerce_int(
        getattr(main_window, "position_min_selected", None),
        getattr(main_window, "position_start_selected", None),
        getattr(main_window, "position_min", None),
        detected[0] if detected else None,
        getattr(main_window, "current_position_number", None),
        1,
    )
    pos_max = coerce_int(
        getattr(main_window, "position_max_selected", None),
        getattr(main_window, "position_end_selected", None),
        getattr(main_window, "position_max", None),
        detected[-1] if detected else None,
        pos_min,
    )

    if pos_max < pos_min:
        pos_min, pos_max = pos_max, pos_min
    return pos_min, pos_max


def position_index_from_number(main_window, pos_number: int | None) -> int:
    """Map a position *number* to its index in ``position_folders``."""
    number = coerce_int(pos_number)
    if number is None:
        return 0
    for i, folder in enumerate(
        getattr(main_window, "position_folders", None) or []
    ):
        if position_number_from_folder(folder) == number:
            return i
    return 0


def position_number_at_current_index(main_window) -> int | None:
    """Return the position number for ``position_folders[current_position_index]``.

    Unlike :func:`current_position_number`, this ignores any cached
    ``main_window.current_position_number`` value and always re-derives the
    number from the folder list.
    """
    folders = getattr(main_window, "position_folders", None) or []
    idx = coerce_int(getattr(main_window, "current_position_index", None))
    if idx is not None and 0 <= idx < len(folders):
        return position_number_from_folder(folders[idx])
    return None


def current_position_number(
    main_window, fallback: Iterable[int] | None = None
) -> int | None:
    """Return the position number currently displayed in the viewer."""
    number = coerce_int(getattr(main_window, "current_position_number", None))
    if number is not None:
        return number

    number = position_number_at_current_index(main_window)
    if number is not None:
        return number

    if fallback:
        return next(iter(fallback), None)
    return None
