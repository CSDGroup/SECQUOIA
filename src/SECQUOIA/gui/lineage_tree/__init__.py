"""Lineage tree view: layout, rendering, controls and interaction."""

import importlib
from typing import TYPE_CHECKING

__all__ = [
    "_clear_track_selection",
    "_select_all_tracks",
    "refresh_lineage_values",
    "reset_lineage_zoom",
]

_SUBMODULE_BY_NAME = {
    "_clear_track_selection": "lineage_selection",
    "_select_all_tracks": "lineage_selection",
    "refresh_lineage_values": "lineage_zoom",
    "reset_lineage_zoom": "lineage_zoom",
}

if TYPE_CHECKING:
    from SECQUOIA.gui.lineage_tree.lineage_selection import (
        _clear_track_selection,
        _select_all_tracks,
    )
    from SECQUOIA.gui.lineage_tree.lineage_zoom import (
        refresh_lineage_values,
        reset_lineage_zoom,
    )


def __getattr__(name: str):
    """Resolve the public entry points from their owning submodule on first access."""
    submodule = _SUBMODULE_BY_NAME.get(name)
    if submodule is not None:
        module = importlib.import_module(f"{__name__}.{submodule}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Advertise the lazily-resolved names to ``dir()`` and tab completion."""
    return sorted([*globals(), *__all__])
