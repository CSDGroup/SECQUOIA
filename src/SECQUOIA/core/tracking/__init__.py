"""Tracking core: track loading, lineage numbering, and lineage editing.

Provides Qt event handlers for interactively editing cell lineage trees
(new IDs, divisions, splits, fuses).
"""

from __future__ import annotations

from SECQUOIA.core.tracking.edits import (
    on_division_clicked,
    on_new_id_clicked,
    on_remove_division_clicked,
    on_split_tree,
)
from SECQUOIA.core.tracking.fuse import on_fuse_trees_clicked
from SECQUOIA.core.tracking.history import (
    on_redo_tracking_clicked,
    on_undo_tracking_clicked,
)

__all__ = [
    "on_division_clicked",
    "on_fuse_trees_clicked",
    "on_new_id_clicked",
    "on_redo_tracking_clicked",
    "on_remove_division_clicked",
    "on_split_tree",
    "on_undo_tracking_clicked",
]
