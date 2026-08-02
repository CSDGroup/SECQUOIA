"""Fusing two lineage trees at a chosen time point."""

from __future__ import annotations

import contextlib
import logging
from typing import NamedTuple

import pandas as pd
from qtpy.QtWidgets import QMessageBox

from SECQUOIA.core.tracking.context import (
    _get_current_time_index,
    _infer_tracknumber_at_t,
    _widget_int,
    _widget_text,
)
from SECQUOIA.core.tracking.history import (
    _snapshot_for_undo,
    _write_back_and_refresh,
)
from SECQUOIA.core.tracking.identity import _new_lineage_identity
from SECQUOIA.core.tracking.numbering import (
    _remap_from_to,
    _remap_tracknumbers,
    _subtree_members,
    _tracknumbers,
)
from SECQUOIA.core.tracking.row_builders import _align_like

LOG = logging.getLogger(__name__)


class _FuseRequest(NamedTuple):
    """What the fuse dialog is asking for."""

    ident1: str
    ident2: str
    t_fuse: int
    g1: int
    g2: int


def _fuse_dialog_parent(main_window):
    """Return the fuse dialog if it is open, otherwise the main window."""
    return getattr(main_window, "fuse_dialog", None) or main_window


def _read_fuse_request(main_window, df: pd.DataFrame) -> _FuseRequest | None:
    """Read the fuse dialog, filling in TrackNumbers left blank by the user."""
    t_fuse = None
    with contextlib.suppress(AttributeError, KeyError, RuntimeError):
        t_fuse = int(main_window.fuse_time_spin.value())
    if t_fuse is None:
        t_fuse = _get_current_time_index(main_window)

    ident1 = _widget_text(main_window, "fuse_tree_id_field_1")
    ident2 = _widget_text(main_window, "fuse_tree_id_field_2")
    if not ident1 or not ident2 or ident1 == ident2:
        LOG.warning("[FuseTrees] Need two distinct Identifications.")
        return None

    g1 = _widget_int(main_window, "fuse_cell_field_1") or (
        _infer_tracknumber_at_t(df, ident1, t_fuse)
    )
    g2 = _widget_int(main_window, "fuse_cell_field_2") or (
        _infer_tracknumber_at_t(df, ident2, t_fuse)
    )
    if g1 is None or g2 is None:
        LOG.warning(
            "[FuseTrees] Could not determine TrackNumbers at t=%s: g1=%s, g2=%s",
            t_fuse,
            g1,
            g2,
        )
        QMessageBox.information(
            _fuse_dialog_parent(main_window),
            "Fuse Trees",
            "Could not determine one or both TrackNumbers at the selected time.",
        )
        return None

    return _FuseRequest(ident1, ident2, t_fuse, int(g1), int(g2))


def _graft_subtree(
    df: pd.DataFrame,
    request: _FuseRequest,
    subtree: list[int],
    new_ident: str,
    new_track_id: int,
) -> pd.DataFrame:
    """Move tree 2's subtree onto tree 1 and rename the result."""
    is_tree1 = df["Identification"].astype(str) == request.ident1
    is_tree2 = df["Identification"].astype(str) == request.ident2
    numbers = _tracknumbers(df)

    grafted = df[
        is_tree2 & (df["t"] >= request.t_fuse) & numbers.isin(subtree)
    ].copy()
    if "TrackNumber" in grafted.columns:
        grafted["TrackNumber"] = _remap_tracknumbers(
            grafted["TrackNumber"],
            lambda number: _remap_from_to(number, request.g2, request.g1),
        )
    grafted["Identification"] = new_ident
    if "track_id" in grafted.columns:
        grafted["track_id"] = int(new_track_id)

    occupied = {
        _remap_from_to(number, request.g2, request.g1) for number in subtree
    }
    displaced = (
        is_tree1 & (df["t"] >= request.t_fuse) & numbers.isin(list(occupied))
    )

    tree1_rows = df[is_tree1 & ~displaced].copy()
    if not tree1_rows.empty:
        tree1_rows["Identification"] = new_ident
        tree1_rows["track_id"] = int(new_track_id)

    untouched = df[~(is_tree1 | is_tree2)].copy()
    if "track_id" not in untouched.columns:
        untouched["track_id"] = 0

    target_cols = list(df.columns)
    if "track_id" not in target_cols:
        target_cols.append("track_id")

    return pd.concat(
        [
            _align_like(untouched, target_cols),
            _align_like(tree1_rows, target_cols),
            _align_like(grafted, target_cols),
        ],
        ignore_index=True,
    )


def on_fuse_trees_clicked(main_window) -> None:
    """Fuse the subtree of one lineage into another at the selected fuse time.

    Reads two Tree IDs and two TrackNumbers from the fuse dialog, grafts tree
    2's subtree onto tree 1's chosen branch, and writes the combined lineage
    out under a fresh Identification.
    """
    df = getattr(main_window, "track_df", None)
    if not isinstance(df, pd.DataFrame) or df.empty:
        LOG.warning("[FuseTrees] track_df missing/empty.")
        return

    request = _read_fuse_request(main_window, df)
    if request is None:
        return

    is_tree1 = df["Identification"].astype(str) == request.ident1
    is_tree2 = df["Identification"].astype(str) == request.ident2
    if not is_tree1.any() or not is_tree2.any():
        LOG.warning(
            "[FuseTrees] One of the Identifications has no rows in track_df."
        )
        return

    subtree = _subtree_members(
        df, is_tree2 & (df["t"] >= request.t_fuse), request.g2
    )
    if not subtree:
        QMessageBox.information(
            _fuse_dialog_parent(main_window),
            "Fuse Trees",
            "No rows from Tree ID 2 in the selected subtree/time to move.",
        )
        return

    _snapshot_for_undo(main_window)

    new_ident, new_track_id = _new_lineage_identity(df, request.ident1)
    out = _graft_subtree(df, request, subtree, new_ident, new_track_id)

    occupied = sorted(
        {_remap_from_to(number, request.g2, request.g1) for number in subtree}
    )
    _write_back_and_refresh(
        main_window,
        out,
        log_msg=(
            f"[FuseTrees] Fused '{request.ident1}' (into TNs {occupied}) with "
            f"subtree of '{request.ident2}' (g2={request.g2}) at "
            f"t={request.t_fuse} -> new '{new_ident}' "
            f"(track_id={new_track_id}). track_df={len(out)}, "
            f"filtered_df={len(getattr(main_window, 'filtered_df', []))}"
        ),
    )

    # Local import: gui.dialogs.track_fuse_dialog imports on_fuse_trees_clicked
    # from this module, so importing it back at module scope would cycle.
    from SECQUOIA.gui.dialogs.track_fuse_dialog import reset_fuse_dialog_fields

    reset_fuse_dialog_fields(
        main_window, new_ident, request.t_fuse, request.g1
    )
