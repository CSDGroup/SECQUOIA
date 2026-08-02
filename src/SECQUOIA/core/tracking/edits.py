"""New-ID, division and split handlers: the single-lineage editing tools."""

from __future__ import annotations

import contextlib
import logging

import pandas as pd
from qtpy.QtWidgets import QInputDialog, QMessageBox

from SECQUOIA.core.tracking.context import (
    _current_t_bounds,
    _existing_time_tracknumbers,
    _forward_t_max,
    _get_position_number,
    _get_time_interval,
    _ident_t_max,
    _ident_t_min,
    _require_data_loaded,
    _resolve_edit_context,
)
from SECQUOIA.core.tracking.history import (
    _refill_realtime,
    _snapshot_for_undo,
    _write_back_and_refresh,
)
from SECQUOIA.core.tracking.identity import (
    _first_identification,
    _new_lineage_identity,
    _next_identification_after,
    _track_id_for_ident,
)
from SECQUOIA.core.tracking.numbering import (
    _descendant_index,
    _remap_from_to,
    _remap_tracknumber_from_root,
    _remap_tracknumbers,
    _subtree_row_index,
)
from SECQUOIA.core.tracking.row_builders import (
    _add_stub_rows_for_t_range,
    _align_like,
    _append_rows_to_track_df,
    _backfill_root_rows,
    _ensure_columns,
    _leave_blank_continuation,
    _stub_rows_for_missing_slots,
)
from SECQUOIA.gui.curation_tree import update_list
from SECQUOIA.utils.helpers import (
    ensure_current_df_subset,
    jump_to_identification,
)

LOG = logging.getLogger(__name__)


def on_new_id_clicked(main_window) -> None:
    """Create a new Identification with blank rows across the current time range."""
    if not _require_data_loaded(main_window):
        return

    _snapshot_for_undo(main_window)

    df = getattr(main_window, "filtered_df", None)
    starting_fresh = df is None or df.empty

    if starting_fresh:
        next_id = _first_identification(main_window)
        new_track_id = 1
        base_cols = list(
            getattr(main_window, "track_df", pd.DataFrame()).columns
        )
        if "Identification" not in base_cols:
            base_cols.append("Identification")
    else:
        derived = _next_identification_after(df.get("Identification"))
        if derived is None:
            return
        next_id, new_track_id = derived
        base_cols = list(df.columns)

    t_file_min, t_idx_min, t_idx_max = _current_t_bounds(main_window)

    new_df = _add_stub_rows_for_t_range(
        base_cols,
        next_id,
        range(t_idx_min, t_idx_max + 1),
        tracknumber=1,
        position_number=_get_position_number(main_window),
        time_interval=_get_time_interval(main_window),
        has_tidx="t_idx" in base_cols,
        has_tfile="t_file" in base_cols,
        track_id_val=new_track_id,
        t_file_value=t_file_min,
    )

    if starting_fresh:
        main_window.filtered_df = new_df.copy()
        main_window.unique_ids = [next_id]
    else:
        main_window.filtered_df = pd.concat([df, new_df], ignore_index=True)
        main_window.unique_ids = (
            main_window.filtered_df["Identification"].unique().tolist()
        )
        LOG.info(
            "[New ID] Added %d rows for %s. New filtered_df: %d",
            len(new_df),
            next_id,
            len(main_window.filtered_df),
        )

    _append_rows_to_track_df(main_window, new_df)

    main_window.track_df = _refill_realtime(main_window, main_window.track_df)
    main_window.filtered_df = _refill_realtime(
        main_window, main_window.filtered_df
    )

    ensure_current_df_subset(main_window)
    update_list(main_window)
    with contextlib.suppress(
        AttributeError, RuntimeError, ValueError, KeyError
    ):
        main_window.update_display_track_id()

    jump_to_identification(
        main_window, ident=next_id, t=t_idx_min, tracknumber=1
    )


def on_division_clicked(main_window) -> None:
    """Split the current TrackNumber into two daughters from the current time on."""
    context = _resolve_edit_context(main_window, "Divisions")
    if context is None:
        return
    df, ident, t_div, g = context

    _snapshot_for_undo(main_window)

    first_daughter, second_daughter = 2 * g, 2 * g + 1
    t_max = _forward_t_max(main_window, df, ident, default=t_div)
    base_cols = list(df.columns)

    add_df = _stub_rows_for_missing_slots(
        base_cols,
        ident,
        range(t_div, t_max + 1),
        (first_daughter, second_daughter),
        _existing_time_tracknumbers(df, ident),
        position_number=_get_position_number(main_window),
        time_interval=_get_time_interval(main_window),
        track_id_val=_track_id_for_ident(df, ident),
    )

    superseded = _descendant_index(
        df, ident, t_div, root=g, keep=(first_daughter, second_daughter)
    )
    if len(superseded):
        df = df.drop(index=superseded).copy()

    if add_df is not None and not add_df.empty:
        df = _ensure_columns(df, list(add_df.columns))
        df = pd.concat(
            [df, _align_like(add_df, list(df.columns))], ignore_index=True
        )

    main_window.current_TrackNumber_plot = first_daughter
    _write_back_and_refresh(
        main_window,
        df,
        log_msg=(
            f"[Divisions] Split TrackNumber {g} at t={t_div} -> daughters "
            f"{first_daughter},{second_daughter}. "
            f"Added {0 if add_df is None else len(add_df)} rows. "
            f"filtered_df={len(getattr(main_window, 'filtered_df', []))}, "
            f"track_df={len(df)}"
        ),
    )


def _collapse_division_to_parent(
    df: pd.DataFrame,
    ident: str,
    t_div: int,
    g: int,
    *,
    position_number: int,
    time_interval: float,
    track_id_val=None,
) -> pd.DataFrame:
    """Remove both daughters and leave the parent as blank rows."""
    t_max = _ident_t_max(df, ident, t_div)

    superseded = _descendant_index(df, ident, t_div, root=g, keep=(g,))
    if len(superseded):
        df = df.drop(index=superseded).copy()

    return _leave_blank_continuation(
        df,
        ident,
        t_div,
        t_max,
        g,
        position_number=position_number,
        time_interval=time_interval,
        track_id_val=track_id_val,
        keep_active=False,
    )


def _promote_daughter(
    df: pd.DataFrame,
    ident: str,
    t_div: int,
    g: int,
    keep_root: int,
    drop_root: int,
) -> pd.DataFrame:
    """Keep one daughter as the continuation of the parent, discard the other."""
    kept = _descendant_index(df, ident, t_div, root=keep_root)
    discarded = _descendant_index(df, ident, t_div, root=drop_root)

    if len(kept):
        df.loc[kept, "TrackNumber"] = _remap_tracknumbers(
            df.loc[kept, "TrackNumber"],
            lambda number: _remap_from_to(number, keep_root, g),
        )
    if len(discarded):
        df = df.drop(index=discarded).copy()
    return df


def on_remove_division_clicked(main_window) -> None:
    """Undo a division: keep one daughter, or neither."""
    context = _resolve_edit_context(main_window, "RemoveDivision")
    if context is None:
        return
    df, ident, t_div, g = context

    first_daughter, second_daughter = 2 * g, 2 * g + 1
    options = [
        f"Keep {first_daughter}",
        f"Keep {second_daughter}",
        "Keep none",
    ]
    choice, confirmed = QInputDialog.getItem(
        main_window, "Remove Division", "Choose action:", options, 0, False
    )
    if not confirmed or not choice:
        return

    in_scope = (df["Identification"].astype(str) == ident) & (df["t"] >= t_div)
    if not in_scope.any():
        LOG.warning("[RemoveDivision] No rows in scope to modify.")
        return

    _snapshot_for_undo(main_window)

    if choice == "Keep none":
        df = _collapse_division_to_parent(
            df,
            ident,
            t_div,
            g,
            position_number=_get_position_number(main_window),
            time_interval=_get_time_interval(main_window),
            track_id_val=_track_id_for_ident(
                df, ident, fallback_to_name=False
            ),
        )
    else:
        keep_root = (
            first_daughter
            if choice.startswith(f"Keep {first_daughter}")
            else second_daughter
        )
        drop_root = (
            second_daughter if keep_root == first_daughter else first_daughter
        )
        df = _promote_daughter(df, ident, t_div, g, keep_root, drop_root)

    _write_back_and_refresh(
        main_window,
        df,
        log_msg=(
            f"[RemoveDivision] g={g}, t>={t_div}. Action='{choice}'. "
            f"filtered_df={len(getattr(main_window, 'filtered_df', []))}, "
            f"track_df={len(df)}"
        ),
    )


def on_split_tree(main_window):
    """Detach the selected subtree into a brand new lineage."""
    context = _resolve_edit_context(main_window, "SplitTree")
    if context is None:
        return
    df, ident, t_split, g = context

    position_number = _get_position_number(main_window)
    time_interval = _get_time_interval(main_window)
    t_min = _ident_t_min(df, ident, 0)
    t_max = _ident_t_max(df, ident, t_split)

    in_future = (df["Identification"].astype(str) == ident) & (
        df["t"] >= t_split
    )
    move_index = _subtree_row_index(df, in_future, g)
    if len(move_index) == 0:
        QMessageBox.information(
            main_window,
            "Split Tree",
            "No future rows of the selected lineage to split.",
        )
        return

    _snapshot_for_undo(main_window)

    new_ident, new_track_id = _new_lineage_identity(df, ident)

    df_move = df.loc[move_index].copy()
    df_move["Identification"] = new_ident
    if "TrackNumber" in df_move.columns:
        df_move["TrackNumber"] = _remap_tracknumbers(
            df_move["TrackNumber"],
            lambda number: _remap_tracknumber_from_root(number, g),
        )
    if "track_id" in df_move.columns:
        df_move["track_id"] = new_track_id

    df_keep = _leave_blank_continuation(
        df.drop(index=move_index).copy(),
        ident,
        t_split,
        t_max,
        g,
        position_number=position_number,
        time_interval=time_interval,
        track_id_val=_track_id_for_ident(df, ident, fallback_to_name=False),
        keep_active=True,
    )

    df_move = _backfill_root_rows(
        df_move,
        new_ident,
        t_min,
        t_split,
        position_number=position_number,
        time_interval=time_interval,
        track_id_val=new_track_id,
    )

    df_keep = _ensure_columns(df_keep, list(df_move.columns))
    out = pd.concat(
        [df_keep, _align_like(df_move, list(df_keep.columns))],
        ignore_index=True,
    )

    _write_back_and_refresh(
        main_window,
        out,
        log_msg=(
            f"[SplitTree] ident='{ident}' split at t={t_split}, g={g} -> new "
            f"'{new_ident}' (track_id={new_track_id}). "
            f"Moved subtree re-rooted to TrackNumber=1; back-filled pre-split "
            f"rows. track_df={len(out)}, "
            f"filtered_df={len(getattr(main_window, 'filtered_df', []))}"
        ),
    )
    jump_to_identification(
        main_window, ident=new_ident, t=t_split, tracknumber=1
    )
