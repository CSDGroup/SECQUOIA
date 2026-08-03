"""Resolving the outlier the user is currently looking at, and rechecking outlier status for a single point after an edit."""

from __future__ import annotations

import logging

import pandas as pd

from SECQUOIA.config import Rule
from SECQUOIA.core.outlier_detection.detection import (
    RulesPack,
    compare_series_op,
    resolve_feature_columns,
    sliding_window_outlier_mask,
)
from SECQUOIA.core.outlier_detection.track_sync import (
    update_outlier_detection_in_track_df_fast,
    update_unique_outliers_ids,
)
from SECQUOIA.core.segmentation.mask_selection import ensure_current_df_subset

LOG = logging.getLogger(__name__)

__all__ = [
    "_normalize_unique_ids",
    "apply_outlier_selection_for_current_point",
    "resolve_current_ident",
]


def _refresh_outlier_views(
    main_window, *, outcol: str = "Outlier_detection"
) -> None:
    """Rebuild the outlier subset, list and star markers from the current data."""
    # circular-import: gui.outlier imports core.outlier_detection at module load time,
    # so these can only be pulled in once everything has finished loading.
    from SECQUOIA.gui.outlier.markers import update_outlier_marker
    from SECQUOIA.gui.outlier.outlier_list import _rebuild_active_list

    for step in (
        lambda: ensure_current_df_subset(main_window),
        lambda: update_unique_outliers_ids(main_window, outcol=outcol),
        lambda: _clamp_outlier_index(main_window),
        lambda: _rebuild_active_list(main_window),
        lambda: update_outlier_marker(main_window),
    ):
        try:
            step()
        except (
            RuntimeError,
            AttributeError,
            TypeError,
            ValueError,
            KeyError,
        ) as e:
            LOG.warning("[Recheck] View refresh step failed: %s", e)


def _clamp_outlier_index(main_window) -> None:
    """Keep ``current_outlier_index`` inside ``unique_outliers_ids``."""
    outs = getattr(main_window, "unique_outliers_ids", None) or []
    try:
        idx = int(getattr(main_window, "current_outlier_index", 0) or 0)
    except (TypeError, ValueError):
        idx = 0
    main_window.current_outlier_index = (
        0 if not outs else max(0, min(idx, len(outs) - 1))
    )


def _normalize_unique_ids(main_window) -> list:
    """Return ``main_window.unique_ids`` as a plain list, never ``None``."""
    ids = getattr(main_window, "unique_ids", None)
    if ids is None:
        return []
    if isinstance(ids, str):
        return [ids]
    if hasattr(ids, "tolist"):
        try:
            listed = ids.tolist()
        except (RuntimeError, AttributeError, TypeError, ValueError):
            listed = None
        if isinstance(listed, list):
            return listed
    if isinstance(ids, (list | tuple | set)):
        return list(ids)
    try:
        return list(ids)
    except TypeError:
        return [ids]


def _is_usable_ident(value) -> bool:
    """Reject values that clearly are not a single Identification."""
    if value is None or not isinstance(value, str):
        return False
    text = value.strip()
    return (
        bool(text) and text != "None" and "\n" not in text and len(text) <= 200
    )


def resolve_current_ident(
    main_window, df_all=None, *, id_col: str = "Identification"
) -> str | None:
    """Resolve the Identification the user is currently looking at."""
    ids = _normalize_unique_ids(main_window)
    ids_str = [str(x) for x in ids]

    try:
        cur_idx = int(getattr(main_window, "current_ident_index", 0) or 0)
    except (TypeError, ValueError):
        cur_idx = 0

    # 1) Trust an explicitly selected ident when it still exists in the list.
    ident = getattr(main_window, "ident", None)
    if ident is not None and not _is_usable_ident(ident):
        LOG.warning(
            "[Recheck] Discarding unusable main_window.ident (%s); "
            "re-deriving from unique_ids.",
            type(ident).__name__,
        )
        main_window.ident = None
        ident = None
    if _is_usable_ident(ident):
        ident = str(ident)
        if ident in ids_str:
            main_window.current_ident_index = ids_str.index(ident)
            main_window.ident = ident
            return ident

    # 2) Fall back to the index, clamped into the valid range.
    if ids_str:
        cur_idx = max(0, min(cur_idx, len(ids_str) - 1))
        main_window.current_ident_index = cur_idx
        main_window.ident = ids_str[cur_idx]
        return ids_str[cur_idx]

    # 3) Nothing selected at all: take the first Identification in the data.
    if df_all is None:
        df_all = getattr(main_window, "filtered_df", None)
    if df_all is not None and len(df_all) > 0 and id_col in df_all.columns:
        col = df_all[id_col].dropna()
        if not col.empty:
            ident = str(col.iloc[0])
            main_window.current_ident_index = 0
            main_window.ident = ident
            return ident

    return None


def _resolve_current_time(main_window, df_all, ident, id_col, time_col) -> int:
    """Return the current time index, falling back to `ident`'s earliest time on error."""
    try:
        return int(getattr(main_window, "current_time_index", 0))
    except (TypeError, ValueError):
        sub = df_all[df_all[id_col].astype("string") == ident]
        return int(sub[time_col].min()) if not sub.empty else 0


def _resolve_target_row_indices(
    df_all, ident, t_val, track_no, id_col, time_col, track_col
) -> list:
    """Find filtered_df row indices for `ident`/`t_val`[/`track_no`], relaxing the TrackNumber filter to all tracks of `ident` if it matches nothing."""
    id_mask = df_all[id_col].astype(str) == ident
    t_mask = pd.to_numeric(df_all[time_col], errors="coerce") == t_val
    row_mask = id_mask & t_mask
    if track_col in df_all.columns and track_no is not None:
        row_mask &= pd.to_numeric(df_all[track_col], errors="coerce") == int(
            track_no
        )

    target_idxs = df_all.index[row_mask].to_list()
    if (
        not target_idxs
        and track_col in df_all.columns
        and track_no is not None
    ):
        relaxed = df_all.index[id_mask & t_mask].to_list()
        if relaxed:
            LOG.debug(
                "[Recheck] %s=%s matched nothing; falling back to all tracks "
                "of %s=%r.",
                track_col,
                track_no,
                id_col,
                ident,
            )
            target_idxs = relaxed

    return target_idxs


def _rule_operand_hits(df_all, idx, cols, op, val) -> bool:
    """Return True if any of `cols` at row `idx` satisfies operator `op` against `val`."""
    if op is None:
        return False
    for c in cols:
        if idx not in df_all.index:
            continue
        s = df_all.loc[[idx], c]
        if bool(compare_series_op(s, op, val).iloc[0]):
            return True
    return False


def _row_hits_rule(df_all, idx, rule: Rule) -> bool:
    """Return True if row `idx` satisfies `rule`'s threshold condition(s)."""
    cols = resolve_feature_columns(
        df_all, rule.feat, (rule.masks or None), (rule.channels or None)
    )
    if not cols:
        return False

    hit1 = _rule_operand_hits(df_all, idx, cols, rule.op1, rule.val1)
    if rule.op2 is None:
        return hit1

    hit2 = _rule_operand_hits(
        df_all,
        idx,
        cols,
        rule.op2,
        rule.val2 if rule.val2 is not None else 0.0,
    )
    return (hit1 and hit2) if rule.combine.upper() == "AND" else (hit1 or hit2)


def _sliding_window_hits(
    df_all,
    target_idxs,
    pack: RulesPack,
    *,
    id_col,
    tr_col,
    time_col,
) -> dict:
    """Return {row_index: bool} sliding-window outlier flags for `target_idxs`.

    Recomputed from each row's own branch (Identification + TrackNumber) full
    timeline, so this agrees with what a full `run_sliding_windows` pass would
    say instead of silently dropping sliding-window flags that threshold rules
    know nothing about.
    """
    if not target_idxs:
        return {}
    if not pack.sliding_windows:
        return dict.fromkeys(target_idxs, False)

    idents = df_all.loc[target_idxs, id_col].astype(str).unique().tolist()
    scope = df_all[id_col].astype(str).isin(idents)
    if tr_col in df_all.columns:
        tracks = (
            pd.to_numeric(df_all.loc[target_idxs, tr_col], errors="coerce")
            .dropna()
            .unique()
            .tolist()
        )
        if tracks:
            scope &= pd.to_numeric(df_all[tr_col], errors="coerce").isin(
                tracks
            )

    subset = df_all.loc[scope]
    if subset.empty:
        return dict.fromkeys(target_idxs, False)

    mask = sliding_window_outlier_mask(
        subset, pack, id_col=id_col, t_col=time_col, track_col=tr_col
    )
    return {idx: bool(mask.get(idx, False)) for idx in target_idxs}


def _apply_outlier_status_updates(
    df_all,
    target_idxs,
    rules,
    sliding_hits: dict,
    *,
    ident,
    track_no,
    t_val,
    id_col,
    tr_col,
    time_col,
    outcol,
) -> bool:
    """Evaluate `outcol` for each row in `target_idxs` against `rules` and the
    precomputed `sliding_hits`, logging and tracking any changes. Returns True
    if at least one row changed."""
    changed_any = False
    for idx in target_idxs:
        prev = df_all.at[idx, outcol]
        threshold_hit = any(_row_hits_rule(df_all, idx, r) for r in rules)
        new_val = (
            "Outlier"
            if (threshold_hit or sliding_hits.get(idx, False))
            else "OK"
        )
        if prev != new_val:
            df_all.at[idx, outcol] = new_val
            changed_any = True
            LOG.debug(
                "[Recheck] %s=%r, %s=%s, %s=%s: %s -> %s",
                id_col,
                ident,
                tr_col,
                None if track_no is None else int(track_no),
                time_col,
                t_val,
                prev,
                new_val,
            )
    return changed_any


def apply_outlier_selection_for_current_point(main_window) -> None:
    """Re-evaluate outlier status for the currently selected identification/time point.

    This updates the filtered dataframe, synchronizes the corresponding rows in
    ``track_df``, and refreshes the visible outlier list and markers.
    """
    ID_COL, TIME_COL, TR_COL, OUTCOL = (
        "Identification",
        "t",
        "TrackNumber",
        "Outlier_detection",
    )
    df_all = getattr(main_window, "filtered_df", None)
    if df_all is None or len(df_all) == 0:
        LOG.warning("[Recheck] filtered_df is missing or empty.")
        return

    ident = resolve_current_ident(main_window, df_all, id_col=ID_COL)
    if ident is None:
        LOG.warning(
            "[Recheck] No Identification selected and none available in "
            "filtered_df; skipping recheck."
        )
        _refresh_outlier_views(main_window, outcol=OUTCOL)
        return

    t_val = _resolve_current_time(main_window, df_all, ident, ID_COL, TIME_COL)
    track_no = getattr(main_window, "current_TrackNumber_plot", None)

    if OUTCOL not in df_all.columns:
        df_all[OUTCOL] = "OK"

    target_idxs = _resolve_target_row_indices(
        df_all, ident, t_val, track_no, ID_COL, TIME_COL, TR_COL
    )
    if not target_idxs:
        LOG.warning(
            "[Outlier] No filtered_df row for %s=%r, %s=%s, %s=%s.",
            ID_COL,
            ident,
            TR_COL,
            track_no,
            TIME_COL,
            t_val,
        )
        _refresh_outlier_views(main_window, outcol=OUTCOL)
        return

    pack_dict = getattr(main_window, "_last_outlier_rules", None)
    if not pack_dict:
        LOG.warning(
            "[Outlier] No saved outlier rules found. Run the full outlier selection first."
        )
        _refresh_outlier_views(main_window, outcol=OUTCOL)
        return
    pack = RulesPack.from_dict(pack_dict)

    sliding_hits = _sliding_window_hits(
        df_all,
        target_idxs,
        pack,
        id_col=ID_COL,
        tr_col=TR_COL,
        time_col=TIME_COL,
    )

    changed_any = _apply_outlier_status_updates(
        df_all,
        target_idxs,
        pack.rules,
        sliding_hits,
        ident=ident,
        track_no=track_no,
        t_val=t_val,
        id_col=ID_COL,
        tr_col=TR_COL,
        time_col=TIME_COL,
        outcol=OUTCOL,
    )

    main_window.filtered_df = df_all
    if changed_any:
        update_outlier_detection_in_track_df_fast(
            main_window, target_idxs, outcol=OUTCOL
        )
    _refresh_outlier_views(main_window, outcol=OUTCOL)
