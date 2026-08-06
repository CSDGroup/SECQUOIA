"""Tree-ID item construction: curation status and active-state icons,
tooltips, and population of the Tree-ID widget from filtered_df."""

import contextlib
import logging

import pandas as pd
import qtawesome as qta
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QMessageBox,
    QTreeWidgetItem,
)

from SECQUOIA.config import CURATIONROLE, CURATIONSTATUS

LOG = logging.getLogger(__name__)

_CURATION_STATUS_BY_CODE = {
    0: CURATIONSTATUS.CURATION_NOT_CHECKED,
    1: CURATIONSTATUS.CURATION_IN_PROGRESS,
    2: CURATIONSTATUS.CURATION_CHECKED,
}


def _curation_code(values) -> int:
    """Collapse a group's raw 'inspected' values into a single status code (0/1/2)."""
    codes = pd.to_numeric(values, errors="coerce").fillna(0).astype(int)
    if (codes == 0).any():
        return 0
    return 1 if (codes == 1).any() else 2


def _build_curation_maps(df):
    """Build per Identification / per TrackNumber curation status and active state
    lookup dicts from `filtered_df`. Returns (have_data, parent_status, parent_active,
    child_status, child_active)."""
    have = isinstance(df, pd.DataFrame) and not df.empty
    parent_status, parent_active, child_status, child_active = {}, {}, {}, {}
    if not have:
        return have, parent_status, parent_active, child_status, child_active

    df = df.copy()
    if "Identification" in df.columns:
        df["Identification"] = df["Identification"].astype(str)

    if {"Identification", "inspected"}.issubset(df.columns):
        parent_status = (
            df.groupby("Identification")["inspected"]
            .apply(_curation_code)
            .to_dict()
        )

    if {"Identification", "TrackNumber", "inspected"}.issubset(df.columns):
        child_status = (
            df.groupby(["Identification", "TrackNumber"])["inspected"]
            .apply(_curation_code)
            .to_dict()
        )

    if "active" in df.columns:
        df["__active_num__"] = pd.to_numeric(df["active"], errors="coerce")

        if {"Identification", "__active_num__"}.issubset(df.columns):
            parent_active = (
                df.groupby("Identification")["__active_num__"]
                .min()
                .fillna(1)
                .astype(int)
                .to_dict()
            )

        if {"Identification", "TrackNumber", "__active_num__"}.issubset(
            df.columns
        ):
            child_active = (
                df.groupby(["Identification", "TrackNumber"])["__active_num__"]
                .max()
                .fillna(1)
                .astype(int)
                .to_dict()
            )

    return have, parent_status, parent_active, child_status, child_active


def _populate_curation_tree(
    main_window, ident_track_pairs, curation_maps
) -> None:
    """Build one parent tree item per Identification, with one child per
    TrackNumber, applying the curation status and active- tate symbols
    from `curation_maps`.
    """
    have, parent_status, parent_active, child_status, child_active = (
        curation_maps
    )

    for ident_full, tracks in ident_track_pairs:
        ident_str = str(ident_full)

        parent = QTreeWidgetItem(main_window.tree_widget)
        parent.setText(0, ident_str.split("-")[-1])

        ps = _CURATION_STATUS_BY_CODE.get(
            parent_status.get(ident_str, 0),
            CURATIONSTATUS.CURATION_NOT_CHECKED,
        )
        apply_curation_status(parent, ps)
        apply_active_state(parent, bool(parent_active.get(ident_str, 1)))

        for tn in tracks:
            child = QTreeWidgetItem(parent)
            child.setText(0, f"{tn}")

            key = (ident_str, float(tn)) if have else None
            cs = _CURATION_STATUS_BY_CODE.get(
                child_status.get(key, 0), CURATIONSTATUS.CURATION_NOT_CHECKED
            )
            ca = (
                bool(child_active.get(key, 1)) if key in child_active else True
            )

            apply_curation_status(child, cs)
            apply_active_state(child, ca)


def update_list(main_window) -> None:
    """Repopulate the Tree-ID widget from `filtered_df`.

    Each item's active state and inspected status are restored, so
    deactivated cells reappear struck through rather than being dropped.
    """
    from SECQUOIA.utils.helpers import extract_unique_tracknumbers

    main_window.tree_widget.clear()

    with contextlib.suppress(RuntimeError, AttributeError, TypeError):
        main_window.refresh_tracking_button_states()

    try:
        data = extract_unique_tracknumbers(main_window)
    except (RuntimeError, AttributeError, TypeError):
        data = {}

    if not data:
        if not getattr(main_window, "folder_list", None):
            QMessageBox.information(
                main_window, "Load data", "Please load data first"
            )
        else:
            LOG.warning(
                "[Tree-IDs] No tracking data for the current position."
            )
        return

    df = getattr(main_window, "filtered_df", None)
    curation_maps = _build_curation_maps(df)

    _populate_curation_tree(main_window, data.items(), curation_maps)
    main_window.tree_widget.collapseAll()


def curation_icon(status: str) -> QIcon:
    """Return a QIcon based on curation status."""
    if status == CURATIONSTATUS.CURATION_CHECKED:
        return qta.icon("fa5s.check-circle", color="#1db954")
    if status == CURATIONSTATUS.CURATION_IN_PROGRESS:
        return qta.icon("fa5s.question-circle", color="#ffbf00")
    if status == CURATIONSTATUS.CURATION_NOT_CHECKED:
        return qta.icon("fa5s.times-circle", color="#e73ce7")
    return qta.icon("fa5s.circle")


def apply_curation_status(item, status: str) -> None:
    """Update QTreeWidgetItem with a curation status + icon + tooltip."""
    item.setData(0, CURATIONROLE.CURATION_ROLE, status)
    item.setIcon(0, curation_icon(status))
    pretty = {
        CURATIONSTATUS.CURATION_CHECKED: "Checked",
        CURATIONSTATUS.CURATION_IN_PROGRESS: "In progress",
        CURATIONSTATUS.CURATION_NOT_CHECKED: "Not checked",
    }.get(status, "Unknown")
    item.setToolTip(0, f"Curation status: {pretty}")


def apply_active_state(item, is_active: bool) -> None:
    """Mark an item active/deactivated (strikeout + tooltip)."""
    item.setData(0, CURATIONROLE.ACTIVE_ROLE, bool(is_active))

    font = item.font(0)
    font.setStrikeOut(not is_active)
    item.setFont(0, font)

    cur_tip = item.toolTip(0) or ""
    state_txt = "Active" if is_active else "Deactivated"
    if "Cell state:" in cur_tip:
        parts = [
            p for p in cur_tip.split("\n") if not p.startswith("Cell state:")
        ]
        cur_tip = "\n".join(parts)
    tip = (
        cur_tip + ("\n" if cur_tip else "") + f"Cell state: {state_txt}"
    ).strip()
    item.setToolTip(0, tip)
