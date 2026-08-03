"""Builds and syncs the Position -> Identification tree on the Tracking tab."""

from __future__ import annotations

import contextlib

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QTreeWidgetItem

from SECQUOIA.gui.loading.load_data.load_data_signals import reconnect
from SECQUOIA.gui.loading.load_data.load_data_summary import _update_summary

__all__ = [
    "_build_tracking_tree",
    "_select_all_tracking_tree",
    "_sync_tracking_selection",
]


def _fit_lineage_column(tree):
    """Widen column 0 to the widest visible Tree-ID, pushing the comment column right."""
    if tree.columnCount() < 2:
        return
    with contextlib.suppress(Exception):
        tree.resizeColumnToContents(0)
        tree.header().resizeSection(0, max(160, tree.header().sectionSize(0)))


def _build_tracking_tree(main_window):
    """Build a tri-state tree (Positions → Identifications) from main_window.track_df."""
    tree = main_window.tracking_tree
    tree.clear()

    df = getattr(main_window, "track_df", None)
    if (
        df is None
        or "Position" not in df.columns
        or "Identification" not in df.columns
    ):
        main_window.tracking_tree_group.setVisible(False)
        return

    show_pos_comment = (
        getattr(main_window, "tracking_format", "") == "tTt"
    ) and bool(getattr(main_window, "position_comment_map", {}))
    if show_pos_comment:
        tree.setColumnCount(2)
        tree.setHeaderLabels(["Select Cell Lineage Tree", "Position Comment"])
        with contextlib.suppress(Exception):
            tree.header().setStretchLastSection(True)
    else:
        tree.setColumnCount(1)
        tree.setHeaderLabels(["Select Cell Lineage Tree"])

    pos_comment_map = getattr(main_window, "position_comment_map", {})

    try:
        positions = sorted({int(p) for p in df["Position"].dropna().unique()})
    except (ValueError, TypeError):
        positions = sorted(set(df["Position"].dropna().unique()))

    for pos in positions:
        parent = QTreeWidgetItem(
            tree, [f"Position {pos}"] + ([""] if show_pos_comment else [])
        )
        tristate_flag = getattr(Qt, "ItemIsAutoTristate", None)
        if tristate_flag is None:
            tristate_flag = getattr(Qt, "ItemIsTristate", None)

        flags = parent.flags() | Qt.ItemIsUserCheckable
        if tristate_flag is not None:
            flags |= tristate_flag

        parent.setFlags(flags)
        parent.setCheckState(0, Qt.Unchecked)

        if show_pos_comment:
            comment = (
                pos_comment_map.get(int(pos), "")
                if isinstance(pos, int)
                else pos_comment_map.get(pos, "")
            )
            if comment:
                parent.setText(1, str(comment))

        ids = sorted(
            {
                str(x)
                for x in df.loc[df["Position"] == pos, "Identification"]
                .dropna()
                .unique()
            }
        )
        for ident in ids:
            child = QTreeWidgetItem(
                parent, [ident] + ([""] if show_pos_comment else [])
            )
            child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
            child.setCheckState(0, Qt.Checked)

    tree.collapseAll()
    _fit_lineage_column(tree)
    main_window.tracking_tree_group.setVisible(True)

    reconnect(
        tree,
        "itemChanged",
        lambda _i, _c: _sync_tracking_selection(main_window),
    )
    reconnect(tree, "itemExpanded", lambda _i: _fit_lineage_column(tree))
    reconnect(tree, "itemCollapsed", lambda _i: _fit_lineage_column(tree))
    _sync_tracking_selection(main_window)
    _update_summary(main_window)


def _select_all_tracking_tree(main_window, select: bool):
    """Check/uncheck all Identification nodes."""
    tree = main_window.tracking_tree
    state = Qt.Checked if select else Qt.Unchecked
    try:
        tree.blockSignals(True)
        for i in range(tree.topLevelItemCount()):
            parent = tree.topLevelItem(i)
            for j in range(parent.childCount()):
                parent.child(j).setCheckState(0, state)
    finally:
        tree.blockSignals(False)
    _sync_tracking_selection(main_window)


def _sync_tracking_selection(main_window):
    """Update selected_positions/identifications + update the counters label."""
    tree = main_window.tracking_tree
    sel_pos, sel_ids = set(), set()
    total_pos = tree.topLevelItemCount()
    total_identifications = 0
    selected_id_count = 0

    for i in range(total_pos):
        parent = tree.topLevelItem(i)
        try:
            pos = int(str(parent.text(0)).split()[-1])
        except (ValueError, TypeError):
            pos = str(parent.text(0))

        child_count = parent.childCount()
        total_identifications += child_count

        checked = 0
        for j in range(child_count):
            ch = parent.child(j)
            if ch.checkState(0) == Qt.Checked:
                checked += 1
                selected_id_count += 1
                sel_ids.add(ch.text(0))
        if checked:
            sel_pos.add(pos)

    if getattr(main_window, "tracking_counts_label", None):
        main_window.tracking_counts_label.setText(
            f"Selected Positions: {len(sel_pos)} / {total_pos}    "
            f"Selected Lineage Trees: {selected_id_count} / {total_identifications}"
        )
