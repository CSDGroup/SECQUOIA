"""Populates the Segmentation and Channel trees and syncs their selections."""

from __future__ import annotations

import contextlib

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLineEdit, QTreeWidgetItem

from SECQUOIA.gui.loading.experiment_paths import (
    _add_seg_tree_item,
    _find_analysis_dir,
)
from SECQUOIA.gui.loading.load_data.load_data_signals import reconnect
from SECQUOIA.gui.loading.load_data.load_data_summary import _update_summary

__all__ = [
    "_populate_segmentation_ui",
    "_on_seg_checkbox",
    "_populate_channels_ui",
    "_sync_channels_selected",
]


def _populate_segmentation_ui(main_window, seg_names):
    """Populate the Segmentation tree; keep seg paths & n_masks updated."""
    tree = main_window.seg_tree
    tree.blockSignals(True)
    tree.clear()
    analysis_dir = _find_analysis_dir(main_window)

    for name in seg_names:
        _add_seg_tree_item(tree, name, analysis_dir)

    tree.blockSignals(False)
    reconnect(
        tree, "itemChanged", lambda _i, _c: _on_seg_checkbox(main_window)
    )
    _on_seg_checkbox(main_window)


def _on_seg_checkbox(main_window):
    """Update segmentation_paths (absolute) and n_masks based on seg_tree checks."""
    tree = main_window.seg_tree
    selected = []
    for i in range(tree.topLevelItemCount()):
        it = tree.topLevelItem(i)
        if it.checkState(0) == Qt.Checked:
            full = it.data(0, Qt.UserRole)
            selected.append(full)
    main_window.segmentation_paths = selected
    main_window.n_masks = len(selected)
    _update_summary(main_window)


def _populate_channels_ui(main_window, channel_names):
    """Fill the channel tree from the detected channels, all checked, with tTt comments alongside."""
    tree = main_window.chan_tree
    tree.blockSignals(True)
    tree.clear()

    main_window.available_channels = list(channel_names)

    comment_map = getattr(main_window, "channel_comment_map", None)
    tree.setColumnCount(2)
    tree.setHeaderLabels(["Select channels to quantify", "Comment"])

    for ch in channel_names:
        item = QTreeWidgetItem([ch, ""])
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(0, Qt.Checked)  # default select all

        # Column 1: optional comment (if any)
        if comment_map and ch in comment_map:
            item.setText(1, comment_map.get(ch, "") or "")

        tree.addTopLevelItem(item)

    with contextlib.suppress(Exception):
        tree.header().setStretchLastSection(True)
        tree.header().resizeSection(
            0, max(140, tree.header().sectionSizeHint(0))
        )

    tree.blockSignals(False)
    reconnect(
        tree,
        "itemChanged",
        lambda _i, _c: _sync_channels_selected(main_window),
    )
    _sync_channels_selected(main_window)


def _sync_channels_selected(main_window):
    """Push the checked channels onto ``ids_channels`` and ``n_channels``, and rebuild ``FL_inputs``."""
    tree = main_window.chan_tree
    selected = []
    for i in range(tree.topLevelItemCount()):
        it = tree.topLevelItem(i)
        if it.checkState(0) == Qt.Checked:
            selected.append(it.text(0))

    main_window.n_channels = len(selected)
    main_window.ids_channels = selected
    _update_summary(main_window)

    for w in getattr(main_window, "FL_inputs", []):
        with contextlib.suppress(AttributeError, RuntimeError):
            w.deleteLater()

    main_window.FL_inputs = []
    for name in selected:
        le = QLineEdit()
        le.setReadOnly(True)
        le.setText(name)
        main_window.FL_inputs.append(le)
