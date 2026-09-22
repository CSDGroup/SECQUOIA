"""Building and rebuilding the Out tree (outliers and close-mask cases of the track list)."""

from __future__ import annotations

import contextlib
import re

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QMessageBox

from SECQUOIA.config import STYLE
from SECQUOIA.core.outlier_detection.close_masks import (
    unique_review_ids,
    update_unique_close_mask_ids,
)
from SECQUOIA.gui.curation_tree import (
    _build_curation_maps,
    _populate_curation_tree,
    update_list,
)
from SECQUOIA.gui.track_selection import handle_item_click
from SECQUOIA.utils.helpers import extract_unique_tracknumbers

__all__ = [
    "_show_info_dialog",
    "_update_outlier_list",
    "auto_select_first_item",
]


def _outliers_is_empty(outliers) -> bool:
    """Return True if `outliers` is missing or holds zero identifiers."""
    if outliers is None:
        return True
    if hasattr(outliers, "size"):
        return int(getattr(outliers, "size", 0)) == 0
    return len(outliers) == 0


def _show_info_dialog(main_window, title: str, text: str) -> None:
    """Pop up an information dialog in the outlier-dialog style."""
    msg = QMessageBox(main_window)
    msg.setIcon(QMessageBox.Information)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setStyleSheet(
        f"QLabel {{ font-size: {STYLE.FONT_SIZE_outlier}px; "
        f'font-family: "{STYLE.FONT_outlier}"; }}'
    )
    msg.exec_()


def _show_no_outliers_dialog(main_window) -> None:
    """Pop up an information dialog telling the user nothing was found."""
    _show_info_dialog(
        main_window,
        "Outliers",
        "No outliers or close masks have been found!",
    )


def _recheck_all_outliers_toggle(main_window) -> None:
    """Switch back to the ALL view and make sure the tree is actually rebuilt."""
    all_cb = getattr(main_window, "ALL", None)
    if all_cb is None:
        update_list(main_window)
        return

    with contextlib.suppress(RuntimeError, AttributeError):
        already_checked = all_cb.isChecked()
        all_cb.setChecked(True)
        if already_checked:
            update_list(main_window)


def _coerce_to_list(outliers):
    """Normalize an outliers container (ndarray, list, or scalar) into a plain list."""
    try:
        return (
            outliers.tolist()
            if hasattr(outliers, "tolist")
            else list(outliers)
        )
    except TypeError:
        return [outliers]


def _outlier_sort_key(ident):
    """Sort key ordering outlier identifiers by trailing numeric suffix, then lexically."""
    s = str(ident).strip()
    suffix = s.split("-")[-1]

    m = re.search(r"(\d+(?:\.\d+)?)\s*$", suffix)
    if m:
        return (0, float(m.group(1)))

    numbers = re.findall(r"\d+(?:\.\d+)?", s)
    if numbers:
        return (0, float(numbers[-1]))

    return (1, suffix.lower(), s.lower())


def _fill_tree_with_idents(main_window, idents) -> None:
    """Rebuild the tree so it lists only `idents`.

    Keeps the same parent/child structure and the same curation status and
    active symbols as the full tree.
    """
    main_window.tree_widget.clear()

    sorted_idents = sorted(_coerce_to_list(idents), key=_outlier_sort_key)

    try:
        data = extract_unique_tracknumbers(main_window)
    except (RuntimeError, AttributeError, TypeError):
        data = {}

    ident_track_pairs = [
        (ident_full, data.get(str(ident_full), []))
        for ident_full in sorted_idents
    ]

    df = getattr(main_window, "filtered_df", None)
    curation_maps = _build_curation_maps(df)

    _populate_curation_tree(main_window, ident_track_pairs, curation_maps)
    main_window.tree_widget.collapseAll()


def _review_idents(main_window) -> list:
    """The identifications for the Out list, read fresh from ``filtered_df``.

    Outliers come from ``unique_outliers_ids``; the close-mask cases are
    re-read here so the list is right after a change of position.
    """
    update_unique_close_mask_ids(main_window)
    return unique_review_ids(main_window)


def _update_outlier_list(main_window, *, interactive: bool = True) -> None:
    """Rebuild the OUT tree so it lists the outlier and close-mask idents."""
    idents = _review_idents(main_window)
    if _outliers_is_empty(idents):
        if interactive:
            _show_no_outliers_dialog(main_window)
        _recheck_all_outliers_toggle(main_window)
        return

    _fill_tree_with_idents(main_window, idents)


def _in_outlier_view(main_window) -> bool:
    """Return True when the user currently has the OUT list selected."""
    cb = getattr(main_window, "Outliers", None)
    try:
        return bool(cb is not None and cb.isChecked())
    except (RuntimeError, AttributeError):
        return False


def _rebuild_active_list(main_window) -> None:
    """Repopulate whichever tree list the user actually has selected."""
    if not _in_outlier_view(main_window):
        update_list(main_window)
        return

    if not _review_idents(main_window):
        _recheck_all_outliers_toggle(main_window)
        return

    _update_outlier_list(main_window, interactive=False)


def auto_select_first_item(main_window) -> None:
    """Select the first top-level item once control returns to the event loop."""

    def _select():
        tw = getattr(main_window, "tree_widget", None)
        if tw is None:
            return
        first = tw.topLevelItem(0)
        if first is None:
            return
        tw.setCurrentItem(first)
        first.setSelected(True)
        handle_item_click(main_window, first)

    QTimer.singleShot(0, _select)
