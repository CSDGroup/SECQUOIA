"""Helpers for managing add/remove-able rows inside a ``QGridLayout``.

Several dialogs (outlier detection's threshold-rule and sliding-window
tables, and similar row-per-item editors) share the same pattern: a grid of
widget rows, each with its own add/remove buttons, where rows can be
inserted, deleted, or reordered. These helpers implement that pattern once,
independent of what a row's fields actually mean.

Each row is represented as a ``dict`` of widgets with at least a
``"btn_bar"`` (the container for its add/remove buttons), a ``"btn_add"``,
and a ``"btn_rm"``. ``field_keys`` lists the remaining keys, in the order
they should appear as grid columns.
"""

from __future__ import annotations

import contextlib

from qtpy.QtWidgets import QComboBox, QListView

__all__ = [
    "remove_grid_row",
    "reposition_grid_rows",
    "update_row_remove_buttons",
    "use_qt_drawn_popup",
    "wire_row_buttons",
]


def use_qt_drawn_popup(combo: QComboBox) -> QComboBox:
    """Force a plain QComboBox to use Qt's own popup list view instead of the
    platform-native one, so a dark stylesheet actually applies to the
    dropdown. Native popups (notably on macOS) largely ignore QSS, which can
    leave dropdown items unreadable against a dark background."""
    combo.setView(QListView())
    return combo


def reposition_grid_rows(grid, rows, field_keys) -> None:
    """Replace each row's widgets in the grid so rows stay in list order after add/remove."""
    for i, row in enumerate(rows, start=1):
        widgets = [row[key] for key in field_keys] + [row["btn_bar"]]
        for widget in widgets:
            grid.removeWidget(widget)
        for col, widget in enumerate(widgets):
            grid.addWidget(widget, i, col)

        row["btn_add"].setProperty("row_index", i - 1)
        row["btn_rm"].setProperty("row_index", i - 1)


def update_row_remove_buttons(rows) -> None:
    """Enable each row's remove button only if more than one row remains."""
    only_one = len(rows) <= 1
    for row in rows:
        row["btn_rm"].setEnabled(not only_one)


def remove_grid_row(grid, rows, field_keys, idx) -> bool:
    """Pop and delete row ``idx`` from ``rows``/``grid``."""
    if len(rows) <= 1:
        return False
    if idx is None or idx < 0 or idx >= len(rows):
        return False

    row = rows.pop(int(idx))
    for widget in [row[key] for key in field_keys] + [row["btn_bar"]]:
        grid.removeWidget(widget)
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            widget.setParent(None)
            widget.deleteLater()
    return True


def wire_row_buttons(row, on_add_after, on_remove) -> None:
    """Connect a row's add/remove buttons to callbacks taking its row index."""
    row["btn_add"].clicked.connect(
        lambda _=False, btn=row["btn_add"]: on_add_after(
            btn.property("row_index")
        )
    )
    row["btn_rm"].clicked.connect(
        lambda _=False, btn=row["btn_rm"]: on_remove(btn.property("row_index"))
    )
