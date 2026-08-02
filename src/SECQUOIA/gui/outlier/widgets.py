"""Widgets shared by the outlier dialogs."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QStandardItem, QStandardItemModel
from qtpy.QtWidgets import QComboBox, QListView, QSizePolicy

from SECQUOIA.gui.common.ui_utils import use_readable_combo_popup_on_macos

__all__ = ["CheckableComboBox", "set_combo_checks"]


class CheckableComboBox(QComboBox):
    """A combo box whose items carry independent check states."""

    def __init__(self, placeholder: str = "Select...", parent=None):
        """Create an empty checkable combo showing ``placeholder`` when nothing is checked."""
        super().__init__(parent)
        self._placeholder = placeholder
        self.setView(QListView())
        self.setModel(QStandardItemModel(self))
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        use_readable_combo_popup_on_macos(self)
        self.view().pressed.connect(self._on_item_pressed)
        self.update_display()

    def add_check_item(self, text, value, checked: bool = False):
        """Append a checkable item labelled ``text`` carrying ``value``."""
        it = QStandardItem(text)
        it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        it.setData(value, Qt.UserRole)
        it.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self.model().appendRow(it)

    def selected_values(self) -> list:
        """Return the user values of all currently checked items."""
        model = self.model()
        return [
            model.item(r).data(Qt.UserRole)
            for r in range(model.rowCount())
            if model.item(r).checkState() == Qt.Checked
        ]

    def set_checked_values(self, values):
        """Check exactly the items whose value is in ``values``; uncheck the rest."""
        want = set(values or [])
        model = self.model()
        for r in range(model.rowCount()):
            it = model.item(r)
            it.setCheckState(
                Qt.Checked if it.data(Qt.UserRole) in want else Qt.Unchecked
            )
        self.update_display()

    def update_display(self):
        """Refresh the line edit to summarize the checked items."""
        model = self.model()
        texts = [
            model.item(r).text()
            for r in range(model.rowCount())
            if model.item(r).checkState() == Qt.Checked
        ]
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        if not texts:
            self.lineEdit().setText(self._placeholder)
        else:
            shown = ", ".join(texts[:3]) + (
                f" (+{len(texts) - 3})" if len(texts) > 3 else ""
            )
            self.lineEdit().setText(shown)
        self.setCursor(Qt.PointingHandCursor)

    def _on_item_pressed(self, index):
        """Toggle an item's check state when the user clicks it."""
        item = self.model().itemFromIndex(index)
        item.setCheckState(
            Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
        )
        self.update_display()


def set_combo_checks(
    combo: QComboBox, values: list[int | str], *, update_api=None
):
    """Set check states in a multi-select combo box from a value list."""
    combo.set_checked_values(values)
