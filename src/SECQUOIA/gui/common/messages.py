"""Standalone message boxes shown outside any particular dialog.

These live in their own module so that dialogs can warn the user without
importing each other.
"""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QMessageBox, QWidget

__all__ = ["show_folder_warning"]


def show_folder_warning(parent: QWidget | None = None) -> None:
    """Warn that no tracking data and experiment folder are loaded yet."""
    msg = QMessageBox(parent)
    msg.setIcon(QMessageBox.Warning)
    msg.setWindowTitle("Folder Selection Error")
    msg.setText(
        "Please first load tracking data and select a experiment folder."
    )
    msg.setInformativeText("Go to File → Start a new Project (Ctrl+N).")
    msg.setStandardButtons(QMessageBox.Ok)

    msg.setSizeGripEnabled(True)
    msg.setTextInteractionFlags(Qt.TextSelectableByMouse)

    msg.setMinimumWidth(420)

    msg.exec_()
