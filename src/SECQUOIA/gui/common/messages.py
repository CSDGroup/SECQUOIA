"""Standalone message boxes shown outside any particular dialog.

These live in their own module so that dialogs can warn the user without
importing each other.
"""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QCursor, QGuiApplication
from qtpy.QtWidgets import QMessageBox, QWidget

__all__ = ["show_folder_warning"]

_WARNING_WIDTH_FRACTION = 0.35
_WARNING_HEIGHT_FRACTION = 0.22


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

    screen = (
        QGuiApplication.screenAt(QCursor.pos())
        or QGuiApplication.primaryScreen()
    )
    available = screen.availableGeometry()
    msg.resize(
        int(available.width() * _WARNING_WIDTH_FRACTION),
        int(available.height() * _WARNING_HEIGHT_FRACTION),
    )

    msg.setStyleSheet("QLabel{min-width:420px;}")

    msg.exec_()
