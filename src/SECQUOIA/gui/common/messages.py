"""Standalone message boxes shown outside any particular dialog.

These live in their own module so that dialogs can warn the user without
importing each other.
"""

from __future__ import annotations

import sys

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialogButtonBox,
    QMessageBox,
    QProxyStyle,
    QStyle,
    QStyleFactory,
    QWidget,
)

__all__ = [
    "DIALOG_BUTTON_QSS",
    "apply_dialog_platform_style",
    "show_folder_warning",
]

DIALOG_BUTTON_QSS = """
QMessageBox QPushButton {
    min-width: 88px;
    padding: 5px 14px;
    border-radius: 4px;
}
"""

_WIN_BUTTON_LAYOUT = int(
    getattr(QDialogButtonBox.WinLayout, "value", QDialogButtonBox.WinLayout)
)


class _WindowsButtonOrderStyle(QProxyStyle):
    """Fusion, but with the Windows dialog button order."""

    def styleHint(self, hint, option=None, widget=None, returnData=None):
        if hint == QStyle.SH_DialogButtonLayout:
            return _WIN_BUTTON_LAYOUT
        return super().styleHint(hint, option, widget, returnData)


def apply_dialog_platform_style(dlg: QMessageBox) -> None:
    """Give one message box the same look and button order on macOS as on Windows."""
    if sys.platform == "darwin":
        style = _WindowsButtonOrderStyle(QStyleFactory.create("Fusion"))
        style.setParent(dlg)
        dlg.setStyle(style)
        for button in dlg.buttons():
            button.setStyle(style)

    dlg.setStyleSheet(DIALOG_BUTTON_QSS)


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
