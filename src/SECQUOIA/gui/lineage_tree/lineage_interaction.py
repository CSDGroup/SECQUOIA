"""Mouse and context-menu wiring for the lineage tree plot."""

from __future__ import annotations

import contextlib

import pyqtgraph as pg
from qtpy import QtCore
from qtpy.QtCore import Qt

from SECQUOIA.gui.lineage_tree.lineage_render import QT_DRAW_ERRORS

__all__ = [
    "ShiftPanGuard",
    "install_shift_pan",
    "strip_default_menu_actions",
]

# pyqtgraph's own entries that duplicate our toolbar and are removed.
_UNWANTED_ACTIONS = ("View All", "Mouse Mode")


class ShiftPanGuard(QtCore.QObject):
    """Restores rectangle-zoom mode when Shift is released or focus is lost."""

    _RESET_EVENTS = (
        QtCore.QEvent.WindowDeactivate,
        QtCore.QEvent.FocusOut,
        QtCore.QEvent.Leave,
        QtCore.QEvent.MouseButtonRelease,
    )

    def __init__(self, view_box, parent=None) -> None:
        """Watch ``view_box`` for the end of a Shift-pan gesture."""
        super().__init__(parent)
        self._vb = view_box

    def restore(self) -> None:
        """Return the view box to rectangle-zoom mode if it is panning."""
        if getattr(self._vb, "_tmp_shift_pan", False):
            self._vb._tmp_shift_pan = False
            self._vb.setMouseMode(pg.ViewBox.RectMode)

    def eventFilter(self, obj, event) -> bool:
        """Restore rectangle zoom when Shift is released or focus is lost."""
        event_type = event.type()
        if event_type == QtCore.QEvent.KeyRelease:
            if getattr(self._vb, "_tmp_shift_pan", False):
                key = getattr(event, "key", lambda: None)()
                if key == Qt.Key_Shift:
                    self.restore()
        elif event_type in self._RESET_EVENTS:
            self.restore()
        return False


def install_shift_pan(plot: pg.PlotItem) -> ShiftPanGuard:
    """Make Shift+left-drag pan while plain left-drag keeps rectangle zoom."""
    view_box = plot.vb
    view_box.setMouseMode(pg.ViewBox.RectMode)
    view_box._tmp_shift_pan = False

    guard = ShiftPanGuard(view_box, plot.scene())
    original_press = view_box.mousePressEvent
    original_release = view_box.mouseReleaseEvent

    def press(event):
        """Switch to pan mode for the duration of a Shift-left-drag."""
        with contextlib.suppress(*QT_DRAW_ERRORS, ValueError):
            modifiers = (
                event.modifiers()
                if hasattr(event, "modifiers")
                else Qt.NoModifier
            )
            button = event.button() if hasattr(event, "button") else None
            if (modifiers & Qt.ShiftModifier) and button == Qt.LeftButton:
                view_box._tmp_shift_pan = True
                view_box.setMouseMode(pg.ViewBox.PanMode)
        return original_press(event)

    def release(event):
        """Restore rectangle zoom once the mouse button is released."""
        with contextlib.suppress(*QT_DRAW_ERRORS, ValueError):
            guard.restore()
        return original_release(event)

    view_box.mousePressEvent = press
    view_box.mouseReleaseEvent = release
    plot.scene().installEventFilter(guard)
    view_box._shift_guard = guard
    return guard


def _normalize_menu_text(text: str) -> str:
    """Normalize Qt menu text so action matching survives accelerators."""
    return text.replace("&", "").replace("…", "...").strip().lower()


def _remove_plot_options(menu) -> None:
    """Drop pyqtgraph's default 'Plot Options' entry from ``menu``."""
    if not menu:
        return
    for action in list(menu.actions()):
        if _normalize_menu_text(action.text()).startswith("plot options"):
            menu.removeAction(action)


def strip_default_menu_actions(plot: pg.PlotItem, main_window=None) -> None:
    """Install the time submenu and remove pyqtgraph's default entries."""
    plot.setMenuEnabled(True)
    with contextlib.suppress(*QT_DRAW_ERRORS):
        plot.vb.setMenuEnabled(True)
    plot.getMenu()

    # circular-import: gui.main_window eagerly imports MainWindow, which
    # pulls in this module transitively.
    from SECQUOIA.gui.main_window.dynamics_plot_time_menu import (
        install_time_menu,
    )

    install_time_menu(plot, main_window)

    view_menu = getattr(plot.vb, "menu", None)
    if view_menu is not None:
        for action in list(view_menu.actions()):
            if action.text() in _UNWANTED_ACTIONS:
                view_menu.removeAction(action)

    _remove_plot_options(plot.getMenu())
    _remove_plot_options(view_menu)

    plot_menu = plot.getMenu()
    if plot_menu is not None:
        plot_menu.aboutToShow.connect(
            lambda: _remove_plot_options(plot.getMenu())
        )
    if view_menu is not None:
        view_menu.aboutToShow.connect(
            lambda: _remove_plot_options(getattr(plot.vb, "menu", None))
        )
