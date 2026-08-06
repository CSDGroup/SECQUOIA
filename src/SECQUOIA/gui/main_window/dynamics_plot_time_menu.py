"""'Change Time Plotting' context menu wiring for dynamics plot widgets."""

from __future__ import annotations

import pyqtgraph as pg
from qtpy.QtWidgets import QAction, QActionGroup, QMenu

from SECQUOIA.utils.plotting import (
    TIME_MODE_CALC,
    TIME_MODE_REAL,
    TIME_MODE_T,
    update_plot,
)

__all__ = ["install_time_menu"]


def install_time_menu(plot_item: pg.PlotItem, main_window) -> None:
    """Insert 'Change Time Plotting' submenu before 'Export' in the ViewBox menu."""
    try:
        vb_menu: QMenu = plot_item.vb.menu
    except (AttributeError, RuntimeError):
        return
    if vb_menu is None:
        return

    for a in vb_menu.actions():
        sub = a.menu() if hasattr(a, "menu") else None
        if sub is not None and sub.title() == "Change Time Plotting":
            _sync_time_menu_checks(
                sub, getattr(main_window, "_time_mode", TIME_MODE_T)
            )
            return
        if isinstance(a, QMenu) and a.title() == "Change Time Plotting":
            _sync_time_menu_checks(
                a, getattr(main_window, "_time_mode", TIME_MODE_T)
            )
            return

    export_action = None
    for a in vb_menu.actions():
        if hasattr(a, "text") and a.text() == "Export":
            export_action = a
            break

    group = QActionGroup(vb_menu)
    group.setExclusive(True)

    sub = QMenu("Change Time Plotting", vb_menu)

    # These labels are matched by text in _sync_time_menu_checks — keep in sync.
    act_t = QAction("Time point", sub, checkable=True)  # TIME_MODE_T
    act_c = QAction("Time", sub, checkable=True)  # TIME_MODE_CALC
    act_r = QAction("RealTime", sub, checkable=True)  # TIME_MODE_REAL

    for act in (act_t, act_c, act_r):
        act.setActionGroup(group)
        sub.addAction(act)

    def _set_mode(mode):
        """Switch the active time mode and redraw the plots and lineage tree."""
        main_window._time_mode = mode
        update_plot(main_window)
        from SECQUOIA.gui.lineage_tree.lineage_tree import lineage_tree

        lineage_tree(main_window)

    act_t.triggered.connect(lambda: _set_mode(TIME_MODE_T))
    act_c.triggered.connect(lambda: _set_mode(TIME_MODE_CALC))
    act_r.triggered.connect(lambda: _set_mode(TIME_MODE_REAL))

    _sync_time_menu_checks(
        sub, getattr(main_window, "_time_mode", TIME_MODE_T)
    )

    if export_action is not None:
        vb_menu.insertMenu(export_action, sub)
    else:
        vb_menu.addMenu(sub)


def _sync_time_menu_checks(submenu: QMenu, mode: str) -> None:
    """Update the checked action in the time mode submenu to match the active mode."""
    for act in submenu.actions():
        txt = act.text()
        if txt == "Time point":
            act.setChecked(mode == TIME_MODE_T)
        elif txt == "RealTime":
            act.setChecked(mode == TIME_MODE_REAL)
        elif txt == "Time":
            act.setChecked(mode == TIME_MODE_CALC)
