"""Opens, updates and closes the Cell Inspector window."""

from __future__ import annotations

import logging

LOG = logging.getLogger(__name__)


def open_cell_inspector(main_window):
    """Show the Cell Inspector, or close it if it is the active window.

    A second press only closes it when it is in front; if it is behind
    the main window the same key raises it instead. Closing hides the
    window, so the channel, mask and level settings survive.
    """
    if _put_away_if_in_front(main_window):
        return getattr(main_window, "cell_inspector", None)

    controller = getattr(main_window, "cell_inspector", None)
    if controller is None:
        from SECQUOIA.gui.cell_inspector.controller import (
            CellInspectorController,
        )
        from SECQUOIA.gui.cell_inspector.panel import CellInspectorPanel

        try:
            panel = CellInspectorPanel(main_window, main_window)
            controller = CellInspectorController(main_window, panel)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            LOG.warning("cell inspector unavailable: %s", exc)
            return None

        main_window.cell_inspector_panel = panel
        main_window.cell_inspector = controller

    try:
        controller.show()
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        LOG.warning("cell inspector could not be shown: %s", exc)
    return controller


def _put_away_if_in_front(main_window) -> bool:
    """Close the window if it is open and active."""
    panel = getattr(main_window, "cell_inspector_panel", None)
    if panel is None:
        return False
    try:
        if not panel.isVisible() or not panel.isActiveWindow():
            return False
        panel.close()
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        LOG.warning("cell inspector: window would not close: %s", exc)
        return False
    return True


def notify_cell_inspector(main_window, method: str) -> None:
    """Call a method on the inspector, if there is one."""
    controller = getattr(main_window, "cell_inspector", None)
    if controller is None:
        return
    handler = getattr(controller, method, None)
    if handler is None:
        return
    try:
        handler()
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        LOG.warning("cell inspector: %s failed: %s", method, exc)


def detach_cell_inspector(main_window) -> None:
    """Shut the inspector down and drop it from the main window."""
    notify_cell_inspector(main_window, "shutdown")
    panel = getattr(main_window, "cell_inspector_panel", None)
    if panel is not None:
        try:
            panel.close()
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            LOG.warning("cell inspector: window would not close: %s", exc)
    main_window.cell_inspector = None
    main_window.cell_inspector_panel = None
