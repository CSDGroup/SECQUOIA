"""The Cell Inspector window and the views inside it."""

from __future__ import annotations

import logging

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QMenu, QSplitter, QVBoxLayout, QWidget

from SECQUOIA.gui.cell_inspector.pane import InspectorPane

LOG = logging.getLogger(__name__)

DEFAULT_WINDOW_WIDTH = 460
DEFAULT_WINDOW_HEIGHT = 480
EXTRA_PANE_WIDTH = 320
EXTRA_PANE_HEIGHT = 280
MAX_PANES = 8

LEFT = "left"
RIGHT = "right"
ABOVE = "above"
BELOW = "below"

ADD_SIDES = (
    ("Left", LEFT),
    ("Right", RIGHT),
    ("Above", ABOVE),
    ("Below", BELOW),
)

ADD_MENU_TEXT = "Add viewer"
REMOVE_PANE_TEXT = "Remove this viewer"

_HORIZONTAL_SIDES = (LEFT, RIGHT)
_BEFORE_SIDES = (LEFT, ABOVE)


class CellInspectorPanel(QWidget):
    """The Cell Inspector window."""

    pane_added = Signal(object)
    pane_removed = Signal(object)
    closed = Signal()

    def __init__(self, widget_source, parent: QWidget | None = None):
        """`widget_source` is the main window, handed to views to build widgets."""
        super().__init__(parent)
        self.setWindowFlag(Qt.Window)
        self.setWindowTitle("Cell Inspector")
        self.setObjectName("cell_inspector_window")
        self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)

        self._widget_source = widget_source

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(self.splitter, 1)

        self.panes: list[InspectorPane] = []
        self.add_pane()

    def _new_pane(self) -> InspectorPane:
        """Make right-click menu."""
        pane = InspectorPane(self._widget_source)
        pane.context_menu_requested.connect(
            lambda position, p=pane: self._show_pane_menu(p, position)
        )
        self.panes.append(pane)
        return pane

    def add_pane(
        self, relative_to: InspectorPane | None = None, side: str = RIGHT
    ) -> InspectorPane | None:
        """Add a view on one side."""
        if len(self.panes) >= MAX_PANES:
            LOG.info("cell inspector: pane limit of %d reached", MAX_PANES)
            return None

        if relative_to is None or not self.panes:
            pane = self._new_pane()
            self.splitter.addWidget(pane)
        else:
            pane = self._insert_beside(relative_to, side)
            if pane is None:
                return None

        self._grow_for(side)
        self.pane_added.emit(pane)
        return pane

    def _insert_beside(
        self, neighbour: InspectorPane, side: str
    ) -> InspectorPane | None:
        """Put a new view next to another one."""
        parent = neighbour.parentWidget()
        if not isinstance(parent, QSplitter):
            LOG.warning("cell inspector: pane is not in a splitter")
            return None

        wanted = Qt.Horizontal if side in _HORIZONTAL_SIDES else Qt.Vertical
        before = side in _BEFORE_SIDES
        index = parent.indexOf(neighbour)
        pane = self._new_pane()

        if parent.orientation() == wanted or parent.count() == 1:
            parent.setOrientation(wanted)
            parent.insertWidget(index if before else index + 1, pane)
            return pane

        sizes = parent.sizes()
        nested = QSplitter(wanted)
        nested.setChildrenCollapsible(False)

        neighbour.setParent(None)
        if before:
            nested.addWidget(pane)
            nested.addWidget(neighbour)
        else:
            nested.addWidget(neighbour)
            nested.addWidget(pane)

        parent.insertWidget(index, nested)
        parent.setSizes(sizes)
        return pane

    def _grow_for(self, side: str) -> None:
        """Grow the window so a new view does not shrink the others."""
        if len(self.panes) <= 1:
            return
        if side in _HORIZONTAL_SIDES:
            self.resize(self.width() + EXTRA_PANE_WIDTH, self.height())
        else:
            self.resize(self.width(), self.height() + EXTRA_PANE_HEIGHT)

    def remove_pane(self, pane: InspectorPane) -> bool:
        """Remove a view. The last one stays, so the window is never empty."""
        if pane not in self.panes or len(self.panes) <= 1:
            return False

        parent = pane.parentWidget()
        self.pane_removed.emit(pane)
        self.panes.remove(pane)
        pane.setParent(None)
        pane.deleteLater()
        self._collapse(parent)
        return True

    def _collapse(self, splitter) -> None:
        while (
            isinstance(splitter, QSplitter)
            and splitter is not self.splitter
            and splitter.count() == 1
        ):
            grandparent = splitter.parentWidget()
            if not isinstance(grandparent, QSplitter):
                return
            index = grandparent.indexOf(splitter)
            sizes = grandparent.sizes()

            child = splitter.widget(0)
            child.setParent(None)
            splitter.setParent(None)
            splitter.deleteLater()

            grandparent.insertWidget(index, child)
            grandparent.setSizes(sizes)
            splitter = grandparent

    def _show_pane_menu(self, pane: InspectorPane, position) -> None:
        """Show the add and remove menu where the user right-clicked."""
        menu = QMenu(self)

        add_menu = menu.addMenu(ADD_MENU_TEXT)
        add_menu.setEnabled(len(self.panes) < MAX_PANES)
        actions = {}
        for label, side in ADD_SIDES:
            actions[add_menu.addAction(label)] = side

        remove_action = menu.addAction(REMOVE_PANE_TEXT)
        remove_action.setEnabled(len(self.panes) > 1)

        chosen = menu.exec_(position)
        if chosen is None:
            return
        if chosen is remove_action:
            self.remove_pane(pane)
        elif chosen in actions:
            self.add_pane(relative_to=pane, side=actions[chosen])

    def rebuild_all_controls(self) -> None:
        """Rebuild every view's controls after the loaded data changed."""
        for pane in self.panes:
            pane.rebuild_controls()

    def closeEvent(self, event) -> None:
        self.closed.emit()
        super().closeEvent(event)


__all__ = [
    "CellInspectorPanel",
    "InspectorPane",
    "ABOVE",
    "BELOW",
    "LEFT",
    "RIGHT",
]
