"""Preload controls in the "Selection of a specific position" window."""

from __future__ import annotations

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget
from test_position_preload import GB, FakePrefetch, make_window

from SECQUOIA.gui import position_navigation
from SECQUOIA.gui.position_navigation import PositionSelectDialog
from SECQUOIA.gui.position_preload import PositionPreloader


@pytest.fixture
def dialog(qtbot, monkeypatch):
    monkeypatch.setattr(
        position_navigation, "_free_resources", lambda: (11 * GB, 376 * GB)
    )
    window = QWidget()
    qtbot.addWidget(window)
    for name, value in vars(make_window()).items():
        setattr(window, name, value)
    window.folder_list = [f"{p:04d}_exp_p{p:04d}" for p in range(1, 6)]
    window.tracking_format = "tTt"
    window.position_comment_map = {3: "Drift after t400"}
    preloader = PositionPreloader(window)
    preloader._prefetch = FakePrefetch()
    preloader.enabled = True
    window.position_preloader = preloader
    dialog = PositionSelectDialog(window)
    qtbot.addWidget(dialog)
    yield dialog
    preloader.close()


def row(dialog, pnum):
    for i in range(dialog.list_widget.topLevelItemCount()):
        item = dialog.list_widget.topLevelItem(i)
        if item.data(0, Qt.UserRole) == pnum:
            return item
    raise LookupError(pnum)


def menu_for(dialog, monkeypatch, pnum):
    shown = {}

    def capture(menu, *_args):
        shown.update({a.text(): a for a in menu.actions() if a.text()})

    monkeypatch.setattr(position_navigation.QMenu, "exec_", capture)
    rect = dialog.list_widget.visualItemRect(row(dialog, pnum))
    dialog._show_menu(rect.center())
    return shown


class TestFilters:
    def test_positions_without_tracking_are_hidden(self, dialog):
        assert row(dialog, 2).isHidden()
        assert not row(dialog, 3).isHidden()

    def test_unticking_shows_them_again(self, dialog):
        dialog.tracked_only_cb.setChecked(False)

        assert not row(dialog, 2).isHidden()

    def test_the_filter_also_searches_comments(self, dialog):
        dialog.search_edit.setText("drift")

        assert not row(dialog, 3).isHidden()
        assert row(dialog, 4).isHidden()


class TestPreloadColumn:
    def test_the_preloading_position_gets_a_dot(self, dialog):
        dialog._preloader._start()
        dialog._refresh_preload()

        item = row(dialog, 3)
        assert not item.icon(2).isNull()
        assert item.toolTip(2) == "Preloading"

    def test_waiting_list_entries_show_their_place(self, dialog):
        dialog._preloader.queue = [4, 5]
        dialog._preloader._start()
        dialog._refresh_preload()

        assert row(dialog, 5).toolTip(2) == "Waiting, 2 in preload list"

    def test_no_dots_while_preloading_is_off(self, dialog):
        dialog._preloader.queue = [4]
        dialog.preload_cb.setChecked(False)

        assert not dialog._preloader.enabled
        assert row(dialog, 4).icon(2).isNull()

    def test_shows_free_ram_and_disk(self, dialog):
        text = dialog.resources_lbl.text()

        assert "Free RAM 11.0 GB (keeps 3 GB)" in text
        assert "Free scratch disk 376 GB" in text


class TestRightClickMenu:
    def test_preload_next_puts_it_first(self, dialog, monkeypatch):
        dialog._preloader.queue = [4]

        menu_for(dialog, monkeypatch, 5)["Preload next"].trigger()

        assert dialog._preloader.queue == [5, 4]

    def test_add_to_list(self, dialog, monkeypatch):
        menu_for(dialog, monkeypatch, 4)["Add to preload list"].trigger()

        assert dialog._preloader.queue == [4]

    def test_a_listed_position_can_be_removed_but_not_added(
        self, dialog, monkeypatch
    ):
        dialog._preloader.queue = [4]

        menu = menu_for(dialog, monkeypatch, 4)

        assert not menu["Add to preload list"].isEnabled()
        assert menu["Remove from list"].isEnabled()

    def test_the_current_position_cannot_be_preloaded(
        self, dialog, monkeypatch
    ):
        menu = menu_for(dialog, monkeypatch, 1)

        assert "Current position" in menu
        assert not menu["Preload next"].isEnabled()

    def test_a_position_without_tracking_cannot_be_preloaded(
        self, dialog, monkeypatch
    ):
        dialog.tracked_only_cb.setChecked(False)

        menu = menu_for(dialog, monkeypatch, 2)

        assert "No tracking data" in menu
        assert not menu["Add to preload list"].isEnabled()

    def test_clear_empties_the_list(self, dialog, monkeypatch):
        dialog._preloader.queue = [4, 5]

        menu_for(dialog, monkeypatch, 3)["Clear preload list"].trigger()

        assert dialog._preloader.queue == []
