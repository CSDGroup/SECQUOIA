"""Choosing which position to preload, and handing it over on a switch."""

from __future__ import annotations

import concurrent.futures
from types import SimpleNamespace

import pandas as pd
import pytest
from qtpy.QtWidgets import QMessageBox, QWidget

from SECQUOIA.gui import loading_pipeline, position_preload
from SECQUOIA.gui.position_preload import PositionPreloader

GB = 1024**3

TRACKED = {1: 3, 2: 0, 3: 2, 4: 1, 5: 4}


class FakePrefetch:

    def __init__(self):
        self.started: list[int] = []
        self.discarded = 0
        self._pending = None

    def start(self, pnum, folder):
        self.started.append(pnum)
        future = concurrent.futures.Future()
        gate = SimpleNamespace(paused=False, progress=0.0, last_retry=0.0)
        self._pending = (pnum, future, gate)
        return "full"

    def pending(self):
        return self._pending

    def take(self, pnum):
        future = self._pending[1]
        self._pending = None
        return future

    def _discard(self):
        if self._pending is not None:
            self.discarded += 1
        self._pending = None

    def close(self, wait=True):
        self._discard()


def make_window(current=1):
    rows = [
        {"Position": p, "Identification": f"{p}_{i}"}
        for p, n in TRACKED.items()
        for i in range(n)
    ]
    return SimpleNamespace(
        position_folders=[f"/exp/exp_p{p:04d}" for p in TRACKED],
        current_position_number=current,
        time_min_selected=1,
        time_max_selected=2,
        track_df=pd.DataFrame(rows),
    )


@pytest.fixture
def preloader(qapp):
    window = make_window()
    preloader = PositionPreloader(window)
    preloader._prefetch = FakePrefetch()
    preloader.enabled = True
    yield preloader
    preloader.close()


def switch_to(preloader, pnum):
    """What a position switch does: hand over, then plan the next one."""
    preloader._mw.current_position_number = pnum
    future = preloader.take(pnum)
    preloader._start()
    return future


class TestChoosingTheNextPosition:
    def test_skips_positions_without_tracking(self, preloader):
        preloader._start()

        assert preloader.target == 3

    def test_the_users_list_comes_first(self, preloader):
        preloader.queue = [5, 4]

        preloader._start()

        assert preloader.target == 5

    def test_follows_the_direction_of_the_last_move(self, preloader):
        switch_to(preloader, 4)
        switch_to(preloader, 3)

        assert preloader.target == 1

    def test_wraps_around_at_the_end(self, preloader):
        switch_to(preloader, 5)

        assert preloader.target == 1

    def test_without_tracking_data_takes_the_following_position(
        self, preloader
    ):
        preloader._mw.track_df = None

        preloader._start()

        assert preloader.target == 2

    def test_a_running_preload_of_the_same_position_is_kept(self, preloader):
        preloader._start()
        preloader._start()

        assert preloader._prefetch.started == [3]


class TestHandover:
    def test_switching_to_the_preloaded_position_hands_it_over(
        self, preloader
    ):
        preloader._start()

        future = switch_to(preloader, 3)

        assert future is not None
        assert preloader.target == 4

    def test_jumping_elsewhere_drops_an_automatic_preload(self, preloader):
        preloader._start()

        future = switch_to(preloader, 5)

        assert future is None
        assert preloader._prefetch.discarded == 1

    def test_jumping_elsewhere_keeps_a_preload_from_the_list(self, preloader):
        preloader.queue = [4]
        preloader._start()

        future = switch_to(preloader, 5)

        assert future is None
        assert preloader._prefetch.discarded == 0
        assert preloader.target == 4

    def test_arriving_removes_the_position_from_the_list(self, preloader):
        preloader.queue = [3, 5]
        preloader._start()

        switch_to(preloader, 3)

        assert preloader.queue == [5]
        assert preloader.target == 5


class TestUserActions:
    def test_preload_next_moves_a_position_to_the_front(self, preloader):
        preloader.queue = [4, 5]

        preloader.preload_next(5)

        assert preloader.queue == [5, 4]

    def test_adding_keeps_a_running_automatic_preload(self, preloader):
        preloader._start()  # p0003 chosen automatically

        preloader.add(5)
        preloader._start()

        assert preloader.target == 3
        assert preloader.queue == [3, 5]
        assert preloader._prefetch.discarded == 0

    def test_adding_keeps_a_ready_preload(self, preloader):
        preloader._start()
        preloader._prefetch.pending()[1].set_result(SimpleNamespace())

        preloader.add(5)
        preloader._start()

        assert preloader.state() == "ready"
        assert preloader._prefetch.started == [3]

    def test_adding_behind_a_listed_preload_keeps_it(self, preloader):
        preloader.queue = [4]
        preloader._start()

        preloader.add(5)
        preloader._start()

        assert preloader.target == 4
        assert preloader.queue == [4, 5]

    def test_turning_off_drops_the_preload(self, preloader):
        preloader._start()

        preloader.set_enabled(False)

        assert preloader.target is None
        assert preloader.state() == "off"

    def test_while_off_nothing_is_scheduled(self, preloader):
        preloader.enabled = False

        preloader.schedule(0)

        assert not preloader._timer.isActive()


class TestState:
    def test_loading_then_ready(self, preloader):
        preloader._start()
        assert preloader.state() == "loading"

        preloader._prefetch.pending()[1].set_result(SimpleNamespace())

        assert preloader.state() == "ready"

    def test_paused_while_ram_is_low(self, preloader):
        preloader._start()
        preloader._prefetch.pending()[2].paused = True

        assert preloader.state() == "paused"

    def test_refused_when_resources_are_short(self, preloader):
        preloader._prefetch.start = lambda pnum, folder: None

        preloader._start()

        assert preloader.state() == "refused"
        assert preloader.target is None


class TestPreloadQuestion:
    """The question asked once an experiment has loaded."""

    @pytest.fixture
    def window(self, qtbot, monkeypatch):
        window = QWidget()
        qtbot.addWidget(window)
        for name, value in vars(make_window()).items():
            setattr(window, name, value)
        window.isVisible = lambda: True
        preloader = PositionPreloader(window)
        preloader._prefetch = FakePrefetch()
        window.position_preloader = preloader
        monkeypatch.setattr(position_preload, "_prefetch_plan", lambda _w: 1)
        monkeypatch.setattr(
            position_preload, "_prefetch_needs", lambda _w: (GB, GB)
        )
        monkeypatch.setattr(
            position_preload, "_free_resources", lambda: (8 * GB, 50 * GB)
        )
        yield window
        preloader.close()

    def answer(self, monkeypatch, label_start):
        """Answer the question with the button whose label starts so."""
        asked = []

        def click(box):
            asked.append((box.text(), [b.text() for b in box.buttons()]))
            next(
                b for b in box.buttons() if b.text().startswith(label_start)
            ).click()

        monkeypatch.setattr(QMessageBox, "exec_", click)
        return asked

    def test_offers_the_next_tracked_position(self, window, monkeypatch):
        asked = self.answer(monkeypatch, "No")

        position_preload.ask_to_preload(window)

        text, buttons = asked[0]
        assert "p0003" in text
        assert set(buttons) == {"No", "Select positions", "Preload p0003"}

    def test_preload_turns_it_on(self, window, monkeypatch):
        self.answer(monkeypatch, "Preload")

        position_preload.ask_to_preload(window)

        assert window.position_preloader.enabled
        assert window.position_preloader._timer.isActive()

    def test_no_leaves_it_off(self, window, monkeypatch):
        self.answer(monkeypatch, "No")

        position_preload.ask_to_preload(window)

        assert not window.position_preloader.enabled

    def test_select_positions_waits_for_the_users_choice(
        self, window, monkeypatch
    ):
        opened = []
        monkeypatch.setattr(
            position_preload,
            "open_position_window",
            lambda w: opened.append(w),
        )
        self.answer(monkeypatch, "Select")

        position_preload.ask_to_preload(window)

        assert opened == [window]
        assert window.position_preloader.enabled
        assert not window.position_preloader._timer.isActive()

    def test_not_asked_without_enough_ram_or_disk(self, window, monkeypatch):
        monkeypatch.setattr(
            position_preload, "_prefetch_plan", lambda _w: None
        )
        asked = self.answer(monkeypatch, "No")

        position_preload.ask_to_preload(window)

        assert asked == []

    def test_waits_until_the_loading_window_is_closed(
        self, window, monkeypatch, qtbot
    ):
        loading_open = [True]
        monkeypatch.setattr(
            position_preload, "_loading_window_open", lambda: loading_open[0]
        )
        asked = self.answer(monkeypatch, "No")

        position_preload.ask_to_preload(window)
        assert asked == []

        loading_open[0] = False
        qtbot.waitUntil(lambda: len(asked) == 1, timeout=3000)

    def test_a_newer_load_cancels_the_waiting_question(
        self, window, monkeypatch, qtbot
    ):
        monkeypatch.setattr(
            position_preload, "_loading_window_open", lambda: True
        )
        asked = self.answer(monkeypatch, "No")
        position_preload.ask_to_preload(window)

        window.position_preloader = PositionPreloader(window)
        monkeypatch.setattr(
            position_preload, "_loading_window_open", lambda: False
        )
        qtbot.wait(700)

        assert asked == []
        window.position_preloader.close()

    def test_a_modal_window_counts_as_a_loading_window(self, qtbot):
        loading = QWidget()
        qtbot.addWidget(loading)
        loading.setWindowModality(position_preload.Qt.WindowModal)
        loading.show()
        assert position_preload._loading_window_open()

        loading.close()
        assert not position_preload._loading_window_open()

    def test_not_asked_while_the_window_is_hidden(self, window, monkeypatch):
        """A window that is not on screen (as in tests) never blocks on it."""
        window.isVisible = lambda: False
        asked = self.answer(monkeypatch, "No")

        position_preload.ask_to_preload(window)

        assert asked == []

    def test_not_asked_with_nothing_else_to_preload(self, window, monkeypatch):
        window.position_folders = window.position_folders[:1]
        asked = self.answer(monkeypatch, "No")

        position_preload.ask_to_preload(window)

        assert asked == []


class TestAdoptingAPreload:
    def test_a_preload_with_unreadable_frames_is_not_used(self, monkeypatch):
        cleaned = []
        monkeypatch.setattr(
            loading_pipeline,
            "cleanup_memmaps",
            lambda staging, **_: cleaned.append(staging),
        )
        staging = SimpleNamespace(fl_failed_frames=2, basic_done=False)
        future = concurrent.futures.Future()
        future.set_result(staging)

        adopted = loading_pipeline._adopt_prefetched_inputs(
            SimpleNamespace(), future, 3
        )

        assert adopted is None
        assert cleaned == [staging]

    def test_a_stopped_preload_falls_back_to_a_normal_load(self):
        future = concurrent.futures.Future()
        future.set_exception(loading_pipeline.LoadCancelled())

        adopted = loading_pipeline._adopt_prefetched_inputs(
            SimpleNamespace(), future, 3
        )

        assert adopted is None

    def test_a_preload_with_unreadable_masks_is_not_used(self, monkeypatch):
        monkeypatch.setattr(
            loading_pipeline, "cleanup_memmaps", lambda staging, **_: None
        )
        staging = SimpleNamespace(mask_failed_frames=1, basic_done=False)
        future = concurrent.futures.Future()
        future.set_result(staging)

        adopted = loading_pipeline._adopt_prefetched_inputs(
            SimpleNamespace(), future, 3
        )

        assert adopted is None


class TestStatusBar:
    """The one line status in the menu bar's right corner."""

    @pytest.fixture
    def bar(self, preloader, qtbot, monkeypatch):
        monkeypatch.setattr(
            position_preload, "_free_resources", lambda: (11 * GB, 376 * GB)
        )
        preloader._mw.position_preloader = preloader
        bar = position_preload.PreloadStatusBar(preloader._mw)
        qtbot.addWidget(bar)
        return bar

    def gate(self, preloader):
        return preloader._prefetch.pending()[2]

    def test_off(self, bar, preloader):
        preloader.enabled = False

        assert bar.status() == ("Preload off", 0, None)

    def test_loading_shows_position_and_percent(self, bar, preloader):
        preloader._start()
        self.gate(preloader).progress = 0.4

        assert bar.status() == ("p0003 · 40%", 40, "loading")

    def test_ready(self, bar, preloader):
        preloader._start()
        preloader._prefetch.pending()[1].set_result(SimpleNamespace())

        assert bar.status() == ("p0003 · ready", 100, "ready")

    def test_paused_on_low_ram(self, bar, preloader):
        preloader._start()
        self.gate(preloader).paused = True

        assert bar.status()[0] == "Paused · low RAM"

    def test_waiting_for_the_server(self, bar, preloader):
        preloader._start()
        self.gate(preloader).last_retry = position_preload.time.monotonic()

        assert bar.status()[0] == "Waiting for server"

    def test_refused_for_low_resources(self, bar, preloader):
        preloader._prefetch.start = lambda pnum, folder: None
        preloader._start()

        assert bar.status()[0] == "Preload: low RAM/disk"

    def test_tooltip_shows_free_ram_and_disk(self, bar, preloader):
        bar.refresh()

        assert "RAM 11.0 GB · Disk 376.0 GB free" in bar.toolTip()

    def test_left_click_opens_the_position_window(
        self, bar, preloader, qtbot, monkeypatch
    ):
        opened = []
        monkeypatch.setattr(
            position_preload, "open_position_window", opened.append
        )

        qtbot.mouseClick(bar, position_preload.Qt.LeftButton)

        assert opened == [preloader._mw]

    def test_a_low_disk_stop_says_so(self, bar, preloader):
        preloader._start()
        self.gate(preloader).stop_reason = "disk"
        preloader._prefetch.pending()[1].set_exception(
            loading_pipeline.LoadCancelled()
        )

        assert bar.status()[0] == "Stopped · low disk"


def test_interactive_preloads_use_two_reader_threads(qapp, monkeypatch):
    """Only two threads, so curating the current position stays smooth."""
    seen = []
    monkeypatch.setattr(
        loading_pipeline,
        "_load_position_inputs",
        lambda staging, *_args: seen.append(staging),
    )
    monkeypatch.setattr(loading_pipeline, "_prefetch_plan", lambda _w: "full")
    window = make_window()
    window.n_channels = 0
    preloader = PositionPreloader(window)
    preloader.enabled = True

    preloader._start()
    preloader._prefetch.pending()[1].result(timeout=5)
    preloader.close()

    assert seen[0].fl_max_workers == 2
    assert seen[0].basic_max_workers == 2


class TestStoppedWhenSettingsMayChange:
    """A preload made with old load settings must never be used."""

    @staticmethod
    def running_preloader():
        closed = []
        return SimpleNamespace(close=lambda: closed.append(True)), closed

    def test_opening_the_load_window_stops_it(self, dialog_main_window, qtbot):
        from SECQUOIA.gui.loading.load_data.load_data_dialog import (
            load_data_window,
        )

        preloader, closed = self.running_preloader()
        dialog_main_window.position_preloader = preloader
        dialog_main_window.mask_no_window = None

        load_data_window(dialog_main_window)
        qtbot.addWidget(dialog_main_window.mask_no_window)

        assert closed == [True]
        assert dialog_main_window.position_preloader is None

    @pytest.fixture
    def curation_window(self, qtbot, monkeypatch):
        from SECQUOIA.gui.main_window import window_lifecycle

        monkeypatch.setattr(
            window_lifecycle, "open_experiment_loader_window", lambda _w: None
        )
        window = QWidget()
        qtbot.addWidget(window)
        window.folder_list = ["exp_p0001"]
        window._clear_curation_ui = lambda: None
        window.position_preloader, window.closed = self.running_preloader()
        return window

    def answer(self, monkeypatch, label_start):
        monkeypatch.setattr(
            QMessageBox,
            "exec_",
            lambda box: next(
                b for b in box.buttons() if b.text().startswith(label_start)
            ).click(),
        )

    def test_switching_to_cytometric_analysis_stops_it(
        self, curation_window, monkeypatch
    ):
        from SECQUOIA.gui.main_window.window_lifecycle import WindowLifecycle

        self.answer(monkeypatch, "Switch without")

        WindowLifecycle.open_cytometric_analysis(curation_window)

        assert curation_window.closed == [True]
        assert curation_window.position_preloader is None

    def test_cancelling_the_switch_keeps_it(
        self, curation_window, monkeypatch
    ):
        from SECQUOIA.gui.main_window.window_lifecycle import WindowLifecycle

        self.answer(monkeypatch, "Cancel")

        WindowLifecycle.open_cytometric_analysis(curation_window)

        assert curation_window.closed == []
