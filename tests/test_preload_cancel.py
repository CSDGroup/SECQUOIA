"""Stopping a background position preload."""

from __future__ import annotations

import concurrent.futures
import threading
import time
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import pytest
import tifffile

from SECQUOIA.core.fluorescence_loading import load_fl_channels
from SECQUOIA.core.memmap_store import cleanup_memmaps
from SECQUOIA.core.segmentation.mask_io import load_masks
from SECQUOIA.gui import loading_pipeline
from SECQUOIA.utils.io_reload import (
    LoadCancelled,
    raise_if_cancelled,
    read_with_reload,
)


def _cancelled() -> threading.Event:
    event = threading.Event()
    event.set()
    return event


def _always_fails(path):
    raise OSError("server unreachable")


class TestReadWithReloadCancel:
    def test_a_cancel_during_the_wait_stops_at_once(self):
        """The wait between retries can be 30 s; a cancel must not sit it out."""
        cancel = threading.Event()
        threading.Timer(0.05, cancel.set).start()
        started = time.monotonic()

        with pytest.raises(LoadCancelled):
            read_with_reload(
                _always_fails, "frame.tif", base_delay_s=30.0, cancel=cancel
            )

        assert time.monotonic() - started < 5.0

    def test_without_a_cancel_the_retry_still_succeeds(self):
        attempts = []

        def flaky(path):
            attempts.append(path)
            if len(attempts) < 2:
                raise OSError("not ready")
            return "ok"

        result = read_with_reload(
            flaky, "frame.tif", base_delay_s=0.0, cancel=threading.Event()
        )

        assert result == "ok"

    def test_a_first_read_that_works_ignores_the_cancel(self):
        """Cancelling only cuts short waiting; a good read is still returned."""
        assert (
            read_with_reload(lambda p: "ok", "frame.tif", cancel=_cancelled())
            == "ok"
        )


class TestRaiseIfCancelled:
    def test_no_event_means_never_cancelled(self):
        raise_if_cancelled(None)

    def test_an_unset_event_does_nothing(self):
        raise_if_cancelled(threading.Event())

    def test_a_set_event_stops_the_load(self):
        with pytest.raises(LoadCancelled):
            raise_if_cancelled(_cancelled())


class TestLoadersStop:
    def test_mask_loading_stops_when_cancelled(self, tmp_path):
        position = tmp_path / "exp_p0001"
        masks = tmp_path / "Segmentation" / position.name
        masks.mkdir(parents=True)
        frame = np.array([[0, 1], [2, 0]], dtype=np.uint16)
        for t in (1, 2):
            tifffile.imwrite(masks / f"mask_t{t:05d}.tif", frame)

        with pytest.raises(LoadCancelled):
            load_masks(
                str(tmp_path / "Segmentation"),
                str(position),
                "tif",
                cancel=_cancelled(),
            )

    def test_image_loading_stops_when_cancelled(self, tmp_path):
        position = tmp_path / "exp_p0001"
        position.mkdir()
        for t in (1, 2):
            imageio.imwrite(
                position / f"img_t{t:05d}_w01.png",
                np.full((4, 5), t, dtype=np.uint8),
            )
        staging = SimpleNamespace(
            position_selection=str(position),
            image_format="png",
            current_position_number=1,
            time_min_selected=1,
            time_max_selected=2,
            n_channels=1,
            ids_channels=["w01"],
            FL_identifiers_1="w01",
            _load_cancel=_cancelled(),
        )

        try:
            with pytest.raises(LoadCancelled):
                load_fl_channels(staging)
        finally:
            cleanup_memmaps(staging, remove_viewer_image_layers=False)


GB = 1024**3


class TestPreloadGate:
    """A preload leaves 3 GB RAM to the window and stops on a full disk."""

    @pytest.fixture
    def resources(self, monkeypatch):
        """Free (RAM, disk) the gate sees, as a list of readings in turn."""
        readings: list[tuple[int, int]] = []

        def fake():
            return readings.pop(0) if len(readings) > 1 else readings[0]

        monkeypatch.setattr(loading_pipeline, "_free_resources", fake)
        return readings

    @pytest.fixture
    def gate(self):
        gate = loading_pipeline._PreloadGate()
        gate.pause_poll_s = 0.01
        return gate

    def test_enough_ram_and_disk_lets_the_load_continue(self, gate, resources):
        resources.append((8 * GB, 100 * GB))

        raise_if_cancelled(gate)

        assert not gate.is_set()
        assert not gate.paused

    def test_low_ram_pauses_until_memory_frees_up(self, gate, resources):
        resources.extend([(2 * GB, 100 * GB), (2 * GB, 100 * GB)])
        resources.append((6 * GB, 100 * GB))

        raise_if_cancelled(gate)

        assert resources == [(6 * GB, 100 * GB)]
        assert not gate.is_set()
        assert not gate.paused

    def test_the_paused_state_is_visible_while_waiting(self, gate, resources):
        resources.append((1 * GB, 100 * GB))
        seen = threading.Event()

        def watch():
            while not gate.paused:
                time.sleep(0.005)
            seen.set()
            gate.set()

        threading.Thread(target=watch).start()

        with pytest.raises(LoadCancelled):
            raise_if_cancelled(gate)

        assert seen.is_set()

    def test_a_cancel_ends_a_pause(self, gate, resources):
        resources.append((1 * GB, 100 * GB))
        threading.Timer(0.05, gate.set).start()

        with pytest.raises(LoadCancelled):
            raise_if_cancelled(gate)

    def test_a_nearly_full_disk_stops_the_preload(self, gate, resources):
        resources.append((8 * GB, 1 * GB))

        with pytest.raises(LoadCancelled):
            raise_if_cancelled(gate)

        assert gate.stop_reason == "disk"

    def test_free_space_is_read_at_most_once_a_second(self, gate, monkeypatch):
        calls = []
        monkeypatch.setattr(
            loading_pipeline,
            "_free_resources",
            lambda: calls.append(1) or (8 * GB, 100 * GB),
        )

        for _ in range(50):
            raise_if_cancelled(gate)

        assert len(calls) == 1

    def test_unknown_free_space_does_not_stop_the_load(
        self, gate, monkeypatch
    ):
        def unknown():
            raise OSError("no access")

        monkeypatch.setattr(loading_pipeline, "_free_resources", unknown)

        raise_if_cancelled(gate)

        assert not gate.is_set()


class TestPrefetchPlanReserves:
    """Images and masks go to disk, so RAM only has to stay above 3 GB."""

    def _plan(self, monkeypatch, ram_free, disk_free, disk_need=10 * GB):
        monkeypatch.setattr(
            loading_pipeline, "_prefetch_needs", lambda _mw: (disk_need, 0)
        )
        monkeypatch.setattr(
            loading_pipeline, "_free_resources", lambda: (ram_free, disk_free)
        )
        return loading_pipeline._prefetch_plan(SimpleNamespace())

    def test_starts_with_3_gb_ram_free(self, monkeypatch):
        assert self._plan(monkeypatch, 3.1 * GB, 100 * GB) == "full"

    def test_refuses_with_less_than_3_gb_ram_free(self, monkeypatch):
        assert self._plan(monkeypatch, 2.9 * GB, 100 * GB) is None

    def test_refuses_without_disk_for_images_and_masks(self, monkeypatch):
        assert self._plan(monkeypatch, 8 * GB, 13 * GB) is None

    def test_masks_count_towards_disk_not_ram(self):
        window = SimpleNamespace(
            images={"w01": np.zeros(100, np.uint8)},
            labels=np.zeros(50, np.uint16),
            basic=None,
        )

        assert loading_pipeline._prefetch_needs(window) == (200, 0)


class TestDiscardingAPreload:
    def _prefetch_with(self, future, cancel):
        prefetch = object.__new__(loading_pipeline._NextPositionPrefetch)
        prefetch._pending = (7, future, cancel)
        return prefetch

    def test_discard_returns_without_waiting_for_the_load(self):
        cancel = threading.Event()
        release = threading.Event()

        def slow_load():
            release.wait(10)
            raise_if_cancelled(cancel)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(slow_load)
            prefetch = self._prefetch_with(future, cancel)
            started = time.monotonic()

            prefetch._discard()

            assert time.monotonic() - started < 1.0
            assert cancel.is_set()
            assert prefetch._pending is None
            release.set()

        assert isinstance(future.exception(), LoadCancelled)

    def test_a_finished_unused_preload_has_its_files_deleted(
        self, monkeypatch
    ):
        cleaned = []
        monkeypatch.setattr(
            loading_pipeline,
            "cleanup_memmaps",
            lambda staging, **_: cleaned.append(staging),
        )
        staging = SimpleNamespace(name="p0007")
        future = concurrent.futures.Future()
        future.set_result(staging)

        self._prefetch_with(future, threading.Event())._discard()

        assert cleaned == [staging]

    def test_a_stopped_preload_is_not_cleaned_twice(self, monkeypatch):
        """A load that stopped already removed its own files on the way out."""
        cleaned = []
        monkeypatch.setattr(
            loading_pipeline,
            "cleanup_memmaps",
            lambda staging, **_: cleaned.append(staging),
        )
        future = concurrent.futures.Future()
        future.set_exception(LoadCancelled())

        self._prefetch_with(future, threading.Event())._discard()

        assert cleaned == []


class TestUnreadableFrames:
    def test_unreadable_mask_files_are_reported(self, tmp_path, monkeypatch):
        from SECQUOIA.core.segmentation import mask_io

        monkeypatch.setattr(
            mask_io,
            "read_with_reload",
            lambda fn, fp, **kw: read_with_reload(fn, fp, max_wait_s=0, **kw),
        )
        position = tmp_path / "exp_p0001"
        masks = tmp_path / "Segmentation" / position.name
        masks.mkdir(parents=True)
        frame = np.array([[0, 1], [2, 0]], dtype=np.uint16)
        tifffile.imwrite(masks / "mask_t00001.tif", frame)
        (masks / "mask_t00002.tif").write_bytes(b"not an image")
        failed: list = []

        stack = load_masks(
            str(tmp_path / "Segmentation"), str(position), "tif", failed=failed
        )

        assert [p.endswith("mask_t00002.tif") for p in failed] == [True]
        assert stack[1].max() == 0

    @pytest.fixture
    def preloaded(self, tmp_path, monkeypatch):
        """A finished preload of a one-frame position with one mask."""
        monkeypatch.setattr(
            loading_pipeline, "_free_resources", lambda: (8 * GB, 100 * GB)
        )
        position = tmp_path / "exp_p0001"
        position.mkdir()
        imageio.imwrite(
            position / "img_t00001_w01.png", np.ones((4, 5), np.uint8)
        )
        masks = tmp_path / "Segmentation" / position.name
        masks.mkdir(parents=True)
        imageio.imwrite(
            masks / "mask_t00001_m.png", np.array([[0, 1], [2, 0]], np.uint16)
        )
        staging = SimpleNamespace(
            position_selection=str(position),
            image_format="png",
            current_position_number=1,
            time_min_selected=1,
            time_max_selected=1,
            n_channels=1,
            ids_channels=["w01"],
            FL_identifiers_1="w01",
            _memmap_dir=None,
            _load_cancel=loading_pipeline._PreloadGate(),
            basic=None,
        )
        loading_pipeline._load_position_inputs(
            staging, str(tmp_path / "Segmentation"), 1, 1
        )
        yield staging
        loading_pipeline._cleanup_staging(staging)

    def test_reports_its_progress_and_failures(self, preloaded):
        assert preloaded._load_cancel.progress == 1.0
        assert preloaded.fl_failed_frames == 0
        assert preloaded.mask_failed_frames == 0

    def test_masks_wait_on_the_scratch_disk(self, preloaded):
        assert isinstance(preloaded.labels, np.memmap)
        assert any("masks_" in p for p in preloaded._memmap_files)

    def test_on_switch_the_masks_move_into_ram(self, preloaded):
        future = concurrent.futures.Future()
        future.set_result(preloaded)
        window = SimpleNamespace()

        loading_pipeline._adopt_prefetched_inputs(window, future, 1)

        assert type(window.labels) is np.ndarray
        assert window.labels[0].tolist() == [[0, 1], [2, 0]]

    def test_masks_in_lists_and_dicts_move_into_ram(self, tmp_path):
        disk = np.memmap(
            tmp_path / "m.dat", dtype=np.uint16, mode="w+", shape=(2,)
        )

        moved = loading_pipeline._masks_to_ram({"a": [disk], "b": None})

        assert type(moved["a"][0]) is np.ndarray
        assert moved["b"] is None


class TestSavedMeasurements:
    """A preload reads the position's saved measurements CSV ahead of time."""

    @pytest.fixture
    def staging(self, tmp_path):
        return SimpleNamespace(
            folder=str(tmp_path / "exp"),
            tracking_format="tTt",
            project_name="Project_1",
            n_masks=2,
            n_channels=1,
            current_position_number=3,
            read_measurements=True,
            _load_cancel=loading_pipeline._PreloadGate(),
        )

    def csv_path(self, staging):
        return loading_pipeline._position_csv_path(staging, 3, 1, 5)

    def test_reads_an_existing_csv(self, staging):
        pd.DataFrame({"Position": [3], "t": [0]}).to_csv(
            self.csv_path(staging), index=False
        )

        df = loading_pipeline._read_saved_measurements(staging, 1, 5)

        assert df["Position"].tolist() == [3]

    def test_nothing_without_a_csv(self, staging):
        assert loading_pipeline._read_saved_measurements(staging, 1, 5) is None

    def test_nothing_unless_asked(self, staging):
        pd.DataFrame({"Position": [3]}).to_csv(
            self.csv_path(staging), index=False
        )
        staging.read_measurements = False

        assert loading_pipeline._read_saved_measurements(staging, 1, 5) is None

    def test_a_broken_csv_is_left_for_the_switch(self, staging):
        with open(self.csv_path(staging), "w") as fh:
            fh.write('a,b\n"unclosed')

        assert loading_pipeline._read_saved_measurements(staging, 1, 5) is None

    def test_the_switch_uses_the_preread_table(self, tmp_path, monkeypatch):
        from conftest import make_fake_main_window

        from SECQUOIA.core.tracking import track_data

        window = make_fake_main_window(tmp_path)
        preread = window.track_df[window.track_df["Position"] == 1].copy()
        reads = []
        monkeypatch.setattr(
            track_data, "read_with_reload", lambda *a, **k: reads.append(a)
        )

        df = track_data._load_cached_position_measurements(
            window, 1, 1, 2, preloaded=preread
        )

        assert reads == []
        assert len(df) == len(preread)
        assert window.filtered_df is df
