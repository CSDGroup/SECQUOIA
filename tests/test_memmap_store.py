"""Tests for SECQUOIA temporary image stacks.

This is the only part of SECQUOIA that deletes a folder tree, so most of
these check what must not be deleted. The data is a temporary folder holding
two small memmap files and a lock file. The cleanup routine reads the
system temp folder, so those tests point it at a test folder first.
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

from SECQUOIA.core.memmap_store import (
    _close_memmap,
    _delete_file_list,
    _detach_viewer_images,
    _flush_close_attr,
    _rmtree_retry,
    acquire_dir_lock,
    cleanup_memmaps,
    ensure_memmap_dir,
    mask_file_allocator,
    reap_stale_memmap_dirs,
    release_dir_lock,
)


def make_memmap(path, shape=(2, 4, 4)) -> np.memmap:
    arr = np.memmap(str(path), dtype=np.uint16, mode="w+", shape=shape)
    arr[:] = 1
    return arr


class RecordingViewer:
    """A viewer that records every read of ``layers``."""

    def __init__(self, touched: list):
        self._touched = touched
        self._layers: list = []

    @property
    def layers(self) -> list:
        self._touched.append(1)
        return self._layers


class TestDirLock:
    def test_taking_a_lock_creates_the_lock_file(self, tmp_path):
        lock = acquire_dir_lock(str(tmp_path))

        try:
            assert lock is not None
            assert (tmp_path / ".owner.lock").exists()
        finally:
            release_dir_lock(lock)

    def test_releasing_a_lock_closes_the_handle(self, tmp_path):
        lock = acquire_dir_lock(str(tmp_path))

        release_dir_lock(lock)

        assert lock.closed

    def test_a_released_lock_can_be_taken_again(self, tmp_path):
        release_dir_lock(acquire_dir_lock(str(tmp_path)))

        second = acquire_dir_lock(str(tmp_path))

        try:
            assert second is not None
        finally:
            release_dir_lock(second)

    def test_a_missing_directory_cannot_be_locked(self, tmp_path):
        assert acquire_dir_lock(str(tmp_path / "nope")) is None

    def test_releasing_nothing_is_not_an_error(self):
        release_dir_lock(None)

    def test_releasing_twice_is_not_an_error(self, tmp_path):
        lock = acquire_dir_lock(str(tmp_path))

        release_dir_lock(lock)
        release_dir_lock(lock)


# Reaping abandoned directories
@pytest.fixture
def fake_tempdir(tmp_path, monkeypatch):
    """Point the reaper at a temporary directory, never the real one."""
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    return tmp_path


class TestReapStaleMemmapDirs:
    def test_removes_an_abandoned_directory(self, fake_tempdir):
        stale = fake_tempdir / "SECQUOIA_memmaps_old"
        stale.mkdir()
        (stale / "stack.dat").write_bytes(b"leftover")

        reap_stale_memmap_dirs()

        assert not stale.exists()

    def test_keeps_a_directory_another_session_still_holds(self, fake_tempdir):
        """The lock is what distinguishes abandoned from in-use."""
        live = fake_tempdir / "SECQUOIA_memmaps_live"
        live.mkdir()
        lock = acquire_dir_lock(str(live))

        try:
            reap_stale_memmap_dirs()
            assert live.exists()
        finally:
            release_dir_lock(lock)

    def test_ignores_directories_that_are_not_ours(self, fake_tempdir):
        other = fake_tempdir / "someone_elses_data"
        other.mkdir()
        (other / "important.txt").write_text("do not delete")

        reap_stale_memmap_dirs()

        assert (other / "important.txt").exists()

    def test_ignores_a_file_that_matches_the_pattern(self, fake_tempdir):
        decoy = fake_tempdir / "SECQUOIA_memmaps_notadir"
        decoy.write_text("just a file")

        reap_stale_memmap_dirs()

        assert decoy.exists()

    def test_an_empty_temp_directory_is_fine(self, fake_tempdir):
        reap_stale_memmap_dirs()


# Closing arrays
class TestCloseMemmap:
    def test_closes_a_memmap(self, tmp_path):
        arr = make_memmap(tmp_path / "stack.dat")

        _close_memmap(arr)

        assert arr._mmap.closed

    def test_ignores_a_plain_array(self):
        _close_memmap(np.zeros((2, 2)))

    def test_ignores_none(self):
        _close_memmap(None)


class TestFlushCloseAttr:
    def test_empties_a_dict_of_stacks(self, fake_main_window, tmp_path):
        arr = make_memmap(tmp_path / "a.dat")
        main_window = fake_main_window(images={"w01": arr})

        _flush_close_attr(main_window, "images")

        assert main_window.images == {}
        assert arr._mmap.closed

    def test_empties_a_list_of_stacks(self, fake_main_window, tmp_path):
        arr = make_memmap(tmp_path / "b.dat")
        main_window = fake_main_window(images=[arr])

        _flush_close_attr(main_window, "images")

        assert main_window.images == []
        assert arr._mmap.closed

    def test_leaves_other_types_alone(self, fake_main_window):
        main_window = fake_main_window(images="not a container")

        _flush_close_attr(main_window, "images")

        assert main_window.images == "not a container"

    def test_a_missing_attribute_is_not_an_error(self, fake_main_window):
        _flush_close_attr(fake_main_window(), "images")


class TestDeleteFileList:
    def test_deletes_every_tracked_file(self, fake_main_window, tmp_path):
        paths = []
        for name in ("a.dat", "b.dat"):
            p = tmp_path / name
            p.write_bytes(b"x")
            paths.append(str(p))
        main_window = fake_main_window(_memmap_files=paths)

        _delete_file_list(main_window, "_memmap_files", retries=2, sleep_s=0)

        assert not any(os.path.exists(p) for p in paths)
        assert main_window._memmap_files == []

    def test_an_already_deleted_file_is_not_an_error(
        self, fake_main_window, tmp_path
    ):
        main_window = fake_main_window(
            _memmap_files=[str(tmp_path / "gone.dat")]
        )

        _delete_file_list(main_window, "_memmap_files", retries=2, sleep_s=0)

        assert main_window._memmap_files == []

    def test_empty_entries_are_skipped(self, fake_main_window):
        main_window = fake_main_window(_memmap_files=["", None])

        _delete_file_list(main_window, "_memmap_files", retries=2, sleep_s=0)

        assert main_window._memmap_files == []

    def test_a_non_list_attribute_is_left_alone(self, fake_main_window):
        main_window = fake_main_window(_memmap_files="not a list")

        _delete_file_list(main_window, "_memmap_files", retries=2, sleep_s=0)

        assert main_window._memmap_files == "not a list"


class TestRmtreeRetry:
    def test_removes_a_tree(self, tmp_path):
        target = tmp_path / "tree"
        (target / "nested").mkdir(parents=True)
        (target / "nested" / "file.txt").write_text("x")

        assert _rmtree_retry(str(target), retries=2, sleep_s=0) is True
        assert not target.exists()

    def test_a_missing_directory_counts_as_removed(self, tmp_path):
        assert _rmtree_retry(str(tmp_path / "nope"), 2, 0) is True

    def test_an_empty_path_counts_as_removed(self):
        assert _rmtree_retry("", 2, 0) is True

    def test_a_file_is_not_a_tree(self, tmp_path):
        """Only directories are removed; a file path is a no-op."""
        target = tmp_path / "file.txt"
        target.write_text("x")

        assert _rmtree_retry(str(target), 2, 0) is True
        assert target.exists()


class TestDetachViewerImages:
    def test_removes_image_and_mask_layers_and_keeps_the_rest(self):

        class Image:
            def __init__(self):
                self.data = np.zeros((2, 2))

        class Labels(Image):
            pass

        class Points(Image):
            pass

        image, labels, points = Image(), Labels(), Points()
        layers = [image, labels, points]

        _detach_viewer_images(SimpleNamespace(layers=layers))

        assert layers == [points]
        assert image.data is None

    def test_no_viewer_is_not_an_error(self):
        _detach_viewer_images(None)


# Full cleanup
@pytest.fixture
def loaded_session(fake_main_window, tmp_path):
    """A window holding memmaps in a directory it owns."""
    mm_dir = tmp_path / "SECQUOIA_memmaps_session"
    mm_dir.mkdir()
    files = []
    arrays = {}
    for name in ("w00", "w01"):
        path = mm_dir / f"{name}.dat"
        arrays[name] = make_memmap(path)
        files.append(str(path))

    return fake_main_window(
        images=arrays,
        corrected_images={},
        corrected_images_ratioflat={},
        corrected_images_noratioflat={},
        _memmap_files=files,
        _ratioflat_memmap_files=[],
        _noratioflat_memmap_files=[],
        _memmap_dir=str(mm_dir),
        _memmap_lock=acquire_dir_lock(str(mm_dir)),
        viewer_1=None,
        viewer_2=None,
    )


class TestCleanupMemmaps:
    def test_removes_the_memmap_directory(self, loaded_session):
        mm_dir = loaded_session._memmap_dir

        cleanup_memmaps(loaded_session, retries=2, sleep_s=0)

        assert not os.path.isdir(mm_dir)

    def test_clears_the_recorded_directory(self, loaded_session):
        cleanup_memmaps(loaded_session, retries=2, sleep_s=0)

        assert loaded_session._memmap_dir == ""

    def test_empties_the_image_containers(self, loaded_session):
        cleanup_memmaps(loaded_session, retries=2, sleep_s=0)

        assert loaded_session.images == {}

    def test_clears_the_tracked_file_lists(self, loaded_session):
        cleanup_memmaps(loaded_session, retries=2, sleep_s=0)

        assert loaded_session._memmap_files == []

    def test_releases_the_directory_lock(self, loaded_session):
        cleanup_memmaps(loaded_session, retries=2, sleep_s=0)

        assert loaded_session._memmap_lock is None

    def test_the_directory_can_be_reclaimed_afterwards(
        self, loaded_session, tmp_path
    ):
        """A cleaned up directory must not stay locked against a later run."""
        cleanup_memmaps(loaded_session, retries=2, sleep_s=0)
        recreated = tmp_path / "SECQUOIA_memmaps_session"
        recreated.mkdir()

        lock = acquire_dir_lock(str(recreated))

        try:
            assert lock is not None
        finally:
            release_dir_lock(lock)

    def test_a_window_with_nothing_loaded_is_fine(self, fake_main_window):
        cleanup_memmaps(fake_main_window(_memmap_dir=""), retries=2, sleep_s=0)

    def test_viewer_detaching_can_be_switched_off(self, loaded_session):
        touched = []
        loaded_session.viewer_1 = RecordingViewer(touched)

        cleanup_memmaps(
            loaded_session,
            remove_viewer_image_layers=False,
            retries=2,
            sleep_s=0,
        )

        assert touched == []

    def test_viewer_detaching_is_on_by_default(self, loaded_session):
        touched = []
        loaded_session.viewer_1 = RecordingViewer(touched)

        cleanup_memmaps(loaded_session, retries=2, sleep_s=0)

        assert touched


class TestMaskFiles:
    @pytest.fixture
    def owner(self, monkeypatch, tmp_path):
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        owner = SimpleNamespace(_memmap_dir="")
        yield owner
        release_dir_lock(getattr(owner, "_memmap_lock", None))

    def test_the_scratch_folder_is_made_once(self, owner):
        folder = ensure_memmap_dir(owner)

        assert ensure_memmap_dir(owner) == folder
        assert os.path.basename(folder).startswith("SECQUOIA_memmaps_")
        assert owner._memmap_lock is not None

    def test_each_mask_stack_gets_its_own_tracked_file(self, owner):
        alloc = mask_file_allocator(owner)

        first = alloc((2, 3, 3), np.uint16)
        second = alloc((2, 3, 3), np.uint16)

        assert isinstance(first, np.memmap) and not first.any()
        assert len(set(owner._mask_memmap_files)) == 2
        for path in owner._mask_memmap_files:
            assert os.path.dirname(path) == owner._memmap_dir
        second[1, 0, 0] = 7  # edits land in the file
        second.flush()
        assert np.fromfile(owner._mask_memmap_files[1], np.uint16)[9] == 7

    def test_cleanup_releases_and_deletes_disk_masks(
        self, owner, fake_main_window
    ):
        alloc = mask_file_allocator(owner)
        window = fake_main_window(
            labels=[alloc((2, 3, 3), np.uint16)],
            _memmap_dir=owner._memmap_dir,
            _memmap_lock=owner._memmap_lock,
            _mask_memmap_files=owner._mask_memmap_files,
            viewer_1=None,
            viewer_2=None,
        )
        files = list(owner._mask_memmap_files)

        cleanup_memmaps(window, retries=2, sleep_s=0)

        assert window.labels == []
        assert window._mask_memmap_files == []
        assert not any(os.path.exists(path) for path in files)

    def test_cleanup_leaves_masks_in_ram_alone(self, fake_main_window):
        masks = [np.zeros((2, 3, 3), np.uint16)]
        window = fake_main_window(labels=masks, _memmap_dir="")

        cleanup_memmaps(window, retries=2, sleep_s=0)

        assert window.labels is masks
