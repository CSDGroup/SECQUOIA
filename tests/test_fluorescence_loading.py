"""Behavioral tests for the fluorescence loading pipeline."""

from __future__ import annotations

from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest

from SECQUOIA.core.fluorescence_loading import load_fl_channels
from SECQUOIA.core.memmap_store import cleanup_memmaps

# Windows built by make_fake_main_window
_WINDOWS: list[SimpleNamespace] = []


@pytest.fixture(autouse=True)
def release_memmaps():
    """Close the stacks each test allocates, as the app does on window close.

    ``load_fl_channels`` opens a memmap directory and holds an exclusive lock
    on it. In the app ``cleanup_memmaps`` releases both; without it here the
    lock file handle stays open and every test emits a ResourceWarning.
    """
    _WINDOWS.clear()
    yield
    while _WINDOWS:
        cleanup_memmaps(_WINDOWS.pop(), remove_viewer_image_layers=False)


def _make_frame(value: int, shape=(4, 5)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


def make_fake_main_window(tmp_path, *, second_channel_identifier=None):
    position_folder = tmp_path / "fake_p0001"
    position_folder.mkdir(parents=True)

    # t=1 and t=3 exist for channel "w01"; t=2 is intentionally missing.
    imageio.imwrite(position_folder / "img_t00001_w01.png", _make_frame(10))
    imageio.imwrite(position_folder / "img_t00003_w01.png", _make_frame(30))

    ids_channels = ["w01"]
    n_channels = 1
    extra = {}
    if second_channel_identifier is not None:
        ids_channels = ["w01", "w02"]
        n_channels = 2
        extra["FL_identifiers_2"] = second_channel_identifier

    main_window = SimpleNamespace(
        position_selection=str(position_folder),
        image_format="png",
        current_position_number=1,
        time_min_selected=1,
        time_max_selected=3,
        n_channels=n_channels,
        ids_channels=ids_channels,
        FL_identifiers_1="w01",
        **extra,
    )
    _WINDOWS.append(main_window)
    return main_window


def test_loads_frames_into_memmap_and_tracks_missing_ones(tmp_path):
    main_window = make_fake_main_window(tmp_path)

    load_fl_channels(main_window)

    stack = main_window.images["w01"]
    present = main_window.image_present["w01"]

    assert stack.shape == (3, 4, 5)
    np.testing.assert_array_equal(present, [True, False, True])
    np.testing.assert_array_equal(stack[0], _make_frame(10))
    np.testing.assert_array_equal(stack[1], np.zeros((4, 5), dtype=np.uint8))
    np.testing.assert_array_equal(stack[2], _make_frame(30))


def test_channel_without_identifier_is_marked_empty(tmp_path):
    main_window = make_fake_main_window(
        tmp_path, second_channel_identifier=None
    )
    main_window.n_channels = 2
    main_window.ids_channels = ["w01", "w02"]
    # No FL_identifiers_2 attribute set -> channel 2 has no identifier.

    load_fl_channels(main_window)

    assert main_window.images["w02"] is None
    assert main_window.image_present["w02"].shape == (0,)
    assert main_window.images["w01"].shape == (3, 4, 5)


def test_channel_with_no_matching_files_is_marked_empty(tmp_path):
    main_window = make_fake_main_window(
        tmp_path, second_channel_identifier="w02"
    )
    # w02 has an identifier but no files on disk in the selected range.

    load_fl_channels(main_window)

    assert main_window.images["w02"] is None
    assert main_window.image_present["w02"].shape == (0,)


def test_zero_length_t_range_marks_all_channels_empty(tmp_path):
    main_window = make_fake_main_window(tmp_path)
    main_window.time_min_selected = 5
    main_window.time_max_selected = 1

    import SECQUOIA.core.fluorescence_loading as fluorescence_loading

    original = fluorescence_loading._current_t_range
    fluorescence_loading._current_t_range = lambda mw: (5, 4, 4, 3)
    try:
        load_fl_channels(main_window)
    finally:
        fluorescence_loading._current_t_range = original

    assert main_window.images["w01"] is None
    assert main_window.image_present["w01"].shape == (0,)


def test_progress_callback_reaches_completion(tmp_path):
    main_window = make_fake_main_window(tmp_path)
    calls = []

    def progress(frac, msg):
        calls.append((frac, msg))

    load_fl_channels(main_window, progress_cb={"fl": progress})

    assert calls
    assert calls[-1][0] == 1.0


def test_progress_callback_as_bare_callable(tmp_path):
    main_window = make_fake_main_window(tmp_path)
    calls = []

    load_fl_channels(
        main_window, progress_cb=lambda frac, msg: calls.append((frac, msg))
    )

    assert calls
    assert calls[-1][0] == 1.0
