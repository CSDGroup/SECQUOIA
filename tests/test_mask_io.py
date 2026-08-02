"""Tests for SECQUOIA mask files.

The test folder holds masks for time points 1, 2 and 4, with 3 missing on
purpose, so the loader can be checked to leave a gap rather than shift the
later frames up. Every mask carries two labelled objects, because a mask
holding only one is read as binary and relabelled from 1.
"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from SECQUOIA.core.segmentation.mask_io import (
    _mask_to_2d,
    _mask_worker_count,
    _n_distinct_values,
    fallback_mask_filename,
    find_exact_mask,
    infer_w_token,
    labels2d_from_container,
    load_masks,
    mask_reader_for_format,
    position_dir_for_mask,
    position_name,
    t_file_from_idx,
)


# Reader selection and small helpers
class TestMaskReaderForFormat:
    @pytest.mark.parametrize("fmt", ["tif", "tiff", "TIF", ".tif"])
    def test_tiff_reads_without_channel_swapping(self, fmt):
        _read_fn, is_bgr = mask_reader_for_format(fmt)

        assert is_bgr is False

    def test_an_unknown_format_still_returns_a_reader(self):
        read_fn, _is_bgr = mask_reader_for_format("not-a-format")

        assert callable(read_fn)

    def test_no_format_still_returns_a_reader(self):
        read_fn, _is_bgr = mask_reader_for_format(None)

        assert callable(read_fn)


class TestMaskWorkerCount:
    def test_honours_an_explicit_request(self):
        assert _mask_worker_count(4) == 4

    def test_clamps_to_at_least_one(self):
        assert _mask_worker_count(-5) == 1

    def test_clamps_to_at_most_eight(self):
        assert _mask_worker_count(99) == 8

    def test_falls_back_to_the_machine_when_unset(self):
        assert 1 <= _mask_worker_count(None) <= 8

    def test_falls_back_when_the_request_is_unparsable(self):
        assert 1 <= _mask_worker_count("many") <= 8


class TestMaskTo2d:
    def test_passes_a_plane_through(self):
        arr = np.zeros((4, 5), dtype=np.uint16)

        assert _mask_to_2d(arr, False).shape == (4, 5)

    def test_takes_the_red_channel_of_a_bgr_image(self):
        arr = np.zeros((4, 5, 3), dtype=np.uint8)
        arr[..., 2] = 7  # cv2 puts red last

        assert _mask_to_2d(arr, True).tolist() == np.full((4, 5), 7).tolist()

    def test_takes_the_first_channel_otherwise(self):
        arr = np.zeros((4, 5, 3), dtype=np.uint8)
        arr[..., 0] = 3

        assert _mask_to_2d(arr, False).tolist() == np.full((4, 5), 3).tolist()


class TestNDistinctValues:
    def test_counts_labels_in_an_integer_image(self):
        img = np.array([[0, 1, 1], [2, 2, 3]], dtype=np.uint16)

        assert _n_distinct_values(img) == 4

    def test_an_all_zero_image_has_one_value(self):
        assert _n_distinct_values(np.zeros((4, 4), dtype=np.uint16)) == 1

    def test_an_empty_image_has_none(self):
        assert _n_distinct_values(np.array([], dtype=np.uint16)) == 0

    def test_handles_float_images(self):
        img = np.array([[0.0, 0.5], [0.5, 1.0]])

        assert _n_distinct_values(img) == 3

    def test_handles_negative_values(self):
        img = np.array([[-1, 0], [1, 1]], dtype=np.int16)

        assert _n_distinct_values(img) == 3

    def test_agrees_with_numpy_unique(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 50, size=(20, 20)).astype(np.uint16)

        assert _n_distinct_values(img) == np.unique(img).size


# Loading stacks


def write_masks(folder, position: str, frames: dict[int, np.ndarray]):
    """Write ``{t_file_number: plane}`` into ``folder/position`` as TIFFs."""
    target = folder / position
    target.mkdir(parents=True, exist_ok=True)
    for t, plane in frames.items():
        tifffile.imwrite(target / f"{position}_t{t:05d}_w01_mask.tif", plane)
    return target


def labelled_plane(marker: int, shape=(6, 6)) -> np.ndarray:
    """A plane with two labelled objects, the second one carrying ``marker``.

    Three distinct values matter: a plane with only two (background plus one
    object) is treated as a *binary* mask and relabelled from 1, which would
    erase the marker that identifies which frame this is.
    """
    plane = np.zeros(shape, dtype=np.uint16)
    plane[1:3, 1:3] = 1
    plane[4:6, 4:6] = marker
    return plane


@pytest.fixture
def seg_folder(tmp_path):
    """A segmentation folder with frames 1, 2 and 4 written (3 is missing)."""
    seg = tmp_path / "Segmentation1"
    write_masks(
        seg,
        "exp_p0001",
        {1: labelled_plane(2), 2: labelled_plane(3), 4: labelled_plane(5)},
    )
    return seg


@pytest.fixture
def position(tmp_path):
    return str(tmp_path / "images" / "exp_p0001")


class TestLoadMasks:
    def test_loads_every_frame_in_the_folder(self, seg_folder, position):
        stack = load_masks(str(seg_folder), position, "tif")

        assert stack.shape == (3, 6, 6)

    def test_a_t_range_allocates_the_whole_range(self, seg_folder, position):
        """Four frames requested, three on disk: the stack still has four."""
        stack = load_masks(
            str(seg_folder), position, "tif", t_file_min=1, t_file_max=4
        )

        assert stack.shape == (4, 6, 6)

    def test_a_missing_frame_leaves_a_gap_rather_than_shifting(
        self, seg_folder, position
    ):
        """This is the convention that matters: row index follows the filename."""
        stack = load_masks(
            str(seg_folder), position, "tif", t_file_min=1, t_file_max=4
        )

        assert stack[0].max() == 2
        assert stack[1].max() == 3
        assert stack[2].max() == 0  # frame 3 was never acquired
        assert stack[3].max() == 5

    def test_the_range_is_rebased_to_zero(self, seg_folder, position):
        stack = load_masks(
            str(seg_folder), position, "tif", t_file_min=2, t_file_max=4
        )

        assert stack.shape == (3, 6, 6)
        assert stack[0].max() == 3  # t-file 2 becomes row 0

    def test_files_outside_the_range_are_ignored(self, seg_folder, position):
        stack = load_masks(
            str(seg_folder), position, "tif", t_file_min=1, t_file_max=2
        )

        assert stack.shape == (2, 6, 6)

    def test_a_binary_mask_is_relabelled_into_objects(
        self, tmp_path, position
    ):
        """A two-value image is a binary mask; each blob needs its own label."""
        seg = tmp_path / "Segmentation1"
        binary = np.zeros((8, 8), dtype=np.uint8)
        binary[1:3, 1:3] = 255
        binary[5:7, 5:7] = 255
        write_masks(seg, "exp_p0001", {1: binary})

        stack = load_masks(str(seg), position, "tif")

        assert sorted(np.unique(stack[0]).tolist()) == [0, 1, 2]

    def test_an_already_labelled_mask_is_left_alone(
        self, seg_folder, position
    ):
        """Three distinct values means it is already labelled, not binary."""
        stack = load_masks(str(seg_folder), position, "tif")

        assert sorted(np.unique(stack[0]).tolist()) == [0, 1, 2]

    def test_the_output_dtype_is_honoured(self, seg_folder, position):
        stack = load_masks(str(seg_folder), position, "tif", dtype=np.uint32)

        assert stack.dtype == np.uint32

    def test_a_list_of_folders_returns_a_list_of_stacks(
        self, tmp_path, seg_folder, position
    ):
        second = tmp_path / "Segmentation2"
        write_masks(second, "exp_p0001", {1: labelled_plane(9)})

        stacks = load_masks([str(seg_folder), str(second)], position, "tif")

        assert isinstance(stacks, list)
        assert len(stacks) == 2
        assert stacks[1][0].max() == 9

    def test_a_dict_of_folders_keeps_its_keys(
        self, tmp_path, seg_folder, position
    ):
        second = tmp_path / "Segmentation2"
        write_masks(second, "exp_p0001", {1: labelled_plane(9)})

        stacks = load_masks(
            {"a": str(seg_folder), "b": str(second)}, position, "tif"
        )

        assert set(stacks) == {"a", "b"}
        assert stacks["b"][0].max() == 9

    def test_an_unsupported_argument_returns_nothing(self, position):
        assert load_masks(42, position, "tif") is None

    def test_an_empty_path_returns_nothing(self, position):
        assert load_masks("", position, "tif") is None

    def test_a_folder_with_no_masks_returns_nothing(self, tmp_path, position):
        empty = tmp_path / "Segmentation1"
        (empty / "exp_p0001").mkdir(parents=True)

        assert load_masks(str(empty), position, "tif") is None

    def test_a_range_that_matches_no_file_returns_nothing(
        self, seg_folder, position
    ):
        assert (
            load_masks(
                str(seg_folder), position, "tif", t_file_min=90, t_file_max=99
            )
            is None
        )

    def test_a_single_worker_gives_the_same_result(self, seg_folder, position):
        one = load_masks(str(seg_folder), position, "tif", max_workers=1)
        many = load_masks(str(seg_folder), position, "tif", max_workers=8)

        assert np.array_equal(one, many)


# Slicing a loaded container
class TestLabels2dFromContainer:
    STACK = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)

    def test_slices_a_bare_array_by_time(self):
        out = labels2d_from_container(self.STACK, 1, 2)

        assert np.array_equal(out, self.STACK[2])

    def test_a_two_dimensional_array_is_only_frame_zero(self):
        plane = np.zeros((4, 5), dtype=np.uint16)

        assert labels2d_from_container(plane, 1, 0) is plane
        assert labels2d_from_container(plane, 1, 1) is None

    def test_an_out_of_range_frame_returns_nothing(self):
        assert labels2d_from_container(self.STACK, 1, 99) is None

    def test_picks_the_mask_out_of_a_list(self):
        containers = [self.STACK, self.STACK + 100]

        out = labels2d_from_container(containers, 2, 0)

        assert np.array_equal(out, self.STACK[0] + 100)

    def test_an_out_of_range_mask_index_returns_nothing(self):
        assert labels2d_from_container([self.STACK], 5, 0) is None

    def test_picks_the_mask_out_of_a_dict_by_key(self):
        containers = {"1": self.STACK, "2": self.STACK + 100}

        out = labels2d_from_container(containers, 2, 0)

        assert np.array_equal(out, self.STACK[0] + 100)

    def test_falls_back_to_sorted_dict_order(self):
        """Keys may be folder names rather than mask numbers."""
        containers = {"Seg_b": self.STACK + 100, "Seg_a": self.STACK}

        out = labels2d_from_container(containers, 1, 0)

        assert np.array_equal(out, self.STACK[0])

    def test_no_container_returns_nothing(self):
        assert labels2d_from_container(None, 1, 0) is None


# Path resolution
class TestPositionName:
    def test_reads_the_folder_name(self, fake_main_window):
        main_window = fake_main_window(
            position_selection="/data/exp/exp_p0007"
        )

        assert position_name(main_window) == "exp_p0007"

    def test_ignores_a_trailing_separator(self, fake_main_window):
        main_window = fake_main_window(
            position_selection="/data/exp/exp_p0007/"
        )

        assert position_name(main_window) == "exp_p0007"

    def test_an_unset_position_normalises_to_the_current_directory(
        self, fake_main_window
    ):
        assert position_name(fake_main_window(position_selection="")) == "."


class TestPositionDirForMask:
    def test_joins_the_position_onto_the_mask_folder(self):
        result = position_dir_for_mask("/a/Segmentation1", "exp_p0001")

        assert result.endswith(
            ("Segmentation1/exp_p0001", "Segmentation1\\exp_p0001")
        )


class TestTFileFromIdx:
    def test_maps_the_first_index_of_a_selection_to_its_first_file(
        self, fake_main_window
    ):
        """t_idx is an absolute 0-based file index, not a stack row.

        For a selection of t-files 5..10, ``current_t_range`` reports the
        index range 4..9, so index 4 -- not 0 -- is the first frame.
        """
        main_window = fake_main_window(
            time_min_selected=5, time_max_selected=10
        )

        assert t_file_from_idx(main_window, 4) == 5
        assert t_file_from_idx(main_window, 9) == 10

    def test_is_always_one_more_than_the_index(self, fake_main_window):
        """The selected range cancels out of the arithmetic entirely."""
        for lo, hi in ((1, 10), (5, 10), (3, 6)):
            main_window = fake_main_window(
                time_min_selected=lo, time_max_selected=hi
            )
            assert t_file_from_idx(main_window, 4) == 5


class TestFindExactMask:
    @pytest.fixture
    def folder(self, tmp_path):
        for name in (
            "exp_p0001_t00001_z001_w01_m00_mask.tif",
            "exp_p0001_t00002_z001_w01_m00_mask.tif",
            "exp_p0002_t00001_z001_w01_m00_mask.tif",
            "notes.txt",
        ):
            (tmp_path / name).touch()
        return str(tmp_path)

    def test_finds_the_file_for_a_position_and_frame(self, folder):
        found = find_exact_mask(folder, "exp_p0001", 2, "tif")

        assert found.endswith("exp_p0001_t00002_z001_w01_m00_mask.tif")

    def test_does_not_match_another_position(self, folder):
        found = find_exact_mask(folder, "exp_p0002", 2, "tif")

        assert found is None

    def test_does_not_match_another_format(self, folder):
        assert find_exact_mask(folder, "exp_p0001", 1, "png") is None

    def test_returns_none_for_a_missing_frame(self, folder):
        assert find_exact_mask(folder, "exp_p0001", 99, "tif") is None

    def test_a_missing_folder_reports_no_match(self):
        """A folder that was renamed or unmounted is a miss, not a failure."""
        assert (
            find_exact_mask("/does/not/exist", "exp_p0001", 1, "tif") is None
        )


class TestInferWToken:
    def test_reads_the_channel_token_from_a_filename(self, tmp_path):
        (tmp_path / "exp_p0001_t00001_z001_w04_m00_mask.tif").touch()

        assert infer_w_token(str(tmp_path)) == "w04"

    def test_returns_none_without_a_token(self, tmp_path):
        (tmp_path / "readme.txt").touch()

        assert infer_w_token(str(tmp_path)) is None

    def test_a_missing_folder_reports_no_token(self):
        assert infer_w_token("/does/not/exist") is None

    def test_a_save_path_can_still_be_built_for_a_missing_folder(self):
        name = fallback_mask_filename("/does/not/exist", "exp_p0001", 7, "tif")

        assert name.endswith("exp_p0001_t00007_z001_w01_m00_mask.tif")


class TestFallbackMaskFilename:
    def test_builds_a_name_using_the_inferred_channel(self, tmp_path):
        (tmp_path / "exp_p0001_t00001_z001_w04_m00_mask.tif").touch()

        name = fallback_mask_filename(str(tmp_path), "exp_p0001", 7, "tif")

        assert name.endswith("exp_p0001_t00007_z001_w04_m00_mask.tif")

    def test_falls_back_to_the_default_channel(self, tmp_path):
        name = fallback_mask_filename(str(tmp_path), "exp_p0001", 7, "tif")

        assert name.endswith("exp_p0001_t00007_z001_w01_m00_mask.tif")

    def test_pads_the_frame_number_to_five_digits(self, tmp_path):
        name = fallback_mask_filename(str(tmp_path), "exp_p0001", 42, "png")

        assert "_t00042_" in name
        assert name.endswith(".png")
