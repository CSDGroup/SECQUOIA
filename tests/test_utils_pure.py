"""Tests for the small SECQUOIA helpers that need no Qt.

Four groups: position numbers read out of folder names, the interval maths
behind the outlier histograms, the retry wrapper for reads that fail on a
synced drive, and the folder discovery inside an experiment. The data is
either a plain string or a temporary experiment folder with two positions
and a few images.
"""

from __future__ import annotations

import numpy as np
import pytest

from SECQUOIA.core.experiment_layout import (
    detect_time_max,
    detect_unique_w_channels,
    find_analysis_dir,
    find_basic_folders,
    find_example_position_folder,
    find_images_csvs,
)
from SECQUOIA.utils.intervals import (
    compute_bins,
    intersect_intervals,
    resolve_multi,
    threshold_regions,
    union_intervals,
)
from SECQUOIA.utils.io_reload import (
    read_text_with_reload,
    read_with_reload,
    write_text_with_reload,
)
from SECQUOIA.utils.positions import (
    coerce_int,
    current_position_number,
    detected_position_numbers,
    position_index_from_number,
    position_number_at_current_index,
    position_number_from_folder,
    resolve_position_range,
)


# utils.positions
class TestCoerceInt:
    def test_returns_the_first_convertible_candidate(self):
        assert coerce_int(None, "nope", "7", 9) == 7

    def test_accepts_floats_by_truncating(self):
        assert coerce_int(3.9) == 3

    def test_returns_none_when_nothing_converts(self):
        assert coerce_int(None, "abc", [], {}) is None

    def test_returns_none_for_no_candidates(self):
        assert coerce_int() is None


class TestPositionNumberFromFolder:
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("/data/exp_p0001", 1),
            ("/data/exp_p0042", 42),
            ("/data/exp_p7", 7),
            ("/data/exp_p0001/", 1),  # trailing separator
            ("exp_p0003", 3),
            ("/data/250615MA40_p0002", 2),
        ],
    )
    def test_extracts_the_trailing_number(self, path, expected):
        assert position_number_from_folder(path) == expected

    @pytest.mark.parametrize(
        "path",
        [
            "/data/Analysis",  # no _p token at all
            "/data/exp_pXYZ",  # token present but not a number
            "",
        ],
    )
    def test_returns_none_when_there_is_no_position_number(self, path):
        assert position_number_from_folder(path) is None

    def test_uses_only_the_last_path_component(self):
        """A _p token higher up the path must not be picked up."""
        assert position_number_from_folder("/exp_p0009/Analysis") is None


class TestDetectedPositionNumbers:
    def test_returns_sorted_numbers(self, fake_main_window):
        main_window = fake_main_window(
            position_folders=["/e/exp_p0003", "/e/exp_p0001", "/e/exp_p0002"]
        )
        assert detected_position_numbers(main_window) == [1, 2, 3]

    def test_skips_folders_without_a_number(self, fake_main_window):
        main_window = fake_main_window(
            position_folders=["/e/exp_p0001", "/e/Analysis", "/e/exp_p0002"]
        )
        assert detected_position_numbers(main_window) == [1, 2]

    def test_clips_to_the_requested_bounds(self, fake_main_window):
        main_window = fake_main_window(
            position_folders=[f"/e/exp_p{n:04d}" for n in (1, 2, 3, 4, 5)]
        )
        assert detected_position_numbers(main_window, 2, 4) == [2, 3, 4]

    def test_handles_a_window_with_no_folders(self, fake_main_window):
        assert detected_position_numbers(fake_main_window()) == []


class TestResolvePositionRange:
    def test_prefers_the_explicit_selection(self, fake_main_window):
        main_window = fake_main_window(
            position_min_selected=4,
            position_max_selected=6,
            position_min=1,
            position_max=99,
            position_folders=["/e/exp_p0001", "/e/exp_p0002"],
        )
        assert resolve_position_range(main_window) == (4, 6)

    def test_falls_back_to_the_detected_folders(self, fake_main_window):
        main_window = fake_main_window(
            position_folders=["/e/exp_p0002", "/e/exp_p0005"]
        )
        assert resolve_position_range(main_window) == (2, 5)

    def test_falls_back_to_one_when_nothing_is_known(self, fake_main_window):
        assert resolve_position_range(fake_main_window()) == (1, 1)

    def test_swaps_an_inverted_range(self, fake_main_window):
        main_window = fake_main_window(
            position_min_selected=9, position_max_selected=2
        )
        assert resolve_position_range(main_window) == (2, 9)


class TestPositionIndexFromNumber:
    def test_maps_a_number_to_its_folder_index(self, fake_main_window):
        main_window = fake_main_window(
            position_folders=["/e/exp_p0002", "/e/exp_p0005", "/e/exp_p0009"]
        )
        assert position_index_from_number(main_window, 5) == 1

    def test_falls_back_to_zero_for_an_unknown_number(self, fake_main_window):
        main_window = fake_main_window(position_folders=["/e/exp_p0002"])
        assert position_index_from_number(main_window, 404) == 0

    def test_falls_back_to_zero_for_a_missing_number(self, fake_main_window):
        main_window = fake_main_window(position_folders=["/e/exp_p0002"])
        assert position_index_from_number(main_window, None) == 0


class TestCurrentPositionNumber:
    def test_prefers_the_cached_number(self, fake_main_window):
        main_window = fake_main_window(
            current_position_number=8,
            current_position_index=0,
            position_folders=["/e/exp_p0001"],
        )
        assert current_position_number(main_window) == 8

    def test_derives_from_the_current_index(self, fake_main_window):
        main_window = fake_main_window(
            current_position_number=None,
            current_position_index=1,
            position_folders=["/e/exp_p0001", "/e/exp_p0004"],
        )
        assert current_position_number(main_window) == 4

    def test_uses_the_fallback_iterable_last(self, fake_main_window):
        main_window = fake_main_window()
        assert current_position_number(main_window, fallback=[11, 12]) == 11

    def test_returns_none_when_nothing_is_available(self, fake_main_window):
        assert current_position_number(fake_main_window()) is None

    def test_at_current_index_ignores_the_cached_number(
        self, fake_main_window
    ):
        """This is the whole reason the second function exists."""
        main_window = fake_main_window(
            current_position_number=8,  # stale cache
            current_position_index=0,
            position_folders=["/e/exp_p0001"],
        )
        assert position_number_at_current_index(main_window) == 1

    def test_at_current_index_rejects_an_out_of_range_index(
        self, fake_main_window
    ):
        main_window = fake_main_window(
            current_position_index=5, position_folders=["/e/exp_p0001"]
        )
        assert position_number_at_current_index(main_window) is None


# ---------------------------------------------------------------------------
# utils.intervals
# ---------------------------------------------------------------------------


class TestResolveMulti:
    def test_an_empty_selection_means_everything(self):
        assert resolve_multi([], [1, 2, 3]) == [1, 2, 3]

    def test_a_real_selection_is_kept(self):
        assert resolve_multi([2], [1, 2, 3]) == [2]

    def test_none_is_not_everything(self):
        """``None`` means "nothing chosen yet", unlike ``[]``."""
        assert resolve_multi(None, [1, 2, 3]) == []

    def test_an_empty_selection_with_no_values_stays_empty(self):
        assert resolve_multi([], []) == []


class TestComputeBins:
    def test_uses_freedman_diaconis_on_spread_data(self):
        rng = np.random.default_rng(0)
        bins = compute_bins(rng.normal(size=1000))
        assert 5 <= bins <= 200

    def test_falls_back_to_fifty_without_spread(self):
        """A zero IQR gives no bin width, so the rule cannot apply."""
        assert compute_bins(np.ones(100)) == 50

    def test_clamps_to_the_lower_bound(self):
        assert compute_bins(np.array([0.0, 1.0])) >= 5

    def test_returns_fifty_for_input_it_cannot_handle(self):
        assert compute_bins(None) == 50

    def test_returns_fifty_for_an_empty_array(self):
        assert compute_bins(np.array([])) == 50


class TestThresholdRegions:
    EDGES = [0.0, 1.0, 2.0, 3.0, 4.0]

    @pytest.mark.parametrize("op", ["<", "<="])
    def test_less_than_covers_everything_below(self, op):
        assert threshold_regions(op, 2.5, self.EDGES) == [(0.0, 2.5)]

    @pytest.mark.parametrize("op", [">", ">="])
    def test_greater_than_covers_everything_above(self, op):
        assert threshold_regions(op, 2.5, self.EDGES) == [(2.5, 4.0)]

    def test_equality_becomes_a_band_one_bin_wide(self):
        ((low, high),) = threshold_regions("=", 2.0, self.EDGES)
        assert low == pytest.approx(1.5)
        assert high == pytest.approx(2.5)

    def test_an_unknown_operator_selects_nothing(self):
        assert threshold_regions("!=", 2.0, self.EDGES) == []


class TestUnionIntervals:
    def test_merges_overlapping_intervals(self):
        assert union_intervals([(0, 2), (1, 3)]) == [(0, 3)]

    def test_keeps_disjoint_intervals_apart(self):
        assert union_intervals([(0, 1), (2, 3)]) == [(0, 1), (2, 3)]

    def test_merges_intervals_that_only_touch(self):
        assert union_intervals([(0, 1), (1, 2)]) == [(0, 2)]

    def test_normalises_reversed_bounds(self):
        assert union_intervals([(3, 1)]) == [(1, 3)]

    def test_sorts_the_result(self):
        assert union_intervals([(5, 6), (0, 1)]) == [(0, 1), (5, 6)]

    def test_of_nothing_is_nothing(self):
        assert union_intervals([]) == []


class TestIntersectIntervals:
    def test_returns_the_overlap(self):
        assert intersect_intervals([(0, 5)], [(3, 8)]) == [(3, 5)]

    def test_returns_nothing_when_disjoint(self):
        assert intersect_intervals([(0, 1)], [(2, 3)]) == []

    def test_touching_intervals_do_not_intersect(self):
        """A zero-width overlap is not a region."""
        assert intersect_intervals([(0, 1)], [(1, 2)]) == []

    def test_intersects_every_pair_and_unions_the_result(self):
        result = intersect_intervals([(0, 4), (6, 10)], [(2, 8)])
        assert result == [(2, 4), (6, 8)]

    def test_against_nothing_is_nothing(self):
        assert intersect_intervals([(0, 1)], []) == []


# ---------------------------------------------------------------------------
# utils.io_reload
# ---------------------------------------------------------------------------


class TestReadWithReload:
    def test_returns_the_value_on_first_success(self):
        assert read_with_reload(lambda p: f"read {p}", "file.csv") == (
            "read file.csv"
        )

    def test_retries_a_transient_failure_and_then_succeeds(self):
        attempts = []

        def flaky(path):
            attempts.append(path)
            if len(attempts) < 3:
                raise OSError("device not ready")
            return "ok"

        result = read_with_reload(
            flaky, "file.csv", base_delay_s=0.0, max_delay_s=0.0
        )

        assert result == "ok"
        assert len(attempts) == 3

    def test_reports_every_retry_to_the_callback(self):
        seen = []

        def flaky(path):
            if len(seen) < 2:
                raise OSError("nope")
            return "ok"

        read_with_reload(
            flaky,
            "file.csv",
            base_delay_s=0.0,
            max_delay_s=0.0,
            on_reload=lambda fp, attempt, exc: seen.append((fp, attempt)),
        )

        assert seen == [("file.csv", 1), ("file.csv", 2)]

    def test_gives_up_and_reraises_once_the_budget_is_spent(self):
        def always_fails(path):
            raise OSError("gone for good")

        with pytest.raises(OSError, match="gone for good"):
            read_with_reload(
                always_fails,
                "file.csv",
                max_wait_s=0.0,
                base_delay_s=0.0,
            )

    def test_does_not_retry_an_unexpected_exception_type(self):
        """Only transient I/O errors are worth retrying."""
        calls = []

        def bad_logic(path):
            calls.append(path)
            raise KeyError("a bug, not a flaky disk")

        with pytest.raises(KeyError):
            read_with_reload(bad_logic, "file.csv", base_delay_s=0.0)

        assert len(calls) == 1

    def test_text_round_trip(self, tmp_path):
        path = str(tmp_path / "note.txt")

        write_text_with_reload(path, "hello")

        assert read_text_with_reload(path) == "hello"


# ---------------------------------------------------------------------------
# core.experiment_layout
# ---------------------------------------------------------------------------


@pytest.fixture
def experiment(tmp_path):
    """A realistic experiment folder layout."""
    root = tmp_path / "250615MA40"
    (root / "Analysis" / "BaSiC_w01").mkdir(parents=True)
    (root / "Analysis" / "BaSiC_w02").mkdir(parents=True)
    (root / "Analysis" / "Segmentation1").mkdir(parents=True)
    (root / "250615MA40_p0001").mkdir()
    (root / "250615MA40_p0002").mkdir()
    (root / "images.csv").write_text("a,b\n")
    (root / "images_extra.csv").write_text("a,b\n")
    (root / "notes.txt").write_text("ignore me\n")

    position = root / "250615MA40_p0001"
    for t in (1, 2, 17):
        for channel in ("w00", "w01"):
            (position / f"250615MA40_p0001_t{t:05d}_{channel}.png").touch()
    return root


class TestFindAnalysisDir:
    def test_finds_the_analysis_folder(self, experiment):
        assert find_analysis_dir(str(experiment)).endswith("Analysis")

    def test_is_case_insensitive(self, tmp_path):
        (tmp_path / "analysis").mkdir()
        assert find_analysis_dir(str(tmp_path)) is not None

    @pytest.mark.parametrize("folder", [None, "", "/does/not/exist"])
    def test_returns_none_without_a_usable_folder(self, folder):
        assert find_analysis_dir(folder) is None

    def test_returns_none_when_there_is_no_analysis_folder(self, tmp_path):
        assert find_analysis_dir(str(tmp_path)) is None


class TestFindBasicFolders:
    def test_returns_sorted_basic_folder_names(self, experiment):
        assert find_basic_folders(str(experiment)) == [
            "BaSiC_w01",
            "BaSiC_w02",
        ]

    def test_ignores_other_analysis_subfolders(self, experiment):
        assert "Segmentation1" not in find_basic_folders(str(experiment))

    def test_returns_empty_without_an_analysis_folder(self, tmp_path):
        assert find_basic_folders(str(tmp_path)) == []


class TestFindImagesCsvs:
    def test_returns_sorted_images_csv_names(self, experiment):
        assert find_images_csvs(str(experiment)) == [
            "images.csv",
            "images_extra.csv",
        ]

    def test_ignores_other_files(self, experiment):
        assert "notes.txt" not in find_images_csvs(str(experiment))

    def test_returns_empty_for_a_missing_folder(self):
        assert find_images_csvs("/does/not/exist") == []


class TestFindExamplePositionFolder:
    def test_prefers_position_one(self, experiment):
        found = find_example_position_folder(str(experiment), "250615MA40")
        assert found.endswith("250615MA40_p0001")

    def test_falls_back_to_position_two(self, tmp_path):
        (tmp_path / "exp_p0002").mkdir()
        found = find_example_position_folder(str(tmp_path), "exp")
        assert found.endswith("exp_p0002")

    def test_falls_back_to_the_first_listed_folder(self, tmp_path):
        found = find_example_position_folder(
            str(tmp_path), "exp", ["/elsewhere/exp_p0099"]
        )
        assert found == "/elsewhere/exp_p0099"

    def test_returns_none_without_an_experiment_name(self, experiment):
        assert find_example_position_folder(str(experiment), None) is None

    def test_returns_none_when_nothing_matches(self, tmp_path):
        assert find_example_position_folder(str(tmp_path), "exp") is None


class TestDetectUniqueWChannels:
    def test_returns_the_channels_found_in_the_filenames(self, experiment):
        position = str(experiment / "250615MA40_p0001")
        assert detect_unique_w_channels(position) == ["w00", "w01"]

    def test_sorts_numerically_not_lexically(self, tmp_path):
        for channel in ("w01", "w02", "w10"):
            (tmp_path / f"img_t00001_{channel}.png").touch()
        assert detect_unique_w_channels(str(tmp_path)) == ["w01", "w02", "w10"]

    def test_synthesises_names_from_the_fallback_count(self, tmp_path):
        assert detect_unique_w_channels(str(tmp_path), 3) == [
            "w01",
            "w02",
            "w03",
        ]

    def test_returns_empty_when_there_is_nothing_to_go_on(self):
        assert detect_unique_w_channels(None, 0) == []


class TestDetectTimeMax:
    def test_returns_the_highest_time_point(self, experiment):
        position = str(experiment / "250615MA40_p0001")
        assert detect_time_max(position) == 17

    def test_returns_one_for_a_folder_with_no_timestamps(self, tmp_path):
        (tmp_path / "readme.txt").touch()
        assert detect_time_max(str(tmp_path)) == 1

    def test_returns_one_without_a_folder(self):
        assert detect_time_max(None) == 1

    def test_never_returns_less_than_one(self, tmp_path):
        (tmp_path / "img_t00000_w01.png").touch()
        assert detect_time_max(str(tmp_path)) == 1
