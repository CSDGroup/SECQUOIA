"""Tests for SECQUOIA BaSiC background correction.

It reads a darkfield image, a flatfield image and two per frame text
series off disk, and applies them. So these tests write a small
BaSiC folder into a temporary directory instead of mocking anything,
and the result is the same on every run.

``apply_basic_correction`` documents the two corrections it writes:

    NoRatioFlat(t) = (I(t) - D) / F - Base(t)
    RatioFlat(t)   = (I(t) - D) / (1 + (F - 1) * Amp(t)) - Base(t)

The darkfield and flatfield are constant, which keeps the arithmetic
checkable by hand and gives three cases where the answer is forced:

    Amp = 1  ->  1 + (F - 1) * 1 = F, so both corrections are the same
    Amp = 0  ->  the denominator is 1, so RatioFlat skips the flatfield
    F   = 1  ->  both reduce to (I - D) - Base, whatever Amp is

The images are the same artificial dataset as ``test_quantify.py``, shared
through ``conftest.py``:

- one channel, w01, over two time points
- a 3 x 3 region at [2:5, 2:5] holding 10 at t=0 and 20 at t=1
- mask 1 is that 3 x 3 region, area 9
- mask 2 is the 2 x 2 region nested inside it, area 4
"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile
from conftest import assert_column_values, make_fake_main_window

from SECQUOIA.core.basic_correction import (
    DataSetBaSiC,
    _prepare_basic_correction_inputs,
    _series_to_dense,
    apply_basic_correction,
)
from SECQUOIA.core.memmap_store import cleanup_memmaps
from SECQUOIA.core.quantification import quantify

RAW = (10.0, 20.0)
CENTRE = (3, 3)
BACKGROUND = (0, 0)
_SESSIONS: list = []


@pytest.fixture(autouse=True)
def release_memmaps():
    """Close the corrected stacks, as closing the window does in the app."""
    _SESSIONS.clear()
    yield
    while _SESSIONS:
        cleanup_memmaps(
            _SESSIONS.pop(),
            remove_viewer_image_layers=False,
            retries=2,
            sleep_s=0,
        )


def write_series(path, values, t_start: int = 1) -> None:
    """Write a ratioflat/basefluor file: a header, then ``t00001:1.0``."""
    lines = ["time:value"]
    for offset, value in enumerate(values):
        lines.append(f"t{t_start + offset:05d}:{value}")
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def basic_folder(tmp_path):
    """Factory writing a BaSiC folder for one position and channel."""
    counter = {"n": 0}

    def _write(
        *,
        dark: float = 0.0,
        flat: float = 1.0,
        base=(0.0, 0.0),
        amp=(1.0, 1.0),
        channel: str = "w01",
        exp_name: str = "fake_experiment",
        position: int = 1,
        shape=(10, 10),
        omit: tuple[str, ...] = (),
        field_shape=None,
    ) -> str:
        counter["n"] += 1
        root = tmp_path / f"BaSiC_{counter['n']}"
        position_dir = root / f"{exp_name}_p{position:04d}"
        position_dir.mkdir(parents=True, exist_ok=True)

        field_shape = shape if field_shape is None else field_shape
        if "darkfield" not in omit:
            tifffile.imwrite(
                position_dir / f"darkfield_{channel}.tif",
                np.full(field_shape, dark, dtype=np.float32),
            )
        if "flatfield" not in omit:
            tifffile.imwrite(
                position_dir / f"flatfield_{channel}.tif",
                np.full(field_shape, flat, dtype=np.float32),
            )
        if "amp" not in omit:
            write_series(position_dir / f"ratioflat_{channel}.txt", amp)
        if "base" not in omit:
            write_series(position_dir / f"basefluor_{channel}.txt", base)
        return str(root)

    return _write


@pytest.fixture
def basic_session(tmp_path, basic_folder):
    """Factory for a session with BaSiC configured the way the app does."""
    counter = {"n": 0}

    def _make(*, present=(True, True), n_frames: int = 2, **field_kwargs):
        counter["n"] += 1
        session = tmp_path / f"session_{counter['n']}"
        memmap_dir = session / "memmaps"
        memmap_dir.mkdir(parents=True)

        path_basic = basic_folder(**field_kwargs)

        basic = DataSetBaSiC(flag=False)
        basic.flag = True  # what ticking the checkbox does

        main_window = make_fake_main_window(session / "window", basic=basic)
        main_window.time_min_selected = 1
        main_window.time_max_selected = n_frames
        main_window.image_present = {"w01": np.asarray(present, dtype=bool)}
        main_window.background_correction_path = path_basic
        main_window._memmap_dir = str(memmap_dir)
        main_window.basic_max_workers = 1

        _prepare_basic_correction_inputs(main_window)
        _SESSIONS.append(main_window)
        return main_window

    return _make


def corrected(main_window) -> tuple[np.ndarray, np.ndarray]:
    """The (NoRatioFlat, RatioFlat) stacks for channel w01, as plain arrays."""
    return (
        np.asarray(main_window.corrected_images_noratioflat["w01"]),
        np.asarray(main_window.corrected_images_ratioflat["w01"]),
    )


class TestSeriesToDense:
    """Maps the sparse text-file timeline onto the dense image timeline."""

    def series(self, times, values) -> dict:
        return {
            "times": np.asarray(times, dtype=np.int32),
            "values": np.asarray(values, dtype=np.float32),
        }

    def test_places_each_value_at_its_frame(self):
        dense, _ = _series_to_dense(
            self.series([1, 2, 3], [5.0, 6.0, 7.0]), 1, 3, fill_value=0.0
        )

        assert dense.tolist() == [5.0, 6.0, 7.0]

    def test_rebases_onto_the_start_of_the_range(self):
        """t_file 5 is row 0 when the selection starts at 5."""
        dense, _ = _series_to_dense(
            self.series([5, 6], [1.0, 2.0]), 5, 6, fill_value=0.0
        )

        assert dense.tolist() == [1.0, 2.0]

    def test_fills_frames_with_no_entry(self):
        dense, _ = _series_to_dense(
            self.series([1, 3], [5.0, 7.0]), 1, 3, fill_value=-1.0
        )

        assert dense.tolist() == [5.0, -1.0, 7.0]

    def test_reports_which_frames_had_an_entry(self):
        _, has_value = _series_to_dense(
            self.series([1, 3], [5.0, 7.0]), 1, 3, fill_value=0.0
        )

        assert has_value.tolist() == [True, False, True]

    def test_drops_times_before_the_range(self):
        dense, has_value = _series_to_dense(
            self.series([0, 1], [9.0, 5.0]), 1, 2, fill_value=0.0
        )

        assert dense.tolist() == [5.0, 0.0]
        assert has_value.tolist() == [True, False]

    def test_drops_times_after_the_range(self):
        dense, has_value = _series_to_dense(
            self.series([1, 9], [5.0, 9.0]), 1, 2, fill_value=0.0
        )

        assert dense.tolist() == [5.0, 0.0]
        assert has_value.tolist() == [True, False]

    def test_no_series_is_all_fill_and_nothing_marked(self):
        dense, has_value = _series_to_dense(None, 1, 3, fill_value=1.0)

        assert dense.tolist() == [1.0, 1.0, 1.0]
        assert not has_value.any()

    def test_a_single_frame_range(self):
        dense, has_value = _series_to_dense(
            self.series([1], [4.0]), 1, 1, fill_value=0.0
        )

        assert dense.tolist() == [4.0]
        assert has_value.tolist() == [True]


class TestDataSetBaSiCLoading:
    def loaded(self, path_basic, *, t_range=(1, 2), channels=("w01",)):
        basic = DataSetBaSiC(flag=False)
        basic.flag = True
        basic.exp_name = "fake_experiment"
        basic.channels_valid = list(channels)
        basic.path_basic = path_basic
        basic.t_min, basic.t_max = t_range
        basic.update_all(1)
        return basic

    def test_loads_the_fields_keyed_by_channel_token(self, basic_folder):
        basic = self.loaded(basic_folder(dark=3.0, flat=4.0))

        assert list(basic.darkfield) == ["w01"]
        assert list(basic.flatfield) == ["w01"]
        assert basic.darkfield["w01"][0, 0] == 3.0
        assert basic.flatfield["w01"][0, 0] == 4.0

    def test_loads_the_amp_and_base_series(self, basic_folder):
        basic = self.loaded(basic_folder(amp=(0.5, 0.75), base=(1.0, 2.0)))

        assert basic.amp["w01"]["values"].tolist() == [0.5, 0.75]
        assert basic.base["w01"]["values"].tolist() == [1.0, 2.0]
        assert basic.amp["w01"]["times"].tolist() == [1, 2]

    def test_windows_the_series_to_the_selected_time_range(self, basic_folder):
        """Entries outside [t_min, t_max] are dropped at read time."""
        folder = basic_folder(amp=(1.0, 2.0, 3.0, 4.0), base=(0.0,) * 4)

        basic = self.loaded(folder, t_range=(2, 3))

        assert basic.amp["w01"]["times"].tolist() == [2, 3]
        assert basic.amp["w01"]["values"].tolist() == [2.0, 3.0]

    def test_ignores_channels_outside_the_valid_list(self, basic_folder):
        folder = basic_folder(channel="w02")

        basic = self.loaded(folder, channels=("w01",))

        assert basic.darkfield == {}

    def test_a_disabled_dataset_loads_nothing(self, basic_folder):
        """Every component stays None while the checkbox is off."""
        basic = DataSetBaSiC(flag=False)
        basic.exp_name = "fake_experiment"
        basic.channels_valid = ["w01"]
        basic.path_basic = basic_folder()
        basic.t_min, basic.t_max = 1, 2

        basic.update_all(1)

        assert basic.darkfield is None
        assert basic.flatfield is None
        assert basic.amp is None
        assert basic.base is None

    def test_a_missing_component_file_is_simply_absent(self, basic_folder):
        basic = self.loaded(basic_folder(omit=("flatfield",)))

        assert list(basic.darkfield) == ["w01"]
        assert basic.flatfield == {}


class TestDataSetBaSiCConfig:
    def test_add_t_range_swaps_a_reversed_pair(self):
        basic = DataSetBaSiC(flag=False)
        basic.flag = True

        basic.add_t_range((9, 2))

        assert (basic.t_min, basic.t_max) == (2, 9)

    def test_add_path_basic_rejects_a_folder_without_basic_in_its_name(
        self, tmp_path
    ):
        """The name is the only thing distinguishing a BaSiC folder."""
        other = tmp_path / "Segmentation1"
        other.mkdir()
        basic = DataSetBaSiC(flag=False)
        basic.flag = True

        basic.add_path_basic(str(other))

        assert basic.path_basic is None

    def test_add_path_basic_rejects_a_folder_that_does_not_exist(self):
        basic = DataSetBaSiC(flag=False)
        basic.flag = True

        basic.add_path_basic("/does/not/exist/BaSiC_w01")

        assert basic.path_basic is None

    def test_add_path_basic_accepts_a_real_basic_folder(self, basic_folder):
        basic = DataSetBaSiC(flag=False)
        basic.flag = True
        folder = basic_folder()

        basic.add_path_basic(folder)

        assert basic.path_basic == folder

    def test_the_setters_are_ignored_while_correction_is_off(self, tmp_path):
        basic = DataSetBaSiC(flag=False)

        basic.add_exp_name("exp")
        basic.add_t_range((1, 5))
        basic.add_channels_valid(["w01"])
        basic.update_position(7)

        assert basic.exp_name is None
        assert basic.t_min is None
        assert basic.channels_valid is None
        assert basic.position is None


class TestCorrectionArithmetic:
    def test_neutral_fields_return_the_image_unchanged(self, basic_session):
        """D=0, F=1, Base=0 and Amp=1 cancel out, so nothing may change."""
        main_window = basic_session(
            dark=0.0, flat=1.0, base=(0.0, 0.0), amp=(1.0, 1.0)
        )

        apply_basic_correction(main_window)

        norf, ratio = corrected(main_window)
        raw = main_window.images["w01"]
        np.testing.assert_allclose(norf, raw)
        np.testing.assert_allclose(ratio, raw)

    def test_an_amp_of_one_makes_the_two_variants_identical(
        self, basic_session
    ):
        """1 + (F - 1) * 1 is F, so both formulas become the same one.

        No expected value is written down here. The point is that two
        separate code paths have to agree wherever the algebra says so.
        """
        main_window = basic_session(
            dark=2.0, flat=4.0, base=(1.0, 3.0), amp=(1.0, 1.0)
        )

        apply_basic_correction(main_window)

        norf, ratio = corrected(main_window)
        np.testing.assert_allclose(norf, ratio)

    def test_an_amp_of_zero_bypasses_the_flatfield(self, basic_session):
        """With Amp at 0 the denominator is 1, so the flatfield drops out."""
        main_window = basic_session(
            dark=2.0, flat=2.0, base=(1.0, 1.0), amp=(0.0, 0.0)
        )

        apply_basic_correction(main_window)

        _norf, ratio = corrected(main_window)
        assert ratio[0][CENTRE] == pytest.approx(7.0)
        assert ratio[1][CENTRE] == pytest.approx(17.0)

    def test_the_two_variants_differ_once_amp_is_not_one(self, basic_session):
        """Both stacks are written in the same loop, so they could be crossed."""
        main_window = basic_session(
            dark=2.0, flat=2.0, base=(1.0, 1.0), amp=(0.0, 0.0)
        )

        apply_basic_correction(main_window)

        norf, ratio = corrected(main_window)
        assert norf[0][CENTRE] == pytest.approx(3.0)
        assert ratio[0][CENTRE] == pytest.approx(7.0)
        assert not np.allclose(norf, ratio)

    def test_the_hand_derived_values_for_a_real_field(self, basic_session):
        """D=2, F=2, Base=1: (10 - 2) / 2 - 1 = 3, (20 - 2) / 2 - 1 = 8."""
        main_window = basic_session(
            dark=2.0, flat=2.0, base=(1.0, 1.0), amp=(1.0, 1.0)
        )

        apply_basic_correction(main_window)

        norf, _ratio = corrected(main_window)
        assert norf[0][CENTRE] == pytest.approx(3.0)
        assert norf[1][CENTRE] == pytest.approx(8.0)

    def test_a_flat_field_of_one_reduces_both_to_dark_and_base(
        self, basic_session
    ):
        """F=1 makes (F - 1) zero, so Amp cannot matter."""
        main_window = basic_session(
            dark=2.0, flat=1.0, base=(1.0, 1.0), amp=(0.25, 0.75)
        )

        apply_basic_correction(main_window)

        norf, ratio = corrected(main_window)
        assert norf[0][CENTRE] == pytest.approx(7.0)
        assert ratio[0][CENTRE] == pytest.approx(7.0)
        assert norf[1][CENTRE] == pytest.approx(17.0)
        assert ratio[1][CENTRE] == pytest.approx(17.0)

    def test_the_baseline_is_applied_per_frame(self, basic_session):
        """Base is a series, not a constant; frame 1 must use its own value."""
        main_window = basic_session(
            dark=0.0, flat=1.0, base=(1.0, 5.0), amp=(1.0, 1.0)
        )

        apply_basic_correction(main_window)

        norf, _ratio = corrected(main_window)
        assert norf[0][CENTRE] == pytest.approx(RAW[0] - 1.0)
        assert norf[1][CENTRE] == pytest.approx(RAW[1] - 5.0)

    def test_the_correction_applies_outside_the_masks_too(self, basic_session):
        """Background is corrected too: raw 0 gives (0 - 2) / 2 - 1 = -2."""
        main_window = basic_session(
            dark=2.0, flat=2.0, base=(1.0, 1.0), amp=(1.0, 1.0)
        )

        apply_basic_correction(main_window)

        norf, _ratio = corrected(main_window)
        assert norf[0][BACKGROUND] == pytest.approx(-2.0)

    def test_a_zero_in_the_flatfield_leaves_that_pixel_at_zero(
        self, basic_session
    ):
        """Division is masked by ``flat != 0``, so it must not produce inf."""
        main_window = basic_session(
            dark=0.0, flat=0.0, base=(0.0, 0.0), amp=(1.0, 1.0)
        )

        apply_basic_correction(main_window)

        norf, _ratio = corrected(main_window)
        assert np.isfinite(norf).all()
        assert norf[0][CENTRE] == pytest.approx(0.0)


class TestCorrectionGuards:
    def test_a_disabled_correction_returns_without_touching_anything(
        self, basic_session
    ):
        main_window = basic_session()
        main_window.basic.flag = False
        main_window.corrected_images_ratioflat = "untouched"

        apply_basic_correction(main_window)

        assert main_window.corrected_images_ratioflat == "untouched"

    def test_a_channel_without_a_flatfield_is_skipped(self, basic_session):
        main_window = basic_session(omit=("flatfield",))

        apply_basic_correction(main_window)

        assert main_window.corrected_images_ratioflat["w01"] is None
        assert main_window.corrected_images_noratioflat["w01"] is None

    def test_a_channel_without_an_amp_series_is_skipped(self, basic_session):
        main_window = basic_session(omit=("amp",))

        apply_basic_correction(main_window)

        assert main_window.corrected_images_noratioflat["w01"] is None

    def test_a_field_of_the_wrong_shape_is_skipped(self, basic_session):
        """A field from a different camera crop must not be broadcast in."""
        main_window = basic_session(field_shape=(4, 4))

        apply_basic_correction(main_window)

        assert main_window.corrected_images_noratioflat["w01"] is None

    def test_a_channel_with_no_acquired_frames_is_skipped(self, basic_session):
        main_window = basic_session(present=(False, False))

        apply_basic_correction(main_window)

        assert main_window.corrected_images_noratioflat["w01"] is None

    def test_the_valid_frames_are_recorded(self, basic_session):
        main_window = basic_session(present=(True, False))

        apply_basic_correction(main_window)

        assert main_window.basic_valid_frames["w01"].tolist() == [True, False]

    def test_valid_frames_intersect_the_images_with_the_series(
        self, basic_session
    ):
        """A frame with no base/amp entry is not valid even if acquired."""
        main_window = basic_session(
            base=(0.0,), amp=(1.0,), present=(True, True)
        )

        apply_basic_correction(main_window)

        assert main_window.basic_valid_frames["w01"].tolist() == [True, False]

    def test_an_unacquired_frame_is_left_as_zeros(self, basic_session):
        """Only valid rows are written; the memmap starts zero filled."""
        main_window = basic_session(
            dark=0.0,
            flat=1.0,
            base=(0.0, 0.0),
            amp=(1.0, 1.0),
            present=(True, False),
        )

        apply_basic_correction(main_window)

        norf, _ratio = corrected(main_window)
        assert norf[0][CENTRE] == pytest.approx(RAW[0])
        assert np.count_nonzero(norf[1]) == 0

    def test_the_output_files_are_recorded_for_cleanup(self, basic_session):
        """The memmaps are deleted by name later, so the paths must be kept."""
        main_window = basic_session()

        apply_basic_correction(main_window)

        (ratio_path,) = main_window._ratioflat_memmap_files
        (norf_path,) = main_window._noratioflat_memmap_files
        assert "RatioFlat" in ratio_path
        assert "NoRatioFlat" in norf_path
        assert ratio_path.endswith("_t00001-00002.dat")

    def test_a_skipped_channel_still_records_a_placeholder(
        self, basic_session
    ):
        """The file lists stay aligned with the channel list."""
        main_window = basic_session(omit=("flatfield",))

        apply_basic_correction(main_window)

        assert main_window._ratioflat_memmap_files == [None]
        assert main_window._noratioflat_memmap_files == [None]


# Concurrency and progress
def test_every_frame_is_correct_under_concurrent_workers(basic_session):
    n_frames = 24
    main_window = basic_session(
        dark=0.0,
        flat=1.0,
        base=tuple(float(t) for t in range(n_frames)),
        amp=(1.0,) * n_frames,
        present=(True,) * n_frames,
        n_frames=n_frames,
    )
    stack = np.zeros((n_frames, 10, 10), dtype=np.float32)
    for t in range(n_frames):
        stack[t, 2:5, 2:5] = 100.0 + t
    main_window.images["w01"] = stack
    main_window.basic_max_workers = 8

    apply_basic_correction(main_window)

    norf, _ratio = corrected(main_window)
    for t in range(n_frames):
        assert norf[t][CENTRE] == pytest.approx(100.0), f"frame {t} is wrong"


@pytest.mark.gui
def test_the_progress_callback_runs_to_completion(basic_session, qapp):
    """The callback calls ``QApplication.processEvents``, so it needs Qt."""
    calls = []
    main_window = basic_session()

    apply_basic_correction(
        main_window, progress_cb=lambda frac, msg: calls.append((frac, msg))
    )

    assert calls
    assert calls[-1][0] == 1.0


def quantified(main_window, no_progress):
    """Correct, then measure, and hand back the track frame sorted by time."""
    apply_basic_correction(main_window)
    quantify(main_window, progress_cb=no_progress, max_pixel_distance=5)
    return main_window.track_df.sort_values("t").reset_index(drop=True)


class TestBasicReachesQuantify:
    """The corrected stacks have to arrive in the measured columns."""

    def test_the_basic_column_families_are_written(
        self, basic_session, no_progress
    ):
        df = quantified(basic_session(), no_progress)

        assert "MeanBaSiCBgCorrectedRatioFlatCh01M1" in df.columns
        assert "MeanBaSiCBgCorrectedNoRatioFlatCh01M1" in df.columns
        assert "MeanNoBgCorrectedCh01M1" in df.columns

    def test_no_basic_columns_when_the_correction_is_off(
        self, basic_session, no_progress
    ):
        main_window = basic_session()
        main_window.basic.flag = False

        df = quantified(main_window, no_progress)

        assert not [c for c in df.columns if "BaSiC" in c]
        assert_column_values(df, "MeanNoBgCorrectedCh01M1", [10, 20])

    def test_neutral_fields_measure_the_same_as_the_raw_channel(
        self, basic_session, no_progress
    ):
        """All three variants must agree when the correction changes nothing.

        That ties the BaSiC columns to the raw numbers ``test_quantify.py``
        already pins, without adding a single new constant here.
        """
        df = quantified(
            basic_session(dark=0.0, flat=1.0, base=(0.0, 0.0), amp=(1.0, 1.0)),
            no_progress,
        )

        for prefix in (
            "MeanNoBgCorrected",
            "MeanBaSiCBgCorrectedRatioFlat",
            "MeanBaSiCBgCorrectedNoRatioFlat",
        ):
            assert_column_values(df, f"{prefix}Ch01M1", [10, 20])

    def test_a_real_field_is_measured_through_both_masks(
        self, basic_session, no_progress
    ):
        """Corrected pixels are 3 then 8; mask 1 is 9 px and mask 2 is 4.

        So the sums are 27 then 72 for mask 1, and 12 then 32 for mask 2.
        """
        df = quantified(
            basic_session(dark=2.0, flat=2.0, base=(1.0, 1.0), amp=(1.0, 1.0)),
            no_progress,
        )

        assert_column_values(
            df, "MeanBaSiCBgCorrectedNoRatioFlatCh01M1", [3, 8]
        )
        assert_column_values(
            df, "SumBaSiCBgCorrectedNoRatioFlatCh01M1", [27, 72]
        )
        assert_column_values(
            df, "MeanBaSiCBgCorrectedNoRatioFlatCh01M2", [3, 8]
        )
        assert_column_values(
            df, "SumBaSiCBgCorrectedNoRatioFlatCh01M2", [12, 32]
        )
        # The uncorrected channel is untouched by any of this.
        assert_column_values(df, "MeanNoBgCorrectedCh01M1", [10, 20])

    def test_the_two_variants_land_in_different_columns(
        self, basic_session, no_progress
    ):
        """Two formulas, two columns, two sets of numbers, all the way out.

        With Amp at 0 RatioFlat skips the flatfield, so the two columns have
        to disagree in the measured output and not just in the arrays.
        """
        df = quantified(
            basic_session(dark=2.0, flat=2.0, base=(1.0, 1.0), amp=(0.0, 0.0)),
            no_progress,
        )

        assert_column_values(
            df, "MeanBaSiCBgCorrectedNoRatioFlatCh01M1", [3, 8]
        )
        assert_column_values(
            df, "MeanBaSiCBgCorrectedRatioFlatCh01M1", [7, 17]
        )
