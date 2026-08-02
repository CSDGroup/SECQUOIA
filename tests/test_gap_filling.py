"""Tests for SECQUOIA gap filling.

The filler invents a row for every frame a track should have been measured
in but was not. The test lineage is one identification over six frames:
track 1 runs to frame 2 and divides into tracks 2 and 3. Frame 1 and frame 4
are left out on purpose, so there is a gap on either side of the division.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from SECQUOIA.core.gap_filling import (
    SYNTHETIC_CELLFATE,
    CoordinatePolicy,
    GapFillContext,
    child_tracks,
    ensure_track_schema,
    extract_track_id,
    fill_missing_frames,
    mask_absent_measurements,
    pair_mate,
    parent_track,
    representative_xy,
    sibling_group,
    sort_columns,
    track_windows,
)


# Binary lineage arithmetic
class TestParentTrack:
    @pytest.mark.parametrize(
        ("track", "expected"),
        [(2, 1), (3, 1), (4, 2), (5, 2), (6, 3), (7, 3), (100, 50)],
    )
    def test_halves_the_track_number(self, track, expected):
        assert parent_track(track) == expected

    def test_the_founder_has_no_parent(self):
        assert parent_track(1) is None

    @pytest.mark.parametrize("track", [None, np.nan, "not a number", 0])
    def test_returns_none_for_unusable_input(self, track):
        assert parent_track(track) is None

    def test_accepts_a_numeric_string(self):
        assert parent_track("4") == 2

    def test_accepts_a_float(self):
        assert parent_track(4.0) == 2


class TestChildTracks:
    @pytest.mark.parametrize(
        ("track", "expected"), [(1, (2, 3)), (2, (4, 5)), (3, (6, 7))]
    )
    def test_doubles_into_a_daughter_pair(self, track, expected):
        assert child_tracks(track) == expected

    @pytest.mark.parametrize("track", [None, np.nan, 0, "x"])
    def test_returns_nothing_for_unusable_input(self, track):
        assert child_tracks(track) == ()


class TestSiblingGroup:
    @pytest.mark.parametrize(
        ("track", "expected"),
        [(2, (2, 3)), (3, (2, 3)), (4, (4, 5)), (5, (4, 5)), (6, (6, 7))],
    )
    def test_pairs_each_track_with_its_sister(self, track, expected):
        assert sibling_group(track) == expected

    def test_the_founder_is_alone(self):
        assert sibling_group(1) == (1,)


class TestPairMate:
    @pytest.mark.parametrize(
        ("track", "expected"), [(2, 3), (3, 2), (4, 5), (5, 4)]
    )
    def test_returns_the_sister(self, track, expected):
        assert pair_mate(track) == expected

    def test_the_founder_has_no_sister(self):
        assert pair_mate(1) is None

    @pytest.mark.parametrize("track", [None, np.nan, 0])
    def test_returns_none_for_unusable_input(self, track):
        assert pair_mate(track) is None


# Fill windows
class TestTrackWindows:
    FRAMES = range(6)  # frames 0..5

    def test_a_parent_stops_the_frame_before_its_daughters(self):
        spans = {1: (0, 2), 2: (3, 5), 3: (3, 5)}

        windows = track_windows(spans, self.FRAMES)

        assert list(windows[1]) == [0, 1, 2]

    def test_daughters_run_to_the_end_of_the_range(self):
        spans = {1: (0, 2), 2: (3, 5), 3: (3, 5)}

        windows = track_windows(spans, self.FRAMES)

        assert list(windows[2]) == [3, 4, 5]
        assert list(windows[3]) == [3, 4, 5]

    def test_the_founder_is_extended_back_to_the_range_start(self):
        """Track 1 was only measured from frame 2, but the range starts at 0."""
        spans = {1: (2, 5)}

        windows = track_windows(spans, self.FRAMES)

        assert list(windows[1]) == [0, 1, 2, 3, 4, 5]

    def test_a_late_detected_daughter_is_born_with_its_sister(self):
        """This is the point of the sister rule.

        Daughter 3 was first measured at frame 5, but its sister 2 appeared at
        frame 3, so the division happened at 3 and daughter 3 gets rows back
        to there.
        """
        spans = {1: (0, 2), 2: (3, 5), 3: (5, 5)}

        windows = track_windows(spans, self.FRAMES)

        assert list(windows[3]) == [3, 4, 5]

    def test_a_track_whose_window_would_be_empty_is_dropped(self):
        """A parent whose daughters appear at frame 0 has nowhere to live."""
        spans = {1: (0, 0), 2: (0, 5), 3: (0, 5)}

        windows = track_windows(spans, self.FRAMES)

        assert 1 not in windows

    def test_windows_are_clipped_to_the_frame_range(self):
        spans = {1: (0, 99)}

        windows = track_windows(spans, range(3))

        assert list(windows[1]) == [0, 1, 2]

    def test_no_spans_means_no_windows(self):
        assert track_windows({}, self.FRAMES) == {}

    def test_no_frames_means_no_windows(self):
        assert track_windows({1: (0, 2)}, range(0)) == {}

    def test_a_dataframe_without_tracknumbers_uses_a_single_none_track(self):
        """``_track_spans`` keys on ``None`` when there is no TrackNumber."""
        windows = track_windows({None: (1, 3)}, self.FRAMES)

        assert list(windows[None]) == [0, 1, 2, 3, 4, 5]


# Small helpers
class TestEnsureTrackSchema:
    def test_adds_every_required_column(self):
        out = ensure_track_schema(pd.DataFrame({"t": [0, 1]}))

        for col in ("Position", "Identification", "XMorphology", "active"):
            assert col in out.columns

    def test_leaves_existing_values_alone(self):
        df = pd.DataFrame({"t": [0], "Position": [7], "active": [0]})

        out = ensure_track_schema(df)

        assert out["Position"].tolist() == [7]
        assert out["active"].tolist() == [0]

    def test_does_not_mutate_the_input(self):
        df = pd.DataFrame({"t": [0, 1]})

        ensure_track_schema(df)

        assert list(df.columns) == ["t"]

    def test_nullable_columns_get_their_declared_dtype(self):
        out = ensure_track_schema(pd.DataFrame({"t": [0, 1]}))

        assert out["track_id"].dtype == "Int64"
        assert out["track_id"].isna().all()
        assert out["Calculated_Time"].dtype == float


class TestRepresentativeXY:
    def test_returns_the_median_of_usable_coordinates(self):
        rows = pd.DataFrame(
            {"XMorphology": [10.0, 20.0, 30.0], "YMorphology": [1.0, 2.0, 3.0]}
        )

        assert representative_xy(rows) == (20.0, 2.0)

    def test_ignores_zero_coordinates(self):
        """(0, 0) is the sentinel for "no position", not a real measurement."""
        rows = pd.DataFrame(
            {"XMorphology": [0.0, 20.0, 0.0], "YMorphology": [0.0, 2.0, 0.0]}
        )

        assert representative_xy(rows) == (20.0, 2.0)

    def test_returns_none_when_nothing_is_usable(self):
        rows = pd.DataFrame(
            {"XMorphology": [0.0, np.nan], "YMorphology": [0.0, np.nan]}
        )

        assert representative_xy(rows) is None

    def test_returns_none_for_an_empty_frame(self):
        assert representative_xy(pd.DataFrame()) is None


class TestExtractTrackId:
    @pytest.mark.parametrize(
        ("ident", "expected"),
        [
            ("250615MA40-p0002-001", 1),
            ("250615MA40-p0002-42", 42),
            ("exp-p0001-7", 7),
        ],
    )
    def test_reads_the_numeric_tail(self, ident, expected):
        assert extract_track_id(ident) == expected

    @pytest.mark.parametrize("ident", ["no-tail-here-x", "plain", ""])
    def test_returns_na_without_a_numeric_tail(self, ident):
        assert pd.isna(extract_track_id(ident))

    def test_returns_na_for_a_missing_identification(self):
        assert pd.isna(extract_track_id(np.nan))


class TestSortColumns:
    def test_includes_tracknumber_when_present(self):
        assert sort_columns(True) == [
            "Position",
            "Identification",
            "t",
            "TrackNumber",
        ]

    def test_omits_tracknumber_when_absent(self):
        assert sort_columns(False) == ["Position", "Identification", "t"]


# GapFillContext
class TestGapFillContext:
    def test_calculated_time_converts_frames_to_minutes(self):
        ctx = GapFillContext(
            position=1, frames=range(6), time_interval_s=180.0
        )

        assert ctx.calculated_time(0) == 0.0
        assert ctx.calculated_time(2) == 6.0  # 2 frames x 3 minutes

    def test_calculated_time_is_zero_without_an_interval(self):
        ctx = GapFillContext(position=1, frames=range(6))

        assert ctx.calculated_time(5) == 0.0

    def test_presence_flags_are_looked_up_by_channel_suffix(self):
        """Column ``...Ch01M1`` carries suffix "01", which maps to channel w01."""
        ctx = GapFillContext(
            position=1,
            frames=range(3),
            channels=("w00", "w01"),
            presence={"w00": [1, 1, 1], "w01": [1, 0, 1]},
        )

        assert list(ctx.presence_flags("01")) == [1, 0, 1]

    def test_presence_flags_are_none_for_an_unknown_channel(self):
        ctx = GapFillContext(
            position=1, frames=range(3), channels=("w00",), presence={}
        )

        assert ctx.presence_flags("99") is None

    def test_presence_flags_are_none_for_an_empty_flag_list(self):
        ctx = GapFillContext(
            position=1,
            frames=range(3),
            channels=("w01",),
            presence={"w01": []},
        )

        assert ctx.presence_flags("01") is None

    def test_frame_is_absent_reads_the_zero_based_flag(self):
        ctx = GapFillContext(
            position=1,
            frames=range(3),
            channels=("w01",),
            presence={"w01": [True, False, True]},
        )

        assert ctx.frame_is_absent("01", 0) is False
        assert ctx.frame_is_absent("01", 1) is True
        assert ctx.frame_is_absent("01", 2) is False

    def test_frame_is_absent_falls_back_to_one_based_indexing(self):
        """Frame numbers sometimes start at 1, so t == len(flags) is allowed."""
        ctx = GapFillContext(
            position=1,
            frames=range(3),
            channels=("w01",),
            presence={"w01": [True, False, True]},
        )

        assert ctx.frame_is_absent("01", 3) is False

    def test_a_frame_is_never_absent_without_evidence(self):
        """No presence information must never be read as "not acquired"."""
        ctx = GapFillContext(position=1, frames=range(3))

        assert ctx.frame_is_absent("01", 0) is False

    def test_a_frame_beyond_every_index_is_not_absent(self):
        ctx = GapFillContext(
            position=1,
            frames=range(3),
            channels=("w01",),
            presence={"w01": [True]},
        )

        assert ctx.frame_is_absent("01", 99) is False


# End to end filling
def gappy_frame() -> pd.DataFrame:
    """One lineage that divides at frame 3, with frame 1 and frame 4 missing.

    Measured frames:

        TrackNumber 1   0, 2       (1 is missing)
        TrackNumber 2   3, 5       (4 is missing)
        TrackNumber 3   3, 4, 5
    """
    rows = [
        (1, 0),
        (1, 2),
        (2, 3),
        (2, 5),
        (3, 3),
        (3, 4),
        (3, 5),
    ]
    return pd.DataFrame(
        {
            "Position": [2] * len(rows),
            "Identification": ["250615MA40-p0002-001"] * len(rows),
            "TrackNumber": [r[0] for r in rows],
            "t": [r[1] for r in rows],
            "XMorphology": [100.0] * len(rows),
            "YMorphology": [200.0] * len(rows),
            "active": [1] * len(rows),
            "Cellfate": ["Healthy"] * len(rows),
            "AreaMorphologyM1": [50.0] * len(rows),
            "MeanNoBgCorrectedCh01M1": [1000.0] * len(rows),
        }
    )


@pytest.fixture
def ctx() -> GapFillContext:
    return GapFillContext(position=2, frames=range(6), time_interval_s=180.0)


class TestFillMissingFrames:
    def test_fills_the_missing_frames_of_every_track(self, ctx):
        out = fill_missing_frames(gappy_frame(), ctx)

        frames = out.groupby("TrackNumber")["t"].apply(list)
        assert frames[1] == [0, 1, 2]
        assert frames[2] == [3, 4, 5]
        assert frames[3] == [3, 4, 5]

    def test_does_not_extend_a_parent_over_its_daughters(self, ctx):
        out = fill_missing_frames(gappy_frame(), ctx)

        parent = out[out["TrackNumber"] == 1]
        assert parent["t"].max() == 2

    def test_marks_synthetic_rows_with_the_synthetic_cellfate(self, ctx):
        out = fill_missing_frames(gappy_frame(), ctx)

        invented = out[(out["TrackNumber"] == 1) & (out["t"] == 1)]
        assert invented["Cellfate"].tolist() == [SYNTHETIC_CELLFATE]

    def test_synthetic_rows_are_active(self, ctx):
        out = fill_missing_frames(gappy_frame(), ctx)

        invented = out[(out["TrackNumber"] == 2) & (out["t"] == 4)]
        assert invented["active"].tolist() == [1]

    def test_synthetic_rows_carry_the_calculated_time(self, ctx):
        out = fill_missing_frames(gappy_frame(), ctx)

        invented = out[(out["TrackNumber"] == 1) & (out["t"] == 1)]
        assert invented["Calculated_Time"].tolist() == [3.0]

    def test_synthetic_rows_take_the_track_id_from_the_identification(
        self, ctx
    ):
        out = fill_missing_frames(gappy_frame(), ctx)

        invented = out[(out["TrackNumber"] == 1) & (out["t"] == 1)]
        assert invented["track_id"].tolist() == [1]

    def test_measurements_default_to_zero_not_to_a_copied_value(self, ctx):
        """An invented row must never look like it was measured."""
        out = fill_missing_frames(gappy_frame(), ctx)

        invented = out[(out["TrackNumber"] == 1) & (out["t"] == 1)]
        assert invented["AreaMorphologyM1"].tolist() == [0]

    def test_coordinates_default_to_the_sentinel(self, ctx):
        out = fill_missing_frames(gappy_frame(), ctx)

        invented = out[(out["TrackNumber"] == 1) & (out["t"] == 1)]
        assert invented["XMorphology"].tolist() == [0]
        assert invented["YMorphology"].tolist() == [0]

    def test_the_median_policy_places_rows_on_the_track(self):
        ctx = GapFillContext(
            position=2,
            frames=range(6),
            coordinates=CoordinatePolicy.MEDIAN,
        )

        out = fill_missing_frames(gappy_frame(), ctx)

        invented = out[(out["TrackNumber"] == 1) & (out["t"] == 1)]
        assert invented["XMorphology"].tolist() == [100.0]
        assert invented["YMorphology"].tolist() == [200.0]

    def test_existing_rows_are_never_overwritten(self, ctx):
        out = fill_missing_frames(gappy_frame(), ctx)

        measured = out[(out["TrackNumber"] == 1) & (out["t"] == 0)]
        assert measured["AreaMorphologyM1"].tolist() == [50.0]
        assert measured["XMorphology"].tolist() == [100.0]

    def test_the_result_is_sorted_canonically(self, ctx):
        out = fill_missing_frames(gappy_frame(), ctx)

        expected = out.sort_values(
            ["Position", "Identification", "t", "TrackNumber"],
            kind="mergesort",
        ).reset_index(drop=True)
        pd.testing.assert_frame_equal(out, expected)

    def test_rows_of_other_positions_are_left_untouched(self, ctx):
        df = gappy_frame()
        other = df.iloc[[0]].copy()
        other["Position"] = 99
        df = pd.concat([df, other], ignore_index=True)

        out = fill_missing_frames(df, ctx)

        assert (out["Position"] == 99).sum() == 1

    def test_a_position_with_no_rows_changes_nothing(self):
        df = gappy_frame()

        out = fill_missing_frames(
            df, GapFillContext(position=404, frames=range(6))
        )

        assert len(out) == len(df)

    def test_an_empty_frame_is_returned_as_is(self, ctx):
        empty = pd.DataFrame()

        assert fill_missing_frames(empty, ctx).empty

    def test_none_becomes_an_empty_frame(self, ctx):
        assert fill_missing_frames(None, ctx).empty

    def test_rows_with_no_identification_are_skipped(self, ctx):
        df = gappy_frame()
        df.loc[df.index[0], "Identification"] = np.nan

        out = fill_missing_frames(df, ctx)

        assert out["Identification"].isna().sum() == 1

    def test_filling_is_idempotent(self, ctx):
        once = fill_missing_frames(gappy_frame(), ctx)

        twice = fill_missing_frames(once, ctx)

        pd.testing.assert_frame_equal(once, twice)


class TestMaskAbsentMeasurements:
    def test_blanks_intensities_on_frames_that_were_not_acquired(self):
        """Zero must mean "measured zero", so unacquired frames become NaN."""
        ctx = GapFillContext(
            position=2,
            frames=range(3),
            channels=("w01",),
            presence={"w01": [True, False, True]},
        )
        add_df = pd.DataFrame(
            {
                "t": [0, 1, 2],
                "MeanNoBgCorrectedCh01M1": [0.0, 0.0, 0.0],
                "AreaMorphologyM1": [0.0, 0.0, 0.0],
            }
        )

        out = mask_absent_measurements(add_df, ctx)

        assert out["MeanNoBgCorrectedCh01M1"].isna().tolist() == [
            False,
            True,
            False,
        ]

    def test_leaves_non_intensity_columns_alone(self):
        """Only ``Ch<n>M<n>`` columns depend on a channel being acquired."""
        ctx = GapFillContext(
            position=2,
            frames=range(3),
            channels=("w01",),
            presence={"w01": [True, False, True]},
        )
        add_df = pd.DataFrame(
            {
                "t": [0, 1, 2],
                "MeanNoBgCorrectedCh01M1": [0.0, 0.0, 0.0],
                "AreaMorphologyM1": [1.0, 2.0, 3.0],
            }
        )

        out = mask_absent_measurements(add_df, ctx)

        assert out["AreaMorphologyM1"].tolist() == [1.0, 2.0, 3.0]

    def test_does_nothing_without_presence_information(self):
        ctx = GapFillContext(position=2, frames=range(3))
        add_df = pd.DataFrame(
            {"t": [0, 1], "MeanNoBgCorrectedCh01M1": [0.0, 0.0]}
        )

        out = mask_absent_measurements(add_df, ctx)

        assert out["MeanNoBgCorrectedCh01M1"].notna().all()

    def test_does_nothing_when_masking_is_switched_off(self):
        ctx = GapFillContext(
            position=2,
            frames=range(3),
            channels=("w01",),
            presence={"w01": [True, False]},
            mask_absent_frames=False,
        )
        add_df = pd.DataFrame(
            {"t": [0, 1], "MeanNoBgCorrectedCh01M1": [0.0, 0.0]}
        )

        out = mask_absent_measurements(add_df, ctx)

        assert out["MeanNoBgCorrectedCh01M1"].notna().all()

    def test_an_empty_frame_passes_through(self):
        ctx = GapFillContext(
            position=2,
            frames=range(3),
            channels=("w01",),
            presence={"w01": [True]},
        )

        assert mask_absent_measurements(pd.DataFrame(), ctx).empty

    def test_fill_missing_frames_applies_the_mask(self):
        """The end-to-end path must reach the masking step."""
        ctx = GapFillContext(
            position=2,
            frames=range(6),
            channels=("w01",),
            presence={"w01": [True, False, True, True, True, True]},
        )

        out = fill_missing_frames(gappy_frame(), ctx)

        invented = out[(out["TrackNumber"] == 1) & (out["t"] == 1)]
        assert invented["MeanNoBgCorrectedCh01M1"].isna().all()
