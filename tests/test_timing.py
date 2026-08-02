"""Tests for SECQUOIA time point handling.

Two ways of counting meet here: file names count time points from 1 and the
dataframe counts from 0. The data is file names of the form
exp_p0001_t00001_w01.png, plus a small imported real time table covering two
positions, two frames and two channels.
"""

from __future__ import annotations

import pandas as pd
import pytest

from SECQUOIA.core.gap_filling import GapFillContext
from SECQUOIA.utils.timing import (
    RT_PREFIX,
    apply_realtime,
    build_realtime_lookup,
    calculate_time,
    current_t_range,
    filename_t_file_number,
    realtime_columns,
)


# Filename parsing
class TestFilenameTFileNumber:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("exp_p0001_t00001_w01.png", 1),
            ("exp_p0001_t00042_w01.png", 42),
            ("exp_p0001_t99999_w01.png", 99999),
            ("/a/long/path/exp_p0001_t00007_w02.tif", 7),
        ],
    )
    def test_reads_the_five_digit_time_token(self, name, expected):
        assert filename_t_file_number(name) == expected

    @pytest.mark.parametrize(
        "name",
        [
            "exp_p0001_w01.png",
            "exp_t1_w01.png",
            "exp_t000001_w01.png",
            "readme.txt",
        ],
    )
    def test_returns_none_without_a_five_digit_token(self, name):
        assert filename_t_file_number(name) is None

    def test_looks_only_at_the_basename(self):
        """A time token in a parent directory must not be picked up."""
        assert filename_t_file_number("/data_t00005_/image.png") is None


# Range conversion
class TestCurrentTRange:
    def test_converts_one_based_filenames_to_zero_based_indices(
        self, fake_main_window
    ):
        main_window = fake_main_window(
            time_min_selected=1, time_max_selected=10
        )

        assert current_t_range(main_window) == (1, 10, 0, 9)

    def test_defaults_to_a_single_first_time_point(self, fake_main_window):
        assert current_t_range(fake_main_window()) == (1, 1, 0, 0)

    def test_a_missing_max_follows_the_min(self, fake_main_window):
        main_window = fake_main_window(time_min_selected=5)

        assert current_t_range(main_window) == (5, 5, 4, 4)

    def test_an_inverted_range_collapses_to_the_min(self, fake_main_window):
        """A max below the min would otherwise produce a negative span."""
        main_window = fake_main_window(
            time_min_selected=8, time_max_selected=3
        )

        assert current_t_range(main_window) == (8, 8, 7, 7)

    def test_accepts_numeric_strings(self, fake_main_window):
        main_window = fake_main_window(
            time_min_selected="2", time_max_selected="4"
        )

        assert current_t_range(main_window) == (2, 4, 1, 3)


class TestCalculateTime:
    def test_writes_calculated_time_in_minutes(self, fake_main_window):
        main_window = fake_main_window(
            dt_seconds=180.0,
            track_df=pd.DataFrame({"t": [0, 1, 2]}),
        )

        calculate_time(main_window)

        assert main_window.track_df["Calculated_Time"].tolist() == [
            0.0,
            3.0,
            6.0,
        ]

    def test_caches_the_interval_on_the_window(self, fake_main_window):
        main_window = fake_main_window(
            dt_seconds=60.0, track_df=pd.DataFrame({"t": [0]})
        )

        calculate_time(main_window)

        assert main_window.time_interval == 60.0

    def test_does_nothing_to_a_frame_without_a_time_column(
        self, fake_main_window
    ):
        df = pd.DataFrame({"other": [1, 2]})
        main_window = fake_main_window(dt_seconds=60.0, track_df=df)

        calculate_time(main_window)

        assert "Calculated_Time" not in main_window.track_df.columns

    def test_replaces_the_frame_rather_than_mutating_it(
        self, fake_main_window
    ):
        """Callers hold references to the old frame; it must stay unchanged."""
        original = pd.DataFrame({"t": [0, 1]})
        main_window = fake_main_window(dt_seconds=60.0, track_df=original)

        calculate_time(main_window)

        assert "Calculated_Time" not in original.columns
        assert "Calculated_Time" in main_window.track_df.columns


# Imported real time
class TestRealtimeColumns:
    def test_finds_the_prefixed_columns(self):
        cols = ["t", f"{RT_PREFIX}1", f"{RT_PREFIX}2", "AreaMorphologyM1"]

        assert realtime_columns(cols) == [f"{RT_PREFIX}1", f"{RT_PREFIX}2"]

    def test_ignores_non_string_columns(self):
        """Pivot tables leave integer column labels behind."""
        assert realtime_columns([1, 2, None, f"{RT_PREFIX}1"]) == [
            f"{RT_PREFIX}1"
        ]

    def test_returns_empty_when_there_are_none(self):
        assert realtime_columns(["t", "Position"]) == []


def rt_frame() -> pd.DataFrame:
    """An imported real-time table: two positions, two frames, two channels."""
    rows = []
    for position in (1, 2):
        for t_file in (1, 2):
            for channel in (1, 2):
                rows.append(
                    (
                        f"exp_p{position:04d}_t{t_file:05d}_w{channel:02d}.png",
                        # 60000 ms == 1 minute, made distinct per row.
                        60_000 * (t_file + 10 * channel + 100 * position),
                    )
                )
    return pd.DataFrame(rows, columns=["Image File", "Measurement Time (ms)"])


class TestBuildRealtimeLookup:
    def test_builds_a_position_and_frame_indexed_table(self, fake_main_window):
        main_window = fake_main_window(
            import_rt_df=rt_frame(), n_channels=2, _rt_wide=None
        )

        wide = build_realtime_lookup(main_window)

        assert set(wide.columns) == {"Position", "t", 1, 2}
        assert sorted(wide["Position"].unique()) == [1, 2]

    def test_shifts_filename_time_points_to_zero_based(self, fake_main_window):
        """``_t00001_`` in a filename is frame 0 in the dataframe."""
        main_window = fake_main_window(
            import_rt_df=rt_frame(), n_channels=2, _rt_wide=None
        )

        wide = build_realtime_lookup(main_window)

        assert sorted(wide["t"].unique()) == [0, 1]

    def test_converts_milliseconds_to_minutes(self, fake_main_window):
        main_window = fake_main_window(
            import_rt_df=rt_frame(), n_channels=2, _rt_wide=None
        )

        wide = build_realtime_lookup(main_window)

        row = wide[(wide["Position"] == 1) & (wide["t"] == 0)]
        # t_file=1, channel=1, position=1 -> 60000 * 111 ms
        assert row[1].iloc[0] == pytest.approx(111.0)

    def test_renumbers_channels_to_be_consecutive(self, fake_main_window):
        """Files may use w03/w07; downstream columns must still be Ch1/Ch2."""
        frame = pd.DataFrame(
            {
                "Image File": [
                    "exp_p0001_t00001_w03.png",
                    "exp_p0001_t00001_w07.png",
                ],
                "Measurement Time (ms)": [60_000, 120_000],
            }
        )
        main_window = fake_main_window(
            import_rt_df=frame, n_channels=2, _rt_wide=None
        )

        wide = build_realtime_lookup(main_window)

        assert set(wide.columns) == {"Position", "t", 1, 2}

    def test_drops_channels_beyond_the_configured_count(
        self, fake_main_window
    ):
        main_window = fake_main_window(
            import_rt_df=rt_frame(), n_channels=1, _rt_wide=None
        )

        wide = build_realtime_lookup(main_window)

        assert set(wide.columns) == {"Position", "t", 1}

    def test_returns_the_cached_table_by_default(self, fake_main_window):
        sentinel = pd.DataFrame({"cached": [1]})
        main_window = fake_main_window(
            import_rt_df=rt_frame(), n_channels=2, _rt_wide=sentinel
        )

        assert build_realtime_lookup(main_window) is sentinel

    def test_force_rebuilds_past_the_cache(self, fake_main_window):
        sentinel = pd.DataFrame({"cached": [1]})
        main_window = fake_main_window(
            import_rt_df=rt_frame(), n_channels=2, _rt_wide=sentinel
        )

        wide = build_realtime_lookup(main_window, force=True)

        assert "cached" not in wide.columns

    def test_returns_none_without_an_imported_table(self, fake_main_window):
        main_window = fake_main_window(import_rt_df=None, _rt_wide=None)

        assert build_realtime_lookup(main_window) is None
        assert main_window._rt_wide is None

    def test_returns_none_when_the_required_columns_are_missing(
        self, fake_main_window
    ):
        main_window = fake_main_window(
            import_rt_df=pd.DataFrame({"Image File": ["x"]}), _rt_wide=None
        )

        assert build_realtime_lookup(main_window) is None

    def test_returns_none_when_no_filename_can_be_parsed(
        self, fake_main_window
    ):
        frame = pd.DataFrame(
            {
                "Image File": ["nonsense.png"],
                "Measurement Time (ms)": [60_000],
            }
        )
        main_window = fake_main_window(
            import_rt_df=frame, n_channels=2, _rt_wide=None
        )

        assert build_realtime_lookup(main_window) is None

    def test_skips_rows_with_an_unparsable_timestamp(self, fake_main_window):
        frame = rt_frame()
        frame["Measurement Time (ms)"] = frame["Measurement Time (ms)"].astype(
            object
        )
        frame.loc[0, "Measurement Time (ms)"] = "not a number"
        main_window = fake_main_window(
            import_rt_df=frame, n_channels=2, _rt_wide=None
        )

        wide = build_realtime_lookup(main_window)

        assert int(wide[[1, 2]].isna().sum().sum()) == 1
        dropped = wide[(wide["Position"] == 1) & (wide["t"] == 0)]
        assert dropped[1].isna().all()
        assert dropped[2].iloc[0] == pytest.approx(121.0)


class TestApplyRealtime:
    def test_joins_the_real_time_columns_onto_the_frame(
        self, fake_main_window
    ):
        main_window = fake_main_window(
            import_rt_df=rt_frame(), n_channels=2, _rt_wide=None
        )
        wide = build_realtime_lookup(main_window)
        df = pd.DataFrame({"Position": [1, 1], "t": [0, 1], "v": [9, 9]})

        out = apply_realtime(df, wide)

        assert f"{RT_PREFIX}1" in out.columns
        assert f"{RT_PREFIX}2" in out.columns

    def test_renames_integer_channel_columns_to_the_prefix(
        self, fake_main_window
    ):
        """The pivot leaves integer labels; none may survive into track_df."""
        main_window = fake_main_window(
            import_rt_df=rt_frame(), n_channels=2, _rt_wide=None
        )
        wide = build_realtime_lookup(main_window)
        df = pd.DataFrame({"Position": [1], "t": [0]})

        out = apply_realtime(df, wide)

        assert not [c for c in out.columns if isinstance(c, int)]

    def test_unmatched_rows_get_missing_values(self, fake_main_window):
        main_window = fake_main_window(
            import_rt_df=rt_frame(), n_channels=2, _rt_wide=None
        )
        wide = build_realtime_lookup(main_window)
        df = pd.DataFrame({"Position": [99], "t": [0]})

        out = apply_realtime(df, wide)

        assert out[f"{RT_PREFIX}1"].isna().all()

    def test_keeps_the_row_count(self, fake_main_window):
        main_window = fake_main_window(
            import_rt_df=rt_frame(), n_channels=2, _rt_wide=None
        )
        wide = build_realtime_lookup(main_window)
        df = pd.DataFrame({"Position": [1, 1, 2], "t": [0, 1, 0]})

        assert len(apply_realtime(df, wide)) == 3

    def test_returns_the_frame_unchanged_without_a_lookup(self):
        df = pd.DataFrame({"Position": [1], "t": [0]})

        assert apply_realtime(df, None) is df

    def test_returns_none_for_no_frame(self):
        assert apply_realtime(None, pd.DataFrame({"Position": [1]})) is None

    def test_returns_the_frame_unchanged_without_position_and_t(self):
        df = pd.DataFrame({"other": [1]})

        assert apply_realtime(df, pd.DataFrame({"Position": [1]})) is df


# The gap filler
class TestGapFillContextFromMainWindow:
    def test_reads_the_frame_range_off_the_window(self, fake_main_window):
        main_window = fake_main_window(
            time_min_selected=1,
            time_max_selected=6,
            current_position_number=2,
            time_interval=180.0,
            ids_channels=["w01"],
            image_present={"w01": [1, 1, 1, 1, 1, 1]},
        )

        ctx = GapFillContext.from_main_window(main_window)

        assert list(ctx.frames) == [0, 1, 2, 3, 4, 5]

    def test_reads_position_channels_and_interval(self, fake_main_window):
        main_window = fake_main_window(
            time_min_selected=1,
            time_max_selected=3,
            current_position_number=7,
            time_interval=60.0,
            ids_channels=["w01", "w02"],
            image_present={"w01": [1, 1, 1]},
        )

        ctx = GapFillContext.from_main_window(main_window)

        assert ctx.position == 7
        assert ctx.channels == ("w01", "w02")
        assert ctx.time_interval_s == 60.0

    def test_a_non_dict_presence_map_is_ignored(self, fake_main_window):
        main_window = fake_main_window(
            time_min_selected=1,
            time_max_selected=2,
            current_position_number=1,
            time_interval=60.0,
            ids_channels=["w01"],
            image_present=["not", "a", "dict"],
        )

        ctx = GapFillContext.from_main_window(main_window)

        assert ctx.presence == {}

    def test_an_unusable_interval_becomes_zero(self, fake_main_window):
        main_window = fake_main_window(
            time_min_selected=1,
            time_max_selected=2,
            current_position_number=1,
            time_interval="not a number",
            ids_channels=[],
            image_present={},
        )

        ctx = GapFillContext.from_main_window(main_window)

        assert ctx.time_interval_s == 0.0

    def test_a_broken_time_range_is_reported_as_a_value_error(
        self, fake_main_window
    ):
        main_window = fake_main_window(
            time_min_selected="not a number",
            current_position_number=1,
        )

        with pytest.raises(ValueError, match="current_t_range failed"):
            GapFillContext.from_main_window(main_window)

    def test_the_context_is_immutable(self, fake_main_window):
        """Frozen, so the filler cannot be surprised halfway through."""
        main_window = fake_main_window(
            time_min_selected=1,
            time_max_selected=2,
            current_position_number=1,
            time_interval=60.0,
            ids_channels=[],
            image_present={},
        )

        ctx = GapFillContext.from_main_window(main_window)

        with pytest.raises(AttributeError):
            ctx.position = 99
