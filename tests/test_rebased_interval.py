"""A time interval is loaded rebased: ``track_df['t']`` starts at 0.

These tests load t-files 700..800 (so ``t`` runs 0..100) and check that
everything working in ``t`` terms stays inside 0..100, while the files written
to disk keep the original file time points.
"""

from __future__ import annotations

import pandas as pd
import pytest

from SECQUOIA.core.tracking.clt_io import CLTParser
from SECQUOIA.core.tracking.context import (
    _current_t_bounds,
    _forward_t_max,
)
from SECQUOIA.utils.timing import (
    apply_realtime,
    build_realtime_lookup,
    t_rebase_offset,
)


@pytest.fixture
def late_window(fake_main_window):
    return fake_main_window(time_min_selected=700, time_max_selected=800)


def frame(ts):
    return pd.DataFrame(
        {
            "t": ts,
            "Position": 1,
            "TrackNumber": 1,
            "Identification": "exp_p0001",
            "XMorphology": 5.0,
            "YMorphology": 6.0,
        }
    )


def time_points(block: str) -> list[int]:
    return [
        int(line.split(";")[0]) for line in block.splitlines() if ";" in line
    ]


class TestRebaseOffset:
    def test_is_the_zero_based_first_frame_of_the_interval(self, late_window):
        assert t_rebase_offset(late_window) == 699

    def test_is_zero_when_no_interval_is_set(self, fake_main_window):
        assert t_rebase_offset(fake_main_window(time_min_selected=None)) == 0
        assert t_rebase_offset(fake_main_window()) == 0


class TestEditBounds:
    def test_stay_inside_the_rebased_range(self, late_window):
        assert _current_t_bounds(late_window) == (700, 0, 100)

    def test_a_division_extends_to_the_last_rebased_frame(self, late_window):
        assert (
            _forward_t_max(late_window, pd.DataFrame(), "x", default=0) == 100
        )


class TestCltExportKeepsOriginalTimePoints:
    def test_quantification_block(self):
        block = CLTParser(None)._build_quantification_block_from_df(
            frame([0, 1, 100]), "u", "Q", "SECQUOIA", None, 0, 1, t_offset=699
        )

        assert time_points(block) == [700, 701, 800]

    def test_tracking_data_block(self):
        block = CLTParser(None)._build_trackingdata_block_from_df(
            frame([0, 1, 100]), t_offset=699
        )

        assert time_points(block) == [700, 701, 800]

    def test_without_an_offset_the_time_points_are_unchanged(self):
        block = CLTParser(None)._build_trackingdata_block_from_df(
            frame([0, 1, 2])
        )

        assert time_points(block) == [1, 2, 3]

    def test_a_new_file_carries_the_offset(self):
        text = CLTParser(None)._new_clt_text(
            frame([0, 100]), "tester", t_offset=699
        )

        assert "\t\t700;1;1;0;" in text
        assert "\t\t800;1;1;0;" in text


class TestRealtimeLookup:
    def rt_frame(self):
        return pd.DataFrame(
            {
                "Image File": [
                    "x_p0001_t00700_z001_w00.tif",
                    "x_p0001_t00701_z001_w00.tif",
                    "x_p0001_t00001_z001_w00.tif",
                ],
                "Measurement Time (ms)": [300_000, 360_000, 1_000],
            }
        )

    def test_rebased_t_gets_the_time_of_its_own_file(self, fake_main_window):
        main_window = fake_main_window(
            time_min_selected=700,
            time_max_selected=800,
            n_channels=1,
            import_rt_df=self.rt_frame(),
        )
        wide = build_realtime_lookup(main_window, force=True)

        merged = apply_realtime(
            pd.DataFrame({"Position": [1, 1], "t": [0, 1]}), wide
        )

        assert merged["RealTimeMinutes_Ch1"].tolist() == [5.0, 6.0]

    def test_frames_before_the_interval_are_dropped(self, fake_main_window):
        main_window = fake_main_window(
            time_min_selected=700,
            time_max_selected=800,
            n_channels=1,
            import_rt_df=self.rt_frame(),
        )

        wide = build_realtime_lookup(main_window, force=True)

        assert wide["t"].tolist() == [0, 1]

    def test_a_selection_from_the_start_is_unchanged(self, fake_main_window):
        main_window = fake_main_window(
            time_min_selected=1,
            time_max_selected=10,
            n_channels=1,
            import_rt_df=self.rt_frame().iloc[[2]],
        )

        wide = build_realtime_lookup(main_window, force=True)

        assert wide["t"].tolist() == [0]
