"""Tests for SECQUOIA outlier detection.

The column layout is the part of the test data that matters: masks 1, 2 and
10, and channels 00, 01 and 10. That way a rule asking for mask 1 can be
shown not to also read mask 10. The values are a short series per track,
held steady for the sliding window tests apart from one large spike.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from SECQUOIA.config import CloseMaskSettings, Rule, SlidingWindow
from SECQUOIA.core.outlier_detection.close_masks import CLOSE_MASK_FLAG
from SECQUOIA.core.outlier_detection.detection import (
    RulesPack,
    compare_series_op,
    load_outlier_rules_from_disk,
    resolve_feature_columns,
    rule_is_active,
    run_outlier_pipeline,
    run_outlier_pipeline_for_all_positions,
    run_sliding_windows,
    run_threshold_rules,
)
from SECQUOIA.utils.paths import project_analysis_dir

COLUMNS = [
    "AreaMorphologyM1",
    "AreaMorphologyM2",
    "AreaMorphologyM10",
    "MeanNoBgCorrectedCh1M1",
    "MeanNoBgCorrectedCh2M1",
    "MeanNoBgCorrectedCh1M10",
    "MeanNoBgCorrectedCh10M1",
    "Ratio",
]


@pytest.fixture
def wide_df() -> pd.DataFrame:
    """Three rows spanning every column layout, plus a non-numeric column."""
    data = {col: [1.0, 2.0, 3.0] for col in COLUMNS}
    data["Identification"] = ["a", "a", "a"]
    data["t"] = [0, 1, 2]
    return pd.DataFrame(data)


# Feature column resolution
class TestResolveFeatureColumns:
    def test_mask_one_does_not_match_mask_ten(self, wide_df):
        """The collision that a plain substring check would get wrong."""
        cols = resolve_feature_columns(wide_df, "AreaMorphology", [1], None)

        assert cols == ["AreaMorphologyM1"]

    def test_channel_one_does_not_match_channel_ten(self, wide_df):
        cols = resolve_feature_columns(
            wide_df, "MeanNoBgCorrected", [1], ["1"]
        )

        assert cols == ["MeanNoBgCorrectedCh1M1"]

    def test_mask_ten_is_reachable(self, wide_df):
        cols = resolve_feature_columns(wide_df, "AreaMorphology", [10], None)

        assert cols == ["AreaMorphologyM10"]

    def test_several_masks_resolve_together(self, wide_df):
        cols = resolve_feature_columns(wide_df, "AreaMorphology", [1, 2], None)

        assert cols == ["AreaMorphologyM1", "AreaMorphologyM2"]

    def test_no_mask_filter_matches_every_mask(self, wide_df):
        cols = resolve_feature_columns(wide_df, "AreaMorphology", None, None)

        assert cols == [
            "AreaMorphologyM1",
            "AreaMorphologyM10",
            "AreaMorphologyM2",
        ]

    def test_a_channel_agnostic_column_ignores_the_channel_filter(
        self, wide_df
    ):
        """Area has no channel, so asking for channel 2 must still find it."""
        cols = resolve_feature_columns(wide_df, "AreaMorphology", [1], ["2"])

        assert "AreaMorphologyM1" in cols

    def test_a_fully_qualified_feature_name_with_a_filter_is_exact(
        self, wide_df
    ):
        cols = resolve_feature_columns(
            wide_df, "MeanNoBgCorrectedCh1M1", [1], None
        )

        assert cols == ["MeanNoBgCorrectedCh1M1"]

    def test_a_fully_qualified_name_needs_no_filter_to_be_exact(self, wide_df):
        """M1 is mask 1 and M10 is mask 10, so one must never match the other.

        The feature name carries its own mask token, so it is used as the
        filter even when the caller passes none.
        """
        assert resolve_feature_columns(
            wide_df, "AreaMorphologyM1", None, None
        ) == ["AreaMorphologyM1"]

    def test_a_partially_qualified_name_still_spans_every_mask(self, wide_df):
        """A channel-only name pins the channel and leaves the mask open."""
        assert resolve_feature_columns(
            wide_df, "MeanNoBgCorrectedCh1", None, None
        ) == ["MeanNoBgCorrectedCh1M1", "MeanNoBgCorrectedCh1M10"]

    def test_a_fully_qualified_name_pins_the_channel_too(self, wide_df):
        assert resolve_feature_columns(
            wide_df, "MeanNoBgCorrectedCh1M1", None, None
        ) == ["MeanNoBgCorrectedCh1M1"]

    def test_a_column_without_tokens_resolves_by_its_own_name(self, wide_df):
        cols = resolve_feature_columns(wide_df, "Ratio", None, None)

        assert cols == ["Ratio"]

    def test_non_numeric_columns_are_never_returned(self, wide_df):
        cols = resolve_feature_columns(wide_df, "Identification", None, None)

        assert cols == []

    def test_an_unknown_feature_resolves_to_nothing(self, wide_df):
        assert resolve_feature_columns(wide_df, "Ghost", [1], None) == []

    def test_the_result_is_sorted(self, wide_df):
        cols = resolve_feature_columns(
            wide_df, "MeanNoBgCorrected", [1, 10], ["1"]
        )

        assert cols == sorted(cols)


# Comparison operators
class TestCompareSeriesOp:
    SERIES = pd.Series([1.0, 5.0, 10.0])

    @pytest.mark.parametrize(
        ("op", "expected"),
        [
            ("<", [True, False, False]),
            ("<=", [True, True, False]),
            ("=", [False, True, False]),
            (">=", [False, True, True]),
            (">", [False, False, True]),
        ],
    )
    def test_every_supported_operator(self, op, expected):
        assert compare_series_op(self.SERIES, op, 5.0).tolist() == expected

    def test_an_unknown_operator_matches_nothing(self):
        result = compare_series_op(self.SERIES, "!=", 5.0)

        assert result.tolist() == [False, False, False]

    def test_non_numeric_values_never_match(self):
        series = pd.Series(["1.0", "oops", "10.0"])

        result = compare_series_op(series, ">", 5.0)

        assert result.tolist() == [False, False, True]


# Threshold rules
def threshold_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Identification": ["a"] * 5,
            "t": [0, 1, 2, 3, 4],
            "AreaMorphologyM1": [10.0, 50.0, 100.0, 150.0, 200.0],
            "AreaMorphologyM2": [10.0, 10.0, 10.0, 10.0, 999.0],
        }
    )


def pack(rules=(), windows=()) -> RulesPack:
    return RulesPack(
        version=1,
        m_n=2,
        ch_n=1,
        rules=list(rules),
        sliding_windows=list(windows),
    )


class TestRunThresholdRules:
    def test_flags_the_rows_above_a_threshold(self):
        rule = Rule(
            feat="AreaMorphology", masks=[1], channels=[], op1=">", val1=120.0
        )

        out = run_threshold_rules(threshold_df(), pack([rule]))

        assert out["Outlier_detection"].tolist() == [
            "OK",
            "OK",
            "OK",
            "Outlier",
            "Outlier",
        ]

    def test_creates_the_output_column_when_missing(self):
        out = run_threshold_rules(threshold_df(), pack())

        assert (out["Outlier_detection"] == "OK").all()

    def test_a_rule_without_values_is_not_active(self):
        rule = Rule(
            feat="AreaMorphology", masks=[1], channels=[], op1="<", val1=0.0
        )

        assert not rule_is_active(rule)

    def test_legacy_disabled_rules_are_dropped_on_load(self):
        loaded = RulesPack.from_dict(
            {
                "version": 1,
                "m_n": 1,
                "ch_n": 0,
                "rules": [
                    {
                        "feat": "AreaMorphology",
                        "masks": [1],
                        "channels": [],
                        "op1": ">",
                        "val1": 120.0,
                        "enabled": False,
                    }
                ],
                "sliding_windows": [],
            }
        )

        assert loaded.rules == []

    def test_two_operands_combined_with_and(self):
        """A band: flagged only between the two thresholds."""
        rule = Rule(
            feat="AreaMorphology",
            masks=[1],
            channels=[],
            op1=">",
            val1=40.0,
            op2="<",
            val2=120.0,
            combine="AND",
        )

        out = run_threshold_rules(threshold_df(), pack([rule]))

        assert out["Outlier_detection"].tolist() == [
            "OK",
            "Outlier",
            "Outlier",
            "OK",
            "OK",
        ]

    def test_two_operands_combined_with_or(self):
        """The complement of a band: flagged outside the two thresholds."""
        rule = Rule(
            feat="AreaMorphology",
            masks=[1],
            channels=[],
            op1="<",
            val1=40.0,
            op2=">",
            val2=120.0,
            combine="OR",
        )

        out = run_threshold_rules(threshold_df(), pack([rule]))

        assert out["Outlier_detection"].tolist() == [
            "Outlier",
            "OK",
            "OK",
            "Outlier",
            "Outlier",
        ]

    def test_a_rule_over_several_masks_flags_a_hit_in_either(self):
        rule = Rule(
            feat="AreaMorphology",
            masks=[1, 2],
            channels=[],
            op1=">",
            val1=500.0,
        )

        out = run_threshold_rules(threshold_df(), pack([rule]))

        # Only mask 2 exceeds 500, at the last row.
        assert out["Outlier_detection"].tolist() == [
            "OK",
            "OK",
            "OK",
            "OK",
            "Outlier",
        ]

    def test_rules_accumulate_and_never_clear_a_flag(self):
        rules = [
            Rule(
                feat="AreaMorphology",
                masks=[1],
                channels=[],
                op1=">",
                val1=180.0,
            ),
            Rule(
                feat="AreaMorphology",
                masks=[1],
                channels=[],
                op1="<",
                val1=20.0,
            ),
        ]

        out = run_threshold_rules(threshold_df(), pack(rules))

        assert out["Outlier_detection"].tolist() == [
            "Outlier",
            "OK",
            "OK",
            "OK",
            "Outlier",
        ]

    def test_a_rule_matching_no_column_is_skipped(self):
        rule = Rule(feat="Ghost", masks=[1], channels=[], op1=">", val1=0.0)

        out = run_threshold_rules(threshold_df(), pack([rule]))

        assert (out["Outlier_detection"] == "OK").all()

    def test_the_progress_callback_fires_once_per_rule(self):
        calls = []
        rules = [
            Rule(
                feat="AreaMorphology",
                masks=[1],
                channels=[],
                op1=">",
                val1=1e9,
            ),
            Rule(feat="Ghost", masks=[1], channels=[], op1=">", val1=0.0),
        ]

        run_threshold_rules(
            threshold_df(), pack(rules), on_rule_done=lambda: calls.append(1)
        )

        assert len(calls) == 2


# Sliding windows
def sliding_df(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Identification": ["a"] * len(values),
            "t": list(range(len(values))),
            "AreaMorphologyM1": values,
        }
    )


def window(sd_factor: float = 2.0, t_min: float = 2.0, t_max: float = 2.0):
    return SlidingWindow(
        feature="AreaMorphology",
        mask=1,
        channel=None,
        t_min=t_min,
        t_max=t_max,
        sd_factor=sd_factor,
    )


STEADY = [100.0] * 4 + [100.0, 100.0]
WITH_JUMP = [100.0] * 4 + [5000.0, 100.0]


class TestRunSlidingWindows:
    def test_a_steady_track_produces_no_outliers(self):
        out = run_sliding_windows(sliding_df(STEADY), pack(windows=[window()]))

        assert (out["Outlier_detection"] == "OK").all()

    def test_a_sudden_jump_is_flagged(self):
        out = run_sliding_windows(
            sliding_df(WITH_JUMP), pack(windows=[window()])
        )

        assert out.loc[out["t"] == 4, "Outlier_detection"].tolist() == [
            "Outlier"
        ]

    def test_the_recovery_after_a_jump_is_not_flagged(self):
        """The band around the spike is wide, so the return to normal fits."""
        out = run_sliding_windows(
            sliding_df(WITH_JUMP), pack(windows=[window()])
        )

        assert out.loc[out["t"] == 5, "Outlier_detection"].tolist() == ["OK"]

    def test_a_zero_sd_factor_disables_the_window(self):
        out = run_sliding_windows(
            sliding_df(WITH_JUMP), pack(windows=[window(sd_factor=0.0)])
        )

        assert (out["Outlier_detection"] == "OK").all()

    def test_tracks_are_evaluated_independently(self):
        """A jump in one track must not flag a steady row in another."""
        steady = sliding_df(STEADY)
        jumpy = sliding_df(WITH_JUMP)
        jumpy["Identification"] = "b"
        df = pd.concat([steady, jumpy], ignore_index=True)

        out = run_sliding_windows(df, pack(windows=[window()]))

        assert (
            out.loc[out["Identification"] == "a", "Outlier_detection"] == "OK"
        ).all()
        assert (
            out.loc[out["Identification"] == "b", "Outlier_detection"]
            == "Outlier"
        ).any()

    def test_no_windows_still_initialises_the_output_column(self):
        """The column is created before the empty-config early return."""
        out = run_sliding_windows(sliding_df(WITH_JUMP), pack())

        assert (out["Outlier_detection"] == "OK").all()

    def test_a_window_matching_no_column_is_skipped(self):
        df = sliding_df([100.0, 5000.0, 99.0])
        cfg = SlidingWindow(
            feature="Ghost",
            mask=1,
            channel=None,
            t_min=2.0,
            t_max=2.0,
            sd_factor=2.0,
        )

        out = run_sliding_windows(df, pack(windows=[cfg]))

        assert (out["Outlier_detection"] == "OK").all()

    def test_an_empty_frame_passes_through(self):
        empty = pd.DataFrame()

        assert run_sliding_windows(empty, pack(windows=[window()])).empty

    def test_none_passes_through(self):
        assert run_sliding_windows(None, pack(windows=[window()])) is None

    def test_a_frame_without_the_key_columns_passes_through(self):
        df = pd.DataFrame({"AreaMorphologyM1": [1.0, 2.0]})

        out = run_sliding_windows(df, pack(windows=[window()]))

        assert "Outlier_detection" not in out.columns

    def test_a_single_time_point_cannot_be_an_outlier(self):
        """There is no previous point to build a band from."""
        df = sliding_df([100.0])

        out = run_sliding_windows(df, pack(windows=[window()]))

        assert (out["Outlier_detection"] == "OK").all()

    def test_missing_values_do_not_crash_the_window(self):
        df = sliding_df([100.0, np.nan, 99.0, np.nan, 100.5, 99.5])

        out = run_sliding_windows(df, pack(windows=[window()]))

        assert len(out) == 6

    def test_the_progress_callback_fires_once_per_window(self):
        calls = []
        df = sliding_df([100.0, 101.0, 99.0])

        run_sliding_windows(
            df,
            pack(windows=[window(), window()]),
            on_window_done=lambda: calls.append(1),
        )

        assert len(calls) == 2


class TestRunOutlierPipeline:
    def test_runs_both_stages(self):
        df = sliding_df([100.0, 101.0, 99.0, 100.5, 5000.0, 99.5])
        rule = Rule(
            feat="AreaMorphology",
            masks=[1],
            channels=[],
            op1="<",
            val1=99.8,
        )

        out = run_outlier_pipeline(df, pack([rule], [window()]))

        flagged = out.loc[out["Outlier_detection"] == "Outlier", "t"].tolist()
        assert 4 in flagged  # from the sliding window
        assert 2 in flagged  # from the threshold rule

    def test_resets_previous_results(self):
        """A rerun must not keep flags from the previous run."""
        df = sliding_df([100.0, 101.0, 99.0])
        df["Outlier_detection"] = "Outlier"

        out = run_outlier_pipeline(df, pack())

        assert (out["Outlier_detection"] == "OK").all()

    def test_keeps_reviewed_rows_by_default(self):
        df = sliding_df([100.0, 101.0, 99.0])
        df["Outlier_detection"] = ["Reviewed", "OK", "OK"]

        out = run_outlier_pipeline(df, pack())

        assert out.loc[out["t"] == 0, "Outlier_detection"].item() == "Reviewed"

    def test_keep_reviewed_false_re_evaluates_reviewed_rows_too(self):
        df = sliding_df([100.0, 101.0, 99.0])
        df["Outlier_detection"] = ["Reviewed", "OK", "OK"]

        out = run_outlier_pipeline(df, pack(), keep_reviewed=False)

        assert (out["Outlier_detection"] == "OK").all()


class TestRulesPackRoundTrip:
    def test_survives_a_dict_round_trip(self):
        original = pack(
            [
                Rule(
                    feat="AreaMorphology",
                    masks=[1],
                    channels=["1"],
                    op1=">",
                    val1=10.0,
                )
            ],
            [window()],
        )

        restored = RulesPack.from_dict(original.to_dict())

        assert restored.rules == original.rules
        assert restored.sliding_windows == original.sliding_windows
        assert (restored.m_n, restored.ch_n) == (original.m_n, original.ch_n)

    def test_survives_a_json_round_trip(self, tmp_path):
        original = pack(
            [
                Rule(
                    feat="AreaMorphology",
                    masks=[1, 2],
                    channels=[],
                    op1="<",
                    val1=5.0,
                    op2=">",
                    val2=1.0,
                    combine="AND",
                )
            ],
            [window(sd_factor=3.5)],
        )
        path = tmp_path / "rules.json"
        path.write_text(json.dumps(original.to_dict()))

        restored = RulesPack.from_dict(load_outlier_rules_from_disk(str(path)))

        assert restored.rules == original.rules
        assert restored.sliding_windows == original.sliding_windows

    def test_to_dict_keeps_the_legacy_single_window_key(self):
        """Older saved files carry one window under "sliding_window"."""
        payload = pack(windows=[window()]).to_dict()

        assert payload["sliding_window"] is not None
        assert payload["sliding_windows"][0] == payload["sliding_window"]

    def test_the_legacy_single_window_key_is_still_read(self):
        payload = pack(windows=[window()]).to_dict()
        payload["sliding_windows"] = []

        restored = RulesPack.from_dict(payload)

        assert len(restored.sliding_windows) == 1

    def test_an_empty_dict_yields_an_empty_pack(self):
        restored = RulesPack.from_dict({})

        assert restored.rules == []
        assert restored.sliding_windows == []
        assert restored.version == 1

    def test_loading_a_missing_file_returns_none(self, tmp_path):
        assert (
            load_outlier_rules_from_disk(str(tmp_path / "nope.json")) is None
        )

    def test_loading_malformed_json_returns_none(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not json")

        assert load_outlier_rules_from_disk(str(path)) is None


class TestRunOutlierPipelineForAllPositions:
    """Positions 1 and 2 are quantified; 3 only exists as a folder on disk;
    4 has tracking rows but no mask matching yet,
    both 3 and 4 must be skipped, not just 3."""

    _POSITIONS = (1, 2, 3, 4)

    @staticmethod
    def _df() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Position": [1, 1, 2, 2, 4, 4],
                "Identification": [
                    "p1-a",
                    "p1-a",
                    "p2-a",
                    "p2-a",
                    "p4-a",
                    "p4-a",
                ],
                "TrackNumber": [1, 1, 1, 1, 1, 1],
                "t": [0, 1, 0, 1, 0, 1],
                "AreaMorphologyM1": [10.0, 200.0, 10.0, 11.0, np.nan, np.nan],
                "alt_dist_px_m1": [
                    np.nan,
                    3.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                "label_id_m1": [1, 1, 1, 1, np.nan, np.nan],
                "Outlier_detection": [
                    "Reviewed",
                    "OK",
                    "OK",
                    "OK",
                    "OK",
                    "OK",
                ],
            }
        )

    @staticmethod
    def _rule() -> Rule:
        return Rule(
            feat="AreaMorphology", masks=[1], channels=[], op1=">", val1=100.0
        )

    def _window(self, fake_main_window, tmp_path, **extra):
        root = tmp_path / "exp"
        for p in self._POSITIONS:
            (root / f"exp_p{p:04d}").mkdir(parents=True)
        attrs = {
            "folder": str(root),
            "tracking_format": "tTt",
            "project_name": "Project_1",
            "time_min_selected": 1,
            "time_max_selected": 2,
            "n_masks": 1,
            "n_channels": 1,
            "track_df": self._df(),
            "position_folders": [
                str(root / f"exp_p{p:04d}") for p in self._POSITIONS
            ],
        }
        attrs.update(extra)
        return fake_main_window(**attrs)

    def test_re_evaluates_every_quantified_position_from_scratch(
        self, fake_main_window, tmp_path
    ):
        window = self._window(fake_main_window, tmp_path)

        processed, skipped = run_outlier_pipeline_for_all_positions(
            window, pack([self._rule()])
        )

        assert processed == [1, 2]
        out = window.track_df.set_index(["Position", "t"])["Outlier_detection"]
        assert out.loc[(1, 0)] == "OK"
        assert out.loc[(1, 1)] == "Outlier"
        assert out.loc[(2, 0)] == "OK"
        assert out.loc[(2, 1)] == "OK"

    def test_reports_unquantified_positions_as_skipped(
        self, fake_main_window, tmp_path
    ):
        """Both p3 (no rows at all) and p4 (tracked but not mask-matched)."""
        window = self._window(fake_main_window, tmp_path)

        _processed, skipped = run_outlier_pipeline_for_all_positions(
            window, pack([self._rule()])
        )

        assert skipped == [3, 4]

    def test_applies_close_mask_detection_when_configured(
        self, fake_main_window, tmp_path
    ):
        window = self._window(fake_main_window, tmp_path)
        rules_pack = RulesPack(
            version=1,
            m_n=1,
            ch_n=1,
            rules=[],
            sliding_windows=[],
            close_masks=CloseMaskSettings(distance=5.0),
        )

        run_outlier_pipeline_for_all_positions(window, rules_pack)

        flags = window.track_df.set_index(["Position", "t"])[CLOSE_MASK_FLAG]
        assert flags.loc[(1, 1)] == "Flagged"
        assert flags.loc[(1, 0)] == "OK"

    def test_calls_on_position_with_progress(self, fake_main_window, tmp_path):
        window = self._window(fake_main_window, tmp_path)
        calls = []

        run_outlier_pipeline_for_all_positions(
            window,
            pack(),
            on_position=lambda i, total, p: calls.append((i, total, p)),
        )

        assert calls == [(1, 2, 1), (2, 2, 2)]

    def test_writes_a_csv_only_for_processed_positions(
        self, fake_main_window, tmp_path
    ):
        window = self._window(fake_main_window, tmp_path)

        run_outlier_pipeline_for_all_positions(window, pack([self._rule()]))

        folder = Path(project_analysis_dir(window))
        assert list(folder.glob("SECQUOIA_p0001_*.csv"))
        assert list(folder.glob("SECQUOIA_p0002_*.csv"))
        assert not list(folder.glob("SECQUOIA_p0003_*.csv"))
        assert not list(folder.glob("SECQUOIA_p0004_*.csv"))

    def test_returns_empty_without_a_track_df(self, fake_main_window):
        window = fake_main_window(track_df=None)

        assert run_outlier_pipeline_for_all_positions(window, pack()) == (
            [],
            [],
        )
