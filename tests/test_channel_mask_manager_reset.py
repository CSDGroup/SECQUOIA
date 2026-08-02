"""Tests for the state a channel/mask/BaSiC change invalidates.

Re-measuring with a different configuration rewrites the columns the outlier
rules and the calculated metrics were built on, so applying a change clears
the outlier verdicts and rules, deletes every calculated metric, and forgets
plot-row features that no longer exist.
"""

from __future__ import annotations

import pandas as pd
import pytest

from SECQUOIA.gui.loading.channel_mask_manager_dialog import (
    _calculated_metric_names,
    _clear_calculated_metrics,
    _clear_outlier_state,
    _columns_lost_by,
    _drop_missing_row_features,
)
from SECQUOIA.gui.main_window.dynamics_plot_row_labels import (
    DynamicsPlotRowLabels,
)

pytestmark = pytest.mark.gui

CHANNELS = ["C00", "C01"]
VARIANTS = ("NoBgCorrected", "BaSiCBgCorrectedRatioFlat")


def measured_frame(channels=CHANNELS, *, variants=VARIANTS, masks=(1,)):
    """A track_df measured under the given configuration."""
    data = {}
    for mask in masks:
        for variant in variants:
            for channel in channels:
                data[f"Mean{variant}Ch{channel[1:]}M{mask}"] = [1.0, 2.0]
        data[f"AreaMorphologyM{mask}"] = [10.0, 11.0]
        data[f"label_id_m{mask}"] = [1, 2]
    return pd.DataFrame({"Position": [1, 1], "t": [0, 1], **data})


def metric(name, col_a, col_b):
    """One entry of the ``_derived_features`` registry."""
    return {
        name: {
            "template": f"{col_a}/{col_b}",
            "display": name,
            "formula": {
                "source_col_a": col_a,
                "source_col_b": col_b,
                "target_col": f"{col_a}/{col_b}",
                "op": "/",
            },
        }
    }


class MetricWindow(DynamicsPlotRowLabels):
    """A main window stand-in carrying only what metric deletion touches."""

    def __init__(self, **attrs):
        for name, value in attrs.items():
            setattr(self, name, value)


def test_no_columns_are_lost_when_the_configuration_is_unchanged(
    fake_main_window,
):
    main_window = fake_main_window(track_df=measured_frame())

    assert _columns_lost_by(main_window, CHANNELS, basic=True, n_masks=1) == []


def test_turning_basic_off_reports_the_basic_columns(fake_main_window):
    main_window = fake_main_window(track_df=measured_frame())

    lost = _columns_lost_by(main_window, CHANNELS, basic=False, n_masks=1)

    assert sorted(lost) == [
        "MeanBaSiCBgCorrectedRatioFlatCh00M1",
        "MeanBaSiCBgCorrectedRatioFlatCh01M1",
    ]


def test_dropping_a_channel_reports_its_columns(fake_main_window):
    main_window = fake_main_window(track_df=measured_frame())

    lost = _columns_lost_by(main_window, ["C00"], basic=True, n_masks=1)

    assert "MeanNoBgCorrectedCh01M1" in lost
    assert "MeanNoBgCorrectedCh00M1" not in lost


def test_a_window_without_data_reports_nothing(fake_main_window):
    main_window = fake_main_window()

    assert (
        _columns_lost_by(main_window, CHANNELS, basic=False, n_masks=1) == []
    )


@pytest.fixture
def window_with_metrics():
    """A session with two calculated metrics, one of them still computable."""
    frame = measured_frame()
    frame["MeanBaSiCBgCorrectedRatioFlatCh00M1/AreaMorphologyM1"] = [0.1, 0.2]
    frame["AreaMorphologyM1/label_id_m1"] = [10.0, 5.5]
    return MetricWindow(
        track_df=frame,
        filtered_df=frame.copy(),
        df_subset=frame.copy(),
        _derived_features={
            **metric(
                "BaSiC ratio",
                "MeanBaSiCBgCorrectedRatioFlatCh00M1",
                "AreaMorphologyM1",
            ),
            **metric("Area per label", "AreaMorphologyM1", "label_id_m1"),
        },
        _feature_defs={"BaSiC ratio": {}, "Area per label": {}},
        selected_feature_by_row={1: "BaSiC ratio", 2: "Area per label"},
        last_run_config={1: {"feature_key": "BaSiC ratio"}},
        row_tools={},
    )


def test_the_metric_names_are_listed_for_the_confirmation(window_with_metrics):
    assert _calculated_metric_names(window_with_metrics) == [
        "Area per label",
        "BaSiC ratio",
    ]


def test_every_metric_is_deleted_even_a_computable_one(window_with_metrics):
    """The chosen policy: any applied change wipes all calculated metrics."""
    deleted = _clear_calculated_metrics(window_with_metrics)

    assert deleted == ["Area per label", "BaSiC ratio"]
    assert window_with_metrics._derived_features == {}


def test_deleting_removes_the_metric_columns_from_every_frame(
    window_with_metrics,
):
    _clear_calculated_metrics(window_with_metrics)

    for attr in ("track_df", "filtered_df", "df_subset"):
        columns = getattr(window_with_metrics, attr).columns
        assert "MeanBaSiCBgCorrectedRatioFlatCh00M1/AreaMorphologyM1" not in (
            columns
        )
        assert "AreaMorphologyM1/label_id_m1" not in columns
        assert "AreaMorphologyM1" in columns


def test_deleting_clears_the_feature_defs_and_row_state(window_with_metrics):
    """Nothing may keep pointing at a metric that no longer exists."""
    _clear_calculated_metrics(window_with_metrics)

    assert window_with_metrics._feature_defs == {}
    assert window_with_metrics.selected_feature_by_row == {}
    assert window_with_metrics.last_run_config == {}


def test_deleting_is_safe_without_any_metrics():
    main_window = MetricWindow(
        track_df=measured_frame(), _derived_features={}, row_tools={}
    )

    assert _clear_calculated_metrics(main_window) == []


def test_a_window_that_cannot_delete_is_tolerated(fake_main_window):
    """A stand-in without the mixin must not break the apply."""
    main_window = fake_main_window(_derived_features=metric("R", "a", "b"))

    assert _clear_calculated_metrics(main_window) == []


@pytest.fixture
def flagged_window(fake_main_window):
    """A session with outlier rules, flags and a marker cache in place."""
    frame = measured_frame()
    frame["Identification"] = ["p1-001", "p1-001"]
    frame["Outlier_detection"] = ["Outlier", "OK"]
    return fake_main_window(
        track_df=frame,
        filtered_df=frame.copy(),
        unique_ids=["p1-001"],
        unique_outliers_ids=["p1-001"],
        current_outlier_index=1,
        current_ident_index=0,
        _last_outlier_rules={"rules": [{"feat": "MeanNoBgCorrected"}]},
        _outlier_marker_items={},
    )


def test_applying_discards_the_saved_rules(flagged_window):
    """They must be gone, not re-applied on the next position switch."""
    _clear_outlier_state(flagged_window)

    assert not getattr(flagged_window, "_last_outlier_rules", None)


def test_applying_resets_the_flags_in_both_frames(flagged_window):
    _clear_outlier_state(flagged_window)

    assert (flagged_window.track_df["Outlier_detection"] == "OK").all()
    assert (flagged_window.filtered_df["Outlier_detection"] == "OK").all()


def test_applying_clears_the_outlier_list(flagged_window):
    _clear_outlier_state(flagged_window)

    assert list(flagged_window.unique_outliers_ids) == []
    assert flagged_window.current_outlier_index == 0


def test_clearing_is_safe_without_any_outlier_state(fake_main_window):
    main_window = fake_main_window(track_df=measured_frame())

    _clear_outlier_state(main_window)

    assert not getattr(main_window, "_last_outlier_rules", None)


def test_a_row_on_a_removed_feature_is_forgotten(fake_main_window):
    """The selection is persisted, so a dead key would survive restarts."""
    main_window = fake_main_window(
        track_df=measured_frame(variants=("NoBgCorrected",)),
        selected_feature_by_row={
            1: "MeanNoBgCorrected",
            2: "MeanBaSiCBgCorrectedRatioFlat",
        },
        _derived_features={},
    )

    cleared = _drop_missing_row_features(main_window)

    assert cleared == [2]
    assert main_window.selected_feature_by_row == {1: "MeanNoBgCorrected"}


def test_rows_on_surviving_features_are_kept(fake_main_window):
    main_window = fake_main_window(
        track_df=measured_frame(),
        selected_feature_by_row={1: "MeanBaSiCBgCorrectedRatioFlat"},
        _derived_features={},
    )

    assert _drop_missing_row_features(main_window) == []


def test_nothing_to_do_without_selections(fake_main_window):
    main_window = fake_main_window(
        track_df=measured_frame(), selected_feature_by_row={}
    )

    assert _drop_missing_row_features(main_window) == []
