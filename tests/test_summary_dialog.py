"""Tests for the column-matching logic in the outlier summary dialog."""

from __future__ import annotations

import pandas as pd
import pytest

from SECQUOIA.gui.outlier.summary_dialog import find_feature_columns

COLUMNS = [
    "AreaMorphologyM1",
    "AreaMorphologyM2",
    "AreaMorphologyM10",
    "MeanNoBgCorrectedCh00M1",
    "MeanNoBgCorrectedCh01M1",
    "MeanNoBgCorrectedCh00M10",
    "Identification",  # non-numeric, must never be returned
    "TrackNumber",
]


@pytest.fixture
def df() -> pd.DataFrame:
    data = {c: [1, 2, 3] for c in COLUMNS}
    data["Identification"] = ["a", "b", "c"]
    return pd.DataFrame(data)


def test_single_digit_mask_does_not_match_double_digit_mask(df):
    assert find_feature_columns(df, "AreaMorphology", [1], []) == [
        "AreaMorphologyM1"
    ]


def test_double_digit_mask_matches_only_itself(df):
    assert find_feature_columns(df, "AreaMorphology", [10], []) == [
        "AreaMorphologyM10"
    ]


def test_channel_and_mask_combined(df):
    assert find_feature_columns(df, "MeanNoBgCorrected", [1], ["00"]) == [
        "MeanNoBgCorrectedCh00M1"
    ]
    assert find_feature_columns(df, "MeanNoBgCorrected", [10], ["00"]) == [
        "MeanNoBgCorrectedCh00M10"
    ]


def test_empty_masks_and_channels_means_all(df):
    assert find_feature_columns(df, "AreaMorphology", [], []) == [
        "AreaMorphologyM1",
        "AreaMorphologyM10",
        "AreaMorphologyM2",
    ]


def test_multiple_masks_are_unioned(df):
    assert find_feature_columns(df, "AreaMorphology", [1, 2], []) == [
        "AreaMorphologyM1",
        "AreaMorphologyM2",
    ]


def test_feature_name_already_carrying_a_suffix(df):
    """M1 is mask 1 and M10 is mask 10, so the suffix must be matched exactly."""
    assert find_feature_columns(df, "AreaMorphologyM1", [], []) == [
        "AreaMorphologyM1"
    ]


def test_feature_name_carrying_only_a_channel_still_spans_every_mask(df):
    assert find_feature_columns(df, "MeanNoBgCorrectedCh00", [], []) == [
        "MeanNoBgCorrectedCh00M1",
        "MeanNoBgCorrectedCh00M10",
    ]


def test_non_numeric_columns_are_excluded(df):
    assert find_feature_columns(df, "Identification", [], []) == []


def test_no_match_returns_empty_list(df):
    assert find_feature_columns(df, "DoesNotExist", [1], ["00"]) == []
