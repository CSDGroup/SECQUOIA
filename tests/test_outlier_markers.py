"""Tests for SECQUOIA outlier marker columns.

These work out which dataframe column a dynamics plot row is showing. The
test data is a small feature dictionary with three entries: one carrying a
channel and a mask, one carrying only a mask, and one plain column name.
That covers every template shape the markers have to fill in.
"""

from __future__ import annotations

from SECQUOIA.gui.outlier.markers import _column_for_row, _new_star_symbol


class _MarkerMainWindow:
    def __init__(self, feature, m=1, ch=1):
        self.selected_feature_by_row = {1: feature}
        self.selected_m_by_channel = {1: m}
        self.selected_ch_by_channel = {1: ch}


CH_M_DEFS = {
    "MeanNoBgCorrected": {
        "template": "MeanNoBgCorrectedCh{ch}M{m}",
        "has_ch": True,
        "has_m": True,
    },
    "AreaMorphology": {
        "template": "AreaMorphologyM{m}",
        "has_ch": False,
        "has_m": True,
    },
    "MyRatio": {"template": "MyRatioColumn", "has_ch": False, "has_m": False},
}


def test_column_for_row_resolves_a_channel_and_mask_feature():
    col, feat = _column_for_row(
        _MarkerMainWindow("MeanNoBgCorrected", m=2, ch="01"), CH_M_DEFS, 1
    )
    assert col == "MeanNoBgCorrectedCh01M2"
    assert feat == "MeanNoBgCorrected"


def test_column_for_row_resolves_a_mask_only_feature():
    col, _ = _column_for_row(
        _MarkerMainWindow("AreaMorphology", m=3), CH_M_DEFS, 1
    )
    assert col == "AreaMorphologyM3"


def test_column_for_row_resolves_a_direct_column():
    col, _ = _column_for_row(_MarkerMainWindow("MyRatio"), CH_M_DEFS, 1)
    assert col == "MyRatioColumn"


def test_column_for_row_returns_none_for_an_unknown_feature():
    assert _column_for_row(_MarkerMainWindow("Ghost"), CH_M_DEFS, 1) == (
        None,
        None,
    )


def test_column_for_row_returns_none_when_no_feature_is_selected():
    main_window = _MarkerMainWindow("AreaMorphology")
    main_window.selected_feature_by_row = {}
    assert _column_for_row(main_window, CH_M_DEFS, 1) == (None, None)


def test_new_star_symbol_prefers_star_when_supported():
    assert _new_star_symbol() == "star"
