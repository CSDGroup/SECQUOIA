"""Tests for the Qt-free parts of SECQUOIA.utils.plotting."""

from __future__ import annotations

import pandas as pd
import pyqtgraph as pg
import pytest

from SECQUOIA.utils.plotting import (
    DEFAULT_FEATURE,
    DEFAULT_TRACK_COLOR,
    EXCLUDED_FEATURES,
    TIME_MODE_CALC,
    TIME_MODE_REAL,
    TIME_MODE_T,
    Z_HIGHLIGHT_BASE,
    Z_HIGHLIGHT_TOP,
    _discover_features,
    _feature_combo_needs_reload,
    _finite_max,
    _finite_min,
    _global_y_range,
    _initial_feature_index,
    _row_axis_metrics,
    _row_curve_plan,
    _row_xmax,
    _style_row_axes,
    _track_color_map,
    _track_xy,
    _visible_tracks,
    x_column_for,
)

REAL_LIKE_COLUMNS = [
    "Position",
    "t",
    "TrackNumber",
    "Identification",
    "track_id",
    "Cellfate",
    "active",
    "inspected",
    "Calculated_Time",
    "XMorphology",
    "YMorphology",
    "AreaMorphologyM1",
    "AreaMorphologyM2",
    "PerimeterMorphologyM1",
    "MeanNoBgCorrectedCh00M1",
    "MeanNoBgCorrectedCh01M1",
    "MeanNoBgCorrectedCh02M1",
    "MeanNoBgCorrectedCh00M2",
    "MaxNoBgCorrectedCh00M1",
    "nn_dist_px_M1",
    "label_id_M1",
    "step_disp_px_M1",
]


# Shape of the result
def test_discovers_channel_and_mask_features():
    features = _discover_features(REAL_LIKE_COLUMNS)

    assert features["MeanNoBgCorrected"] == {
        "template": "MeanNoBgCorrectedCh{ch}M{m}",
        "has_ch": True,
        "has_m": True,
        "display": "MeanNoBgCorrected",
    }
    assert features["MaxNoBgCorrected"]["has_ch"] is True


def test_discovers_mask_only_features():
    features = _discover_features(REAL_LIKE_COLUMNS)

    assert features["AreaMorphology"] == {
        "template": "AreaMorphologyM{m}",
        "has_ch": False,
        "has_m": True,
        "display": "AreaMorphology",
    }
    assert "PerimeterMorphology" in features


def test_template_round_trips_to_a_real_column():
    """The whole point of the template is that formatting it finds a column."""
    features = _discover_features(REAL_LIKE_COLUMNS)

    intensity = features["MeanNoBgCorrected"]["template"].format(ch="01", m=1)
    assert intensity in REAL_LIKE_COLUMNS

    morphology = features["AreaMorphology"]["template"].format(m=2)
    assert morphology in REAL_LIKE_COLUMNS


def test_bookkeeping_columns_are_not_features():
    features = _discover_features(REAL_LIKE_COLUMNS)
    for column in ("Position", "t", "TrackNumber", "Identification"):
        assert column not in features


@pytest.mark.parametrize("excluded", sorted(EXCLUDED_FEATURES))
def test_excluded_features_never_appear(excluded):
    features = _discover_features([f"{excluded}M1", f"{excluded}Ch00M1"])
    assert excluded not in features


def test_step_displacement_is_offered():
    """step_disp_px_ is deliberately not in EXCLUDED_FEATURES."""
    features = _discover_features(REAL_LIKE_COLUMNS)
    assert "step_disp_px_" in features


def test_channel_columns_do_not_leak_in_as_mask_only_features():
    features = _discover_features(["MeanNoBgCorrectedCh00M1"])

    assert set(features) == {"MeanNoBgCorrected"}
    assert "MeanNoBgCorrectedCh00" not in features


def test_a_channel_column_upgrades_an_existing_mask_only_entry():
    """Column order must not change the outcome."""
    mask_first = _discover_features(["FooM1", "FooCh00M1"])
    channel_first = _discover_features(["FooCh00M1", "FooM1"])

    assert mask_first["Foo"]["has_ch"] is True
    assert mask_first["Foo"]["template"] == "FooCh{ch}M{m}"
    assert mask_first["Foo"] == channel_first["Foo"]


def test_only_the_trailing_numbers_are_templated():
    """A feature whose own name contains digits keeps them."""
    features = _discover_features(["Ratio2Ch00M1"])
    assert features["Ratio2"]["template"] == "Ratio2Ch{ch}M{m}"


def test_multi_digit_channel_and_mask_numbers():
    features = _discover_features(["FooCh12M34"])
    assert features["Foo"]["template"] == "FooCh{ch}M{m}"


# Derived features
def test_derived_feature_is_added_when_its_column_exists():
    features = _discover_features(
        REAL_LIKE_COLUMNS,
        {"MyRatio": {"template": "AreaMorphologyM1", "display": "My Ratio"}},
    )

    assert features["MyRatio"] == {
        "template": "AreaMorphologyM1",
        "has_ch": False,
        "has_m": False,
        "display": "My Ratio",
    }


def test_derived_feature_is_skipped_when_its_column_is_missing():
    features = _discover_features(
        REAL_LIKE_COLUMNS,
        {"Ghost": {"template": "NotAColumn", "display": "Ghost"}},
    )
    assert "Ghost" not in features


def test_derived_feature_falls_back_to_its_key_for_display():
    features = _discover_features(
        REAL_LIKE_COLUMNS, {"MyRatio": {"template": "AreaMorphologyM1"}}
    )
    assert features["MyRatio"]["display"] == "MyRatio"


def test_derived_feature_overrides_a_discovered_one_of_the_same_name():
    features = _discover_features(
        REAL_LIKE_COLUMNS,
        {
            "AreaMorphology": {
                "template": "AreaMorphologyM1",
                "display": "Area (fixed mask)",
            }
        },
    )
    assert features["AreaMorphology"]["has_m"] is False
    assert features["AreaMorphology"]["display"] == "Area (fixed mask)"


def test_derived_registry_may_be_none_or_empty():
    assert _discover_features(REAL_LIKE_COLUMNS, None) == _discover_features(
        REAL_LIKE_COLUMNS, {}
    )


def test_derived_entry_without_a_template_is_ignored():
    features = _discover_features(REAL_LIKE_COLUMNS, {"Bad": {}})
    assert "Bad" not in features


# Degenerate input
def test_no_columns_yields_no_features():
    assert _discover_features([]) == {}


def test_columns_without_a_mask_suffix_yield_no_features():
    assert _discover_features(["Position", "t", "Identification"]) == {}


# Small numeric helpers
def test_finite_max_and_min_ignore_nan_and_inf():
    values = [1.0, float("nan"), 5.0, float("inf"), -2.0]
    assert _finite_max(values) == 5.0
    assert _finite_min(values) == -2.0


def test_finite_helpers_fall_back_when_nothing_is_finite():
    assert _finite_max([float("nan")], default=7.0) == 7.0
    assert _finite_min([], default=-7.0) == -7.0


# X-axis column selection
def test_x_column_defaults_to_the_time_index():
    assert x_column_for(TIME_MODE_T, False, None, ["t"]) == "t"


def test_x_column_uses_calculated_time_when_present():
    columns = ["t", "Calculated_Time"]
    assert x_column_for(TIME_MODE_CALC, False, None, columns) == (
        "Calculated_Time"
    )


def test_x_column_falls_back_when_calculated_time_is_missing():
    assert x_column_for(TIME_MODE_CALC, False, None, ["t"]) == "t"


def test_x_column_picks_the_realtime_column_for_the_channel():
    columns = ["t", "RealTimeMinutes_Ch1", "RealTimeMinutes_Ch2"]
    assert x_column_for(TIME_MODE_REAL, True, 2, columns) == (
        "RealTimeMinutes_Ch2"
    )


def test_x_column_realtime_falls_back_to_the_first_channel():
    columns = ["t", "RealTimeMinutes_Ch1"]
    assert x_column_for(TIME_MODE_REAL, True, 9, columns) == (
        "RealTimeMinutes_Ch1"
    )


def test_x_column_realtime_falls_back_to_the_time_index():
    assert x_column_for(TIME_MODE_REAL, True, 1, ["t"]) == "t"


# Curve data extraction
def _plot_df() -> pd.DataFrame:
    """Two tracks; track 2 has a NaN gap and track 3 is entirely missing."""
    return pd.DataFrame(
        {
            "TrackNumber": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "t": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "Value": [
                10.0,
                11.0,
                12.0,
                20.0,
                float("nan"),
                22.0,
                float("nan"),
                float("nan"),
                float("nan"),
            ],
        }
    )


def test_track_xy_returns_finite_points_only():
    x, y = _track_xy(_plot_df(), 2, "t", "Value")
    assert x.tolist() == [0.0, 2.0]
    assert y.tolist() == [20.0, 22.0]


def test_track_xy_keeps_a_complete_track_intact():
    x, y = _track_xy(_plot_df(), 1, "t", "Value")
    assert x.tolist() == [0.0, 1.0, 2.0]
    assert y.tolist() == [10.0, 11.0, 12.0]


def test_track_xy_returns_none_for_an_all_nan_track():
    assert _track_xy(_plot_df(), 3, "t", "Value") is None


def test_track_xy_returns_none_for_an_absent_track():
    assert _track_xy(_plot_df(), 99, "t", "Value") is None


def test_track_xy_drops_points_with_a_missing_x():
    df = _plot_df()
    df["t"] = df["t"].astype(float)
    df.loc[df["TrackNumber"] == 1, "t"] = [0.0, float("nan"), 2.0]
    x, y = _track_xy(df, 1, "t", "Value")
    assert x.tolist() == [0.0, 2.0]
    assert y.tolist() == [10.0, 12.0]


# Draw order
def _colors(tracks):
    return {t: pg.mkColor("#112233") for t in tracks}


def test_plain_mode_draws_each_track_once():
    plan = list(
        _row_curve_plan(
            [1, 2, 3],
            highlight_on=False,
            color_map={},
            track_colors=_colors([1, 2, 3]),
            line_w=2,
            hl_line_w=4,
        )
    )
    assert [track for track, _ in plan] == [1, 2, 3]
    assert all(style.z is None for _, style in plan)


def test_highlight_mode_draws_a_white_base_layer_first():
    """Highlighted tracks are drawn twice: white underneath, colour on top."""
    plan = list(
        _row_curve_plan(
            [1, 2, 3],
            highlight_on=True,
            color_map={2: pg.mkColor("#FF0000")},
            track_colors=_colors([1, 2, 3]),
            line_w=2,
            hl_line_w=4,
        )
    )

    assert [track for track, _ in plan] == [1, 2, 3, 2]
    assert [style.z for _, style in plan] == [
        Z_HIGHLIGHT_BASE,
        Z_HIGHLIGHT_BASE,
        Z_HIGHLIGHT_BASE,
        Z_HIGHLIGHT_TOP,
    ]


def test_highlight_mode_shares_one_pen_for_the_base_layer():
    plan = list(
        _row_curve_plan(
            [1, 2],
            highlight_on=True,
            color_map={},
            track_colors=_colors([1, 2]),
            line_w=2,
            hl_line_w=4,
        )
    )
    assert plan[0][1].pen is plan[1][1].pen


def test_highlight_mode_without_any_painted_colors_draws_the_base_only():
    plan = list(
        _row_curve_plan(
            [1, 2],
            highlight_on=True,
            color_map={},
            track_colors=_colors([1, 2]),
            line_w=2,
            hl_line_w=4,
        )
    )
    assert [track for track, _ in plan] == [1, 2]


def test_plan_is_empty_when_no_tracks_are_visible():
    assert (
        list(
            _row_curve_plan(
                [],
                highlight_on=False,
                color_map={},
                track_colors={},
                line_w=2,
                hl_line_w=4,
            )
        )
        == []
    )


def test_default_track_color_is_only_a_safety_net():
    plan = list(
        _row_curve_plan(
            [7],
            highlight_on=False,
            color_map={},
            track_colors={},
            line_w=2,
            hl_line_w=4,
        )
    )
    assert plan[0][1].symbol_brush == pg.mkColor(DEFAULT_TRACK_COLOR)


# Axis sizing and styling
class _RecordingAxis:
    """Duck-typed stand-in for a pyqtgraph AxisItem."""

    def __init__(self, side, log):
        self.side, self.log = side, log

    def setStyle(self, **kw):
        self.log.append((self.side, "setStyle", sorted(kw)))

    def setPen(self, pen):
        self.log.append((self.side, "setPen"))

    def setTextPen(self, pen):
        self.log.append((self.side, "setTextPen"))

    def setTickPen(self, pen):
        self.log.append((self.side, "setTickPen"))


class _RecordingPlotItem:
    def __init__(self, left=True, bottom=True):
        self.log = []
        self._axes = {
            "left": _RecordingAxis("left", self.log) if left else None,
            "bottom": _RecordingAxis("bottom", self.log) if bottom else None,
        }

    def getAxis(self, side):
        return self._axes[side]


class _FakeMainWindow:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_row_axis_metrics_reads_the_configured_font():
    metrics = _row_axis_metrics(
        _FakeMainWindow(
            _plot_params={"axis_font_size": 14}, _shared_left_axis_width=42
        )
    )
    assert metrics == (14, 42, int(14 * 2.6))


def test_row_axis_metrics_falls_back_to_ten_point():
    axis_fs, _, _ = _row_axis_metrics(_FakeMainWindow(_plot_params={}))
    assert axis_fs == 10


def test_bottom_axis_height_grows_with_the_font():
    """Tick labels are clipped if the bottom axis does not scale."""
    _, _, small = _row_axis_metrics(
        _FakeMainWindow(_plot_params={"axis_font_size": 10})
    )
    _, _, large = _row_axis_metrics(
        _FakeMainWindow(_plot_params={"axis_font_size": 22})
    )
    assert large > small


def test_bottom_axis_height_has_a_floor():
    _, _, height = _row_axis_metrics(
        _FakeMainWindow(_plot_params={"axis_font_size": 4})
    )
    assert height == 24


def test_style_row_axes_sets_both_axes():
    plot_item = _RecordingPlotItem()
    _style_row_axes(plot_item, 12)

    left = [entry for entry in plot_item.log if entry[0] == "left"]
    bottom = [entry for entry in plot_item.log if entry[0] == "bottom"]
    assert ("left", "setPen") in left
    assert ("bottom", "setTickPen") in bottom
    # The tick font is applied last, after the pens.
    assert plot_item.log[-1] == ("bottom", "setStyle", ["tickFont"])


def test_style_row_axes_tolerates_a_missing_axis():
    plot_item = _RecordingPlotItem(left=False)
    _style_row_axes(plot_item, 12)  # must not raise
    assert all(entry[0] == "bottom" for entry in plot_item.log)


def test_row_xmax_reads_the_last_finite_value():
    df = pd.DataFrame({"t": [0.0, 1.0, 5.0]})
    assert _row_xmax(df, "t", default=99) == 5.0


def test_row_xmax_ignores_non_finite_values():
    df = pd.DataFrame({"t": [0.0, float("inf"), 3.0]})
    assert _row_xmax(df, "t", default=99) == 3.0


def test_row_xmax_falls_back_for_a_missing_column():
    df = pd.DataFrame({"t": [0.0, 1.0]})
    assert _row_xmax(df, "not_a_column", default=99) == 99


def test_row_xmax_falls_back_when_nothing_is_finite():
    df = pd.DataFrame({"t": [float("nan"), float("nan")]})
    assert _row_xmax(df, "t", default=99) == 99


# Feature combo synchronisation
class _FakeCombo:
    """Duck-typed QComboBox holding (text, data) pairs."""

    def __init__(self, items=()):
        self._items = list(items)

    def count(self):
        return len(self._items)

    def itemData(self, index):
        return self._items[index][1] if 0 <= index < len(self._items) else None


def test_no_reload_when_the_feature_list_is_unchanged():
    combo = _FakeCombo([("A", "A"), ("B", "B")])
    assert _feature_combo_needs_reload(combo, ["A", "B"]) is False


def test_reload_when_a_feature_is_added():
    combo = _FakeCombo([("A", "A")])
    assert _feature_combo_needs_reload(combo, ["A", "B"]) is True


def test_reload_when_a_feature_is_renamed():
    combo = _FakeCombo([("A", "A"), ("B", "B")])
    assert _feature_combo_needs_reload(combo, ["A", "C"]) is True


def test_reload_when_the_order_changes():
    combo = _FakeCombo([("A", "A"), ("B", "B")])
    assert _feature_combo_needs_reload(combo, ["B", "A"]) is True


def test_reload_from_an_empty_combo():
    assert _feature_combo_needs_reload(_FakeCombo(), ["A"]) is True


# Which feature gets selected after a rebuild
def test_the_users_previous_choice_wins():
    keys = ["AreaMorphology", DEFAULT_FEATURE, "Other"]
    assert _initial_feature_index(keys, "Other", features_defaulted=False) == 2


def test_first_draw_preselects_the_default_feature():
    keys = ["AreaMorphology", DEFAULT_FEATURE]
    assert _initial_feature_index(keys, None, features_defaulted=False) == 1


def test_later_rebuilds_do_not_reimpose_the_default():
    """Once a session has defaulted, a rebuild must not yank the selection."""
    keys = ["AreaMorphology", DEFAULT_FEATURE]
    assert _initial_feature_index(keys, None, features_defaulted=True) == 0


def test_first_draw_without_the_default_feature_falls_back_to_the_first():
    keys = ["AreaMorphology", "Other"]
    assert _initial_feature_index(keys, None, features_defaulted=False) == 0


def test_a_previous_choice_that_no_longer_exists_is_ignored():
    keys = ["AreaMorphology", DEFAULT_FEATURE]
    assert (
        _initial_feature_index(keys, "Deleted", features_defaulted=False) == 1
    )


def test_no_features_at_all_selects_index_zero():
    assert _initial_feature_index([], None, features_defaulted=False) == 0


# Which tracks are drawn
def test_all_tracks_are_visible_by_default():
    assert _visible_tracks([1, 2, 3], selected=set(), zoom=set()) == [1, 2, 3]


def test_selection_narrows_the_visible_tracks():
    assert _visible_tracks([1, 2, 3], selected={2, 3}, zoom=set()) == [2, 3]


def test_zoom_narrows_further_within_the_selection():
    assert _visible_tracks([1, 2, 3], selected={2, 3}, zoom={3}) == [3]


def test_zoom_applies_without_a_selection():
    assert _visible_tracks([1, 2, 3], selected=set(), zoom={1}) == [1]


def test_a_selection_matching_nothing_hides_everything():
    assert _visible_tracks([1, 2, 3], selected={99}, zoom=set()) == []


def test_visible_tracks_preserves_data_order():
    assert _visible_tracks([3, 1, 2], selected={1, 2, 3}, zoom=set()) == [
        3,
        1,
        2,
    ]


# Track colours
def test_every_track_gets_a_colour():
    colors = _track_color_map([1, 2, 3])
    assert set(colors) == {1, 2, 3}


def test_track_colours_are_distinct():
    colors = _track_color_map([1, 2, 3, 4])
    assert len({repr(c) for c in colors.values()}) == 4


def test_no_tracks_yields_no_colours():
    assert _track_color_map([]) == {}


# Shared y range
class _MainWindowStub:
    def __init__(self, features, channels=None, masks=None):
        self.selected_feature_by_row = features
        self.selected_ch_by_channel = channels or {}
        self.selected_m_by_channel = masks or {}


def test_global_y_range_spans_the_selected_columns():
    df = pd.DataFrame({"AM{m}".replace("{m}", "1"): [1.0, 9.0]})
    defs = {"A": {"template": "AM{m}"}}
    low, high = _global_y_range(
        _MainWindowStub({1: "A"}, masks={1: 1}), df, defs
    )
    assert (low, high) == (1.0, 9.0)


def test_global_y_range_falls_back_when_the_column_is_absent():
    df = pd.DataFrame({"Other": [1.0, 2.0]})
    defs = {"A": {"template": "AM{m}"}}
    assert _global_y_range(_MainWindowStub({1: "A"}), df, defs) == (0.0, 1.0)


def test_global_y_range_always_spans_at_least_one_unit():
    """A perfectly flat feature would otherwise give a zero-height axis."""
    df = pd.DataFrame({"AM1": [42.0, 42.0]})
    defs = {"A": {"template": "AM{m}"}}
    low, high = _global_y_range(
        _MainWindowStub({1: "A"}, masks={1: 1}), df, defs
    )
    assert high > low


def test_global_y_range_ignores_unknown_features():
    df = pd.DataFrame({"AM1": [1.0, 9.0]})
    defs = {"A": {"template": "AM{m}"}}
    assert _global_y_range(
        _MainWindowStub({1: "Ghost"}, masks={1: 1}), df, defs
    ) == (0.0, 1.0)


def test_global_y_range_ignores_non_finite_values():
    df = pd.DataFrame({"AM1": [1.0, float("inf"), 5.0]})
    defs = {"A": {"template": "AM{m}"}}
    low, high = _global_y_range(
        _MainWindowStub({1: "A"}, masks={1: 1}), df, defs
    )
    assert (low, high) == (1.0, 5.0)
