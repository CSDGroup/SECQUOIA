"""Tests for close-mask detection: the flag, the tab and the purple stars.

The data is a small tracking table with one identification over four time
points. Mask 1 and mask 2 each carry a second-candidate distance
(``alt_dist_px_m*``); mask 1 has candidates at t=1 (3 px) and t=2 (5 px),
mask 2 at t=3 (2 px).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from conftest import make_fake_main_window
from qtpy.QtCore import QPoint

from SECQUOIA.config import PLOTPARAMETERS, CloseMaskSettings
from SECQUOIA.core.outlier_detection import (
    RulesPack,
    apply_close_mask_detection,
    find_close_mask_cases,
    sync_close_mask_flag_to_track_df,
    update_unique_close_mask_ids,
)
from SECQUOIA.core.outlier_detection.close_masks import (
    CLOSE_MASK_FLAG,
    FLAG_FLAGGED,
    FLAG_OK,
    FLAG_REVIEWED,
    close_mask_hits,
    close_mask_summary,
    mask_indices_with_candidates,
    replay_close_mask_detection,
    reset_close_mask_state,
    unique_review_ids,
    update_flags_after_edit,
)
from SECQUOIA.core.quantification import (
    FeatureNaming,
    quantify,
    stale_measurement_columns,
)
from SECQUOIA.gui.lineage_tree.lineage_geometry import FeatureCatalog
from SECQUOIA.gui.outlier import navigation, outlier_list, reset
from SECQUOIA.gui.outlier.markers import (
    update_close_mask_marker,
    update_outlier_marker,
)
from SECQUOIA.utils.plotting.feature_discovery import _discover_features

NAN = np.nan


def close_frame() -> pd.DataFrame:
    """Four time points of one identification with second-mask distances."""
    return pd.DataFrame(
        {
            "Position": [1, 1, 1, 1],
            "Identification": ["exp-p0001-001"] * 4,
            "TrackNumber": [1, 1, 1, 1],
            "t": [0, 1, 2, 3],
            "AreaMorphologyM1": [10.0, 11.0, 12.0, 13.0],
            "alt_label_id_m1": pd.array([0, 7, 8, 0], dtype="Int64"),
            "alt_dist_px_m1": [NAN, 3.0, 5.0, NAN],
            "alt_label_id_m2": pd.array([0, 0, 0, 9], dtype="Int64"),
            "alt_dist_px_m2": [NAN, NAN, NAN, 2.0],
        }
    )


def flags(df: pd.DataFrame) -> list[str]:
    return df[CLOSE_MASK_FLAG].tolist()


def test_the_column_is_created_and_flags_every_close_row():
    df = close_frame()

    find_close_mask_cases(df, 5.0)

    assert flags(df) == [FLAG_OK, FLAG_FLAGGED, FLAG_FLAGGED, FLAG_FLAGGED]


def test_the_distance_boundary_is_inclusive():
    at_limit, just_below = close_frame(), close_frame()

    find_close_mask_cases(at_limit, 5.0, masks=[1])
    find_close_mask_cases(just_below, 4.99, masks=[1])

    assert flags(at_limit) == [FLAG_OK, FLAG_FLAGGED, FLAG_FLAGGED, FLAG_OK]
    assert flags(just_below) == [FLAG_OK, FLAG_FLAGGED, FLAG_OK, FLAG_OK]


def test_a_row_is_flagged_when_any_selected_mask_has_a_candidate():
    df = close_frame()

    find_close_mask_cases(df, 5.0, masks=[2])

    assert flags(df) == [FLAG_OK, FLAG_OK, FLAG_OK, FLAG_FLAGGED]


def test_all_masks_are_checked_when_none_is_given():
    assert mask_indices_with_candidates(close_frame().columns) == [1, 2]
    assert close_mask_hits(close_frame(), 5.0).tolist() == [
        False,
        True,
        True,
        True,
    ]


def test_a_mask_without_candidate_columns_is_ignored():
    df = close_frame()

    find_close_mask_cases(df, 5.0, masks=[1, 5])

    assert flags(df) == [FLAG_OK, FLAG_FLAGGED, FLAG_FLAGGED, FLAG_OK]


def test_reviewed_rows_stay_reviewed_when_the_detection_runs_again():
    df = close_frame()
    find_close_mask_cases(df, 5.0)
    df.loc[df["t"] == 1, CLOSE_MASK_FLAG] = FLAG_REVIEWED

    find_close_mask_cases(df, 5.0)

    assert flags(df) == [FLAG_OK, FLAG_REVIEWED, FLAG_FLAGGED, FLAG_FLAGGED]


def test_keep_reviewed_false_re_evaluates_reviewed_rows_too():
    df = close_frame()
    find_close_mask_cases(df, 5.0)
    df.loc[df["t"] == 1, CLOSE_MASK_FLAG] = FLAG_REVIEWED

    find_close_mask_cases(df, 5.0, keep_reviewed=False)

    assert flags(df) == [FLAG_OK, FLAG_FLAGGED, FLAG_FLAGGED, FLAG_FLAGGED]


def test_flagged_rows_that_no_longer_qualify_return_to_ok():
    df = close_frame()
    find_close_mask_cases(df, 5.0)

    find_close_mask_cases(df, 3.0)

    assert flags(df) == [FLAG_OK, FLAG_FLAGGED, FLAG_OK, FLAG_FLAGGED]


def test_unknown_or_missing_flag_values_are_read_as_ok():
    df = close_frame()
    df[CLOSE_MASK_FLAG] = ["maybe", None, FLAG_REVIEWED, NAN]

    find_close_mask_cases(df, 4.0)

    assert flags(df) == [FLAG_OK, FLAG_FLAGGED, FLAG_REVIEWED, FLAG_FLAGGED]


def test_requantifying_discards_the_flags(tmp_path, no_progress):
    main_window = make_fake_main_window(tmp_path)
    quantify(main_window, progress_cb=no_progress, max_pixel_distance=5)
    main_window.track_df[CLOSE_MASK_FLAG] = FLAG_REVIEWED

    quantify(main_window, progress_cb=no_progress, max_pixel_distance=5)

    assert (
        FLAG_REVIEWED
        not in main_window.track_df.get(
            CLOSE_MASK_FLAG, pd.Series(dtype=object)
        ).tolist()
    )


def make_session(df: pd.DataFrame | None = None) -> SimpleNamespace:
    df = close_frame() if df is None else df
    return SimpleNamespace(filtered_df=df.copy(), track_df=df.copy())


def test_unique_ids_list_only_identifications_with_a_flag():
    df = pd.concat(
        [
            close_frame(),
            close_frame().assign(Identification="exp-p0001-002"),
        ],
        ignore_index=True,
    )
    df[CLOSE_MASK_FLAG] = FLAG_OK
    df.loc[[1, 5], CLOSE_MASK_FLAG] = [FLAG_FLAGGED, FLAG_REVIEWED]
    main_window = make_session(df)

    update_unique_close_mask_ids(main_window)

    assert main_window.unique_close_mask_ids == ["exp-p0001-001"]


def test_unique_ids_are_empty_without_flags():
    main_window = make_session()

    update_unique_close_mask_ids(main_window)

    assert main_window.unique_close_mask_ids == []


def test_the_flag_is_synced_into_the_track_table():
    main_window = make_session()
    find_close_mask_cases(main_window.filtered_df, 5.0)

    sync_close_mask_flag_to_track_df(main_window)

    assert flags(main_window.track_df) == [
        FLAG_OK,
        FLAG_FLAGGED,
        FLAG_FLAGGED,
        FLAG_FLAGGED,
    ]


def test_syncing_leaves_other_positions_alone():
    other = close_frame().assign(Position=2)
    other[CLOSE_MASK_FLAG] = FLAG_REVIEWED
    track_df = pd.concat([close_frame(), other], ignore_index=True)
    main_window = SimpleNamespace(filtered_df=close_frame(), track_df=track_df)
    find_close_mask_cases(main_window.filtered_df, 5.0)

    sync_close_mask_flag_to_track_df(main_window)

    synced = main_window.track_df
    assert flags(synced[synced["Position"] == 1]) == [
        FLAG_OK,
        FLAG_FLAGGED,
        FLAG_FLAGGED,
        FLAG_FLAGGED,
    ]
    assert flags(synced[synced["Position"] == 2]) == [FLAG_REVIEWED] * 4


def test_syncing_carries_reviewed_over_and_clears_what_no_longer_flags():
    main_window = make_session()
    main_window.track_df[CLOSE_MASK_FLAG] = FLAG_FLAGGED
    main_window.filtered_df[CLOSE_MASK_FLAG] = [
        FLAG_OK,
        FLAG_REVIEWED,
        FLAG_OK,
        FLAG_FLAGGED,
    ]

    sync_close_mask_flag_to_track_df(main_window)

    assert flags(main_window.track_df) == [
        FLAG_OK,
        FLAG_REVIEWED,
        FLAG_OK,
        FLAG_FLAGGED,
    ]


def test_apply_detection_updates_flags_track_table_and_ids():
    main_window = make_session()

    n_flagged = apply_close_mask_detection(main_window, 5.0)

    assert n_flagged == 3
    assert flags(main_window.track_df).count(FLAG_FLAGGED) == 3
    assert main_window.unique_close_mask_ids == ["exp-p0001-001"]


def test_apply_detection_without_data_flags_nothing():
    main_window = SimpleNamespace(filtered_df=None, track_df=None)

    assert apply_close_mask_detection(main_window, 5.0) == 0
    assert main_window.unique_close_mask_ids == []


def test_the_summary_counts_flagged_reviewed_and_identifications():
    df = close_frame()
    df[CLOSE_MASK_FLAG] = [FLAG_OK, FLAG_FLAGGED, FLAG_REVIEWED, FLAG_FLAGGED]

    assert close_mask_summary(df) == {
        "flagged": 2,
        "reviewed": 1,
        "identifications": 1,
    }
    assert close_mask_summary(close_frame())["flagged"] == 0
    assert close_mask_summary(None)["flagged"] == 0


# The flag is not a measurement
def test_the_flag_column_is_never_pruned_as_a_stale_measurement():
    naming = FeatureNaming.for_config(["C01"], basic=False)

    stale = stale_measurement_columns([CLOSE_MASK_FLAG], naming, [1])

    assert stale == []


def test_the_flag_column_stays_out_of_the_feature_lists():
    df = close_frame()
    find_close_mask_cases(df, 5.0)
    columns = list(df.columns)

    assert CLOSE_MASK_FLAG not in _discover_features(columns)
    assert CLOSE_MASK_FLAG not in FeatureCatalog.from_columns(columns).features


class StarWindow(SimpleNamespace):
    """The parts of a main window the star markers read."""


@pytest.fixture
def star_window(qtbot):
    plot = pg.PlotWidget()
    qtbot.addWidget(plot)
    df = close_frame()
    df["Outlier_detection"] = ["OK", "OK", "Outlier", "OK"]
    df[CLOSE_MASK_FLAG] = [FLAG_OK, FLAG_FLAGGED, FLAG_FLAGGED, FLAG_REVIEWED]
    return StarWindow(
        filtered_df=df,
        unique_ids=["exp-p0001-001"],
        current_ident_index=0,
        n_channels=1,
        n_masks=2,
        _max_plot_rows=1,
        plot_widget_1=plot,
        _feature_defs={
            "AreaMorphology": {
                "template": "AreaMorphologyM{m}",
                "has_ch": False,
                "has_m": True,
            }
        },
        selected_feature_by_row={1: "AreaMorphology"},
        selected_m_by_channel={1: 1},
        selected_ch_by_channel={1: 1},
        _time_mode="t",
    )


def scatter_items(plot: pg.PlotWidget) -> list:
    """The scatter (star) items currently on `plot`."""
    return [
        item
        for item in plot.getPlotItem().items
        if isinstance(item, pg.ScatterPlotItem)
    ]


def star_item(main_window, attr: str):
    return getattr(main_window, attr)[1]["item"]


def test_purple_stars_sit_on_the_flagged_time_points(star_window):
    update_close_mask_marker(star_window)

    item = star_item(star_window, "_close_mask_marker_items")
    assert item.data["x"].tolist() == [1.0, 2.0]
    assert item.data["y"].tolist() == [11.0, 12.0]


def test_purple_stars_are_purple_and_orange_stars_stay_orange(star_window):
    update_outlier_marker(star_window)

    purple = star_item(star_window, "_close_mask_marker_items")
    orange = star_item(star_window, "_outlier_marker_items")
    assert purple.opts["pen"].color().getRgb() == PLOTPARAMETERS.PURPLE
    assert orange.opts["pen"].color().getRgb() == PLOTPARAMETERS.ORANGE
    assert orange.data["x"].tolist() == [2.0]


def test_a_purple_star_on_an_outlier_is_larger_and_drawn_below(star_window):
    update_outlier_marker(star_window)

    purple = star_item(star_window, "_close_mask_marker_items")
    orange = star_item(star_window, "_outlier_marker_items")
    small, large = purple.data["size"].tolist()
    assert small == PLOTPARAMETERS.OUTLIERSIZE
    assert large > PLOTPARAMETERS.OUTLIERSIZE
    assert purple.zValue() < orange.zValue()


def test_reviewed_rows_get_no_star(star_window):
    update_close_mask_marker(star_window)

    item = star_item(star_window, "_close_mask_marker_items")
    assert 3.0 not in item.data["x"].tolist()


def test_redrawing_replaces_the_stars_instead_of_stacking_them(star_window):
    update_close_mask_marker(star_window)
    update_close_mask_marker(star_window)

    assert len(scatter_items(star_window.plot_widget_1)) == 1


def test_stars_disappear_once_nothing_is_flagged(star_window):
    update_close_mask_marker(star_window)
    star_window.filtered_df[CLOSE_MASK_FLAG] = FLAG_REVIEWED

    update_close_mask_marker(star_window)

    assert star_window._close_mask_marker_items == {}
    assert scatter_items(star_window.plot_widget_1) == []


def test_stars_are_cleared_when_the_flag_column_is_gone(star_window):
    update_close_mask_marker(star_window)
    star_window.filtered_df = star_window.filtered_df.drop(
        columns=[CLOSE_MASK_FLAG]
    )

    update_close_mask_marker(star_window)

    assert star_window._close_mask_marker_items == {}


def test_only_the_current_identification_is_drawn(star_window):
    other = star_window.filtered_df.assign(Identification="exp-p0001-002")
    star_window.filtered_df = pd.concat(
        [star_window.filtered_df, other], ignore_index=True
    )

    update_close_mask_marker(star_window)

    item = star_item(star_window, "_close_mask_marker_items")
    assert item.data["x"].tolist() == [1.0, 2.0]


@pytest.fixture
def close_window(dialog_main_window, qtbot):
    from SECQUOIA.gui.outlier.detection.window import OutlierDetectionWindow

    dialog_main_window.threshold = 5
    for mask, dists in ((1, [NAN, 3.0, 5.0, NAN]), (2, [NAN, NAN, NAN, 2.0])):
        dialog_main_window.filtered_df[f"alt_dist_px_m{mask}"] = dists
        dialog_main_window.filtered_df[f"alt_label_id_m{mask}"] = pd.array(
            [0 if np.isnan(d) else 4 for d in dists], dtype="Int64"
        )
    dialog_main_window.track_df = dialog_main_window.filtered_df.copy()

    window = OutlierDetectionWindow(dialog_main_window)
    qtbot.addWidget(window)
    return window


def tab_labels(window) -> list[str]:
    return [window.tabs.tabText(i) for i in range(window.tabs.count())]


def test_the_window_has_a_close_masks_tab(close_window):
    assert "Close masks" in tab_labels(close_window)


def test_the_close_masks_tab_follows_the_sliding_window_tab(close_window):
    assert tab_labels(close_window) == [
        "Threshold Rules",
        "Sliding window",
        "Close masks",
        "Load Rules",
        "Run",
    ]


def test_every_tab_still_shows_its_own_page(close_window):
    tabs = close_window.tabs
    pages = {
        "Threshold Rules": close_window.threshold_tab.page,
        "Sliding window": close_window.sliding_tab.page,
        "Close masks": close_window.close_mask_tab.page,
        "Load Rules": close_window.load_tab.page,
        "Run": close_window.run_tab.page,
    }

    for index, label in enumerate(tab_labels(close_window)):
        assert tabs.widget(index) is pages[label]


def test_the_run_summary_still_refreshes_when_the_run_tab_is_shown(
    close_window,
):
    close_window.close_mask_tab.find_btn.click()
    close_window.tabs.setCurrentWidget(close_window.run_tab.page)

    assert "CLOSE MASKS" in close_window.run_tab.summary_box.toPlainText()


def test_the_tab_starts_at_the_matching_tolerance_which_is_also_the_maximum(
    close_window,
):
    tab = close_window.close_mask_tab

    assert tab.dist_spin.value() == 5.0
    assert tab.dist_spin.maximum() == 5.0
    assert tab.effective_threshold() == 5.0


def test_the_distance_cannot_go_above_the_tolerance(close_window):
    tab = close_window.close_mask_tab

    tab.dist_spin.setValue(12.0)

    assert tab.dist_spin.value() == 5.0
    assert tab.effective_threshold() == 5.0

    tab.dist_spin.setValue(4.0)

    assert tab.effective_threshold() == 4.0


def test_there_is_no_warning_note_about_the_tolerance(close_window):
    tab = close_window.close_mask_tab

    tab.dist_spin.setValue(12.0)

    assert not hasattr(tab, "cap_note")
    assert "Re-quantify" not in tab.result_label.text()


def test_next_on_the_sliding_window_tab_goes_to_close_masks(close_window):
    close_window.tabs.setCurrentWidget(close_window.sliding_tab.page)

    close_window.sliding_tab.next_btn.click()

    assert (
        close_window.tabs.currentWidget() is close_window.close_mask_tab.page
    )


def test_next_on_the_close_masks_tab_goes_to_run(close_window):
    close_window.tabs.setCurrentWidget(close_window.close_mask_tab.page)

    close_window.close_mask_tab.next_btn.click()

    assert close_window.tabs.currentWidget() is close_window.run_tab.page


def test_no_mask_checked_means_every_mask(close_window):
    tab = close_window.close_mask_tab

    assert tab.selected_masks() is None

    tab.mask_combo.set_checked_values([2])

    assert tab.selected_masks() == [2]


def test_find_flags_the_rows_and_reports_the_count(
    close_window, dialog_main_window
):
    tab = close_window.close_mask_tab

    tab.find_btn.click()

    assert close_mask_summary(dialog_main_window.filtered_df)["flagged"] == 3
    assert "3 flagged" in tab.result_label.text()
    assert flags(dialog_main_window.track_df).count(FLAG_FLAGGED) == 3


def test_find_respects_the_distance_and_mask_selection(
    close_window, dialog_main_window
):
    tab = close_window.close_mask_tab
    tab.dist_spin.setValue(3.0)
    tab.mask_combo.set_checked_values([1])

    tab.find_btn.click()

    assert flags(dialog_main_window.filtered_df) == [
        FLAG_OK,
        FLAG_FLAGGED,
        FLAG_OK,
        FLAG_OK,
    ]


def test_find_does_not_touch_the_outlier_flags(
    close_window, dialog_main_window
):
    dialog_main_window.filtered_df["Outlier_detection"] = [
        "OK",
        "Outlier",
        "OK",
        "OK",
    ]

    close_window.close_mask_tab.find_btn.click()

    assert dialog_main_window.filtered_df["Outlier_detection"].tolist() == [
        "OK",
        "Outlier",
        "OK",
        "OK",
    ]


def test_reset_removes_the_flags_instead_of_finding_them_again(
    close_window, dialog_main_window
):
    tab = close_window.close_mask_tab
    tab.find_btn.click()
    assert close_mask_summary(dialog_main_window.filtered_df)["flagged"] == 3

    tab.reset_btn.click()

    for df in (dialog_main_window.filtered_df, dialog_main_window.track_df):
        assert (df[CLOSE_MASK_FLAG] == FLAG_OK).all()
    assert close_mask_summary(dialog_main_window.filtered_df)["flagged"] == 0
    assert "removed" in tab.result_label.text()


def test_the_reset_button_removes_reviewed_flags_too(
    close_window, dialog_main_window
):
    tab = close_window.close_mask_tab
    tab.find_btn.click()
    dialog_main_window.filtered_df[CLOSE_MASK_FLAG] = FLAG_REVIEWED

    tab.reset_btn.click()

    assert close_mask_summary(dialog_main_window.filtered_df)["reviewed"] == 0


def test_reset_stops_the_detection_until_find_is_pressed_again(
    close_window, dialog_main_window
):
    tab = close_window.close_mask_tab
    tab.find_btn.click()
    assert tab.in_use()

    tab.reset_btn.click()

    assert not tab.in_use()
    assert tab.settings_for_rules() is None
    assert "close_masks" not in dialog_main_window._last_outlier_rules

    tab.find_btn.click()

    assert close_mask_summary(dialog_main_window.filtered_df)["flagged"] == 3


def test_reset_leaves_the_outlier_flags_alone(
    close_window, dialog_main_window
):
    dialog_main_window.filtered_df["Outlier_detection"] = [
        "OK",
        "Outlier",
        "OK",
        "OK",
    ]
    tab = close_window.close_mask_tab
    tab.find_btn.click()

    tab.reset_btn.click()

    assert dialog_main_window.filtered_df["Outlier_detection"].tolist() == [
        "OK",
        "Outlier",
        "OK",
        "OK",
    ]


def test_the_tab_asks_for_quantification_when_no_candidates_exist(
    dialog_main_window, qtbot
):
    from SECQUOIA.gui.outlier.detection.window import OutlierDetectionWindow

    dialog_main_window.threshold = 5
    window = OutlierDetectionWindow(dialog_main_window)
    qtbot.addWidget(window)

    window.close_mask_tab.find_btn.click()

    assert "quantification" in window.close_mask_tab.result_label.text()
    assert CLOSE_MASK_FLAG not in dialog_main_window.filtered_df.columns


def test_the_run_tab_summarises_the_close_mask_flags(
    close_window, dialog_main_window
):
    close_window.close_mask_tab.find_btn.click()

    close_window.run_tab.refresh_summary()

    text = close_window.run_tab.summary_box.toPlainText()
    assert "CLOSE MASKS" in text
    assert "3 flagged time point(s) in 1 identification(s)" in text


def test_finding_close_masks_switches_the_list_to_the_out_view(
    close_window, dialog_main_window, qtbot
):
    from qtpy.QtWidgets import QCheckBox

    dialog_main_window.Outliers = QCheckBox()
    qtbot.addWidget(dialog_main_window.Outliers)

    close_window.close_mask_tab.find_btn.click()

    assert dialog_main_window.Outliers.isChecked()


def test_finding_nothing_leaves_the_view_alone(
    close_window, dialog_main_window, qtbot
):
    from qtpy.QtWidgets import QCheckBox

    dialog_main_window.Outliers = QCheckBox()
    qtbot.addWidget(dialog_main_window.Outliers)
    close_window.close_mask_tab.dist_spin.setValue(0.5)

    close_window.close_mask_tab.find_btn.click()

    assert not dialog_main_window.Outliers.isChecked()


def test_the_review_ids_list_outliers_then_close_masks_without_repeats():
    window = SimpleNamespace(
        unique_outliers_ids=["a", "b"], unique_close_mask_ids=["b", "c"]
    )

    assert unique_review_ids(window) == ["a", "b", "c"]


def test_the_review_ids_work_with_either_list_missing_or_an_array():
    assert unique_review_ids(SimpleNamespace()) == []
    assert unique_review_ids(
        SimpleNamespace(unique_outliers_ids=np.array(["a"]))
    ) == ["a"]
    assert unique_review_ids(SimpleNamespace(unique_close_mask_ids=["z"])) == [
        "z"
    ]


def test_review_ids_compare_identifications_as_text():
    window = SimpleNamespace(
        unique_outliers_ids=[1], unique_close_mask_ids=["1", "2"]
    )

    assert unique_review_ids(window) == [1, "2"]


@pytest.fixture
def list_window(fake_main_window_widget, qtbot):
    from qtpy.QtWidgets import QCheckBox, QTreeWidget

    rows = []
    for number in (1, 2, 3):
        for t in range(2):
            rows.append(
                {
                    "Position": 1,
                    "Identification": f"exp-p0001-00{number}",
                    "TrackNumber": 1,
                    "t": t,
                    "active": 1,
                    "inspected": 0,
                    "Outlier_detection": "OK",
                    CLOSE_MASK_FLAG: FLAG_OK,
                }
            )
    df = pd.DataFrame(rows)
    window = fake_main_window_widget(
        filtered_df=df,
        track_df=df.copy(),
        tree_widget=QTreeWidget(),
        ALL=QCheckBox(),
        Outliers=QCheckBox(),
        unique_outliers_ids=[],
        unique_close_mask_ids=[],
        folder_list=["x"],
    )
    qtbot.addWidget(window.tree_widget)
    return window


def listed(window) -> list[str]:
    tree = window.tree_widget
    return [
        tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())
    ]


def flag_ident(window, number: int, value: str = FLAG_FLAGGED) -> None:
    df = window.filtered_df
    df.loc[
        df["Identification"] == f"exp-p0001-00{number}", CLOSE_MASK_FLAG
    ] = value


def test_the_out_list_shows_outliers_and_close_mask_cases(list_window):
    list_window.unique_outliers_ids = ["exp-p0001-001"]
    flag_ident(list_window, 3)

    outlier_list._update_outlier_list(list_window, interactive=False)

    assert listed(list_window) == ["001", "003"]


def test_the_out_list_works_with_close_masks_alone(list_window):
    flag_ident(list_window, 2)

    outlier_list._update_outlier_list(list_window, interactive=False)

    assert listed(list_window) == ["002"]


def test_an_identification_with_both_is_listed_once(list_window):
    list_window.unique_outliers_ids = ["exp-p0001-002"]
    flag_ident(list_window, 2)

    outlier_list._update_outlier_list(list_window, interactive=False)

    assert listed(list_window) == ["002"]


def test_reviewed_close_masks_drop_out_of_the_list(list_window):
    flag_ident(list_window, 1)
    flag_ident(list_window, 2, FLAG_REVIEWED)

    outlier_list._update_outlier_list(list_window, interactive=False)

    assert listed(list_window) == ["001"]


def test_an_empty_out_list_says_so_and_returns_to_all(
    list_window, monkeypatch
):
    shown = []
    monkeypatch.setattr(
        outlier_list,
        "_show_info_dialog",
        lambda window, title, text: shown.append(text),
    )

    outlier_list._update_outlier_list(list_window)

    assert shown == ["No outliers or close masks have been found!"]
    assert list_window.ALL.isChecked()


def test_the_active_out_list_follows_new_flags(list_window):
    list_window.Outliers.setChecked(True)
    flag_ident(list_window, 1)
    outlier_list._rebuild_active_list(list_window)
    assert listed(list_window) == ["001"]

    flag_ident(list_window, 3)
    outlier_list._rebuild_active_list(list_window)

    assert listed(list_window) == ["001", "003"]


def test_the_out_list_falls_back_to_all_once_nothing_is_left(list_window):
    list_window.Outliers.setChecked(True)
    flag_ident(list_window, 1)
    outlier_list._rebuild_active_list(list_window)

    flag_ident(list_window, 1, FLAG_REVIEWED)
    outlier_list._rebuild_active_list(list_window)

    assert list_window.ALL.isChecked()


def test_the_all_list_is_untouched_by_the_flags(list_window):
    flag_ident(list_window, 1)

    outlier_list._rebuild_active_list(list_window)

    assert listed(list_window) == ["001", "002", "003"]


def test_the_outlier_jump_visits_close_mask_cases_too(monkeypatch):
    seen = []
    monkeypatch.setattr(
        navigation, "go_to_ident", lambda window, ident: seen.append(ident)
    )
    df = pd.DataFrame(
        {
            "Identification": ["A", "B", "C"],
            CLOSE_MASK_FLAG: [FLAG_OK, FLAG_FLAGGED, FLAG_FLAGGED],
        }
    )
    window = SimpleNamespace(
        track_df=df,
        filtered_df=df,
        unique_outliers_ids=[],
        current_outlier_index=0,
    )

    navigation.change_outlier(window, "next")
    navigation.change_outlier(window, "next")
    navigation.change_outlier(window, "previous")

    assert seen == ["C", "B", "C"]


def test_the_outlier_jump_needs_some_detection(monkeypatch, caplog):
    seen = []
    monkeypatch.setattr(
        navigation, "go_to_ident", lambda window, ident: seen.append(ident)
    )
    window = SimpleNamespace(
        track_df=pd.DataFrame({"Identification": ["A"]}),
        unique_outliers_ids=[],
        current_outlier_index=0,
    )

    navigation.change_outlier(window, "next")

    assert seen == []
    assert "perform outlier detection first" in caplog.text


def reset_session(star_window) -> SimpleNamespace:
    """The star window, flagged, with the state a detection leaves behind."""
    df = star_window.filtered_df
    star_window.track_df = df.copy()
    star_window.unique_outliers_ids = ["exp-p0001-001"]
    star_window.unique_close_mask_ids = ["exp-p0001-001"]
    star_window.close_mask_threshold = 5.0
    star_window.close_mask_masks = [1]
    star_window.close_mask_pinned_row = ("exp-p0001-001", 1, 2)
    return star_window


def reset_all(window, **kwargs) -> None:
    reset.reset_outlier_state(
        window, refresh_plots=False, show_message=False, **kwargs
    )


def test_reset_clears_the_close_mask_flags_in_both_tables(star_window):
    window = reset_session(star_window)

    reset_all(window)

    for df in (window.filtered_df, window.track_df):
        assert (df[CLOSE_MASK_FLAG] == FLAG_OK).all()
        assert (df["Outlier_detection"] == "OK").all()


def test_reset_removes_reviewed_flags_too(star_window):
    window = reset_session(star_window)
    assert FLAG_REVIEWED in window.filtered_df[CLOSE_MASK_FLAG].tolist()

    reset_all(window)

    assert FLAG_REVIEWED not in window.filtered_df[CLOSE_MASK_FLAG].tolist()


def test_reset_forgets_the_list_the_run_and_the_pinned_row(star_window):
    window = reset_session(star_window)

    reset_all(window)

    assert window.unique_close_mask_ids == []
    assert window.unique_outliers_ids == []
    assert window.close_mask_threshold is None
    assert window.close_mask_masks is None
    assert window.close_mask_pinned_row is None


def test_reset_removes_the_purple_stars(star_window):
    window = reset_session(star_window)
    update_close_mask_marker(window)
    assert scatter_items(window.plot_widget_1)

    reset_all(window)

    assert window._close_mask_marker_items == {}
    assert scatter_items(window.plot_widget_1) == []


def test_reset_clears_the_napari_highlight(star_window, monkeypatch):
    window = reset_session(star_window)
    cleared = []
    monkeypatch.setattr(
        reset, "clear_close_mask_view", lambda w: cleared.append(w)
    )

    reset_all(window)

    assert cleared == [window]


def test_reset_is_harmless_without_any_close_mask_state(fake_main_window):
    window = fake_main_window(
        filtered_df=pd.DataFrame({"Identification": ["A"], "t": [0]}),
        track_df=pd.DataFrame({"Identification": ["A"], "t": [0]}),
    )

    reset_all(window)

    assert CLOSE_MASK_FLAG not in window.track_df.columns
    assert window.unique_close_mask_ids == []


def test_an_edit_after_a_reset_never_flags_again(star_window):
    window = reset_session(star_window)
    reset_all(window)
    window.track_df.loc[:, "alt_dist_px_m1"] = 1.0
    window.track_df.loc[:, "alt_label_id_m1"] = pd.array(
        [3] * 4, dtype="Int64"
    )

    update_flags_after_edit(window, window.track_df.index, edited_row=0)

    assert (window.track_df[CLOSE_MASK_FLAG] == FLAG_OK).all()


def test_the_detection_can_be_run_again_after_a_reset(star_window):
    window = reset_session(star_window)
    reset_all(window)

    n_flagged = apply_close_mask_detection(window, 5.0)

    assert n_flagged == 3
    assert close_mask_summary(window.filtered_df)["reviewed"] == 0


def test_resetting_the_state_alone_works_on_plain_tables():
    window = SimpleNamespace(
        filtered_df=close_frame().assign(**{CLOSE_MASK_FLAG: FLAG_FLAGGED}),
        track_df=close_frame().assign(**{CLOSE_MASK_FLAG: FLAG_REVIEWED}),
    )

    reset_close_mask_state(window)

    assert (window.filtered_df[CLOSE_MASK_FLAG] == FLAG_OK).all()
    assert (window.track_df[CLOSE_MASK_FLAG] == FLAG_OK).all()


def rules_pack(**kwargs) -> RulesPack:
    return RulesPack(
        version=1, m_n=2, ch_n=2, rules=[], sliding_windows=[], **kwargs
    )


def test_a_rules_pack_without_close_masks_saves_none():
    assert rules_pack().to_dict()["close_masks"] is None


def test_close_mask_settings_survive_a_save_and_load():
    pack = rules_pack(close_masks=CloseMaskSettings(3.5, [1, 2]))

    restored = RulesPack.from_dict(json.loads(json.dumps(pack.to_dict())))

    assert restored.close_masks == CloseMaskSettings(3.5, [1, 2])
    assert restored == pack


def test_all_masks_are_saved_as_none():
    payload = rules_pack(close_masks=CloseMaskSettings(2.0)).to_dict()

    assert payload["close_masks"] == {"distance": 2.0, "masks": None}
    assert RulesPack.from_dict(payload).close_masks.masks is None


def test_rules_files_from_before_close_masks_still_load():
    old = rules_pack().to_dict()
    del old["close_masks"]

    assert RulesPack.from_dict(old).close_masks is None


@pytest.mark.parametrize(
    "damaged",
    [
        "yes",
        {},
        {"distance": "far"},
        {"distance": -1},
        {"distance": float("nan")},
        {"distance": 3, "masks": ["x"]},
    ],
)
def test_damaged_close_mask_entries_are_ignored_not_fatal(damaged):
    payload = rules_pack().to_dict()
    payload["close_masks"] = damaged

    assert RulesPack.from_dict(payload).close_masks is None


def test_the_rules_snapshot_carries_the_tab_settings_once_in_use(
    close_window, dialog_main_window
):
    from SECQUOIA.gui.outlier.rules import snapshot_outlier_ui_to_pack

    assert snapshot_outlier_ui_to_pack(dialog_main_window).close_masks is None

    tab = close_window.close_mask_tab
    tab.dist_spin.setValue(3.0)
    tab.mask_combo.set_checked_values([2])
    tab.find_btn.click()

    pack = snapshot_outlier_ui_to_pack(dialog_main_window)

    assert pack.close_masks == CloseMaskSettings(3.0, [2])


def test_finding_keeps_the_settings_with_the_saved_rules(
    close_window, dialog_main_window
):
    close_window.close_mask_tab.find_btn.click()

    saved = dialog_main_window._last_outlier_rules["close_masks"]

    assert saved == {"distance": 5.0, "masks": None}


def test_ctrl_r_forgets_the_close_mask_settings_of_the_saved_rules(
    close_window, dialog_main_window
):
    close_window.close_mask_tab.find_btn.click()

    reset.reset_outlier_state(
        dialog_main_window, refresh_plots=False, show_message=False
    )

    assert "close_masks" not in (
        getattr(dialog_main_window, "_last_outlier_rules", None) or {}
    )
    assert dialog_main_window.close_mask_threshold is None


def test_apply_on_the_run_tab_detects_close_masks_and_saves_the_settings(
    close_window, dialog_main_window, qtbot, monkeypatch
):
    from qtpy.QtWidgets import QCheckBox, QTreeWidget

    from SECQUOIA.gui.outlier.detection import run_tab

    monkeypatch.setattr(run_tab, "auto_select_first_item", lambda w: None)

    dialog_main_window.Outliers = QCheckBox()
    dialog_main_window.tree_widget = QTreeWidget()
    dialog_main_window.folder_list = ["x"]
    qtbot.addWidget(dialog_main_window.tree_widget)
    tab = close_window.close_mask_tab
    tab.apply_settings(CloseMaskSettings(5.0, None))  # chosen, not yet found
    assert close_mask_summary(dialog_main_window.filtered_df)["flagged"] == 0

    close_window.run_tab._on_apply()

    assert close_mask_summary(dialog_main_window.filtered_df)["flagged"] == 3
    assert flags(dialog_main_window.track_df).count(FLAG_FLAGGED) == 3
    saved = sorted(
        Path(dialog_main_window.folder).rglob("Outlier_detection/rules_*.json")
    )
    assert saved, "no rules file was written"
    assert json.loads(saved[-1].read_text())["close_masks"] == {
        "distance": 5.0,
        "masks": None,
    }


def test_apply_without_close_mask_settings_flags_nothing(
    close_window, dialog_main_window, qtbot, monkeypatch
):
    from qtpy.QtWidgets import QCheckBox, QTreeWidget

    from SECQUOIA.gui.outlier.detection import run_tab

    monkeypatch.setattr(run_tab, "auto_select_first_item", lambda w: None)
    monkeypatch.setattr(
        outlier_list, "_show_info_dialog", lambda *args, **kwargs: None
    )

    dialog_main_window.Outliers = QCheckBox()
    dialog_main_window.tree_widget = QTreeWidget()
    dialog_main_window.folder_list = ["x"]
    qtbot.addWidget(dialog_main_window.tree_widget)

    close_window.run_tab._on_apply()

    assert close_mask_summary(dialog_main_window.filtered_df)["flagged"] == 0


def test_loading_rules_fills_the_tab_and_makes_the_settings_current(
    close_window, dialog_main_window
):
    payload = rules_pack(close_masks=CloseMaskSettings(3.0, [1])).to_dict()

    close_window.load_tab._apply_loaded_rules(payload)

    tab = close_window.close_mask_tab
    assert tab.dist_spin.value() == 3.0
    assert tab.selected_masks() == [1]
    assert dialog_main_window.close_mask_threshold == 3.0
    assert dialog_main_window.close_mask_masks == [1]
    assert dialog_main_window._last_outlier_rules["close_masks"] == {
        "distance": 3.0,
        "masks": [1],
    }


def test_loaded_distances_above_the_tolerance_are_capped(
    close_window, dialog_main_window
):
    payload = rules_pack(close_masks=CloseMaskSettings(50.0)).to_dict()

    close_window.load_tab._apply_loaded_rules(payload)

    assert close_window.close_mask_tab.dist_spin.value() == 5.0
    assert dialog_main_window.close_mask_threshold == 5.0


def test_loading_rules_without_close_masks_leaves_the_tab_alone(
    close_window, dialog_main_window
):
    tab = close_window.close_mask_tab
    tab.dist_spin.setValue(2.0)
    tab.find_btn.click()

    close_window.load_tab._apply_loaded_rules(rules_pack().to_dict())

    assert tab.dist_spin.value() == 2.0
    assert dialog_main_window.close_mask_threshold == 2.0


def test_a_reopened_window_starts_from_the_settings_in_use(
    dialog_main_window, qtbot
):
    from SECQUOIA.gui.outlier.detection.window import OutlierDetectionWindow

    dialog_main_window.threshold = 5
    dialog_main_window.close_mask_threshold = 3.0
    dialog_main_window.close_mask_masks = [2]

    window = OutlierDetectionWindow(dialog_main_window)
    qtbot.addWidget(window)

    assert window.close_mask_tab.dist_spin.value() == 3.0
    assert window.close_mask_tab.selected_masks() == [2]


def test_the_run_summary_lists_the_settings(close_window):
    tab = close_window.close_mask_tab
    close_window.run_tab.refresh_summary()
    assert "not used" in close_window.run_tab.summary_box.toPlainText()

    tab.dist_spin.setValue(3.0)
    tab.mask_combo.set_checked_values([1, 2])
    tab.find_btn.click()
    close_window.run_tab.refresh_summary()

    text = close_window.run_tab.summary_box.toPlainText()
    assert "distance <= 3 px | masks: 1, 2" in text


def test_a_position_switch_repeats_the_saved_detection():
    window = make_session()
    pack = rules_pack(close_masks=CloseMaskSettings(5.0, None))

    n_flagged = replay_close_mask_detection(window, pack)

    assert n_flagged == 3
    assert window.close_mask_threshold == 5.0
    assert flags(window.track_df).count(FLAG_FLAGGED) == 3
    assert window.unique_close_mask_ids == ["exp-p0001-001"]


def test_a_position_switch_keeps_the_reviewed_cases_it_already_has():
    window = make_session()
    window.filtered_df[CLOSE_MASK_FLAG] = [
        FLAG_OK,
        FLAG_REVIEWED,
        FLAG_OK,
        FLAG_OK,
    ]

    replay_close_mask_detection(
        window, rules_pack(close_masks=CloseMaskSettings(5.0))
    )

    assert flags(window.filtered_df) == [
        FLAG_OK,
        FLAG_REVIEWED,
        FLAG_FLAGGED,
        FLAG_FLAGGED,
    ]


def test_a_position_switch_without_saved_settings_detects_nothing():
    window = make_session()

    assert replay_close_mask_detection(window, rules_pack()) == 0
    assert CLOSE_MASK_FLAG not in window.filtered_df.columns


def test_both_position_switch_paths_repeat_the_detection():
    """`load_position` and the position dialog each replay the saved rules."""
    import inspect

    from SECQUOIA.gui import position_navigation

    source = inspect.getsource(position_navigation)

    assert source.count("replay_close_mask_detection(main_window, pack)") == 2


# The layout of the Close masks tab
def shown_close_tab(close_window, qtbot):
    close_window.resize(760, 420)
    close_window.show()
    close_window.tabs.setCurrentWidget(close_window.close_mask_tab.page)
    qtbot.wait(50)
    return close_window.close_mask_tab


def left_edge(widget, page) -> int:
    return widget.mapTo(page, QPoint(0, 0)).x()


def top_edge(widget, page) -> int:
    return widget.mapTo(page, QPoint(0, 0)).y()


def test_the_mask_selector_comes_first_and_the_distance_follows_in_the_same_row(
    close_window,
):
    grid = close_window.close_mask_tab.controls_grid
    tab = close_window.close_mask_tab

    mask = grid.getItemPosition(grid.indexOf(tab.mask_combo))
    distance = grid.getItemPosition(grid.indexOf(tab.dist_spin))

    assert mask[0] == distance[0] == 0
    assert mask[1] < distance[1]


def test_each_button_sits_in_the_row_below_and_the_column_of_its_control(
    close_window,
):
    grid = close_window.close_mask_tab.controls_grid
    tab = close_window.close_mask_tab

    def cell(widget):
        row, column, *_ = grid.getItemPosition(grid.indexOf(widget))
        return row, column

    assert cell(tab.find_btn) == (1, cell(tab.mask_combo)[1])
    assert cell(tab.reset_btn) == (1, cell(tab.dist_spin)[1])


def test_the_buttons_line_up_with_the_controls_above_them(close_window, qtbot):
    tab = shown_close_tab(close_window, qtbot)

    assert left_edge(tab.find_btn, tab.page) == left_edge(
        tab.mask_combo, tab.page
    )
    assert left_edge(tab.reset_btn, tab.page) == left_edge(
        tab.dist_spin, tab.page
    )
    assert tab.find_btn.width() == tab.mask_combo.width()
    assert tab.reset_btn.width() == tab.dist_spin.width()
    assert top_edge(tab.find_btn, tab.page) > top_edge(
        tab.mask_combo, tab.page
    )
    assert top_edge(tab.mask_combo, tab.page) == pytest.approx(
        top_edge(tab.dist_spin, tab.page), abs=2
    )


def test_the_buttons_no_longer_repeat_close_masks_in_their_names(close_window):
    tab = close_window.close_mask_tab

    assert tab.find_btn.text() == "Find"
    assert tab.reset_btn.text() == "Reset flags"


def test_every_control_of_the_close_masks_tab_has_a_tooltip(close_window):
    from qtpy.QtWidgets import QDoubleSpinBox, QLabel, QPushButton

    page = close_window.close_mask_tab.page
    controls = [
        *page.findChildren(QPushButton),
        *page.findChildren(QDoubleSpinBox),
        close_window.close_mask_tab.mask_combo,
        *[
            label
            for label in page.findChildren(QLabel)
            if label.text() in ("Mask:", "Distance ≤:")
        ],
    ]

    assert len(controls) >= 7  # Find, Reset, Next, Exit, spin, combo, 2 labels
    assert not [control for control in controls if not control.toolTip()]


def test_both_next_buttons_have_a_tooltip(close_window):
    assert close_window.sliding_tab.next_btn.toolTip()
    assert close_window.close_mask_tab.next_btn.toolTip()
    assert close_window.sliding_tab.next_btn.toolTip() != (
        close_window.close_mask_tab.next_btn.toolTip()
    )


def test_the_menu_entry_for_checking_a_case_has_a_tooltip():
    from SECQUOIA.config import TOOLTIPSTEXT

    assert "next one" in TOOLTIPSTEXT.CLOSE_CHECK
    for text in (
        TOOLTIPSTEXT.OUT_NEXT,
        TOOLTIPSTEXT.OUT_PREVIOUS,
        TOOLTIPSTEXT.OUT_RESET,
    ):
        assert "close" in text


def parameters_dialog(window, qtbot, **rules):
    from SECQUOIA.gui.outlier.summary_dialog import OutlierParametersDialog

    window._last_outlier_rules = {"m_n": 2, "ch_n": 2, **rules}
    dialog = OutlierParametersDialog(window)
    qtbot.addWidget(dialog)
    return dialog


def table_with_header(dialog, first_columns: list[str]):
    from qtpy.QtWidgets import QTableWidget

    for table in dialog.findChildren(QTableWidget):
        header = [
            table.horizontalHeaderItem(c).text()
            for c in range(table.columnCount())
        ]
        if header[: len(first_columns)] == first_columns:
            return table
    return None


def table_rows(table) -> list[list[str]]:
    return [
        [table.item(r, c).text() for c in range(table.columnCount())]
        for r in range(table.rowCount())
    ]


def header_texts(dialog) -> list[str]:
    from qtpy.QtWidgets import QLabel

    return [label.text() for label in dialog.findChildren(QLabel)]


def test_the_close_mask_settings_are_a_table_like_the_other_parameters(
    dialog_main_window, qtbot
):
    dialog = parameters_dialog(
        dialog_main_window,
        qtbot,
        close_masks={"distance": 12.5, "masks": [1]},
    )

    table = table_with_header(dialog, ["#", "Distance ≤ (px)", "Masks"])

    assert table is not None
    assert table_rows(table) == [["1", "12.5", "1"]]
    assert "Close-mask detection:" in header_texts(dialog)


def test_all_masks_are_shown_the_way_the_rules_table_shows_them(
    dialog_main_window, qtbot
):
    dialog = parameters_dialog(
        dialog_main_window, qtbot, close_masks={"distance": 20, "masks": None}
    )

    table = table_with_header(dialog, ["#", "Distance ≤ (px)"])

    assert table_rows(table) == [["1", "20", "ALL (1..2)"]]


def test_several_masks_are_listed(dialog_main_window, qtbot):
    dialog = parameters_dialog(
        dialog_main_window, qtbot, close_masks={"distance": 3, "masks": [1, 2]}
    )

    table = table_with_header(dialog, ["#", "Distance ≤ (px)"])

    assert table_rows(table) == [["1", "3", "1, 2"]]


def test_the_close_mask_table_is_only_one_row_tall(dialog_main_window, qtbot):
    dialog = parameters_dialog(
        dialog_main_window, qtbot, close_masks={"distance": 3, "masks": None}
    )
    table = table_with_header(dialog, ["#", "Distance ≤ (px)"])

    assert table.maximumHeight() < 120


def test_no_close_mask_table_without_saved_settings(dialog_main_window, qtbot):
    dialog = parameters_dialog(dialog_main_window, qtbot)

    assert table_with_header(dialog, ["#", "Distance ≤ (px)"]) is None
    assert "Close-mask detection:" not in header_texts(dialog)


def test_the_no_parameters_note_only_shows_when_nothing_is_set(
    dialog_main_window, qtbot
):
    empty = parameters_dialog(dialog_main_window, qtbot)
    close_only = parameters_dialog(
        dialog_main_window, qtbot, close_masks={"distance": 3, "masks": None}
    )

    assert "No parameters have been selected." in header_texts(empty)
    assert "No parameters have been selected." not in header_texts(close_only)


def test_the_rules_and_sliding_tables_are_unchanged_next_to_the_new_one(
    dialog_main_window, qtbot
):
    dialog = parameters_dialog(
        dialog_main_window,
        qtbot,
        rules=[
            {
                "feat": "AreaMorphology",
                "masks": [1],
                "channels": [],
                "masks_raw": [1],
                "channels_raw": [],
                "op1": ">",
                "val1": 5.0,
            }
        ],
        sliding_windows=[
            {
                "feature": "AreaMorphology",
                "mask": 1,
                "channel": None,
                "t_min": 1,
                "t_max": 3,
                "sd_factor": 2.0,
            }
        ],
        close_masks={"distance": 4, "masks": None},
    )

    assert table_with_header(dialog, ["#", "Feature", "Masks"]) is not None
    assert table_with_header(dialog, ["#", "Feature", "Mask"]) is not None
    assert table_with_header(dialog, ["#", "Distance ≤ (px)"]) is not None
    assert "1 rule(s) currently configured:" in header_texts(dialog)


def test_the_mask_selector_reads_all_until_a_mask_is_checked(close_window):
    combo = close_window.close_mask_tab.mask_combo

    assert combo.lineEdit().text() == "All"

    combo.set_checked_values([2])

    assert combo.lineEdit().text() == "2"
