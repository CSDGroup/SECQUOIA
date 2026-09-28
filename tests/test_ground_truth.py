"""Tests for the ground truth scoring bar (developer mode)."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from SECQUOIA.core.ground_truth import (
    CATEGORIES,
    CORRECT,
    GT_ANNOTATION_ENV,
    GT_COLUMNS,
    INCORRECT,
    MISSED,
    AnnotationStore,
    annotation_path,
    annotation_store,
    current_row,
    default_records,
    gt_annotation_mode,
    measured_masks,
    next_open_point,
    open_masks,
    record_all_defaults,
    record_correction,
    record_judgment,
    record_key,
    record_note,
    snapshot,
)
from SECQUOIA.gui import ground_truth_review
from SECQUOIA.gui.ground_truth_bar import GroundTruthBar
from SECQUOIA.gui.ground_truth_review import (
    accept_current_point,
    advance_if_complete,
)

IDENT = "exp-p0001-001"
OTHER = "exp-p0001-002"


def make_frame() -> pd.DataFrame:
    """Two cells with two masks.

    At t=2 mask 1 of the first cell is unassigned and its columns hold
    placeholders; mask 2 of the second cell has no label at all.
    """
    return pd.DataFrame(
        {
            "Identification": [IDENT] * 3 + [OTHER],
            "Position": [1, 1, 1, 2],
            "TrackNumber": [1, 1, 1, 1],
            "t": [0, 1, 2, 0],
            "XMorphology": [10.0, 11.0, 12.0, 50.0],
            "YMorphology": [20.0, 21.0, 22.0, 60.0],
            "label_id_m1": pd.array([7, 7, 0, 3], dtype="Int64"),
            "nn_dist_px_m1": [2.0, 3.5, np.nan, 1.0],
            "alt_label_id_m1": pd.array([12, 13, 20, 0], dtype="Int64"),
            "alt_dist_px_m1": [6.0, 6.5, 8.0, np.nan],
            "XMorphologyM1": [11.0, 12.0, 0.0, 51.0],
            "YMorphologyM1": [21.0, 22.0, 0.0, 61.0],
            "AreaMorphologyM1": [100.0, 110.0, 0.0, 90.0],
            "label_id_m2": pd.array([9, 9, 9, pd.NA], dtype="Int64"),
            "nn_dist_px_m2": [4.0, 4.5, 5.0, np.nan],
            "alt_label_id_m2": pd.array([15, 16, 0, 0], dtype="Int64"),
            "alt_dist_px_m2": [7.0, 7.5, np.nan, np.nan],
            "XMorphologyM2": [12.0, 13.0, 14.0, np.nan],
            "YMorphologyM2": [22.0, 23.0, 24.0, np.nan],
            "AreaMorphologyM2": [300.0, 310.0, 320.0, np.nan],
        }
    )


def make_other_position() -> pd.DataFrame:
    """The table of another position: one cell, two time points."""
    frame = make_frame().iloc[:2].copy()
    frame["Identification"] = "exp-p0002-001"
    return frame


def make_window(tmp_path, **overrides) -> SimpleNamespace:
    """A main window stand-in showing the first time point of the first cell."""
    window = SimpleNamespace(
        filtered_df=make_frame(),
        ident=IDENT,
        current_TrackNumber_plot=1,
        current_time_index=0,
        folder=str(tmp_path),
        tracking_format="tTt",
        project_name="run",
        threshold=20.0,
        n_masks=2,
        unique_ids=[IDENT, OTHER],
    )
    for name, value in overrides.items():
        setattr(window, name, value)
    return window


def gt_path(tmp_path) -> Path:
    return (
        tmp_path
        / "Analysis"
        / "SECQUOIA_files_tTt"
        / "run"
        / "ground_truth_annotations.csv"
    )


def saved(tmp_path) -> pd.DataFrame:
    """Every row of the file."""
    return pd.read_csv(gt_path(tmp_path))


def clicked(tmp_path) -> pd.DataFrame:
    """The rows somebody clicked, which are the ones with a timestamp."""
    rows = saved(tmp_path)
    return rows[rows["timestamp"].notna()]


def synced_store(window) -> AnnotationStore:
    """The window's store, with a row for every tracked point and mask."""
    store = annotation_store(window)
    store.sync_defaults(window.filtered_df, [1, 2], window.threshold)
    return store


@pytest.mark.parametrize("value", ["1", "true", " Yes ", "ON"])
def test_the_switch_selects_one_shared_bar(value, monkeypatch):
    monkeypatch.setenv(GT_ANNOTATION_ENV, value)

    assert gt_annotation_mode() == 1


@pytest.mark.parametrize("value", ["2", " 2 "])
def test_the_switch_selects_a_bar_under_each_viewer(value, monkeypatch):
    monkeypatch.setenv(GT_ANNOTATION_ENV, value)

    assert gt_annotation_mode() == 2


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "3"])
def test_the_switch_is_off_for_anything_else(value, monkeypatch):
    monkeypatch.setenv(GT_ANNOTATION_ENV, value)

    assert gt_annotation_mode() == 0


def test_the_switch_is_off_when_the_variable_is_not_set(monkeypatch):
    monkeypatch.delenv(GT_ANNOTATION_ENV, raising=False)

    assert gt_annotation_mode() == 0


def test_the_file_is_next_to_the_measurement_csvs_of_the_project(tmp_path):
    assert annotation_path(make_window(tmp_path)) == str(gt_path(tmp_path))


def test_each_project_has_its_own_file(tmp_path):
    first = make_window(tmp_path, project_name="threshold_10")
    second = make_window(tmp_path, project_name="threshold_20")

    assert annotation_path(first) != annotation_path(second)


@pytest.mark.parametrize(
    "overrides", [{"folder": None}, {"tracking_format": None}]
)
def test_there_is_no_file_before_the_project_is_known(tmp_path, overrides):
    assert annotation_path(make_window(tmp_path, **overrides)) is None


def test_current_row_is_the_row_of_the_cell_track_and_time_point(tmp_path):
    window = make_window(tmp_path, current_time_index=1)

    row = current_row(window)

    assert (row["Identification"], row["t"], row["XMorphology"]) == (
        IDENT,
        1,
        11.0,
    )


def test_current_row_does_not_fall_back_to_another_time_point(tmp_path):
    window = make_window(tmp_path, current_time_index=99)

    assert current_row(window) is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"filtered_df": None},
        {"filtered_df": pd.DataFrame()},
        {"ident": "unknown"},
        {"current_TrackNumber_plot": None},
    ],
)
def test_current_row_is_none_when_nothing_matches(tmp_path, overrides):
    assert current_row(make_window(tmp_path, **overrides)) is None


def test_current_row_is_none_before_the_window_has_a_current_cell():
    assert current_row(SimpleNamespace(filtered_df=make_frame())) is None


def test_snapshot_records_the_assignment_of_the_chosen_mask(tmp_path):
    row = current_row(make_window(tmp_path))

    assert snapshot(row, 2) == {
        "assigned_label_id": 9,
        "distance_px": 4.0,
        "alt_label_id": 15,
        "alt_distance_px": 7.0,
        "tracking_x": 10.0,
        "tracking_y": 20.0,
        "mask_centroid_x": 12.0,
        "mask_centroid_y": 22.0,
        "mask_area_px": 300.0,
    }


def test_snapshot_leaves_the_mask_values_empty_without_an_assigned_mask(
    tmp_path,
):
    """Even so, a second candidate mask can exist and is still reported."""
    row = current_row(make_window(tmp_path, current_time_index=2))

    assert snapshot(row, 1) == {
        "assigned_label_id": 0,
        "distance_px": None,
        "alt_label_id": 20,
        "alt_distance_px": 8.0,
        "tracking_x": 12.0,
        "tracking_y": 22.0,
        "mask_centroid_x": None,
        "mask_centroid_y": None,
        "mask_area_px": None,
    }


def test_snapshot_reads_a_missing_label_as_no_mask():
    row = make_frame().iloc[3]  # label_id_m2 is <NA>

    assert snapshot(row, 2)["assigned_label_id"] == 0


def test_snapshot_survives_columns_a_table_does_not_have():
    row = pd.Series({"XMorphology": 1.0, "YMorphology": 2.0})

    assert snapshot(row, 1)["assigned_label_id"] == 0
    assert snapshot(row, 1)["alt_label_id"] == 0


def test_snapshot_reports_no_second_candidate_as_zero_and_no_distance():
    row = make_frame().iloc[3]  # alt_label_id_m1 is 0 for this row

    assert snapshot(row, 1)["alt_label_id"] == 0
    assert snapshot(row, 1)["alt_distance_px"] is None


def test_record_key_is_identification_track_time_and_mask(tmp_path):
    row = current_row(make_window(tmp_path, current_time_index=1))

    assert record_key(row, 2) == (IDENT, 1, 1, 2)


def test_default_records_have_a_row_for_every_tracked_point_and_mask():
    records = default_records(make_frame(), [1, 2], 20.0)

    assert list(records.columns) == GT_COLUMNS
    assert sorted(
        zip(
            records["Identification"],
            records["t"],
            records["mask_idx"],
            strict=True,
        )
    ) == [
        (IDENT, 0, 1),
        (IDENT, 0, 2),
        (IDENT, 1, 1),
        (IDENT, 1, 2),
        (IDENT, 2, 1),
        (IDENT, 2, 2),
        (OTHER, 0, 1),
        (OTHER, 0, 2),
    ]


def test_a_point_with_an_assigned_mask_starts_correct_and_one_without_missed():
    records = default_records(make_frame(), [1, 2], 20.0).set_index(
        ["Identification", "t", "mask_idx"]
    )

    assert records["category"].to_dict() == {
        (IDENT, 0, 1): CORRECT,
        (IDENT, 0, 2): CORRECT,
        (IDENT, 1, 1): CORRECT,
        (IDENT, 1, 2): CORRECT,
        (IDENT, 2, 1): MISSED,
        (IDENT, 2, 2): CORRECT,
        (OTHER, 0, 1): CORRECT,
        (OTHER, 0, 2): MISSED,
    }


def test_checked_label_id_starts_as_a_copy_of_the_assigned_one():
    records = default_records(make_frame(), [1, 2], 20.0).set_index(
        ["Identification", "t", "mask_idx"]
    )

    assert records.loc[(IDENT, 0, 1), "checked_label_id"] == 7
    assert records.loc[(IDENT, 2, 1), "checked_label_id"] == 0  # unassigned


def test_default_records_carry_the_snapshot_but_no_note_and_no_timestamp():
    records = default_records(make_frame(), [1, 2], 20.0).set_index(
        ["Identification", "t", "mask_idx"]
    )

    first = records.loc[(IDENT, 0, 2)]
    assert first["assigned_label_id"] == 9
    assert first["checked_label_id"] == 9
    assert first["distance_px"] == 4.0
    assert first["alt_label_id"] == 15
    assert first["alt_distance_px"] == 7.0
    assert first["mask_area_px"] == 300.0
    assert first["threshold_px"] == 20.0
    assert first["notes"] == ""
    assert records["timestamp"].isna().all()
    assert records.loc[(IDENT, 2, 1), "mask_centroid_x"] != 0.0  # a NaN


def test_default_records_carry_the_position():
    records = default_records(make_frame(), [1, 2], 20.0).set_index(
        ["Identification", "t", "mask_idx"]
    )

    assert records.loc[(IDENT, 0, 1), "Position"] == 1
    assert records.loc[(OTHER, 0, 1), "Position"] == 2


def test_default_records_carry_the_second_candidate_even_when_unassigned():
    records = default_records(make_frame(), [1, 2], 20.0).set_index(
        ["Identification", "t", "mask_idx"]
    )

    unassigned = records.loc[(IDENT, 2, 1)]
    assert (
        unassigned["alt_label_id"],
        unassigned["alt_distance_px"],
    ) == (
        20,
        8.0,
    )


def test_default_records_agree_with_the_snapshot_of_each_row():
    frame = make_frame()
    records = default_records(frame, [1, 2], 20.0).set_index(
        ["Identification", "t", "mask_idx"]
    )

    for _, row in frame.iterrows():
        for mask_idx in (1, 2):
            expected = snapshot(row, mask_idx)
            stored = records.loc[(row["Identification"], row["t"], mask_idx)]
            for name, value in expected.items():
                if value is None:
                    assert pd.isna(stored[name])
                else:
                    assert stored[name] == value


def test_default_records_skip_masks_the_table_has_no_label_column_for():
    records = default_records(make_frame(), [1, 2, 3], 20.0)

    assert set(records["mask_idx"]) == {1, 2}


def test_default_records_skip_points_without_a_track_or_time_point():
    frame = make_frame()
    frame.loc[1, "t"] = np.nan

    assert len(default_records(frame, [1, 2], 20.0)) == 6


def test_default_records_do_not_depend_on_the_index_of_the_table():
    frame = make_frame()
    frame.index = [0, 0, 1, 1]

    assert len(default_records(frame, [1, 2], 20.0)) == 8


def test_default_records_are_empty_for_a_table_without_the_key_columns():
    records = default_records(pd.DataFrame({"a": [1]}), [1], 20.0)

    assert records.empty
    assert list(records.columns) == GT_COLUMNS


def test_syncing_lists_every_point_and_mask_sorted_by_key(tmp_path):
    synced_store(make_window(tmp_path))

    rows = saved(tmp_path)
    assert len(rows) == 8
    assert rows["timestamp"].isna().all()
    keys = rows[["Identification", "TrackNumber", "t", "mask_idx"]].values
    assert keys.tolist() == sorted(keys.tolist())


def test_syncing_twice_adds_nothing(tmp_path):
    window = make_window(tmp_path)
    store = synced_store(window)

    store.sync_defaults(window.filtered_df, [1, 2], window.threshold)

    assert len(saved(tmp_path)) == 8


def test_syncing_keeps_the_rows_of_other_positions(tmp_path):
    window = make_window(tmp_path)
    store = synced_store(window)

    store.sync_defaults(make_other_position(), [1, 2], 20.0)
    assert len(saved(tmp_path)) == 12

    store.sync_defaults(window.filtered_df, [1, 2], 20.0)
    assert len(saved(tmp_path)) == 12


def test_syncing_refreshes_untouched_rows_and_keeps_clicked_ones(tmp_path):
    window = make_window(tmp_path)
    store = synced_store(window)
    record_judgment(window, 1, INCORRECT, "judged")  # (t=0, mask 1)

    reassigned = make_frame()
    reassigned["label_id_m1"] = pd.array([5, 5, 0, 3], dtype="Int64")
    store.sync_defaults(reassigned, [1, 2], 20.0)

    assert store.get((IDENT, 1, 0, 1))["assigned_label_id"] == 7
    assert store.get((IDENT, 1, 0, 1))["category"] == INCORRECT
    assert store.get((IDENT, 1, 1, 1))["assigned_label_id"] == 5
    assert len(saved(tmp_path)) == 8


def test_a_missed_default_and_a_correct_default_survive_a_round_trip(tmp_path):
    synced_store(make_window(tmp_path))

    reopened = AnnotationStore(str(gt_path(tmp_path)))

    assert reopened.get((IDENT, 1, 2, 1))["category"] == MISSED
    assert pd.isna(reopened.get((IDENT, 1, 2, 1))["timestamp"])
    assert reopened.get((IDENT, 1, 0, 1))["category"] == CORRECT
    assert pd.isna(reopened.get((IDENT, 1, 0, 1))["timestamp"])


def test_a_missed_default_can_still_be_explicitly_judged(tmp_path):
    window = make_window(tmp_path, current_time_index=2)
    store = synced_store(window)

    record_judgment(window, 1, MISSED)

    row = store.get((IDENT, 1, 2, 1))
    assert row["category"] == MISSED
    assert pd.notna(row["timestamp"])
    assert len(saved(tmp_path)) == 8


def test_a_judgment_is_written_with_the_snapshot_and_the_decision(tmp_path):
    window = make_window(tmp_path)

    record = record_judgment(window, 1, INCORRECT, "  touching neighbor ")

    written = saved(tmp_path)
    assert list(written.columns) == GT_COLUMNS
    assert written.loc[0].to_dict() == {
        "Identification": IDENT,
        "Position": 1,
        "TrackNumber": 1,
        "t": 0,
        "mask_idx": 1,
        "assigned_label_id": 7,
        "distance_px": 2.0,
        "checked_label_id": 0,
        "alt_label_id": 12,
        "alt_distance_px": 6.0,
        "tracking_x": 10.0,
        "tracking_y": 20.0,
        "mask_centroid_x": 11.0,
        "mask_centroid_y": 21.0,
        "mask_area_px": 100.0,
        "category": INCORRECT,
        "notes": "touching neighbor",
        "threshold_px": 20.0,
        "timestamp": record["timestamp"],
    }


def test_correct_and_missed_both_copy_the_assigned_value_when_clicked(
    tmp_path,
):
    window = make_window(tmp_path)

    record_judgment(window, 1, CORRECT)

    assert saved(tmp_path)["checked_label_id"].tolist() == [7]


def test_incorrect_always_sets_the_checked_label_to_zero(tmp_path):
    window = make_window(tmp_path, current_time_index=2)

    record_judgment(window, 1, INCORRECT)

    row = saved(tmp_path).iloc[0]
    assert (row["assigned_label_id"], row["checked_label_id"]) == (0, 0)


def test_judging_again_replaces_the_judgment_instead_of_adding_one(tmp_path):
    window = make_window(tmp_path)

    record_judgment(window, 1, INCORRECT, "first look")
    record_judgment(window, 1, CORRECT, "second look")

    written = saved(tmp_path)
    assert written["category"].tolist() == [CORRECT]
    assert written["notes"].tolist() == ["second look"]


def test_each_mask_and_time_point_is_judged_on_its_own(tmp_path):
    window = make_window(tmp_path)

    record_judgment(window, 1, CORRECT)
    record_judgment(window, 2, MISSED)
    window.current_time_index = 1
    record_judgment(window, 1, INCORRECT)

    written = saved(tmp_path)
    assert len(written) == 3
    assert sorted(
        zip(
            written["t"],
            written["mask_idx"],
            written["category"],
            strict=True,
        )
    ) == [
        (0, 1, CORRECT),
        (0, 2, MISSED),
        (1, 1, INCORRECT),
    ]


def test_every_category_can_be_recorded(tmp_path):
    assert CATEGORIES == (CORRECT, INCORRECT, MISSED)
    window = make_window(tmp_path)

    for mask_idx, category in enumerate(CATEGORIES, start=1):
        record_judgment(window, mask_idx, category)

    assert saved(tmp_path)["category"].tolist() == list(CATEGORIES)


def test_an_unknown_category_is_refused_and_nothing_is_written(tmp_path):
    window = make_window(tmp_path)

    with pytest.raises(ValueError, match="Invalid category"):
        record_judgment(window, 1, "Fine")

    assert not gt_path(tmp_path).exists()


def test_nothing_is_judged_without_a_row_or_an_experiment(tmp_path):
    assert (
        record_judgment(
            make_window(tmp_path, current_time_index=99), 1, CORRECT
        )
        is None
    )
    assert (
        record_judgment(make_window(tmp_path, folder=None), 1, CORRECT) is None
    )
    assert not gt_path(tmp_path).exists()


def test_a_judgment_never_touches_the_measurement_table(tmp_path):
    window = make_window(tmp_path)
    before = window.filtered_df.copy()

    synced_store(window)
    record_judgment(window, 1, INCORRECT)

    pd.testing.assert_frame_equal(window.filtered_df, before)


def test_the_file_is_replaced_whole_and_leaves_no_temporary_copy(tmp_path):
    record_judgment(make_window(tmp_path), 1, CORRECT)

    assert os.listdir(gt_path(tmp_path).parent) == [
        "ground_truth_annotations.csv"
    ]


def test_the_csv_column_order_is_the_canonical_one_however_the_row_was_made(
    tmp_path,
):
    """A record built as a dict can list its fields in any order; the file must not."""
    window = make_window(tmp_path)
    synced_store(window)
    record_judgment(window, 1, INCORRECT)

    assert list(saved(tmp_path).columns) == GT_COLUMNS


def test_a_note_is_saved_with_the_category_the_row_already_has(tmp_path):
    window = make_window(tmp_path)
    synced_store(window)

    record = record_note(window, 1, "  division doublet ")

    assert record["category"] == CORRECT
    row = clicked(tmp_path).iloc[0]
    assert (row["category"], row["notes"]) == (CORRECT, "division doublet")


def test_a_note_keeps_a_category_that_was_clicked(tmp_path):
    window = make_window(tmp_path)
    synced_store(window)
    record_judgment(window, 1, INCORRECT, "first")

    record_note(window, 1, "second")

    row = clicked(tmp_path).iloc[0]
    assert (row["category"], row["notes"]) == (INCORRECT, "second")


def test_a_note_is_refused_for_a_mask_that_was_never_synced(tmp_path):
    window = make_window(tmp_path)
    synced_store(window)

    assert record_note(window, 3, "no such mask") is None
    assert clicked(tmp_path).empty


def test_no_note_is_saved_without_a_row_or_an_experiment(tmp_path):
    assert record_note(make_window(tmp_path, folder=None), 1, "x") is None
    window = make_window(tmp_path, current_time_index=99)
    synced_store(window)
    assert record_note(window, 1, "x") is None
    assert record_note(make_window(tmp_path, ident="unknown"), 1, "x") is None


def edit_target(mask_idx: int, label_id: int) -> SimpleNamespace:
    """A stand-in for `EditTarget`: `record_correction` only reads these."""
    return SimpleNamespace(
        ident=IDENT, track_no=1, t=0, mask_idx=mask_idx, label_id=label_id
    )


def corrected_row(tmp_path) -> pd.Series:
    """The (IDENT, t=0, mask 1) row that `edit_target` above always targets."""
    rows = saved(tmp_path)
    hit = rows[
        (rows["Identification"] == IDENT)
        & (rows["t"] == 0)
        & (rows["mask_idx"] == 1)
    ]
    assert len(hit) == 1
    return hit.iloc[0]


def test_a_correction_on_a_missed_default_names_the_true_mask(tmp_path):
    window = make_window(tmp_path, current_time_index=2)  # mask 1 unassigned
    synced_store(window)
    target = edit_target(1, 42)
    target.t = 2

    record_correction(window, target)

    rows = saved(tmp_path)
    row = rows[
        (rows["Identification"] == IDENT)
        & (rows["t"] == 2)
        & (rows["mask_idx"] == 1)
    ].iloc[0]
    assert (row["assigned_label_id"], row["checked_label_id"]) == (0, 42)
    assert row["category"] == MISSED
    assert pd.notna(row["timestamp"])


def test_a_correction_confirming_the_assigned_mask_is_correct(tmp_path):
    window = make_window(tmp_path)
    synced_store(window)

    record_correction(window, edit_target(1, 7))

    row = corrected_row(tmp_path)
    assert (row["checked_label_id"], row["category"]) == (7, CORRECT)


def test_a_correction_to_a_different_mask_is_incorrect(tmp_path):
    window = make_window(tmp_path)
    synced_store(window)

    record_correction(window, edit_target(1, 99))

    row = corrected_row(tmp_path)
    assert (row["checked_label_id"], row["category"]) == (99, INCORRECT)


def test_a_correction_preserves_the_original_assigned_label(tmp_path):
    """The edit has already changed the real data by the time this runs."""
    window = make_window(tmp_path)
    synced_store(window)

    record_correction(window, edit_target(1, 99))

    row = corrected_row(tmp_path)
    assert row["assigned_label_id"] == 7


def test_a_correction_keeps_the_note_and_the_second_candidate(tmp_path):
    window = make_window(tmp_path)
    synced_store(window)
    record_judgment(window, 1, INCORRECT, "first look")

    record_correction(window, edit_target(1, 99))

    row = corrected_row(tmp_path)
    assert row["notes"] == "first look"
    assert row["alt_label_id"] == 12


def test_a_correction_can_be_redone_by_a_later_correction(tmp_path):
    window = make_window(tmp_path)
    synced_store(window)
    record_correction(window, edit_target(1, 99))

    record_correction(window, edit_target(1, 7))

    row = corrected_row(tmp_path)
    assert (row["checked_label_id"], row["category"]) == (7, CORRECT)


def test_nothing_is_corrected_without_a_prior_row_or_a_project(tmp_path):
    window = make_window(tmp_path)

    assert record_correction(window, edit_target(1, 42)) is None
    assert not gt_path(tmp_path).exists()
    assert (
        record_correction(
            make_window(tmp_path, folder=None), edit_target(1, 1)
        )
        is None
    )


def judge_point(window, ident: str, t: int) -> None:
    """Judge both masks of a point, so it counts as done."""
    window.ident, window.current_time_index = ident, t
    record_judgment(window, 1, CORRECT)
    record_judgment(window, 2, CORRECT)


def test_measured_masks_are_the_masks_with_a_label_column():
    assert measured_masks(make_frame(), [1, 2, 3]) == [1, 2]


def test_every_measured_mask_is_open_before_anything_is_judged(tmp_path):
    window = make_window(tmp_path)
    synced_store(window)

    assert open_masks(window, [1, 2]) == [1, 2]


def test_a_judged_mask_is_no_longer_open(tmp_path):
    window = make_window(tmp_path)
    synced_store(window)

    record_judgment(window, 1, INCORRECT)
    assert open_masks(window, [1, 2]) == [2]

    record_judgment(window, 2, CORRECT)
    assert open_masks(window, [1, 2]) == []


def test_no_mask_is_open_where_there_is_nothing_to_judge(tmp_path):
    window = make_window(tmp_path)
    synced_store(window)

    assert open_masks(make_window(tmp_path, folder=None), [1, 2]) is None
    assert open_masks(window, [3]) is None  # not measured
    window.current_time_index = 99
    assert open_masks(window, [1, 2]) is None


def test_accepting_confirms_every_open_mask_as_its_default(tmp_path):
    window = make_window(tmp_path)
    synced_store(window)

    record_all_defaults(window, [1, 2])

    rows = clicked(tmp_path)
    assert rows["category"].tolist() == [CORRECT, CORRECT]
    assert rows["mask_idx"].tolist() == [1, 2]
    assert open_masks(window, [1, 2]) == []


def test_accepting_keeps_a_judgment_that_was_already_made(tmp_path):
    window = make_window(tmp_path)
    synced_store(window)
    record_judgment(window, 1, INCORRECT, "kept")

    record_all_defaults(window, [1, 2])

    rows = clicked(tmp_path).set_index("mask_idx")
    assert rows.loc[1, "category"] == INCORRECT
    assert rows.loc[1, "notes"] == "kept"
    assert rows.loc[2, "category"] == CORRECT


def test_accepting_confirms_a_missed_default_rather_than_turning_it_correct(
    tmp_path,
):
    window = make_window(tmp_path, current_time_index=2)
    synced_store(window)

    record_all_defaults(window, [1, 2])

    rows = clicked(tmp_path).set_index("mask_idx")
    assert rows.loc[1, "category"] == MISSED
    assert rows.loc[1, "checked_label_id"] == 0
    assert rows.loc[2, "category"] == CORRECT
    assert open_masks(window, [1, 2]) == []


def test_accepting_does_nothing_where_there_is_no_point(tmp_path):
    window = make_window(tmp_path, current_time_index=99)
    synced_store(window)

    record_all_defaults(window, [1, 2])

    assert clicked(tmp_path).empty


def test_the_next_open_point_follows_the_current_one_in_time(tmp_path):
    window = make_window(tmp_path)
    synced_store(window)

    assert next_open_point(window) == (IDENT, 1, 1)


def test_points_that_are_done_are_skipped(tmp_path):
    window = make_window(tmp_path)
    synced_store(window)
    judge_point(window, IDENT, 1)
    window.current_time_index = 0

    assert next_open_point(window) == (IDENT, 1, 2)


def test_after_the_last_point_of_an_identification_comes_the_next_one(
    tmp_path,
):
    window = make_window(tmp_path)
    synced_store(window)
    for t in (0, 1, 2):
        judge_point(window, IDENT, t)

    assert next_open_point(window) == (OTHER, 1, 0)


def test_the_search_wraps_around_to_points_that_were_skipped(tmp_path):
    window = make_window(tmp_path)
    synced_store(window)
    judge_point(window, OTHER, 0)

    assert next_open_point(window) == (IDENT, 1, 0)


def test_no_point_is_next_once_every_point_is_done(tmp_path):
    window = make_window(tmp_path)
    synced_store(window)
    for ident, t in [(IDENT, 0), (IDENT, 1), (IDENT, 2), (OTHER, 0)]:
        judge_point(window, ident, t)

    assert next_open_point(window) is None


def test_the_tree_order_decides_which_identification_comes_first(tmp_path):
    window = make_window(tmp_path, ident="unknown")
    synced_store(window)

    window.unique_ids = [IDENT, OTHER]
    assert next_open_point(window) == (IDENT, 1, 0)

    window.unique_ids = [OTHER, IDENT]
    assert next_open_point(window) == (OTHER, 1, 0)


def test_points_that_left_the_loaded_table_are_not_offered(tmp_path):
    window = make_window(tmp_path)
    synced_store(window)

    window.filtered_df = make_frame().drop(index=1)

    assert next_open_point(window) == (IDENT, 1, 2)


def test_there_is_no_next_point_without_a_table_or_a_project(tmp_path):
    assert next_open_point(make_window(tmp_path, folder=None)) is None
    window = make_window(tmp_path)
    synced_store(window)
    window.filtered_df = None
    assert next_open_point(window) is None


def test_judgments_already_in_the_file_are_kept_and_can_be_replaced(tmp_path):
    record_judgment(make_window(tmp_path), 1, INCORRECT, "note")

    resumed = make_window(tmp_path)
    store = annotation_store(resumed)
    assert store.get((IDENT, 1, 0, 1))["notes"] == "note"
    record_judgment(resumed, 2, CORRECT)
    record_judgment(resumed, 1, MISSED)

    written = saved(tmp_path)
    assert sorted(
        zip(written["mask_idx"], written["category"], strict=True)
    ) == [
        (1, MISSED),
        (2, CORRECT),
    ]


def test_an_empty_note_survives_a_round_trip_as_an_empty_string(tmp_path):
    record_judgment(make_window(tmp_path), 1, CORRECT)

    store = AnnotationStore(str(gt_path(tmp_path)))

    assert store.get((IDENT, 1, 0, 1))["notes"] == ""


def test_a_file_that_is_not_a_ground_truth_file_is_not_overwritten(tmp_path):
    gt_path(tmp_path).parent.mkdir(parents=True)
    gt_path(tmp_path).write_text("a,b\n1,2\n")

    with pytest.raises(ValueError, match="not a ground-truth file"):
        record_judgment(make_window(tmp_path), 1, CORRECT)

    assert gt_path(tmp_path).read_text() == "a,b\n1,2\n"


def test_the_store_follows_the_loaded_project(tmp_path):
    window = make_window(tmp_path)
    first = annotation_store(window)
    assert annotation_store(window) is first

    window.project_name = "another"

    assert annotation_store(window) is not first


def test_there_is_no_store_before_an_experiment_is_loaded(tmp_path):
    assert annotation_store(make_window(tmp_path, folder=None)) is None


@pytest.fixture
def shown(monkeypatch):
    """The points the view was asked to show, instead of really moving it."""
    points = []
    monkeypatch.setattr(
        ground_truth_review,
        "_show_point",
        lambda window, ident, track, t: (
            points.append((ident, track, t)) or True
        ),
    )
    return points


@pytest.fixture
def dialogs(monkeypatch):
    """The messages the review would have shown in a dialog."""
    messages = []
    monkeypatch.setattr(
        ground_truth_review,
        "_show_info_dialog",
        lambda window, title, text: messages.append((title, text)),
    )
    return messages


def test_the_view_stays_while_a_mask_of_the_point_is_open(
    tmp_path, shown, dialogs
):
    window = make_window(tmp_path)
    synced_store(window)
    record_judgment(window, 1, CORRECT)

    advance_if_complete(window)

    assert shown == [] and dialogs == []


def test_the_view_moves_on_once_every_mask_of_the_point_is_judged(
    tmp_path, shown, dialogs
):
    window = make_window(tmp_path)
    synced_store(window)
    judge_point(window, IDENT, 0)

    advance_if_complete(window)

    assert shown == [(IDENT, 1, 1)]


def test_the_view_stays_where_there_is_no_point_to_judge(
    tmp_path, shown, dialogs
):
    window = make_window(tmp_path, current_time_index=99)
    synced_store(window)

    advance_if_complete(window)

    assert shown == [] and dialogs == []


def test_the_user_is_told_when_every_point_is_done(tmp_path, shown, dialogs):
    window = make_window(tmp_path)
    synced_store(window)
    for ident, t in [(IDENT, 0), (IDENT, 1), (IDENT, 2), (OTHER, 0)]:
        judge_point(window, ident, t)

    advance_if_complete(window)

    assert shown == []
    assert dialogs == [("Ground truth", "All points checked")]


def test_a_point_that_cannot_be_shown_is_logged_not_raised(
    tmp_path, dialogs, monkeypatch, caplog
):
    window = make_window(tmp_path)
    synced_store(window)
    judge_point(window, IDENT, 0)

    def fail(*args):
        raise RuntimeError("no viewer")

    monkeypatch.setattr(ground_truth_review, "_show_point", fail)
    with caplog.at_level("WARNING", logger=ground_truth_review.__name__):
        advance_if_complete(window)

    assert "Could not show the point" in caplog.text


def test_a_point_of_an_unknown_identification_is_logged(
    tmp_path, dialogs, monkeypatch, caplog
):
    window = make_window(tmp_path)
    synced_store(window)
    judge_point(window, IDENT, 0)
    monkeypatch.setattr(
        ground_truth_review, "_show_point", lambda *args: False
    )

    with caplog.at_level("WARNING", logger=ground_truth_review.__name__):
        advance_if_complete(window)

    assert "Could not show the point" in caplog.text


@pytest.fixture
def navigation(monkeypatch):
    """The calls the real navigation would have received."""
    calls = []

    def go_to_ident(window, ident):
        calls.append(("ident", ident))
        window.ident, window.current_TrackNumber_plot = ident, 1
        return ident != "unknown"

    monkeypatch.setattr(ground_truth_review, "go_to_ident", go_to_ident)
    monkeypatch.setattr(
        ground_truth_review,
        "update_plot",
        lambda window: calls.append(("plot", None)),
    )
    monkeypatch.setattr(
        ground_truth_review,
        "_jump_to_time",
        lambda window, t: calls.append(("time", t)),
    )
    return calls


def test_showing_a_point_of_the_same_track_only_changes_the_time(
    tmp_path, navigation
):
    window = make_window(tmp_path)

    assert ground_truth_review._show_point(window, IDENT, 1, 5)

    assert navigation == [("time", 5)]


def test_showing_another_track_of_the_identification_redraws_the_plot(
    tmp_path, navigation
):
    window = make_window(tmp_path)

    assert ground_truth_review._show_point(window, IDENT, 2, 5)

    assert window.current_TrackNumber_plot == 2
    assert navigation == [("plot", None), ("time", 5)]


def test_showing_another_identification_goes_there_first(tmp_path, navigation):
    window = make_window(tmp_path)

    assert ground_truth_review._show_point(window, OTHER, 3, 5)

    assert (window.ident, window.current_TrackNumber_plot) == (OTHER, 3)
    assert navigation == [("ident", OTHER), ("plot", None), ("time", 5)]


def test_an_unknown_identification_leaves_the_view_alone(tmp_path, navigation):
    window = make_window(tmp_path)

    assert not ground_truth_review._show_point(window, "unknown", 1, 5)

    assert navigation == [("ident", "unknown")]


def test_the_x_key_hands_over_to_the_first_bar():
    accepted = []
    window = SimpleNamespace(
        gt_bars=[
            SimpleNamespace(accept_all=lambda: accepted.append("first")),
            SimpleNamespace(accept_all=lambda: accepted.append("second")),
        ]
    )

    accept_current_point(window)

    assert accepted == ["first"]


def test_the_x_key_does_nothing_without_a_bar():
    accept_current_point(SimpleNamespace())
    accept_current_point(SimpleNamespace(gt_bars=[]))


@pytest.fixture
def bar(tmp_path, qtbot, shown, dialogs):
    window = make_window(tmp_path)
    widget = GroundTruthBar(window)
    qtbot.addWidget(widget)
    return widget


def click(bar: GroundTruthBar, category: str) -> None:
    bar.buttons[category].click()


def test_the_bar_lists_the_masks_and_says_what_a_click_would_record(bar):
    assert [
        bar.mask_combo.itemData(i) for i in range(bar.mask_combo.count())
    ] == [1, 2]
    assert bar.status.text() == (
        f"Judging → Mask 1 | {IDENT} | Track 1 | t=0 | assigned label 7 | "
        "dist 2.0 px | next nearest: label 12 (6.0 px) — default: Correct"
    )
    assert all(button.isEnabled() for button in bar.buttons.values())


def test_the_status_line_says_nothing_about_a_second_candidate_without_one(
    bar,
):
    bar.main_window.ident = OTHER
    bar.refresh()

    assert "next nearest" not in bar.status.text()


def test_the_bar_has_one_button_per_category(bar):
    assert list(bar.buttons) == list(CATEGORIES)


def test_the_file_lists_every_point_and_mask_before_anything_is_clicked(
    bar, tmp_path
):
    assert len(saved(tmp_path)) == 8
    assert clicked(tmp_path).empty


def test_a_click_judges_the_mask_chosen_in_the_bar(bar, tmp_path):
    bar.mask_combo.setCurrentIndex(bar.mask_combo.findData(2))

    assert "Mask 2" in bar.status.text()
    assert "assigned label 9" in bar.status.text()
    click(bar, INCORRECT)

    assert clicked(tmp_path)[
        ["mask_idx", "assigned_label_id", "category"]
    ].values.tolist() == [[2, 9, INCORRECT]]


def test_the_line_shows_that_a_judgment_is_recorded(bar):
    click(bar, MISSED)

    assert bar.status.text().endswith(f"— recorded: {MISSED}")


def test_switching_the_mask_shows_that_mask_as_not_yet_clicked(bar):
    click(bar, INCORRECT)

    bar.mask_combo.setCurrentIndex(bar.mask_combo.findData(2))

    assert bar.status.text().endswith("— default: Correct")


def test_the_note_is_saved_with_the_click_and_shown_again_on_return(
    bar, tmp_path
):
    bar.notes.setText("division doublet")
    click(bar, INCORRECT)
    assert clicked(tmp_path)["notes"].tolist() == ["division doublet"]

    bar.main_window.current_time_index = 1
    bar.refresh()
    assert bar.notes.text() == ""

    bar.main_window.current_time_index = 0
    bar.refresh()
    assert bar.notes.text() == "division doublet"


def test_a_note_typed_for_one_time_point_does_not_follow_to_the_next(
    bar, tmp_path
):
    bar.notes.setText("meant for t=0")

    bar.main_window.current_time_index = 1
    bar.refresh()
    click(bar, CORRECT)

    assert clicked(tmp_path)["notes"].isna().tolist() == [True]


def test_enter_saves_the_note_and_keeps_the_category(bar, tmp_path):
    bar.notes.setText("touching neighbor")

    bar.notes.returnPressed.emit()

    row = clicked(tmp_path).iloc[0]
    assert (row["category"], row["notes"]) == (CORRECT, "touching neighbor")
    assert bar.status.text().endswith("— recorded: Correct")


def test_enter_keeps_a_category_that_was_clicked(bar, tmp_path):
    click(bar, INCORRECT)
    bar.notes.setText("wrong neighbor")

    bar.notes.returnPressed.emit()

    row = clicked(tmp_path).iloc[0]
    assert (row["category"], row["notes"]) == (INCORRECT, "wrong neighbor")


def test_enter_saves_a_note_on_a_missed_default_keeping_it_missed(
    bar, tmp_path
):
    bar.main_window.current_time_index = 2
    bar.refresh()
    bar.notes.setText("segmentation missed it")

    bar.notes.returnPressed.emit()

    row = clicked(tmp_path).iloc[0]
    assert (row["category"], row["notes"]) == (
        MISSED,
        "segmentation missed it",
    )


def test_enter_takes_the_focus_off_the_notes_box(bar, qtbot):
    bar.show()
    bar.notes.setFocus()
    qtbot.waitUntil(bar.notes.hasFocus)

    bar.notes.returnPressed.emit()

    assert not bar.notes.hasFocus()


def test_the_bar_follows_a_change_of_the_current_cell(bar):
    bar.main_window.ident = OTHER
    bar.main_window.current_time_index = 0
    bar.refresh()

    assert OTHER in bar.status.text()
    assert "assigned label 3" in bar.status.text()


def test_an_unassigned_mask_is_described_as_none_and_defaults_to_missed(bar):
    bar.main_window.current_time_index = 2
    bar.refresh()

    assert "assigned label none | dist –" in bar.status.text()
    assert bar.status.text().endswith(f"— default: {MISSED}")
    assert all(button.isEnabled() for button in bar.buttons.values())


def test_the_buttons_are_off_where_the_cell_has_no_tracked_point(bar):
    bar.main_window.current_time_index = 99
    bar.refresh()

    assert not any(button.isEnabled() for button in bar.buttons.values())
    assert bar.status.text() == (
        "No tracked point for the current cell at this time point."
    )


def test_the_buttons_are_off_before_an_experiment_is_loaded(tmp_path, qtbot):
    widget = GroundTruthBar(make_window(tmp_path, folder=None))
    qtbot.addWidget(widget)

    assert not any(button.isEnabled() for button in widget.buttons.values())
    assert widget.status.text() == "Load an experiment first."


def test_the_bar_can_be_built_before_the_window_has_any_state(qtbot):
    widget = GroundTruthBar(SimpleNamespace())
    qtbot.addWidget(widget)

    assert widget.status.text() == "Load an experiment first."


def test_the_chosen_mask_is_kept_when_the_mask_list_is_rebuilt(bar):
    bar.mask_combo.setCurrentIndex(bar.mask_combo.findData(2))

    bar.main_window.n_masks = 3
    bar.refresh()

    assert bar.mask_combo.currentData() == 2
    assert bar.mask_combo.count() == 3


def test_a_mask_the_table_does_not_have_gets_no_rows(bar, tmp_path):
    bar.main_window.n_masks = 3

    bar.refresh()

    assert set(saved(tmp_path)["mask_idx"]) == {1, 2}


def test_switching_position_adds_its_points_and_keeps_the_others(
    bar, tmp_path
):
    click(bar, INCORRECT)

    bar.main_window.filtered_df = make_other_position()
    bar.main_window.ident = "exp-p0002-001"
    bar.refresh()

    rows = saved(tmp_path)
    assert len(rows) == 12
    assert len(clicked(tmp_path)) == 1
    assert "exp-p0002-001" in bar.status.text()


def test_untouched_rows_follow_a_reloaded_table_but_clicked_ones_do_not(
    bar, tmp_path
):
    click(bar, INCORRECT)

    reassigned = make_frame()
    reassigned["label_id_m1"] = pd.array([5, 5, 0, 3], dtype="Int64")
    bar.main_window.filtered_df = reassigned
    bar.refresh()

    rows = saved(tmp_path).set_index(["Identification", "t", "mask_idx"])
    assert rows.loc[(IDENT, 0, 1), "assigned_label_id"] == 7
    assert rows.loc[(IDENT, 0, 1), "category"] == INCORRECT
    assert rows.loc[(IDENT, 1, 1), "assigned_label_id"] == 5


def test_the_list_is_only_rebuilt_when_the_table_changes(
    tmp_path, qtbot, monkeypatch
):
    calls = []
    original = AnnotationStore.sync_defaults

    def counting(self, *args):
        calls.append(1)
        return original(self, *args)

    monkeypatch.setattr(AnnotationStore, "sync_defaults", counting)
    widget = GroundTruthBar(make_window(tmp_path))
    qtbot.addWidget(widget)

    widget.refresh()
    widget.refresh()
    assert len(calls) == 1

    widget.main_window.filtered_df = make_frame()
    widget.refresh()
    assert len(calls) == 2


def test_a_file_that_cannot_be_read_is_reported_once_not_on_every_refresh(
    tmp_path, qtbot, caplog
):
    gt_path(tmp_path).parent.mkdir(parents=True)
    gt_path(tmp_path).write_text("a,b\n1,2\n")

    with caplog.at_level("ERROR", logger="SECQUOIA.gui.ground_truth_bar"):
        widget = GroundTruthBar(make_window(tmp_path))
        qtbot.addWidget(widget)
        widget.refresh()
        widget.refresh()

    assert widget.status.text().startswith("Cannot open the annotation file")
    assert not any(button.isEnabled() for button in widget.buttons.values())
    assert len(caplog.records) == 1


def test_a_failed_save_is_shown_and_cleared_by_the_next_good_one(
    bar, monkeypatch
):
    original = AnnotationStore.save

    def fail(self):
        raise OSError("disk full")

    monkeypatch.setattr(AnnotationStore, "save", fail)
    click(bar, CORRECT)
    assert bar.status.text().startswith(
        "Could not save the judgment: disk full"
    )

    monkeypatch.setattr(AnnotationStore, "save", original)
    click(bar, CORRECT)
    assert not bar.status.text().startswith("Could not save")


def test_a_failed_save_of_the_list_is_shown(tmp_path, qtbot, monkeypatch):
    def fail(self):
        raise OSError("disk full")

    monkeypatch.setattr(AnnotationStore, "save", fail)
    widget = GroundTruthBar(make_window(tmp_path))
    qtbot.addWidget(widget)

    assert widget.status.text().startswith(
        "Could not save the annotation file: disk full"
    )


def test_a_click_takes_the_focus_off_the_notes_box(bar, qtbot):
    bar.show()
    bar.notes.setFocus()
    qtbot.waitUntil(bar.notes.hasFocus)

    click(bar, CORRECT)

    assert not bar.notes.hasFocus()


def pick_mask(bar: GroundTruthBar, mask_idx: int) -> None:
    """Choose a mask in the bar the way a user does."""
    bar.mask_combo.setCurrentIndex(bar.mask_combo.findData(mask_idx))
    bar.mask_combo.activated.emit(bar.mask_combo.currentIndex())


def test_a_click_moves_on_only_when_it_completes_the_point(bar, shown):
    click(bar, CORRECT)  # mask 1 of 2
    assert shown == []

    pick_mask(bar, 2)
    click(bar, CORRECT)

    assert shown == [(IDENT, 1, 1)]


def test_a_click_does_not_move_on_while_a_mask_is_unassigned(bar, shown):
    bar.main_window.current_time_index = 2  # mask 1 has no assigned mask
    bar.refresh()
    pick_mask(bar, 2)

    click(bar, CORRECT)

    assert shown == []


def test_a_note_saved_with_enter_never_moves_on(bar, shown):
    click(bar, INCORRECT)
    pick_mask(bar, 2)
    bar.notes.setText("looks fine")

    bar.notes.returnPressed.emit()

    assert shown == []


def test_accepting_marks_the_open_masks_and_moves_on(bar, shown, tmp_path):
    bar.accept_all()

    assert clicked(tmp_path)["category"].tolist() == [CORRECT, CORRECT]
    assert shown == [(IDENT, 1, 1)]


def test_accepting_keeps_what_was_clicked_before(bar, shown, tmp_path):
    click(bar, INCORRECT)  # mask 1

    bar.accept_all()

    rows = clicked(tmp_path).set_index("mask_idx")
    assert rows["category"].to_dict() == {1: INCORRECT, 2: CORRECT}
    assert shown == [(IDENT, 1, 1)]


def test_accepting_confirms_missed_rather_than_correct_and_moves_on(
    bar, shown, tmp_path
):
    bar.main_window.current_time_index = 2
    bar.refresh()

    bar.accept_all()

    rows = clicked(tmp_path).set_index("mask_idx")
    assert rows.loc[1, "category"] == MISSED
    assert rows.loc[2, "category"] == CORRECT
    assert shown == [(OTHER, 1, 0)]


def test_accepting_reports_a_failed_save(bar, shown, monkeypatch):
    def fail(self):
        raise OSError("disk full")

    monkeypatch.setattr(AnnotationStore, "save", fail)

    bar.accept_all()

    assert bar.status.text().startswith("Could not save the judgment")
    assert shown == []


def test_a_bar_can_start_on_a_preset_mask(tmp_path, qtbot, shown, dialogs):
    widget = GroundTruthBar(make_window(tmp_path), mask_idx=2)
    qtbot.addWidget(widget)

    assert widget.mask_combo.currentData() == 2
    assert "Mask 2" in widget.status.text()


def test_a_mask_picked_by_the_user_beats_the_preset(
    tmp_path, qtbot, shown, dialogs
):
    widget = GroundTruthBar(make_window(tmp_path), mask_idx=2)
    qtbot.addWidget(widget)

    pick_mask(widget, 1)
    widget.main_window.n_masks = 3
    widget.refresh()

    assert widget.mask_combo.currentData() == 1


def test_the_preset_is_applied_once_the_masks_are_known(
    tmp_path, qtbot, shown, dialogs
):
    window = make_window(tmp_path, filtered_df=None, n_masks=0)
    widget = GroundTruthBar(window, mask_idx=2)
    qtbot.addWidget(widget)
    assert widget.mask_combo.currentData() == 1

    window.filtered_df, window.n_masks = make_frame(), 2
    widget.refresh()

    assert widget.mask_combo.currentData() == 2


def test_two_bars_build_the_list_only_once(
    tmp_path, qtbot, shown, dialogs, monkeypatch
):
    calls = []
    original = AnnotationStore.sync_defaults

    def counting(self, *args):
        calls.append(1)
        return original(self, *args)

    monkeypatch.setattr(AnnotationStore, "sync_defaults", counting)
    window = make_window(tmp_path)
    first = GroundTruthBar(window, mask_idx=1)
    second = GroundTruthBar(window, mask_idx=2)
    qtbot.addWidget(first)
    qtbot.addWidget(second)

    first.refresh()
    second.refresh()

    assert len(calls) == 1
    assert (
        first.mask_combo.currentData(),
        second.mask_combo.currentData(),
    ) == (
        1,
        2,
    )
