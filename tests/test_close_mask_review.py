"""Tests for reviewing close-mask cases: flag updates after an edit, and the C key.

The review data is a small table with two identifications:

    A: flagged at t=1 and t=3, ok at t=0 and t=2
    B: flagged at t=2

"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from qtpy.QtWidgets import QApplication, QLineEdit, QWidget

from SECQUOIA.core.outlier_detection.close_masks import (
    CLOSE_MASK_FLAG,
    FLAG_FLAGGED,
    FLAG_OK,
    FLAG_REVIEWED,
    close_mask_row_key,
    flagged_times,
    is_pinned_close_mask_row,
    next_flagged_ident,
    next_flagged_time,
    set_close_mask_flag,
    update_flags_after_edit,
)
from SECQUOIA.core.outlier_detection.track_sync import (
    update_unique_outliers_ids,
)
from SECQUOIA.gui.outlier import close_mask_review, navigation
from SECQUOIA.gui.outlier.close_mask_review import (
    mark_close_mask_checked,
    mark_outlier_reviewed,
    review_current_point,
)

NAN = np.nan


def review_frame() -> pd.DataFrame:
    rows = []
    for ident, flagged in (("A", (1, 3)), ("B", (2,))):
        for t in range(4):
            close = t in flagged
            rows.append(
                {
                    "Position": 1,
                    "Identification": ident,
                    "TrackNumber": 1,
                    "t": t,
                    "XMorphology": 5.0,
                    "YMorphology": 5.0,
                    "label_id_m1": 1,
                    "alt_label_id_m1": 2 if close else 0,
                    "alt_dist_px_m1": 3.0 if close else NAN,
                    CLOSE_MASK_FLAG: FLAG_FLAGGED if close else FLAG_OK,
                }
            )
    return pd.DataFrame(rows)


def make_window(*, ident="A", t=1) -> SimpleNamespace:
    df = review_frame()
    return SimpleNamespace(
        filtered_df=df.copy(),
        track_df=df.copy(),
        unique_ids=["A", "B"],
        current_ident_index=0 if ident == "A" else 1,
        current_TrackNumber_plot=1,
        current_time_index=t,
        close_mask_threshold=5.0,
        close_mask_masks=None,
    )


def flag_at(df: pd.DataFrame, ident: str, t: int) -> str:
    return df[(df["Identification"] == ident) & (df["t"] == t)][
        CLOSE_MASK_FLAG
    ].iloc[0]


def outlier_window(*, ident="A", t=0) -> SimpleNamespace:
    """`review_frame()` plus a plain Outlier_detection column: A is an
    outlier at t=0, B at t=0; neither overlaps a close-mask-flagged time.
    """
    df = review_frame()
    df["Outlier_detection"] = "OK"
    df.loc[
        (df["Identification"].isin(["A", "B"])) & (df["t"] == 0),
        "Outlier_detection",
    ] = "Outlier"
    return SimpleNamespace(
        filtered_df=df.copy(),
        track_df=df.copy(),
        unique_ids=["A", "B"],
        current_ident_index=0 if ident == "A" else 1,
        current_TrackNumber_plot=1,
        current_time_index=t,
        close_mask_threshold=5.0,
        close_mask_masks=None,
    )


def outlier_flag_at(df: pd.DataFrame, ident: str, t: int) -> str:
    return df[(df["Identification"] == ident) & (df["t"] == t)][
        "Outlier_detection"
    ].iloc[0]


def test_the_next_flagged_time_is_the_first_one_after_the_current():
    assert next_flagged_time(np.array([1, 4, 7]), 1) == 4
    assert next_flagged_time(np.array([1, 4, 7]), 3) == 4


def test_the_next_flagged_time_wraps_around():
    assert next_flagged_time(np.array([1, 4, 7]), 7) == 1
    assert next_flagged_time(np.array([1, 4, 7]), 9) == 1


def test_there_is_no_next_time_without_flags():
    assert next_flagged_time(np.array([], dtype=int), 3) is None


def test_flagged_times_are_per_identification():
    df = review_frame()

    assert flagged_times(df, "A").tolist() == [1, 3]
    assert flagged_times(df, "B").tolist() == [2]
    assert flagged_times(df, "C").tolist() == []


def test_reviewed_rows_are_not_flagged_times():
    df = review_frame()
    df.loc[(df["Identification"] == "A") & (df["t"] == 1), CLOSE_MASK_FLAG] = (
        FLAG_REVIEWED
    )

    assert flagged_times(df, "A").tolist() == [3]


def test_the_next_ident_follows_the_list_order():
    assert next_flagged_ident(["A", "B", "C"], "A", ["B", "C"]) == "B"
    assert next_flagged_ident(["A", "B", "C"], "B", ["A", "C"]) == "C"


def test_the_next_ident_wraps_around_and_skips_the_current_one():
    assert next_flagged_ident(["A", "B", "C"], "C", ["A", "B"]) == "A"
    assert next_flagged_ident(["A", "B"], "A", ["A"]) is None


def test_the_next_ident_is_none_when_nothing_remains():
    assert next_flagged_ident(["A"], "A", []) is None


def test_the_next_ident_keeps_the_type_of_the_remaining_ids():
    assert next_flagged_ident([1, 2, 3], 1, [3]) == 3


def test_setting_a_flag_updates_the_matching_rows_in_both_tables():
    window = make_window()
    target = window.filtered_df.index[
        (window.filtered_df["Identification"] == "A")
        & (window.filtered_df["t"] == 1)
    ]

    set_close_mask_flag(window, target, FLAG_REVIEWED)

    assert flag_at(window.filtered_df, "A", 1) == FLAG_REVIEWED
    assert flag_at(window.track_df, "A", 1) == FLAG_REVIEWED
    assert flag_at(window.track_df, "A", 3) == FLAG_FLAGGED
    assert flag_at(window.track_df, "B", 2) == FLAG_FLAGGED


def test_setting_a_flag_never_touches_other_positions():
    window = make_window()
    other = window.track_df.assign(Position=2)
    other[CLOSE_MASK_FLAG] = FLAG_FLAGGED
    window.track_df = pd.concat([window.track_df, other], ignore_index=True)
    target = window.filtered_df.index[
        (window.filtered_df["Identification"] == "A")
        & (window.filtered_df["t"] == 1)
    ]

    set_close_mask_flag(window, target, FLAG_REVIEWED)

    position_two = window.track_df[window.track_df["Position"] == 2]
    assert (position_two[CLOSE_MASK_FLAG] == FLAG_FLAGGED).all()


def edit_table() -> pd.DataFrame:
    """Two tracking points at one frame, each with the other's mask as candidate."""
    return pd.DataFrame(
        {
            "Position": [1, 1],
            "Identification": ["A", "B"],
            "TrackNumber": [1, 1],
            "t": [0, 0],
            "alt_label_id_m1": pd.array([2, 1], dtype="Int64"),
            "alt_dist_px_m1": [4.0, 4.0],
            CLOSE_MASK_FLAG: [FLAG_FLAGGED, FLAG_FLAGGED],
        }
    )


def edit_window(df: pd.DataFrame | None = None, **attrs) -> SimpleNamespace:
    window = SimpleNamespace(
        track_df=edit_table() if df is None else df,
        close_mask_threshold=5.0,
        close_mask_masks=None,
    )
    window.__dict__.update(attrs)
    return window


def test_a_row_that_comes_within_the_distance_is_flagged():
    df = edit_table()
    df[CLOSE_MASK_FLAG] = FLAG_OK
    window = edit_window(df)
    window.track_df.loc[1, "alt_dist_px_m1"] = 3.0

    update_flags_after_edit(window, window.track_df.index, edited_row=0)

    assert window.track_df.loc[1, CLOSE_MASK_FLAG] == FLAG_FLAGGED


def test_a_row_that_moves_out_of_reach_is_unflagged():
    window = edit_window()
    window.track_df.loc[1, "alt_dist_px_m1"] = 6.0

    update_flags_after_edit(window, window.track_df.index, edited_row=0)

    assert window.track_df.loc[1, CLOSE_MASK_FLAG] == FLAG_OK


def test_a_row_that_lost_its_candidate_is_unflagged():
    window = edit_window()
    window.track_df.loc[1, ["alt_label_id_m1", "alt_dist_px_m1"]] = [0, NAN]

    update_flags_after_edit(window, window.track_df.index, edited_row=0)

    assert window.track_df.loc[1, CLOSE_MASK_FLAG] == FLAG_OK


def test_reviewed_rows_stay_reviewed_whatever_the_edit_does():
    window = edit_window()
    window.track_df.loc[1, CLOSE_MASK_FLAG] = FLAG_REVIEWED
    window.track_df.loc[1, "alt_dist_px_m1"] = 2.0

    update_flags_after_edit(window, window.track_df.index, edited_row=0)

    assert window.track_df.loc[1, CLOSE_MASK_FLAG] == FLAG_REVIEWED


def test_editing_a_flagged_row_marks_it_reviewed_and_pins_it():
    window = edit_window()

    update_flags_after_edit(window, window.track_df.index, edited_row=0)

    assert window.track_df.loc[0, CLOSE_MASK_FLAG] == FLAG_REVIEWED
    assert window.track_df.loc[1, CLOSE_MASK_FLAG] == FLAG_FLAGGED
    assert window.close_mask_pinned_row == ("A", 1, 0)
    assert is_pinned_close_mask_row(window, window.track_df.loc[0])
    assert not is_pinned_close_mask_row(window, window.track_df.loc[1])


def test_editing_an_unflagged_row_does_not_mark_it_reviewed():
    df = edit_table()
    df[CLOSE_MASK_FLAG] = FLAG_OK
    df["alt_dist_px_m1"] = [NAN, NAN]
    df["alt_label_id_m1"] = pd.array([0, 0], dtype="Int64")
    window = edit_window(df)

    update_flags_after_edit(window, window.track_df.index, edited_row=0)

    assert window.track_df[CLOSE_MASK_FLAG].tolist() == [FLAG_OK, FLAG_OK]
    assert getattr(window, "close_mask_pinned_row", None) is None


def test_editing_a_flagged_row_whose_candidate_vanished_still_reviews_it():
    window = edit_window()
    window.track_df.loc[0, ["alt_label_id_m1", "alt_dist_px_m1"]] = [0, NAN]

    update_flags_after_edit(window, window.track_df.index, edited_row=0)

    assert window.track_df.loc[0, CLOSE_MASK_FLAG] == FLAG_REVIEWED


def test_nothing_happens_before_a_detection_created_the_flags():
    df = edit_table().drop(columns=[CLOSE_MASK_FLAG])
    window = edit_window(df)

    update_flags_after_edit(window, window.track_df.index, edited_row=0)

    assert CLOSE_MASK_FLAG not in window.track_df.columns


def test_without_a_known_distance_flags_are_only_ever_cleared():
    df = edit_table()
    df.loc[0, CLOSE_MASK_FLAG] = FLAG_OK
    window = edit_window(df, close_mask_threshold=None)
    window.track_df.loc[1, ["alt_label_id_m1", "alt_dist_px_m1"]] = [0, NAN]

    update_flags_after_edit(window, window.track_df.index, edited_row=0)

    assert window.track_df.loc[0, CLOSE_MASK_FLAG] == FLAG_OK  # not added
    assert window.track_df.loc[1, CLOSE_MASK_FLAG] == FLAG_OK  # cleared


def test_the_mask_selection_of_the_detection_is_respected():
    df = edit_table()
    df[CLOSE_MASK_FLAG] = FLAG_OK
    df["alt_label_id_m2"] = pd.array([5, 5], dtype="Int64")
    df["alt_dist_px_m2"] = [1.0, 1.0]
    window = edit_window(df, close_mask_masks=[1])

    update_flags_after_edit(window, window.track_df.index, edited_row=0)

    assert window.track_df.loc[0, CLOSE_MASK_FLAG] == FLAG_FLAGGED  # mask 1
    window.track_df.loc[:, "alt_dist_px_m1"] = 9.0
    df[CLOSE_MASK_FLAG] = FLAG_OK

    update_flags_after_edit(window, window.track_df.index, edited_row=0)

    assert window.track_df[CLOSE_MASK_FLAG].tolist() == [FLAG_OK, FLAG_OK]


def test_a_repeated_edit_keeps_the_pin_of_the_first():
    """Right-click on all masks edits the same row once per mask."""
    window = edit_window()

    update_flags_after_edit(window, window.track_df.index, edited_row=0)
    update_flags_after_edit(window, window.track_df.index, edited_row=0)

    assert window.track_df.loc[0, CLOSE_MASK_FLAG] == FLAG_REVIEWED
    assert window.close_mask_pinned_row == close_mask_row_key(
        window.track_df.loc[0]
    )


class Recorder:
    """Stands in for the navigation the key triggers."""

    def __init__(self):
        self.times: list[int] = []
        self.idents: list[str] = []
        self.dialogs: list[str] = []
        self.cleared = 0


@pytest.fixture
def recorder(monkeypatch):
    rec = Recorder()

    def jump(window, t):
        rec.times.append(t)
        window.current_time_index = t

    def go(window, ident):
        rec.idents.append(ident)
        window.current_ident_index = window.unique_ids.index(ident)
        window.ident = ident
        window.current_time_index = 0
        return True

    monkeypatch.setattr(close_mask_review, "_jump_to_time", jump)
    monkeypatch.setattr(close_mask_review, "go_to_ident", go)
    monkeypatch.setattr(
        close_mask_review,
        "_show_info_dialog",
        lambda window, title, text: rec.dialogs.append(text),
    )
    monkeypatch.setattr(
        close_mask_review,
        "clear_close_mask_view",
        lambda window: setattr(rec, "cleared", rec.cleared + 1),
    )
    monkeypatch.setattr(
        close_mask_review, "_refresh_close_mask_views", refresh_ids
    )
    monkeypatch.setattr(
        close_mask_review, "_refresh_outlier_views", refresh_outlier_ids
    )
    monkeypatch.setattr(
        close_mask_review, "select_ident_in_tree", lambda w: None
    )
    return rec


def refresh_ids(window) -> None:
    """What the real refresh does to the data: rebuild the flagged id list."""
    from SECQUOIA.core.outlier_detection.close_masks import (
        update_unique_close_mask_ids,
    )

    update_unique_close_mask_ids(window)


def refresh_outlier_ids(window) -> None:
    """What the real refresh does to the data: rebuild the outlier id list."""
    update_unique_outliers_ids(window)


def test_c_marks_the_row_reviewed_in_both_tables_and_jumps_to_the_next_time(
    recorder,
):
    window = make_window(ident="A", t=1)

    mark_close_mask_checked(window)

    assert flag_at(window.filtered_df, "A", 1) == FLAG_REVIEWED
    assert flag_at(window.track_df, "A", 1) == FLAG_REVIEWED
    assert recorder.times == [3]
    assert recorder.idents == []


def test_the_jump_wraps_to_the_first_flagged_time(recorder):
    window = make_window(ident="A", t=3)

    mark_close_mask_checked(window)

    assert flag_at(window.filtered_df, "A", 3) == FLAG_REVIEWED
    assert recorder.times == [1]


def test_an_identification_without_flags_left_moves_on_to_the_next(recorder):
    window = make_window(ident="A", t=1)
    mark_close_mask_checked(window)
    recorder.times.clear()

    mark_close_mask_checked(window)

    assert flag_at(window.filtered_df, "A", 3) == FLAG_REVIEWED
    assert recorder.idents == ["B"]
    assert recorder.times == [2]
    assert window.unique_close_mask_ids == ["B"]


def test_the_last_case_restores_the_normal_view_and_says_so(recorder):
    window = make_window(ident="B", t=2)

    mark_close_mask_checked(window)
    assert recorder.idents == ["A"]
    assert recorder.dialogs == []
    recorder.idents.clear()
    window.current_ident_index, window.current_time_index = 0, 1
    mark_close_mask_checked(window)
    mark_close_mask_checked(window)

    assert recorder.dialogs == ["All close-mask cases checked"]
    assert recorder.cleared == 1
    assert window.unique_close_mask_ids == []


def test_a_single_remaining_case_ends_the_review(recorder):
    window = make_window(ident="B", t=2)
    window.filtered_df.loc[
        window.filtered_df["Identification"] == "A", CLOSE_MASK_FLAG
    ] = FLAG_REVIEWED
    window.track_df.loc[
        window.track_df["Identification"] == "A", CLOSE_MASK_FLAG
    ] = FLAG_REVIEWED

    mark_close_mask_checked(window)

    assert recorder.dialogs == ["All close-mask cases checked"]
    assert recorder.times == []
    assert recorder.idents == []


def test_a_row_that_is_not_flagged_is_left_alone(recorder):
    window = make_window(ident="A", t=0)

    mark_close_mask_checked(window)

    assert flag_at(window.filtered_df, "A", 0) == FLAG_OK
    assert flagged_times(window.filtered_df, "A").tolist() == [1, 3]
    assert recorder.times == [] and recorder.idents == []


def test_a_row_reviewed_earlier_is_not_touched_by_c(recorder):
    window = make_window(ident="A", t=1)
    set_close_mask_flag(window, [1], FLAG_REVIEWED)

    mark_close_mask_checked(window)

    assert recorder.times == []


def test_a_row_just_reviewed_by_an_edit_counts_as_checked(recorder):
    window = make_window(ident="A", t=1)
    set_close_mask_flag(window, [1], FLAG_REVIEWED)
    window.close_mask_pinned_row = close_mask_row_key(
        window.filtered_df.loc[1]
    )

    mark_close_mask_checked(window)

    assert recorder.times == [3]
    assert window.close_mask_pinned_row is None
    assert flag_at(window.filtered_df, "A", 1) == FLAG_REVIEWED


def test_c_before_any_detection_does_nothing(recorder, caplog):
    window = make_window()
    window.filtered_df = window.filtered_df.drop(columns=[CLOSE_MASK_FLAG])

    mark_close_mask_checked(window)

    assert recorder.times == [] and recorder.dialogs == []
    assert "press Find" in caplog.text


def test_reviewed_flags_are_what_gets_saved(recorder):
    window = make_window(ident="A", t=1)

    mark_close_mask_checked(window)

    saved = window.track_df[window.track_df[CLOSE_MASK_FLAG] == FLAG_REVIEWED]
    assert saved["t"].tolist() == [1]


def test_a_second_pass_of_the_detection_keeps_the_checked_cases(recorder):
    from SECQUOIA.core.outlier_detection import apply_close_mask_detection

    window = make_window(ident="A", t=1)
    mark_close_mask_checked(window)

    apply_close_mask_detection(window, 5.0)

    assert flag_at(window.filtered_df, "A", 1) == FLAG_REVIEWED
    assert flag_at(window.track_df, "A", 1) == FLAG_REVIEWED


def test_mark_outlier_reviewed_marks_the_row_and_moves_to_the_next_identification(
    recorder,
):
    window = outlier_window(ident="A", t=0)

    mark_outlier_reviewed(window)

    assert outlier_flag_at(window.filtered_df, "A", 0) == "Reviewed"
    assert outlier_flag_at(window.track_df, "A", 0) == "Reviewed"
    assert recorder.idents == ["B"]
    assert recorder.times == [0]


def test_mark_outlier_reviewed_shows_the_all_reviewed_dialog_once_nothing_is_left(
    recorder,
):
    window = outlier_window(ident="A", t=0)
    mark_outlier_reviewed(window)
    recorder.idents.clear()

    mark_outlier_reviewed(window)

    assert recorder.dialogs == ["All outliers reviewed"]
    assert window.unique_outliers_ids == []


def test_mark_outlier_reviewed_leaves_a_non_outlier_row_alone(recorder):
    window = outlier_window(ident="A", t=1)

    mark_outlier_reviewed(window)

    assert outlier_flag_at(window.filtered_df, "A", 1) == "OK"
    assert recorder.times == [] and recorder.idents == []


def test_mark_outlier_reviewed_before_any_detection_does_nothing(
    recorder, caplog
):
    window = outlier_window()
    window.filtered_df = window.filtered_df.drop(columns=["Outlier_detection"])

    mark_outlier_reviewed(window)

    assert recorder.times == [] and recorder.dialogs == []
    assert "run outlier detection" in caplog.text


def test_a_second_pass_of_the_detection_keeps_the_reviewed_outlier(recorder):
    from SECQUOIA.core.outlier_detection.detection import (
        RulesPack,
        run_outlier_pipeline,
    )

    window = outlier_window(ident="A", t=0)
    mark_outlier_reviewed(window)
    pack = RulesPack(
        version=1,
        m_n=1,
        ch_n=1,
        rules=[],
        sliding_windows=[],
    )

    window.filtered_df = run_outlier_pipeline(window.filtered_df, pack)

    assert outlier_flag_at(window.filtered_df, "A", 0) == "Reviewed"


def test_review_current_point_reviews_a_close_mask_case_first(recorder):
    window = make_window(ident="A", t=1)

    review_current_point(window)

    assert flag_at(window.filtered_df, "A", 1) == FLAG_REVIEWED


def test_review_current_point_reviews_a_plain_outlier(recorder):
    window = outlier_window(ident="A", t=0)

    review_current_point(window)

    assert outlier_flag_at(window.filtered_df, "A", 0) == "Reviewed"


def test_review_current_point_does_nothing_on_an_unflagged_row(recorder):
    window = outlier_window(ident="A", t=2)

    review_current_point(window)

    assert recorder.times == [] and recorder.idents == []
    assert outlier_flag_at(window.filtered_df, "A", 2) == "OK"
    assert flag_at(window.filtered_df, "A", 2) == FLAG_OK


@pytest.fixture
def hotkey_host(qtbot, monkeypatch):
    from SECQUOIA.gui.main_window import key_bindings

    class Host(key_bindings.KeyBindings, QWidget):
        def __getattr__(self, name):
            if name in ("_global_hotkeys_installed", "_shortcuts"):
                raise AttributeError(name)
            return lambda *args, **kwargs: None

    host = Host()
    host._brush_shortcuts = []
    qtbot.addWidget(host)
    calls = []
    monkeypatch.setattr(
        key_bindings, "review_current_point", lambda w: calls.append(w)
    )
    host.install_global_hotkeys()
    return host, calls


def shortcut_for(host, key: str):
    return next(sc for sc in host._shortcuts if sc.key().toString() == key)


def test_the_c_key_reviews_the_current_point(hotkey_host):
    host, calls = hotkey_host

    shortcut_for(host, "C").activated.emit()

    assert calls == [host]


def test_the_c_key_is_ignored_while_typing(hotkey_host, qtbot, monkeypatch):
    host, calls = hotkey_host
    field = QLineEdit(host)
    qtbot.addWidget(field)
    monkeypatch.setattr(
        QApplication, "focusWidget", staticmethod(lambda: field)
    )

    shortcut_for(host, "C").activated.emit()

    assert calls == []


def test_the_c_key_works_again_once_the_text_field_loses_focus(
    hotkey_host, monkeypatch
):
    host, calls = hotkey_host
    monkeypatch.setattr(
        QApplication, "focusWidget", staticmethod(lambda: None)
    )

    shortcut_for(host, "C").activated.emit()

    assert calls == [host]


def navigation_window(**attrs) -> SimpleNamespace:
    return SimpleNamespace(
        unique_ids=["A", "B"],
        current_ident_index=0,
        current_time_index=5,
        current_TrackNumber_plot=3,
        zoom_in=lambda: None,
        _keep_lineage_collapsed_after_update=lambda: None,
        fit_all_plots=lambda: None,
        _refresh_all_row_igt=lambda: None,
        _refresh_all_row_summaries=lambda: None,
        **attrs,
    )


@pytest.fixture
def navigation_calls(monkeypatch):
    calls = []
    for name in (
        "ensure_current_df_subset",
        "update_plot",
        "lineage_tree",
        "set_active_layers_from_header_buttons",
        "select_ident_in_tree",
    ):
        monkeypatch.setattr(
            navigation, name, lambda *a, _name=name, **k: calls.append(_name)
        )
    return calls


def test_going_to_an_ident_starts_at_its_first_time_point(navigation_calls):
    window = navigation_window()

    assert navigation.go_to_ident(window, "B") is True

    assert window.current_ident_index == 1
    assert window.ident == "B"
    assert window.current_time_index == 0
    assert window.current_TrackNumber_plot == 1
    assert "select_ident_in_tree" in navigation_calls


def test_going_to_an_unknown_ident_changes_nothing(navigation_calls):
    window = navigation_window()

    assert navigation.go_to_ident(window, "Z") is False

    assert window.current_ident_index == 0
    assert window.current_time_index == 5
    assert navigation_calls == []


def test_the_outlier_jump_still_goes_through_go_to_ident(monkeypatch):
    seen = []
    monkeypatch.setattr(
        navigation, "go_to_ident", lambda window, ident: seen.append(ident)
    )
    window = navigation_window(
        track_df=pd.DataFrame({"Outlier_detection": ["OK"]}),
        unique_outliers_ids=["B", "A"],
        current_outlier_index=0,
    )

    navigation.change_outlier(window, "next")
    navigation.change_outlier(window, "next")

    assert seen == ["A", "B"]


@pytest.fixture
def time_jumps(monkeypatch):
    jumps = []
    monkeypatch.setattr(
        navigation, "_on_time_index_changed", lambda window, t: jumps.append(t)
    )
    return jumps


def outlier_times(window, ident: str, times) -> None:
    df = window.filtered_df
    df["Outlier_detection"] = "OK"
    df.loc[
        (df["Identification"] == ident) & (df["t"].isin(times)),
        "Outlier_detection",
    ] = "Outlier"


def test_the_arrow_jump_finds_close_mask_time_points(time_jumps):
    window = make_window(ident="A", t=1)

    navigation.change_to_next_outlier(window, +1)
    window.current_time_index = 3
    navigation.change_to_next_outlier(window, +1)
    navigation.change_to_next_outlier(window, -1)

    assert time_jumps == [3, 1, 1]


def test_the_arrow_jump_walks_back_through_close_mask_time_points(time_jumps):
    window = make_window(ident="A", t=3)

    navigation.change_to_next_outlier(window, -1)

    assert time_jumps == [1]


def test_the_arrow_jump_combines_outliers_and_close_masks(time_jumps):
    window = make_window(ident="A", t=1)
    outlier_times(window, "A", [2])

    navigation.change_to_next_outlier(window, +1)
    window.current_time_index = 2
    navigation.change_to_next_outlier(window, +1)

    assert time_jumps == [2, 3]


def test_the_arrow_jump_still_works_for_outliers_alone(time_jumps):
    window = make_window(ident="A", t=0)
    window.filtered_df[CLOSE_MASK_FLAG] = FLAG_OK
    outlier_times(window, "A", [2])

    navigation.change_to_next_outlier(window, +1)

    assert time_jumps == [2]


def test_the_arrow_jump_skips_reviewed_close_masks(time_jumps):
    window = make_window(ident="A", t=0)
    set_close_mask_flag(window, [1], FLAG_REVIEWED)

    navigation.change_to_next_outlier(window, +1)

    assert time_jumps == [3]


def test_the_arrow_jump_stays_within_the_current_identification(time_jumps):
    window = make_window(ident="B", t=0)

    navigation.change_to_next_outlier(window, +1)

    assert time_jumps == [2]


def test_the_arrow_jump_does_nothing_without_any_time_point(time_jumps):
    window = make_window(ident="A", t=1)
    window.filtered_df[CLOSE_MASK_FLAG] = FLAG_OK

    navigation.change_to_next_outlier(window, +1)

    assert time_jumps == []


def test_the_arrow_jump_needs_neither_column_to_exist(time_jumps):
    window = make_window(ident="A", t=1)
    window.filtered_df = window.filtered_df.drop(columns=[CLOSE_MASK_FLAG])

    navigation.change_to_next_outlier(window, +1)

    assert time_jumps == []
