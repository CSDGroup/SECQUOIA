"""Characterization tests for SECQUOIA.core.tracking lineage-edit handlers."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from SECQUOIA.core import tracking
from SECQUOIA.core.tracking import (
    on_division_clicked,
    on_fuse_trees_clicked,
    on_new_id_clicked,
    on_remove_division_clicked,
    on_split_tree,
)
from SECQUOIA.core.tracking.identity import _next_ident_like
from SECQUOIA.core.tracking.numbering import (
    _is_desc,
    _path_from_root,
    _remap_from_to,
    _remap_tracknumber_from_root,
)
from SECQUOIA.gui.common import messages as _messages
from SECQUOIA.gui.dialogs import track_fuse_dialog

IDENT_1 = "250615MA40-p0002-001"
IDENT_2 = "250615MA40-p0002-002"
IDENT_3 = "250615MA40-p0002-003"

COLUMNS = [
    "Position",
    "t",
    "XMorphology",
    "YMorphology",
    "AreaMorphologyM1",
    "MeanNoBgCorrectedCh00M1",
    "track_id",
    "TrackNumber",
    "Cellfate",
    "Identification",
    "active",
    "inspected",
    "Calculated_Time",
]

TIME_INTERVAL_SECONDS = 300.0


# Fakes
class FakeLineEdit:
    """Replacement for Qt QLineEdit."""

    def __init__(self, value: str = "") -> None:
        self.value = value

    def text(self) -> str:
        return self.value

    def clear(self) -> None:
        self.value = ""


class FakeSpinBox:
    """Replacement for Qt QSpinBox."""

    def __init__(self, value: int) -> None:
        self.value_ = value

    def value(self) -> int:
        return self.value_


class FakeInputDialog:
    """Scriptable stand-in for QInputDialog."""

    CANCELLED = object()

    def __init__(self, choice=None) -> None:
        self.choice = choice
        self.calls: list[tuple] = []

    def getItem(self, parent, title, label, options, index=0, editable=False):
        self.calls.append((title, tuple(options)))
        if self.choice is self.CANCELLED:
            return "", False
        if self.choice is None:
            return options[index], True
        return self.choice, True


class FakeMessageBox:
    """Records information() calls instead of showing a modal."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def information(self, parent, title, text) -> None:
        self.messages.append((title, text))


def _row(t: int, track_id: int, tracknumber: int, base: float) -> dict:
    return {
        "Position": 2,
        "t": t,
        "XMorphology": base + t,
        "YMorphology": base + 10 + t,
        "AreaMorphologyM1": base * 10 + t,
        "MeanNoBgCorrectedCh00M1": base / 10 + t,
        "track_id": track_id,
        "TrackNumber": tracknumber,
        "Cellfate": "Healthy",
        "Identification": IDENT_1 if track_id == 1 else IDENT_2,
        "active": 1,
        "inspected": 0,
        "Calculated_Time": t * TIME_INTERVAL_SECONDS / 60.0,
    }


def make_track_df() -> pd.DataFrame:
    """Two lineages; the first divides at t=3."""
    rows = [_row(t, 1, 1, 10.0) for t in range(3)]
    for t in range(3, 6):
        rows.append(_row(t, 1, 2, 30.0))
        rows.append(_row(t, 1, 3, 40.0))
    rows.extend(_row(t, 2, 1, 50.0) for t in range(6))
    return pd.DataFrame(rows, columns=COLUMNS)


def make_main_window(
    df: pd.DataFrame | None = None,
    ident: str = IDENT_1,
    t: int = 3,
    tracknumber: int | None = 1,
) -> SimpleNamespace:
    """Smallest main_window-like object the edit handlers touch."""
    df = make_track_df() if df is None else df
    main_window = SimpleNamespace(
        folder_list=["/exp"],
        track_df=df.copy(),
        filtered_df=df.copy(),
        unique_ids=df["Identification"].unique().tolist(),
        experiment_name="250615MA40",
        position_folders=["/exp/250615MA40_p0002"],
        current_position_index=0,
        current_time_index=t,
        ident=ident,
        current_TrackNumber_plot=tracknumber,
        Time_input=FakeLineEdit(str(TIME_INTERVAL_SECONDS)),
        fuse_time_spin=FakeSpinBox(t),
        fuse_tree_id_field_1=FakeLineEdit(),
        fuse_tree_id_field_2=FakeLineEdit(),
        fuse_cell_field_1=FakeLineEdit(),
        fuse_cell_field_2=FakeLineEdit(),
        fuse_dialog=None,
        _fuse_next_slot=0,
        jumped=None,
    )
    main_window.update_display_track_id = lambda: None
    return main_window


@pytest.fixture
def silence_ui(monkeypatch):
    """Neutralise the GUI refresh calls made by _write_back_and_refresh."""

    def fake_jump(main_window, ident=None, t=None, tracknumber=None):
        main_window.jumped = (ident, t, tracknumber)

    monkeypatch.setattr(tracking.edits, "jump_to_identification", fake_jump)
    monkeypatch.setattr(track_fuse_dialog, "jump_to_identification", fake_jump)
    monkeypatch.setattr(
        tracking.edits, "ensure_current_df_subset", lambda mw: None
    )
    monkeypatch.setattr(
        tracking.history, "ensure_current_df_subset", lambda mw: None
    )
    monkeypatch.setattr(tracking.edits, "update_list", lambda mw: None)
    monkeypatch.setattr(tracking.history, "update_list", lambda mw: None)
    monkeypatch.setattr(tracking.history, "update_plot", lambda mw: None)
    monkeypatch.setattr(tracking.history, "lineage_tree", lambda mw: None)
    monkeypatch.setattr(
        tracking.context, "_current_t_range", lambda mw: (1, 6, 0, 5)
    )
    monkeypatch.setattr(
        tracking.history, "build_realtime_lookup", lambda mw: None
    )
    monkeypatch.setattr(_messages, "show_folder_warning", lambda mw=None: None)
    return None


@pytest.fixture
def message_box(monkeypatch):
    box = FakeMessageBox()
    monkeypatch.setattr(tracking.edits, "QMessageBox", box)
    monkeypatch.setattr(tracking.fuse, "QMessageBox", box)
    return box


def spans(df: pd.DataFrame) -> dict[tuple[str, int], tuple[int, int, int]]:
    """Map (Identification, TrackNumber) -> (t_min, t_max, row_count)."""
    grouped = (
        df.groupby(["Identification", "TrackNumber"])["t"]
        .agg(["min", "max", "count"])
        .astype(int)
    )
    return {
        (str(ident), int(tn)): tuple(int(v) for v in row)
        for (ident, tn), row in grouped.iterrows()
    }


def assert_no_duplicate_rows(df: pd.DataFrame) -> None:
    """(Identification, t, TrackNumber) must stay unique after every edit."""
    duplicated = df.duplicated(
        subset=["Identification", "t", "TrackNumber"]
    ).sum()
    assert (
        duplicated == 0
    ), f"{duplicated} duplicate (ident, t, TrackNumber) rows"


# Helpers
@pytest.mark.parametrize(
    ("node", "root", "expected"),
    [
        (1, 1, True),
        (2, 1, True),
        (3, 1, True),
        (5, 2, True),
        (4, 2, True),
        (5, 3, False),
        (2, 3, False),
        (1, 2, False),
    ],
)
def test_is_desc(node, root, expected):
    assert _is_desc(node, root) is expected


def test_is_desc_rejects_garbage():
    assert _is_desc("abc", 1) is False
    assert _is_desc(None, 1) is False


@pytest.mark.parametrize(
    ("node", "root", "expected"),
    [(1, 1, []), (2, 1, ["L"]), (3, 1, ["R"]), (5, 1, ["L", "R"])],
)
def test_path_from_root(node, root, expected):
    assert _path_from_root(node, root) == expected


@pytest.mark.parametrize(
    ("node", "root", "expected"),
    [(4, 4, 1), (8, 4, 2), (9, 4, 3), (18, 4, 6), (1, 1, 1), (3, 1, 3)],
)
def test_remap_tracknumber_from_root(node, root, expected):
    assert _remap_tracknumber_from_root(node, root) == expected


@pytest.mark.parametrize(
    ("node", "src", "dst", "expected"),
    [(1, 1, 2, 2), (2, 1, 2, 4), (3, 1, 2, 5), (4, 2, 6, 12)],
)
def test_remap_from_to(node, src, dst, expected):
    assert _remap_from_to(node, src, dst) == expected


def test_next_ident_like_increments_numeric_suffix():
    ids = pd.Series([IDENT_1, IDENT_2])
    assert _next_ident_like(IDENT_1, ids) == IDENT_3


def test_next_ident_like_skips_gaps_and_takes_the_maximum():
    ids = pd.Series([IDENT_1, "250615MA40-p0002-007"])
    assert _next_ident_like(IDENT_1, ids) == "250615MA40-p0002-008"


# on_new_id_clicked
def test_new_id_appends_next_free_identification(silence_ui):
    main_window = make_main_window()
    on_new_id_clicked(main_window)

    layout = spans(main_window.track_df)
    assert (IDENT_3, 1) in layout
    assert layout[(IDENT_3, 1)] == (0, 5, 6)
    assert_no_duplicate_rows(main_window.track_df)


def test_new_id_rows_are_blank_but_carry_defaults(silence_ui):
    main_window = make_main_window()
    on_new_id_clicked(main_window)

    new_rows = main_window.track_df[
        main_window.track_df["Identification"] == IDENT_3
    ]
    assert list(new_rows["TrackNumber"]) == [1] * 6
    assert list(new_rows["Cellfate"]) == ["Healthy"] * 6
    assert list(new_rows["active"]) == [1] * 6
    assert list(new_rows["inspected"]) == [0] * 6
    assert list(new_rows["Position"]) == [2] * 6
    # Measurement columns start empty.
    assert new_rows["AreaMorphologyM1"].eq(0).all()
    assert new_rows["MeanNoBgCorrectedCh00M1"].eq(0).all()
    # Calculated_Time is derived from t and the Time_input field.
    assert list(new_rows["Calculated_Time"]) == [
        t * TIME_INTERVAL_SECONDS / 60.0 for t in range(6)
    ]


def test_new_id_moves_the_selection_to_the_new_track(silence_ui):
    main_window = make_main_window()
    on_new_id_clicked(main_window)
    assert main_window.jumped == (IDENT_3, 0, 1)


def test_new_id_leaves_existing_lineages_untouched(silence_ui):
    main_window = make_main_window()
    before = spans(main_window.track_df)
    on_new_id_clicked(main_window)
    after = spans(main_window.track_df)
    for key, value in before.items():
        assert after[key] == value


def test_new_id_on_empty_frame_creates_the_first_identification(silence_ui):
    empty = make_track_df().iloc[0:0]
    main_window = make_main_window(df=empty)
    on_new_id_clicked(main_window)

    idents = main_window.track_df["Identification"].unique().tolist()
    assert idents == ["250615MA40-p0002-001"]
    assert len(main_window.track_df) == 6


# on_division_clicked
def test_division_creates_two_daughters_and_truncates_the_parent(silence_ui):
    main_window = make_main_window(t=4, tracknumber=2)
    on_division_clicked(main_window)

    layout = spans(main_window.track_df)
    # Parent 2 now stops at t=3, the frame before the division.
    assert layout[(IDENT_1, 2)] == (3, 3, 1)
    # Daughters 4 and 5 = 2*g and 2*g+1 run from the division to the end.
    assert layout[(IDENT_1, 4)] == (4, 5, 2)
    assert layout[(IDENT_1, 5)] == (4, 5, 2)
    # The sibling subtree is untouched.
    assert layout[(IDENT_1, 3)] == (3, 5, 3)
    assert_no_duplicate_rows(main_window.track_df)


def test_division_selects_the_first_daughter(silence_ui):
    main_window = make_main_window(t=4, tracknumber=2)
    on_division_clicked(main_window)
    assert main_window.current_TrackNumber_plot == 4


def test_division_daughter_rows_are_blank_with_defaults(silence_ui):
    main_window = make_main_window(t=4, tracknumber=2)
    on_division_clicked(main_window)

    daughters = main_window.track_df[
        (main_window.track_df["Identification"] == IDENT_1)
        & (main_window.track_df["TrackNumber"].isin([4, 5]))
    ]
    assert daughters["AreaMorphologyM1"].eq(0).all()
    assert list(daughters["Cellfate"]) == ["Healthy"] * 4
    assert daughters["track_id"].eq(1).all()


def test_division_is_idempotent(silence_ui):
    """Dividing the same track twice must not duplicate the daughter rows."""
    main_window = make_main_window(t=4, tracknumber=2)
    on_division_clicked(main_window)
    first = spans(main_window.track_df)

    main_window.current_TrackNumber_plot = 2
    main_window.current_time_index = 4
    on_division_clicked(main_window)

    assert spans(main_window.track_df) == first
    assert_no_duplicate_rows(main_window.track_df)


def test_division_leaves_other_identifications_alone(silence_ui):
    main_window = make_main_window(t=4, tracknumber=2)
    on_division_clicked(main_window)
    layout = spans(main_window.track_df)
    assert layout[(IDENT_2, 1)] == (0, 5, 6)


# on_remove_division_clicked
def test_remove_division_keep_first_daughter_promotes_it(
    silence_ui, monkeypatch
):
    monkeypatch.setattr(
        tracking.edits, "QInputDialog", FakeInputDialog("Keep 2")
    )
    main_window = make_main_window(t=3, tracknumber=1)
    on_remove_division_clicked(main_window)

    layout = spans(main_window.track_df)
    # The division is gone: one continuous track over the full range.
    assert layout == {(IDENT_1, 1): (0, 5, 6), (IDENT_2, 1): (0, 5, 6)}
    assert_no_duplicate_rows(main_window.track_df)


def test_remove_division_keep_first_daughter_keeps_its_measurements(
    silence_ui, monkeypatch
):
    monkeypatch.setattr(
        tracking.edits, "QInputDialog", FakeInputDialog("Keep 2")
    )
    main_window = make_main_window(t=3, tracknumber=1)
    before = make_track_df()
    expected = before.loc[
        (before["Identification"] == IDENT_1) & (before["TrackNumber"] == 2),
        "AreaMorphologyM1",
    ].tolist()

    on_remove_division_clicked(main_window)

    kept = main_window.track_df[
        (main_window.track_df["Identification"] == IDENT_1)
        & (main_window.track_df["t"] >= 3)
    ]
    assert kept["AreaMorphologyM1"].tolist() == expected


def test_remove_division_keep_none_zeroes_the_parent_forward(
    silence_ui, monkeypatch
):
    monkeypatch.setattr(
        tracking.edits, "QInputDialog", FakeInputDialog("Keep none")
    )
    main_window = make_main_window(t=3, tracknumber=1)
    on_remove_division_clicked(main_window)

    parent = main_window.track_df[
        main_window.track_df["Identification"] == IDENT_1
    ].sort_values("t")
    assert spans(main_window.track_df)[(IDENT_1, 1)] == (0, 5, 6)
    # History before the division survives.
    assert parent.loc[parent["t"] < 3, "AreaMorphologyM1"].tolist() == [
        100.0,
        101.0,
        102.0,
    ]
    # From the division onward every measurement is blanked.
    forward = parent[parent["t"] >= 3]
    assert forward["AreaMorphologyM1"].eq(0).all()
    assert forward["XMorphology"].eq(0).all()
    assert_no_duplicate_rows(main_window.track_df)


def test_remove_division_keep_none_preserves_bookkeeping_columns(
    silence_ui, monkeypatch
):
    """SAFE_KEEP_NUMERIC columns must survive the blanking."""
    monkeypatch.setattr(
        tracking.edits, "QInputDialog", FakeInputDialog("Keep none")
    )
    main_window = make_main_window(t=3, tracknumber=1)
    on_remove_division_clicked(main_window)

    forward = main_window.track_df[
        (main_window.track_df["Identification"] == IDENT_1)
        & (main_window.track_df["t"] >= 3)
    ]
    assert forward["t"].tolist() == [3, 4, 5]
    assert forward["Position"].eq(2).all()
    assert forward["TrackNumber"].eq(1).all()


def test_remove_division_cancelled_changes_nothing(silence_ui, monkeypatch):
    monkeypatch.setattr(
        tracking.edits,
        "QInputDialog",
        FakeInputDialog(FakeInputDialog.CANCELLED),
    )
    main_window = make_main_window(t=3, tracknumber=1)
    before = main_window.track_df.copy()
    on_remove_division_clicked(main_window)
    pd.testing.assert_frame_equal(main_window.track_df, before)


def test_remove_division_offers_both_daughters_and_keep_none(
    silence_ui, monkeypatch
):
    dialog = FakeInputDialog(FakeInputDialog.CANCELLED)
    monkeypatch.setattr(tracking.edits, "QInputDialog", dialog)
    main_window = make_main_window(t=3, tracknumber=1)
    on_remove_division_clicked(main_window)

    assert dialog.calls[0][1] == ("Keep 2", "Keep 3", "Keep none")


# on_split_tree
def test_split_tree_moves_the_subtree_to_a_new_identification(silence_ui):
    main_window = make_main_window(t=3, tracknumber=1)
    on_split_tree(main_window)

    layout = spans(main_window.track_df)
    # Everything from t=3 onward now belongs to the new Tree ID.
    assert layout[(IDENT_3, 2)] == (3, 5, 3)
    assert layout[(IDENT_3, 3)] == (3, 5, 3)
    assert_no_duplicate_rows(main_window.track_df)


def test_split_tree_backfills_the_new_identification_to_the_start(silence_ui):
    main_window = make_main_window(t=3, tracknumber=1)
    on_split_tree(main_window)

    layout = spans(main_window.track_df)
    assert layout[(IDENT_3, 1)] == (0, 2, 3)

    stub = main_window.track_df[
        (main_window.track_df["Identification"] == IDENT_3)
        & (main_window.track_df["TrackNumber"] == 1)
    ]
    assert stub["AreaMorphologyM1"].eq(0).all()


def test_split_tree_leaves_a_blank_continuation_on_the_original(silence_ui):
    main_window = make_main_window(t=3, tracknumber=1)
    on_split_tree(main_window)

    layout = spans(main_window.track_df)
    # The original keeps a TrackNumber 1 row for every frame ...
    assert layout[(IDENT_1, 1)] == (0, 5, 6)

    original = main_window.track_df[
        main_window.track_df["Identification"] == IDENT_1
    ].sort_values("t")
    assert original.loc[original["t"] < 3, "AreaMorphologyM1"].tolist() == [
        100.0,
        101.0,
        102.0,
    ]
    assert original.loc[original["t"] >= 3, "AreaMorphologyM1"].eq(0).all()


def test_split_tree_assigns_a_new_track_id(silence_ui):
    main_window = make_main_window(t=3, tracknumber=1)
    on_split_tree(main_window)

    new_rows = main_window.track_df[
        main_window.track_df["Identification"] == IDENT_3
    ]
    assert new_rows["track_id"].nunique() == 1
    assert int(new_rows["track_id"].iloc[0]) == 3


def test_split_tree_selects_the_new_tree(silence_ui):
    main_window = make_main_window(t=3, tracknumber=1)
    on_split_tree(main_window)
    assert main_window.jumped == (IDENT_3, 3, 1)


def test_split_tree_with_no_future_rows_is_a_no_op(silence_ui, message_box):
    main_window = make_main_window(t=99, tracknumber=1)
    before = main_window.track_df.copy()
    on_split_tree(main_window)

    pd.testing.assert_frame_equal(main_window.track_df, before)
    assert message_box.messages and message_box.messages[0][0] == "Split Tree"


# on_fuse_trees_clicked
def _prepare_fuse(main_window, cell_1: str = "2", cell_2: str = "1"):
    main_window.fuse_tree_id_field_1 = FakeLineEdit(IDENT_1)
    main_window.fuse_tree_id_field_2 = FakeLineEdit(IDENT_2)
    main_window.fuse_cell_field_1 = FakeLineEdit(cell_1)
    main_window.fuse_cell_field_2 = FakeLineEdit(cell_2)
    return main_window


def test_fuse_grafts_the_second_subtree_onto_the_first(silence_ui):
    main_window = _prepare_fuse(make_main_window(t=3, tracknumber=1))
    on_fuse_trees_clicked(main_window)

    layout = spans(main_window.track_df)
    assert {ident for ident, _ in layout} == {IDENT_3}
    assert layout[(IDENT_3, 2)] == (3, 5, 3)
    assert layout[(IDENT_3, 1)] == (0, 2, 3)
    assert layout[(IDENT_3, 3)] == (3, 5, 3)
    assert_no_duplicate_rows(main_window.track_df)


def test_fuse_carries_the_second_trees_measurements(silence_ui):
    main_window = _prepare_fuse(make_main_window(t=3, tracknumber=1))
    source = make_track_df()
    expected = source.loc[
        (source["Identification"] == IDENT_2) & (source["t"] >= 3),
        "AreaMorphologyM1",
    ].tolist()

    on_fuse_trees_clicked(main_window)

    grafted = main_window.track_df[
        main_window.track_df["TrackNumber"] == 2
    ].sort_values("t")
    assert grafted["AreaMorphologyM1"].tolist() == expected


def test_fuse_assigns_one_track_id_to_the_whole_result(silence_ui):
    main_window = _prepare_fuse(make_main_window(t=3, tracknumber=1))
    on_fuse_trees_clicked(main_window)
    assert main_window.track_df["track_id"].nunique() == 1


def test_fuse_selects_the_merged_tree(silence_ui):
    main_window = _prepare_fuse(make_main_window(t=3, tracknumber=1))
    on_fuse_trees_clicked(main_window)
    assert main_window.jumped == (IDENT_3, 3, 2)


def test_fuse_clears_the_dialog_fields(silence_ui):
    main_window = _prepare_fuse(make_main_window(t=3, tracknumber=1))
    on_fuse_trees_clicked(main_window)
    assert main_window.fuse_tree_id_field_1.text() == ""
    assert main_window.fuse_cell_field_2.text() == ""


def test_fuse_discards_the_second_trees_pre_fuse_history(silence_ui):
    main_window = _prepare_fuse(make_main_window(t=3, tracknumber=1))
    rows_before = len(main_window.track_df)

    on_fuse_trees_clicked(main_window)

    assert len(main_window.track_df) == rows_before - 6
    early = main_window.track_df[main_window.track_df["t"] < 3]
    assert len(early) == 3


def test_fuse_requires_two_distinct_identifications(silence_ui):
    main_window = make_main_window(t=3, tracknumber=1)
    main_window.fuse_tree_id_field_1 = FakeLineEdit(IDENT_1)
    main_window.fuse_tree_id_field_2 = FakeLineEdit(IDENT_1)
    before = main_window.track_df.copy()

    on_fuse_trees_clicked(main_window)

    pd.testing.assert_frame_equal(main_window.track_df, before)


def test_fuse_without_a_matching_subtree_is_a_no_op(silence_ui, message_box):
    main_window = _prepare_fuse(
        make_main_window(t=3, tracknumber=1), cell_2="7"
    )
    before = main_window.track_df.copy()

    on_fuse_trees_clicked(main_window)

    pd.testing.assert_frame_equal(main_window.track_df, before)
    assert message_box.messages


def test_every_edit_keeps_the_column_set_stable(silence_ui, monkeypatch):
    monkeypatch.setattr(
        tracking.edits, "QInputDialog", FakeInputDialog("Keep 2")
    )
    expected = list(make_track_df().columns)

    for handler, kwargs in (
        (on_new_id_clicked, {}),
        (on_division_clicked, {"t": 4, "tracknumber": 2}),
        (on_remove_division_clicked, {"t": 3, "tracknumber": 1}),
        (on_split_tree, {"t": 3, "tracknumber": 1}),
    ):
        main_window = make_main_window(**kwargs)
        handler(main_window)
        assert list(main_window.track_df.columns) == expected, handler.__name__


def test_every_edit_refreshes_filtered_df(silence_ui, monkeypatch):
    monkeypatch.setattr(
        tracking.edits, "QInputDialog", FakeInputDialog("Keep 2")
    )

    for handler, kwargs in (
        (on_division_clicked, {"t": 4, "tracknumber": 2}),
        (on_remove_division_clicked, {"t": 3, "tracknumber": 1}),
        (on_split_tree, {"t": 3, "tracknumber": 1}),
    ):
        main_window = make_main_window(**kwargs)
        handler(main_window)
        assert len(main_window.filtered_df) == len(
            main_window.track_df
        ), handler.__name__


def test_handlers_abort_without_loaded_data(silence_ui):
    for handler in (
        on_new_id_clicked,
        on_division_clicked,
        on_remove_division_clicked,
        on_split_tree,
    ):
        main_window = make_main_window()
        main_window.folder_list = []
        before = main_window.track_df.copy()
        handler(main_window)
        pd.testing.assert_frame_equal(
            main_window.track_df, before, obj=handler.__name__
        )


def test_write_back_refills_realtime_columns(silence_ui, monkeypatch):
    """Every edit that writes back must re-apply the realtime lookup."""
    seen: list[int] = []

    def spy(main_window, df):
        seen.append(0 if df is None else len(df))
        return df

    monkeypatch.setattr(tracking.history, "_refill_realtime", spy)
    main_window = make_main_window(t=3, tracknumber=1)
    on_split_tree(main_window)

    assert seen, "_refill_realtime was never called during on_split_tree"


def test_new_id_refills_realtime_columns(silence_ui, monkeypatch):
    """on_new_id_clicked refreshes by hand and must refill both frames."""
    seen: list[int] = []
    monkeypatch.setattr(
        tracking.edits,
        "_refill_realtime",
        lambda mw, df: (seen.append(0 if df is None else len(df)), df)[1],
    )
    main_window = make_main_window()
    on_new_id_clicked(main_window)

    assert len(seen) >= 2, "track_df and filtered_df should both be refilled"


def test_zero_numeric_except_protects_realtime_columns():
    """Imported wall-clock values are exempt from measurement blanking.

    Tested directly rather than through a handler: in every current edit path
    the blanked rows are freshly built stubs that were already zero, so the
    exemption is only observable at this level.
    """
    df = make_track_df()
    df["RealTime_min"] = [100.0 + i for i in range(len(df))]
    df["AreaMorphologyM1"] = 55.0

    original = tracking.row_builders.realtime_columns
    tracking.row_builders.realtime_columns = lambda cols: ["RealTime_min"]
    try:
        mask = (df["t"] >= 3).fillna(False).astype(bool)
        tracking.row_builders._zero_numeric_except(df, mask)
    finally:
        tracking.row_builders.realtime_columns = original

    assert df.loc[df["t"] >= 3, "AreaMorphologyM1"].eq(0).all()
    assert not df.loc[df["t"] >= 3, "RealTime_min"].eq(0).any()


def test_edits_reapply_the_realtime_lookup_to_blanked_rows(
    silence_ui, monkeypatch
):
    """After an edit, realtime values are restored from the imported lookup."""
    sentinel = object()
    monkeypatch.setattr(
        tracking.history, "build_realtime_lookup", lambda mw: sentinel
    )

    def fake_apply(df, wide):
        assert wide is sentinel
        out = df.copy()
        out["RealTime_min"] = out["t"] * 5.0 + 1000.0
        return out

    monkeypatch.setattr(tracking.history, "apply_realtime", fake_apply)
    monkeypatch.setattr(
        tracking.edits, "QInputDialog", FakeInputDialog("Keep none")
    )

    main_window = make_main_window(t=3, tracknumber=1)
    on_remove_division_clicked(main_window)

    restored = main_window.track_df
    assert "RealTime_min" in restored.columns
    assert restored["RealTime_min"].eq(restored["t"] * 5.0 + 1000.0).all()


def test_split_tree_keeps_the_blanked_continuation_active(silence_ui):
    """The zeroed remainder of the original tree must stay selectable."""
    main_window = make_main_window(t=3, tracknumber=1)
    on_split_tree(main_window)

    blanked = main_window.track_df[
        (main_window.track_df["Identification"] == IDENT_1)
        & (main_window.track_df["t"] >= 3)
    ]
    assert blanked["AreaMorphologyM1"].eq(0).all(), "rows should be blanked"
    assert blanked["active"].eq(1).all(), "blanked rows must remain active"


def test_remove_division_keep_none_clears_the_active_flag(
    silence_ui, monkeypatch
):
    monkeypatch.setattr(
        tracking.edits, "QInputDialog", FakeInputDialog("Keep none")
    )
    main_window = make_main_window(t=3, tracknumber=1)
    on_remove_division_clicked(main_window)

    blanked = main_window.track_df[
        (main_window.track_df["Identification"] == IDENT_1)
        & (main_window.track_df["t"] >= 3)
    ]
    assert blanked["active"].eq(0).all()


def test_new_id_tolerates_a_track_df_without_a_time_column(silence_ui):
    df = make_track_df().drop(columns=["t"])
    main_window = make_main_window(df=df)
    on_new_id_clicked(main_window)  # must not raise
    assert IDENT_3 in set(main_window.filtered_df["Identification"])


def test_existing_time_tracknumbers_lists_occupied_slots():
    df = make_track_df()
    slots = tracking.context._existing_time_tracknumbers(df, IDENT_1)
    assert (0, 1) in slots
    assert (3, 2) in slots and (3, 3) in slots
    assert (3, 1) not in slots, "parent stops before the division"


def test_existing_time_tracknumbers_ignores_missing_values():
    df = make_track_df()
    df.loc[0, "TrackNumber"] = None
    slots = tracking.context._existing_time_tracknumbers(df, IDENT_1)
    assert (0, 1) not in slots


def test_descendant_index_selects_the_subtree_only():
    df = make_track_df()
    index = tracking.numbering._descendant_index(df, IDENT_1, t_from=3, root=1)
    numbers = set(df.loc[index, "TrackNumber"])
    assert numbers == {2, 3}


def test_descendant_index_honours_keep():
    df = make_track_df()
    index = tracking.numbering._descendant_index(
        df, IDENT_1, t_from=3, root=1, keep=(2,)
    )
    assert set(df.loc[index, "TrackNumber"]) == {3}


def test_track_id_for_ident_reads_the_existing_value():
    assert (
        int(tracking.identity._track_id_for_ident(make_track_df(), IDENT_2))
        == 2
    )


def test_track_id_for_ident_falls_back_to_the_name_suffix():
    df = make_track_df()
    df["track_id"] = float("nan")
    assert tracking.identity._track_id_for_ident(df, IDENT_1) == 1


def test_track_id_for_ident_without_the_column():
    df = make_track_df().drop(columns=["track_id"])
    assert tracking.identity._track_id_for_ident(df, IDENT_1) is None


def test_resolve_edit_context_returns_the_current_selection(silence_ui):
    main_window = make_main_window(t=4, tracknumber=2)
    context = tracking.context._resolve_edit_context(main_window, "Test")
    assert context is not None
    assert context.ident == IDENT_1
    assert context.t == 4
    assert context.tracknumber == 2


def test_resolve_edit_context_infers_a_missing_tracknumber(silence_ui):
    main_window = make_main_window(t=0, tracknumber=None)
    context = tracking.context._resolve_edit_context(main_window, "Test")
    assert context is not None and context.tracknumber == 1


def test_resolve_edit_context_bails_on_an_empty_frame(silence_ui):
    main_window = make_main_window(df=make_track_df().iloc[0:0])
    assert tracking.context._resolve_edit_context(main_window, "Test") is None


def test_division_reuses_existing_daughter_rows(silence_ui):
    """A slot that already exists must not be duplicated by a second division."""
    main_window = make_main_window(t=3, tracknumber=1)
    before = len(main_window.track_df)
    on_division_clicked(main_window)
    assert len(main_window.track_df) == before
    assert_no_duplicate_rows(main_window.track_df)


def test_division_discards_superseded_grandchildren(silence_ui):
    """Rows deeper in the subtree are dropped, the new daughters are not."""
    df = make_track_df()
    deeper = df[df["TrackNumber"] == 2].copy()
    deeper["TrackNumber"] = 4
    df = pd.concat([df, deeper], ignore_index=True)

    main_window = make_main_window(df=df, t=3, tracknumber=1)
    on_division_clicked(main_window)

    numbers = set(
        main_window.track_df.loc[
            main_window.track_df["Identification"] == IDENT_1, "TrackNumber"
        ]
    )
    assert 4 not in numbers
    assert {1, 2, 3} <= numbers


def _with_grandchildren(tracknumbers=(4, 5)) -> pd.DataFrame:
    """make_track_df() plus a further division of daughter 2."""
    df = make_track_df()
    for number in tracknumbers:
        deeper = df[df["TrackNumber"] == 2].copy()
        deeper["TrackNumber"] = number
        deeper["AreaMorphologyM1"] = 900.0 + number
        df = pd.concat([df, deeper], ignore_index=True)
    return df


def make_multigeneration_df() -> pd.DataFrame:
    rows = []
    for tracknumber, t_start, t_end in (
        (1, 0, 2),
        (2, 3, 5),
        (3, 3, 12),
        (4, 6, 8),
        (5, 6, 12),
        (8, 9, 12),
        (9, 9, 12),
    ):
        for t in range(t_start, t_end + 1):
            rows.append(
                {
                    "Position": 2,
                    "t": t,
                    "XMorphology": 1.0,
                    "YMorphology": 2.0,
                    "AreaMorphologyM1": 100.0 * tracknumber + t,
                    "MeanNoBgCorrectedCh00M1": 1.0,
                    "track_id": 1,
                    "TrackNumber": tracknumber,
                    "Cellfate": "Healthy",
                    "Identification": IDENT_1,
                    "active": 1,
                    "inspected": 0,
                    "Calculated_Time": t * 5.0,
                }
            )
    return pd.DataFrame(rows, columns=COLUMNS)


def test_remove_division_reroots_the_whole_kept_subtree(
    silence_ui, monkeypatch
):
    """Removing a division keeps every generation below the kept daughter."""
    monkeypatch.setattr(
        tracking.edits, "QInputDialog", FakeInputDialog("Keep 2")
    )
    monkeypatch.setattr(
        tracking.context, "_current_t_range", lambda mw: (1, 13, 0, 12)
    )
    main_window = make_main_window(
        df=make_multigeneration_df(), t=3, tracknumber=1
    )
    on_remove_division_clicked(main_window)

    layout = spans(main_window.track_df)
    assert layout[(IDENT_1, 1)] == (0, 5, 6)
    assert layout[(IDENT_1, 2)] == (6, 8, 3)
    assert layout[(IDENT_1, 3)] == (6, 12, 7)
    assert layout[(IDENT_1, 4)] == (9, 12, 4)
    assert layout[(IDENT_1, 5)] == (9, 12, 4)
    assert_no_duplicate_rows(main_window.track_df)


def test_remove_division_carries_measurements_through_the_reroot(
    silence_ui, monkeypatch
):
    """Renumbering must move the data with the rows, not just relabel them."""
    monkeypatch.setattr(
        tracking.edits, "QInputDialog", FakeInputDialog("Keep 2")
    )
    monkeypatch.setattr(
        tracking.context, "_current_t_range", lambda mw: (1, 13, 0, 12)
    )
    main_window = make_main_window(
        df=make_multigeneration_df(), t=3, tracknumber=1
    )
    on_remove_division_clicked(main_window)

    # Old TrackNumber 8 became 4; its values were 800 + t.
    rerooted = main_window.track_df[
        main_window.track_df["TrackNumber"] == 4
    ].sort_values("t")
    assert rerooted["AreaMorphologyM1"].tolist() == [
        809.0,
        810.0,
        811.0,
        812.0,
    ]


def test_remove_division_reroots_at_any_generation(silence_ui, monkeypatch):
    """The renumbering is relative to the division being removed, not the root."""
    monkeypatch.setattr(
        tracking.edits, "QInputDialog", FakeInputDialog("Keep 4")
    )
    monkeypatch.setattr(
        tracking.context, "_current_t_range", lambda mw: (1, 13, 0, 12)
    )
    main_window = make_main_window(
        df=make_multigeneration_df(), t=6, tracknumber=2
    )
    on_remove_division_clicked(main_window)

    layout = spans(main_window.track_df)
    # 4 -> 2 joins the parent's own frames; 8 -> 4 and 9 -> 5 follow it.
    assert layout[(IDENT_1, 2)] == (3, 8, 6)
    assert layout[(IDENT_1, 4)] == (9, 12, 4)
    assert layout[(IDENT_1, 5)] == (9, 12, 4)
    # The untouched sibling branch keeps its numbering.
    assert layout[(IDENT_1, 3)] == (3, 12, 10)
    assert_no_duplicate_rows(main_window.track_df)


def test_remove_division_can_keep_the_second_daughter(silence_ui, monkeypatch):
    monkeypatch.setattr(
        tracking.edits, "QInputDialog", FakeInputDialog("Keep 3")
    )
    monkeypatch.setattr(
        tracking.context, "_current_t_range", lambda mw: (1, 13, 0, 12)
    )
    main_window = make_main_window(
        df=make_multigeneration_df(), t=3, tracknumber=1
    )
    on_remove_division_clicked(main_window)

    layout = spans(main_window.track_df)
    assert layout == {(IDENT_1, 1): (0, 12, 13)}
    assert_no_duplicate_rows(main_window.track_df)


def test_remove_division_keep_none_removes_the_whole_subtree(
    silence_ui, monkeypatch
):
    monkeypatch.setattr(
        tracking.edits, "QInputDialog", FakeInputDialog("Keep none")
    )
    main_window = make_main_window(
        df=_with_grandchildren(), t=3, tracknumber=1
    )
    on_remove_division_clicked(main_window)

    survivors = main_window.track_df[
        main_window.track_df["Identification"] == IDENT_1
    ]
    assert set(survivors["TrackNumber"]) == {1}
    assert spans(main_window.track_df)[(IDENT_1, 1)] == (0, 5, 6)
    assert_no_duplicate_rows(main_window.track_df)


def test_remove_division_keep_none_spans_the_daughters_full_extent(
    silence_ui, monkeypatch
):
    """The parent must cover every frame the daughters reached.

    The daughters are usually the rows extending furthest in time, so the
    extent has to be measured before they are dropped.
    """
    monkeypatch.setattr(
        tracking.edits, "QInputDialog", FakeInputDialog("Keep none")
    )
    df = make_track_df()
    # Push the daughters two frames past the parent's own last frame.
    for extra_t in (6, 7):
        for number in (2, 3):
            row = df[(df["TrackNumber"] == number) & (df["t"] == 5)].copy()
            row["t"] = extra_t
            df = pd.concat([df, row], ignore_index=True)

    main_window = make_main_window(df=df, t=3, tracknumber=1)
    on_remove_division_clicked(main_window)

    assert spans(main_window.track_df)[(IDENT_1, 1)] == (0, 7, 8)


def test_ident_t_max_reads_the_last_frame():
    assert (
        tracking.context._ident_t_max(make_track_df(), IDENT_1, default=0) == 5
    )


def test_ident_t_max_falls_back_for_an_unknown_identification():
    assert (
        tracking.context._ident_t_max(make_track_df(), "nope", default=42)
        == 42
    )


def test_forward_scope_mask_selects_one_track_forward():
    df = make_track_df()
    mask = tracking.numbering._forward_scope_mask(
        df, IDENT_1, t_from=1, tracknumber=1
    )
    assert df.loc[mask, "t"].tolist() == [1, 2]


def test_track_id_for_ident_can_refuse_the_name_fallback():
    df = make_track_df()
    df["track_id"] = float("nan")
    assert (
        tracking.identity._track_id_for_ident(
            df, IDENT_1, fallback_to_name=False
        )
        is None
    )


def test_split_tree_reroots_a_deeper_subtree(silence_ui):
    """A grandchild keeps its shape: 2->1 means 4->2 and 5->3."""
    df = make_track_df()
    for number in (4, 5):
        deeper = df[df["TrackNumber"] == 2].copy()
        deeper["TrackNumber"] = number
        df = pd.concat([df, deeper], ignore_index=True)

    main_window = make_main_window(df=df, t=4, tracknumber=2)
    on_split_tree(main_window)

    moved = main_window.track_df[
        main_window.track_df["Identification"] == IDENT_3
    ]
    # 2 becomes the new root, its children 4 and 5 become 2 and 3.
    assert set(moved.loc[moved["t"] >= 4, "TrackNumber"]) == {1, 2, 3}


def test_split_tree_backfill_starts_at_the_lineages_first_frame(silence_ui):
    df = make_track_df()
    df = df[df["t"] >= 2].reset_index(drop=True)
    main_window = make_main_window(df=df, t=4, tracknumber=1)
    on_split_tree(main_window)

    layout = spans(main_window.track_df)
    assert layout[(IDENT_3, 1)][0] == 2, "back-fill must not assume t=0"


def test_split_tree_needs_no_backfill_when_splitting_at_the_start(silence_ui):
    main_window = make_main_window(t=0, tracknumber=1)
    on_split_tree(main_window)
    assert_no_duplicate_rows(main_window.track_df)


def test_subtree_row_index_tolerates_missing_tracknumbers():
    df = make_track_df()
    df.loc[(df["t"] == 4) & (df["TrackNumber"] == 2), "TrackNumber"] = None
    scope = (df["Identification"] == IDENT_1) & (df["t"] >= 3)
    index = tracking.numbering._subtree_row_index(df, scope, 1)
    assert (
        len(index) > 0
    ), "a NaN in scope must not hide the rest of the subtree"


def test_subtree_row_index_falls_back_to_the_root_itself():
    df = make_track_df()
    scope = (df["Identification"] == IDENT_2) & (df["t"] >= 0)
    index = tracking.numbering._subtree_row_index(df, scope, 1)
    assert set(df.loc[index, "TrackNumber"]) == {1}


def test_remap_tracknumbers_preserves_missing_values():
    numbers = pd.Series([2.0, None, 4.0])
    result = tracking.numbering._remap_tracknumbers(numbers, lambda n: n * 10)
    assert result[0] == 20
    assert pd.isna(result[1])
    assert result[2] == 40


def test_new_lineage_identity_uses_the_name_suffix():
    df = make_track_df()
    new_ident, new_track_id = tracking.identity._new_lineage_identity(
        df, IDENT_1
    )
    assert new_ident == IDENT_3
    assert new_track_id == 3


def test_new_lineage_identity_numbers_an_unsuffixed_identification():
    df = make_track_df()
    df["Identification"] = "plainname"
    new_ident, new_track_id = tracking.identity._new_lineage_identity(
        df, "plainname"
    )
    assert new_ident == "plainname-001"
    assert new_track_id == 1, "taken from the suffix, not from max(track_id)"


def test_leave_blank_continuation_can_keep_rows_active():
    df = make_track_df()
    result = tracking.row_builders._leave_blank_continuation(
        df.copy(),
        IDENT_1,
        t_from=1,
        t_max=2,
        tracknumber=1,
        position_number=2,
        time_interval=300.0,
        keep_active=True,
    )
    blanked = result[
        (result["Identification"] == IDENT_1)
        & (result["TrackNumber"] == 1)
        & (result["t"] >= 1)
    ]
    assert blanked["AreaMorphologyM1"].eq(0).all()
    assert blanked["active"].eq(1).all()


def test_fuse_reads_tracknumbers_from_the_dialog(silence_ui):
    main_window = _prepare_fuse(make_main_window(t=3, tracknumber=1))
    request = tracking.fuse._read_fuse_request(
        main_window, main_window.track_df
    )
    assert request is not None
    assert (request.ident1, request.ident2) == (IDENT_1, IDENT_2)
    assert (request.g1, request.g2) == (2, 1)
    assert request.t_fuse == 3


def test_fuse_infers_tracknumbers_left_blank(silence_ui):
    """Empty cell fields fall back to whatever is unambiguous at that time."""
    main_window = _prepare_fuse(
        make_main_window(t=0, tracknumber=1), cell_1="", cell_2=""
    )
    main_window.fuse_time_spin = FakeSpinBox(0)
    request = tracking.fuse._read_fuse_request(
        main_window, main_window.track_df
    )
    assert request is not None and (request.g1, request.g2) == (1, 1)


def test_fuse_rejects_a_repeated_identification(silence_ui):
    main_window = make_main_window(t=3, tracknumber=1)
    main_window.fuse_tree_id_field_1 = FakeLineEdit(IDENT_1)
    main_window.fuse_tree_id_field_2 = FakeLineEdit(IDENT_1)
    assert (
        tracking.fuse._read_fuse_request(main_window, main_window.track_df)
        is None
    )


def test_widget_int_rounds_and_tolerates_rubbish(silence_ui):
    main_window = make_main_window()
    main_window.field = FakeLineEdit("2.6")
    assert tracking.context._widget_int(main_window, "field") == 3
    main_window.field = FakeLineEdit("abc")
    assert tracking.context._widget_int(main_window, "field") is None
    main_window.field = FakeLineEdit("")
    assert tracking.context._widget_int(main_window, "field") is None
    assert tracking.context._widget_int(main_window, "not_there") is None


def test_widget_text_tolerates_a_missing_field(silence_ui):
    main_window = make_main_window()
    assert tracking.context._widget_text(main_window, "not_there") == ""


def test_subtree_members_returns_the_branch_in_order():
    df = make_track_df()
    scope = (df["Identification"] == IDENT_1) & (df["t"] >= 3)
    assert tracking.numbering._subtree_members(df, scope, 1) == [2, 3]
    assert tracking.numbering._subtree_members(df, scope, 2) == [2]
    assert tracking.numbering._subtree_members(df, scope, 7) == []


def test_fuse_grafts_a_multi_branch_subtree(silence_ui):
    """Tree 2's own branches keep their shape after being re-rooted."""
    df = make_track_df()
    for number in (2, 3):
        branch = df[df["Identification"] == IDENT_2].copy()
        branch["TrackNumber"] = number
        df = pd.concat([df, branch], ignore_index=True)

    main_window = _prepare_fuse(make_main_window(df=df, t=3, tracknumber=1))
    on_fuse_trees_clicked(main_window)

    # g2=1 maps onto g1=2, so 1->2, 2->4, 3->5.
    numbers = set(
        main_window.track_df.loc[main_window.track_df["t"] >= 3, "TrackNumber"]
    )
    assert {2, 4, 5} <= numbers


def test_fuse_leaves_other_lineages_alone(silence_ui):
    df = make_track_df()
    third = df[df["Identification"] == IDENT_2].copy()
    third["Identification"] = "250615MA40-p0002-005"
    third["track_id"] = 5
    df = pd.concat([df, third], ignore_index=True)

    main_window = _prepare_fuse(make_main_window(df=df, t=3, tracknumber=1))
    on_fuse_trees_clicked(main_window)

    survivors = main_window.track_df[
        main_window.track_df["Identification"] == "250615MA40-p0002-005"
    ]
    assert len(survivors) == 6
    assert survivors["track_id"].eq(5).all()


def test_fuse_closes_the_dialog_and_resets_the_slot(silence_ui):
    closed = []

    class FakeDialog:
        def close(self):
            closed.append(True)

    main_window = _prepare_fuse(make_main_window(t=3, tracknumber=1))
    main_window.fuse_dialog = FakeDialog()
    on_fuse_trees_clicked(main_window)

    assert closed == [True]
    assert main_window._fuse_next_slot == 1
