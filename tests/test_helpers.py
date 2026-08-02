"""Tests for SECQUOIA time and cell navigation.

These functions read their state off the main window, so the stand-in
carries a six frame image stack, one identification per cell, and a channel
presence list saying which frames were acquired. The folder warning is
replaced while the tests run, because it opens a modal box.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import SECQUOIA.gui.common.messages as messages
from SECQUOIA.utils.helpers import (
    _compute_unique_ids,
    _current_ident_df,
    _infer_n_frames_from_viewers,
    _is_image_layer,
    _set_time_on_all_viewers,
    _set_time_on_viewer,
    _update_current_track_number_plot,
    _x_position_for_row,
    change_cell,
    change_time_point,
    change_time_point_in_jump_channel,
    cycle_highlight_color,
    extract_unique_tracknumbers,
    jump_to_identification,
    set_jump_channel,
    synchronize_viewers_tracking,
)
from SECQUOIA.utils.plotting import TIME_MODE_CALC


# Fakes
class FakeDims:
    """Records the time step a viewer was moved to."""

    def __init__(self, n_dims: int = 3, *, broken: bool = False):
        self.current_step = tuple([0] * n_dims)
        self.broken = broken

    def set_current_step(self, axis: int, value: int) -> None:
        if self.broken:
            raise AttributeError("this viewer has no set_current_step")
        steps = list(self.current_step)
        steps[axis] = value
        self.current_step = tuple(steps)


def fake_viewer(*layers, broken_dims: bool = False):
    return SimpleNamespace(
        dims=FakeDims(broken=broken_dims), layers=list(layers)
    )


def image_layer(shape=(6, 10, 10)):
    return SimpleNamespace(_type_string="Image", data=np.zeros(shape))


def labels_layer(shape=(6, 10, 10)):
    return SimpleNamespace(
        _type_string="Labels", data=np.zeros(shape), selected_label=1
    )


@pytest.fixture
def folder_warning(monkeypatch):
    """Replace the modal folder warning with a recorder.

    Without this the real ``QMessageBox.exec_()`` blocks until the per-test
    timeout fires.
    """
    calls = []
    monkeypatch.setattr(
        messages, "show_folder_warning", lambda *a, **k: calls.append(a)
    )
    return calls


# Layer inspection
class TestIsImageLayer:
    def test_recognises_an_image_layer_by_its_type_string(self):
        assert _is_image_layer(image_layer()) is True

    def test_rejects_a_labels_layer(self):
        assert _is_image_layer(labels_layer()) is False

    def test_falls_back_to_duck_typing_without_a_type_string(self):
        """A Labels layer is the one with a selected_label."""
        assert _is_image_layer(SimpleNamespace(data=np.zeros((2, 2)))) is True
        assert (
            _is_image_layer(
                SimpleNamespace(data=np.zeros((2, 2)), selected_label=1)
            )
            is False
        )


class TestInferNFramesFromViewers:
    def test_reads_the_frame_count_off_an_image_layer(self, fake_main_window):
        main_window = fake_main_window(viewer_1=fake_viewer(image_layer()))

        assert _infer_n_frames_from_viewers(main_window) == 6

    def test_a_two_dimensional_layer_is_a_single_frame(self, fake_main_window):
        main_window = fake_main_window(
            viewer_1=fake_viewer(image_layer(shape=(10, 10)))
        )

        assert _infer_n_frames_from_viewers(main_window) == 1

    def test_ignores_labels_layers(self, fake_main_window):
        main_window = fake_main_window(
            viewer_1=fake_viewer(labels_layer(shape=(99, 10, 10))),
            viewer_2=fake_viewer(image_layer(shape=(6, 10, 10))),
        )

        assert _infer_n_frames_from_viewers(main_window) == 6

    def test_falls_back_to_the_stored_stacks(self, fake_main_window):
        main_window = fake_main_window(
            n_channels=1, images=[np.zeros((4, 10, 10))]
        )

        assert _infer_n_frames_from_viewers(main_window) == 4

    def test_returns_zero_when_nothing_is_loaded(self, fake_main_window):
        main_window = fake_main_window(n_channels=0, images=None)

        assert _infer_n_frames_from_viewers(main_window) == 0


class TestSetTimeOnViewer:
    def test_sets_the_step_on_the_first_axis(self):
        viewer = fake_viewer()

        _set_time_on_viewer(viewer, 3)

        assert viewer.dims.current_step == (3, 0, 0)

    def test_falls_back_to_assigning_the_whole_tuple(self):
        """Older viewers have no set_current_step."""
        viewer = fake_viewer(broken_dims=True)

        _set_time_on_viewer(viewer, 2)

        assert viewer.dims.current_step == (2, 0, 0)

    def test_a_viewer_that_cannot_be_moved_is_not_an_error(self):
        _set_time_on_viewer(SimpleNamespace(), 1)

    def test_moves_both_viewers(self, fake_main_window):
        main_window = fake_main_window(
            viewer_1=fake_viewer(), viewer_2=fake_viewer()
        )

        _set_time_on_all_viewers(main_window, 4)

        assert main_window.viewer_1.dims.current_step[0] == 4
        assert main_window.viewer_2.dims.current_step[0] == 4

    def test_a_missing_second_viewer_is_skipped(self, fake_main_window):
        main_window = fake_main_window(viewer_1=fake_viewer(), viewer_2=None)

        _set_time_on_all_viewers(main_window, 4)

        assert main_window.viewer_1.dims.current_step[0] == 4


# Time navigation
def navigable_window(fake_main_window, *, current=0, n_frames=6):
    """A window with a folder loaded and one six-frame image stack."""
    return fake_main_window(
        folder_list=["/experiment"],
        viewer_1=fake_viewer(image_layer(shape=(n_frames, 10, 10))),
        current_time_index=current,
        df_subset=None,
    )


class TestChangeTimePoint:
    def test_steps_forward(self, fake_main_window):
        main_window = navigable_window(fake_main_window, current=2)

        change_time_point(main_window, 1)

        assert main_window.current_time_index == 3

    def test_steps_backward(self, fake_main_window):
        main_window = navigable_window(fake_main_window, current=2)

        change_time_point(main_window, -1)

        assert main_window.current_time_index == 1

    def test_wraps_past_the_last_frame(self, fake_main_window):
        main_window = navigable_window(fake_main_window, current=5)

        change_time_point(main_window, 1)

        assert main_window.current_time_index == 0

    def test_wraps_before_the_first_frame(self, fake_main_window):
        main_window = navigable_window(fake_main_window, current=0)

        change_time_point(main_window, -1)

        assert main_window.current_time_index == 5

    def test_moves_the_viewer_too(self, fake_main_window):
        main_window = navigable_window(fake_main_window, current=1)

        change_time_point(main_window, 1)

        assert main_window.viewer_1.dims.current_step[0] == 2

    def test_a_single_frame_stack_stays_at_zero(self, fake_main_window):
        main_window = navigable_window(fake_main_window, current=0, n_frames=1)

        change_time_point(main_window, 1)

        assert main_window.current_time_index == 0

    def test_warns_and_stops_without_a_loaded_folder(
        self, fake_main_window, folder_warning
    ):
        main_window = fake_main_window(folder_list=[], current_time_index=3)

        change_time_point(main_window, 1)

        assert len(folder_warning) == 1
        assert main_window.current_time_index == 3

    def test_a_reentrant_call_is_ignored(self, fake_main_window):
        """The time-change callback must not recurse into itself."""
        main_window = navigable_window(fake_main_window, current=2)
        main_window._in_time_change_cb = True

        change_time_point(main_window, 1)

        assert main_window.current_time_index == 2


class TestJumpChannelNavigation:
    def test_selects_an_available_channel(self, fake_main_window):
        main_window = fake_main_window(
            image_present={"w01": [1, 0, 1]}, jump_channel=None
        )

        set_jump_channel(main_window, "w01")

        assert main_window.jump_channel == "w01"

    def test_refuses_an_unavailable_channel(self, fake_main_window):
        main_window = fake_main_window(
            image_present={"w01": [1]}, jump_channel="w01"
        )

        set_jump_channel(main_window, "w99")

        assert main_window.jump_channel == "w01"

    def test_jumps_to_the_next_acquired_frame(self, fake_main_window):
        main_window = navigable_window(fake_main_window, current=0)
        main_window.jump_channel = "w01"
        main_window.image_present = {"w01": [1, 0, 0, 1, 0, 1]}

        change_time_point_in_jump_channel(main_window, 1)

        assert main_window.current_time_index == 3

    def test_jumps_to_the_previous_acquired_frame(self, fake_main_window):
        main_window = navigable_window(fake_main_window, current=5)
        main_window.jump_channel = "w01"
        main_window.image_present = {"w01": [1, 0, 0, 1, 0, 1]}

        change_time_point_in_jump_channel(main_window, -1)

        assert main_window.current_time_index == 3

    def test_wraps_forward_to_the_first_acquired_frame(self, fake_main_window):
        main_window = navigable_window(fake_main_window, current=5)
        main_window.jump_channel = "w01"
        main_window.image_present = {"w01": [1, 0, 0, 1, 0, 0]}

        change_time_point_in_jump_channel(main_window, 1)

        assert main_window.current_time_index == 0

    def test_wraps_backward_to_the_last_acquired_frame(self, fake_main_window):
        main_window = navigable_window(fake_main_window, current=0)
        main_window.jump_channel = "w01"
        main_window.image_present = {"w01": [0, 0, 1, 0, 0, 1]}

        change_time_point_in_jump_channel(main_window, -1)

        assert main_window.current_time_index == 5

    def test_does_nothing_without_a_selected_channel(self, fake_main_window):
        main_window = navigable_window(fake_main_window, current=2)
        main_window.jump_channel = None

        change_time_point_in_jump_channel(main_window, 1)

        assert main_window.current_time_index == 2

    def test_does_nothing_without_a_presence_map(self, fake_main_window):
        main_window = navigable_window(fake_main_window, current=2)
        main_window.jump_channel = "w01"
        main_window.image_present = None

        change_time_point_in_jump_channel(main_window, 1)

        assert main_window.current_time_index == 2

    def test_does_nothing_when_the_channel_has_no_images(
        self, fake_main_window
    ):
        main_window = navigable_window(fake_main_window, current=2)
        main_window.jump_channel = "w01"
        main_window.image_present = {"w01": [0, 0, 0, 0, 0, 0]}

        change_time_point_in_jump_channel(main_window, 1)

        assert main_window.current_time_index == 2


class TestSynchronizeViewersTracking:
    def test_pushes_the_current_index_to_the_viewers(self, fake_main_window):
        main_window = fake_main_window(
            current_time_index=3,
            viewer_1=fake_viewer(),
            viewer_2=fake_viewer(),
        )

        synchronize_viewers_tracking(main_window)

        assert main_window.viewer_1.dims.current_step[0] == 3
        assert main_window.viewer_2.dims.current_step[0] == 3


# Track number bookkeeping


def subset_frame() -> pd.DataFrame:
    """One identification whose track divides at t=3."""
    return pd.DataFrame(
        {
            "Identification": ["p1_id1"] * 7,
            "TrackNumber": [1, 1, 1, 2, 3, 2, 3],
            "t": [0, 1, 2, 3, 3, 4, 4],
        }
    )


class TestExtractUniqueTracknumbers:
    def test_maps_each_identification_to_its_track_numbers(
        self, fake_main_window
    ):
        main_window = fake_main_window(filtered_df=subset_frame())

        assert extract_unique_tracknumbers(main_window) == {
            "p1_id1": [1, 2, 3]
        }

    def test_separates_identifications(self, fake_main_window):
        df = subset_frame()
        df.loc[df["t"] >= 3, "Identification"] = "p1_id2"
        main_window = fake_main_window(filtered_df=df)

        result = extract_unique_tracknumbers(main_window)

        assert result == {"p1_id1": [1], "p1_id2": [2, 3]}

    def test_drops_missing_track_numbers(self, fake_main_window):
        df = subset_frame()
        df["TrackNumber"] = df["TrackNumber"].astype(float)
        df.loc[0, "TrackNumber"] = np.nan
        main_window = fake_main_window(filtered_df=df)

        assert extract_unique_tracknumbers(main_window) == {
            "p1_id1": [1, 2, 3]
        }

    def test_returns_empty_without_the_required_columns(
        self, fake_main_window
    ):
        main_window = fake_main_window(filtered_df=pd.DataFrame({"t": [0]}))

        assert extract_unique_tracknumbers(main_window) == {}

    def test_returns_empty_without_a_frame(self, fake_main_window):
        assert extract_unique_tracknumbers(fake_main_window()) == {}


class TestUpdateCurrentTrackNumberPlot:
    def test_keeps_a_track_that_is_still_present(self, fake_main_window):
        main_window = fake_main_window(
            df_subset=subset_frame(),
            current_time_index=4,
            current_TrackNumber_plot=3,
        )

        _update_current_track_number_plot(main_window)

        assert main_window.current_TrackNumber_plot == 3

    def test_switches_to_the_first_track_at_this_time(self, fake_main_window):
        """Track 1 has ended by t=4, so the selection has to move."""
        main_window = fake_main_window(
            df_subset=subset_frame(),
            current_time_index=4,
            current_TrackNumber_plot=1,
        )

        _update_current_track_number_plot(main_window)

        assert main_window.current_TrackNumber_plot == 2

    def test_clears_the_selection_when_no_row_exists(self, fake_main_window):
        main_window = fake_main_window(
            df_subset=subset_frame(),
            current_time_index=99,
            current_TrackNumber_plot=1,
        )

        _update_current_track_number_plot(main_window)

        assert main_window.current_TrackNumber_plot is None

    def test_clears_the_selection_without_a_subset(self, fake_main_window):
        main_window = fake_main_window(
            df_subset=None, current_TrackNumber_plot=1
        )

        _update_current_track_number_plot(main_window)

        assert main_window.current_TrackNumber_plot is None

    def test_clears_the_selection_for_an_empty_subset(self, fake_main_window):
        main_window = fake_main_window(
            df_subset=subset_frame().iloc[0:0], current_TrackNumber_plot=1
        )

        _update_current_track_number_plot(main_window)

        assert main_window.current_TrackNumber_plot is None


# Cell navigation


def multi_ident_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Identification": ["a", "a", "b", "b", "c"],
            "TrackNumber": [1, 1, 1, 1, 1],
            "t": [0, 1, 0, 1, 0],
        }
    )


class TestComputeUniqueIds:
    def test_lists_identifications_in_order_of_appearance(
        self, fake_main_window
    ):
        main_window = fake_main_window(filtered_df=multi_ident_frame())

        assert _compute_unique_ids(main_window) == ["a", "b", "c"]

    def test_skips_missing_identifications(self, fake_main_window):
        df = multi_ident_frame()
        df.loc[0, "Identification"] = np.nan
        main_window = fake_main_window(filtered_df=df)

        assert _compute_unique_ids(main_window) == ["a", "b", "c"]


class TestChangeCellGuards:
    """The paths that return before any plot or viewer work happens."""

    def test_warns_without_a_loaded_folder(
        self, fake_main_window, folder_warning
    ):
        main_window = fake_main_window(folder_list=[])

        change_cell(main_window, "next")

        assert len(folder_warning) == 1

    def test_does_nothing_without_a_filtered_frame(self, fake_main_window):
        main_window = fake_main_window(
            folder_list=["/e"], filtered_df=None, ident="a"
        )

        change_cell(main_window, "next")

        assert main_window.ident == "a"

    def test_does_nothing_without_the_identification_column(
        self, fake_main_window
    ):
        main_window = fake_main_window(
            folder_list=["/e"], filtered_df=pd.DataFrame({"t": [0]}), ident="a"
        )

        change_cell(main_window, "next")

        assert main_window.ident == "a"

    def test_rejects_an_unknown_direction(self, fake_main_window):
        main_window = fake_main_window(
            folder_list=["/e"], filtered_df=multi_ident_frame(), ident="a"
        )

        with pytest.raises(ValueError, match="invalid direction"):
            change_cell(main_window, "sideways")

    def test_next_at_the_last_cell_does_nothing(self, fake_main_window):
        """The boundary returns before touching the viewer or the plots."""
        main_window = fake_main_window(
            folder_list=["/e"],
            filtered_df=multi_ident_frame(),
            ident="c",
            current_ident_index=2,
        )

        change_cell(main_window, "next")

        assert main_window.ident == "c"
        assert main_window.current_ident_index == 2

    def test_previous_at_the_first_cell_does_nothing(self, fake_main_window):
        main_window = fake_main_window(
            folder_list=["/e"],
            filtered_df=multi_ident_frame(),
            ident="a",
            current_ident_index=0,
        )

        change_cell(main_window, "previous")

        assert main_window.ident == "a"
        assert main_window.current_ident_index == 0

    def test_an_empty_frame_leaves_the_selection_alone(self, fake_main_window):
        main_window = fake_main_window(
            folder_list=["/e"],
            filtered_df=multi_ident_frame().iloc[0:0],
            ident="a",
        )

        change_cell(main_window, "next")

        assert main_window.ident == "a"


class TestJumpToIdentification:
    """The GUI refresh is suppressed, so the computed state is observable."""

    def base(self, fake_main_window, **extra):
        return fake_main_window(
            filtered_df=multi_ident_frame(),
            unique_ids=["a", "b", "c"],
            current_ident_index=0,
            ident="a",
            current_time_index=0,
            current_TrackNumber_plot=1,
            **extra,
        )

    def test_selects_the_requested_identification(self, fake_main_window):
        main_window = self.base(fake_main_window)

        jump_to_identification(main_window, "b")

        assert main_window.ident == "b"
        assert main_window.current_ident_index == 1

    def test_jumps_to_the_first_frame_of_that_cell_by_default(
        self, fake_main_window
    ):
        df = multi_ident_frame()
        df.loc[df["Identification"] == "b", "t"] = [7, 8]
        main_window = self.base(fake_main_window)
        main_window.filtered_df = df

        jump_to_identification(main_window, "b")

        assert main_window.current_time_index == 7

    def test_an_explicit_time_wins(self, fake_main_window):
        main_window = self.base(fake_main_window)

        jump_to_identification(main_window, "b", t=5)

        assert main_window.current_time_index == 5

    def test_an_explicit_track_number_is_selected(self, fake_main_window):
        main_window = self.base(fake_main_window)

        jump_to_identification(main_window, "b", tracknumber=3)

        assert main_window.current_TrackNumber_plot == 3

    def test_the_track_number_defaults_to_one(self, fake_main_window):
        main_window = self.base(fake_main_window)

        jump_to_identification(main_window, "b", tracknumber=0)

        assert main_window.current_TrackNumber_plot == 1

    def test_recomputes_the_id_list_when_the_target_is_missing(
        self, fake_main_window
    ):
        """A stale cache must not make a valid identification unreachable."""
        main_window = self.base(fake_main_window)
        main_window.unique_ids = ["a"]  # stale

        jump_to_identification(main_window, "c")

        assert main_window.ident == "c"
        assert main_window.unique_ids == ["a", "b", "c"]

    def test_an_unknown_identification_changes_nothing(self, fake_main_window):
        main_window = self.base(fake_main_window)

        jump_to_identification(main_window, "ghost")

        assert main_window.ident == "a"
        assert main_window.current_ident_index == 0

    def test_an_empty_identification_changes_nothing(self, fake_main_window):
        main_window = self.base(fake_main_window)

        jump_to_identification(main_window, "")

        assert main_window.ident == "a"

    def test_a_window_without_a_filtered_frame_changes_nothing(
        self, fake_main_window
    ):
        main_window = fake_main_window(ident="a")

        jump_to_identification(main_window, "b")

        assert main_window.ident == "a"


class TestCurrentIdentDf:
    def test_returns_the_selected_identification_and_its_rows(
        self, fake_main_window
    ):
        main_window = fake_main_window(
            filtered_df=multi_ident_frame(),
            unique_ids=["a", "b", "c"],
            current_ident_index=1,
        )

        ident, df_id = _current_ident_df(main_window)

        assert ident == "b"
        assert df_id["t"].tolist() == [0, 1]

    def test_falls_back_to_the_first_row_for_an_out_of_range_index(
        self, fake_main_window
    ):
        main_window = fake_main_window(
            filtered_df=multi_ident_frame(),
            unique_ids=["a", "b", "c"],
            current_ident_index=99,
        )

        ident, _ = _current_ident_df(main_window)

        assert ident == "a"

    def test_returns_none_without_any_identifications(self, fake_main_window):
        main_window = fake_main_window(
            filtered_df=multi_ident_frame(), unique_ids=[]
        )

        assert _current_ident_df(main_window) is None

    def test_returns_none_when_the_selection_matches_no_rows(
        self, fake_main_window
    ):
        main_window = fake_main_window(
            filtered_df=multi_ident_frame(),
            unique_ids=["ghost"],
            current_ident_index=0,
        )

        assert _current_ident_df(main_window) is None


class TestXPositionForRow:
    FEATURE_DEFS = {"Area": {"has_ch": False}}

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"t": [0, 1, 2, 3], "Calculated_Time": [0.0, 3.0, 6.0, 9.0]}
        )

    def window(self, fake_main_window):
        return fake_main_window(
            selected_feature_by_row={0: "Area"},
            selected_ch_by_channel={0: 1},
        )

    def test_reads_the_x_value_at_the_current_time(self, fake_main_window):
        x = _x_position_for_row(
            self.window(fake_main_window),
            self.frame(),
            0,
            2,
            TIME_MODE_CALC,
            self.FEATURE_DEFS,
        )

        assert x == 6.0

    def test_falls_back_to_the_nearest_time(self, fake_main_window):
        """A frame the cell was not measured in still needs a marker."""
        df = self.frame()
        df = df[df["t"] != 2]

        x = _x_position_for_row(
            self.window(fake_main_window),
            df,
            0,
            2,
            TIME_MODE_CALC,
            self.FEATURE_DEFS,
        )

        assert x in (3.0, 9.0)

    def test_falls_back_to_the_frame_number_with_no_usable_column(
        self, fake_main_window
    ):
        df = pd.DataFrame({"t": [0, 1, 2]})

        x = _x_position_for_row(
            self.window(fake_main_window),
            df,
            0,
            2,
            TIME_MODE_CALC,
            self.FEATURE_DEFS,
        )

        assert x == 2.0


# Highlight colours


class TestCycleHighlightColor:
    def test_moves_forward_through_the_palette(self, fake_main_window):
        main_window = fake_main_window(
            _hl_palette={"a": "#f00", "b": "#0f0", "c": "#00f"},
            _hl_palette_index=0,
        )

        cycle_highlight_color(main_window, forward=True)

        assert main_window._hl_palette_index == 1
        assert main_window._hl_active_color == "#0f0"

    def test_moves_backward_through_the_palette(self, fake_main_window):
        main_window = fake_main_window(
            _hl_palette={"a": "#f00", "b": "#0f0", "c": "#00f"},
            _hl_palette_index=1,
        )

        cycle_highlight_color(main_window, forward=False)

        assert main_window._hl_active_color == "#f00"

    def test_wraps_at_the_end_of_the_palette(self, fake_main_window):
        main_window = fake_main_window(
            _hl_palette={"a": "#f00", "b": "#0f0"}, _hl_palette_index=1
        )

        cycle_highlight_color(main_window, forward=True)

        assert main_window._hl_palette_index == 0

    def test_wraps_at_the_start_of_the_palette(self, fake_main_window):
        main_window = fake_main_window(
            _hl_palette={"a": "#f00", "b": "#0f0"}, _hl_palette_index=0
        )

        cycle_highlight_color(main_window, forward=False)

        assert main_window._hl_palette_index == 1

    def test_caches_the_palette_order(self, fake_main_window):
        main_window = fake_main_window(
            _hl_palette={"a": "#f00", "b": "#0f0"}, _hl_palette_index=0
        )

        cycle_highlight_color(main_window)

        assert main_window._hl_palette_order == ["#f00", "#0f0"]

    def test_an_empty_palette_is_not_an_error(self, fake_main_window):
        main_window = fake_main_window(_hl_palette={}, _hl_palette_index=0)

        cycle_highlight_color(main_window)

        assert main_window._hl_palette_index == 0

    def test_calls_the_button_style_hook(self, fake_main_window):
        calls = []
        main_window = fake_main_window(
            _hl_palette={"a": "#f00", "b": "#0f0"},
            _hl_palette_index=0,
            _hl_sync_button_style=lambda: calls.append(1),
        )

        cycle_highlight_color(main_window)

        assert len(calls) == 1
