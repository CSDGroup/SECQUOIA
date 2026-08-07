"""Tests for saving and reloading a SECQUOIA project.

A project is saved from one main window and loaded into a fresh one, and the
two are compared. The test experiment has two positions on disk, two
channels, a four row track dataframe and one derived feature, so paths,
formats and calculated features are all covered by the round trip.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from SECQUOIA.core.project_state import (
    _measurements_output_path,
    load_project_state,
    save_position_measurements,
    save_project_state,
    save_track_df,
)

pytestmark = pytest.mark.gui


def track_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Position": [1, 1, 2, 2],
            "Identification": ["p1-001", "p1-001", "p2-001", "p2-001"],
            "TrackNumber": [1, 1, 1, 1],
            "t": [0, 1, 0, 1],
            "AreaMorphologyM1": [10.0, 11.0, 20.0, 21.0],
        }
    )


@pytest.fixture
def experiment(tmp_path):
    """A minimal on disk experiment with two position folders."""
    root = tmp_path / "250615MA40"
    for position in (1, 2):
        (root / f"250615MA40_p{position:04d}").mkdir(parents=True)
    return root


@pytest.fixture
def saved_window(fake_main_window, experiment, qapp, silence_modals):
    """A window whose project has been saved to disk."""
    main_window = fake_main_window(
        folder=str(experiment),
        folder_list=os.listdir(experiment),
        experiment_name="250615MA40",
        project_name="Project_1",
        tracking_format="tTt",
        image_format="png",
        loading_format="standard",
        cp_tracking=False,
        tracking_path=str(experiment / "tracking.csv"),
        segmentation_paths=[str(experiment / "Analysis" / "Segmentation1")],
        background_correction_path=None,
        import_rt_path=None,
        time_min_selected=1,
        time_max_selected=6,
        dt_seconds=180,
        position_min_selected=1,
        position_max_selected=2,
        position_start_selected=1,
        threshold=5,
        min_mask_size=10,
        available_channels=["w00", "w01"],
        ids_channels=["w00", "w01"],
        jump_channel="w01",
        n_channels=2,
        n_masks=1,
        user="tester",
        track_df=track_frame(),
        position_folders=[
            str(experiment / "250615MA40_p0001"),
            str(experiment / "250615MA40_p0002"),
        ],
        current_position_index=0,
        _derived_features={"Ratio": {"template": "RatioM{m}"}},
        last_run_config={1: {"mask": 1}},
        selected_feature_by_row={0: "AreaMorphology"},
    )
    save_track_df(main_window)
    save_project_state(main_window)
    return main_window


# Saving
class TestSaveTrackDf:
    def test_writes_the_full_csv(self, saved_window):
        full = os.path.join(
            os.path.dirname(saved_window.project_state_path),
            "Updated_CSV_File.csv",
        )

        assert os.path.isfile(full)
        assert len(pd.read_csv(full)) == 4

    def test_writes_a_per_position_csv(self, saved_window):
        folder = os.path.dirname(saved_window.project_state_path)
        matches = [
            f for f in os.listdir(folder) if f.startswith("SECQUOIA_p0001")
        ]

        assert len(matches) == 1
        assert len(pd.read_csv(os.path.join(folder, matches[0]))) == 2

    def test_the_per_position_name_records_the_layout(self, saved_window):
        folder = os.path.dirname(saved_window.project_state_path)
        name = next(
            f for f in os.listdir(folder) if f.startswith("SECQUOIA_p0001")
        )

        assert "_t00001-00006_" in name
        assert "_m1_ch2" in name

    def test_warns_and_stops_without_a_loaded_folder(
        self, fake_main_window, silence_modals
    ):
        """A root that does not exist is dropped without a modal warning."""
        main_window = fake_main_window(folder_list=[])

        save_track_df(main_window)

        assert shown_once(silence_modals)

    def test_stops_on_an_invalid_experiment_root(
        self, fake_main_window, silence_modals
    ):
        main_window = fake_main_window(
            folder_list=["something"], folder="/does/not/exist"
        )

        save_track_df(main_window)

        assert not silence_modals


def shown_once(shown) -> bool:
    return shown == ["QMessageBox"]


class TestSavePositionMeasurements:
    def test_writes_the_rows_of_one_position(self, saved_window):
        assert save_position_measurements(saved_window, 2, n_masks=1) is True

        path = _measurements_output_path(saved_window, 2, 1)
        assert pd.read_csv(path)["Position"].unique().tolist() == [2]

    def test_reports_false_for_a_position_with_no_rows(self, saved_window):
        assert save_position_measurements(saved_window, 99, n_masks=1) is False

    def test_reports_false_on_a_broken_window(self, fake_main_window):
        main_window = fake_main_window(track_df=None)

        assert save_position_measurements(main_window, 1, n_masks=1) is False


class TestSaveProjectState:
    def test_writes_the_metadata_next_to_the_csv(self, saved_window):
        assert saved_window.project_state_path.endswith(
            "project_metadata.json"
        )
        assert os.path.isfile(saved_window.project_state_path)

    def test_the_metadata_records_the_experiment(self, saved_window):
        with open(saved_window.project_state_path) as f:
            state = json.load(f)

        assert state["experiment_name"] == "250615MA40"
        assert state["tracking_format"] == "tTt"
        assert state["n_channels"] == 2

    def test_the_metadata_records_the_derived_features(self, saved_window):
        with open(saved_window.project_state_path) as f:
            state = json.load(f)

        assert state["metric_calculations"]["derived_features"] == {
            "Ratio": {"template": "RatioM{m}"}
        }

    def test_the_metadata_points_at_the_track_csv(self, saved_window):
        with open(saved_window.project_state_path) as f:
            state = json.load(f)

        assert os.path.isfile(state["track_df_csv_path"])

    def test_does_nothing_without_an_experiment_root(self, fake_main_window):
        main_window = fake_main_window(folder=None)

        save_project_state(main_window)

        assert not hasattr(main_window, "project_state_path")


# Loading
class TestLoadProjectState:
    def loaded(self, fake_main_window, saved_window):
        target = fake_main_window()
        assert (
            load_project_state(target, saved_window.project_state_path) is True
        )
        return target

    def test_restores_the_experiment_root_under_its_own_name(
        self, fake_main_window, saved_window
    ):
        """The JSON key is experiment_root; the attribute is folder."""
        target = self.loaded(fake_main_window, saved_window)

        assert target.folder == saved_window.folder

    def test_restores_the_formats(self, fake_main_window, saved_window):
        target = self.loaded(fake_main_window, saved_window)

        assert target.tracking_format == "tTt"
        assert target.image_format == "png"

    def test_restores_the_time_range(self, fake_main_window, saved_window):
        target = self.loaded(fake_main_window, saved_window)

        assert target.time_min_selected == 1
        assert target.time_max_selected == 6
        assert target.dt_seconds == 180

    def test_restores_the_channels(self, fake_main_window, saved_window):
        target = self.loaded(fake_main_window, saved_window)

        assert target.ids_channels == ["w00", "w01"]
        assert target.n_channels == 2
        assert target.jump_channel == "w01"

    def test_rebuilds_the_channel_inputs_as_widgets(
        self, fake_main_window, saved_window
    ):
        target = self.loaded(fake_main_window, saved_window)

        assert [w.text() for w in target.FL_inputs] == ["w00", "w01"]
        assert all(w.isReadOnly() for w in target.FL_inputs)

    def test_reloads_the_track_dataframe(self, fake_main_window, saved_window):
        target = self.loaded(fake_main_window, saved_window)

        assert len(target.track_df) == 4
        assert target.track_df["Position"].tolist() == [1, 1, 2, 2]

    def test_restores_the_derived_features(
        self, fake_main_window, saved_window
    ):
        target = self.loaded(fake_main_window, saved_window)

        assert target._derived_features == {"Ratio": {"template": "RatioM{m}"}}

    def test_makes_derived_features_visible_to_the_plots(
        self, fake_main_window, saved_window
    ):
        target = self.loaded(fake_main_window, saved_window)

        assert "Ratio" in target._feature_defs

    def test_restores_integer_keyed_state_as_integers(
        self, fake_main_window, saved_window
    ):
        """JSON turns dict keys into strings; they have to come back as ints."""
        target = self.loaded(fake_main_window, saved_window)

        assert target.last_run_config == {1: {"mask": 1}}
        assert target.selected_feature_by_row == {0: "AreaMorphology"}

    def test_rediscovers_the_position_folders(
        self, fake_main_window, saved_window
    ):
        target = self.loaded(fake_main_window, saved_window)

        assert len(target.position_folders) == 2
        assert target.position_min == 1
        assert target.position_max == 2

    def test_restores_the_clt_parser_for_the_ttt_format(
        self, fake_main_window, saved_window
    ):
        target = self.loaded(fake_main_window, saved_window)

        assert target.clt_parser is not None
        assert target.clt_parser.folder_exp == saved_window.tracking_path

    def test_records_where_it_was_loaded_from(
        self, fake_main_window, saved_window
    ):
        target = self.loaded(fake_main_window, saved_window)

        assert target.project_state_path == saved_window.project_state_path

    def test_refuses_a_missing_metadata_file(self, fake_main_window, tmp_path):
        main_window = fake_main_window()

        assert (
            load_project_state(main_window, str(tmp_path / "nope.json"))
            is False
        )

    def test_refuses_a_project_whose_csv_has_gone(
        self, fake_main_window, saved_window, silence_modals
    ):
        """Losing the CSV must be reported, not silently loaded as empty."""
        with open(saved_window.project_state_path) as f:
            state = json.load(f)
        os.remove(state["track_df_csv_path"])

        result = load_project_state(
            fake_main_window(), saved_window.project_state_path
        )

        assert result is False
        assert "critical" in silence_modals

    def test_a_round_trip_preserves_the_measurements(
        self, fake_main_window, saved_window
    ):
        target = self.loaded(fake_main_window, saved_window)

        pd.testing.assert_frame_equal(
            target.track_df.reset_index(drop=True),
            track_frame().reset_index(drop=True),
        )
