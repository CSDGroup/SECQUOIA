"""Tests for surviving widgets whose C++ object has been deleted."""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest
from qtpy.QtCore import QEvent
from qtpy.QtWidgets import QApplication, QComboBox, QWidget

from SECQUOIA.core.project_state import save_project_state
from SECQUOIA.gui.common.ui_utils import is_widget_alive, reuse_widget

pytestmark = pytest.mark.gui


def delete_now(widget: QWidget) -> None:
    """Destroy `widget` and its children, without spinning an event loop."""
    widget.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)


class DeletedCombo:
    """Stands in for a combo whose C++ object is gone."""

    def currentText(self):
        raise RuntimeError(
            "wrapped C/C++ object of type QComboBox has been deleted"
        )

    def objectName(self):
        raise RuntimeError(
            "wrapped C/C++ object of type QComboBox has been deleted"
        )


def test_a_live_widget_is_reported_alive(qtbot):
    combo = QComboBox()
    qtbot.addWidget(combo)

    assert is_widget_alive(combo) is True


def test_a_deleted_child_is_reported_dead(qtbot):
    parent = QWidget()
    qtbot.addWidget(parent)
    combo = QComboBox(parent)

    delete_now(parent)

    assert is_widget_alive(combo) is False


def test_none_is_reported_dead():
    assert is_widget_alive(None) is False


def test_a_live_widget_is_reused(qtbot, fake_main_window):
    combo = QComboBox()
    qtbot.addWidget(combo)
    main_window = fake_main_window(image_format_combo=combo)

    assert reuse_widget(main_window, "image_format_combo", QComboBox) is combo


def test_a_deleted_widget_is_rebuilt(qtbot, fake_main_window):
    """Reopening the loading window after a run all must not touch the corpse."""
    parent = QWidget()
    qtbot.addWidget(parent)
    combo = QComboBox(parent)
    combo.addItems(["png", "tif"])
    main_window = fake_main_window(image_format_combo=combo)
    delete_now(parent)

    rebuilt = reuse_widget(main_window, "image_format_combo", QComboBox)

    assert rebuilt is not combo
    assert main_window.image_format_combo is rebuilt
    # The caller repopulates an empty combo, which is what makes this safe.
    assert rebuilt.count() == 0


def test_a_missing_widget_is_built(fake_main_window):
    main_window = fake_main_window()

    built = reuse_widget(main_window, "image_format_combo", QComboBox)

    assert isinstance(built, QComboBox)


@pytest.fixture
def window_with_dead_combos(fake_main_window, tmp_path):
    """A saved session whose loading_window combos have been destroyed."""
    root = tmp_path / "exp"
    root.mkdir()
    return fake_main_window(
        folder=str(root),
        folder_list=["exp_p0001"],
        experiment_name="exp",
        project_name="Project_1",
        tracking_format="tTt",
        image_format="png",
        image_format_combo=DeletedCombo(),
        tracking_format_combo=DeletedCombo(),
        ids_channels=["w00"],
        n_channels=1,
        n_masks=1,
        user="tester",
        track_df=pd.DataFrame({"Position": [1], "t": [0]}),
        position_folders=[str(root / "exp_p0001")],
        current_position_index=0,
    )


def test_saving_survives_a_deleted_combo(window_with_dead_combos, qapp):
    """The crash reported at the end of a run all pass."""
    save_project_state(window_with_dead_combos)

    path = window_with_dead_combos.project_state_path
    assert os.path.isfile(path)


def test_the_saved_formats_fall_back_to_the_plain_values(
    window_with_dead_combos, qapp
):
    save_project_state(window_with_dead_combos)

    with open(window_with_dead_combos.project_state_path) as handle:
        state = json.load(handle)

    assert state["image_format"] == "png"
    assert state["tracking_format"] == "tTt"


def test_a_live_combo_still_wins_over_the_plain_value(
    fake_main_window, qtbot, qapp, tmp_path
):
    """While the loading window is open the widget is the source of truth."""
    root = tmp_path / "exp"
    root.mkdir()
    combo = QComboBox()
    qtbot.addWidget(combo)
    combo.addItems(["png", "tif"])
    combo.setCurrentText("tif")

    main_window = fake_main_window(
        folder=str(root),
        folder_list=["exp_p0001"],
        experiment_name="exp",
        project_name="Project_1",
        tracking_format="tTt",
        image_format="png",
        image_format_combo=combo,
        ids_channels=["w00"],
        n_channels=1,
        n_masks=1,
        user="tester",
        track_df=pd.DataFrame({"Position": [1], "t": [0]}),
        position_folders=[str(root / "exp_p0001")],
        current_position_index=0,
    )
    save_project_state(main_window)

    with open(main_window.project_state_path) as handle:
        state = json.load(handle)

    assert state["image_format"] == "tif"
