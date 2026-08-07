"""Tests that the larger SECQUOIA windows build.

The loading window, the experiment loader, the image and movie exporter, and
the lineage tree. They share the stand-in main window used by the dialog
tests, with the experiment folders really present on disk so each window
fills in rather than coming up empty. The loading window also has its tab
locking checked.
"""

from __future__ import annotations

import pytest
from conftest import combo_entries
from qtpy.QtWidgets import (
    QDialog,
    QPushButton,
    QTabWidget,
    QWidget,
)

pytestmark = pytest.mark.gui


@pytest.fixture(autouse=True)
def _no_modals(silence_modals):
    return silence_modals


# Load data window
@pytest.fixture
def load_data_window(dialog_main_window, qtbot):
    from SECQUOIA.gui.loading.load_data.load_data_dialog import (
        load_data_window as build,
    )

    dialog_main_window.mask_no_window = None
    build(dialog_main_window)
    window = dialog_main_window.mask_no_window
    qtbot.addWidget(window)
    return window


class TestLoadDataWindow:
    def test_builds_its_widget_tree(self, load_data_window):
        assert len(load_data_window.findChildren(QWidget)) > 50

    def test_is_stored_on_the_main_window(
        self, load_data_window, dialog_main_window
    ):
        assert dialog_main_window.mask_no_window is load_data_window

    def test_has_the_four_loading_stages(self, load_data_window):
        tabs = load_data_window.findChildren(QTabWidget)[0]

        assert [tabs.tabText(i) for i in range(tabs.count())] == [
            "Tracking",
            "Segmentation Options",
            "Channel Selection",
            "Loading",
        ]

    def test_only_the_first_stage_starts_unlocked(self, load_data_window):
        """The later tabs gate on inputs the user has not supplied yet."""
        tabs = load_data_window.findChildren(QTabWidget)[0]

        assert tabs.isTabEnabled(0) is True
        assert [tabs.isTabEnabled(i) for i in (1, 2, 3)] == [
            False,
            False,
            False,
        ]

    def test_exposes_the_tab_widget_on_the_main_window(
        self, load_data_window, dialog_main_window
    ):
        assert (
            dialog_main_window._tabs
            is load_data_window.findChildren(QTabWidget)[0]
        )

    def test_has_a_close_button_wired_to_the_window(
        self, load_data_window, dialog_main_window
    ):
        assert dialog_main_window.exit_button.text() == "Close"

        dialog_main_window.exit_button.click()

        assert load_data_window.isVisible() is False

    def test_replaces_a_window_that_was_already_open(
        self, dialog_main_window, qtbot
    ):
        """Opening it twice must not leave two windows behind."""
        from SECQUOIA.gui.loading.load_data.load_data_dialog import (
            load_data_window as build,
        )

        dialog_main_window.mask_no_window = None
        build(dialog_main_window)
        first = dialog_main_window.mask_no_window
        qtbot.addWidget(first)

        build(dialog_main_window)
        second = dialog_main_window.mask_no_window
        qtbot.addWidget(second)

        assert second is not first


# Experiment loader
@pytest.fixture
def experiment_loader(dialog_main_window, qtbot):
    """The dialog plus its (select, load, help) buttons."""
    from SECQUOIA.gui.loading.experiment_loader_dialog import (
        create_experiment_loader_dialog,
    )

    dialog, select_btn, load_btn, help_btn = create_experiment_loader_dialog(
        dialog_main_window
    )
    qtbot.addWidget(dialog)
    return dialog, select_btn, load_btn, help_btn


class TestExperimentLoaderDialog:
    def test_builds_a_dialog(self, experiment_loader):
        dialog, *_ = experiment_loader

        assert isinstance(dialog, QDialog)
        assert dialog.findChildren(QWidget)

    def test_returns_the_select_and_load_buttons(self, experiment_loader):
        _dialog, select_btn, load_btn, _help = experiment_loader

        assert isinstance(select_btn, QPushButton)
        assert isinstance(load_btn, QPushButton)

    def test_both_action_buttons_are_labelled(self, experiment_loader):
        _dialog, select_btn, load_btn, _help = experiment_loader

        assert select_btn.text()
        assert load_btn.text()

    def test_the_help_button_is_a_tool_button(self, experiment_loader):
        """The factory's annotation says QPushButton; it is a QToolButton.

        Pinned so a later annotation fix does not silently change the type.
        """
        from qtpy.QtWidgets import QToolButton

        _dialog, _select, _load, help_btn = experiment_loader

        assert isinstance(help_btn, QToolButton)

    def test_every_button_belongs_to_the_dialog(self, experiment_loader):
        from qtpy.QtWidgets import QAbstractButton

        dialog, *buttons = experiment_loader
        owned = set(dialog.findChildren(QAbstractButton))

        assert set(buttons) <= owned


# Image and movie exporter
@pytest.fixture
def exporter(dialog_main_window, qtbot):
    from SECQUOIA.gui.exporter import ImageMovieExporter

    dialog_main_window.FL_identifiers_1 = "w00"
    dialog_main_window.FL_identifiers_2 = "w01"
    instance = ImageMovieExporter(dialog_main_window)
    instance.open_image_movie_export_window(dialog_main_window)
    window = dialog_main_window.gif_window
    qtbot.addWidget(window)
    return window


class TestImageMovieExporter:
    def test_builds_its_widget_tree(self, exporter):
        assert len(exporter.findChildren(QWidget)) > 100

    def test_is_titled(self, exporter):
        assert exporter.windowTitle() == "Image & Movie Exporter"

    def test_offers_the_loaded_channels(self, exporter):
        entries = combo_entries(exporter)
        assert {"w00", "w01"} <= entries
        assert "Channel 1" not in entries

    def test_offers_a_choice_per_loaded_mask(self, exporter):
        """Two masks are loaded, plus the "no mask" entry."""
        entries = combo_entries(exporter)
        assert {"None", "Mask 1", "Mask 2"} <= entries

    def test_offers_the_loaded_identification(self, exporter):
        entries = combo_entries(exporter)
        assert "exp-p0001-001" in entries

    def test_has_export_controls(self, exporter):
        labels = {b.text().lower() for b in exporter.findChildren(QPushButton)}

        assert any("export" in label or "save" in label for label in labels)

    def test_refuses_to_open_without_a_loaded_folder(
        self, dialog_main_window, silence_modals
    ):
        """Exporting needs images, so an unloaded session gets a warning."""
        from SECQUOIA.gui.exporter import ImageMovieExporter

        dialog_main_window.folder_list = []
        dialog_main_window.gif_window = None
        instance = ImageMovieExporter(dialog_main_window)

        instance.open_image_movie_export_window(dialog_main_window)

        assert silence_modals  # the warning box was shown
        assert dialog_main_window.gif_window is None


# Lineage tree view
@pytest.fixture
def lineage_view(dialog_main_window, lineage_df):
    from SECQUOIA.gui.lineage_tree.lineage_view import LineageTreeView

    return LineageTreeView(
        lineage_df, "pos1_id1", main_window=dialog_main_window
    )


class TestLineageTreeView:
    def test_lays_out_every_track_of_the_lineage(self, lineage_view):
        """The default fixture lineage is five tracks over three generations."""
        assert lineage_view.state["tracks"]["TrackNumber"].tolist() == [
            1,
            2,
            3,
            4,
            5,
        ]

    def test_draws_the_tree_into_the_plot(self, lineage_view):
        assert len(lineage_view.plot.items) > 0

    def test_records_the_division_edges(self, lineage_view):
        edges = lineage_view.state["edges"]

        assert set(map(tuple, edges[["parent", "child"]].to_numpy())) == {
            (1, 2),
            (1, 3),
            (2, 4),
            (2, 5),
        }

    def test_the_heatmap_mode_renders_more_than_the_tree(
        self, dialog_main_window, lineage_df
    ):
        """T and H are the two modes the style dialog offers."""
        from SECQUOIA.gui.lineage_tree.lineage_view import LineageTreeView

        tree = LineageTreeView(
            lineage_df,
            "pos1_id1",
            render_mode="T",
            main_window=dialog_main_window,
        )
        heat = LineageTreeView(
            lineage_df,
            "pos1_id1",
            render_mode="H",
            main_window=dialog_main_window,
        )

        assert len(heat.plot.items) > len(tree.plot.items)

    def test_builds_without_a_main_window(self, lineage_df):
        """The view is also used standalone, with no window to read state off."""
        from SECQUOIA.gui.lineage_tree.lineage_view import LineageTreeView

        view = LineageTreeView(lineage_df, "pos1_id1")

        assert view.state["tracks"]["TrackNumber"].tolist() == [1, 2, 3, 4, 5]

    def test_an_unknown_identification_renders_empty(
        self, dialog_main_window, lineage_df
    ):
        """It must come up blank rather than raising or drawing another cell."""
        from SECQUOIA.gui.lineage_tree.lineage_view import LineageTreeView

        view = LineageTreeView(
            lineage_df, "no-such-cell", main_window=dialog_main_window
        )

        assert view.state == {}
        assert view.plot.items == []
