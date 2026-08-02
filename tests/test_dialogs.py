"""Tests that the SECQUOIA dialogs build and respond.

Every dialog is the real class, built without showing a window. The stand-in
main window holds one track over four frames with two masks and two
channels, which is what the dialogs read to fill their dropdowns. Modal
boxes are replaced, since they would otherwise sit and wait for a click.
"""

from __future__ import annotations

import pytest
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QPushButton,
    QSpinBox,
    QWidget,
)

pytestmark = pytest.mark.gui


@pytest.fixture(autouse=True)
def _no_modals(silence_modals):
    """No dialog in this module may block the run."""
    return silence_modals


def combo_entries(widget, *, lower: bool = False) -> set[str]:
    """Every entry offered by every combo box under ``widget``."""
    return {
        combo.itemText(i).lower() if lower else combo.itemText(i)
        for combo in widget.findChildren(QComboBox)
        for i in range(combo.count())
    }


# Plot parameters
@pytest.fixture
def plot_params_dialog(dialog_main_window, qtbot):
    from SECQUOIA.gui.dialogs.plot_params_dialog import PlotParamsDialog

    dialog = PlotParamsDialog(dialog_main_window)
    qtbot.addWidget(dialog)
    return dialog


class TestPlotParamsDialog:
    def test_builds_its_widget_tree(self, plot_params_dialog):
        assert plot_params_dialog.findChildren(QWidget)

    def test_is_parented_to_the_main_window(
        self, plot_params_dialog, dialog_main_window
    ):
        assert plot_params_dialog.parent() is dialog_main_window

    def test_is_not_modal(self, plot_params_dialog):
        """It stays open while the user watches the plots update."""
        assert plot_params_dialog.isModal() is False

    def test_populates_the_symbol_combo(self, plot_params_dialog):
        assert plot_params_dialog.symbol_combo.count() > 0

    def test_shows_the_current_symbol_size(self, plot_params_dialog):
        assert (
            plot_params_dialog.symbol_size_spin.value()
            == plot_params_dialog.params["symbol_size"]
        )

    def test_the_spin_boxes_have_sane_ranges(self, plot_params_dialog):
        spin = plot_params_dialog.symbol_size_spin
        assert spin.minimum() >= 0
        assert spin.maximum() > spin.minimum()

    def test_applying_writes_the_controls_back(
        self, plot_params_dialog, monkeypatch
    ):
        import SECQUOIA.gui.dialogs.plot_params_dialog as module

        monkeypatch.setattr(module, "update_plot", lambda mw: None)
        monkeypatch.setattr(module.lt, "lineage_tree", lambda mw: None)
        plot_params_dialog.symbol_size_spin.setValue(17)
        plot_params_dialog.axis_font_spin.setValue(9)

        plot_params_dialog.apply_changes()

        assert plot_params_dialog.params["symbol_size"] == 17
        assert plot_params_dialog.params["axis_font_size"] == 9

    def test_applying_redraws_the_plots(self, plot_params_dialog, monkeypatch):
        import SECQUOIA.gui.dialogs.plot_params_dialog as module

        redrawn = []
        monkeypatch.setattr(
            module, "update_plot", lambda mw: redrawn.append("plot")
        )
        monkeypatch.setattr(
            module.lt, "lineage_tree", lambda mw: redrawn.append("lineage")
        )

        plot_params_dialog.apply_changes()

        assert redrawn == ["plot", "lineage"]

    def test_ok_applies_and_closes(self, plot_params_dialog, monkeypatch):
        import SECQUOIA.gui.dialogs.plot_params_dialog as module

        monkeypatch.setattr(module, "update_plot", lambda mw: None)
        monkeypatch.setattr(module.lt, "lineage_tree", lambda mw: None)
        plot_params_dialog.symbol_size_spin.setValue(13)

        plot_params_dialog._on_ok()

        assert plot_params_dialog.params["symbol_size"] == 13
        assert plot_params_dialog.result() == QDialog.DialogCode.Accepted

    def test_the_factory_builds_without_showing(
        self, dialog_main_window, qtbot
    ):
        from SECQUOIA.gui.dialogs.plot_params_dialog import (
            create_plot_params_dialog,
        )

        dialog = create_plot_params_dialog(dialog_main_window)
        qtbot.addWidget(dialog)

        assert dialog.isVisible() is False


# Derived metrics
@pytest.fixture
def metric_dialog(dialog_main_window, qtbot):
    from SECQUOIA.gui.dialogs.metric_dialog import MetricDialog

    dialog = MetricDialog(dialog_main_window, 0)
    qtbot.addWidget(dialog)
    return dialog


class TestMetricDialog:
    def test_builds_its_widget_tree(self, metric_dialog):
        assert len(metric_dialog.findChildren(QWidget)) > 20

    def test_offers_the_features_present_in_the_dataframe(self, metric_dialog):
        entries = combo_entries(metric_dialog)

        assert "AreaMorphology" in entries

    def test_offers_both_masks(self, metric_dialog):
        entries = combo_entries(metric_dialog)

        assert {"1", "2"} <= entries

    def test_has_a_run_and_an_exit_button(self, metric_dialog):
        labels = {b.text() for b in metric_dialog.findChildren(QPushButton)}

        assert {"Run", "Exit"} <= labels


# Mask arithmetic
@pytest.fixture
def arithmetic_dialog(dialog_main_window, qtbot):
    from SECQUOIA.gui.dialogs.arithmetic_dialog import MaskArithmeticDialog

    dialog = MaskArithmeticDialog(dialog_main_window)
    qtbot.addWidget(dialog)
    return dialog


class TestMaskArithmeticDialog:
    def test_builds_its_widget_tree(self, arithmetic_dialog):
        assert arithmetic_dialog.findChildren(QWidget)

    def test_offers_a_choice_per_mask(self, arithmetic_dialog):
        """Two masks are loaded, so both must be selectable."""
        entries = combo_entries(arithmetic_dialog)

        assert {"1", "2"} <= entries

    def test_offers_the_bitwise_operations(self, arithmetic_dialog):
        entries = combo_entries(arithmetic_dialog, lower=True)

        assert {"intersection", "union", "mutually exclusive"} <= entries

    def test_offers_a_way_to_ignore_the_second_mask(self, arithmetic_dialog):
        """A single mask operation still needs the dialog."""
        entries = combo_entries(arithmetic_dialog, lower=True)

        assert any("ignore" in entry for entry in entries)

    def test_the_module_imports_on_its_own(self):
        """This module used to be circular with main_window."""
        import importlib

        assert importlib.import_module(
            "SECQUOIA.gui.dialogs.arithmetic_dialog"
        )


# Lineage style
@pytest.fixture
def lineage_tools_combo(dialog_main_window, qtbot):
    """The main window's own T/H selector, which the dialog reads and writes."""
    combo = QComboBox()
    combo.addItems(["T", "H"])
    qtbot.addWidget(combo)
    dialog_main_window.lineage_tools = {"F": combo}
    return combo


@pytest.fixture
def lineage_style_dialog(dialog_main_window, lineage_tools_combo, qtbot):
    from SECQUOIA.gui.dialogs.lineage_style_dialog import LineageStyleDialog

    dialog = LineageStyleDialog(parent=dialog_main_window)
    qtbot.addWidget(dialog)
    return dialog


class TestLineageStyleDialog:
    def test_builds_its_widget_tree(self, lineage_style_dialog):
        assert lineage_style_dialog.findChildren(QWidget)

    def test_is_parented_to_the_main_window(
        self, lineage_style_dialog, dialog_main_window
    ):
        """Guards the trap above: unparented, the dialog reads nothing."""
        assert lineage_style_dialog.parent() is dialog_main_window

    def test_reports_the_mode_selected_on_the_main_window(
        self, lineage_style_dialog, lineage_tools_combo
    ):
        lineage_tools_combo.setCurrentIndex(0)
        assert lineage_style_dialog._get_current_mode() == "T"

        lineage_tools_combo.setCurrentIndex(1)
        assert lineage_style_dialog._get_current_mode() == "H"

    def test_opens_on_the_mode_already_in_use(
        self, dialog_main_window, lineage_tools_combo, qtbot
    ):
        """Reopening the dialog must not silently reset the user to Tree."""
        from SECQUOIA.gui.dialogs.lineage_style_dialog import (
            LineageStyleDialog,
        )

        lineage_tools_combo.setCurrentIndex(1)  # Heatmap

        dialog = LineageStyleDialog(parent=dialog_main_window)
        qtbot.addWidget(dialog)

        assert dialog.mode_combo.currentIndex() == 1

    def test_switching_mode_changes_which_controls_are_shown(
        self, lineage_style_dialog
    ):
        """The tree and heatmap parameter groups are mutually exclusive."""
        mode_combo = lineage_style_dialog.mode_combo
        assert mode_combo.count() == 2

        visibility = []
        for index in range(mode_combo.count()):
            mode_combo.setCurrentIndex(index)
            visibility.append(
                (
                    lineage_style_dialog.tree_grp.isVisibleTo(
                        lineage_style_dialog
                    ),
                    lineage_style_dialog.heat_grp.isVisibleTo(
                        lineage_style_dialog
                    ),
                )
            )

        assert visibility == [(True, False), (False, True)]

    def test_applying_pushes_the_chosen_mode_to_the_main_window(
        self, lineage_style_dialog, lineage_tools_combo
    ):
        lineage_tools_combo.setCurrentIndex(0)  # start on Tree

        lineage_style_dialog._set_current_mode("H")

        assert lineage_tools_combo.currentText() == "H"
        assert lineage_style_dialog._get_current_mode() == "H"

    def test_the_mode_can_be_pushed_back_to_tree(
        self, lineage_style_dialog, lineage_tools_combo
    ):
        """Both directions, so a one-way setter cannot pass."""
        lineage_tools_combo.setCurrentIndex(1)  # start on Heatmap

        lineage_style_dialog._set_current_mode("T")

        assert lineage_tools_combo.currentText() == "T"
        assert lineage_style_dialog._get_current_mode() == "T"

    def test_the_setter_accepts_the_dialogs_descriptive_labels(
        self, lineage_style_dialog, lineage_tools_combo
    ):
        """The dialog's own combo reads "H (Heatmap)"; the setter takes it."""
        lineage_tools_combo.setCurrentIndex(0)

        lineage_style_dialog._set_current_mode("H (Heatmap)")

        assert lineage_tools_combo.currentText() == "H"


# Track fusion
@pytest.fixture
def track_fuse_dialog(dialog_main_window, qtbot):
    from SECQUOIA.gui.dialogs.track_fuse_dialog import TrackFuseDialog

    dialog = TrackFuseDialog(dialog_main_window)
    qtbot.addWidget(dialog)
    return dialog


class TestTrackFuseDialog:
    def test_builds_its_widget_tree(self, track_fuse_dialog):
        assert track_fuse_dialog.findChildren(QWidget)

    def test_offers_the_loaded_time_range(
        self, track_fuse_dialog, dialog_main_window
    ):
        """The window has four frames, so the spin covers indices 0..3."""
        (spin,) = track_fuse_dialog.findChildren(QSpinBox)

        assert spin.minimum() == 0
        assert spin.maximum() == dialog_main_window.time_max_selected - 1

    def test_the_time_range_can_be_refreshed(
        self, track_fuse_dialog, dialog_main_window
    ):
        """Loading a longer position must widen the spin, not leave it stale."""
        (spin,) = track_fuse_dialog.findChildren(QSpinBox)
        assert spin.maximum() == 3

        dialog_main_window.time_max_selected = 12
        track_fuse_dialog.refresh_time_range()

        assert spin.maximum() == 11


# Outlier detection
@pytest.fixture
def outlier_window(dialog_main_window, qtbot):
    from SECQUOIA.gui.outlier.detection.window import OutlierDetectionWindow

    window = OutlierDetectionWindow(dialog_main_window)
    qtbot.addWidget(window)
    return window


class TestOutlierDetectionWindow:
    def test_builds_its_widget_tree(self, outlier_window):
        assert len(outlier_window.findChildren(QWidget)) > 50

    def test_has_more_than_one_tab(self, outlier_window):
        from qtpy.QtWidgets import QTabWidget

        tabs = outlier_window.findChildren(QTabWidget)
        assert tabs
        assert tabs[0].count() > 1

    def test_every_tab_can_be_selected(self, outlier_window):
        from qtpy.QtWidgets import QTabWidget

        tabs = outlier_window.findChildren(QTabWidget)[0]

        for index in range(tabs.count()):
            tabs.setCurrentIndex(index)
            assert tabs.currentIndex() == index

    def test_starts_with_one_rule_row(self, outlier_window):
        assert len(outlier_window.threshold_tab.rows_widgets) == 1

    def test_offers_the_numeric_columns_of_the_dataframe(self, outlier_window):
        """Without derived features, the rule builder falls back to columns."""
        entries = combo_entries(outlier_window.threshold_tab.page)

        assert "AreaMorphologyM1" in entries
        assert "AreaMorphologyM2" in entries

    def test_derived_features_take_priority_over_the_columns(
        self, dialog_main_window, qtbot
    ):
        from SECQUOIA.gui.outlier.detection.window import (
            OutlierDetectionWindow,
        )

        dialog_main_window._feature_defs = {"MyRatio": {"template": "MyRatio"}}
        window = OutlierDetectionWindow(dialog_main_window)
        qtbot.addWidget(window)

        entries = combo_entries(window.threshold_tab.page)

        assert "MyRatio" in entries
        assert "AreaMorphologyM1" not in entries

    def test_more_rule_rows_can_be_added(self, outlier_window):
        tab = outlier_window.threshold_tab
        before = len(tab.rows_widgets)

        for _ in range(3):
            tab.add_row()

        assert len(tab.rows_widgets) == before + 3

    def test_every_rule_row_can_be_cleared(self, outlier_window):
        tab = outlier_window.threshold_tab
        tab.add_row()

        tab.clear_all_rows()

        assert tab.rows_widgets == []

    def test_a_rule_row_offers_the_comparison_operators(self, outlier_window):
        entries = combo_entries(outlier_window.threshold_tab.page)

        assert {"<", "<=", ">", ">=", "="} <= entries

    def test_a_rule_row_offers_every_mask_and_channel(self, outlier_window):
        entries = combo_entries(outlier_window.threshold_tab.page)

        assert {"1", "2"} <= entries  # masks
        assert {"00", "01"} <= entries  # channels


@pytest.fixture
def outlier_parameters_dialog(dialog_main_window, qtbot):
    from SECQUOIA.gui.outlier.summary_dialog import OutlierParametersDialog

    dialog = OutlierParametersDialog(dialog_main_window)
    qtbot.addWidget(dialog)
    return dialog


class TestOutlierParametersDialog:
    def test_is_titled_for_the_current_parameters(
        self, outlier_parameters_dialog
    ):
        assert outlier_parameters_dialog.windowTitle() == (
            "Outlier Detection – Current Parameters"
        )

    def test_offers_a_way_out_and_a_way_to_the_plots(
        self, outlier_parameters_dialog
    ):
        labels = {
            b.text()
            for b in outlier_parameters_dialog.findChildren(QPushButton)
        }

        assert {"OK", "Show Plots"} <= labels

    def test_is_parented_to_the_main_window(
        self, outlier_parameters_dialog, dialog_main_window
    ):
        assert outlier_parameters_dialog.parent() is dialog_main_window
