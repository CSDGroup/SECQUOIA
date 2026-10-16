"""Tests that a dialog can be built without running the application.

This is the smallest example of the pattern the other GUI tests follow. The
plot parameters dialog is built against a stand-in main window holding one
track over four time points, one mask and one channel, which is enough for
the dialog to fill its controls.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytestmark = pytest.mark.gui


@pytest.fixture
def small_track_df() -> pd.DataFrame:
    """One track, four frames, one mask, one channel."""
    return pd.DataFrame(
        {
            "Identification": ["pos1_id1"] * 4,
            "TrackNumber": [1, 1, 1, 1],
            "t": [0, 1, 2, 3],
            "Calculated_Time": [0.0, 3.0, 6.0, 9.0],
            "AreaMorphologyM1": [10.0, 11.0, 12.0, 13.0],
            "MeanNoBgCorrectedCh1M1": [100.0, 110.0, 120.0, 130.0],
        }
    )


def test_plot_params_dialog_builds(fake_main_window_widget, small_track_df):
    from SECQUOIA.gui.dialogs.plot_params_dialog import PlotParamsDialog

    main_window = fake_main_window_widget(
        track_df=small_track_df,
        n_masks=1,
        n_channels=1,
    )

    dialog = PlotParamsDialog(main_window)

    # The dialog really built its widget tree rather than bailing out early.
    assert dialog.findChildren(object)
    assert dialog.parent() is main_window


def _set(**kw):
    from SECQUOIA.core.segmentation.click_segment import (
        ClickSettings,
        ClickSettingsSet,
    )

    return ClickSettingsSet(default=ClickSettings(**kw))


def test_click_segment_dialog_roundtrip(fake_main_window_widget):
    from SECQUOIA.core.segmentation.click_segment import (
        ClickSettings,
        ClickSettingsSet,
    )
    from SECQUOIA.gui.dialogs.click_segment_dialog import ClickSegmentDialog

    start = _set(
        smoothing=2.0,
        edge_sensitivity=0.8,
        cellprob_threshold=-1.0,
        diameter=40.0,
    )
    main_window = fake_main_window_widget()
    dialog = ClickSegmentDialog(main_window, start)
    assert dialog.settings() == start

    dialog.diameter.setValue(0)
    assert dialog.settings().default.diameter is None

    dialog._reset()
    assert dialog.settings() == ClickSettingsSet(default=ClickSettings())


def test_click_segment_dialog_has_the_help_button(fake_main_window_widget):
    from qtpy.QtWidgets import QToolButton

    from SECQUOIA.config import TOOLTIPSTEXT
    from SECQUOIA.core.segmentation.click_segment import ClickSettingsSet
    from SECQUOIA.gui.dialogs.click_segment_dialog import ClickSegmentDialog

    main_window = fake_main_window_widget()
    dialog = ClickSegmentDialog(main_window, ClickSettingsSet())

    tips = [b.toolTip() for b in dialog.findChildren(QToolButton)]
    assert TOOLTIPSTEXT.CLICK_HELP in tips


def test_click_segment_dialog_is_non_modal_and_single(
    fake_main_window_widget,
):
    from SECQUOIA.core.segmentation.click_segment import (
        ClickSettings,
        ClickSettingsSet,
    )
    from SECQUOIA.gui.dialogs.click_segment_dialog import (
        open_click_segment_dialog,
    )

    main_window = fake_main_window_widget()
    open_click_segment_dialog(main_window)
    dialog = main_window._click_segment_dialog
    assert dialog.isVisible()
    assert not dialog.isModal()

    open_click_segment_dialog(main_window)
    assert main_window._click_segment_dialog is dialog

    dialog.edge.setValue(1.5)
    dialog.accept()
    assert not dialog.isVisible()
    assert main_window.click_segment_settings == _set(edge_sensitivity=1.5)

    main_window.click_segment_settings = ClickSettingsSet(
        default=ClickSettings(smoothing=3.0)
    )
    open_click_segment_dialog(main_window)
    assert dialog.smoothing.value() == 3.0
    assert dialog.edge.value() == 0.5
    dialog.reject()


def test_click_segment_dialog_layout_and_mask_size(fake_main_window_widget):
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QFormLayout

    from SECQUOIA.gui.dialogs.click_segment_dialog import ClickSegmentDialog

    main_window = fake_main_window_widget()
    dialog = ClickSegmentDialog(main_window, _set(mask_size=-2))
    assert dialog.mask_size.value() == -2
    assert dialog.settings().default.mask_size == -2

    boxes = (
        dialog.smoothing,
        dialog.edge,
        dialog.cellprob,
        dialog.diameter,
        dialog.mask_size,
    )
    assert (
        len({b.width() or b.maximumWidth() for b in boxes}) == 1
    )  # same size

    form = dialog.findChildren(QFormLayout)[0]
    assert form.labelAlignment() & Qt.AlignLeft


def test_click_segment_dialog_per_mask_overrides(fake_main_window_widget):
    """'All masks' is the default; a mask can override it, and go back to inheriting."""
    from SECQUOIA.core.segmentation.click_segment import (
        ClickSettings,
        ClickSettingsSet,
    )
    from SECQUOIA.gui.dialogs.click_segment_dialog import ClickSegmentDialog

    main_window = fake_main_window_widget()
    dialog = ClickSegmentDialog(main_window, ClickSettingsSet(), n_masks=2)
    assert [dialog.target.itemText(i) for i in range(3)] == [
        "All masks (default)",
        "Mask 1",
        "Mask 2",
    ]
    assert not dialog.inherit.isVisibleTo(dialog)

    dialog.target.setCurrentIndex(2)
    assert dialog.inherit.isChecked() and not dialog.mask_size.isEnabled()
    assert dialog.settings().overrides == {}

    dialog.inherit.setChecked(False)
    dialog.mask_size.setValue(-2)
    result = dialog.settings()
    assert result.for_mask(2).mask_size == -2
    assert result.for_mask(1).mask_size == 0
    assert result.default.mask_size == 0

    dialog.target.setCurrentIndex(0)
    dialog.smoothing.setValue(3.0)
    result = dialog.settings()
    assert result.for_mask(1).smoothing == 3.0
    assert result.for_mask(2).smoothing == 1.0
    assert result.for_mask(2).mask_size == -2

    dialog.target.setCurrentIndex(2)
    dialog.inherit.setChecked(True)
    assert dialog.settings().overrides == {}
    assert dialog.settings().for_mask(2) == ClickSettings(smoothing=3.0)

    dialog.inherit.setChecked(False)
    dialog.mask_size.setValue(4)
    dialog._reset()
    assert dialog.inherit.isChecked()
    assert dialog.settings().overrides == {}
