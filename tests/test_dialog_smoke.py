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
