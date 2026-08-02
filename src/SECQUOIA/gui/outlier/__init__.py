"""Outlier detection and inspection dialogs."""

from __future__ import annotations

from SECQUOIA.gui.outlier.detection import (
    open_outlier_detection_window,
)
from SECQUOIA.gui.outlier.rules import snapshot_outlier_ui_to_pack
from SECQUOIA.gui.outlier.summary_dialog import (
    OutlierParametersDialog,
    OutlierPlotsDialog,
    show_current_outlier_parameters,
)
from SECQUOIA.gui.outlier.widgets import CheckableComboBox

__all__ = [
    "CheckableComboBox",
    "OutlierParametersDialog",
    "OutlierPlotsDialog",
    "open_outlier_detection_window",
    "show_current_outlier_parameters",
    "snapshot_outlier_ui_to_pack",
]
