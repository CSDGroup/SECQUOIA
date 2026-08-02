"""The outlier detection dialog, split into one module per tab."""

from __future__ import annotations

from SECQUOIA.gui.outlier.detection.window import (
    OutlierDetectionWindow,
    open_outlier_detection_window,
)

__all__ = ["OutlierDetectionWindow", "open_outlier_detection_window"]
