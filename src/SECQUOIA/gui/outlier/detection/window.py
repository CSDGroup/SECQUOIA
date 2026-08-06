"""The outlier detection window: composes the threshold, sliding-window, load, and run tabs."""

from __future__ import annotations

import contextlib
import logging

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from SECQUOIA.config import LINKS, TOOLTIPSTEXT
from SECQUOIA.gui.common.messages import show_folder_warning
from SECQUOIA.gui.common.ui_utils import (
    add_progress_bar,
    fit_to_screen,
    make_help_button,
    spinbox_arrow_pngs,
)
from SECQUOIA.gui.outlier.detection.load_tab import LoadTab
from SECQUOIA.gui.outlier.detection.run_tab import RunTab
from SECQUOIA.gui.outlier.detection.sliding_window_tab import SlidingWindowTab
from SECQUOIA.gui.outlier.detection.threshold_rules_tab import (
    ThresholdRulesTab,
)
from SECQUOIA.gui.outlier.style import outlier_window_stylesheet

LOG = logging.getLogger(__name__)

__all__ = ["OutlierDetectionWindow", "open_outlier_detection_window"]


def _resolve_feature_keys(main_window) -> list:
    """Return the feature names to offer in the rule builders.

    Uses the keys of `_feature_defs`, falling back to the numeric columns of
    `filtered_df` and finally to a single default feature name.
    """
    feature_keys = []
    with contextlib.suppress(RuntimeError, AttributeError, TypeError):
        feature_keys = list(getattr(main_window, "_feature_defs", {}).keys())
    if not feature_keys:
        with contextlib.suppress(Exception):
            df_all = getattr(main_window, "filtered_df", None)
            if df_all is not None and not df_all.empty:
                cols = [
                    c
                    for c in df_all.columns
                    if df_all[c].dtype.kind in ("i", "u", "f")
                ]
                feature_keys = cols[:200]
    if not feature_keys:
        feature_keys = ["MeanNoBgCorrected"]
    return feature_keys


class OutlierDetectionWindow(QWidget):
    """Threshold-rule, sliding-window, load, and run tabs for outlier detection."""

    def __init__(self, main_window: QWidget):
        super().__init__()
        self.main_window = main_window

        feature_keys = _resolve_feature_keys(main_window)
        m_n = int(getattr(main_window, "n_masks", 0))
        channel_ids = [
            ch[1:] for ch in getattr(main_window, "ids_channels", [])
        ]
        ch_n = getattr(main_window, "n_channels", 0)

        self.setWindowTitle("Outlier Detection")
        self.setMinimumSize(400, 100)
        with contextlib.suppress(Exception):
            arrows = spinbox_arrow_pngs("#E6E6E6")
            self.setStyleSheet(outlier_window_stylesheet(arrows))

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(6)

        self.tabs = QTabWidget(self)
        outer.addWidget(self.tabs)

        out_bar, out_helpers = add_progress_bar(outer)
        self.set_progress = out_helpers["set"]
        self.finish_progress = out_helpers["finish"]
        self.set_progress_step = out_helpers["step"]

        self.threshold_tab = ThresholdRulesTab(
            main_window, self, feature_keys, m_n, ch_n, channel_ids
        )
        self.sliding_tab = SlidingWindowTab(
            main_window, feature_keys, m_n, ch_n
        )
        self.run_tab = RunTab(
            main_window, self, self.threshold_tab, self.sliding_tab
        )
        self.load_tab = LoadTab(
            main_window,
            self,
            self.tabs,
            self.threshold_tab,
            self.sliding_tab,
            self.run_tab,
        )

        self.tabs.addTab(self.threshold_tab.page, "Threshold Rules")
        self.tabs.addTab(self.sliding_tab.page, "Sliding window")
        self.tabs.addTab(self.load_tab.page, "Load Rules")
        self.tabs.addTab(self.run_tab.page, "Run")

        help_btn = make_help_button(
            self.tabs, TOOLTIPSTEXT.OUT_HELP, LINKS.GITHUB_OUTLIER
        )
        self.tabs.setCornerWidget(help_btn, Qt.TopRightCorner)

        self.threshold_tab.next_btn.clicked.connect(
            lambda *_: self.tabs.setCurrentWidget(self.sliding_tab.page)
        )
        self.sliding_tab.next_btn.clicked.connect(
            lambda *_: self.tabs.setCurrentWidget(self.run_tab.page)
        )
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self.adjustSize()
        fit_to_screen(self, max_frac=0.85, min_size=(700, 320))
        self.show()
        main_window._outlier_window_ref = self

    def _on_tab_changed(self, idx):
        """Refresh the run tab's summary when it becomes the active tab."""
        if self.tabs.widget(idx) is self.run_tab.page:
            self.run_tab.refresh_summary()


def open_outlier_detection_window(main_window: QWidget) -> None:
    """Open the outlier-detection window."""
    if not hasattr(main_window, "folder_list") or not main_window.folder_list:
        LOG.warning("Please first load CSV file and select folder")
        show_folder_warning(main_window)
        return
    OutlierDetectionWindow(main_window)
