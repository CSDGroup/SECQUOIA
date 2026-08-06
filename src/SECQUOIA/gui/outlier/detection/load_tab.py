"""The Load Rules tab: pick a saved rules file and populate the other tabs."""

from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import asdict

from qtpy.QtWidgets import (
    QFileDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from SECQUOIA.config import TOOLTIPSTEXT
from SECQUOIA.core.outlier_detection import (
    RulesPack,
    _outlier_rules_dir,
    load_outlier_rules_from_disk,
)
from SECQUOIA.gui.common.ui_utils import make_tab_scaffold, make_tab_title

LOG = logging.getLogger(__name__)

__all__ = ["LoadTab"]


class LoadTab:
    """File picker for loading a previously saved rules file into the UI."""

    def __init__(
        self, main_window, win, tabs, threshold_tab, sliding_tab, run_tab
    ):
        self.main_window = main_window
        self.win = win
        self.tabs = tabs
        self.threshold_tab = threshold_tab
        self.sliding_tab = sliding_tab
        self.run_tab = run_tab

        self.page = QWidget()
        content_v, side_v = make_tab_scaffold(self.page)

        content_v.addWidget(
            make_tab_title(
                "Load a previously saved outlier rules file (.json)"
            )
        )
        load_hint = QLabel(
            "Choose a rules file from your Outlier_detection folder."
        )
        load_hint.setWordWrap(True)
        content_v.addWidget(load_hint)
        content_v.addStretch(1)

        choose_btn = QPushButton("Select")
        choose_btn.setToolTip(TOOLTIPSTEXT.LOAD_BTN)
        choose_btn.setMinimumHeight(28)
        choose_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        side_v.addWidget(choose_btn)

        choose_btn.clicked.connect(self._on_choose_clicked)

    def _apply_loaded_rules(self, payload: dict):
        """Load saved threshold and sliding-window rules into the UI."""
        try:
            pack = RulesPack.from_dict(payload)
        except (TypeError, ValueError, AttributeError) as e:
            LOG.error("[Rules] Invalid rules file: %s", e)
            return

        m_n, ch_n = self.threshold_tab.m_n, self.threshold_tab.ch_n
        file_m_n = pack.m_n or m_n
        file_ch_n = pack.ch_n or ch_n

        if (file_m_n != m_n) or (file_ch_n != ch_n):
            try:
                QMessageBox.information(
                    self.win,
                    "Info",
                    f"Saved rules used m_n={file_m_n}, ch_n={file_ch_n}; "
                    f"current GUI has m_n={m_n}, ch_n={ch_n}. "
                    "Loading will keep any mask/channel IDs that exist here.",
                )
            except RuntimeError:
                LOG.warning(
                    "[Rules] Mask/channel counts differ; proceeding best-effort."
                )

        self.threshold_tab.clear_all_rows()

        feature_keys = self.threshold_tab.feature_keys
        rules = [asdict(r) for r in pack.rules]
        if not rules:
            self.threshold_tab.add_row()
        else:
            for r in rules:
                self.threshold_tab.append_row_with_values(
                    feat=r.get(
                        "feat",
                        str(
                            feature_keys[0]
                            if feature_keys
                            else "MeanNoBgCorrected"
                        ),
                    ),
                    op1=r.get("op1", "<"),
                    val1=r.get("val1", 0.0),
                    op2=r.get("op2", None),
                    val2=r.get("val2", 0.0),
                    combine=r.get("combine", "OR"),
                    masks=[
                        int(x)
                        for x in r.get("masks", [])
                        if isinstance(x, (int | float | str))
                    ],
                    channels=[
                        x for x in r.get("channels", []) if isinstance(x, str)
                    ],
                )

        sw_list = [
            asdict(s)
            for s in pack.sliding_windows
            if float(getattr(s, "sd_factor", 0) or 0) > 0
            and float(getattr(s, "t_max", 0) or 0)
            > float(getattr(s, "t_min", 0) or 0)
        ]

        try:
            self.sliding_tab.clear_rows()
            if sw_list:
                for sw in sw_list:
                    self.sliding_tab.append_row_with_values(
                        feature=str(
                            sw.get(
                                "feature",
                                (
                                    feature_keys[0]
                                    if feature_keys
                                    else "MeanNoBgCorrected"
                                ),
                            )
                        ),
                        mask=sw.get("mask", None),
                        channel=sw.get("channel", None),
                        t_min=float(sw.get("t_min", 0)),
                        t_max=float(sw.get("t_max", 0)),
                        sd_factor=float(sw.get("sd_factor", 0.0)),
                    )
            else:
                self.sliding_tab.add_row()
        except (TypeError, ValueError, RuntimeError, AttributeError) as e:
            LOG.warning("[Rules] Could not rebuild sliding windows: %s", e)

    def _on_load_rules_returning_success(self) -> bool:
        """Load saved outlier rules and return whether loading succeeded."""
        main_window = self.main_window
        folder = _outlier_rules_dir(main_window) or os.getcwd()
        chosen = None
        dlg = QFileDialog(
            self.win,
            "Load Outlier Rules",
            folder,
            "Rules (*.json);;All Files (*)",
        )
        dlg.setFileMode(QFileDialog.ExistingFile)
        if dlg.exec_():
            files = dlg.selectedFiles()
            if files:
                chosen = files[0]

        if not chosen:
            return False

        payload = load_outlier_rules_from_disk(chosen)
        if not payload:
            with contextlib.suppress(Exception):
                QMessageBox.warning(
                    self.win,
                    "Load Failed",
                    "Could not read the selected rules file.",
                )
            return False

        self._apply_loaded_rules(payload)
        return True

    def _on_choose_clicked(self):
        """Load a rules file and switch to the Run tab when it succeeds."""
        if self._on_load_rules_returning_success():
            self.run_tab.refresh_summary()
            self.tabs.setCurrentWidget(self.run_tab.page)
