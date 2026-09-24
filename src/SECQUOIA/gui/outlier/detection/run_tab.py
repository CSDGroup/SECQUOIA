"""The Run tab: review the configured rules and apply outlier detection."""

from __future__ import annotations

from qtpy.QtWidgets import (
    QApplication,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from SECQUOIA.config import TOOLTIPSTEXT
from SECQUOIA.core.outlier_detection import (
    run_outlier_pipeline_for_all_positions,
    run_sliding_windows,
    run_threshold_rules,
    save_outlier_rules_to_disk,
    update_outlier_detection_in_track_df,
    update_unique_outliers_ids,
)
from SECQUOIA.core.outlier_detection.close_masks import (
    apply_close_mask_detection,
    close_mask_summary,
)
from SECQUOIA.core.project_state import save_track_df
from SECQUOIA.core.segmentation.mask_selection import ensure_current_df_subset
from SECQUOIA.gui.common.ui_utils import make_tab_scaffold, make_tab_title
from SECQUOIA.gui.outlier.markers import update_outlier_marker
from SECQUOIA.gui.outlier.outlier_list import (
    _update_outlier_list,
    auto_select_first_item,
)
from SECQUOIA.gui.outlier.rules import snapshot_outlier_ui_to_pack

__all__ = ["RunTab"]


class RunTab:
    """Summary of the configured rules, plus Apply/Exit actions."""

    def __init__(self, main_window, win, threshold_tab, sliding_tab):
        self.main_window = main_window
        self.win = win
        self.threshold_tab = threshold_tab
        self.sliding_tab = sliding_tab

        self.page = QWidget()
        content_v, side_v = make_tab_scaffold(self.page)

        content_v.addWidget(
            make_tab_title("Review your selections before running:")
        )

        # Summary text area
        self.summary_box = QPlainTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setMinimumHeight(200)
        content_v.addWidget(self.summary_box, 1)  # let it expand on the left

        refresh_summary_btn = QPushButton("Refresh")
        refresh_summary_btn.setMinimumHeight(28)
        refresh_summary_btn.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        refresh_summary_btn.setToolTip(TOOLTIPSTEXT.REFRESH_SUMMARY)

        apply_btn = QPushButton("Apply to this position")
        apply_btn.setToolTip(TOOLTIPSTEXT.APPLY_BTN)
        apply_btn.setMinimumHeight(28)
        apply_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        apply_all_btn = QPushButton("Apply to all positions")
        apply_all_btn.setToolTip(TOOLTIPSTEXT.APPLY_ALL_BTN)
        apply_all_btn.setMinimumHeight(28)
        apply_all_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        exit_btn = QPushButton("Exit")
        exit_btn.setToolTip(TOOLTIPSTEXT.EXIT_BTN)
        exit_btn.setMinimumHeight(28)
        exit_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        side_v.addWidget(refresh_summary_btn)
        side_v.addWidget(apply_btn)
        side_v.addWidget(apply_all_btn)
        side_v.addWidget(exit_btn)

        refresh_summary_btn.clicked.connect(self.refresh_summary)
        apply_btn.clicked.connect(self._on_apply)
        apply_all_btn.clicked.connect(self._on_apply_all)
        exit_btn.clicked.connect(win.close)

    def refresh_summary(self):
        """Refresh the outlier configuration summary display."""
        text_lines = []
        text_lines.append("STATIC THRESHOLD RULES:")
        text_lines += self.threshold_tab.summary_lines()
        text_lines.append("")
        text_lines.append("SLIDING WINDOWS:")
        text_lines += self.sliding_tab.summary_lines()
        text_lines.append("")
        text_lines.append("CLOSE MASKS:")
        text_lines += self._close_mask_lines()
        self.summary_box.setPlainText("\n".join(text_lines))

    def _close_mask_lines(self):
        """Summary lines for the close-mask settings and the flags set so far."""
        close_tab = getattr(self.main_window, "_close_mask_tab", None)
        settings = close_tab.settings_for_rules() if close_tab else None
        if settings is None:
            lines = ["  (not used - see the Close masks tab)"]
        else:
            masks = (
                "all"
                if settings.masks is None
                else ", ".join(str(m) for m in settings.masks)
            )
            lines = [
                f"  distance <= {settings.distance:g} px | masks: {masks}"
            ]

        summary = close_mask_summary(
            getattr(self.main_window, "filtered_df", None)
        )
        if summary["flagged"] or summary["reviewed"]:
            lines.append(
                f"  {summary['flagged']} flagged time point(s) in "
                f"{summary['identifications']} identification(s), "
                f"{summary['reviewed']} reviewed"
            )
        return lines

    def _on_apply(self):
        """Apply the selected outlier rules and update downstream UI state."""
        main_window = self.main_window
        step = self.win.set_progress_step
        setp = self.win.set_progress
        finish = self.win.finish_progress

        setp(0)
        QApplication.processEvents()

        try:
            pack = snapshot_outlier_ui_to_pack(main_window)
            payload = pack.to_dict()

            n_rules = len(pack.rules)
            n_sliding = len(pack.sliding_windows)
            pre_steps = 3
            sync_steps = 7
            total = pre_steps + n_rules + n_sliding + sync_steps
            i = 0

            def advance():
                """Advance the apply-progress dialog by one step."""
                nonlocal i
                i += 1
                step(i, total)
                QApplication.processEvents()

            advance()

            main_window._last_outlier_rules = payload
            save_outlier_rules_to_disk(main_window, payload)
            advance()

            df_all = getattr(main_window, "filtered_df", None)
            advance()

            outcol = "Outlier_detection"
            if df_all is not None and not df_all.empty:
                df_all[outcol] = "OK"
                df_all = run_threshold_rules(
                    df_all, pack, outcol=outcol, on_rule_done=advance
                )
                df_all = run_sliding_windows(
                    df_all, pack, outcol=outcol, on_window_done=advance
                )
                main_window.filtered_df = df_all

                update_outlier_detection_in_track_df(
                    main_window, outcol=outcol
                )
                advance()
                ensure_current_df_subset(main_window)
                advance()
                update_unique_outliers_ids(main_window, outcol=outcol)
                if pack.close_masks is not None:
                    apply_close_mask_detection(
                        main_window,
                        pack.close_masks.distance,
                        pack.close_masks.masks,
                    )
                advance()
                _update_outlier_list(main_window)
                advance()
                main_window.Outliers.setChecked(True)
                advance()
                update_outlier_marker(main_window)
                advance()
                auto_select_first_item(main_window)
                advance()

            while i < total:
                advance()

        finally:
            finish()
            self.win.close()

    def _on_apply_all(self):
        """Apply the selected outlier rules to every already quantified position."""
        main_window = self.main_window
        step = self.win.set_progress_step
        setp = self.win.set_progress
        finish = self.win.finish_progress

        setp(0)
        QApplication.processEvents()

        try:
            pack = snapshot_outlier_ui_to_pack(main_window)
            payload = pack.to_dict()
            main_window._last_outlier_rules = payload
            save_outlier_rules_to_disk(main_window, payload)

            def on_position(i, total, _pnum):
                """Advance the progress bar to position i of total."""
                step(i, total)
                QApplication.processEvents()

            processed, skipped = run_outlier_pipeline_for_all_positions(
                main_window, pack, on_position=on_position
            )
            save_track_df(main_window)

            self._refresh_current_position_view(main_window)
            self._show_apply_all_summary(processed, skipped)
        finally:
            finish()
            self.win.close()

    @staticmethod
    def _refresh_current_position_view(main_window) -> None:
        """Reload filtered_df from track_df for the on-screen position and refresh the outlier UI."""
        cur_pos = getattr(main_window, "current_position_number", None)
        track_df = getattr(main_window, "track_df", None)
        if (
            cur_pos is None
            or track_df is None
            or "Position" not in track_df.columns
        ):
            return

        main_window.filtered_df = track_df[
            track_df["Position"] == cur_pos
        ].copy()
        if "Identification" in main_window.filtered_df.columns:
            main_window.unique_ids = main_window.filtered_df[
                "Identification"
            ].unique()

        ensure_current_df_subset(main_window)
        update_unique_outliers_ids(main_window)
        _update_outlier_list(main_window)
        update_outlier_marker(main_window)
        auto_select_first_item(main_window)

    @staticmethod
    def _show_apply_all_summary(
        processed: list[int], skipped: list[int]
    ) -> None:
        """Popup listing which positions were detected and which were skipped."""

        def fmt(nums):
            """Format position numbers as a comma separated 'pNNNN' list."""
            return ", ".join(f"p{n:04d}" for n in nums)

        lines = [f"Applied to {len(processed)} position(s): {fmt(processed)}."]
        if skipped:
            lines.append(
                f"Skipped {len(skipped)} position(s) - not yet quantified: "
                f"{fmt(skipped)}."
            )

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Apply to all positions")
        msg.setText("\n\n".join(lines))
        msg.exec_()
