"""The Close masks tab: flag tracking points with a second mask close by."""

from __future__ import annotations

import logging

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from SECQUOIA.config import TOOLTIPSTEXT, CloseMaskSettings
from SECQUOIA.core.outlier_detection.close_masks import (
    apply_close_mask_detection,
    close_mask_summary,
    mask_indices_with_candidates,
)
from SECQUOIA.core.outlier_detection.recheck import _refresh_close_mask_views
from SECQUOIA.core.segmentation.close_mask_view import (
    refresh_close_mask_view_at_current,
)
from SECQUOIA.gui.common.ui_utils import (
    compact_combo,
    compact_spin,
    make_tab_scaffold,
    make_tab_title,
)
from SECQUOIA.gui.outlier.outlier_list import auto_select_first_item
from SECQUOIA.gui.outlier.reset import clear_close_mask_flags
from SECQUOIA.gui.outlier.widgets import CheckableComboBox

LOG = logging.getLogger(__name__)

__all__ = ["CloseMaskTab"]

# Width of the mask selector, the distance box and the buttons under them.
_CONTROL_WIDTH = 110


class CloseMaskTab:
    """Distance and mask selection, plus the actions that set the purple flags.

    The distance can be anything from 0 up to the tolerance quantification
    matched with: no second mask is recorded beyond it.
    """

    def __init__(self, main_window, win, m_n):
        self.main_window = main_window
        self.win = win
        main_window._close_mask_tab = self

        self.page = QWidget()
        content_v, side_v = make_tab_scaffold(self.page)

        content_v.addWidget(
            make_tab_title(
                "Flag tracking points that have a second mask close by:"
            )
        )

        self.dist_spin = QDoubleSpinBox()
        self.dist_spin.setDecimals(1)
        self.dist_spin.setSingleStep(0.5)
        # No second mask is recorded beyond the tolerance quantification
        # matched with, so that is the largest distance that can find anything.
        self.dist_spin.setRange(0.0, max(0.0, float(main_window.threshold)))
        self.dist_spin.setSuffix(" px")
        self.dist_spin.setToolTip(TOOLTIPSTEXT.CLOSE_DIST)
        compact_spin(self.dist_spin, max_w=_CONTROL_WIDTH)

        self.mask_combo = CheckableComboBox(placeholder="All")
        compact_combo(self.mask_combo, min_chars=7, max_w=_CONTROL_WIDTH)
        self.mask_combo.setToolTip(TOOLTIPSTEXT.CLOSE_MASK)
        for mask_idx in range(1, int(m_n) + 1):
            self.mask_combo.add_check_item(str(mask_idx), mask_idx)
        self.mask_combo.update_display()  # "All" until a mask is checked

        self.find_btn = QPushButton("Find")
        self.find_btn.setToolTip(TOOLTIPSTEXT.CLOSE_FIND)
        self.reset_btn = QPushButton("Reset flags")
        self.reset_btn.setToolTip(TOOLTIPSTEXT.CLOSE_RESET)
        for button in (self.find_btn, self.reset_btn):
            button.setMinimumHeight(28)
        for widget in (
            self.mask_combo,
            self.dist_spin,
            self.find_btn,
            self.reset_btn,
        ):
            widget.setFixedWidth(_CONTROL_WIDTH)

        # Mask and distance share a row; each button sits under its control.
        self.controls_grid = QGridLayout()
        self.controls_grid.setContentsMargins(0, 0, 0, 0)
        self.controls_grid.setHorizontalSpacing(8)
        self.controls_grid.setVerticalSpacing(8)
        self.controls_grid.addWidget(
            self._label("Mask", TOOLTIPSTEXT.CLOSE_MASK), 0, 0
        )
        self.controls_grid.addWidget(self.mask_combo, 0, 1)
        self.controls_grid.addWidget(
            self._label("Distance ≤", TOOLTIPSTEXT.CLOSE_DIST, gap=16), 0, 2
        )
        self.controls_grid.addWidget(self.dist_spin, 0, 3)
        self.controls_grid.addWidget(self.find_btn, 1, 1)
        self.controls_grid.addWidget(self.reset_btn, 1, 3)
        self.controls_grid.setColumnStretch(4, 1)
        content_v.addLayout(self.controls_grid)

        self.result_label = QLabel()
        self.result_label.setWordWrap(True)
        content_v.addWidget(self.result_label)
        content_v.addStretch(1)

        self.next_btn = QPushButton("Next →")
        self.next_btn.setToolTip(TOOLTIPSTEXT.NEXT_OUT)
        self.next_btn.setMinimumHeight(28)
        self.next_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        exit_btn = QPushButton("Exit")
        exit_btn.setToolTip(TOOLTIPSTEXT.EXIT_BTN)
        exit_btn.setMinimumHeight(28)
        exit_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        side_v.addWidget(self.next_btn)
        side_v.addWidget(exit_btn)

        self._show_settings_in_use()
        self.result_label.setText(self._current_result_text())

        self.find_btn.clicked.connect(lambda *_: self._find())
        self.reset_btn.clicked.connect(lambda *_: self._reset())
        exit_btn.clicked.connect(win.close)

    @staticmethod
    def _label(text: str, tooltip: str, *, gap: int = 0) -> QLabel:
        """A left-aligned caption, `gap` px away from what comes before it."""
        label = QLabel(text + ":")
        label.setToolTip(tooltip)
        label.setContentsMargins(gap, 0, 0, 0)
        label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        return label

    # Settings
    def effective_threshold(self) -> float:
        """The distance in the box; it cannot exceed the matching tolerance."""
        return float(self.dist_spin.value())

    def selected_masks(self) -> list[int] | None:
        """The checked mask indices, or ``None`` (all masks) when none is checked."""
        return self.mask_combo.selected_values() or None

    def in_use(self) -> bool:
        """True once a detection was run (or loaded) and not reset since."""
        return (
            getattr(self.main_window, "close_mask_threshold", None) is not None
        )

    def settings_for_rules(self) -> CloseMaskSettings | None:
        """The settings to save with the outlier rules, or None while unused."""
        if not self.in_use():
            return None
        return CloseMaskSettings(
            distance=self.effective_threshold(), masks=self.selected_masks()
        )

    def apply_settings(self, settings: CloseMaskSettings) -> None:
        """Show saved settings (from a rules file) and make them the ones in use."""
        self.dist_spin.setValue(settings.distance)  # capped by the range
        self.mask_combo.set_checked_values(settings.masks or [])
        self.main_window.close_mask_threshold = self.effective_threshold()
        self.main_window.close_mask_masks = self.selected_masks()
        self._remember_in_rules()
        self.result_label.setText(self._current_result_text())

    def _show_settings_in_use(self) -> None:
        """Start from the settings in use, or from the largest distance."""
        used = getattr(self.main_window, "close_mask_threshold", None)
        self.dist_spin.setValue(
            self.dist_spin.maximum() if used is None else float(used)
        )
        masks = getattr(self.main_window, "close_mask_masks", None)
        if masks:
            self.mask_combo.set_checked_values(masks)

    def _remember_in_rules(self) -> None:
        """Keep the settings with the saved rules, so a position switch repeats them."""
        settings = self.settings_for_rules()
        rules = getattr(self.main_window, "_last_outlier_rules", None)
        rules = dict(rules) if isinstance(rules, dict) else {}
        if settings is None:
            rules.pop("close_masks", None)
        else:
            rules["close_masks"] = {
                "distance": settings.distance,
                "masks": settings.masks,
            }
        self.main_window._last_outlier_rules = rules

    # Actions
    def _current_result_text(self) -> str:
        """Describe the flags already present in the data."""
        return _describe(
            close_mask_summary(getattr(self.main_window, "filtered_df", None))
        )

    def _find(self) -> None:
        """Detect the close masks and refresh the views."""
        main_window = self.main_window
        df = getattr(main_window, "filtered_df", None)
        if df is None or len(df) == 0:
            self.result_label.setText("Load data first.")
            return
        if not mask_indices_with_candidates(df.columns):
            self.result_label.setText(
                "No second-mask columns found. Run quantification first."
            )
            return

        was_in_out_view = self._in_out_view()
        n_flagged = apply_close_mask_detection(
            main_window, self.effective_threshold(), self.selected_masks()
        )
        self._remember_in_rules()
        _refresh_close_mask_views(main_window)

        if n_flagged and not was_in_out_view:
            self._show_out_view()
        else:
            self._refresh_napari_view()

        self.result_label.setText(self._current_result_text())

    def _reset(self) -> None:
        """Remove every close-mask flag (reviewed ones too) and their displays."""
        main_window = self.main_window
        clear_close_mask_flags(main_window)
        _refresh_close_mask_views(main_window)
        self._refresh_napari_view()
        self.result_label.setText("Close-mask flags removed.")

    def _refresh_napari_view(self) -> None:
        """Show, or clear, the napari highlight for the row on screen."""
        try:
            refresh_close_mask_view_at_current(self.main_window)
        except (RuntimeError, AttributeError, TypeError, ValueError) as err:
            LOG.warning("[CloseMasks] napari view refresh failed: %s", err)

    def _in_out_view(self) -> bool:
        checkbox = getattr(self.main_window, "Outliers", None)
        return bool(checkbox is not None and checkbox.isChecked())

    def _show_out_view(self) -> None:
        """Switch the list to the Out view, which now holds the flagged cases."""
        checkbox = getattr(self.main_window, "Outliers", None)
        if checkbox is None:
            return
        checkbox.setChecked(True)
        auto_select_first_item(self.main_window)


def _describe(summary: dict[str, int]) -> str:
    """One sentence for the result label."""
    flagged, reviewed = summary["flagged"], summary["reviewed"]
    if not flagged and not reviewed:
        return "No close masks flagged."
    text = (
        f"{flagged} flagged time point(s) in "
        f"{summary['identifications']} identification(s)"
    )
    return text + (f", {reviewed} reviewed." if reviewed else ".")
