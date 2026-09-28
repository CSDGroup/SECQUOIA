"""The bar for scoring the mask-to-track matching by hand (developer mode)."""

from __future__ import annotations

import logging

import pandas as pd
from qtpy.QtCore import QSignalBlocker, Qt, QTimer
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from SECQUOIA.config import TOOLTIPSTEXT
from SECQUOIA.core.ground_truth import (
    CATEGORIES,
    CORRECT,
    INCORRECT,
    MISSED,
    AnnotationStore,
    annotation_store,
    current_row,
    record_all_defaults,
    record_judgment,
    record_key,
    record_note,
    snapshot,
)
from SECQUOIA.core.segmentation.mask_selection import get_mask_indices
from SECQUOIA.gui.common.ui_utils import compact_combo
from SECQUOIA.gui.ground_truth_review import advance_if_complete

LOG = logging.getLogger(__name__)

__all__ = ["GroundTruthBar"]


_REFRESH_MS = 300

_TOOLTIPS = {
    CORRECT: TOOLTIPSTEXT.GT_CORRECT,
    INCORRECT: TOOLTIPSTEXT.GT_INCORRECT,
    MISSED: TOOLTIPSTEXT.GT_MISSED,
}


class GroundTruthBar(QWidget):
    """Mask selector, the three judgment buttons and a notes field."""

    def __init__(self, main_window, mask_idx: int | None = None) -> None:
        super().__init__()
        self.main_window = main_window
        self._preset = mask_idx
        self._picked = False
        self._key: tuple | None = None
        self._save_error = ""
        self._logged_problem = ""

        self.mask_combo = QComboBox()
        compact_combo(self.mask_combo, min_chars=3, max_w=70)
        self.mask_combo.setToolTip(TOOLTIPSTEXT.GT_MASK)
        self.mask_combo.activated.connect(self._remember_pick)
        self.mask_combo.currentIndexChanged.connect(lambda *_: self.refresh())

        fate_button = getattr(main_window, "btn_healthy", None)
        style = fate_button.styleSheet() if fate_button is not None else ""
        self.buttons: dict[str, QPushButton] = {}
        for category in CATEGORIES:
            button = QPushButton(category)
            button.setToolTip(_TOOLTIPS[category])
            button.setFocusPolicy(Qt.NoFocus)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.setMinimumHeight(24)
            button.setStyleSheet(style)
            button.clicked.connect(lambda *_, c=category: self._judge(c))
            self.buttons[category] = button

        self.notes = QLineEdit()
        self.notes.setPlaceholderText("Notes")
        self.notes.setToolTip(TOOLTIPSTEXT.GT_NOTES)
        self.notes.returnPressed.connect(self._save_note)

        controls = QHBoxLayout()
        controls.setContentsMargins(2, 2, 2, 2)
        controls.setSpacing(4)
        controls.addWidget(QLabel("Mask:"))
        controls.addWidget(self.mask_combo)
        for button in self.buttons.values():
            controls.addWidget(button, 1)
        controls.addWidget(self.notes, 2)

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setContentsMargins(2, 0, 2, 2)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(controls)
        layout.addWidget(self.status)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(_REFRESH_MS)
        self.refresh()

    def refresh(self) -> None:
        """Bring the mask list, the buttons and the line under them up to date."""
        self._update_mask_choices()
        mask_idx = self.mask_combo.currentData()
        row = current_row(self.main_window)
        try:
            store, problem = annotation_store(self.main_window), ""
        except (OSError, ValueError) as err:
            store, problem = None, f"Cannot open the annotation file: {err}"
        if store is not None:
            self._sync_defaults(store)
        problem = self._save_error or problem
        if problem and problem != self._logged_problem:
            LOG.error("[ground truth] %s", problem)
        self._logged_problem = problem

        ready = row is not None and store is not None and mask_idx is not None
        for button in self.buttons.values():
            button.setEnabled(ready)

        if not ready:
            self._sync_notes(None, None)
            self._show(problem or self._why_not_ready(store, mask_idx))
            return

        key = record_key(row, mask_idx)
        saved = store.get(key)
        self._sync_notes(key, saved)
        text = self._describe(key, snapshot(row, mask_idx), saved)
        self._show(f"{problem}\n{text}" if problem else text)

    def _sync_defaults(self, store: AnnotationStore) -> None:
        """List every tracked point and mask of the loaded position in the file."""
        df = getattr(self.main_window, "filtered_df", None)
        masks = get_mask_indices(self.main_window)
        if df is None or df.empty:
            return
        synced = getattr(self.main_window, "_gt_synced", None)
        if (
            synced is not None
            and synced[0] is store
            and synced[1] is df
            and synced[2] == masks
        ):
            return
        self.main_window._gt_synced = (store, df, masks)
        try:
            store.sync_defaults(
                df, masks, getattr(self.main_window, "threshold", None)
            )
        except OSError as err:
            self._save_error = f"Could not save the annotation file: {err}"

    def _update_mask_choices(self) -> None:
        """List the masks of the loaded experiment, keeping the chosen one."""
        masks = get_mask_indices(self.main_window)
        listed = [
            self.mask_combo.itemData(i) for i in range(self.mask_combo.count())
        ]
        if masks == listed:
            return
        preset = self._preset is not None and not self._picked
        chosen = self._preset if preset else self.mask_combo.currentData()
        with QSignalBlocker(self.mask_combo):
            self.mask_combo.clear()
            for mask_idx in masks:
                self.mask_combo.addItem(str(mask_idx), mask_idx)
            self.mask_combo.setCurrentIndex(
                max(self.mask_combo.findData(chosen), 0)
            )

    def _remember_pick(self, *_) -> None:
        """Keep the mask somebody picked, instead of the preset."""
        self._picked = True

    def _sync_notes(self, key: tuple | None, saved: dict | None) -> None:
        """Show the saved note of `key`; a note never follows to another key."""
        if key == self._key:
            return
        self._key = key
        self.notes.setText("" if saved is None else saved["notes"])

    def _show(self, text: str) -> None:
        if text != self.status.text():
            self.status.setText(text)

    @staticmethod
    def _why_not_ready(store, mask_idx) -> str:
        """Why nothing can be judged right now."""
        if store is None:
            return "Load an experiment first."
        if mask_idx is None:
            return "No mask found."
        return "No tracked point for the current cell at this time point."

    @staticmethod
    def _describe(key: tuple, assigned: dict, saved: dict | None) -> str:
        """The line stating what a click would record."""
        identification, track, t, mask_idx = key
        label = assigned["assigned_label_id"] or "none"
        distance = assigned["distance_px"]
        dist = "–" if distance is None else f"{distance:.1f} px"
        text = (
            f"Judging → Mask {mask_idx} | {identification} | Track {track} | "
            f"t={t} | assigned label {label} | dist {dist}"
        )
        if assigned["alt_label_id"]:
            text += (
                f" | next nearest: label {assigned['alt_label_id']} "
                f"({assigned['alt_distance_px']:.1f} px)"
            )
        if saved is None:
            return text
        state = "default" if pd.isna(saved["timestamp"]) else "recorded"
        return f"{text} — {state}: {saved['category']}"

    def _save(self, action, *, advance: bool = False) -> None:
        """Run `action(mask_idx)` for the chosen mask, then read the row."""
        mask_idx = self.mask_combo.currentData()
        saved = False
        if mask_idx is not None:
            try:
                action(mask_idx)
                self._save_error = ""
                saved = True
            except (OSError, ValueError) as err:
                self._save_error = f"Could not save the judgment: {err}"
        if saved and advance:
            advance_if_complete(self.main_window)
        self.notes.clearFocus()
        self.refresh()

    def _judge(self, category: str) -> None:
        """Record `category` for the mask assigned right now."""
        self._save(
            lambda mask_idx: record_judgment(
                self.main_window, mask_idx, category, self.notes.text()
            ),
            advance=True,
        )

    def accept_all(self) -> None:
        """Accept every open mask of the current point as already assigned (the X key)."""
        self._save(
            lambda _mask_idx: record_all_defaults(
                self.main_window, get_mask_indices(self.main_window)
            ),
            advance=True,
        )

    def _save_note(self) -> None:
        """Save the note with the category the row already has (Enter)."""
        self._save(
            lambda mask_idx: record_note(
                self.main_window, mask_idx, self.notes.text()
            )
        )
