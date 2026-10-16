"""Dialog for the click-segment tool settings, stored per experiment."""

from __future__ import annotations

import logging

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from SECQUOIA.config import LINKS, STYLE, TOOLTIPSTEXT
from SECQUOIA.core.project_state import save_project_state
from SECQUOIA.core.segmentation.click_segment import (
    CELLPROB_RANGE,
    DIAMETER_RANGE,
    EDGE_SENSITIVITY_RANGE,
    MASK_SIZE_RANGE,
    SMOOTHING_RANGE,
    ClickSettings,
    ClickSettingsSet,
)
from SECQUOIA.gui.common.ui_utils import make_help_button

LOG = logging.getLogger(__name__)

__all__ = ["ClickSegmentDialog", "open_click_segment_dialog"]

_SPIN_WIDTH = 110


def _spin(bounds, step, tip) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(*bounds)
    box.setSingleStep(step)
    box.setDecimals(2)
    box.setToolTip(tip)
    return box


class ClickSegmentDialog(QDialog):
    """Edit the click-segment values for all masks, with optional per-mask overrides."""

    def __init__(
        self,
        parent: QWidget | None,
        current: ClickSettingsSet,
        n_masks: int = 1,
    ):
        super().__init__(parent)
        self.setWindowTitle("Click-segment settings")
        self.setStyleSheet(f"""
        QWidget {{
            font-size: {STYLE.FONT_SIZE_outlier}pt;
            font-family: "{STYLE.FONT_outlier}";
        }}
        """)

        self.smoothing = _spin(
            SMOOTHING_RANGE, 0.5, TOOLTIPSTEXT.CLICK_SMOOTHING
        )
        self.edge = _spin(
            EDGE_SENSITIVITY_RANGE, 0.1, TOOLTIPSTEXT.CLICK_EDGE_SENSITIVITY
        )
        self.cellprob = _spin(CELLPROB_RANGE, 0.5, TOOLTIPSTEXT.CLICK_CELLPROB)
        self.diameter = QSpinBox()
        self.diameter.setRange(0, int(DIAMETER_RANGE[1]))
        self.diameter.setSpecialValueText("Auto")
        self.diameter.setSuffix(" px")
        self.diameter.setToolTip(TOOLTIPSTEXT.CLICK_DIAMETER)
        self.mask_size = QSpinBox()
        self.mask_size.setRange(*MASK_SIZE_RANGE)
        self.mask_size.setSuffix(" px")
        self.mask_size.setToolTip(TOOLTIPSTEXT.CLICK_MASK_SIZE)
        for box in (
            self.smoothing,
            self.edge,
            self.cellprob,
            self.diameter,
            self.mask_size,
        ):
            box.setFixedWidth(_SPIN_WIDTH)
        self.target = QComboBox()
        self.target.setToolTip(TOOLTIPSTEXT.CLICK_TARGET)
        self.inherit = QCheckBox("Use the All masks values")
        self.inherit.setToolTip(TOOLTIPSTEXT.CLICK_INHERIT)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.addRow("Apply to", self.target)
        form.addRow("", self.inherit)
        form.addRow("Smoothing (watershed)", self.smoothing)
        form.addRow("Edge sensitivity (watershed)", self.edge)
        form.addRow("Cell probability threshold (Cellpose)", self.cellprob)
        form.addRow("Diameter (Cellpose)", self.diameter)
        form.addRow("Mask size (both methods)", self.mask_size)

        reset = QPushButton("Reset to defaults")
        ok = QPushButton("OK")
        cancel = QPushButton("Cancel")
        reset.setAutoDefault(False)
        cancel.setAutoDefault(False)
        ok.setDefault(True)
        reset.clicked.connect(self._reset)
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addWidget(reset)
        buttons.addStretch(1)
        buttons.addWidget(ok)
        buttons.addWidget(cancel)

        help_row = QHBoxLayout()
        help_row.setContentsMargins(0, 0, 0, 0)
        help_row.addStretch(1)
        help_row.addWidget(
            make_help_button(
                self, TOOLTIPSTEXT.CLICK_HELP, LINKS.GITHUB_VIEWER
            )
        )

        layout = QVBoxLayout(self)
        layout.addLayout(help_row)
        layout.addLayout(form)
        layout.addLayout(buttons)

        self.load_set(current, n_masks)
        self.target.currentIndexChanged.connect(self._switch)
        self.inherit.toggled.connect(self._inherit_toggled)

    def load_set(self, current: ClickSettingsSet, n_masks: int = 1) -> None:
        self._default = current.default
        self._overrides = dict(current.overrides)
        self._entry = None
        self.target.blockSignals(True)
        self.target.clear()
        self.target.addItem("All masks (default)", None)
        for idx in range(1, max(1, int(n_masks)) + 1):
            self.target.addItem(f"Mask {idx}", idx)
        self.target.blockSignals(False)
        self._show_entry(None)

    def _show_entry(self, key: int | None) -> None:
        """Fill the controls for 'All masks' (None) or one mask."""
        self._entry = key
        inherited = key is not None and key not in self._overrides
        cfg = (
            self._default if key is None or inherited else self._overrides[key]
        )
        self.inherit.setVisible(key is not None)
        self.inherit.blockSignals(True)
        self.inherit.setChecked(inherited)
        self.inherit.blockSignals(False)
        self.load(cfg)
        self._set_values_enabled(not inherited)

    def _set_values_enabled(self, enabled: bool) -> None:
        for box in (
            self.smoothing,
            self.edge,
            self.cellprob,
            self.diameter,
            self.mask_size,
        ):
            box.setEnabled(enabled)

    def _store(self) -> None:
        """Keep the controls' values for the entry being shown."""
        if self._entry is None:
            self._default = self.values()
        elif self.inherit.isChecked():
            self._overrides.pop(self._entry, None)
        else:
            self._overrides[self._entry] = self.values()

    def _switch(self, _index: int) -> None:
        self._store()
        self._show_entry(self.target.currentData())

    def _inherit_toggled(self, inherited: bool) -> None:
        if self._entry is None:
            return
        if inherited:
            self._overrides.pop(self._entry, None)
            self.load(self._default)
        self._set_values_enabled(not inherited)

    def load(self, settings: ClickSettings) -> None:
        """Show `settings` in the controls."""
        self.smoothing.setValue(settings.smoothing)
        self.edge.setValue(settings.edge_sensitivity)
        self.cellprob.setValue(settings.cellprob_threshold)
        self.diameter.setValue(int(settings.diameter or 0))
        self.mask_size.setValue(settings.mask_size)

    def _reset(self) -> None:
        """'All masks' goes back to the built-in values; a mask goes back to the shared ones."""
        if self._entry is None:
            self.load(ClickSettings())
        else:
            self.inherit.setChecked(True)

    def settings(self) -> ClickSettingsSet:
        self._store()
        return ClickSettingsSet(self._default, dict(self._overrides))

    def values(self) -> ClickSettings:
        return ClickSettings.from_dict(
            {
                "smoothing": self.smoothing.value(),
                "edge_sensitivity": self.edge.value(),
                "cellprob_threshold": self.cellprob.value(),
                "diameter": self.diameter.value() or None,
                "mask_size": self.mask_size.value(),
            }
        )


def _as_set(value) -> ClickSettingsSet:
    """Accept a plain `ClickSettings` too, as the shared default."""
    if isinstance(value, ClickSettingsSet):
        return value
    if isinstance(value, ClickSettings):
        return ClickSettingsSet(default=value)
    return ClickSettingsSet()


def _mask_count(main_window: QWidget) -> int:
    try:
        return max(1, int(getattr(main_window, "n_masks", 1) or 1))
    except (TypeError, ValueError):
        return 1


def open_click_segment_dialog(main_window: QWidget) -> None:
    """Show the settings dialog and save OK'd values into the experiment's metadata."""
    current = _as_set(getattr(main_window, "click_segment_settings", None))
    dialog = getattr(main_window, "_click_segment_dialog", None)
    if dialog is None:
        dialog = ClickSegmentDialog(
            main_window, current, _mask_count(main_window)
        )
        dialog.setModal(False)

        def _apply() -> None:
            main_window.click_segment_settings = dialog.settings()
            save_project_state(main_window)
            LOG.info(
                "[click-segment] settings: %s",
                main_window.click_segment_settings,
            )

        dialog.accepted.connect(_apply)
        main_window._click_segment_dialog = dialog
    else:
        dialog.load_set(current, _mask_count(main_window))
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
