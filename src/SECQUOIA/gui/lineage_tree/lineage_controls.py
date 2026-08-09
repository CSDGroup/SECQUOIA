"""The Feature / Mask / Channel selector bar above the lineage heatmap."""

from __future__ import annotations

from qtpy import QtCore, QtWidgets
from qtpy.QtWidgets import QSizePolicy

from SECQUOIA.config import TOOLTIPSTEXT
from SECQUOIA.gui.common.ui_utils import use_fusion_widget_style
from SECQUOIA.gui.lineage_tree.lineage_geometry import FeatureCatalog
from SECQUOIA.utils.plotting import DEFAULT_FEATURE

__all__ = ["HeatmapControls"]

_LABEL_STYLE = "color: white;"
ALL_CHANNELS = -1


class HeatmapControls(QtWidgets.QWidget):
    """Feature / mask / channel pickers driving the lineage heatmap."""

    selectionChanged = QtCore.Signal()

    def __init__(
        self,
        catalog: FeatureCatalog,
        default_mask: int | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Build the bar for the features described by ``catalog``."""
        super().__init__(parent)
        self._catalog = catalog
        self._default_mask = default_mask

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        _, self._feature = self._add_combo(
            layout, "Feature:", TOOLTIPSTEXT.TREE_FEATURE, 12, stretch=0
        )
        self._mask_label, self._mask = self._add_combo(
            layout, "M:", TOOLTIPSTEXT.TREE_MASK, 6
        )
        self._channel_label, self._channel = self._add_combo(
            layout, "CH:", TOOLTIPSTEXT.TREE_CH, 6
        )
        layout.addStretch(1)

        self._populate_features()
        self._feature.currentIndexChanged.connect(self._on_feature_changed)
        self._mask.currentIndexChanged.connect(self._emit_changed)
        self._channel.currentIndexChanged.connect(self._emit_changed)
        self._repopulate_dependents()

    def _add_combo(
        self,
        layout: QtWidgets.QHBoxLayout,
        text: str,
        tooltip: str,
        min_chars: int,
        stretch: int = 1,
    ) -> tuple[QtWidgets.QLabel, QtWidgets.QComboBox]:
        """Add a ``label: combo`` pair to ``layout`` and return both."""
        label = QtWidgets.QLabel(text)
        label.setStyleSheet(_LABEL_STYLE)
        label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        layout.addWidget(label)

        combo = QtWidgets.QComboBox()
        use_fusion_widget_style(combo)
        combo.setToolTip(tooltip)
        combo.setStyleSheet(_LABEL_STYLE)
        combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        combo.setMinimumContentsLength(min_chars)
        layout.addWidget(combo, stretch)
        return label, combo

    def _populate_features(self) -> None:
        """Fill the feature list, preferring the default feature if present."""
        self._feature.blockSignals(True)
        self._feature.clear()
        for name in self._catalog.selectable_features():
            self._feature.addItem(name)
        index = self._feature.findText(DEFAULT_FEATURE)
        self._feature.setCurrentIndex(index if index >= 0 else 0)
        self._feature.blockSignals(False)

    def _repopulate_dependents(self) -> None:
        """Rebuild the mask and channel lists for the selected feature."""
        feature = self._feature.currentText()

        masks = self._catalog.masks_for(feature)
        self._mask.blockSignals(True)
        self._mask.clear()
        if masks:
            for mask in masks:
                self._mask.addItem(str(mask), mask)
            if self._default_mask in masks:
                self._mask.setCurrentIndex(masks.index(self._default_mask))
        else:
            self._mask.addItem("N/A", None)
        self._mask.blockSignals(False)
        self._mask.setEnabled(bool(masks))
        self._mask_label.setEnabled(bool(masks))

        has_channel = self._catalog.has_channel.get(feature, False)
        self._channel.blockSignals(True)
        self._channel.clear()
        if has_channel:
            self._channel.addItem("All", ALL_CHANNELS)
            for channel in self._catalog.channels_for(feature):
                self._channel.addItem(f"{channel:02d}", channel)
        else:
            self._channel.addItem("N/A", ALL_CHANNELS)
        self._channel.blockSignals(False)

        self._channel.setEnabled(has_channel)
        self._channel_label.setEnabled(has_channel)

    def _on_feature_changed(self, _index: int) -> None:
        """Refresh the dependent combos, then report a single change."""
        self._repopulate_dependents()
        self._emit_changed()

    def _emit_changed(self, *_args) -> None:
        """Announce that the selection changed."""
        self.selectionChanged.emit()

    def selection(self) -> tuple[str, int | None, int | None]:
        """Current ``(feature, mask, channel)``."""
        return (
            self._feature.currentText(),
            self._mask.currentData(),
            self._channel.currentData(),
        )

    def resolve(self) -> list[str]:
        """Dataframe column(s) the current selection maps to.
        Empty means "nothing to colour by", which the caller should treat as
        a request to fall back to the plain tree.
        """
        return self._catalog.resolve(*self.selection())
