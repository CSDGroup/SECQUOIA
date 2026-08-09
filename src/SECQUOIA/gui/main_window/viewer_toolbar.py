"""Channel/mask dropdowns and assembly of each viewer's header toolbar."""

import contextlib
import functools
import logging

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from SECQUOIA.config import STYLE, TOOLTIPSTEXT
from SECQUOIA.gui.cell_inspector.integration import notify_cell_inspector
from SECQUOIA.gui.common.ui_utils import (
    TOOLBAR_ICON_PX,
    dpi_icon_size,
)

LOG = logging.getLogger(__name__)


class ViewerToolbar:
    """Channel/mask combo boxes and per viewer header row assembly."""

    @staticmethod
    def _select_by_data(combo: QComboBox, target_val: int) -> bool:
        """Select the first combo-box item whose data matches target_val."""
        for idx in range(combo.count()):
            if combo.itemData(idx) == target_val:
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)
                return True
        return False

    def _style_combo_and_label(self, combo: QComboBox, label: QLabel) -> None:
        """Apply the default 13px style, then try to switch to the configured FONT_SIZE."""
        combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        combo.setStyleSheet("QComboBox{font-size:13px; padding:0 4px;}")
        label.setStyleSheet("QLabel{font-size:13px;}")
        try:
            combo.setStyleSheet(
                f"QComboBox{{font-size:{STYLE.FONT_SIZE}px; padding:0 4px;}}"
            )
            label.setStyleSheet(f"QLabel{{font-size:{STYLE.FONT_SIZE}px;}}")
        except (RuntimeError, AttributeError, TypeError):
            combo.setStyleSheet("")
            label.setStyleSheet("")

    def _build_channel_combo(self) -> tuple[QWidget, QComboBox]:
        """Build the CH: label + dropdown container for a viewer row."""
        ch_container = QWidget()
        ch_box = QHBoxLayout(ch_container)
        ch_box.setContentsMargins(0, 0, 0, 0)
        ch_box.setSpacing(6)

        ch_label = QLabel("CH:")
        ch_label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        ch_box.addWidget(ch_label)

        ch_combo = QComboBox()
        ch_combo.setMinimumContentsLength(6)
        ch_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        ch_combo.setToolTip(TOOLTIPSTEXT.CH_VIEWER)
        for channel in self.ids_channels:
            ch_combo.addItem(channel[1:], channel[1:])
        ch_box.addWidget(ch_combo)

        self._style_combo_and_label(ch_combo, ch_label)
        return ch_container, ch_combo

    def _build_mask_combo(self, m_n: int) -> tuple[QWidget, QComboBox]:
        """Build the M: label + dropdown container for a viewer row."""
        m_container = QWidget()
        m_box = QHBoxLayout(m_container)
        m_box.setContentsMargins(0, 0, 0, 0)
        m_box.setSpacing(6)

        m_label = QLabel("M:")
        m_label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        m_box.addWidget(m_label)

        m_combo = QComboBox()
        m_combo.setMinimumContentsLength(6)
        m_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        m_combo.setToolTip(TOOLTIPSTEXT.M_VIEWER)
        m_combo.addItem("ALL", 0)
        for i in range(1, m_n + 1):
            m_combo.addItem(str(i), i)
        m_box.addWidget(m_combo)

        self._style_combo_and_label(m_combo, m_label)
        return m_container, m_combo

    @staticmethod
    def _find_header_row(outer_vbox) -> QWidget | None:
        """Return the widget named "header_row" in a viewer wrapper's layout.

        Falls back to the layout's first item if no widget carries that name.
        """
        for i in range(outer_vbox.count()):
            w = outer_vbox.itemAt(i).widget()
            if isinstance(w, QWidget) and w.objectName() == "header_row":
                return w
        return outer_vbox.itemAt(0).widget()

    def _build_eye_button(self, row_idx: int, row: QWidget) -> QPushButton:
        """Build the show/hide toggle button placed at the end of a viewer row."""
        eye_btn = QPushButton()
        eye_btn.setCheckable(True)
        eye_btn.setFocusPolicy(Qt.NoFocus)
        eye_btn.setMinimumSize(20, 20)
        eye_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        eye_btn.setFlat(True)
        eye_btn.setStyleSheet("""
            QPushButton {
                border: none;
                background: transparent;
                padding: 0;
            }
            QPushButton:hover { background: transparent; }
            QPushButton:checked { background: transparent; }
            """)

        eye_btn.setToolTip(TOOLTIPSTEXT.EYEV)
        eye_btn.setIcon(self._eye_icon(visible=True))
        eye_btn.setIconSize(dpi_icon_size(eye_btn, TOOLBAR_ICON_PX))

        eye_btn.setChecked(True)
        eye_btn.toggled.connect(
            lambda checked, idx=row_idx: self._set_viewer_minimized(
                idx, minimized=(not checked)
            )
        )

        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            row.parent()._eye_btn = eye_btn

        return eye_btn

    def _on_mask_combo_changed(self, index, v, combo, ts, oc) -> None:
        """Handle mask-dropdown changes by updating segmentation visibility, tool state, and opacity UI."""
        if v is None or index < 0:
            return
        m_val = combo.itemData(index)
        if m_val == 0:
            self._set_all_segs_visible(v)
        else:
            self._set_active_seg_only(v, int(m_val))
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            self._enforce_tool_state_for_mask(v, ts)

        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            self._sync_opacity_ui_from_mask(v, oc)

    def _on_channel_combo_changed(
        self, index, v, chc, mc, cc, ts, oc, ch_n
    ) -> None:
        """Handle channel-dropdown changes by updating visible channels, mask state, contrast UI, and opacity UI."""
        if v is None or index < 0:
            return
        channel = chc.itemData(index)
        if channel is None:
            return

        self._set_active_channel_only(v, f"w{channel}")

        picked = self._select_by_data(chc, channel) if ch_n else False
        m_idx = mc.currentIndex()
        if not picked:
            self._set_all_segs_visible(v)
        else:
            self._set_active_seg_only(v, int(m_idx))

        self._sync_contrast_ui_from_layer(v, cc)
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            self._enforce_tool_state_for_mask(v, ts)

        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            self._sync_opacity_ui_from_mask(v, oc)

    def _apply_default_row_selection(
        self,
        row_idx: int,
        viewer,
        ch_combo: QComboBox,
        m_combo: QComboBox,
        tool_strip,
        opacity_container,
        contrast_container,
        ch_n: int,
        m_n: int,
    ) -> None:
        """Pick default channel/mask selections for a freshly (re)built viewer row and sync dependent UI."""
        if viewer is None:
            return

        if ch_n and ch_combo.count():
            default_ch = None
            for channel in self.ids_channels:
                lyr = self._find_layer_by_name(
                    viewer, self._ch_layer_name_for(channel)
                )
                if lyr is not None:
                    if getattr(lyr, "visible", False):
                        default_ch = channel[1:]
                        break
                    if default_ch is None:
                        default_ch = channel[1:]
            if default_ch is not None:
                for idx in range(ch_combo.count()):
                    if ch_combo.itemData(idx) == default_ch:
                        ch_combo.blockSignals(True)
                        ch_combo.setCurrentIndex(idx)
                        ch_combo.blockSignals(False)
                        self._set_active_channel_only(viewer, f"w{default_ch}")
                        break

        has_seg = any(
            self._is_seg_layer(lyr) for lyr in getattr(viewer, "layers", [])
        )
        desired_m = (
            2 if (row_idx == 1 and m_n >= 2) else (1 if m_n >= 1 else 0)
        )

        if m_combo.count():
            if desired_m > 0 and self._select_by_data(m_combo, desired_m):
                with contextlib.suppress(TypeError, ValueError):
                    self._set_active_seg_only(viewer, desired_m)
            else:
                self._select_by_data(m_combo, 0)
                try:
                    if has_seg:
                        self._set_all_segs_visible(viewer)
                except (TypeError, ValueError):
                    pass

            with contextlib.suppress(TypeError, ValueError):
                self._enforce_tool_state_for_mask(viewer, tool_strip)
            with contextlib.suppress(TypeError, ValueError):
                self._sync_opacity_ui_from_mask(viewer, opacity_container)

        self._sync_contrast_ui_from_layer(viewer, contrast_container)
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            contrast_container._armed = True
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            self._enforce_tool_state_for_mask(viewer, tool_strip)

    def _build_row_controls(
        self, row_idx: int, wrapper: QWidget, ch_n: int, m_n: int
    ) -> None:
        """Rebuild the CH/M dropdowns, contrast/opacity/tool controls, and eye button for one viewer row."""
        outer_vbox = wrapper.layout()
        if outer_vbox is None or outer_vbox.count() == 0:
            return

        row = self._find_header_row(outer_vbox)
        if row is None:
            return

        hbox = row.layout()
        if hbox is None:
            return

        while hbox.count():
            item = hbox.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        viewer = self._viewer_for_row_index(row_idx)

        ch_container, ch_combo = self._build_channel_combo()
        self.channel_combos.append(ch_combo)

        m_container, m_combo = self._build_mask_combo(m_n)
        self.mask_combos.append(m_combo)

        contrast_container = self._build_contrast_controls(viewer)
        contrast_container.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        self.contrast_widgets.append(contrast_container)

        opacity_container = self._build_opacity_control(viewer)
        self.opacity_widgets.append(opacity_container)

        tool_strip = self._make_viewer_tools(viewer)
        self.tool_strips.append(tool_strip)

        ch_combo.currentIndexChanged.connect(
            functools.partial(
                self._on_channel_combo_changed,
                v=viewer,
                chc=ch_combo,
                mc=m_combo,
                cc=contrast_container,
                ts=tool_strip,
                oc=opacity_container,
                ch_n=ch_n,
            )
        )
        m_combo.currentIndexChanged.connect(
            functools.partial(
                self._on_mask_combo_changed,
                v=viewer,
                combo=m_combo,
                ts=tool_strip,
                oc=opacity_container,
            )
        )

        hbox.addWidget(m_container)
        if m_n:
            hbox.addSpacing(1)

        hbox.addWidget(opacity_container)
        hbox.addSpacing(1)

        hbox.addWidget(ch_container)
        if ch_n:
            hbox.addSpacing(1)

        hbox.addWidget(contrast_container)
        hbox.addSpacing(1)

        hbox.addWidget(tool_strip)
        hbox.addStretch(1)

        eye_btn = self._build_eye_button(row_idx, row)
        hbox.addSpacing(1)
        hbox.addWidget(eye_btn)

        tool_strip._btn_brush.setToolTip(TOOLTIPSTEXT.BTN_BRUSH)
        tool_strip._btn_erase.setToolTip(TOOLTIPSTEXT.BTN_ERASE)
        tool_strip._btn_pan.setToolTip(TOOLTIPSTEXT.BTN_PAN)

        self._apply_default_row_selection(
            row_idx,
            viewer,
            ch_combo,
            m_combo,
            tool_strip,
            opacity_container,
            contrast_container,
            ch_n,
            m_n,
        )

    def update_channel_mask_dropdowns(self) -> None:
        """Rebuild the CH/M dropdowns, contrast, opacity, tools and eye button for every viewer row."""

        ch_n = max(0, int(getattr(self, "n_channels", 0)))
        m_n = max(0, int(getattr(self, "n_masks", 0)))

        self.channel_combos = []
        self.mask_combos = []
        self.contrast_widgets = []
        self.tool_strips = []
        self.opacity_widgets = []

        if hasattr(self, "viewer_wrappers"):
            for row_idx, wrapper in enumerate(self.viewer_wrappers):
                self._build_row_controls(row_idx, wrapper, ch_n, m_n)

            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                self._apply_minimize_layout_effects()
        notify_cell_inspector(self, "refresh_sources")
