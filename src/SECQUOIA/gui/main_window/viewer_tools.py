"""Brush/erase/pan drawing-mode controls and new mask-ID creation."""

import contextlib
import logging

import numpy as np
import qtawesome as qta
from napari.layers import Labels
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QMenu,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from SECQUOIA.config import STYLE
from SECQUOIA.core.segmentation.mask_selection import (
    show_all_masks,
    show_current_mask,
)
from SECQUOIA.gui.common.ui_utils import (
    TOOLBAR_ICON_PX,
    dpi_icon_size,
)

LOG = logging.getLogger(__name__)


class ViewerTools:
    """Brush/erase/pan drawing tools and new mask-ID creation."""

    def _labels_layer_for_viewer(self, viewer) -> Labels | None:
        """Return the active (or last) Labels layer in *viewer*, or None."""
        if viewer is None:
            return None
        try:
            lyr = getattr(viewer.layers.selection, "active", None)
            if lyr is not None and isinstance(lyr, Labels):
                return lyr
        except (RuntimeError, AttributeError, TypeError):
            pass
        try:
            for lyr in reversed(list(viewer.layers)):
                if isinstance(lyr, Labels):
                    return lyr
        except (RuntimeError, AttributeError, TypeError):
            pass
        return None

    def _focus_canvas(self, viewer):
        """Force the viewer into 2D display and give its canvas keyboard focus."""
        try:
            qt_viewer = viewer.window._qt_viewer
            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                viewer.dims.ndisplay = 2
            if hasattr(qt_viewer, "canvas") and hasattr(
                qt_viewer.canvas, "native"
            ):
                qt_viewer.canvas.native.setFocus(Qt.OtherFocusReason)
        except (RuntimeError, AttributeError, TypeError):
            pass

    def _set_labels_mode(self, viewer, mode_str: str) -> None:
        """Select a Labels layer, make it editable, set its interaction mode, and refocus the canvas."""
        lyr = self._labels_layer_for_viewer(viewer)
        if lyr is None:
            LOG.warning("[tools] No Labels layer in this viewer.")
            return
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            viewer.layers.selection.active = lyr
        try:
            if hasattr(lyr, "editable"):
                lyr.editable = True
        except (RuntimeError, AttributeError, TypeError):
            pass
        try:
            from napari.layers.labels._labels_constants import (
                Mode as LabelsMode,
            )

            mapping = {
                "paint": LabelsMode.PAINT,
                "erase": LabelsMode.ERASE,
                "pan_zoom": LabelsMode.PAN_ZOOM,
            }
            lyr.mode = mapping.get(mode_str, mode_str)
        except (RuntimeError, AttributeError, TypeError):
            lyr.mode = mode_str
        self._focus_canvas(viewer)

    def _make_viewer_tools(self, viewer):
        """Build the brush, erase, and pan tool controls for a viewer."""
        container = QWidget()
        box = QHBoxLayout(container)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)

        btn_brush = QPushButton()
        btn_erase = QPushButton()
        btn_pan = QPushButton()

        for b in (btn_brush, btn_erase, btn_pan):
            b.setCheckable(True)
            b.setFocusPolicy(Qt.NoFocus)
            b.setMinimumWidth(26)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.setProperty("vtool", "1")
            b.setToolTip("")

        used_icons = False
        try:
            candidates = [
                (
                    "fa6s",
                    {
                        "brush": "paintbrush",
                        "erase": "eraser",
                        "pan": "up-down-left-right",
                    },
                ),
                (
                    "fa5s",
                    {
                        "brush": "paint-brush",
                        "erase": "eraser",
                        "pan": "arrows-alt",
                    },
                ),
                (
                    "fa",
                    {
                        "brush": "paint-brush",
                        "erase": "eraser",
                        "pan": "arrows",
                    },
                ),
                (
                    "mdi",
                    {
                        "brush": "brush",
                        "erase": "eraser",
                        "pan": "cursor-move",
                    },
                ),
            ]
            for prefix, names in candidates:
                try:
                    btn_brush.setIcon(
                        qta.icon(f"{prefix}.{names['brush']}", color="white")
                    )
                    btn_erase.setIcon(
                        qta.icon(f"{prefix}.{names['erase']}", color="white")
                    )
                    btn_pan.setIcon(
                        qta.icon(f"{prefix}.{names['pan']}", color="white")
                    )
                    for b in (btn_brush, btn_erase, btn_pan):
                        b.setIconSize(dpi_icon_size(b, TOOLBAR_ICON_PX))
                    used_icons = True
                    break
                except (RuntimeError, AttributeError, TypeError):
                    continue
            if not used_icons:
                raise RuntimeError("no qtawesome prefix available")
        except (RuntimeError, AttributeError, TypeError):
            btn_brush.setText("B")
            btn_erase.setText("E")
            btn_pan.setText("P")

        group = QButtonGroup(container)
        group.setExclusive(True)
        for b in (btn_brush, btn_erase, btn_pan):
            group.addButton(b)
        btn_pan.setChecked(True)

        def _guarded_mode_switch(on: bool, v=viewer, mode="pan_zoom"):
            """Switch label-editing tools while preventing paint or erase mode when all masks are selected."""
            if not on:
                return
            if self._mask_selection_for_viewer(v) == 0 and mode in (
                "paint",
                "erase",
            ):
                with contextlib.suppress(
                    RuntimeError, AttributeError, TypeError
                ):
                    container._btn_pan.setChecked(True)
                return
            self._set_labels_mode(v, mode)

        btn_brush.toggled.connect(
            lambda on, v=viewer: _guarded_mode_switch(on, v, "paint")
        )
        btn_erase.toggled.connect(
            lambda on, v=viewer: _guarded_mode_switch(on, v, "erase")
        )
        btn_pan.toggled.connect(
            lambda on, v=viewer: _guarded_mode_switch(on, v, "pan_zoom")
        )

        container.setStyleSheet(f"""
            QWidget#tools_container {{
                border: 1px solid #4a4a4a;
                border-radius: 8px;
                padding: 4px;
                background: transparent;
            }}
            QPushButton[vtool="1"] {{
                padding: 1px 6px;
                border: none;
                border-radius: 6px;
                background: #2b2b2b;
                color: #e6e6e6;
                font-size: {STYLE.FONT_SIZE}px;
            }}
            QPushButton[vtool="1"]:hover {{ border-color: #6a6a6a; }}
            QPushButton[vtool="1"]:checked {{
                background: #16a34a;
                color: white;
                border-color: #0e7a36;
            }}
            """)

        def _open_mask_menu(btn, pos):
            """Open the mask context menu for showing all masks or creating a new mask ID."""
            menu = QMenu(btn)
            act_show_all = menu.addAction("Show all masks (Ctrl+M)")
            act_show_current = menu.addAction(
                "Show current Tree-ID's mask (Ctrl+Shift+M)"
            )
            act_new_id = menu.addAction("Create a new mask ID (M)")
            act_show_all.triggered.connect(lambda *_: show_all_masks(self))
            act_show_current.triggered.connect(
                lambda *_: show_current_mask(self)
            )
            act_new_id.triggered.connect(
                lambda *_: self._create_new_mask_id(viewer)
            )
            menu.exec_(btn.mapToGlobal(pos))

        for b in (btn_brush, btn_erase):
            b.setContextMenuPolicy(Qt.CustomContextMenu)
            b.customContextMenuRequested.connect(
                lambda pos, bb=b: _open_mask_menu(bb, pos)
            )

        box.addWidget(btn_brush)
        box.addWidget(btn_erase)
        box.addWidget(btn_pan)
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        container._btn_brush = btn_brush
        container._btn_erase = btn_erase
        container._btn_pan = btn_pan
        container._uses_icons = used_icons
        return container

    def _create_new_mask_id(self, viewer):
        """Create a new label ID in the current segmentation layer for the given viewer."""
        show_all_masks(self)

        # Determine current segmentation layer for this viewer
        m_sel = int(self._mask_selection_for_viewer(viewer) or 0)
        sel = getattr(getattr(viewer, "layers", None), "selection", None)
        layer = getattr(sel, "active", None) if sel is not None else None
        if layer is None or not hasattr(layer, "data"):
            layer = self._get_seg_layer(viewer, m_sel if m_sel > 0 else 1)

        if layer is None:
            LOG.warning("[new mask id] No segmentation layer found.")
            return

        t = int(getattr(self, "current_time_index", 0))
        data = np.asarray(layer.data)
        max_lbl = int(np.nanmax(data[t]))
        label_id = max_lbl + 1
        layer.selected_label = int(label_id)

        v_name = (
            "viewer_1"
            if viewer is getattr(self, "viewer_1", None)
            else (
                "viewer_2"
                if viewer is getattr(self, "viewer_2", None)
                else str(viewer)
            )
        )
        LOG.info(
            "[new mask id] viewer=%s layer=%s new_label_id=%s",
            v_name,
            getattr(layer, "name", "?"),
            label_id,
        )

    def _mask_selection_for_viewer(self, viewer) -> int:
        """Return current mask id for this viewer."""
        if viewer is None:
            return 0
        try:
            vi = self.viewer_fluorescence.index(viewer)
            if hasattr(self, "mask_combos") and vi < len(self.mask_combos):
                mc = self.mask_combos[vi]
                val = mc.currentData()
                return int(val) if val is not None else 0
        except (RuntimeError, AttributeError, TypeError):
            pass
        return 0

    def _enforce_tool_state_for_mask(self, viewer, tool_strip):
        """Enable or disable Brush/Erase according to the mask selection.

        With "ALL" selected (mask id 0) there is no single target layer to draw
        into, so both are disabled and Pan is forced on.
        """
        if viewer is None or tool_strip is None:
            return
        is_all = self._mask_selection_for_viewer(viewer) == 0
        try:
            tool_strip._btn_brush.setEnabled(not is_all)
            tool_strip._btn_erase.setEnabled(not is_all)
            tool_strip._btn_pan.setEnabled(True)
            if is_all and not tool_strip._btn_pan.isChecked():
                tool_strip._btn_pan.setChecked(True)
        except (RuntimeError, AttributeError, TypeError):
            pass
