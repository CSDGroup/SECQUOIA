"""Draggable preview grid layout."""

import logging
import math
from collections.abc import Iterable
from contextlib import suppress

import numpy as np
from qtpy.QtCore import (
    QPoint,
    QPointF,
    Qt,
)
from qtpy.QtGui import (
    QColor,
)
from qtpy.QtWidgets import (
    QFrame,
    QGraphicsItem,
    QGraphicsProxyWidget,
    QLabel,
    QLayout,
    QMenu,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from SECQUOIA.config import EXPORT, TOOLTIPSTEXT

LOG = logging.getLogger(__name__)


class _Panels:
    """Draggable preview grid layout."""

    class SnapProxy(QGraphicsProxyWidget):
        """Graphics proxy that keeps preview panels aligned to the layout grid."""

        def __init__(self, controller, main_window, parent=None):
            """Proxy widget that snaps to nearest free grid cell."""
            super().__init__(parent)
            self._ctrl = controller
            self._mw = main_window
            self.setFlag(QGraphicsItem.ItemIsMovable, False)
            self.setFlag(QGraphicsItem.ItemIsSelectable, True)
            self.setZValue(0)

        def itemChange(self, change, value):
            """Snap movement to free grid positions during drag."""
            if change == QGraphicsItem.ItemPositionChange and isinstance(
                value, QPointF
            ):
                return self._ctrl._nearest_free_point(
                    self._mw, value, ignore_proxy=self
                )
            return super().itemChange(change, value)

    def _fl_block_base_index(self, main_window) -> int:
        """Return base index offset for per channel image blocks."""
        n_ch = int(getattr(main_window, "n_channels", 0) or 0)
        images = getattr(main_window, "images", None)
        if not isinstance(images, dict) or n_ch <= 0 or len(images) < n_ch:
            return 0
        return len(images) - n_ch

    def _as_rgb_tuple(self, val):
        """Normalize various color inputs to an (R,G,B) tuple."""
        if isinstance(val, Iterable) and not isinstance(
            val, (str | bytes | QColor)
        ):
            with suppress(Exception):
                r, g, b = val
                return int(r), int(g), int(b)
        if isinstance(val, QColor):
            return val.red(), val.green(), val.blue()
        return (255, 255, 255)

    def _channel_identifiers(self, main_window):
        """Collect unique channel identifiers."""
        seen, out = set(), []
        n_channels = int(getattr(main_window, "n_channels", 0) or 0)
        for i in range(1, n_channels + 1):
            ident = getattr(main_window, f"FL_identifiers_{i}", None)
            if ident and ident not in seen:
                out.append(ident)
                seen.add(ident)
        if not out:
            for w in getattr(main_window, "FL_inputs", []):
                with suppress(AttributeError):
                    txt = w.text().strip()
                    if txt and txt not in seen:
                        out.append(txt)
                        seen.add(txt)
        return out or ["Channel 1"]

    def _populate_channel_combo_for_panel(self, main_window, combo):
        """Fill a channel combo with available channels for panels."""
        with self.blocked(combo):
            combo.clear()
            base = self._fl_block_base_index(main_window)
            for i, ident in enumerate(self._channel_identifiers(main_window)):
                combo.addItem(ident, base + i)
            combo.setCurrentIndex(0)

    def _mask_count(self, main_window) -> int:
        """Return the number of available mask stacks."""
        declared = max(0, int(getattr(main_window, "n_masks", 0)))
        labels = getattr(main_window, "labels", None)
        if labels is None:
            return 0
        if isinstance(labels, (list | tuple)):
            available = len(labels)
        elif isinstance(labels, np.ndarray):
            available = labels.shape[0] if labels.ndim == 4 else 1
        else:
            available = 1
        return max(0, min(declared, available))

    def _get_mask_stack(self, main_window, idx: int):
        """Return the mask stack at index or None if missing."""
        labels = getattr(main_window, "labels", None)
        if labels is None:
            return None
        if isinstance(labels, (list | tuple)):
            return labels[idx] if 0 <= idx < len(labels) else None
        if isinstance(labels, np.ndarray):
            if labels.ndim == 4:
                return labels[idx] if 0 <= idx < labels.shape[0] else None
            if labels.ndim == 3:
                return labels if idx == 0 else None
        return None

    def _populate_mask_combo_for_panel(self, main_window, combo):
        """Fill mask combo with ‘None’ and available mask stacks."""
        with self.blocked(combo):
            combo.clear()
            combo.addItem("None", None)
            for i in range(self._mask_count(main_window)):
                combo.addItem(f"Mask {i + 1}", i)
            combo.setCurrentIndex(0)

    def _ident_list(self, main_window):
        """Return available identification values."""
        with suppress(Exception):
            ids = list(getattr(main_window, "unique_ids", []))
            return (
                ids
                if ids
                else [str(getattr(main_window, "identification", "ID_1"))]
            )
        return ["ID_1"]

    def _populate_ident_combo_for_panel(self, main_window, combo):
        """Populate identification combo and select current default."""
        with self.blocked(combo):
            combo.clear()
            for s in self._ident_list(main_window):
                combo.addItem(str(s))
            with suppress(Exception):
                current_default = main_window.unique_ids[
                    main_window.current_ident_index
                ]
                idx = combo.findText(str(current_default))
                combo.setCurrentIndex(idx if idx >= 0 else 0)

    def _cell_w(self):
        """Return the horizontal grid step for one preview tile."""
        return EXPORT.PREVIEW_W + EXPORT.GRID_GAP

    def _cell_h(self):
        """Return the vertical grid step for one preview tile."""
        return EXPORT.PREVIEW_H + EXPORT.GRID_GAP

    def _cell_from_point(self, pt: QPointF):
        """Convert scene point to grid cell coords."""
        return (
            int(
                math.floor(
                    max(0, pt.x() - EXPORT.TILE_PAD) / max(1, self._cell_w())
                )
            ),
            int(
                math.floor(
                    max(0, pt.y() - EXPORT.TILE_PAD) / max(1, self._cell_h())
                )
            ),
        )

    def _point_from_cell(self, cell):
        """Convert (col,row) grid cell to scene QPointF origin."""
        c, r = cell
        return QPointF(
            EXPORT.TILE_PAD + max(0, c * self._cell_w()),
            EXPORT.TILE_PAD + max(0, r * self._cell_h()),
        )

    def _occupied_cells(self, main_window, ignore_proxy=None):
        """Return set of grid cells currently occupied by panels."""
        occ = set()
        for p in getattr(main_window, "panels", []):
            pr = p["proxy"]
            if ignore_proxy is not None and pr is ignore_proxy:
                continue
            occ.add(self._cell_from_point(pr.pos()))
        return occ

    def _nearest_free_cell(
        self, main_window, wanted_cell, ignore_proxy=None, max_search=100
    ):
        """Find nearest unoccupied cell to the desired one."""
        occ = self._occupied_cells(main_window, ignore_proxy=ignore_proxy)
        if wanted_cell not in occ:
            return wanted_cell
        c0, r0 = wanted_cell
        for radius in range(1, max_search):
            for dc in range(-radius, radius + 1):
                for dr in (-radius, radius):
                    cand = (c0 + dc, r0 + dr)
                    if cand not in occ and cand[0] >= 0 and cand[1] >= 0:
                        return cand
            for dr in range(-radius + 1, radius):
                for dc in (-radius, radius):
                    cand = (c0 + dc, r0 + dr)
                    if cand not in occ and cand[0] >= 0 and cand[1] >= 0:
                        return cand
        return wanted_cell

    def _nearest_free_point(self, main_window, pt: QPointF, ignore_proxy=None):
        """Snap a scene point to nearest free cell’s origin point."""
        return self._point_from_cell(
            self._nearest_free_cell(
                main_window,
                self._cell_from_point(pt),
                ignore_proxy=ignore_proxy,
            )
        )

    def _panel_dict(
        self, container, obj_name, label, proxy: QGraphicsProxyWidget
    ):
        """Create the initial state dictionary for a panel tile."""
        return {
            "container": container,
            "obj_name": obj_name,
            "label": label,
            "proxy": proxy,
            "ident": "",
            "chan_abs_idx": 0,
            "chan_label_text": "CH",
            "black_point": EXPORT.BW_DEFAULT_BLACK,
            "white_point": EXPORT.BW_DEFAULT_WHITE,
            "show_ch_lbl_chk": False,
            "ch_font_size": 14,
            "ch_color_rgb": (255, 255, 255),
            "ch_label_text": "",
            "font_name": "",
            "mask_idx": None,
            "mask_alpha": 0.35,
            "mask_mode": "Full",
            "contour_px": 1,
            "mask_color_rgb": (255, 255, 255),
            "time_bar_chk": False,
            "time_text_chk": False,
            "time_font_size": 14,
            "time_color_rgb": (255, 255, 255),
            "indicator_px": 10,
            "indicator_color_rgb": (255, 255, 255),
            "lineage_path": [],
            "_buf": None,
        }

    def _highlight_panel(self, panel, selected: bool):
        """Visually mark a panel as selected."""
        obj = panel.get("obj_name", "")
        if not obj:
            return

        with suppress(Exception):
            panel["proxy"].setZValue(100 if selected else 0)

        c = panel["container"]
        with suppress(Exception):
            st = c.style()
            if st:
                st.unpolish(c)

        c.setStyleSheet(
            "border: 2px solid #2e7dff;"
            if selected
            else "border: 1px solid #444;"
        )

        with suppress(Exception):
            st = c.style()
            if st:
                st.polish(c)

        with suppress(Exception):
            w, h = c.width(), c.height()
            panel["proxy"].prepareGeometryChange()
            panel["proxy"].setMinimumSize(w, h)
            panel["proxy"].setMaximumSize(w, h)
            panel["proxy"].resize(w, h)
            panel["proxy"].updateGeometry()

        c.update()
        with suppress(Exception):
            panel["proxy"].update()

    def _set_active_panel(self, main_window, panel):
        """Set active panel and sync sidebar (and B/W Inspector, if open) from its state."""
        main_window.active_panel = panel
        for p in getattr(main_window, "panels", []):
            self._highlight_panel(p, p is panel)
        self._load_active_into_sidebar(main_window)
        self._retarget_bw_inspector(main_window)
        with suppress(Exception):
            main_window.view.resetCachedContent()
            main_window.scene.update()
            main_window.view.viewport().update()

    def _adjacent_pos(
        self, ref_proxy: QGraphicsProxyWidget, side: str
    ) -> QPointF:
        """Return the scene position of a cell adjacent to ref panel."""
        base = ref_proxy.pos()
        if side == "right":
            return QPointF(max(0, base.x() + self._cell_w()), max(0, base.y()))
        if side == "left":
            return QPointF(max(0, base.x() - self._cell_w()), max(0, base.y()))
        if side == "above":
            return QPointF(max(0, base.x()), max(0, base.y() - self._cell_h()))
        return QPointF(max(0, base.x()), max(0, base.y() + self._cell_h()))

    def _side_from_click(self, container: QWidget, pos: QPoint) -> str:
        """Infer context-menu side from click."""
        w = max(1, container.width())
        h = max(1, container.height())
        x, y = pos.x(), pos.y()
        if x >= 0.6 * w:
            return "right"
        if x <= 0.4 * w:
            return "left"
        if y <= 0.5 * h:
            return "above"
        return "below"

    def _bind_panel_mouse(self, main_window, panel_dict):
        """Attach mouse handlers and context menu to a panel tile."""
        base_frame_press = panel_dict["container"].mousePressEvent

        def _on_frame_press(e):
            if e.button() == Qt.RightButton:
                click_side = self._side_from_click(
                    panel_dict["container"], e.pos()
                )
                menu = QMenu(panel_dict["container"])
                act_add = menu.addAction("Add image")
                act_add.setToolTip(TOOLTIPSTEXT.CONTEXT_ADD_ADJACENT)
                act_remove = menu.addAction("Remove")
                act_remove.setToolTip(TOOLTIPSTEXT.CONTEXT_REMOVE_TILE)
                act_reset = menu.addAction("Reset")
                act = menu.exec_(panel_dict["container"].mapToGlobal(e.pos()))
                if act is act_add:
                    pos = self._nearest_free_point(
                        main_window,
                        self._adjacent_pos(panel_dict["proxy"], click_side),
                    )
                    self._add_panel(main_window, pos)
                elif act is act_remove:
                    self._remove_panel(main_window, panel_dict)
                elif act is act_reset:
                    self._reset_layout_to_defaults(main_window)
                return
            self._set_active_panel(main_window, panel_dict)
            with suppress(Exception):
                base_frame_press(e)

        panel_dict["container"].mousePressEvent = _on_frame_press

    def _create_tile_widget(self, main_window):
        """Create a fixed size panel widget and preview label."""
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        obj_name = f"preview_panel_{id(panel)}"
        panel.setObjectName(obj_name)
        panel.setStyleSheet("border: 1px solid #444;")
        panel.setToolTip(TOOLTIPSTEXT.PREVIEW_TILE)

        root_v = QVBoxLayout(panel)
        root_v.setContentsMargins(0, 0, 0, 0)
        root_v.setSpacing(0)
        root_v.setSizeConstraint(QLayout.SetFixedSize)

        label = self.JLabelFixed(EXPORT.PREVIEW_W, EXPORT.PREVIEW_H)
        label.setObjectName(f"tile_label_{id(label)}")
        label.setToolTip(TOOLTIPSTEXT.PREVIEW_LABEL)
        label.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        root_v.addWidget(label, 1)
        panel.setFixedSize(EXPORT.PREVIEW_W, EXPORT.PREVIEW_H)
        panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        return panel, label, obj_name

    class JLabelFixed(QLabel):
        """Fixed size QLabel used as the rendered preview for a tile."""

        def __init__(self, w, h, *args, **kwargs):
            """Label with fixed size and transparent background for previews."""
            super().__init__(*args, **kwargs)
            self.setAlignment(Qt.AlignCenter)
            self.setFixedSize(w, h)
            self.setText("No preview yet")
            self.setStyleSheet("background: transparent;")
            self.setAttribute(Qt.WA_StyledBackground, False)

    def _next_free_pos(self, main_window) -> QPointF:
        """Return scene position for the next free grid slot."""
        panels = getattr(main_window, "panels", [])
        if not panels:
            return QPointF(EXPORT.TILE_PAD, EXPORT.TILE_PAD)
        occ = self._occupied_cells(main_window)
        r = 0
        while r <= 1000:
            c = 0
            while c <= 1000:
                if (c, r) not in occ:
                    return self._point_from_cell((c, r))
                c += 1
            r += 1
        return QPointF(EXPORT.TILE_PAD, EXPORT.TILE_PAD)

    def _add_panel(self, main_window, pos: QPointF = None):
        """Create, place, and register a new panel tile at position."""
        container, label, obj_name = self._create_tile_widget(main_window)
        proxy = self.SnapProxy(self, main_window)
        proxy.setWidget(container)
        with suppress(Exception):
            proxy.setCacheMode(QGraphicsItem.NoCache)
        main_window.scene.addItem(proxy)

        proxy.setPos(
            self._nearest_free_point(main_window, pos)
            if isinstance(pos, QPointF)
            else self._next_free_pos(main_window)
        )
        d = self._panel_dict(container, obj_name, label, proxy)

        txt = self._get_combo_text(
            getattr(main_window, "sel_ident_combo", None)
        )
        d["ident"] = txt or self._ident_list(main_window)[0]

        chan_combo = getattr(main_window, "sel_chan_combo", None)
        if chan_combo:
            d["chan_abs_idx"] = self._get_combo_data(chan_combo)
            d["chan_label_text"] = self._get_combo_text(chan_combo) or "CH"
        else:
            base = self._fl_block_base_index(main_window)
            d["chan_abs_idx"] = base
            d["chan_label_text"] = self._channel_identifiers(main_window)[0]

        with suppress(Exception):
            d["black_point"] = int(main_window.sel_black_spin.value())
            d["white_point"] = int(main_window.sel_white_spin.value())

        self._bind_panel_mouse(main_window, d)
        main_window.panels.append(d)
        self._update_single_panel_preview(main_window, d)
        self._set_active_panel(main_window, d)
        self._refresh_scene_rect(main_window)

    def _remove_panel(self, main_window, panel_dict):
        """Remove a panel tile and maintain a valid active panel."""
        if panel_dict in getattr(main_window, "panels", []):
            with suppress(Exception):
                main_window.scene.removeItem(panel_dict["proxy"])
            main_window.panels.remove(panel_dict)
        if not main_window.panels:
            self._add_panel(main_window)
        else:
            self._set_active_panel(main_window, main_window.panels[0])
        self._refresh_scene_rect(main_window)
