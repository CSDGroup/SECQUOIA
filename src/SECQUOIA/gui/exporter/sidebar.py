"""Syncing sidebar controls with the active panel's state."""

import logging

from qtpy.QtGui import (
    QColor,
)
from qtpy.QtWidgets import (
    QColorDialog,
)

from SECQUOIA.config import EXPORT

LOG = logging.getLogger(__name__)


class _Sidebar:
    """Syncing sidebar controls with the active panel's state."""

    def _wire_selection_to_active(self, main_window):
        """Bind selection controls to update the active panel state."""

        def apply_ident():
            p = getattr(main_window, "active_panel", None)
            if not p:
                return
            p["ident"] = self._get_combo_text(main_window.sel_ident_combo)
            self._populate_cell_combo(main_window)
            self._load_mask_style_into_sidebar(main_window)
            self._update_single_panel_preview(main_window, p)

        def apply_channel():
            p = getattr(main_window, "active_panel", None)
            if not p:
                return
            p["chan_abs_idx"] = self._get_combo_data(
                main_window.sel_chan_combo
            )
            p["chan_label_text"] = (
                self._get_combo_text(main_window.sel_chan_combo) or "CH"
            )
            self._update_single_panel_preview(main_window, p)

        def _apply_bw():
            p = getattr(main_window, "active_panel", None)
            if not p:
                return
            b = int(main_window.sel_black_spin.value())
            w = int(main_window.sel_white_spin.value())
            if w <= b:
                w = b + 1
                with self.blocked(main_window.sel_white_spin):
                    main_window.sel_white_spin.setValue(w)
            p["black_point"], p["white_point"] = b, w
            self._update_single_panel_preview(main_window, p)

        def _apply_w():
            p = getattr(main_window, "active_panel", None)
            if not p:
                return
            b = int(main_window.sel_black_spin.value())
            w = int(main_window.sel_white_spin.value())
            if w <= b:
                b = w - 1
                with self.blocked(main_window.sel_black_spin):
                    main_window.sel_black_spin.setValue(b)
            p["black_point"], p["white_point"] = b, w
            self._update_single_panel_preview(main_window, p)

        main_window.sel_ident_combo.currentIndexChanged.connect(
            lambda *_: apply_ident()
        )
        main_window.sel_chan_combo.currentIndexChanged.connect(
            lambda *_: apply_channel()
        )
        main_window.sel_black_spin.valueChanged.connect(lambda *_: _apply_bw())
        main_window.sel_white_spin.valueChanged.connect(lambda *_: _apply_w())
        main_window.sel_gen_btn.clicked.connect(
            lambda *_: self._open_generations_dialog(
                main_window, getattr(main_window, "active_panel", {})
            )
        )

    def _load_active_into_sidebar(self, main_window):
        """Load active panel settings into sidebar widgets."""
        p = getattr(main_window, "active_panel", None)
        if not p:
            return

        with self.blocked(
            main_window.sel_ident_combo,
            main_window.sel_chan_combo,
            main_window.sel_black_spin,
            main_window.sel_white_spin,
        ):
            ident = (p.get("ident") or "").strip()
            if ident:
                idx = main_window.sel_ident_combo.findText(ident)
                if idx < 0:
                    main_window.sel_ident_combo.addItem(ident)
                    idx = main_window.sel_ident_combo.findText(ident)
                if idx >= 0:
                    main_window.sel_ident_combo.setCurrentIndex(idx)

            ch_abs = int(p.get("chan_abs_idx", 0))
            target_idx = -1
            for i in range(main_window.sel_chan_combo.count()):
                if int(main_window.sel_chan_combo.itemData(i)) == ch_abs:
                    target_idx = i
                    break
            if target_idx >= 0:
                main_window.sel_chan_combo.setCurrentIndex(target_idx)

            main_window.sel_black_spin.setValue(
                int(p.get("black_point", EXPORT.BW_DEFAULT_BLACK))
            )
            main_window.sel_white_spin.setValue(
                int(p.get("white_point", EXPORT.BW_DEFAULT_WHITE))
            )

        with self.blocked(
            main_window.adv_show_ch_lbl_chk,
            main_window.adv_ch_font_spin,
            main_window.adv_ch_label_edit,
            main_window.adv_font_combo,
            main_window.adv_mask_combo,
            main_window.sel_sisters_chk,
            main_window.adv_cell_label_chk,
            main_window.adv_cell_label_spin,
            main_window.adv_time_text_chk,
            main_window.adv_time_font_spin,
            main_window.adv_time_bar_chk,
            main_window.adv_indicator_px_spin,
            main_window.adv_scale_bar_chk,
        ):
            main_window.adv_scale_bar_chk.setChecked(
                bool(p.get("scale_bar_chk", False))
            )
            main_window.adv_show_ch_lbl_chk.setChecked(
                bool(p.get("show_ch_lbl_chk", False))
            )
            main_window.adv_ch_font_spin.setValue(
                int(p.get("ch_font_size", 14))
            )
            main_window.adv_ch_label_edit.setText(p.get("ch_label_text", ""))
            font_name = str(p.get("font_name", "") or "")
            if font_name:
                idx = main_window.adv_font_combo.findText(font_name)
                if idx < 0:
                    main_window.adv_font_combo.addItem(font_name)
                    idx = main_window.adv_font_combo.findText(font_name)
                if idx >= 0:
                    main_window.adv_font_combo.setCurrentIndex(idx)
            else:
                main_window.adv_font_combo.setCurrentIndex(0)
            self._btn_color(
                main_window.adv_ch_color_btn,
                tuple(p.get("ch_color_rgb", (255, 255, 255))),
            )

            main_window.adv_mask_combo.clear()
            main_window.adv_mask_combo.addItem("None", None)
            for i in range(self._mask_count(main_window)):
                main_window.adv_mask_combo.addItem(f"Mask {i + 1}", i)
            mask_idx_wanted = p.get("mask_idx", None)
            set_idx = 0
            for i in range(main_window.adv_mask_combo.count()):
                if main_window.adv_mask_combo.itemData(i) == mask_idx_wanted:
                    set_idx = i
                    break
            main_window.adv_mask_combo.setCurrentIndex(set_idx)
            main_window.adv_mask_combo.set_checked(p.get("shown_masks", []))
            main_window.sel_sisters_chk.setChecked(
                bool(p.get("show_sisters", False))
            )
            self._populate_cell_combo(main_window)
            main_window.adv_cell_label_chk.setChecked(
                bool(p.get("show_cell_labels", False))
            )
            main_window.adv_cell_label_spin.setValue(
                int(p.get("cell_label_size", 12))
            )

            main_window.adv_time_text_chk.setChecked(
                bool(p.get("time_text_chk", False))
            )
            main_window.adv_time_font_spin.setValue(
                int(p.get("time_font_size", 14))
            )
            self._btn_color(
                main_window.adv_time_color_btn,
                tuple(p.get("time_color_rgb", (255, 255, 255))),
            )
            main_window.adv_time_bar_chk.setChecked(
                bool(p.get("time_bar_chk", False))
            )
            main_window.adv_indicator_px_spin.setValue(
                int(p.get("indicator_px", 10))
            )
            self._btn_color(
                main_window.adv_indicator_color_btn,
                tuple(p.get("indicator_color_rgb", (255, 255, 255))),
            )
        self._load_mask_style_into_sidebar(main_window)

    def _populate_cell_combo(self, main_window):
        """List 'All' and every track of the active Tree-ID in the Cell box."""
        p = getattr(main_window, "active_panel", None) or {}
        combo = main_window.adv_cell_combo
        with self.blocked(combo):
            combo.clear()
            combo.addItem("All", None)
            for n in self._available_track_numbers_for_ident(
                main_window, p.get("ident", "")
            ):
                combo.addItem(str(n), n)

    def _picked_mask_style(self, main_window) -> dict:
        """Style of the mask (M) and cell picked in the sidebar."""
        return self._mask_style(
            getattr(main_window, "active_panel", None) or {},
            main_window.adv_mask_combo.currentData(),
            main_window.adv_cell_combo.currentData(),
        )

    def _load_mask_style_into_sidebar(self, main_window):
        """Show the picked mask and cell's style in the style controls."""
        s = self._picked_mask_style(main_window)
        with self.blocked(
            main_window.adv_mask_alpha_spin,
            main_window.adv_mask_mode_combo,
            main_window.adv_contour_px_spin,
        ):
            main_window.adv_mask_alpha_spin.setValue(float(s["mask_alpha"]))
            main_window.adv_mask_mode_combo.setCurrentText(s["mask_mode"])
            main_window.adv_contour_px_spin.setValue(int(s["contour_px"]))
        self._btn_color(
            main_window.adv_mask_color_btn, tuple(s["mask_color_rgb"])
        )
        self._btn_color(
            main_window.adv_cell_label_color_btn,
            self._picked_label_color(main_window),
        )

    def _picked_label_color(self, main_window) -> tuple:
        """Label colour of the cell picked in the sidebar."""
        return self._label_color(
            getattr(main_window, "active_panel", None) or {},
            main_window.adv_cell_combo.currentData(),
        )

    def _set_mask_style(self, main_window, key, value, target=None):
        """Store a style value for the picked cell and mask."""
        p = getattr(main_window, "active_panel", None)
        if not p:
            return
        mask_idx = target or main_window.adv_mask_combo.currentData()
        cell = main_window.adv_cell_combo.currentData()
        if mask_idx is None:
            p[key] = value
        else:
            k = str(mask_idx) if cell is None else f"{mask_idx}:{cell}"
            styles = p.get("mask_styles") or {}
            p["mask_styles"] = {**styles, k: {**styles.get(k, {}), key: value}}
        self._update_single_panel_preview(main_window, p)

    def _wire_sidebar_to_active(self, main_window):
        """Bind advanced sidebar controls to live-update active panel."""

        def _pick_color(attr_color_key, btn_widget, start=None, apply=None):
            start_tuple = self._as_rgb_tuple(
                start
                or getattr(main_window, "active_panel", {}).get(
                    attr_color_key, (255, 255, 255)
                )
                if isinstance(getattr(main_window, "active_panel", None), dict)
                else (255, 255, 255)
            )
            start_qc = QColor(*start_tuple)
            c = QColorDialog.getColor(
                start_qc, main_window.gif_window, "Choose color"
            )
            if c.isValid() and getattr(main_window, "active_panel", None):
                t = (c.red(), c.green(), c.blue())
                self._btn_color(btn_widget, t)
                if apply:
                    apply(t)
                    return
                main_window.active_panel[attr_color_key] = t
                self._update_single_panel_preview(
                    main_window, main_window.active_panel
                )

        main_window.adv_ch_color_btn.clicked.connect(
            lambda *_: _pick_color(
                "ch_color_rgb", main_window.adv_ch_color_btn
            )
        )
        main_window.adv_time_color_btn.clicked.connect(
            lambda *_: _pick_color(
                "time_color_rgb", main_window.adv_time_color_btn
            )
        )
        main_window.adv_indicator_color_btn.clicked.connect(
            lambda *_: _pick_color(
                "indicator_color_rgb", main_window.adv_indicator_color_btn
            )
        )
        main_window.adv_mask_color_btn.clicked.connect(
            lambda *_: _pick_color(
                "mask_color_rgb",
                main_window.adv_mask_color_btn,
                start=self._picked_mask_style(main_window)["mask_color_rgb"],
                apply=lambda t: self._set_mask_style(
                    main_window, "mask_color_rgb", t
                ),
            )
        )
        main_window.adv_cell_label_color_btn.clicked.connect(
            lambda *_: _pick_color(
                "label_color_rgb",
                main_window.adv_cell_label_color_btn,
                start=self._picked_label_color(main_window),
                apply=lambda t: self._set_mask_style(
                    main_window, "label_color_rgb", t, target="label"
                ),
            )
        )

        def bind_chk(qchk, key):
            qchk.toggled.connect(
                lambda v: self._sidebar_apply_bool(main_window, key, v)
            )

        def bind_spin(qspin, key):
            qspin.valueChanged.connect(
                lambda v: self._sidebar_apply_int(main_window, key, v)
            )

        bind_chk(main_window.adv_show_ch_lbl_chk, "show_ch_lbl_chk")
        bind_spin(main_window.adv_ch_font_spin, "ch_font_size")
        main_window.adv_ch_label_edit.textChanged.connect(
            lambda s: self._sidebar_apply_text(main_window, "ch_label_text", s)
        )
        main_window.adv_font_combo.currentTextChanged.connect(
            lambda s: self._sidebar_apply_text(main_window, "font_name", s)
        )

        main_window.adv_mask_combo.currentIndexChanged.connect(
            lambda *_: self._sidebar_apply_combo_mask(
                main_window, "mask_idx", main_window.adv_mask_combo
            )
        )
        main_window.adv_cell_combo.currentIndexChanged.connect(
            lambda *_: self._load_mask_style_into_sidebar(main_window)
        )
        main_window.adv_mask_alpha_spin.valueChanged.connect(
            lambda v: self._set_mask_style(main_window, "mask_alpha", float(v))
        )
        main_window.adv_mask_mode_combo.currentTextChanged.connect(
            lambda s: self._set_mask_style(main_window, "mask_mode", s)
        )
        main_window.adv_contour_px_spin.valueChanged.connect(
            lambda v: self._set_mask_style(main_window, "contour_px", int(v))
        )
        bind_chk(main_window.sel_sisters_chk, "show_sisters")
        bind_chk(main_window.adv_cell_label_chk, "show_cell_labels")
        bind_spin(main_window.adv_cell_label_spin, "cell_label_size")
        main_window.adv_mask_combo.toggled.connect(
            lambda: self._sidebar_apply_list(
                main_window,
                "shown_masks",
                main_window.adv_mask_combo.checked(),
            )
        )
        bind_chk(main_window.adv_time_text_chk, "time_text_chk")
        bind_spin(main_window.adv_time_font_spin, "time_font_size")
        bind_chk(main_window.adv_time_bar_chk, "time_bar_chk")
        bind_spin(main_window.adv_indicator_px_spin, "indicator_px")
        bind_chk(main_window.adv_scale_bar_chk, "scale_bar_chk")
        main_window.adv_scale_bar_btn.clicked.connect(
            lambda *_: self._open_scale_bar_dialog(main_window)
        )

    def _reset_advanced_settings(self, main_window, panel):
        """Put this image's Advanced settings back to defaults."""
        keep = {"ident", "chan_abs_idx", "chan_label_text", "lineage_path"}
        keep |= {"black_point", "white_point", "_buf"}
        fresh = self._panel_dict(
            panel["container"],
            panel["obj_name"],
            panel["label"],
            panel["proxy"],
        )
        panel.update({k: v for k, v in fresh.items() if k not in keep})
        if panel is getattr(main_window, "active_panel", None):
            self._load_active_into_sidebar(main_window)
        self._update_single_panel_preview(main_window, panel)

    def _sidebar_apply_bool(self, main_window, key, val):
        """Apply a boolean sidebar value to the active panel and refresh."""
        p = getattr(main_window, "active_panel", None)
        if not p:
            return
        p[key] = bool(val)
        self._update_single_panel_preview(main_window, p)

    def _sidebar_apply_int(self, main_window, key, val):
        """Apply an integer sidebar value to the active panel and refresh."""
        p = getattr(main_window, "active_panel", None)
        if not p:
            return
        p[key] = int(val)
        self._update_single_panel_preview(main_window, p)

    def _sidebar_apply_list(self, main_window, key, values):
        """Apply a list sidebar value to the active panel and refresh."""
        p = getattr(main_window, "active_panel", None)
        if not p:
            return
        p[key] = list(values)
        self._update_single_panel_preview(main_window, p)

    def _sidebar_apply_text(self, main_window, key, text):
        """Apply a text sidebar value to the active panel and refresh."""
        p = getattr(main_window, "active_panel", None)
        if not p:
            return
        p[key] = str(text or "")
        self._update_single_panel_preview(main_window, p)

    def _sidebar_apply_combo_mask(self, main_window, key, combo):
        """Pick the mask to style."""
        p = getattr(main_window, "active_panel", None)
        if not p:
            return
        p[key] = mask_idx = combo.currentData()
        shown = p.get("shown_masks", [])
        if mask_idx is not None and mask_idx not in shown:
            p["shown_masks"] = [*shown, mask_idx]
            combo.set_checked(p["shown_masks"])
        self._load_mask_style_into_sidebar(main_window)
        self._update_single_panel_preview(main_window, p)
