"""Panel layout preset save/load and default layout reset."""

import datetime as _dt
import json
import logging
import os
from contextlib import suppress

from SECQUOIA.config import EXPORT
from SECQUOIA.utils.paths import project_analysis_dir
from SECQUOIA.utils.timing import current_t_range as _current_t_range

LOG = logging.getLogger(__name__)


class _Presets:
    """Panel layout preset save/load and default layout reset."""

    def _active_ident_token(self, main_window) -> str:
        """Return a filesystem-safe token for the active identification."""
        ident = ""
        with suppress(Exception):
            p = getattr(main_window, "active_panel", None)
            if p:
                ident = (p.get("ident") or "").strip()
            elif getattr(main_window, "panels", []):
                ident = (main_window.panels[0].get("ident") or "").strip()
        return self._safe_name(ident or "ID")

    def _settings_dir(self, main_window) -> str:
        """Return (and create) the presets directory for this dataset."""
        experiment_root = getattr(
            main_window, "experiment_root", None
        ) or getattr(main_window, "folder", os.getcwd())
        d = os.path.join(
            project_analysis_dir(main_window, experiment_root),
            "SECQUOIA_export_settings",
        )
        os.makedirs(d, exist_ok=True)
        return d

    def _rgb_to_list(self, rgb):
        """Convert an RGB tuple to a 3-element int list."""
        with suppress(Exception):
            r, g, b = rgb
            return [int(r), int(g), int(b)]
        return [255, 255, 255]

    def _serialize_state(self, main_window) -> dict:
        """Serialize global settings and panels into a JSON-ready dict."""
        state = {
            "meta": {
                "version": 2,
                "timestamp": _dt.datetime.now().isoformat(),
                "preview_w": EXPORT.PREVIEW_W,
                "preview_h": EXPORT.PREVIEW_H,
            },
            "global": {
                "start_t": int(main_window.start_t_input.value()),
                "end_t": int(main_window.end_t_input.value()),
                "crop_full_x": int(main_window.crop_full_x_input.value()),
                "crop_full_y": int(main_window.crop_full_y_input.value()),
                "tile_w": int(EXPORT.PREVIEW_W),
                "tile_h": int(EXPORT.PREVIEW_H),
            },
            "panels": [],
        }
        for p in getattr(main_window, "panels", []):
            pos = p["proxy"].pos()
            col, row = self._cell_from_point(pos)
            entry = {
                "col": int(col),
                "row": int(row),
                "chan_abs_idx": int(p.get("chan_abs_idx", 0)),
                "chan_label_text": str(p.get("chan_label_text", "CH")),
                "black_point": int(
                    p.get("black_point", EXPORT.BW_DEFAULT_BLACK)
                ),
                "white_point": int(
                    p.get("white_point", EXPORT.BW_DEFAULT_WHITE)
                ),
                "show_ch_lbl_chk": bool(p.get("show_ch_lbl_chk", False)),
                "ch_font_size": int(p.get("ch_font_size", 14)),
                "ch_color_rgb": self._rgb_to_list(
                    p.get("ch_color_rgb", (255, 255, 255))
                ),
                "ch_label_text": str(p.get("ch_label_text", "")),
                "font_name": str(p.get("font_name", "")),
                "mask_idx": (
                    None
                    if p.get("mask_idx", None) is None
                    else int(p.get("mask_idx"))
                ),
                "mask_alpha": float(p.get("mask_alpha", 0.35)),
                "mask_mode": str(p.get("mask_mode", "Full")),
                "contour_px": int(p.get("contour_px", 1)),
                "mask_color_rgb": self._rgb_to_list(
                    p.get("mask_color_rgb", (255, 255, 255))
                ),
                "time_bar_chk": bool(p.get("time_bar_chk", False)),
                "time_text_chk": bool(p.get("time_text_chk", False)),
                "time_font_size": int(p.get("time_font_size", 14)),
                "time_color_rgb": self._rgb_to_list(
                    p.get("time_color_rgb", (255, 255, 255))
                ),
                "indicator_px": int(p.get("indicator_px", 10)),
                "indicator_color_rgb": self._rgb_to_list(
                    p.get("indicator_color_rgb", (255, 255, 255))
                ),
            }
            state["panels"].append(entry)
        return state

    def _clear_all_panels(self, main_window):
        """Remove all panels from scene and reset state."""
        for p in list(getattr(main_window, "panels", [])):
            with suppress(Exception):
                main_window.scene.removeItem(p["proxy"])
        main_window.panels = []
        main_window.active_panel = None
        self._refresh_scene_rect(main_window)

    def _reset_layout_to_defaults(self, main_window):
        """Restore default tile sizes, time range, and single panel."""
        with self.blocked(main_window.tile_w_input, main_window.tile_h_input):
            main_window.tile_w_input.setValue(
                EXPORT.PREVIEW_W if EXPORT.PREVIEW_W else 200
            )
            main_window.tile_h_input.setValue(
                EXPORT.PREVIEW_H if EXPORT.PREVIEW_H else 200
            )
        EXPORT.PREVIEW_W = 200
        EXPORT.PREVIEW_H = 200

        lo, hi = 0, 9999
        try:
            t_file_min, t_file_max, _, _ = _current_t_range(main_window)
            n_frames = max(0, int(t_file_max) - int(t_file_min) + 1)
            hi = max(0, n_frames - 1)
        except (AttributeError, TypeError, ValueError):
            pass

        main_window.start_t_input.setRange(lo, hi)
        main_window.end_t_input.setRange(lo, hi)
        main_window.start_t_input.setValue(lo)
        main_window.end_t_input.setValue(hi)

        self._clear_all_panels(main_window)
        self._add_panel(main_window)
        p = main_window.active_panel
        if p:
            p.update(
                {
                    "chan_label_text": "CH",
                    "black_point": EXPORT.BW_DEFAULT_BLACK,
                    "white_point": EXPORT.BW_DEFAULT_WHITE,
                    "show_ch_lbl_chk": False,
                    "ch_font_size": 14,
                    "ch_color_rgb": (255, 255, 255),
                    "ch_label_text": "",
                    "font_name": getattr(EXPORT, "DEFAULT_FONT_NAME", "Arial"),
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
                }
            )

        self._load_active_into_sidebar(main_window)
        self._update_single_panel_preview(
            main_window, main_window.active_panel
        )
        self._refresh_scene_rect(main_window)

    def _apply_panel_saved_settings(self, main_window, panel_dict, data: dict):
        """Copy saved panel settings dict into a live panel."""
        with suppress(Exception):
            ch_abs = int(data.get("chan_abs_idx", 0))
            if not (0 <= ch_abs < len(getattr(main_window, "images", []))):
                ch_abs = 0
            panel_dict["chan_abs_idx"] = ch_abs
        panel_dict["chan_label_text"] = str(data.get("chan_label_text", "CH"))
        panel_dict["black_point"] = int(
            data.get("black_point", EXPORT.BW_DEFAULT_BLACK)
        )
        panel_dict["white_point"] = int(
            data.get("white_point", EXPORT.BW_DEFAULT_WHITE)
        )
        panel_dict["show_ch_lbl_chk"] = bool(
            data.get("show_ch_lbl_chk", False)
        )
        panel_dict["ch_font_size"] = int(data.get("ch_font_size", 14))
        panel_dict["ch_color_rgb"] = tuple(
            data.get("ch_color_rgb", [255, 255, 255])
        )
        panel_dict["ch_label_text"] = str(data.get("ch_label_text", ""))
        panel_dict["font_name"] = str(data.get("font_name", ""))

        mask_idx = data.get("mask_idx")
        if isinstance(mask_idx, int) and 0 <= mask_idx < self._mask_count(
            main_window
        ):
            panel_dict["mask_idx"] = mask_idx
        else:
            panel_dict["mask_idx"] = None

        panel_dict["mask_alpha"] = float(data.get("mask_alpha", 0.35))
        panel_dict["mask_mode"] = str(data.get("mask_mode", "Full"))
        panel_dict["contour_px"] = int(data.get("contour_px", 1))
        panel_dict["mask_color_rgb"] = tuple(
            data.get("mask_color_rgb", [255, 255, 255])
        )
        panel_dict["time_bar_chk"] = bool(data.get("time_bar_chk", False))
        panel_dict["time_text_chk"] = bool(data.get("time_text_chk", False))
        panel_dict["time_font_size"] = int(data.get("time_font_size", 14))
        panel_dict["time_color_rgb"] = tuple(
            data.get("time_color_rgb", [255, 255, 255])
        )
        panel_dict["indicator_px"] = int(data.get("indicator_px", 10))
        panel_dict["indicator_color_rgb"] = tuple(
            data.get("indicator_color_rgb", [255, 255, 255])
        )

    def _apply_state(self, main_window, state: dict):
        """Apply a saved preset state to rebuild panels."""
        g = state.get("global", {})
        with suppress(Exception):
            main_window.start_t_input.setValue(
                int(g.get("start_t", main_window.start_t_input.value()))
            )
        with suppress(Exception):
            main_window.end_t_input.setValue(
                int(g.get("end_t", main_window.end_t_input.value()))
            )
        with suppress(Exception):
            main_window.crop_full_x_input.setValue(
                int(
                    g.get("crop_full_x", main_window.crop_full_x_input.value())
                )
            )
        with suppress(Exception):
            main_window.crop_full_y_input.setValue(
                int(
                    g.get("crop_full_y", main_window.crop_full_y_input.value())
                )
            )
        with suppress(Exception):
            main_window.tile_w_input.setValue(
                int(g.get("tile_w", EXPORT.PREVIEW_W))
            )
            main_window.tile_h_input.setValue(
                int(g.get("tile_h", EXPORT.PREVIEW_H))
            )

        self._clear_all_panels(main_window)
        for panel_data in state.get("panels", []):
            col = int(panel_data.get("col", 0))
            row = int(panel_data.get("row", 0))
            pos = self._point_from_cell((max(0, col), max(0, row)))
            self._add_panel(main_window, pos)
            new_panel = main_window.panels[-1]
            self._apply_panel_saved_settings(
                main_window, new_panel, panel_data
            )
            self._update_single_panel_preview(main_window, new_panel)

        if getattr(main_window, "panels", []):
            self._set_active_panel(main_window, main_window.panels[0])
        self._update_all_previews(main_window)
        self._refresh_scene_rect(main_window)

    def _refresh_presets_list(self, main_window):
        """Refresh the presets combo with available JSON files."""
        d = self._settings_dir(main_window)
        files = []
        with suppress(OSError):
            for fn in os.listdir(d):
                if fn.lower().endswith(".json"):
                    files.append(fn)
        files.sort()
        cb = main_window.preset_list_combo
        with self.blocked(cb):
            cb.clear()
            for fn in files:
                name = os.path.splitext(fn)[0]
                cb.addItem(name, os.path.join(d, fn))

    def _save_current_preset_clicked(self, main_window):
        """Save current layout/settings to a JSON preset file."""
        name = main_window.preset_name_edit.text().strip()
        if not name:
            self._warn(
                main_window, "Save Settings", "Please enter a preset name."
            )
            return
        data = self._serialize_state(main_window)
        d = self._settings_dir(main_window)
        safe = self._safe_name(name)
        path = os.path.join(d, f"{safe}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self._refresh_presets_list(main_window)
            idx = main_window.preset_list_combo.findText(safe)
            if idx >= 0:
                main_window.preset_list_combo.setCurrentIndex(idx)
            self._info(main_window, "Save Settings", f"Saved preset:\n{path}")
        except (OSError, TypeError, ValueError) as e:
            self._error(main_window, "Save Settings", f"Failed to save:\n{e}")

    def _load_selected_preset_clicked(self, main_window):
        """Load the selected JSON preset and apply its state."""
        cb = main_window.preset_list_combo
        path = cb.currentData()
        if not path:
            self._warn(main_window, "Load Settings", "No preset selected.")
            return
        try:
            with open(path, encoding="utf-8") as f:
                state = json.load(f)
            self._apply_state(main_window, state)
            self._info(main_window, "Load Settings", f"Loaded preset:\n{path}")
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            self._error(main_window, "Load Settings", f"Failed to load:\n{e}")
