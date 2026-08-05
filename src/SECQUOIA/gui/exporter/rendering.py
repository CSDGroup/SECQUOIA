"""Per frame image composition: crop, levels, overlays, text."""

import logging
import math
from contextlib import suppress

import numpy as np
from PIL import Image, ImageDraw
from qtpy.QtGui import (
    QImage,
    QPixmap,
)

from SECQUOIA.config import EXPORT

LOG = logging.getLogger(__name__)


class _Rendering:
    """Per frame image composition: crop, levels, overlays, text."""

    def _compute_contour(
        self, mask_bool: np.ndarray, thickness: int
    ) -> np.ndarray:
        """Return a dilated edge map for a binary mask."""
        inner = np.zeros_like(mask_bool, dtype=bool)
        if mask_bool.size:
            inner[1:-1, 1:-1] = (
                mask_bool[1:-1, 1:-1]
                & mask_bool[:-2, 1:-1]
                & mask_bool[2:, 1:-1]
                & mask_bool[1:-1, :-2]
                & mask_bool[1:-1, 2:]
                & mask_bool[:-2, :-2]
                & mask_bool[:-2, 2:]
                & mask_bool[2:, :-2]
                & mask_bool[2:, 2:]
            )
        edges = mask_bool & (~inner)
        return (
            self._binary_dilate(edges, max(0, thickness - 1))
            if thickness > 1
            else edges
        )

    def _gif_safe_duration_ms(self, fps: int) -> tuple[int, bool]:
        """Return (duration_ms, was_clamped) for a GIF-safe per frame delay."""
        safe_fps = min(max(1, int(fps)), EXPORT.GIF_SAFE_MAX_FPS)
        duration_cs = max(1, round(100.0 / safe_fps))
        return duration_cs * 10, safe_fps < fps

    def _quantize_frames_per_frame(self, frames_rgb, colors=256):
        """Palette quantize frames individually for GIF saving."""
        out = []
        for fr in frames_rgb:
            out.append(
                fr.quantize(
                    colors=colors,
                    method=Image.MEDIANCUT,
                    dither=Image.FLOYDSTEINBERG,
                )
            )
        return out

    def _has_frame(self, main_window, chan_abs_idx, t_index) -> bool:
        """Check if a channel has data at a given time index."""
        with suppress(Exception):
            pres = main_window.image_present[
                main_window.ids_channels[chan_abs_idx]
            ]
            return 0 <= t_index < len(pres) and bool(pres[t_index])
        return False

    def _find_prev_next(self, main_window, chan_abs_idx, t_index):
        """Find nearest previous and next available frames for a time."""
        pres = main_window.image_present[
            main_window.ids_channels[chan_abs_idx]
        ]
        p = t_index
        while p >= 0 and not pres[p]:
            p -= 1
        n = t_index
        L = len(pres)
        while n < L and not pres[n]:
            n += 1
        return (
            p if 0 <= p < L and pres[p] else None,
            n if n < L and pres[n] else None,
        )

    def _resolve_sample_t(
        self, main_window, chan_abs_idx, t_index, policy: str
    ):
        """Map requested time to actual sample index per missing data policy."""
        if self._has_frame(main_window, chan_abs_idx, t_index):
            return t_index
        if policy == "Strict":
            return None
        prev_i, next_i = self._find_prev_next(
            main_window, chan_abs_idx, t_index
        )
        if policy == "Hold last":
            return prev_i
        if policy == "Nearest":
            if prev_i is None:
                return next_i
            if next_i is None:
                return prev_i
            return (
                prev_i if (t_index - prev_i) <= (next_i - t_index) else next_i
            )
        return None

    def _get_panel_state_for_render(self, panel):
        """Return a copy of `panel` with render fields coerced and clamped."""
        st = dict(panel)
        st["chan_abs_idx"] = int(panel.get("chan_abs_idx", 0))
        st["contour_px"] = max(1, int(panel.get("contour_px", 1)))
        st["indicator_px"] = max(4, int(panel.get("indicator_px", 10)))
        st["time_font_size"] = int(panel.get("time_font_size", 14))
        st["ch_font_size"] = int(panel.get("ch_font_size", 14))
        st["mask_alpha"] = float(panel.get("mask_alpha", 0.35))
        st["mask_color"] = tuple(panel.get("mask_color_rgb", (255, 255, 255)))
        st["time_color"] = tuple(panel.get("time_color_rgb", (255, 255, 255)))
        st["ch_color"] = tuple(panel.get("ch_color_rgb", (255, 255, 255)))
        st["indicator_color"] = tuple(
            panel.get("indicator_color_rgb", (255, 255, 255))
        )
        st["black_point"] = int(
            panel.get("black_point", EXPORT.BW_DEFAULT_BLACK)
        )
        st["white_point"] = int(
            panel.get("white_point", EXPORT.BW_DEFAULT_WHITE)
        )
        if (panel.get("ch_label_text") or "").strip():
            st["chan_label_text"] = panel["ch_label_text"].strip()
        return st

    def _apply_levels_gray(
        self, pil_img: Image.Image, black: int, white: int
    ) -> Image.Image:
        """Apply simple grayscale levels between black and white points."""
        b = max(EXPORT.BW_MIN, min(white - 1, int(black)))
        w = min(EXPORT.BW_MAX, max(b + 1, int(white)))
        if b <= 0 and w >= 255:
            return pil_img.convert("L")
        span = max(1, w - b)
        lut = [0] * 256
        for i in range(256):
            lut[i] = (
                0
                if i <= b
                else (255 if i >= w else int(round((i - b) * 255.0 / span)))
            )
        return pil_img.convert("L").point(lut)

    def _resolve_sec_per_frame(self, main_window) -> float:
        """Return seconds/frame, defaulting to 1.0."""
        cached = getattr(main_window, "time_interval", None)
        if cached is not None:
            with suppress(ValueError, TypeError):
                return float(cached)
        with suppress(Exception):
            return float(main_window.time_input.value())
        return 1.0

    def _validate_render_channel(self, main_window, st: dict) -> str:
        """Validate the panel's channel index and return its channel id."""
        if st["chan_abs_idx"] is None:
            st["chan_abs_idx"] = 0
        if st["chan_abs_idx"] < 0 or st["chan_abs_idx"] >= len(
            main_window.images
        ):
            raise ValueError(
                f"Selected channel {st['chan_abs_idx']} not available."
            )
        return main_window.ids_channels[st["chan_abs_idx"]]

    def _load_leveled_frame(
        self, main_window, channel_id, st: dict, t_index: int, policy: str
    ) -> Image.Image:
        """Resolve the sample frame for t_index and apply black/white levels."""
        stack = main_window.images[channel_id]
        sel_t = self._resolve_sample_t(
            main_window, st["chan_abs_idx"], t_index, policy
        )
        if sel_t is None:
            raise KeyError("MISSING_FRAME")
        pil_img = Image.fromarray(stack[sel_t])
        return self._apply_levels_gray(
            pil_img, st["black_point"], st["white_point"]
        )

    def _pick_coord(self, row, m1_col: str, fallback_col: str) -> float | None:
        """First non-missing, non-NaN value of `m1_col` then `fallback_col`."""
        for col in (m1_col, fallback_col):
            val = row.get(col)
            if val is None:
                continue
            with suppress(TypeError, ValueError):
                fval = float(val)
                if not np.isnan(fval):
                    return fval
        return None

    def _extract_centroid(self, row) -> tuple[float, float]:
        """Return the (x, y) centroid from a track row, preferring M1 columns."""
        x = self._pick_coord(row, "XMorphologyM1", "XMorphology")
        y = self._pick_coord(row, "YMorphologyM1", "YMorphology")
        if x is None or y is None:
            raise ValueError("No usable centroid in track row.")
        return x, y

    def _compute_crop_box(
        self, x, y, crop_w: int, crop_h: int, img_w: int, img_h: int
    ) -> tuple[int, int, int, int]:
        """Return a centered (left, top, right, bottom) crop box clamped to image bounds."""
        rx, ry = int(round(crop_w / 2)), int(round(crop_h / 2))
        left, right = int(round(x)) - rx, int(round(x)) + rx
        top, bottom = int(round(y)) - ry, int(round(y)) + ry
        left, top = max(0, left), max(0, top)
        right, bottom = min(img_w, right), min(img_h, bottom)
        if right <= left or bottom <= top:
            return 0, 0, min(img_w, crop_w), min(img_h, crop_h)
        return left, top, right, bottom

    def _crop_and_resize(
        self, leveled: Image.Image, box: tuple[int, int, int, int]
    ) -> Image.Image:
        """Crop the leveled frame to box and resize it to the preview tile size."""
        cropped = leveled.crop(box).convert("RGB")
        return cropped.resize(
            (EXPORT.PREVIEW_W, EXPORT.PREVIEW_H), Image.NEAREST
        )

    def _overlay_mask(
        self,
        main_window,
        cropped: Image.Image,
        st: dict,
        box: tuple[int, int, int, int],
        t_index: int,
    ) -> Image.Image:
        """Blend the selected mask (full fill or contour) onto the cropped tile."""
        mask_idx = st.get("mask_idx")
        if mask_idx is None:
            return cropped
        mask_stack = self._get_mask_stack(main_window, int(mask_idx))
        if not isinstance(mask_stack, np.ndarray) or t_index >= len(
            mask_stack
        ):
            return cropped

        left, top, right, bottom = box
        mask_crop = mask_stack[t_index][top:bottom, left:right]
        mask_arr = np.asarray(
            Image.fromarray(mask_crop.astype(np.int32)).resize(
                (EXPORT.PREVIEW_W, EXPORT.PREVIEW_H), Image.NEAREST
            )
        )
        mask_bool = mask_arr > 0
        if st.get("mask_mode", "Full") == "Contour":
            mask_bool = self._compute_contour(mask_bool, st["contour_px"])
        if not mask_bool.any():
            return cropped

        base = np.asarray(cropped, dtype=np.uint8).astype(np.float32)
        overlay_color = np.array(list(st["mask_color"]), dtype=np.float32)
        alpha = float(np.clip(st["mask_alpha"], 0.0, 1.0))
        base[mask_bool] = (1 - alpha) * base[mask_bool] + alpha * overlay_color
        return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))

    def _draw_time_indicator(
        self, draw, main_window, st: dict, t_index: int, sec_per_frame: float
    ) -> int:
        """Draw the sliding time marker; return its top y offset (0 if hidden)."""
        y_ind = 0
        with suppress(Exception):
            y_ind = (
                EXPORT.TIME_INDICATOR_Y if st.get("time_bar_chk", False) else 0
            )
            if y_ind:
                t_end_ui = int(main_window.end_t_input.value())
                t_now_sec = max(0.0, t_index * sec_per_frame)
                t_end_sec = max(t_now_sec, t_end_ui * sec_per_frame)
                frac = (
                    0.0 if t_end_sec <= 0 else min(1.0, t_now_sec / t_end_sec)
                )
                pad = EXPORT.TEXT_PAD_L
                sz = st["indicator_px"]
                x0 = int(pad + frac * (EXPORT.PREVIEW_W - 2 * pad - sz))
                draw.rectangle(
                    [x0, y_ind, x0 + sz, y_ind + sz],
                    fill=st["indicator_color"],
                )
        return y_ind

    def _draw_time_text(
        self, draw, st: dict, t_index: int, sec_per_frame: float, line_y: int
    ) -> None:
        """Draw the time HH:MM:SS label if enabled."""
        if not st.get("time_text_chk", False):
            return
        font_t = self._get_font(st["time_font_size"], st.get("font_name"))
        text = self._fmt_hhmmss(max(0.0, t_index * sec_per_frame))
        with suppress(Exception):
            draw.text(
                (EXPORT.TEXT_PAD_L, line_y),
                text,
                fill=st["time_color"],
                font=font_t,
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )

    def _draw_channel_label(self, draw, st: dict, line_y: int) -> None:
        """Draw the right aligned channel label if enabled."""
        if not st.get("show_ch_lbl_chk", False):
            return
        font_ch = self._get_font(st["ch_font_size"], st.get("font_name"))
        txt = st.get("chan_label_text", "CH")
        tw = 60
        with suppress(Exception):
            tw = draw.textbbox((0, 0), txt, font=font_ch)[2]
        x_text = EXPORT.PREVIEW_W - tw - EXPORT.TEXT_PAD_R
        with suppress(Exception):
            draw.text(
                (x_text, line_y),
                txt,
                fill=st["ch_color"],
                font=font_ch,
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )

    def _render_panel_pil(
        self, main_window, panel, t_index: int
    ) -> Image.Image:
        """Render a single panel tile as a PIL image for time t."""
        sec_per_frame = self._resolve_sec_per_frame(main_window)

        st = self._get_panel_state_for_render(panel)
        if not st.get("ident", ""):
            raise ValueError("Identification is missing.")

        channel_id = self._validate_render_channel(main_window, st)
        policy = (
            self._get_combo_text(
                getattr(main_window, "missing_policy_combo", None)
            )
            or "Strict"
        )
        leveled = self._load_leveled_frame(
            main_window, channel_id, st, t_index, policy
        )

        x, y = self._resolve_centroid(
            main_window,
            st["ident"],
            t_index,
            st.get("lineage_path", []),
            policy,
        )

        crop_w = int(main_window.crop_full_x_input.value())
        crop_h = int(main_window.crop_full_y_input.value())
        box = self._compute_crop_box(
            x, y, crop_w, crop_h, leveled.width, leveled.height
        )
        cropped = self._crop_and_resize(leveled, box)
        cropped = self._overlay_mask(main_window, cropped, st, box, t_index)

        draw = ImageDraw.Draw(cropped)
        y_ind = self._draw_time_indicator(
            draw, main_window, st, t_index, sec_per_frame
        )
        line_y = (
            EXPORT.TEXT_PAD_L
            if y_ind == 0
            else (y_ind + st["indicator_px"] + EXPORT.LABEL_GAP)
        )
        self._draw_time_text(draw, st, t_index, sec_per_frame, line_y)
        self._draw_channel_label(draw, st, line_y)

        return cropped

    def _update_single_panel_preview(self, main_window, panel):
        """Render and update the on-canvas preview for a panel."""
        if not panel:
            return
        try:
            t = int(main_window.start_t_input.value())
            img = self._render_panel_pil(main_window, panel, t)
            arr = np.ascontiguousarray(
                np.asarray(img.convert("RGB"), dtype=np.uint8)
            )
            h, w, ch = arr.shape
            bytes_per_line = ch * w
            q_img = QImage(
                arr.data, w, h, bytes_per_line, QImage.Format_RGB888
            )
            panel["_buf"] = arr
            panel["label"].setPixmap(QPixmap.fromImage(q_img))
        except KeyError as e:
            if str(e) == "'MISSING_FRAME'":
                with suppress(Exception):
                    self._stop_play(main_window)
                panel["label"].setText("Preview unavailable")
            else:
                self._warn(main_window, "Preview Error", str(e))
        except (ValueError, IndexError, RuntimeError, OSError) as e:
            with suppress(Exception):
                self._stop_play(main_window)
            panel["label"].setText("Preview unavailable")
            self._warn(main_window, "Preview Error", str(e))

    def _canvas_bbox(self, main_window, export_gap_px: int):
        """Return (minx, miny, maxx, maxy) laid out with the export gap."""
        if not getattr(main_window, "panels", []):
            return 0, 0, EXPORT.PREVIEW_W, EXPORT.PREVIEW_H
        xs, ys, xe, ye = [], [], [], []
        step_x = EXPORT.PREVIEW_W + export_gap_px
        step_y = EXPORT.PREVIEW_H + export_gap_px
        for p in main_window.panels:
            col, row = self._cell_from_point(p["proxy"].pos())
            x = col * step_x
            y = row * step_y
            xs.append(x)
            ys.append(y)
            xe.append(x + EXPORT.PREVIEW_W)
            ye.append(y + EXPORT.PREVIEW_H)
        return min(xs), min(ys), max(xe), max(ye)

    def _render_canvas_frame(
        self, main_window, t_index: int
    ) -> Image.Image | None:
        """Render the full canvas image for time t."""
        export_gap = (
            int(getattr(main_window, "export_gap_spin", None).value())
            if hasattr(main_window, "export_gap_spin")
            else EXPORT.EXPORT_GAP_DEFAULT
        )
        minx, miny, maxx, maxy = self._canvas_bbox(main_window, export_gap)
        pad = EXPORT.CANVAS_MARGIN
        width = int(math.ceil(maxx - minx)) + 2 * pad
        height = int(math.ceil(maxy - miny)) + 2 * pad
        canvas = Image.new("RGB", (max(1, width), max(1, height)), (0, 0, 0))
        policy = (
            self._get_combo_text(
                getattr(main_window, "missing_policy_combo", None)
            )
            or "Strict"
        )

        any_missing = False
        tiles_and_positions = []

        step_x = EXPORT.PREVIEW_W + export_gap
        step_y = EXPORT.PREVIEW_H + export_gap
        for p in main_window.panels:
            col, row = self._cell_from_point(p["proxy"].pos())
            x = col * step_x - minx + pad
            y = row * step_y - miny + pad
            try:
                tile = self._render_panel_pil(main_window, p, t_index).convert(
                    "RGB"
                )
                tiles_and_positions.append((tile, (x, y)))
            except KeyError as e:
                if str(e) == "'MISSING_FRAME'":
                    any_missing = True
                    tiles_and_positions.append((None, (x, y)))
                else:
                    raise

        if any_missing and policy == "Strict":
            return None
        for tile, (x, y) in tiles_and_positions:
            if tile is not None:
                canvas.paste(tile, (x, y))
        return canvas
