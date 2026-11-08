"""Drag the time text, CH label and scale bar inside an image."""

from SECQUOIA.config import EXPORT

OVERLAY_DEFAULTS = {
    "off_time_text": (0.0, 0.0),
    "off_ch_label": (0.0, 0.0),
    "off_scale_bar": (0.0, 0.0),
}


def _clamp(v, size, limit):
    """Keep [v, v + size] inside [0, limit]."""
    return min(max(0, v), max(0, limit - size))


class _OverlayDrag:
    """Place overlays from their stored offset and move them by mouse."""

    def _place(self, st: dict, key: str, x, y, w, h) -> tuple[int, int]:
        """Top-left for an overlay: default + offset, snapped inside an image."""
        W, H = EXPORT.PREVIEW_W, EXPORT.PREVIEW_H
        dx, dy = st.get(key, (0.0, 0.0))
        px = int(round(_clamp(x + dx * W, w, W)))
        py = int(round(_clamp(y + dy * H, h, H)))
        st["_hit"][key] = (px, py, w, h, x, y)
        return px, py

    def _overlay_press(self, panel: dict, pos) -> bool:
        """Start dragging the topmost overlay under `pos`, if any."""
        panel["_drag"] = None
        for key, box in reversed(list(panel.get("_hit", {}).items())):
            x, y, w, h = box[:4]
            if x <= pos.x() <= x + w and y <= pos.y() <= y + h:
                panel["_drag"] = (key, pos.x(), pos.y(), box)
                return True
        return False

    def _overlay_move(self, main_window, panel: dict, pos) -> None:
        """Move the dragged overlay, never past the tile edges."""
        drag = panel.get("_drag")
        if not drag:
            return
        key, sx, sy, (x, y, w, h, x0, y0) = drag
        W, H = EXPORT.PREVIEW_W, EXPORT.PREVIEW_H
        nx = _clamp(x + pos.x() - sx, w, W)
        ny = _clamp(y + pos.y() - sy, h, H)
        panel[key] = ((nx - x0) / W, (ny - y0) / H)
        self._update_single_panel_preview(main_window, panel)
