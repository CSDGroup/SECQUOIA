"""Per-image scale bar settings dialog and drawing."""

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QPushButton,
    QSpinBox,
)

from SECQUOIA.config import EXPORT, TOOLTIPSTEXT
from SECQUOIA.gui.common.ui_utils import use_fusion_combos

UNIT_TO_UM = {"nm": 0.001, "µm": 1.0, "mm": 1000.0}

SCALE_BAR_DEFAULTS = {
    "scale_bar_chk": False,
    "scale_bar_len": 5.0,
    "scale_bar_unit": "µm",
    "scale_bar_text_chk": True,
    "scale_bar_font": EXPORT.DEFAULT_FONT_NAME,
    "scale_bar_font_size": 12,
    "scale_bar_px": 4,
    "scale_bar_color_rgb": (255, 255, 255),
}


def _spin(box, lo, hi, value, decimals=None):
    """Configure a spin box's range, precision and value in one place."""
    if decimals is not None:
        box.setDecimals(decimals)
    box.setRange(lo, hi)
    box.setValue(type(lo)(value))
    return box


class _ScaleBar:
    """Scale bar sized from µm/px, crop width and tile width."""

    def _um_per_px(self, main_window) -> float:
        """User override for this experiment, else the TATexp value, else 1.0."""
        xml = getattr(main_window, "xml_path", None)
        saved = getattr(main_window, "export_um_per_px", None)
        if saved and saved[0] == xml:
            return saved[1]
        tat = getattr(getattr(main_window, "clt_parser", None), "tat", None)
        return float(getattr(tat, "um_per_px", 0) or 0) or 1.0

    def _draw_scale_bar(self, draw, main_window, st: dict, box) -> None:
        """Draw the bar bottom right; skip it if it does not fit the tile."""
        if not st.get("scale_bar_chk", False):
            return
        um = st["scale_bar_len"] * UNIT_TO_UM.get(st["scale_bar_unit"], 1.0)
        tile_px_per_um = (
            EXPORT.PREVIEW_W / (box[2] - box[0]) / self._um_per_px(main_window)
        )
        length = round(um * tile_px_per_um)
        pad = EXPORT.TEXT_PAD_R
        if length < 1 or length > EXPORT.PREVIEW_W - 2 * pad:
            return
        thick = int(st["scale_bar_px"])
        color = tuple(st["scale_bar_color_rgb"])
        text, tw, th = "", 0, -2
        if st.get("scale_bar_text_chk", True):
            font = self._get_font(
                st["scale_bar_font_size"], st["scale_bar_font"]
            )
            text = f"{st['scale_bar_len']:g} {st['scale_bar_unit']}"
            _, _, tw, th = draw.textbbox((0, 0), text, font, stroke_width=1)
        gw, gh = max(length, tw), th + 2 + thick
        gx, gy = self._place(
            st,
            "off_scale_bar",
            EXPORT.PREVIEW_W - pad - gw,
            EXPORT.PREVIEW_H - pad - gh,
            gw,
            gh,
        )
        bx, top = gx + (gw - length) // 2, gy + gh - thick
        draw.rectangle([bx, top, bx + length - 1, gy + gh - 1], fill=color)
        if text:
            draw.text(
                (gx + (gw - tw) // 2, gy),
                text,
                fill=color,
                font=font,
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )

    def _open_scale_bar_dialog(self, main_window) -> None:
        """Edit µm/px and the active image's bar settings."""
        p = getattr(main_window, "active_panel", None)
        if not p:
            self._info(
                main_window.gif_window, "Scale bar", "Select a tile first."
            )
            return
        dlg = QDialog(main_window.gif_window)
        dlg.setWindowTitle("Scale bar")
        form = QFormLayout(dlg)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        um_px = _spin(
            QDoubleSpinBox(), 0.0001, 1000.0, self._um_per_px(main_window), 4
        )
        length = _spin(QDoubleSpinBox(), 0.01, 1e5, p["scale_bar_len"], 2)
        unit = QComboBox()
        unit.addItems(list(UNIT_TO_UM))
        unit.setCurrentText(p["scale_bar_unit"])
        font = QComboBox()
        font.addItems(list(EXPORT.FONT_NAMES) or [EXPORT.DEFAULT_FONT_NAME])
        font.setCurrentText(p["scale_bar_font"])
        show_text = QCheckBox()
        show_text.setChecked(bool(p["scale_bar_text_chk"]))
        font_size = _spin(QSpinBox(), 8, 72, p["scale_bar_font_size"])
        thick = _spin(QSpinBox(), 1, 40, p["scale_bar_px"])
        color_btn = QPushButton()
        color = [tuple(p["scale_bar_color_rgb"])]
        self._btn_color(color_btn, color[0])

        saved_panel = {k: p[k] for k in SCALE_BAR_DEFAULTS}
        saved_um = getattr(main_window, "export_um_per_px", None)

        def apply(*_):
            """Write the dialog values to the tile and refresh the previews."""
            main_window.export_um_per_px = (
                getattr(main_window, "xml_path", None),
                um_px.value(),
            )
            p.update(
                scale_bar_len=length.value(),
                scale_bar_unit=unit.currentText(),
                scale_bar_text_chk=show_text.isChecked(),
                scale_bar_font=font.currentText(),
                scale_bar_font_size=font_size.value(),
                scale_bar_px=thick.value(),
                scale_bar_color_rgb=color[0],
            )
            self._update_all_previews(main_window)

        def pick_color():
            c = QColorDialog.getColor(QColor(*color[0]), dlg, "Choose color")
            if c.isValid():
                color[0] = (c.red(), c.green(), c.blue())
                self._btn_color(color_btn, color[0])
                apply()

        color_btn.clicked.connect(pick_color)
        show_text.toggled.connect(apply)
        for field in (um_px, length, font_size, thick):
            field.setMinimumWidth(180)
            field.valueChanged.connect(apply)
        for field in (unit, font):
            field.setMinimumWidth(180)
            field.currentTextChanged.connect(apply)

        form.addRow("µm / px:", um_px)
        form.addRow("Length:", length)
        form.addRow("Unit:", unit)
        form.addRow("Show text:", show_text)
        form.addRow("Font:", font)
        form.addRow("Text size:", font_size)
        form.addRow("Thickness:", thick)
        form.addRow("Color:", color_btn)
        for field, tip in (
            (um_px, TOOLTIPSTEXT.SCALE_BAR_UM_PER_PX),
            (length, TOOLTIPSTEXT.SCALE_BAR_LENGTH),
            (unit, TOOLTIPSTEXT.SCALE_BAR_UNIT),
            (show_text, TOOLTIPSTEXT.SCALE_BAR_SHOW_TEXT),
            (font, TOOLTIPSTEXT.SCALE_BAR_FONT),
            (font_size, TOOLTIPSTEXT.SCALE_BAR_TEXT_SIZE),
            (thick, TOOLTIPSTEXT.SCALE_BAR_THICKNESS),
            (color_btn, TOOLTIPSTEXT.SCALE_BAR_COLOR),
        ):
            field.setToolTip(tip)
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        use_fusion_combos(dlg)
        if dlg.exec_() == QDialog.Accepted:
            return
        main_window.export_um_per_px = saved_um
        p.update(saved_panel)
        self._update_all_previews(main_window)
