"""Shared style constants for the Load Data dialog and its tabs."""

from __future__ import annotations

from SECQUOIA.config import STYLE

__all__ = ["Theme", "UiSize"]


class Theme:
    bg_base = "#2b2b2b"
    bg_panel = "#333333"
    bg_selected = "#3c3c3c"
    border = "#555555"
    text = "#ffffff"
    ctrl_bg = "#404040"
    ctrl_hover = "#4a4a4a"


class UiSize:
    font_pt = STYLE.FONT_SIZE
    select_w = 240
    numeric_w = 170
    min_w_editor = 170
    min_w_select = 240
    min_win = (700, 750)
