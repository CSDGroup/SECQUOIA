"""Small, reusable Qt widget helpers."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import qtawesome as qta
from qtpy.QtCore import QEventLoop, QPoint, QSize, Qt, QUrl
from qtpy.QtGui import QColor, QDesktopServices, QPainter, QPixmap, QPolygon
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyleFactory,
    QToolButton,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)

__all__ = [
    "ContentWidthScrollArea",
    "TOOLBAR_ICON_PX",
    "add_progress_bar",
    "checkbox_stylesheet",
    "clear_layout",
    "compact_combo",
    "compact_spin",
    "dpi_icon_size",
    "drop_dead_widget_attrs",
    "equalize_min_widths",
    "fit_to_screen",
    "fixed_label",
    "form_row",
    "glyph_png",
    "harmonize_form_labels",
    "is_widget_alive",
    "make_help_button",
    "make_tab_scaffold",
    "make_tab_title",
    "make_vertical_separator",
    "open_at_screen_frac",
    "restore_button_loading",
    "set_button_icon",
    "set_fixed_width",
    "set_min_expanding",
    "set_tree_rows",
    "spinbox_arrow_pngs",
    "style_button_loading",
    "themed_icon",
    "use_fusion_widget_style",
    "use_fusion_combos",
    "use_readable_combo_popup_on_macos",
    "reuse_widget",
    "widen_to_hint",
]


def is_widget_alive(widget) -> bool:
    """Return True if the widget exists and its C++ object is not deleted."""
    if widget is None:
        return False
    try:
        widget.objectName()
    except RuntimeError:
        return False
    return True


def drop_dead_widget_attrs(owner) -> list[str]:
    dropped = []
    for name, value in list(vars(owner).items()):
        if isinstance(value, QWidget) and not is_widget_alive(value):
            setattr(owner, name, None)
            dropped.append(name)
    return dropped


def reuse_widget(main_window, attr: str, factory):
    """Return ``main_window.<attr>`` if it is still usable, else build it a new."""
    widget = getattr(main_window, attr, None)
    if not is_widget_alive(widget):
        widget = factory()
    setattr(main_window, attr, widget)
    return widget


_fusion_style = None


def use_fusion_widget_style(widget: QWidget) -> None:
    """Give one widget (and its children) Fusion metrics on every platform."""
    global _fusion_style
    if not is_widget_alive(_fusion_style):
        _fusion_style = QStyleFactory.create("Fusion")
    widget.setStyle(_fusion_style)


def use_fusion_combos(root: QWidget) -> None:
    """Give every combo under ``root`` Fusion metrics and a themed popup."""
    for combo in root.findChildren(QComboBox):
        use_fusion_widget_style(combo)


def use_readable_combo_popup_on_macos(combo: QComboBox) -> None:
    """Apply the Fusion style to a combo so its popup honors the stylesheet."""
    if sys.platform != "darwin":
        return
    use_fusion_widget_style(combo)


# Base icon size in 96-DPI pixels, scaled up on high-DPI screens.
_BASE_ICON_PX = 20
TOOLBAR_ICON_PX = 14


_LOADING_STYLESHEET = (
    "QPushButton { background-color: #2e7d32; color: #ffffff; "
    "border: 1px solid #1b5e20; }"
)

_PROGRESS_STYLESHEET = """
QProgressBar {
    border: 1pt solid #d0d0d0;
    border-radius: 6pt;
    padding: 2pt;
    background: #f5f5f5;
    color: #000;                  /* black text */
    text-align: center;           /* center the text */
    font-weight: bold;            /* make it bold */
}
QProgressBar::chunk {
    background-color: #4caf50;    /* green fill */
    margin: 0.5pt;
    border-radius: 4pt;
}
"""


def themed_icon(name: str, **kwargs):
    """Create a qtawesome icon with SECQUOIA's default white-on-dark colors."""
    kwargs.setdefault(
        "options",
        [
            {
                "color": "white",
                "color_active": "white",
                "color_disabled": "#aaaaaa",
            }
        ],
    )
    return qta.icon(name, **kwargs)


def dpi_icon_size(widget: QWidget, base: int = _BASE_ICON_PX) -> QSize:
    """Return an icon size scaled for the screen the widget lives on."""
    screen = None
    window_handle = widget.window().windowHandle() if widget.window() else None
    if window_handle:
        screen = window_handle.screen()
    if not screen:
        screen = QApplication.primaryScreen()

    scale = 1.0
    if screen and hasattr(screen, "logicalDotsPerInch"):
        with contextlib.suppress(Exception):
            scale = max(1.0, float(screen.logicalDotsPerInch()) / 96.0)

    size = int(round(base * scale))
    return QSize(size, size)


def set_button_icon(
    btn: QPushButton,
    name: str,
    size: int | None = None,
    reference: QWidget | None = None,
    **icon_kwargs,
) -> None:
    """Set a themed icon and a matching icon size on a button."""
    btn.setIcon(themed_icon(name, **icon_kwargs))
    if size is None:
        size = dpi_icon_size(reference or btn).width()
    btn.setIconSize(QSize(size, size))


def widen_to_hint(
    widget: QWidget, extra_px: int = 24, fix_policy: bool = True
) -> None:
    """Fix a widget's width to its size hint plus padding."""
    if fix_policy:
        widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    widget.setFixedWidth(widget.sizeHint().width() + max(0, int(extra_px)))


def set_fixed_width(widget: QWidget, width: int) -> None:
    """Apply a fixed width and a fixed size policy to a widget."""
    widget.setFixedWidth(int(width))
    widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)


def set_min_expanding(widget: QWidget, min_w: int) -> None:
    """Set a minimum width while letting the widget expand horizontally."""
    widget.setMinimumWidth(int(min_w))
    widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)


def compact_combo(combo: QComboBox, min_chars: int = 8, max_w: int = 180):
    """Keep a combo from widening to its longest entry."""
    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(int(min_chars))
    combo.setMaximumWidth(int(max_w))
    combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
    return combo


def compact_spin(spin: QWidget, max_w: int = 96):
    """Cap a spin box's width so its numeric range does not dictate the layout."""
    spin.setMaximumWidth(int(max_w))
    spin.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
    return spin


def fit_to_screen(
    window: QWidget,
    max_frac: float = 0.85,
    min_size: tuple[int, int] = (640, 300),
) -> None:
    """Size a window to its hint, capped at a fraction of the screen."""
    hint = window.sizeHint()
    screen = window.screen() if hasattr(window, "screen") else None
    if screen is None:
        screen = QApplication.primaryScreen()
    avail = (
        screen.availableGeometry()
        if screen is not None
        else QApplication.desktop().availableGeometry()
    )
    width = min(hint.width(), int(avail.width() * max_frac))
    height = min(hint.height(), int(avail.height() * max_frac))
    window.setMinimumSize(
        min(min_size[0], width),
        min(min_size[1], height),
    )
    window.resize(max(width, 1), max(height, 1))


def open_at_screen_frac(window: QWidget, frac: float = 0.8) -> None:
    """Size a window to a share of the screen, but never below its hint."""
    screen = window.screen() or QApplication.primaryScreen()
    avail = screen.availableGeometry()
    hint = window.sizeHint()
    window.resize(
        min(max(hint.width(), int(avail.width() * frac)), avail.width()),
        min(max(hint.height(), int(avail.height() * frac)), avail.height()),
    )


def equalize_min_widths(
    widgets: list[QWidget | None], target_width: int | None = None
) -> None:
    """Give a group of widgets a common width."""
    hints = [w.sizeHint().width() for w in widgets if w]
    if not hints:
        return
    width = max(hints) if target_width is None else int(target_width)
    for widget in widgets:
        if widget is None:
            continue
        set_fixed_width(widget, width)


def harmonize_form_labels(root: QWidget) -> None:
    """Left align every ``formLabel`` under ``root`` and equalize their width."""
    labels = root.findChildren(QLabel, "formLabel")
    if not labels:
        return
    for label in labels:
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    max_width = max(label.sizeHint().width() for label in labels)
    for label in labels:
        label.setFixedWidth(max_width)


def set_tree_rows(tree: QTreeWidget, approx_rows: int = 8) -> None:
    """Size a tree widget's minimum height to show roughly N rows."""
    metrics = tree.fontMetrics()
    row_h = max(20, metrics.height() + 6)
    header_h = (
        tree.header().sizeHint().height()
        if tree.header()
        else metrics.height() + 8
    )
    tree.setMinimumHeight(row_h * int(approx_rows) + header_h + 8)
    tree.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


def clear_layout(layout: QLayout | None) -> None:
    """Remove and delete all widgets and sub-layouts from ``layout``."""
    if not isinstance(layout, QLayout):
        return

    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()

        if w is not None:
            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                w.setParent(None)
            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                w.deleteLater()
            continue

        sub = item.layout()
        if sub is not None:
            clear_layout(sub)


def make_help_button(parent: QWidget, tooltip: str, url: str) -> QToolButton:
    """Build a themed help-icon button that opens ``url`` when clicked."""
    btn = QToolButton(parent)
    btn.setAutoRaise(True)
    btn.setCursor(Qt.PointingHandCursor)
    btn.setIcon(themed_icon("mdi.help-circle-outline"))
    btn.setIconSize(dpi_icon_size(btn))
    btn.setToolTip(tooltip)
    btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(url)))
    return btn


def make_vertical_separator(parent: QWidget | None = None) -> QFrame:
    """Create a thin vertical separator frame for a tab layout."""
    line = QFrame(parent)
    line.setFrameShape(QFrame.VLine)
    line.setFrameShadow(QFrame.Sunken)
    line.setFixedWidth(1)
    line.setStyleSheet("QFrame { background: #444; }")
    return line


class ContentWidthScrollArea(QScrollArea):
    """A scroll area that asks for the full width of the widget it holds."""

    def sizeHint(self) -> QSize:
        """Return the held widget's preferred width, plus frame and scroll bar."""
        hint = super().sizeHint()
        inner = self.widget()
        if inner is None:
            return hint
        width = inner.sizeHint().width() + 2 * self.frameWidth()
        if self.verticalScrollBarPolicy() != Qt.ScrollBarAlwaysOff:
            width += self.verticalScrollBar().sizeHint().width()
        return QSize(max(hint.width(), width), hint.height())


def make_tab_scaffold(
    tab_widget: QWidget,
) -> tuple[QVBoxLayout, QVBoxLayout]:
    """Split a tab into a wide content column and a narrow side-button column."""
    outer_h = QHBoxLayout(tab_widget)
    outer_h.setContentsMargins(8, 8, 8, 8)
    outer_h.setSpacing(8)

    content = QWidget()
    content_v = QVBoxLayout(content)
    content_v.setContentsMargins(0, 0, 0, 0)
    content_v.setSpacing(8)

    scroll = ContentWidthScrollArea(tab_widget)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setWidget(content)

    side = QWidget(tab_widget)
    side.setFixedWidth(130)
    side_v = QVBoxLayout(side)
    side_v.setContentsMargins(0, 0, 0, 0)
    side_v.setSpacing(8)
    side_v.setAlignment(Qt.AlignTop)

    outer_h.addWidget(scroll, 1)
    outer_h.addWidget(make_vertical_separator(tab_widget))
    outer_h.addWidget(side, 0)
    return content_v, side_v


def make_tab_title(text: str) -> QLabel:
    """Create a compact, word-wrapped title label for a tab."""
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setFixedHeight(28)
    lab.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    lab.setStyleSheet("QLabel { margin: 0; padding: 0; }")
    return lab


def fixed_label(text: str) -> QLabel:
    """Create a label that does not stretch."""
    label = QLabel(text)
    label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
    return label


def form_row(label_text: str, widget_or_row) -> QHBoxLayout:
    """Build a label-plus-field row that emulates a form layout entry."""
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(8)

    label = QLabel(label_text)
    label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    label.setObjectName("formLabel")
    row.addWidget(label)

    if isinstance(widget_or_row, QWidget):
        row.addWidget(widget_or_row, 1)
    else:
        row.addLayout(widget_or_row, 1)
    return row


def style_button_loading(btn: QPushButton) -> None:
    """Put a button into its disabled "Loading…" state."""
    if not hasattr(btn, "_orig_text"):
        btn._orig_text = btn.text()
    if not hasattr(btn, "_orig_stylesheet"):
        btn._orig_stylesheet = btn.styleSheet()

    btn.setText("Loading…")
    btn.setStyleSheet(_LOADING_STYLESHEET)
    btn.setEnabled(False)
    btn.repaint()
    QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)


def restore_button_loading(
    btn: QPushButton, fallback_text: str = "Load data"
) -> None:
    """Restore a button from its "Loading…" state to its original text/style."""
    original_text = getattr(btn, "_orig_text", fallback_text) or fallback_text
    btn.setText(original_text)
    btn.setStyleSheet(getattr(btn, "_orig_stylesheet", ""))
    btn.setEnabled(True)


def spinbox_arrow_pngs(color: str = "#ffffff") -> dict[str, str]:
    """Render up/down spinbox arrows to PNG files and return their paths."""
    icon_dir = Path.home() / ".SECQUOIA" / "icons"
    icon_dir.mkdir(parents=True, exist_ok=True)

    scale = 3
    size = 8 * scale
    shapes = {
        "up": [
            (size * 0.5, size * 0.28),
            (size * 0.82, size * 0.70),
            (size * 0.18, size * 0.70),
        ],
        "down": [
            (size * 0.18, size * 0.30),
            (size * 0.82, size * 0.30),
            (size * 0.5, size * 0.72),
        ],
    }

    paths = {}
    for name, points in shapes.items():
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(color))
        painter.drawPolygon(
            QPolygon([QPoint(int(x), int(y)) for x, y in points])
        )
        painter.end()

        file_path = icon_dir / f"spin_{name}.png"
        pixmap.save(str(file_path), "PNG")
        paths[name] = file_path.as_posix()
    return paths


def add_progress_bar(
    parent_layout: QVBoxLayout | None,
) -> tuple[QProgressBar, dict]:
    """Create a green progress bar and return it with its update helpers."""
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    bar.setTextVisible(True)
    bar.setFormat("%p%")
    bar.setAlignment(Qt.AlignCenter)
    bar.setStyleSheet(_PROGRESS_STYLESHEET)

    if parent_layout is not None:
        parent_layout.addWidget(bar)

    def set_progress(pct: int):
        """Set the bar to an explicit percentage, clamped to 0-100."""
        bar.setValue(max(0, min(100, int(pct))))

    def bump_progress(delta: int):
        """Increase the bar by a relative amount."""
        set_progress(bar.value() + int(delta))

    def finish_progress():
        """Mark the bar as complete."""
        set_progress(100)

    def step_progress(idx: int, total: int):
        """Set the bar from an item count, e.g. position 3 of 12."""
        pct = round((idx / total) * 100) if total > 0 else 0
        set_progress(pct)

    helpers = {
        "set": set_progress,
        "bump": bump_progress,
        "finish": finish_progress,
        "step": step_progress,
    }
    return bar, helpers


def glyph_png(name: str, color: str = "#ffffff", size: int = 24) -> str:
    """Render a qtawesome glyph to a PNG and return its path for QSS url()."""
    icon_dir = Path.home() / ".SECQUOIA" / "icons"
    icon_dir.mkdir(parents=True, exist_ok=True)
    path = icon_dir / f"{name.replace('.', '_')}_{color.lstrip('#')}.png"
    qta.icon(name, color=color).pixmap(QSize(size, size)).save(
        str(path), "PNG"
    )
    return path.as_posix()


def checkbox_stylesheet(accent: str = "#0a84ff") -> str:
    """QSS drawing every checkbox identically on macOS and Windows."""
    check = glyph_png("mdi.check-bold")
    dash = glyph_png("mdi.minus-thick")
    return f"""
    QCheckBox::indicator,
    QTreeView::indicator,
    QListView::indicator {{
        width: 14px;
        height: 14px;
        border: 1px solid #6f6f6f;
        border-radius: 3px;
        background: #3a3a3a;
    }}
    QCheckBox::indicator:checked,
    QTreeView::indicator:checked,
    QListView::indicator:checked {{
        background: {accent};
        border: 1px solid {accent};
        padding: 2px;
        image: url("{check}");
    }}
    QCheckBox::indicator:indeterminate,
    QTreeView::indicator:indeterminate,
    QListView::indicator:indeterminate {{
        background: {accent};
        border: 1px solid {accent};
        padding: 2px;
        image: url("{dash}");
    }}
    QCheckBox::indicator:disabled,
    QTreeView::indicator:disabled,
    QListView::indicator:disabled {{
        background: #333333;
        border: 1px solid #444444;
        image: none;
    }}
    """
