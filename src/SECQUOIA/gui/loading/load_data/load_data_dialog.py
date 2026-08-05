"""The Load data window: choosing what to load, and loading it."""

from __future__ import annotations

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QPushButton,
    QStyle,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from SECQUOIA.config import LINKS, TOOLTIPSTEXT
from SECQUOIA.gui.common.ui_utils import (
    drop_dead_widget_attrs,
    harmonize_form_labels,
    is_widget_alive,
    make_help_button,
    spinbox_arrow_pngs,
    widen_to_hint,
)
from SECQUOIA.gui.loading.experiment_paths import _update_load_button_enabled
from SECQUOIA.gui.loading.load_data.load_data_summary import _update_summary
from SECQUOIA.gui.loading.load_data.load_data_tabs import (
    build_channel_tab,
    build_loading_tab,
    build_segmentation_tab,
    build_tracking_tab,
)
from SECQUOIA.gui.loading.load_data.load_data_theme import Theme, UiSize

__all__ = ["load_data_window"]


def load_data_window(main_window: QWidget) -> None:
    """Open the Load Data window: four tabs, with only Tracking enabled at first."""

    drop_dead_widget_attrs(main_window)

    prev = getattr(main_window, "mask_no_window", None)
    if prev is not None:
        try:
            prev.close()
            prev.deleteLater()
        except RuntimeError:
            pass
        main_window.mask_no_window = None

    w = QWidget(main_window)
    w.setWindowTitle("Load Experiment Data")
    w.setWindowModality(Qt.WindowModal)
    w.setWindowFlag(Qt.Tool, True)
    w.setMinimumSize(*UiSize.min_win)
    w.resize(*UiSize.min_win)
    _arrows = spinbox_arrow_pngs(Theme.text)
    w.setStyleSheet(f"""
        QWidget {{
            font-size: {UiSize.font_pt}pt;
            color: {Theme.text};
            background-color: {Theme.bg_base};
        }}
        QTabWidget::pane {{
            border: 1px solid {Theme.border};
            background: {Theme.bg_base};
        }}
        QTabBar::tab {{
            background: {Theme.bg_panel};
            color: {Theme.text};
            padding: 6px 14px;
            border: 1px solid {Theme.border};
            margin-right: 2px;
        }}
        QTabBar::tab:selected {{ background: {Theme.bg_selected}; }}
        QTabBar::tab:hover    {{ background: {Theme.ctrl_hover}; }}

        QGroupBox, QFrame {{
            background-color: {Theme.bg_base};
            border: 1px solid {Theme.border};
            border-radius: 4px;
        }}

        QLineEdit, QComboBox {{
            background-color: {Theme.ctrl_bg};
            color: {Theme.text};
            border: 1px solid {Theme.border};
            padding: 2px 4px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {Theme.ctrl_bg};
            color: {Theme.text};
            selection-background-color: {Theme.bg_selected};
            border: 1px solid {Theme.border};
        }}

        QTreeWidget {{
            background-color: {Theme.ctrl_bg};
            color: {Theme.text};
            border: 1px solid {Theme.border};
        }}
        QHeaderView::section {{
            background-color: {Theme.bg_panel};
            color: {Theme.text};
            border: 1px solid {Theme.border};
            padding: 4px;
        }}

        QPushButton {{
            background-color: {Theme.ctrl_bg};
            color: {Theme.text};
            border: 1px solid {Theme.border};
            padding: 6px 12px;
            border-radius: 4px;
        }}
        QPushButton:hover {{ background-color: {Theme.ctrl_hover}; }}
        QSpinBox, QDoubleSpinBox {{
            background-color: {Theme.ctrl_bg};
            color: {Theme.text};
            border: 1px solid {Theme.border};
            padding: 2px 20px 2px 4px;   /* right room for the buttons */
        }}
        QSpinBox::up-button, QDoubleSpinBox::up-button {{
            subcontrol-origin: border; subcontrol-position: top right;
            width: 16px; background-color: {Theme.bg_panel};
            border-left: 1px solid {Theme.border};
            border-bottom: 1px solid {Theme.border};
        }}
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            subcontrol-origin: border; subcontrol-position: bottom right;
            width: 16px; background-color: {Theme.bg_panel};
            border-left: 1px solid {Theme.border};
        }}
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
            background-color: {Theme.ctrl_hover};
        }}
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
            image: url("{_arrows['up']}");
            width: 8px; height: 8px;
        }}
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
            image: url("{_arrows['down']}");
            width: 8px; height: 8px;
        }}
        QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled,
        QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{
            image: none;
        }}

    """)

    outer = QVBoxLayout(w)
    outer.setContentsMargins(10, 10, 10, 10)
    outer.setSpacing(8)

    tabs = QTabWidget()
    outer.addWidget(tabs)
    main_window._tabs = tabs

    tabs.addTab(build_tracking_tab(main_window, UiSize), "Tracking")
    tabs.addTab(
        build_segmentation_tab(main_window, UiSize), "Segmentation Options"
    )
    tabs.addTab(build_channel_tab(main_window, UiSize), "Channel Selection")
    tabs.addTab(build_loading_tab(main_window, UiSize, w), "Loading")

    help_btn = make_help_button(
        tabs, TOOLTIPSTEXT.HELP_BTN, LINKS.GITHUB_LOADING_WINDOW
    )
    tabs.setCornerWidget(help_btn, Qt.TopRightCorner)

    for i in (1, 2, 3):
        tabs.setTabEnabled(i, False)

    loading_idx = tabs.indexOf(tabs.widget(3))

    def _on_tab_changed(idx):
        """Refresh summary information."""
        if idx == loading_idx:
            _update_summary(main_window)

    tabs.currentChanged.connect(_on_tab_changed)

    btn_row = QWidget()
    btn_h = QHBoxLayout(btn_row)
    btn_h.setContentsMargins(0, 0, 0, 0)
    btn_h.setSpacing(8)
    exit_icon = QApplication.style().standardIcon(QStyle.SP_DialogCloseButton)
    main_window.exit_button = QPushButton("Close")
    main_window.exit_button.setToolTip(TOOLTIPSTEXT.CLOSE_BTN_LOADING)
    main_window.exit_button.setIcon(exit_icon)
    widen_to_hint(main_window.exit_button, 24)
    main_window.exit_button.clicked.connect(w.close)
    btn_h.addStretch()
    btn_h.addWidget(main_window.exit_button)
    outer.addWidget(btn_row)

    _update_load_button_enabled(main_window)
    harmonize_form_labels(w)

    def _grow_on_screen_change(_s):
        """Resize the load data window to fit the newly selected screen."""
        wh = w.windowHandle()
        if not wh:
            return
        ag = wh.screen().availableGeometry()
        w.resize(
            max(w.width(), max(UiSize.min_win[0], int(ag.width() * 0.8))),
            max(w.height(), max(UiSize.min_win[1], int(ag.height() * 0.8))),
        )

    def _watch_screen_changes():
        """Follow the window to another screen, once it has a native handle."""
        if not is_widget_alive(w):
            return
        handle = w.windowHandle()
        if handle is not None:
            handle.screenChanged.connect(_grow_on_screen_change)

    QTimer.singleShot(0, _watch_screen_changes)

    main_window.mask_no_window = w
    w.show()
    w.raise_()
    w.activateWindow()
