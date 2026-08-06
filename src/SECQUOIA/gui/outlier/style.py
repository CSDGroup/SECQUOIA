"""Dark theme stylesheet for the outlier detection window."""

from __future__ import annotations

from SECQUOIA.config import STYLE

__all__ = ["outlier_window_stylesheet"]


def outlier_window_stylesheet(arrows: dict[str, str]) -> str:
    """Build the outlier detection window's QSS, using rendered spinbox arrow PNGs."""
    return f"""
    QWidget {{
        font-size: {STYLE.FONT_SIZE_outlier}pt;
        font-family: "{STYLE.FONT_outlier}";
        color: #E6E6E6;           /* light text everywhere */
        background: #2B2B2B;      /* dark background matching the main window  */
    }}

    /* --- Tabs --- */
    QTabWidget::pane {{
        border: 1px solid #444;
        top: -1px;                /* make the selected tab look connected */
        background: #2B2B2B;      /* same as window background */
    }}
    QTabBar {{
        background: #2B2B2B;      /* remove white strip behind the tabs */
    }}
    QTabBar::tab {{
        background: #3A3A3A;
        color: #E0E0E0;
        padding: 4px 8px;
        border: 1px solid #555;
        border-bottom: none;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        margin-right: 2px;        /* small gap between tabs */
    }}
    QTabBar::tab:selected {{
        background: #232323;      /* darker for the active tab */
        color: #FFFFFF;
        border-color: #666;
    }}
    QTabBar::tab:!selected {{
        background: #3A3A3A;
        color: #C8C8C8;
    }}
    QTabBar::tab:hover {{
        background: #2E2E2E;
    }}
    QTabBar::tab:disabled {{
        color: #777;
    }}

    /* Buttons and combos in the same dark theme */
    QPushButton {{
        background: #3A3A3A;
        border: 1px solid #555;
        padding: 4px 10px;
    }}
    QPushButton:hover {{ background: #2E2E2E; }}
    QPushButton:pressed {{ background: #1F1F1F; }}

    QComboBox, QDoubleSpinBox {{
        background: #3A3A3A;
        border: 1px solid #555;
        selection-background-color: #555;
        selection-color: #FFF;
    }}
    QComboBox:disabled, QDoubleSpinBox:disabled {{
        background: #333333;
        color: #808080;
        border: 1px solid #444;
    }}
    QComboBox QAbstractItemView {{
        background: #3A3A3A;
        color: #E6E6E6;
        selection-background-color: #555;
        selection-color: #FFF;
    }}

    /* --- Spin box arrows: identical rendering on macOS + Windows --- */
    QDoubleSpinBox, QSpinBox {{
        padding-right: 20px;      /* leave room for the up/down buttons */
    }}
    QDoubleSpinBox::up-button, QSpinBox::up-button {{
        subcontrol-origin: border; subcontrol-position: top right;
        width: 16px; background: #3A3A3A;
        border-left: 1px solid #555; border-bottom: 1px solid #555;
    }}
    QDoubleSpinBox::down-button, QSpinBox::down-button {{
        subcontrol-origin: border; subcontrol-position: bottom right;
        width: 16px; background: #3A3A3A;
        border-left: 1px solid #555;
    }}
    QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover,
    QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {{
        background: #2E2E2E;
    }}
    QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {{
        image: url("{arrows['up']}"); width: 8px; height: 8px;
    }}
    QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {{
        image: url("{arrows['down']}"); width: 8px; height: 8px;
    }}
    QDoubleSpinBox::up-arrow:disabled, QSpinBox::up-arrow:disabled,
    QDoubleSpinBox::down-arrow:disabled, QSpinBox::down-arrow:disabled {{
        image: none;
    }}
    """
