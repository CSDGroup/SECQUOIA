"""Prepares Qt, then builds and shows the SECQUOIA main window."""

import contextlib
import os
import sys

os.environ.setdefault("QT_API", "pyside6")
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
os.environ.setdefault("QT_SCALE_FACTOR_ROUNDING_POLICY", "PassThrough")

if os.name == "nt":
    with contextlib.suppress(Exception):
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    with contextlib.suppress(Exception):
        import ctypes

        ctypes.windll.user32.SetProcessDPIAware()

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication


def set_windows_app_id(app_id: str = "SECQUOIA.SECQUOIA.desktop.1") -> None:
    """Give the process its own Windows AppUserModelID."""
    if os.name != "nt":
        return
    with contextlib.suppress(Exception):
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)


def main():
    """Launch the SECQUOIA desktop application."""
    from SECQUOIA.core.logging_setup import configure_logging
    from SECQUOIA.core.memmap_store import reap_stale_memmap_dirs
    from SECQUOIA.gui.loading.loading_dialogs import show_splash_with_bar
    from SECQUOIA.gui.main_window import MainWindow

    configure_logging()
    with contextlib.suppress(Exception):
        reap_stale_memmap_dirs()
    with contextlib.suppress(Exception):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    with contextlib.suppress(Exception):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    from SECQUOIA.resources import app_icon

    try:
        from SECQUOIA.config import STYLE

        f = app.font()
        sz = (
            int(getattr(STYLE, "FONT_SIZE", 10))
            if getattr(STYLE, "FONT_SIZE", None)
            else 10
        )
        f.setPointSize(max(8, sz))
        app.setFont(f)
    except (ImportError, AttributeError, TypeError, ValueError):
        pass

    app.setStyleSheet("""
        QWidget {
            background-color: #2b2b2b;
            color: #f0f0f0;
        }
        QPushButton {
            background-color: #3c3c3c;
            border: 1px solid #555;
            padding: 5px;
        }
        QPushButton:hover {
            background-color: #444444;
        }
        QHeaderView::section {
            background-color: #444444;
            color: #f0f0f0;
            font-weight: bold;
        }

        /* Dark menubar */
        QMenuBar {
            background-color: #2b2b2b;
            color: #f0f0f0;
            border: none;
        }
        QMenuBar::item {
            background: transparent;
            padding: 4px 10px;
        }
        QMenuBar::item:selected {
            background: #3c3c3c;
            color: #ffffff;
        }
        QMenuBar::item:disabled {
            color: #8a8a8a;
        }

        /* Dark popup menus */
        QMenu {
            background-color: #2b2b2b;
            color: #f0f0f0;
            border: 1px solid #555;
            padding: 4px;
        }
        QMenu::separator {
            height: 1px;
            background: #555;
            margin: 4px 6px;
        }
        QMenu::item {
            background: transparent;
            padding: 5px 18px;
        }
        QMenu::item:selected {
            background: #3c3c3c;
            color: #ffffff;
        }
        QMenu::item:disabled {
            color: #8a8a8a;
        }
        """)

    splash, progress, finish = show_splash_with_bar(app)
    progress(10, "Initializing…")

    main_window = MainWindow()

    set_windows_app_id()
    icon = app_icon()
    app.setWindowIcon(icon)
    main_window.setWindowIcon(icon)

    progress(70, "Finalizing…")

    main_window.show()
    progress(100, "Ready")
    finish(main_window)

    run = getattr(app, "exec", None) or getattr(app, "exec_", None)
    sys.exit(run())


if __name__ == "__main__":
    main()
