"""Loading and switching what the application has open."""

from __future__ import annotations

import logging

from qtpy.QtCore import QCoreApplication, Qt, QTimer
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStyleFactory,
    QVBoxLayout,
    QWidget,
)

from SECQUOIA.config import LINKS, TOOLTIPSTEXT
from SECQUOIA.core.project_state import load_project_state
from SECQUOIA.gui.common.ui_utils import (
    add_progress_bar,
    drop_dead_widget_attrs,
    make_help_button,
)
from SECQUOIA.gui.loading_pipeline import run_loading
from SECQUOIA.gui.position_navigation import resolve_position_index
from SECQUOIA.resources import splash_pixmap

LOG = logging.getLogger(__name__)

__all__ = [
    "open_load_previous_project_gui",
    "show_splash_with_bar",
]


def show_splash_with_bar(app):
    """Show a splash screen with the SECQUOIA banner and a progress bar."""
    splash = QDialog()
    splash.setWindowTitle("Loading…")
    splash.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
    splash.setModal(False)

    layout = QVBoxLayout(splash)
    layout.setContentsMargins(20, 20, 20, 20)
    layout.setSpacing(15)

    label = QLabel()
    label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)

    pixmap = splash_pixmap()
    if pixmap is not None:
        screen = QGuiApplication.primaryScreen()
        ratio = screen.devicePixelRatio() if screen else 1.0
        pixmap = pixmap.scaledToWidth(
            int(800 * ratio), Qt.SmoothTransformation
        )
        pixmap.setDevicePixelRatio(ratio)
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)

    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    bar.setStyle(QStyleFactory.create("Fusion"))
    bar.setTextVisible(True)
    bar.setAlignment(Qt.AlignCenter)
    bar.setMinimumHeight(22)
    bar.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

    bar.setStyleSheet("""
        QProgressBar {
            background: #333;
            border: 1pt solid #222;
            border-radius: 6pt;
            color: white;
        }
        QProgressBar::chunk {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 #1e3a8a, stop:1 #60a5fa
            );
            border-radius: 6pt;
        }
        """)

    layout.addWidget(label)
    layout.addWidget(bar)

    splash.adjustSize()
    splash.show()
    app.processEvents()

    def progress_fn(value: int, text: str | None = None):
        """Update the splash progress bar and optional status text."""
        bar.setValue(max(0, min(100, value)))
        if text:
            bar.setFormat(text + " (%p%)")
        QCoreApplication.processEvents()

    def finish_fn(main_window):
        """Close the splash screen and return focus to the main window."""
        splash.close()
        splash.deleteLater()
        main_window.raise_()
        main_window.activateWindow()
        QCoreApplication.processEvents()

    return splash, progress_fn, finish_fn


def open_load_previous_project_gui(main_window: QWidget) -> None:
    """Open a dialog for selecting and loading a saved project metadata file."""
    d = QDialog(main_window)
    d.setWindowTitle("Load previous project")

    header = QHBoxLayout()
    t = QLabel("Load previous project")

    help_btn = make_help_button(
        d, TOOLTIPSTEXT.HELP_PROJECT, LINKS.GITHUB_LOADING_WINDOW
    )

    header.addWidget(t)
    header.addStretch(1)
    header.addWidget(help_btn, 0, Qt.AlignRight)

    le = QLineEdit()
    le.setPlaceholderText("Select project_metadata.json ...")
    le.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    bsel = QPushButton("Select")
    bsel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    brun = QPushButton("Run")
    brun.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    bexit = QPushButton("Exit")
    bexit.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    bexit.clicked.connect(d.reject)
    bexit.setToolTip(TOOLTIPSTEXT.EXIT_PROJECT)

    bsel.clicked.connect(
        lambda: (lambda p: le.setText(p) if p else None)(
            QFileDialog.getOpenFileName(
                d, "Select metadata JSON", "", "JSON files (*.json)"
            )[0]
        )
    )

    row = QHBoxLayout()
    row.addWidget(QLabel("Metadata (json.file):"))
    row.addWidget(le, 1)
    row.addWidget(bsel)

    le.setToolTip(TOOLTIPSTEXT.PROJECT_PATH)
    bsel.setToolTip(TOOLTIPSTEXT.SELECT_PROJECT_PATH)
    brun.setToolTip(TOOLTIPSTEXT.LOAD_PROJECT)

    pbar, _ = add_progress_bar(None)
    pbar.setToolTip(TOOLTIPSTEXT.PROGRESS_PROJECT)

    msg = QLabel("")
    msg.setWordWrap(True)
    msg.setStyleSheet("border: none; background: transparent;")
    msg.setToolTip(TOOLTIPSTEXT.MSG_LOADING_PROJECT)
    msg.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def _on_finish_progress():
        pbar.setValue(100)
        QApplication.restoreOverrideCursor()
        d.accept()

    main_window.set_progress = lambda v: pbar.setValue(int(v))
    main_window.finish_progress = _on_finish_progress
    main_window.progress_msg_label = msg

    def _run():
        """Load the selected project and bring its viewers and plots up."""
        try:
            drop_dead_widget_attrs(main_window)
            if not load_project_state(main_window, le.text().strip()):
                return
            QApplication.setOverrideCursor(Qt.WaitCursor)
            resolve_position_index(main_window)
            main_window.create_napari_viewers()
            main_window.add_viewers_to_layout2()
            main_window.update_channel_mask_dropdowns()
            main_window.add_plot_widgets()
            QTimer.singleShot(
                0, lambda: main_window._force_max_layout(ratio=0.30)
            )
            run_loading(main_window)
        except (RuntimeError, ValueError, OSError) as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(main_window, "Error", str(e))

    brun.clicked.connect(_run)

    panel = QWidget(d)
    panel.setObjectName("panel")
    panel.setStyleSheet("""
        QWidget#panel {
            background: transparent;        /* <-- no white box fill */
            border: 1px solid #e6e6e6;      /* <-- outline */
            border-radius: 8px;
        }
    """)
    panel_lay = QVBoxLayout(panel)
    panel_lay.setContentsMargins(12, 12, 12, 12)
    panel_lay.setSpacing(8)

    panel_lay.addLayout(row)
    panel_lay.addWidget(pbar)
    panel_lay.addWidget(msg)
    btn_row = QHBoxLayout()
    btn_row.addWidget(brun, 1)
    btn_row.addWidget(bexit)
    panel_lay.addLayout(btn_row)

    lay = QVBoxLayout(d)
    lay.addLayout(header)
    lay.addWidget(panel)

    (d.exec_() if hasattr(d, "exec_") else d.exec())
