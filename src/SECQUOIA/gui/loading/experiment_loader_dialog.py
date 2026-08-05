"""The Cytometric analysis loader window."""

from __future__ import annotations

import os
from functools import partial

from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from SECQUOIA.config import LINKS, TOOLTIPSTEXT
from SECQUOIA.core.cytometric_analysis import run_cytometric_analysis
from SECQUOIA.gui.common.ui_utils import (
    add_progress_bar,
    dpi_icon_size,
    make_help_button,
    restore_button_loading,
    set_button_icon,
    style_button_loading,
)
from SECQUOIA.gui.loading.experiment_paths import (
    _add_seg_tree_item,
    _collect_segmentation_folders,
    _find_analysis_dir,
    _find_basic_folders,
    _get_experiments_root,
    _install_experiment_mount_menu,
    _select_experiment_folder,
)

__all__ = [
    "create_experiment_loader_dialog",
    "open_experiment_loader_window",
]


def create_experiment_loader_dialog(
    main_window: QWidget,
) -> tuple[QDialog, QPushButton, QPushButton, QPushButton]:
    """Create the GUI for cytometric analysis.

    Returns the dialog along with its select/load/help buttons as
    `(dlg, select_btn, load_btn, help_btn)`.
    """

    dlg = QDialog(main_window)
    dlg.setWindowTitle("Load Experiment for Cytometric analysis")
    dlg.setModal(True)

    def px(v: int) -> int:
        """Scale a pixel value from 96-DPI units to the dialog's current DPI."""
        return int(v * dlg.logicalDpiX() / 96)

    _icon_px = dpi_icon_size(dlg).width()
    _set_btn_icon = partial(
        set_button_icon,
        size=_icon_px,
        reference=dlg,
        options=[{"color": "white"}],
    )

    root = QVBoxLayout(dlg)
    root.setContentsMargins(12, 12, 12, 12)
    root.setSpacing(10)

    # Header
    header = QHBoxLayout()
    title = QLabel("Cytometric analysis", dlg)
    title.setStyleSheet("font-weight:700; font-size:14pt;")
    header.addWidget(title)
    header.addStretch(1)

    help_btn = make_help_button(
        dlg, TOOLTIPSTEXT.HELP_CYTOMETRIC, LINKS.GITHUB_CYTOMETRIC
    )
    header.addWidget(help_btn, 0, Qt.AlignRight)
    root.addLayout(header)

    # Boxed load row
    box = QGroupBox("", dlg)
    box_l = QVBoxLayout(box)
    box_l.setContentsMargins(10, 10, 10, 10)
    box_l.setSpacing(8)

    row = QHBoxLayout()
    row.setSpacing(8)
    row.addWidget(QLabel("Experiment folder:", box))

    select_btn = QPushButton("Select", box)

    mounted_root = _get_experiments_root()
    if mounted_root:
        select_btn.setToolTip(
            f"Mounted experiments path:\n{mounted_root}\n\n"
            "Left-click to select an experiment folder.\n"
            "Right-click to change the mounted path."
        )
    else:
        select_btn.setToolTip(
            TOOLTIPSTEXT.SELECT_BTN
            + "\n\nRight-click to mount experiments path."
        )

    _install_experiment_mount_menu(select_btn, main_window)

    load_btn = QPushButton("Load Experiment Data", box)
    load_btn.setToolTip(TOOLTIPSTEXT.LOAD_BTN_CYTOMETRIC)
    load_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    select_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    # Map for existing handlers + requested icon call
    _set_btn_icon(load_btn, "mdi.file-restore-outline")
    _set_btn_icon(select_btn, "mdi.folder-open-outline")

    row.addWidget(select_btn)
    row.addWidget(load_btn)
    row.addStretch(1)
    box_l.addLayout(row)
    root.addWidget(box)

    # Boxed options
    opt = QGroupBox("", dlg)
    opt_l = QVBoxLayout(opt)
    opt_l.setContentsMargins(10, 10, 10, 10)
    opt_l.setSpacing(8)

    image_format_row = QHBoxLayout()
    image_format_row.setSpacing(8)
    image_format_row.addWidget(QLabel("Image Format:", opt))
    image_format_combo = QComboBox(opt)
    image_format_combo.addItems(["png", "tif", "jpg"])
    image_format_combo.setToolTip(TOOLTIPSTEXT.IMAGEFORMAT)
    image_format_row.addWidget(image_format_combo, 1)
    opt_l.addLayout(image_format_row)

    seg_header_row = QHBoxLayout()
    seg_header_row.setSpacing(8)

    seg_label = QLabel("Segmentation mask:", opt)

    seg_select_all_btn = QPushButton("Select all", opt)
    seg_unselect_all_btn = QPushButton("Unselect all", opt)
    seg_select_all_btn.setEnabled(False)
    seg_unselect_all_btn.setEnabled(False)

    seg_select_all_btn.setToolTip(TOOLTIPSTEXT.SELECT_ALL_BTN)
    seg_unselect_all_btn.setToolTip(TOOLTIPSTEXT.UNSELECT_ALL_BTN)
    _set_btn_icon(seg_select_all_btn, "mdi.check-all")
    _set_btn_icon(seg_unselect_all_btn, "mdi.selection-off")

    seg_header_row.addWidget(seg_label)
    seg_header_row.addStretch(1)
    seg_header_row.addWidget(seg_select_all_btn)
    seg_header_row.addWidget(seg_unselect_all_btn)
    opt_l.addLayout(seg_header_row)

    seg_tree = QTreeWidget(opt)
    seg_tree.setHeaderLabels(["Select existing Segmentation (One or more)"])
    header_font = seg_tree.header().font()
    header_font.setPointSize(header_font.pointSize())
    header_font.setBold(True)
    seg_tree.header().setFont(header_font)

    tree_font = seg_tree.font()
    tree_font.setPointSize(tree_font.pointSize())
    seg_tree.setFont(tree_font)

    seg_tree.setColumnCount(1)
    seg_tree.setUniformRowHeights(True)
    seg_tree.setRootIsDecorated(False)
    seg_tree.setToolTip(TOOLTIPSTEXT.SEG_TREE_INFO)
    seg_tree.headerItem().setToolTip(0, TOOLTIPSTEXT.SEG_TREE)
    seg_tree.setEnabled(False)
    opt_l.addWidget(seg_tree)

    seg_format_row = QHBoxLayout()
    seg_format_row.setSpacing(8)
    seg_format_row.addWidget(QLabel("Segmentation Format:", opt))
    seg_format_combo = QComboBox(opt)
    seg_format_combo.addItems(["png", "tif", "jpg"])
    seg_format_combo.setToolTip(TOOLTIPSTEXT.IMAGEFORMAT)
    seg_format_row.addWidget(seg_format_combo, 1)
    opt_l.addLayout(seg_format_row)

    threshold_row = QHBoxLayout()
    threshold_row.setSpacing(8)
    threshold_row.addWidget(QLabel("Distance between masks (px):", opt))
    threshold_spin = QSpinBox(opt)
    threshold_spin.setRange(0, 10**9)
    threshold_spin.setSingleStep(5)
    threshold_spin.setValue(20)
    threshold_spin.setKeyboardTracking(True)
    threshold_spin.setToolTip(TOOLTIPSTEXT.THRESHOLD)
    threshold_row.addWidget(threshold_spin, 1)
    opt_l.addLayout(threshold_row)

    basic_row = QHBoxLayout()
    basic_row.setSpacing(8)
    basic_row.addWidget(QLabel("BaSiC:", opt))

    basic_combo = QComboBox(opt)
    basic_combo.setToolTip(TOOLTIPSTEXT.BASIC_FOLDER_CYTOMETRIC)
    basic_combo.setEnabled(False)
    basic_chk = QCheckBox(opt)
    basic_chk.setEnabled(False)
    basic_chk.setToolTip(TOOLTIPSTEXT.BASIC_CYTOMETRIC)
    basic_row.addWidget(basic_combo, 1)
    basic_row.addWidget(basic_chk)
    opt_l.addLayout(basic_row)

    root.addWidget(opt)
    footer = QHBoxLayout()

    progress, _ = add_progress_bar(None)
    progress.setMinimumHeight(
        max(px(18), int(dlg.fontMetrics().height() * 1.2))
    )
    progress.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    run_btn = QPushButton("Run", dlg)
    run_btn.setToolTip(TOOLTIPSTEXT.RUN_CYTOMETRIC_BTN)
    _set_btn_icon(run_btn, "mdi.play")

    footer.addWidget(progress, 1)
    footer.addWidget(run_btn)
    root.addLayout(footer)
    root.addStretch(1)

    dlg.seg_tree = seg_tree
    dlg.basic_combo = basic_combo
    dlg.basic_chk = basic_chk
    dlg.threshold_spin = threshold_spin
    dlg.run_btn = run_btn
    dlg.progress_bar = progress

    def _sync_seg():
        """Synchronize selected segmentation paths and mask count with the tree state."""
        selected = []
        for i in range(seg_tree.topLevelItemCount()):
            it = seg_tree.topLevelItem(i)
            if it.checkState(0) == Qt.Checked:
                full = it.data(0, Qt.UserRole)
                if full:
                    selected.append(full)

        main_window.segmentation_paths = selected
        main_window.n_masks = len(selected)

    def _populate_loader_segmentation_ui(seg_names):
        """Populate the loader segmentation tree with detected segmentation folders."""
        seg_tree.blockSignals(True)
        seg_tree.clear()

        analysis_dir = _find_analysis_dir(main_window)

        if seg_names:
            for name in seg_names:
                _add_seg_tree_item(seg_tree, name, analysis_dir)

            seg_tree.setEnabled(True)
            seg_select_all_btn.setEnabled(True)
            seg_unselect_all_btn.setEnabled(True)
        else:
            item = QTreeWidgetItem(["(no Segmentation folders found)"])
            seg_tree.addTopLevelItem(item)
            seg_tree.setEnabled(False)
            seg_select_all_btn.setEnabled(False)
            seg_unselect_all_btn.setEnabled(False)

        seg_tree.blockSignals(False)
        _sync_seg()

    def _sync_basic(_state=None):
        """Synchronize BaSiC background-correction settings."""
        ok = basic_chk.isChecked() and basic_combo.count() > 0
        basic_combo.setEnabled(ok)

        analysis = _find_analysis_dir(main_window)
        sel = basic_combo.currentText()
        if ok and analysis and sel and not sel.startswith("("):
            main_window.background_correction_path = os.path.join(
                analysis, sel
            )
        else:
            main_window.background_correction_path = None

        if hasattr(main_window, "basic"):
            main_window.basic.flag = basic_chk.isChecked()
        main_window.use_background_correction = basic_chk.isChecked()

    def _on_select_folder(*_):
        """Handle experiment folder selection and update the "Select" button tooltip."""
        old = getattr(main_window, "folder_button", None)
        main_window.folder_button = select_btn
        try:
            _select_experiment_folder(main_window, dlg)
        finally:
            main_window.folder_button = old

        folder = getattr(main_window, "folder", None)
        if folder:
            select_btn.setToolTip(folder)

    basic_chk.toggled.connect(lambda *_: _sync_basic())
    basic_combo.currentTextChanged.connect(lambda *_: _sync_basic())
    select_btn.clicked.connect(_on_select_folder)
    image_format_combo.currentTextChanged.connect(lambda *_: _sync_formats())
    seg_format_combo.currentTextChanged.connect(lambda *_: _sync_formats())
    seg_tree.itemChanged.connect(lambda *_: _sync_seg())

    def _set_all_seg_checked(state):
        """Check or uncheck all selectable segmentation tree items."""
        seg_tree.blockSignals(True)
        for i in range(seg_tree.topLevelItemCount()):
            it = seg_tree.topLevelItem(i)
            if it.flags() & Qt.ItemIsUserCheckable:
                it.setCheckState(0, state)
        seg_tree.blockSignals(False)
        _sync_seg()

    seg_select_all_btn.clicked.connect(
        lambda: _set_all_seg_checked(Qt.Checked)
    )
    seg_unselect_all_btn.clicked.connect(
        lambda: _set_all_seg_checked(Qt.Unchecked)
    )

    def _on_load_experiment_clicked():
        """Load experiment metadata and populate segmentation and BaSiC options."""
        if not getattr(main_window, "folder", None):
            QMessageBox.warning(
                dlg,
                "Select Experiment Folder",
                "Please select an experiment folder first.",
            )
            return

        style_button_loading(load_btn)
        try:
            seg_names = _collect_segmentation_folders(main_window)
            _populate_loader_segmentation_ui(seg_names)

            basic_names = _find_basic_folders(main_window)
            basic_combo.clear()
            if basic_names:
                basic_combo.addItems(basic_names)
                basic_chk.setEnabled(True)
            else:
                basic_combo.addItem("(no BaSiC folders found)")
                basic_chk.setChecked(False)
                basic_chk.setEnabled(False)

            _sync_seg()
            _sync_basic()
        finally:
            restore_button_loading(load_btn, fallback_text="Load Experiment")

    load_btn.clicked.connect(_on_load_experiment_clicked)

    def _sync_formats():
        """Store the selected image and segmentation formats on the main window."""
        main_window.image_format = image_format_combo.currentText()
        main_window.seg_format = seg_format_combo.currentText()

    run_btn.clicked.connect(
        lambda *_: (
            _sync_formats(),
            _sync_basic(),
            run_cytometric_analysis(main_window),
        )
    )

    dlg.setSizeGripEnabled(True)
    dlg.adjustSize()
    screen = dlg.screen() or QGuiApplication.primaryScreen()
    avail = screen.availableGeometry()
    dlg.resize(min(dlg.width(), int(avail.width() * 0.9)), dlg.height())
    return dlg, select_btn, load_btn, help_btn


def open_experiment_loader_window(main_window: QWidget, *_) -> QDialog:
    """Opens the experiment loader as a non-modal independent tool window."""
    # Reuse existing window if already open
    dlg = getattr(main_window, "experiment_loader_window", None)
    if dlg is not None:
        dlg.raise_()
        dlg.activateWindow()
        return dlg

    dlg, select_btn, load_btn, help_btn = create_experiment_loader_dialog(
        main_window
    )

    # Make it independent
    dlg.setModal(False)
    dlg.setWindowModality(Qt.NonModal)
    dlg.setWindowFlag(Qt.Tool, True)
    dlg.setAttribute(Qt.WA_DeleteOnClose, True)

    main_window.experiment_loader_window = dlg
    dlg.destroyed.connect(
        lambda *_: setattr(main_window, "experiment_loader_window", None)
    )

    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    return dlg
