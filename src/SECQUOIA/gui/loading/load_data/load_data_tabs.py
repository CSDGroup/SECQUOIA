"""The four tabs of the Load Data dialog: Tracking, Segmentation Options,
Channel Selection, and Loading."""

from __future__ import annotations

import contextlib
import os
from functools import partial

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)

from SECQUOIA.config import TOOLTIPSTEXT
from SECQUOIA.gui.common.ui_utils import (
    add_progress_bar,
    dpi_icon_size,
    equalize_min_widths,
    form_row,
    reuse_widget,
    set_fixed_width,
    set_min_expanding,
    set_tree_rows,
    themed_icon,
    use_readable_combo_popup_on_macos,
    widen_to_hint,
)
from SECQUOIA.gui.loading.experiment_paths import (
    _get_experiments_root,
    _get_tracking_root,
    _install_experiment_mount_menu,
    _install_tracking_mount_menu,
    _make_bg_sync_handler,
    _select_experiment_folder,
    _select_tracking_folder,
    _update_load_button_enabled,
)
from SECQUOIA.gui.loading.load_data.load_data_handlers import (
    _on_load_data_clicked,
    _on_run_clicked,
)
from SECQUOIA.gui.loading.load_data.load_data_summary import _update_summary
from SECQUOIA.gui.loading.load_data.load_data_tracking_tree import (
    _select_all_tracking_tree,
)

__all__ = [
    "build_tracking_tab",
    "build_segmentation_tab",
    "build_channel_tab",
    "build_loading_tab",
]

# Tracking labels
_TRACKING_SELECTOR_LABELS = {
    "Ultrack": "UltrackTrackWorkFolder:",
    "CTC": "CTCWorkFolder:",
    "btrack": "btrackTrackWorkFolder:",
}
CURATION_GREEN = "#1db954"


def _set_btn_icon(btn, name, **icon_kw):
    """Set a themed icon at the screen's DPI-scaled size."""
    btn.setIcon(themed_icon(name, **icon_kw))
    btn.setIconSize(dpi_icon_size(btn))


# Tracking Tab
def build_tracking_tab(main_window, UiSize):
    """Build the tracking setup tab for experiment and tracking inputs."""
    tab = QWidget()
    t_v = QVBoxLayout(tab)
    t_v.setContentsMargins(8, 8, 8, 8)
    t_v.setSpacing(8)
    form = QFormLayout()
    form.setHorizontalSpacing(12)
    form.setVerticalSpacing(8)

    # Project name
    row = QWidget()
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(8)
    project_input = QLineEdit()
    project_input.setText("Project_1")
    project_input.setMaxLength(128)
    set_fixed_width(project_input, UiSize.select_w)

    def _sync_project_name(text: str):
        """Store the edited project name on the main window."""
        main_window.project_name = text.strip()

    project_input.textChanged.connect(_sync_project_name)
    _sync_project_name(project_input.text())
    h.addWidget(project_input)
    h.addStretch()
    form.addRow("Project name:", row)

    # User initials
    row = QWidget()
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(8)
    user_input = QLineEdit()
    user_input.setToolTip(TOOLTIPSTEXT.USER)
    user_input.setText("MA")
    user_input.setMaxLength(16)
    set_fixed_width(user_input, UiSize.select_w)
    existing_user = getattr(main_window, "user", "")
    if existing_user:
        user_input.setText(str(existing_user))

    def _sync_user_initials(text: str):
        """Store the edited user initials on the main window."""
        main_window.user = text.strip()

    user_input.textChanged.connect(_sync_user_initials)
    _sync_user_initials(user_input.text())
    h.addWidget(user_input)
    h.addStretch()
    form.addRow("User:", row)

    # Experiment folder
    row = QWidget()
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(8)
    main_window.folder_button = QPushButton("Select")
    _set_btn_icon(main_window.folder_button, "mdi.folder-open-outline")

    mounted_root = _get_experiments_root()
    if mounted_root:
        main_window.folder_button.setToolTip(
            f"Mounted experiments path:\n{mounted_root}\n\n"
            "Left-click to select an experiment folder.\n"
            "Right-click to change the mounted path."
        )
    else:
        main_window.folder_button.setToolTip(
            TOOLTIPSTEXT.EXPFOLDER
            + "\n\nRight-click to mount experiments path."
        )

    main_window.folder_button.clicked.connect(
        partial(_select_experiment_folder, main_window)
    )

    _install_experiment_mount_menu(main_window.folder_button, main_window)

    set_fixed_width(main_window.folder_button, UiSize.select_w)

    cur_folder = getattr(main_window, "folder", None)
    if cur_folder:
        main_window.folder_button.setText(os.path.basename(cur_folder))
    h.addWidget(main_window.folder_button)
    h.addStretch()
    form.addRow("Experiment folder:", row)

    # Image Format
    row = QWidget()
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(8)

    reuse_widget(main_window, "image_format_combo", QComboBox)
    if main_window.image_format_combo.count() == 0:
        main_window.image_format_combo.addItems(["png", "tif", "jpg"])
        main_window.image_format_combo.setToolTip(TOOLTIPSTEXT.IMAGEFORMAT)

    main_window.image_format_combo.setSizeAdjustPolicy(
        QComboBox.AdjustToMinimumContentsLengthWithIcon
    )
    main_window.image_format_combo.setMinimumContentsLength(10)
    set_fixed_width(main_window.image_format_combo, UiSize.select_w)
    use_readable_combo_popup_on_macos(main_window.image_format_combo)

    h.addWidget(main_window.image_format_combo, 0)
    h.addStretch()
    form.addRow("Image format:", row)

    # Tracking format
    row = QWidget()
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(8)
    reuse_widget(main_window, "tracking_format_combo", QComboBox)
    if main_window.tracking_format_combo.count() == 0:
        main_window.tracking_format_combo.addItems(
            [
                "btrack",
                "CTC",
                "tTt",
                "Ultrack",
            ]
        )
        main_window.tracking_format_combo.setToolTip(
            TOOLTIPSTEXT.TRACKINGFORMAT
        )
        main_window.tracking_format_combo.setCurrentText("tTt")
    main_window.tracking_format_combo.setSizeAdjustPolicy(
        QComboBox.AdjustToMinimumContentsLengthWithIcon
    )
    main_window.tracking_format_combo.setMinimumContentsLength(10)
    set_fixed_width(main_window.tracking_format_combo, UiSize.select_w)
    use_readable_combo_popup_on_macos(main_window.tracking_format_combo)
    h.addWidget(main_window.tracking_format_combo, 0)
    h.addStretch()
    form.addRow("Tracking format:", row)

    # Format-specific selector
    main_window.tracking_selector_row = QWidget()
    ts_h = QHBoxLayout(main_window.tracking_selector_row)
    ts_h.setContentsMargins(0, 0, 0, 0)
    ts_h.setSpacing(8)

    main_window.tracking_button = QPushButton("Select")
    _set_btn_icon(main_window.tracking_button, "mdi.file-outline")
    set_fixed_width(main_window.tracking_button, UiSize.select_w)
    _install_tracking_mount_menu(main_window.tracking_button, main_window)
    ts_h.addWidget(main_window.tracking_button, 0)

    # Will be included in the future
    # main_window.cp_tracking_chk = QCheckBox(
    #    "Enable cross-position tracking"
    # )
    # main_window.cp_tracking_chk.setToolTip(TOOLTIPSTEXT.CP)
    # main_window.cp_tracking = False
    # main_window.cp_tracking_chk.stateChanged.connect(
    #    lambda s: setattr(main_window, "cp_tracking", s == Qt.Checked)
    # )
    # main_window.cp_tracking_chk.stateChanged.connect(
    #    lambda _: _update_summary(main_window)
    # )
    # ts_h.addWidget(main_window.cp_tracking_chk, 0)
    main_window.cp_tracking = False
    ts_h.addStretch(1)

    main_window.tracking_selector_label = QLabel("")

    def _wire_tracking_controls(fmt: str):
        """Configure tracking controls for the selected tracking format."""
        with contextlib.suppress(TypeError, RuntimeError):
            main_window.tracking_button.clicked.disconnect()

        # Will be included in the future
        # main_window.cp_tracking_chk.setVisible(fmt == "tTt")
        main_window.tracking_path = None
        main_window.tracking_button.setText("Select")

        dialog_fmt = fmt if fmt in _TRACKING_SELECTOR_LABELS else "tTt"
        selector_label = _TRACKING_SELECTOR_LABELS.get(fmt, "tTtWorkFolder:")

        _set_btn_icon(main_window.tracking_button, "mdi.folder-open-outline")
        main_window.tracking_button.setToolTip(f"Select {dialog_fmt} folder")
        main_window.tracking_button.clicked.connect(
            partial(
                _select_tracking_folder,
                main_window,
                dialog_fmt,
                f"Select {dialog_fmt} folder",
            )
        )
        main_window.tracking_selector_label.setText(selector_label)

        mounted_tracking_root = _get_tracking_root(fmt)
        if mounted_tracking_root:
            main_window.tracking_button.setToolTip(
                f"Mounted {fmt} tracking path:\n{mounted_tracking_root}\n\n"
                "Left-click to select a tracking folder.\n"
                "Right-click to change the mounted path."
            )
        else:
            current_tip = main_window.tracking_button.toolTip()
            main_window.tracking_button.setToolTip(
                current_tip + "\n\nRight-click to mount tracking path."
            )
        _update_load_button_enabled(main_window)
        _update_summary(main_window)

    def _on_fmt_change(_txt):
        """Handle tracking-format changes by rewiring related controls."""
        _wire_tracking_controls(
            main_window.tracking_format_combo.currentText()
        )

    main_window.tracking_format_combo.currentTextChanged.connect(
        _on_fmt_change
    )
    _on_fmt_change(main_window.tracking_format_combo.currentText())

    form.addRow(
        main_window.tracking_selector_label,
        main_window.tracking_selector_row,
    )

    # Load Experiment
    load_row = QWidget()
    load_h = QHBoxLayout(load_row)
    load_h.setContentsMargins(0, 0, 0, 0)
    load_h.setSpacing(8)
    main_window.load_data_btn = QPushButton("Load Experiment")
    _set_btn_icon(main_window.load_data_btn, "mdi.file-restore-outline")
    main_window.load_data_btn.setToolTip(TOOLTIPSTEXT.LOAD_DATA_BTN)
    main_window.load_data_btn.clicked.connect(
        partial(_on_load_data_clicked, main_window)
    )
    set_fixed_width(main_window.load_data_btn, UiSize.select_w)
    load_h.addStretch(1)
    load_h.addWidget(main_window.load_data_btn, 0)
    form.addRow("", load_row)

    # Result tree
    main_window.tracking_tree_group = QGroupBox("")
    main_window.tracking_tree_group.setVisible(False)
    g_v = QVBoxLayout(main_window.tracking_tree_group)
    g_v.setContentsMargins(8, 8, 8, 8)
    g_v.setSpacing(8)

    actions = QWidget()
    a_h = QHBoxLayout(actions)
    a_h.setContentsMargins(0, 0, 0, 0)
    a_h.setSpacing(8)
    btn_sel_all = QPushButton("Select all")
    btn_sel_all.setToolTip(TOOLTIPSTEXT.SELECT_ALL_BTN)
    _set_btn_icon(btn_sel_all, "mdi.check-all")
    btn_unsel_all = QPushButton("Unselect all")
    btn_unsel_all.setToolTip(TOOLTIPSTEXT.UNSELECT_ALL_BTN)
    _set_btn_icon(btn_unsel_all, "mdi.selection-off")
    widen_to_hint(btn_sel_all)
    widen_to_hint(btn_unsel_all)
    btn_sel_all.clicked.connect(
        lambda: _select_all_tracking_tree(main_window, True)
    )
    btn_unsel_all.clicked.connect(
        lambda: _select_all_tracking_tree(main_window, False)
    )
    a_h.addStretch()
    a_h.addWidget(btn_sel_all)
    a_h.addWidget(btn_unsel_all)
    g_v.addWidget(actions)

    main_window.tracking_tree = QTreeWidget()
    main_window.tracking_tree.setColumnCount(1)
    main_window.tracking_tree.setHeaderLabels(["Cell Lineage Tree"])
    main_window.tracking_tree.headerItem().setToolTip(
        0, TOOLTIPSTEXT.TRACKING_TREE
    )
    main_window.tracking_tree.setUniformRowHeights(True)
    main_window.tracking_tree.setRootIsDecorated(True)
    main_window.tracking_tree.setExpandsOnDoubleClick(True)
    main_window.tracking_tree.setToolTip(TOOLTIPSTEXT.TRACKING_TREE_INFO)
    g_v.addWidget(main_window.tracking_tree)
    main_window.tracking_counts_label = QLabel("")
    g_v.addWidget(main_window.tracking_counts_label)

    t_v.addLayout(form)
    t_v.addWidget(main_window.tracking_tree_group)
    t_v.addStretch(1)

    set_tree_rows(main_window.tracking_tree, approx_rows=10)
    return tab


# Segmentation Options Tab
def build_segmentation_tab(main_window, UiSize):
    """Build the segmentation options tab."""
    tab = QWidget()
    t_v = QVBoxLayout(tab)
    t_v.setContentsMargins(8, 8, 8, 8)
    t_v.setSpacing(8)

    seg_group = QGroupBox("")
    seg_v = QVBoxLayout(seg_group)
    seg_v.setContentsMargins(8, 8, 8, 8)
    seg_v.setSpacing(8)

    main_window.seg_tree = QTreeWidget()
    main_window.seg_tree.setHeaderLabels(
        ["Select existing Segmentation (One or more)"]
    )
    main_window.seg_tree.setColumnCount(1)
    main_window.seg_tree.setUniformRowHeights(True)
    main_window.seg_tree.setRootIsDecorated(False)
    main_window.seg_tree.setToolTip(TOOLTIPSTEXT.SEG_TREE_INFO)
    main_window.seg_tree.headerItem().setToolTip(0, TOOLTIPSTEXT.SEG_TREE)
    seg_v.addWidget(main_window.seg_tree)

    t_v.addWidget(seg_group)

    # Background correction widgets
    main_window.bg_correct_chk = QCheckBox()
    main_window.bg_correct_chk.setToolTip(TOOLTIPSTEXT.BG)
    main_window.bg_correct_combo = QComboBox()
    main_window.bg_correct_combo.setToolTip(TOOLTIPSTEXT.BG_COMBO)
    main_window.bg_correct_combo.setSizeAdjustPolicy(
        QComboBox.AdjustToMinimumContentsLengthWithIcon
    )
    main_window.bg_correct_combo.setMinimumContentsLength(0)
    set_fixed_width(main_window.bg_correct_combo, UiSize.min_w_select)

    _sync_bg_ui = _make_bg_sync_handler(
        main_window, on_sync=lambda: _update_summary(main_window)
    )

    main_window.bg_correct_chk.stateChanged.connect(_sync_bg_ui)
    main_window.bg_correct_combo.currentTextChanged.connect(
        lambda _t: _sync_bg_ui()
    )

    # Import Real time [ms] widgets
    main_window.import_rt_chk = QCheckBox()
    main_window.import_rt_chk.setToolTip(
        "Enable importing real-time (ms) from an 'images*.csv' file in the experiment folder."
    )

    main_window.import_rt_combo = QComboBox()
    main_window.import_rt_combo.setToolTip(
        "Select an 'images*.csv' file detected in the experiment folder."
    )
    main_window.import_rt_combo.setSizeAdjustPolicy(
        QComboBox.AdjustToMinimumContentsLengthWithIcon
    )
    main_window.import_rt_combo.setMinimumContentsLength(0)
    set_min_expanding(main_window.import_rt_combo, UiSize.min_w_editor)

    main_window.use_import_rt = False
    main_window.import_rt_path = None

    def _sync_import_rt_ui(_state=None):
        """Synchronize real-time import UI state and selected CSV path."""
        ok = (
            main_window.import_rt_chk.isChecked()
            and main_window.import_rt_combo.count() > 0
        )
        main_window.import_rt_combo.setEnabled(ok)
        folder = getattr(main_window, "folder", None)
        sel = main_window.import_rt_combo.currentText()
        if ok and folder and sel and not sel.startswith("("):
            main_window.import_rt_path = os.path.join(folder, sel)
        else:
            main_window.import_rt_path = None
        main_window.use_import_rt = main_window.import_rt_chk.isChecked()
        _update_summary(main_window)

    main_window.import_rt_chk.stateChanged.connect(_sync_import_rt_ui)
    main_window.import_rt_combo.currentTextChanged.connect(
        lambda _t: _sync_import_rt_ui()
    )

    # Editors
    threshold_spin = QSpinBox()
    threshold_spin.setRange(0, 10**9)
    threshold_spin.setSingleStep(5)
    threshold_spin.setValue(20)
    threshold_spin.setKeyboardTracking(True)
    threshold_spin.setToolTip(TOOLTIPSTEXT.THRESHOLD)

    min_mask_size_spin = QSpinBox()
    min_mask_size_spin.setRange(0, 10**9)
    min_mask_size_spin.setSingleStep(10)
    min_mask_size_spin.setValue(5)
    min_mask_size_spin.setToolTip(TOOLTIPSTEXT.AREA)

    set_min_expanding(main_window.bg_correct_combo, UiSize.min_w_editor)
    set_min_expanding(threshold_spin, UiSize.min_w_editor)
    set_min_expanding(min_mask_size_spin, UiSize.min_w_editor)

    # BaSiC background correction
    bg_row = QWidget()
    bg_h = QHBoxLayout(bg_row)
    bg_h.setContentsMargins(0, 0, 0, 0)
    bg_h.setSpacing(8)
    bg_h.addWidget(main_window.bg_correct_combo, 0)
    bg_h.addWidget(main_window.bg_correct_chk, 0)
    bg_h.addStretch(1)
    t_v.addLayout(form_row("Select BaSiC background correction:", bg_row))

    # Import Real time [ms]
    rt_row = QWidget()
    rt_h = QHBoxLayout(rt_row)
    rt_h.setContentsMargins(0, 0, 0, 0)
    rt_h.setSpacing(8)
    rt_h.addWidget(main_window.import_rt_combo, 0)
    rt_h.addWidget(main_window.import_rt_chk, 0)
    rt_h.addStretch(1)
    t_v.addLayout(form_row("Import Real time [ms]:", rt_row))

    # Minimum cell size
    min_size_row = QWidget()
    min_size_h = QHBoxLayout(min_size_row)
    min_size_h.setContentsMargins(0, 0, 0, 0)
    min_size_h.setSpacing(8)
    min_size_h.addWidget(min_mask_size_spin, 0)
    min_size_h.addStretch()
    t_v.addLayout(form_row("Minimum cell size (px):", min_size_row))

    # Maximum distance
    distance_row = QWidget()
    distance_h = QHBoxLayout(distance_row)
    distance_h.setContentsMargins(0, 0, 0, 0)
    distance_h.setSpacing(8)
    distance_h.addWidget(threshold_spin, 0)
    distance_h.addStretch()
    t_v.addLayout(
        form_row(
            "Maximum distance of trackpoint from segmentation mask:",
            distance_row,
        )
    )

    main_window._threshold_spin = threshold_spin
    main_window._min_mask_size_spin = min_mask_size_spin
    main_window.threshold = threshold_spin.value()
    main_window.min_mask_size = min_mask_size_spin.value()
    threshold_spin.valueChanged.connect(
        lambda v: setattr(main_window, "threshold", v)
    )
    min_mask_size_spin.valueChanged.connect(
        lambda v: setattr(main_window, "min_mask_size", v)
    )
    threshold_spin.valueChanged.connect(
        lambda _v: _update_summary(main_window)
    )
    min_mask_size_spin.valueChanged.connect(
        lambda _v: _update_summary(main_window)
    )

    set_tree_rows(main_window.seg_tree, approx_rows=10)
    t_v.addStretch(1)
    return tab


# Channel Selection Tab
def build_channel_tab(main_window, UiSize):
    """Build the channel selection tab and time/position controls."""
    tab = QWidget()
    t_v = QVBoxLayout(tab)
    t_v.setContentsMargins(8, 8, 8, 8)
    t_v.setSpacing(8)

    ch_group = QGroupBox("")
    ch_v = QVBoxLayout(ch_group)
    ch_v.setContentsMargins(8, 8, 8, 8)
    ch_v.setSpacing(8)

    main_window.chan_tree = QTreeWidget()
    main_window.chan_tree.setUniformRowHeights(True)
    main_window.chan_tree.setRootIsDecorated(False)
    main_window.chan_tree.setToolTip(TOOLTIPSTEXT.CH_TREE_INFO)
    if main_window.chan_tree.headerItem():
        main_window.chan_tree.headerItem().setToolTip(0, TOOLTIPSTEXT.CH_TREE)
    ch_v.addWidget(main_window.chan_tree)
    t_v.addWidget(ch_group)

    # Time point inputs
    main_window.time_min_spin = QSpinBox()
    main_window.time_min_spin.setToolTip(TOOLTIPSTEXT.MINTIME)
    main_window.time_min_spin.setRange(0, 10**9)

    main_window.time_max_spin = QSpinBox()
    main_window.time_max_spin.setToolTip(TOOLTIPSTEXT.MAXTIME)
    main_window.time_max_spin.setRange(0, 10**9)

    set_min_expanding(main_window.time_min_spin, UiSize.min_w_editor)
    set_min_expanding(main_window.time_max_spin, UiSize.min_w_editor)

    # Row: Min time point
    min_tp_row = QWidget()
    min_tp_h = QHBoxLayout(min_tp_row)
    min_tp_h.setContentsMargins(0, 0, 0, 0)
    min_tp_h.setSpacing(8)
    min_tp_h.addWidget(main_window.time_min_spin, 0)
    min_tp_h.addStretch()
    t_v.addLayout(form_row("Minimum time point for analysis:", min_tp_row))

    # Row: Max time point
    max_tp_row = QWidget()
    max_tp_h = QHBoxLayout(max_tp_row)
    max_tp_h.setContentsMargins(0, 0, 0, 0)
    max_tp_h.setSpacing(8)
    max_tp_h.addWidget(main_window.time_max_spin, 0)
    max_tp_h.addStretch()
    t_v.addLayout(form_row("Maximum time point for analysis:", max_tp_row))

    # Time [s] between time points
    main_window.time_input = QSpinBox()
    main_window.time_input.setToolTip(TOOLTIPSTEXT.TIME)
    main_window.time_input.setRange(0, 10**9)
    main_window.time_input.setSingleStep(30)
    main_window.time_input.setKeyboardTracking(True)
    cached = getattr(main_window, "time_interval", None)
    if cached is not None:
        cur_val = int(cached)
    else:
        try:
            ti = getattr(main_window, "Time_input", None)
            cur_val = int(ti.text()) if ti is not None else 300
        except (AttributeError, ValueError, RuntimeError):
            cur_val = 300
    main_window.time_input.setValue(cur_val)
    main_window.time_interval = float(cur_val)
    main_window.time_input.text = lambda: str(main_window.time_input.value())
    main_window.time_input.setText = lambda s: main_window.time_input.setValue(
        int(s)
    )
    main_window.Time_input = main_window.time_input
    set_min_expanding(main_window.time_input, UiSize.min_w_editor)

    time_input_row = QWidget()
    time_input_h = QHBoxLayout(time_input_row)
    time_input_h.setContentsMargins(0, 0, 0, 0)
    time_input_h.setSpacing(8)
    time_input_h.addWidget(main_window.time_input, 0)
    time_input_h.addStretch()
    t_v.addLayout(form_row("Time [s] between time points:", time_input_row))

    def _on_time_input_changed(v):
        """Mirror the spinbox value into a plain attribute so core
        modules can read the interval after this dialog is gone."""
        main_window.time_interval = float(v)
        _update_summary(main_window)

    main_window.time_input.valueChanged.connect(_on_time_input_changed)

    # Start position
    pos_start_row = QWidget()
    pos_start_h = QHBoxLayout(pos_start_row)
    pos_start_h.setContentsMargins(0, 0, 0, 0)
    pos_start_h.setSpacing(8)
    main_window.positions_start_combo = QSpinBox()
    main_window.positions_start_combo.setToolTip(TOOLTIPSTEXT.STARTPOSITION)
    set_min_expanding(main_window.positions_start_combo, UiSize.min_w_editor)
    pos_start_h.addWidget(main_window.positions_start_combo, 0)
    pos_start_h.addStretch()
    t_v.addLayout(form_row("Start position of curation:", pos_start_row))

    equalize_min_widths(
        [
            (
                main_window.bg_correct_combo
                if hasattr(main_window, "bg_correct_combo")
                else QWidget()
            ),
            main_window.import_rt_combo,
            main_window.time_min_spin,
            main_window.time_max_spin,
            main_window.time_input,
            main_window.positions_start_combo,
        ],
        target_width=UiSize.min_w_editor,
    )

    set_tree_rows(main_window.chan_tree, approx_rows=10)
    t_v.addStretch(1)
    return tab


# Loading Tab
def build_loading_tab(main_window, UiSize, dialog):
    """Build the loading summary tab with run controls and progress bars."""
    tab = QWidget()
    t_v = QVBoxLayout(tab)
    t_v.setContentsMargins(8, 8, 8, 8)
    t_v.setSpacing(8)

    # Summary panel
    summary_group = QGroupBox("Summary")
    summary_v = QVBoxLayout(summary_group)
    summary_v.setContentsMargins(10, 10, 10, 10)
    summary_v.setSpacing(8)
    summary_group.setStyleSheet("""
        QGroupBox {
            border: 1px solid #555555;
            border-radius: 4px;
            margin-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
            font-weight: 700;
            font-size: 16pt;
            color: #ffffff;
        }
    """)
    main_window.summary_label = QLabel("")
    main_window.summary_label.setWordWrap(True)
    main_window.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    main_window.summary_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
    summary_v.addWidget(main_window.summary_label)
    summary_v.addStretch(1)
    t_v.addWidget(summary_group)

    # Loading format
    loading_row = QWidget()
    loading_row.setToolTip(TOOLTIPSTEXT.LOADING_FORMAT)
    loading_h = QHBoxLayout(loading_row)
    loading_h.setContentsMargins(0, 0, 0, 0)
    loading_h.setSpacing(8)

    main_window.loading_all_chk = QCheckBox("All")
    main_window.loading_all_chk.setToolTip(TOOLTIPSTEXT.RUN_ALL)
    widen_to_hint(main_window.loading_all_chk, 10)

    main_window.loading_positions_chk = QCheckBox("Positions")
    main_window.loading_positions_chk.setToolTip(TOOLTIPSTEXT.RUN_POSITION)
    widen_to_hint(main_window.loading_positions_chk, 10)

    main_window.loading_positions_bg_chk = QCheckBox(
        "Positions+Background loading"
    )
    main_window.loading_positions_bg_chk.setToolTip(
        TOOLTIPSTEXT.RUN_POSITION_BACKGROUND
    )
    widen_to_hint(main_window.loading_positions_bg_chk, 10)

    main_window.loading_group = QButtonGroup(loading_row)
    main_window.loading_group.setExclusive(True)
    main_window.loading_group.addButton(main_window.loading_all_chk)
    main_window.loading_group.addButton(main_window.loading_positions_chk)
    main_window.loading_group.addButton(main_window.loading_positions_bg_chk)
    main_window.loading_positions_chk.setChecked(True)
    main_window.loading_format = "Positions"

    def _sync_loading_format(_btn=None):
        """Update the selected loading mode from the checkboxes."""
        if main_window.loading_all_chk.isChecked():
            main_window.loading_format = "All"
        elif main_window.loading_positions_bg_chk.isChecked():
            main_window.loading_format = "Positions+Background loading"
        else:
            main_window.loading_format = "Positions"
        _update_summary(main_window)

    main_window.loading_group.buttonClicked.connect(_sync_loading_format)
    _sync_loading_format()

    loading_h.addWidget(main_window.loading_positions_chk)
    loading_h.addWidget(main_window.loading_all_chk)
    loading_h.addStretch()
    set_fixed_width(loading_row, UiSize.numeric_w)

    frm = form_row("Loading format:", loading_row)
    frm.setStretch(1, 0)
    frm.setAlignment(loading_row, Qt.AlignLeft)
    t_v.addLayout(frm)

    # Progress bars
    main_window.global_progress_bar, _ = add_progress_bar(t_v)
    main_window.global_progress_bar.setFormat("Position %v/%m")
    main_window.global_progress_bar.setRange(0, 1)
    main_window.global_progress_bar.setVisible(False)
    main_window.global_progress_bar.setSizePolicy(
        QSizePolicy.Expanding, QSizePolicy.Fixed
    )

    main_window.progress_bar, progress = add_progress_bar(t_v)
    main_window.set_progress = progress["set"]
    main_window.bump_progress = progress["bump"]
    main_window.finish_progress = progress["finish"]
    main_window.set_progress_step = progress["step"]

    main_window.progress_msg_label = QLabel("")
    main_window.progress_msg_label.setObjectName("progressMsgLabel")
    main_window.progress_msg_label.setWordWrap(True)
    t_v.addWidget(main_window.progress_msg_label)

    # Run / Start curation
    run_row = QWidget()
    run_h = QHBoxLayout(run_row)
    run_h.setContentsMargins(0, 0, 0, 0)
    run_h.setSpacing(8)
    style = QApplication.style()

    main_window.apply_button = QPushButton("RUN")
    main_window.apply_button.setToolTip(TOOLTIPSTEXT.RUN_BTN)
    main_window.apply_button.setIcon(style.standardIcon(QStyle.SP_MediaPlay))
    main_window.apply_button.clicked.connect(
        partial(_on_run_clicked, main_window)
    )
    widen_to_hint(main_window.apply_button, 24)

    main_window.start_curation_btn = QPushButton("Start Curation")
    _set_btn_icon(
        main_window.start_curation_btn,
        "fa5s.arrow-right",
        options=[{"color": CURATION_GREEN, "color_disabled": "#5a5a5a"}],
    )
    main_window.start_curation_btn.setEnabled(False)
    main_window.start_curation_btn.setToolTip(TOOLTIPSTEXT.CURATION_BTN)
    main_window.start_curation_btn.clicked.connect(dialog.close)
    widen_to_hint(main_window.start_curation_btn, 24)
    main_window.start_curation_btn.hide()

    run_h.addStretch()
    run_h.addWidget(main_window.apply_button)
    run_h.addWidget(main_window.start_curation_btn)
    run_h.addStretch()
    t_v.addWidget(run_row)
    return tab
