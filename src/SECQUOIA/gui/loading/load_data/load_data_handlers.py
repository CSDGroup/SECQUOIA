"""Click handlers for the 'Load Experiment' and 'RUN' buttons."""

from __future__ import annotations

import glob
import os

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QApplication, QMessageBox

from SECQUOIA.config import TOOLTIPSTEXT
from SECQUOIA.core.logging_setup import attach_experiment_log
from SECQUOIA.core.project_state import save_project_state
from SECQUOIA.core.tracking.track_data import (
    BTRACK_MISSING_MESSAGE,
    _load_tracking_data,
    btrack_is_available,
    filter_track_df_to_selected,
)
from SECQUOIA.gui.common.ui_utils import (
    restore_button_loading,
    style_button_loading,
)
from SECQUOIA.gui.loading.experiment_paths import (
    _collect_segmentation_folders,
    _confirm_overwrite_project_dir,
    _current_tracking_format,
    _detect_time_max_in_position_folder,
    _detect_unique_w_channels,
    _ensure_analysis_dirs,
    _find_basic_folders,
    _find_images_csvs,
)
from SECQUOIA.gui.loading.load_data.load_data_realtime_import import (
    _maybe_import_realtime,
    _maybe_load_ttt_channel_comments,
    _maybe_load_ttt_position_comments,
)
from SECQUOIA.gui.loading.load_data.load_data_reset import (
    _reset_dataset_state,
    _reset_loaded_experiment,
)
from SECQUOIA.gui.loading.load_data.load_data_segmentation_channels import (
    _populate_channels_ui,
    _populate_segmentation_ui,
)
from SECQUOIA.gui.loading.load_data.load_data_summary import _update_summary
from SECQUOIA.gui.loading.load_data.load_data_tracking_tree import (
    _build_tracking_tree,
)
from SECQUOIA.gui.loading_pipeline import run_loading
from SECQUOIA.gui.position_navigation import set_current_position_from_spinbox

__all__ = ["_on_load_data_clicked", "_on_run_clicked"]


def _on_load_data_clicked(main_window):
    """Validate the inputs, then fill the dialog from what is on disk."""
    if not getattr(main_window, "folder", None) or not getattr(
        main_window, "tracking_path", None
    ):
        QMessageBox.warning(
            main_window.mask_no_window,
            "Required inputs missing",
            "Please select an experiment folder and the tracking folder first.",
        )
        return

    if (
        _current_tracking_format(main_window) == "btrack"
        and not btrack_is_available()
    ):
        QMessageBox.warning(
            main_window.mask_no_window,
            "btrack is not installed",
            BTRACK_MISSING_MESSAGE,
        )
        return

    style_button_loading(main_window.load_data_btn)
    try:
        _reset_dataset_state(main_window)
        _reset_loaded_experiment(main_window)
        main_window.tracking_format = (
            main_window.tracking_format_combo.currentText()
        )

        if not _confirm_overwrite_project_dir(main_window):
            return

        _ensure_analysis_dirs(main_window)
        attach_experiment_log(main_window.folder)

        if main_window.tracking_format == "tTt":
            exp_folder = getattr(main_window, "folder", None)
            xml_path = None
            if exp_folder and os.path.isdir(exp_folder):

                candidates = sorted(
                    glob.glob(os.path.join(exp_folder, "*_TATexp.xml"))
                )
                xml_path = candidates[0] if candidates else None
            main_window.xml_path = xml_path

        # Segmentation list
        seg_names = _collect_segmentation_folders(main_window)
        _populate_segmentation_ui(main_window, seg_names)

        # BaSiC
        basic_names = _find_basic_folders(main_window)
        main_window.bg_correct_combo.clear()
        if basic_names:
            main_window.bg_correct_combo.addItems(basic_names)
            main_window.bg_correct_combo.setEnabled(True)
            main_window.bg_correct_chk.setEnabled(True)
        else:
            main_window.bg_correct_combo.addItem("(no BaSiC folders found)")
            main_window.bg_correct_combo.setEnabled(False)
            main_window.bg_correct_chk.setChecked(False)
            main_window.bg_correct_chk.setEnabled(False)
            main_window.bg_correct_chk.setToolTip(TOOLTIPSTEXT.BG_NONE)

        # Import Real time [ms] (images*.csv in experiment folder)
        images_csv = _find_images_csvs(main_window)
        main_window.import_rt_combo.blockSignals(True)
        main_window.import_rt_combo.clear()
        if images_csv:
            main_window.import_rt_combo.addItems(images_csv)
            main_window.import_rt_combo.setEnabled(True)
        else:
            main_window.import_rt_combo.addItem("(no images*.csv found)")
            main_window.import_rt_combo.setEnabled(False)
        main_window.import_rt_combo.blockSignals(False)

        ok = (
            main_window.import_rt_chk.isChecked()
            and main_window.import_rt_combo.count() > 0
            and not main_window.import_rt_combo.currentText().startswith("(")
        )
        main_window.import_rt_combo.setEnabled(ok)
        if ok:
            main_window.import_rt_path = os.path.join(
                getattr(main_window, "folder", ""),
                main_window.import_rt_combo.currentText(),
            )
        else:
            main_window.import_rt_path = None
        main_window.use_import_rt = main_window.import_rt_chk.isChecked()
        _update_summary(main_window)

        _maybe_load_ttt_channel_comments(main_window)
        _maybe_load_ttt_position_comments(main_window)

        # Channels
        chans = _detect_unique_w_channels(main_window)
        _populate_channels_ui(main_window, chans)

        # Positions
        det_min = int(getattr(main_window, "position_min", 1))
        det_max = int(getattr(main_window, "position_max", max(1, det_min)))

        sp_start = main_window.positions_start_combo
        sp_start.blockSignals(True)
        sp_start.setRange(det_min, det_max)
        sp_start.setValue(det_min)
        sp_start.blockSignals(False)

        main_window.position_min_selected = det_min
        main_window.position_max_selected = det_max

        def _sync_start():
            """Synchronize the selected starting position with main-window state."""
            st = sp_start.value()
            main_window.position_min_selected = det_min
            main_window.position_max_selected = det_max
            main_window.position_start_selected = st
            set_current_position_from_spinbox(main_window)

        sp_start.valueChanged.connect(_sync_start)
        _sync_start()

        t_detected = int(
            getattr(main_window, "t_max_detected", 0)
        ) or _detect_time_max_in_position_folder(main_window)
        main_window.t_max_detected = max(1, t_detected)
        main_window.time_min_spin.setRange(1, main_window.t_max_detected)
        main_window.time_max_spin.setRange(1, main_window.t_max_detected)
        main_window.time_min_spin.setValue(1)
        main_window.time_max_spin.setValue(main_window.t_max_detected)

        def _sync_time():
            """Synchronize the selected time range."""
            mn = main_window.time_min_spin.value()
            mx = main_window.time_max_spin.value()
            if mx < mn:
                main_window.time_max_spin.blockSignals(True)
                main_window.time_max_spin.setValue(mn)
                main_window.time_max_spin.blockSignals(False)
                mx = mn
            main_window.time_min_selected = mn
            main_window.time_max_selected = mx
            _update_summary(main_window)

        main_window.time_min_spin.valueChanged.connect(_sync_time)
        main_window.time_max_spin.valueChanged.connect(_sync_time)
        _sync_time()

        # Parser inputs
        main_window.cp_tracking = False
        # Will be included in the future
        # main_window.cp_tracking = (
        #    main_window.cp_tracking_chk.isChecked()
        #    if hasattr(main_window, "cp_tracking_chk")
        #    else bool(getattr(main_window, "cp_tracking", False))
        # )
        main_window.threshold = 0

        # Load tracking data
        if main_window.tracking_format == "tTt" and not getattr(
            main_window, "xml_path", None
        ):
            QMessageBox.warning(
                main_window.mask_no_window,
                "TAT XML not found",
                "tTt mode requires a TAT experiment XML (e.g. *_TATexp.xml) in the Experiment folder.\nPlease add it and try again.",
            )
            return

        try:
            _load_tracking_data(main_window)
        except (OSError, ValueError, KeyError, RuntimeError) as e:
            QMessageBox.warning(
                main_window.mask_no_window,
                "Tracking data load failed",
                f"An error occurred while loading tracking data:\n{e}",
            )
            return

        # Build selection tree
        if getattr(main_window, "track_df", None) is not None:
            _build_tracking_tree(main_window)

        for i in (1, 2, 3):
            main_window._tabs.setTabEnabled(i, True)

    finally:
        restore_button_loading(main_window.load_data_btn)


def _on_run_clicked(main_window):
    """Start the run: build the viewers and plots, then load the selected positions."""
    if not getattr(main_window, "folder", None):
        QMessageBox.warning(
            main_window.mask_no_window,
            "Select Experiment Folder",
            "Please select an experiment folder before running.",
        )
        return
    label = getattr(main_window, "progress_msg_label", None)
    if label is not None:
        label.setStyleSheet("")
        label.setText("Preparing to run… (building viewers)")

    QApplication.setOverrideCursor(Qt.WaitCursor)
    QApplication.processEvents()

    if not getattr(main_window, "_experiment_reset_done", False):
        _reset_loaded_experiment(main_window)
    main_window._experiment_reset_done = False

    if hasattr(main_window, "_threshold_spin"):
        main_window.threshold = main_window._threshold_spin.value()
    if hasattr(main_window, "_min_mask_size_spin"):
        main_window.min_mask_size = main_window._min_mask_size_spin.value()
    # CP will be included in the future
    # if hasattr(main_window, "cp_tracking_chk"):
    #    main_window.cp_tracking = main_window.cp_tracking_chk.isChecked()
    main_window.cp_tracking = False
    main_window.tracking_format = (
        main_window.tracking_format_combo.currentText()
    )
    main_window.image_format = main_window.image_format_combo.currentText()
    main_window.dt_seconds = int(main_window.time_input.value())
    _maybe_import_realtime(main_window)
    main_window.create_napari_viewers()
    main_window.add_viewers_to_layout2()
    main_window.add_plot_widgets()
    QTimer.singleShot(0, lambda: main_window._force_max_layout(ratio=0.30))
    filter_track_df_to_selected(main_window)
    run_loading(main_window)
    _update_summary(main_window)
    main_window.jump_channel = main_window.ids_channels[0]
    save_project_state(main_window)
    QApplication.restoreOverrideCursor()
