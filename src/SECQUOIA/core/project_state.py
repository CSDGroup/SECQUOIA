"""Project state: track_df CSV exports and project metadata JSON."""

import datetime as _dt
import glob
import json
import logging
import os
import re
from contextlib import suppress

import pandas as pd
from qtpy.QtWidgets import QLineEdit, QMessageBox

from SECQUOIA.core.tracking.clt_io import CLTParser
from SECQUOIA.utils.io_reload import dump_json_with_reload, to_csv_with_reload
from SECQUOIA.utils.paths import project_analysis_dir
from SECQUOIA.utils.positions import position_number_at_current_index
from SECQUOIA.utils.timing import current_t_range as _current_t_range

LOG = logging.getLogger(__name__)


def save_track_df(main_window, label_entries=None) -> None:
    """Save track_df to CSV files.

    1) The full track_df to Analysis/SECQUOIA_files_<tracking_format>/Updated_CSV_File.csv
    2) Only the current position rows into its per position file.
    """
    if not hasattr(main_window, "folder_list") or not main_window.folder_list:
        LOG.warning("Please first load a CSV file and select a folder.")
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText("Please first load a CSV file and select a folder")
        msg.setWindowTitle("Folder Selection Error")
        msg.exec_()
        return

    experiment_root = getattr(main_window, "folder", None)
    if not experiment_root or not os.path.isdir(experiment_root):
        LOG.warning("Invalid experiment root; aborting save.")
        return

    save_folder = project_analysis_dir(main_window, experiment_root)
    os.makedirs(save_folder, exist_ok=True)

    #  Save full CSV
    full_csv_path = os.path.join(save_folder, "Updated_CSV_File.csv")
    try:
        to_csv_with_reload(main_window.track_df, full_csv_path, index=False)
    except (OSError, ValueError, RuntimeError) as e:
        LOG.error("Failed to save full track_df CSV after reloading: %r", e)
        return
    LOG.info("The updated CSV file has been stored here: %s", full_csv_path)

    position = position_number_at_current_index(main_window)
    if position is None:
        LOG.warning("Could not determine current position.")
        return

    if "Position" not in main_window.track_df.columns:
        LOG.warning(
            "Column 'Position' not found in track_df; skipping per position export."
        )
        return

    df_pos = main_window.track_df[
        main_window.track_df["Position"] == position
    ].copy()
    if df_pos.empty:
        LOG.warning(
            "No rows found for Position %s; nothing to save.", position
        )
        return

    pattern = os.path.join(
        save_folder, f"SECQUOIA_p{position:04d}_t*-*_m*_ch*.csv"
    )
    matches = glob.glob(pattern)

    if matches:
        out_path = max(matches, key=os.path.getmtime)
    else:
        try:
            t_file_min, t_file_max, _, _ = _current_t_range(main_window)
        except (RuntimeError, AttributeError, TypeError):
            t_file_min, t_file_max = 0, 0

        num_masks = (
            len(label_entries)
            if label_entries is not None
            else getattr(main_window, "n_masks", "NA")
        )
        num_channels = getattr(main_window, "n_channels", "NA")

        filename = (
            f"SECQUOIA_p{position:04d}"
            f"_t{t_file_min:05d}-{t_file_max:05d}"
            f"_m{num_masks}_ch{num_channels}.csv"
        )
        out_path = os.path.join(save_folder, filename)

    # Save/overwrite
    try:
        to_csv_with_reload(df_pos, out_path, index=False)
    except (OSError, ValueError, RuntimeError) as e:
        LOG.error("Failed to save per position CSV after reloading: %r", e)
        return
    LOG.info("Saved per position (p%04d) CSV to: %s", position, out_path)


def _measurements_output_path(main_window, position, n_masks: int) -> str:
    """Build the per position measurements path under the project folder.

    ``SECQUOIA_p<pos>_t<min>-<max>_m<n_masks>_ch<n_channels>.csv``, created
    inside ``Analysis/SECQUOIA_files_<fmt>/<project>/``.
    """
    save_folder = project_analysis_dir(main_window)
    os.makedirs(save_folder, exist_ok=True)

    t_min, t_max, _, _ = _current_t_range(main_window)
    filename = (
        f"SECQUOIA_p{position:04d}"
        f"_t{t_min:05d}-{t_max:05d}"
        f"_m{n_masks}_ch{main_window.n_channels}.csv"
    )
    return os.path.join(save_folder, filename)


def save_position_measurements(main_window, position, n_masks: int) -> bool:
    """Write the rows of `position` to CSV. Returns True when a file was written."""
    try:
        df_pos = main_window.track_df[
            main_window.track_df["Position"] == position
        ].copy()
        if df_pos.empty:
            LOG.warning(
                "No rows found for Position %s; nothing to save.", position
            )
            return False

        out_path = _measurements_output_path(main_window, position, n_masks)
        to_csv_with_reload(df_pos, out_path, index=False)
        LOG.info("Saved fluorescence measurements to %s", out_path)
        return True
    except (
        RuntimeError,
        AttributeError,
        TypeError,
        OSError,
        ValueError,
    ) as err:
        LOG.error("Error while saving fluorescence measurements: %s", err)
        return False


def _combo_text(main_window, combo_attr: str, fallback_attr: str):
    """Current text of a combo box, falling back to the plain attribute."""
    combo = getattr(main_window, combo_attr, None)
    if combo is not None:
        with suppress(RuntimeError, AttributeError, TypeError):
            return combo.currentText()
    return getattr(main_window, fallback_attr, None)


def save_project_state(main_window) -> None:
    """Save the current project metadata to ``project_metadata.json``."""
    experiment_root = getattr(main_window, "folder", None)
    if not experiment_root:
        return

    save_folder = project_analysis_dir(main_window, experiment_root)
    os.makedirs(save_folder, exist_ok=True)

    fl_inputs_widgets = getattr(main_window, "FL_inputs", None) or []
    fl_inputs_text = []
    for w in fl_inputs_widgets:
        with suppress(AttributeError, RuntimeError, TypeError):
            fl_inputs_text.append(w.text())

    ids_channels = list(getattr(main_window, "ids_channels", []) or [])
    if ids_channels:
        fl_inputs_text = ids_channels

    expected_track_csv = os.path.join(save_folder, "Updated_CSV_File.csv")

    state = {
        "project_name": getattr(main_window, "project_name", None),
        "experiment_name": getattr(main_window, "experiment_name", None),
        "experiment_root": experiment_root,
        # Formats and modes
        "image_format": _combo_text(
            main_window, "image_format_combo", "image_format"
        ),
        "tracking_format": _combo_text(
            main_window, "tracking_format_combo", "tracking_format"
        ),
        "loading_format": getattr(main_window, "loading_format", None),
        "cp_tracking": getattr(main_window, "cp_tracking", False),
        # Paths
        "tracking_path": getattr(main_window, "tracking_path", None),
        "segmentation_paths": list(
            getattr(main_window, "segmentation_paths", []) or []
        ),
        "basic_flag": bool(
            getattr(getattr(main_window, "basic", None), "flag", False)
        ),
        "background_correction_path": getattr(
            main_window, "background_correction_path", None
        ),
        "import_rt_path": getattr(main_window, "import_rt_path", None),
        # Time and positions
        "time_min_selected": getattr(main_window, "time_min_selected", None),
        "time_max_selected": getattr(main_window, "time_max_selected", None),
        "dt_seconds": getattr(main_window, "dt_seconds", None),
        "position_min": getattr(main_window, "position_min_selected", None),
        "position_max": getattr(main_window, "position_max_selected", None),
        "position_selection": getattr(
            main_window, "position_start_selected", None
        ),
        # Thresholds
        "threshold": getattr(main_window, "threshold", None),
        "min_mask_size": getattr(main_window, "min_mask_size", None),
        # Channels
        "available_channels": list(
            getattr(main_window, "available_channels", []) or []
        ),
        "ids_channels": ids_channels,
        "jump_channel": getattr(main_window, "jump_channel", None),
        "n_channels": int(
            getattr(main_window, "n_channels", len(ids_channels) or 0) or 0
        ),
        "n_masks": getattr(main_window, "n_masks", None),
        "FL_inputs": list(fl_inputs_text),
        # Metric calculations
        "metric_calculations": {
            "derived_features": getattr(main_window, "_derived_features", {}),
            "last_run_config": getattr(main_window, "last_run_config", {}),
            "selected_feature_by_row": getattr(
                main_window, "selected_feature_by_row", {}
            ),
        },
        "track_df_csv_path": expected_track_csv,
        "user": getattr(main_window, "user", None),
        "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }

    json_path = os.path.join(save_folder, "project_metadata.json")
    try:
        dump_json_with_reload(state, json_path, indent=2, default=str)
    except (OSError, ValueError, RuntimeError) as e:
        LOG.error(
            "Failed to save project metadata JSON after reloading: %r", e
        )
        return

    main_window.project_state_path = json_path


def _rehydrate_after_project_load(main_window, s: dict) -> None:
    """Rebuild the channel widgets and position bookkeeping from saved state."""
    fl_names = list(s.get("FL_inputs") or []) or list(
        s.get("ids_channels") or []
    )
    main_window.FL_inputs = []
    for name in fl_names:
        le = QLineEdit()
        le.setReadOnly(True)
        le.setText(str(name))
        main_window.FL_inputs.append(le)

    main_window.ids_channels = fl_names
    main_window.n_channels = int(s.get("n_channels") or len(fl_names) or 0)
    main_window.jump_channel = s.get("jump_channel")

    folder = getattr(main_window, "folder", None)
    exp_name = getattr(
        main_window, "experiment_name", None
    ) or os.path.basename(folder or "")
    main_window.experiment_name = exp_name

    position_folders = []
    position_indices = []
    if folder and os.path.isdir(folder) and exp_name:
        pattern = re.compile(rf"^{re.escape(exp_name)}_p(\d+)$")
        for d in os.listdir(folder):
            full = os.path.join(folder, d)
            if os.path.isdir(full):
                m = pattern.match(d)
                if m:
                    position_folders.append(full)
                    position_indices.append(int(m.group(1)))

    position_folders.sort()
    position_indices.sort()

    main_window.position_folders = position_folders
    main_window.folder_list = (
        os.listdir(folder) if folder and os.path.isdir(folder) else []
    )

    if position_indices:
        main_window.position_min = position_indices[0]
        main_window.position_max = position_indices[-1]
        main_window.position_indices = position_indices
    else:
        main_window.position_min = 1
        main_window.position_max = 1
        main_window.position_indices = []

    # Set current_position_index from saved selection
    sel = int(s.get("position_selection") or 1)
    main_window.current_position_number = sel
    main_window.position_start_selected = sel
    try:
        main_window.current_position_index = max(
            0, min(int(sel) - 1, len(position_folders) - 1)
        )
    except (TypeError, ValueError):
        main_window.current_position_index = 0


def _restore_clt_parser(main_window) -> None:
    """Resolve the experiment's TAT XML metadata and (re)build ``clt_parser``."""
    exp_folder = getattr(main_window, "folder", None)
    xml_path = None
    if exp_folder and os.path.isdir(exp_folder):
        candidates = sorted(
            glob.glob(os.path.join(exp_folder, "*_TATexp.xml"))
        )
        xml_path = candidates[0] if candidates else None
    main_window.xml_path = xml_path
    main_window.clt_parser = CLTParser(xml_path)
    main_window.clt_parser.folder_exp = getattr(
        main_window, "tracking_path", None
    )


def _restore_bg_widgets(main_window, basic_flag: bool) -> None:
    """Mirror the restored BaSiC state onto the loading dialog's widgets."""
    chk = getattr(main_window, "bg_correct_chk", None)
    if chk is not None:
        with suppress(RuntimeError, AttributeError, TypeError):
            chk.blockSignals(True)
            chk.setChecked(basic_flag)
            chk.blockSignals(False)

    combo = getattr(main_window, "bg_correct_combo", None)
    if combo is None:
        return

    with suppress(RuntimeError, AttributeError, TypeError):
        combo.setEnabled(bool(basic_flag and combo.count() > 0))

    bg_path = getattr(main_window, "background_correction_path", None)
    if not bg_path:
        return

    with suppress(RuntimeError, AttributeError, TypeError):
        idx = combo.findText(os.path.basename(bg_path))
        if idx >= 0:
            combo.blockSignals(True)
            combo.setCurrentIndex(idx)
            combo.blockSignals(False)


def load_project_state(main_window, json_path=None) -> bool:
    """Load project metadata from ``project_metadata.json`` and restore state."""
    p = json_path or getattr(main_window, "project_state_path", None)
    if not p or not os.path.isfile(p):
        return False
    with open(p, encoding="utf-8") as f:
        s = json.load(f)

    csv_path = s.get("track_df_csv_path")
    if not csv_path or not os.path.isfile(csv_path):
        QMessageBox.critical(
            main_window,
            "Missing updated CSV file",
            "This project cannot be loaded because its updated CSV file is "
            f"missing:\n\n{csv_path or '(no path stored in project_metadata.json)'}\n\n",
        )
        return False

    for k in (
        "project_name",
        "experiment_name",
        "experiment_root",
        "cp_tracking",
        "image_format",
        "tracking_format",
        "tracking_path",
        "segmentation_paths",
        "background_correction_path",
        "import_rt_path",
        "time_min_selected",
        "time_max_selected",
        "dt_seconds",
        "position_min",
        "position_max",
        "position_selection",
        "threshold",
        "min_mask_size",
        "available_channels",
        "ids_channels",
        "jump_channel",
        "n_channels",
        "n_masks",
        "track_df_csv_path",
        "user",
        "basic_flag",
        "saved_at",
    ):
        setattr(main_window, "folder" if k == "experiment_root" else k, s[k])

    if main_window.tracking_format == "tTt":
        _restore_clt_parser(main_window)

    basic_flag = bool(s.get("basic_flag", False))

    if hasattr(main_window, "basic"):
        main_window.basic.flag = basic_flag

    main_window.use_background_correction = basic_flag

    _restore_bg_widgets(main_window, basic_flag)

    try:
        main_window.track_df = pd.read_csv(s["track_df_csv_path"])
    except (KeyError, FileNotFoundError, OSError, pd.errors.ParserError) as e:
        LOG.warning("CSV load failed: %s (%s)", s.get("track_df_csv_path"), e)
        main_window.track_df = None

    metric_state = s.get("metric_calculations", {})
    main_window._derived_features = metric_state.get("derived_features", {})

    main_window.last_run_config = {
        int(k): v for k, v in metric_state.get("last_run_config", {}).items()
    }
    main_window.selected_feature_by_row = {
        int(k): v
        for k, v in metric_state.get("selected_feature_by_row", {}).items()
    }
    # Make restored calculated features visible to dynamics plots
    main_window._feature_defs = getattr(main_window, "_feature_defs", {}) or {}
    main_window._feature_defs.update(main_window._derived_features)
    _rehydrate_after_project_load(main_window, s)

    main_window.project_state_path = p
    return True
