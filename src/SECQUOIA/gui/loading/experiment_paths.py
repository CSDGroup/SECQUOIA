"""Shared helpers for the experiment-loading dialogs."""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil

from qtpy.QtCore import QSettings, Qt
from qtpy.QtWidgets import (
    QFileDialog,
    QMenu,
    QMessageBox,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

from SECQUOIA.config import (
    EXPERIMENTS_ROOT_KEY,
    TOOLTIPSTEXT,
    TRACKING_ROOT_KEYS,
)
from SECQUOIA.core.experiment_layout import (
    detect_time_max,
    detect_unique_w_channels,
    find_analysis_dir,
    find_basic_folders,
    find_example_position_folder,
    find_images_csvs,
)
from SECQUOIA.core.logging_setup import attach_experiment_log
from SECQUOIA.utils.paths import project_analysis_dir

LOG = logging.getLogger(__name__)

__all__ = [
    "_add_seg_tree_item",
    "_clear_saved_dir",
    "_collect_segmentation_folders",
    "_confirm_overwrite_project_dir",
    "_current_tracking_format",
    "_detect_time_max_in_position_folder",
    "_detect_unique_w_channels",
    "_ensure_analysis_dirs",
    "_find_analysis_dir",
    "_find_basic_folders",
    "_find_example_position_folder",
    "_find_images_csvs",
    "_get_experiments_root",
    "_get_saved_dir",
    "_get_tracking_root",
    "_install_experiment_mount_menu",
    "_install_tracking_mount_menu",
    "_make_bg_sync_handler",
    "_select_experiment_folder",
    "_select_tracking_folder",
    "_set_saved_dir",
    "_settings",
    "_tracking_root_key",
    "_update_load_button_enabled",
]


def _settings():
    """Return the app-wide QSettings object."""
    return QSettings("SECQUOIA", "SECQUOIA")


def _get_saved_dir(key: str) -> str:
    """Return a saved directory path if it still exists."""
    path = _settings().value(key, "", type=str)
    return path if path and os.path.isdir(path) else ""


def _set_saved_dir(key: str, path: str) -> None:
    """Save an existing directory path under the given settings key."""
    if path and os.path.isdir(path):
        settings = _settings()
        LOG.debug("QSettings file: %s", settings.fileName())
        settings.setValue(key, path)
        settings.sync()


def _clear_saved_dir(key: str) -> None:
    """Remove a saved directory path from QSettings."""
    settings = _settings()
    settings.remove(key)
    settings.sync()


def _get_experiments_root() -> str:
    """Return the mounted parent folder for experiment folders."""
    return _get_saved_dir(EXPERIMENTS_ROOT_KEY)


def _tracking_root_key(fmt: str) -> str:
    """Return the QSettings key for a tracking format."""
    return TRACKING_ROOT_KEYS.get(str(fmt), f"paths/tracking_root_{fmt}")


def _get_tracking_root(fmt: str) -> str:
    """Return the mounted parent folder for a tracking format."""
    return _get_saved_dir(_tracking_root_key(fmt))


def _find_analysis_dir(main_window):
    """Return '<experiment_folder>/Analysis' for the loaded experiment."""
    return find_analysis_dir(getattr(main_window, "folder", None))


def _find_basic_folders(main_window):
    """Return BaSiC folder names inside the experiment's Analysis folder."""
    return find_basic_folders(getattr(main_window, "folder", None))


def _find_images_csvs(main_window):
    """Return names of 'images*.csv' files in the experiment folder."""
    return find_images_csvs(getattr(main_window, "folder", None))


def _find_example_position_folder(main_window):
    """Return a representative position folder for the loaded experiment."""
    return find_example_position_folder(
        getattr(main_window, "folder", None),
        getattr(main_window, "experiment_name", None),
        getattr(main_window, "position_folders", None),
    )


def _detect_unique_w_channels(main_window):
    """Return the 'wNN' channel identifiers found in one sample position folder."""
    return detect_unique_w_channels(
        _find_example_position_folder(main_window),
        getattr(main_window, "n_channels", 1),
    )


def _detect_time_max_in_position_folder(main_window):
    """Return the highest time point index present in the experiment."""
    return detect_time_max(_find_example_position_folder(main_window))


def _collect_segmentation_folders(main_window):
    """Return folder NAMES in <experiment>/Analysis that start with 'Segmentation'."""
    analysis = _find_analysis_dir(main_window)
    if not analysis:
        return []
    names = []
    with contextlib.suppress(OSError), os.scandir(analysis) as it:
        for e in it:
            if e.is_dir() and e.name.startswith("Segmentation"):
                names.append(e.name)
    return sorted(names)


def _make_bg_sync_handler(main_window, *, on_sync=None):
    """Build a handler that syncs the BaSiC checkbox/combo state onto main_window."""

    def _sync_bg_ui(_state=None):
        ok = (
            main_window.bg_correct_chk.isChecked()
            and main_window.bg_correct_combo.count() > 0
        )
        main_window.bg_correct_combo.setEnabled(ok)
        analysis = _find_analysis_dir(main_window)
        sel = main_window.bg_correct_combo.currentText()
        if ok and analysis and sel and not sel.startswith("("):
            main_window.background_correction_path = os.path.join(
                analysis, sel
            )
        else:
            main_window.background_correction_path = None
        main_window.basic.flag = main_window.bg_correct_chk.isChecked()
        main_window.use_background_correction = (
            main_window.bg_correct_chk.isChecked()
        )
        if on_sync is not None:
            on_sync()

    return _sync_bg_ui


def _add_seg_tree_item(
    tree: QTreeWidget,
    name: str,
    analysis_dir: str | None,
    *,
    checked: bool = False,
    extra_columns: int = 0,
) -> QTreeWidgetItem:
    """Add a checkable segmentation folder item to a tree, storing its full path in UserRole."""
    item = QTreeWidgetItem([name] + [""] * extra_columns)
    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
    item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
    full = os.path.join(analysis_dir, name) if analysis_dir else name
    item.setData(0, Qt.UserRole, full)
    tree.addTopLevelItem(item)
    return item


def _ensure_analysis_dirs(main_window):
    """Create ``<experiment>/Analysis/SECQUOIA_files_<tracking_format>/<project_name>``."""
    folder = getattr(main_window, "folder", None)
    if not folder:
        return
    try:
        project_dir = project_analysis_dir(main_window, folder)
        os.makedirs(project_dir, exist_ok=True)
    except (OSError, RuntimeError, AttributeError, TypeError) as e:
        if hasattr(main_window, "status_label") and main_window.status_label:
            main_window.status_label.setText(
                f"Status: Could not prepare Analysis folders ({e})."
            )


def _confirm_overwrite_project_dir(main_window) -> bool:
    """Check whether the selected project folder already exists. If it exists, ask the user whether to overwrite it."""
    folder = getattr(main_window, "folder", None)
    if not folder:
        return False

    tracking_format = getattr(main_window, "tracking_format", None)
    if not tracking_format:
        tracking_format = main_window.tracking_format_combo.currentText()

    proj = (getattr(main_window, "project_name", "") or "Project_1").strip()

    if not proj:
        proj = "Project_1"
        main_window.project_name = proj

    project_dir = project_analysis_dir(main_window, folder, tracking_format)

    if not os.path.isdir(project_dir):
        main_window.project_dir = project_dir
        return True

    parent = getattr(main_window, "mask_no_window", main_window)

    reply = QMessageBox.question(
        parent,
        "Project already exists",
        (
            f"The project folder already exists:\n\n"
            f"{project_dir}\n\n"
            "Do you want to overwrite it?\n\n"
            "This will permanently delete the old project folder and "
            "create a new empty one."
        ),
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No,
    )

    if reply != QMessageBox.Yes:
        if hasattr(main_window, "status_label") and main_window.status_label:
            main_window.status_label.setText(
                "Status: Project creation cancelled."
            )
        return False

    try:
        shutil.rmtree(project_dir)
    except OSError as e:
        QMessageBox.warning(
            parent,
            "Could not overwrite project",
            f"Could not delete the existing project folder:\n\n{e}",
        )
        return False

    main_window.project_dir = project_dir
    return True


def _current_tracking_format(main_window) -> str:
    """Return the currently selected tracking format."""
    combo = getattr(main_window, "tracking_format_combo", None)
    if combo is not None:
        txt = combo.currentText()
        if txt:
            return txt
    return getattr(main_window, "tracking_format", "tTt") or "tTt"


def _update_load_button_enabled(main_window):
    """Enable the "Load Experiment" button only once a folder and a tracking path are set."""
    has_folder = bool(getattr(main_window, "folder", None))
    path_ok = bool(getattr(main_window, "tracking_path", None))
    btn = getattr(main_window, "load_data_btn", None)
    if btn is not None:
        btn.setEnabled(has_folder and path_ok)


def _resolve_experiment_dialog_parent(main_window, dlg):
    """Return the widget to parent the experiment-folder dialog to, if any."""
    parent = getattr(main_window, "mask_no_window", main_window)
    if isinstance(parent, QWidget):
        return parent
    if isinstance(dlg, QWidget):
        return dlg
    return main_window if isinstance(main_window, QWidget) else None


def _apply_experiment_folder_to_widgets(main_window, folder):
    """Reflect the chosen folder in the folder button/label/apply-button, and return its base name.

    These widgets may belong to a different window (e.g. the main Loading
    window vs. the Cytometric Analysis dialog) that has since been closed,
    leaving a stale Python reference to an already-deleted Qt widget — guard
    each touch so that doesn't crash the currently open dialog.
    """
    base = os.path.basename(folder)
    if hasattr(main_window, "folder_button") and main_window.folder_button:
        with contextlib.suppress(RuntimeError):
            main_window.folder_button.setText(base)
    if hasattr(main_window, "folder_label") and main_window.folder_label:
        with contextlib.suppress(RuntimeError):
            main_window.folder_label.setText(folder)
    if hasattr(main_window, "apply_button") and main_window.apply_button:
        with contextlib.suppress(RuntimeError):
            main_window.apply_button.setEnabled(True)
    return base


def _scan_position_folders(folder, experiment_name):
    """Return (position_folders, position_indices) for '<experiment_name>_p<N>' subfolders of folder."""
    pattern = re.compile(rf"^{re.escape(experiment_name)}_p(\d+)$")

    position_folders = []
    position_indices = []
    try:
        for d in os.listdir(folder):
            full = os.path.join(folder, d)
            if not os.path.isdir(full):
                continue
            m = pattern.match(d)
            if m:
                position_folders.append(full)
                position_indices.append(int(m.group(1)))
    except (OSError, RuntimeError, AttributeError, TypeError):
        return [], []

    position_folders.sort()
    position_indices.sort()
    return position_folders, position_indices


def _apply_position_folders(main_window, position_folders, position_indices):
    """Store scanned position folders/indices on main_window, defaulting to a single position."""
    main_window.position_folders = position_folders
    if position_indices:
        main_window.position_min = position_indices[0]
        main_window.position_max = position_indices[-1]
        main_window.position_indices = position_indices
    else:
        main_window.position_min = 1
        main_window.position_max = 1
        main_window.position_indices = []


def _ask_for_directory(parent, title: str, start_dir: str) -> str:
    """Ask the user for a directory"""
    path = QFileDialog.getExistingDirectory(parent, title, start_dir)
    return os.path.normpath(path) if path else ""


def _select_experiment_folder(main_window, dlg=None):
    """Open a folder dialog to pick an experiment folder and refresh the dependent UI/state."""
    parent = _resolve_experiment_dialog_parent(main_window, dlg)

    start_dir = (
        _get_experiments_root()
        or getattr(main_window, "folder", None)
        or os.path.expanduser("~")
    )

    folder = _ask_for_directory(parent, "Select Experiment Folder", start_dir)

    if not folder:
        if parent:
            parent.raise_()
            parent.activateWindow()
        return

    main_window.folder = folder
    base = _apply_experiment_folder_to_widgets(main_window, folder)

    _ensure_analysis_dirs(main_window)
    attach_experiment_log(folder)

    main_window.experiment_name = base

    position_folders, position_indices = _scan_position_folders(
        folder, main_window.experiment_name
    )
    _apply_position_folders(main_window, position_folders, position_indices)

    main_window.folder_list = [
        f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))
    ]

    parent.raise_()
    parent.activateWindow()


def _select_tracking_folder(main_window, fmt: str, title: str):
    """Select a tracking folder, starting from the mounted tracking path."""
    parent = getattr(main_window, "mask_no_window", main_window)
    if not isinstance(parent, QWidget):
        parent = main_window if isinstance(main_window, QWidget) else None

    start_dir = (
        _get_tracking_root(fmt)
        or getattr(main_window, "tracking_path", None)
        or _get_experiments_root()
        or os.path.expanduser("~")
    )

    tracking_path = _ask_for_directory(parent, title, start_dir)

    if tracking_path:
        main_window.tracking_path = tracking_path
        main_window.tracking_button.setText(os.path.basename(tracking_path))
        main_window.tracking_button.setToolTip(tracking_path)

    _update_load_button_enabled(main_window)

    if parent:
        parent.raise_()
        parent.activateWindow()


def _resolve_mount_dialog_parent(main_window):
    """Find a QWidget to parent a mount-folder file dialog on."""
    parent = getattr(main_window, "mask_no_window", main_window)
    if not isinstance(parent, QWidget):
        parent = main_window if isinstance(main_window, QWidget) else None
    return parent


def _exec_mount_menu(menu, button, pos):
    """Show ``menu`` at ``pos`` and return the chosen action, across Qt bindings."""
    try:
        return menu.exec_(button.mapToGlobal(pos))
    except AttributeError:
        return menu.exec(button.mapToGlobal(pos))


def _install_experiment_mount_menu(button, main_window):
    """Right-click lets the user mount a parent folder: a directory containing many experiment folders."""
    button.setContextMenuPolicy(Qt.CustomContextMenu)

    def _show_menu(pos):
        """Show the context menu for mounting or clearing a saved folder path."""
        menu = QMenu(button)

        current_root = _get_experiments_root()

        if current_root:
            current_action = menu.addAction(f"Mounted path: {current_root}")
            current_action.setEnabled(False)
            menu.addSeparator()

        mount_action = menu.addAction("Mount experiments path...")
        clear_action = menu.addAction("Clear mounted path")

        clear_action.setEnabled(bool(current_root))

        chosen = _exec_mount_menu(menu, button, pos)

        if chosen == mount_action:
            parent = _resolve_mount_dialog_parent(main_window)

            start_dir = current_root or os.path.expanduser("~")

            mounted = _ask_for_directory(
                parent,
                "Select parent folder containing experiments",
                start_dir,
            )

            if mounted:
                _set_saved_dir(EXPERIMENTS_ROOT_KEY, mounted)
                button.setToolTip(
                    f"Mounted experiments path:\n{mounted}\n\n"
                    "Left-click to select an experiment folder."
                )

        elif chosen == clear_action:
            _clear_saved_dir(EXPERIMENTS_ROOT_KEY)
            button.setToolTip(TOOLTIPSTEXT.EXPFOLDER)

    button.customContextMenuRequested.connect(_show_menu)


def _install_tracking_mount_menu(button, main_window):
    """Right-click lets the user mount a parent folder: a directory containing
    tracking folders for the current tracking format.
    """
    if button.property("tracking_mount_menu_installed"):
        return

    button.setProperty("tracking_mount_menu_installed", True)
    button.setContextMenuPolicy(Qt.CustomContextMenu)

    def _show_menu(pos):
        """Show the context menu for mounting or clearing a saved folder path."""
        fmt = _current_tracking_format(main_window)
        key = _tracking_root_key(fmt)
        current_root = _get_saved_dir(key)

        menu = QMenu(button)

        if current_root:
            current_action = menu.addAction(
                f"Mounted {fmt} tracking path: {current_root}"
            )
            current_action.setEnabled(False)
            menu.addSeparator()

        mount_action = menu.addAction(f"Mount {fmt} tracking path...")
        clear_action = menu.addAction(f"Clear {fmt} mounted path")
        clear_action.setEnabled(bool(current_root))

        chosen = _exec_mount_menu(menu, button, pos)

        if chosen == mount_action:
            parent = _resolve_mount_dialog_parent(main_window)

            start_dir = (
                current_root
                or _get_experiments_root()
                or os.path.expanduser("~")
            )

            mounted = _ask_for_directory(
                parent,
                f"Select parent folder containing {fmt} tracking folders",
                start_dir,
            )

            if mounted:
                _set_saved_dir(key, mounted)
                button.setToolTip(
                    f"Mounted {fmt} tracking path:\n{mounted}\n\n"
                    "Left-click to select a tracking folder."
                )

        elif chosen == clear_action:
            _clear_saved_dir(key)
            button.setToolTip(
                f"Select {fmt} folder\n\n"
                "Right-click to mount tracking path."
            )

    button.customContextMenuRequested.connect(_show_menu)
