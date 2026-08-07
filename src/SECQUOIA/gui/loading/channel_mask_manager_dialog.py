"""Post-load Channel/Mask Manager.

Add or remove channels and segmentation masks after a project
has already been loaded, without restarting the experiment.
"""

from __future__ import annotations

import contextlib
import os

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from SECQUOIA.config import LINKS, STYLE, TOOLTIPSTEXT
from SECQUOIA.core.memmap_store import cleanup_memmaps
from SECQUOIA.core.project_state import save_project_state
from SECQUOIA.core.quantification import (
    FeatureNaming,
    stale_measurement_columns,
)
from SECQUOIA.core.tracking.track_data import (
    _position_csv_path,
    update_track_df,
)
from SECQUOIA.gui.common.ui_utils import (
    add_progress_bar,
    form_row,
    make_help_button,
    set_min_expanding,
    set_tree_rows,
)
from SECQUOIA.gui.loading.experiment_paths import (
    _add_seg_tree_item,
    _collect_segmentation_folders,
    _detect_unique_w_channels,
    _find_analysis_dir,
    _find_basic_folders,
)
from SECQUOIA.gui.loading_pipeline import _load_data, run_all_positions
from SECQUOIA.gui.outlier.reset import reset_outlier_state
from SECQUOIA.utils.plotting import _discover_features, update_plot
from SECQUOIA.utils.timing import current_t_range

__all__ = ["open_channel_mask_manager"]

_DARK_THEME_QSS = """
QDialog, QWidget {{
    font-size: {font_pt}pt;
    color: #ffffff;
    background-color: #2b2b2b;
}}
QTabWidget::pane {{
    border: 1px solid #555555;
    background: #2b2b2b;
}}
QTabBar::tab {{
    background: #333333;
    color: #ffffff;
    padding: 6px 14px;
    border: 1px solid #555555;
    margin-right: 2px;
}}
QTabBar::tab:selected {{ background: #3c3c3c; }}
QTabBar::tab:hover    {{ background: #4a4a4a; }}
QTreeWidget {{
    background-color: #404040;
    color: #ffffff;
    border: 1px solid #555555;
}}
QHeaderView::section {{
    background-color: #333333;
    color: #ffffff;
    border: 1px solid #555555;
    padding: 4px;
}}
QPushButton {{
    background-color: #404040;
    color: #ffffff;
    border: 1px solid #555555;
    padding: 6px 12px;
    border-radius: 4px;
}}
QPushButton:hover {{ background-color: #4a4a4a; }}
QPushButton:disabled {{ color: #888888; }}
QCheckBox {{ color: #ffffff; }}
QLabel {{ color: #ffffff; }}
"""


def _current_position_number(main_window) -> int | None:
    """Return the current position number, from the selected folder or the cached attribute."""
    sel = getattr(main_window, "position_selection", None)
    if sel:
        with contextlib.suppress(ValueError, IndexError):
            return int(os.path.basename(sel).split("_p")[-1])
    return getattr(main_window, "current_position_number", None)


def _invalidate_current_position_cache(main_window) -> None:
    """Delete the current position's cached measurements CSV."""
    position = _current_position_number(main_window)
    if position is None:
        return
    t_file_min, t_file_max, _, _ = current_t_range(main_window)
    path = _position_csv_path(main_window, position, t_file_min, t_file_max)
    if os.path.isfile(path):
        with contextlib.suppress(OSError):
            os.remove(path)


def _invalidate_all_position_caches(main_window) -> int:
    """Delete every cached measurements CSV of this project; return how many were removed."""
    position = _current_position_number(main_window) or 0
    t_file_min, t_file_max, _, _ = current_t_range(main_window)
    folder = os.path.dirname(
        _position_csv_path(main_window, position, t_file_min, t_file_max)
    )

    removed = 0
    with contextlib.suppress(OSError), os.scandir(folder) as it:
        for entry in it:
            if (
                entry.is_file()
                and entry.name.startswith("SECQUOIA_p")
                and entry.name.endswith(".csv")
            ):
                with contextlib.suppress(OSError):
                    os.remove(entry.path)
                    removed += 1
    return removed


def _columns_lost_by(main_window, channels, *, basic: bool, n_masks: int):
    """Return the ``track_df`` measurement columns that the pending configuration would orphan."""
    df = getattr(main_window, "track_df", None)
    if df is None or not hasattr(df, "columns"):
        return []
    return stale_measurement_columns(
        df.columns,
        FeatureNaming.for_config(channels, basic=basic),
        range(1, int(n_masks) + 1),
    )


def _calculated_metric_names(main_window) -> list[str]:
    """Names of the calculated metrics currently defined."""
    registry = getattr(main_window, "_derived_features", None) or {}
    return sorted(str(key) for key in registry)


def _clear_calculated_metrics(main_window) -> list[str]:
    """Delete every calculated metric, returning the names that were removed."""
    delete_all = getattr(main_window, "delete_all_derived_metrics", None)
    if not callable(delete_all):
        return []
    with contextlib.suppress(
        RuntimeError, AttributeError, TypeError, ValueError, KeyError
    ):
        return list(delete_all())
    return []


def _clear_outlier_state(main_window) -> None:
    """Drop the outlier flags, saved rules, list and markers."""
    with contextlib.suppress(
        RuntimeError, AttributeError, TypeError, ValueError, KeyError
    ):
        reset_outlier_state(
            main_window, refresh_plots=False, show_message=False
        )


def _drop_missing_row_features(main_window) -> list:
    """Forget per row plot features that the new configuration no longer offers."""
    selected = getattr(main_window, "selected_feature_by_row", None)
    df = getattr(main_window, "track_df", None)
    if not isinstance(selected, dict) or not selected:
        return []
    if df is None or not hasattr(df, "columns"):
        return []

    available = set(
        _discover_features(
            list(df.columns),
            getattr(main_window, "_derived_features", None) or {},
        )
    )
    cleared = [row for row, feat in selected.items() if feat not in available]
    for row in cleared:
        del selected[row]
    return cleared


def _sync_loader_bg_widgets(main_window, flag: bool, path: str | None) -> None:
    """Mirror the applied BaSiC state onto the loading dialog's widgets, if they still exist."""
    chk = getattr(main_window, "bg_correct_chk", None)
    combo = getattr(main_window, "bg_correct_combo", None)

    if chk is not None:
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            chk.blockSignals(True)
            chk.setChecked(bool(flag))
            chk.blockSignals(False)

    if combo is not None:
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            if path:
                idx = combo.findText(os.path.basename(path))
                if idx >= 0:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(idx)
                    combo.blockSignals(False)
            combo.setEnabled(bool(flag) and combo.count() > 0)


def open_channel_mask_manager(main_window: QWidget) -> None:
    """Open the Channel/Mask Manager dialog for an already loaded project."""
    if not getattr(main_window, "folder", None) or not getattr(
        main_window, "ids_channels", None
    ):
        QMessageBox.warning(
            main_window,
            "No project loaded",
            "Load a project first (File ▸ Start a new Project) before "
            "managing channels and masks.",
        )
        return

    prev = getattr(main_window, "_channel_mask_manager_window", None)
    if prev is not None:
        try:
            prev.close()
            prev.deleteLater()
        except RuntimeError:
            pass
        main_window._channel_mask_manager_window = None

    dlg = QDialog(main_window)
    dlg.setWindowTitle("Channel/Segmentation manager")
    dlg.setWindowModality(Qt.WindowModal)
    dlg.setMinimumSize(480, 560)
    dlg.setStyleSheet(_DARK_THEME_QSS.format(font_pt=STYLE.FONT_SIZE))

    outer = QVBoxLayout(dlg)
    outer.setContentsMargins(10, 10, 10, 10)
    outer.setSpacing(8)

    tabs = QTabWidget()
    outer.addWidget(tabs)

    # Channels tab
    chan_tab = QWidget()
    chan_v = QVBoxLayout(chan_tab)
    chan_tree = QTreeWidget()
    chan_tree.setHeaderLabels(["Channels"])
    chan_tree.setColumnCount(1)
    chan_tree.setRootIsDecorated(False)
    chan_tree.setToolTip(TOOLTIPSTEXT.CH_TREE_INFO)
    chan_tree.headerItem().setToolTip(0, TOOLTIPSTEXT.CH_TREE)
    chan_v.addWidget(chan_tree)
    set_tree_rows(chan_tree, approx_rows=10)

    current_channels = list(getattr(main_window, "ids_channels", []))
    all_channels = list(
        _detect_unique_w_channels(main_window)
        or getattr(main_window, "available_channels", [])
        or []
    )
    for ch in current_channels:
        if ch not in all_channels:
            all_channels.append(ch)

    for ch in all_channels:
        item = QTreeWidgetItem([ch])
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(
            0, Qt.Checked if ch in current_channels else Qt.Unchecked
        )
        chan_tree.addTopLevelItem(item)

    tabs.addTab(chan_tab, "Channel Selection")

    # Segmentation tab
    mask_tab = QWidget()
    mask_v = QVBoxLayout(mask_tab)
    mask_tree = QTreeWidget()
    mask_tree.setHeaderLabels(["Segmentations", "Mask #"])
    mask_tree.setColumnCount(2)
    mask_tree.setRootIsDecorated(False)
    mask_tree.setToolTip(TOOLTIPSTEXT.SEG_TREE_INFO)
    mask_tree.headerItem().setToolTip(0, TOOLTIPSTEXT.SEG_TREE)
    mask_tree.header().setStretchLastSection(False)
    mask_v.addWidget(mask_tree)
    set_tree_rows(mask_tree, approx_rows=10)

    analysis_dir = _find_analysis_dir(main_window)
    current_paths = list(getattr(main_window, "segmentation_paths", []))
    current_names = {os.path.basename(p) for p in current_paths}
    seg_names = list(_collect_segmentation_folders(main_window))
    for name in current_names:
        if name not in seg_names:
            seg_names.append(name)
    seg_names = sorted(seg_names)

    for name in seg_names:
        _add_seg_tree_item(
            mask_tree,
            name,
            analysis_dir,
            checked=name in current_names,
            extra_columns=1,
        )

    mask_tree.setColumnWidth(0, 340)
    mask_tree.setColumnWidth(1, 90)

    def _refresh_mask_numbers():
        """Show each checked mask's resulting M-number; blank for unchecked ones."""
        mask_tree.blockSignals(True)
        try:
            counter = 0
            for i in range(mask_tree.topLevelItemCount()):
                it = mask_tree.topLevelItem(i)
                if it.checkState(0) == Qt.Checked:
                    counter += 1
                    it.setText(1, f"M{counter}")
                else:
                    it.setText(1, "")
        finally:
            mask_tree.blockSignals(False)

    _refresh_mask_numbers()
    mask_tree.itemChanged.connect(lambda _i, _c: _refresh_mask_numbers())

    initial_basic_flag = bool(getattr(main_window.basic, "flag", False))
    initial_basic_path = getattr(
        main_window, "background_correction_path", None
    )

    bg_correct_chk = QCheckBox()
    bg_correct_chk.setToolTip(TOOLTIPSTEXT.BG)
    bg_correct_combo = QComboBox()
    bg_correct_combo.setToolTip(TOOLTIPSTEXT.BG_COMBO)
    bg_correct_combo.setSizeAdjustPolicy(
        QComboBox.AdjustToMinimumContentsLengthWithIcon
    )
    bg_correct_combo.setMinimumContentsLength(0)
    set_min_expanding(bg_correct_combo, 170)

    basic_names = _find_basic_folders(main_window)
    if basic_names:
        bg_correct_combo.addItems(basic_names)
    else:
        bg_correct_combo.addItem("(no BaSiC folders found)")

    if initial_basic_path:
        idx = bg_correct_combo.findText(os.path.basename(initial_basic_path))
        if idx >= 0:
            bg_correct_combo.setCurrentIndex(idx)

    bg_correct_chk.setEnabled(bool(basic_names))
    bg_correct_chk.setChecked(initial_basic_flag and bool(basic_names))
    if not basic_names:
        bg_correct_chk.setToolTip(TOOLTIPSTEXT.BG_NONE)

    def _selected_basic_path() -> str | None:
        """Return the BaSiC folder the dialog currently selects, or None."""
        if not bg_correct_chk.isChecked() or not basic_names:
            return None
        sel = bg_correct_combo.currentText()
        if not sel or sel.startswith("(") or not analysis_dir:
            return None
        return os.path.join(analysis_dir, sel)

    def _sync_bg_ui(_state=None):
        """Enable the folder combo only while the checkbox is on and BaSiC folders exist."""
        bg_correct_combo.setEnabled(
            bg_correct_chk.isChecked() and bool(basic_names)
        )

    bg_correct_chk.stateChanged.connect(_sync_bg_ui)
    _sync_bg_ui()

    bg_row = QWidget()
    bg_h = QHBoxLayout(bg_row)
    bg_h.setContentsMargins(0, 0, 0, 0)
    bg_h.setSpacing(8)
    bg_h.addWidget(bg_correct_combo, 1)
    bg_h.addWidget(bg_correct_chk, 0)
    mask_v.addLayout(form_row("BaSiC background correction:", bg_row))

    tabs.addTab(mask_tab, "Segmentation Options")

    help_btn = make_help_button(
        tabs, TOOLTIPSTEXT.HELP_BTN, LINKS.GITHUB_LOADING_WINDOW
    )
    tabs.setCornerWidget(help_btn, Qt.TopRightCorner)

    # Bottom bar: apply-scope, status, progress, actions
    apply_all_chk = QCheckBox("Apply to all positions")
    apply_all_chk.setChecked(True)
    apply_all_chk.setToolTip(TOOLTIPSTEXT.CM_APPLY_ALL)
    outer.addWidget(apply_all_chk)

    status_label = QLabel("")
    status_label.setWordWrap(True)
    outer.addWidget(status_label)

    progress_bar, progress = add_progress_bar(outer)
    progress_bar.setVisible(False)

    btn_row = QWidget()
    btn_h = QHBoxLayout(btn_row)
    btn_h.setContentsMargins(0, 0, 0, 0)
    close_btn = QPushButton("Close")
    close_btn.setToolTip(TOOLTIPSTEXT.EXIT_BTN)
    close_btn.clicked.connect(dlg.close)
    apply_btn = QPushButton("Apply")
    apply_btn.setToolTip(TOOLTIPSTEXT.CM_APPLY_BTN)
    btn_h.addStretch()
    btn_h.addWidget(close_btn)
    btn_h.addWidget(apply_btn)
    outer.addWidget(btn_row)

    def _checked_items(tree):
        """Return the checked top-level items of a tree, in display order."""
        return [
            tree.topLevelItem(i)
            for i in range(tree.topLevelItemCount())
            if tree.topLevelItem(i).checkState(0) == Qt.Checked
        ]

    _UI_ERRORS = (
        RuntimeError,
        AttributeError,
        TypeError,
        ValueError,
        KeyError,
    )

    def _refresh_ui_after_load():
        """Rebuild the CH/M dropdowns and plots once the new layers and values exist."""
        with contextlib.suppress(*_UI_ERRORS):
            main_window.update_channel_mask_dropdowns()
        with contextlib.suppress(*_UI_ERRORS):
            _drop_missing_row_features(main_window)
        with contextlib.suppress(*_UI_ERRORS):
            main_window.add_plot_widgets()
        with contextlib.suppress(*_UI_ERRORS):
            update_track_df(main_window)
        with contextlib.suppress(*_UI_ERRORS):
            update_plot(main_window)
        with contextlib.suppress(*_UI_ERRORS):
            main_window.fit_all_plots()

    def _finalize():
        """Refresh the UI, save the project and close the dialog once loading is done."""
        _refresh_ui_after_load()
        save_project_state(main_window)
        QApplication.restoreOverrideCursor()
        apply_btn.setEnabled(True)
        close_btn.setEnabled(True)
        progress["set"](100)
        status_label.setText("Done.")
        QTimer.singleShot(400, dlg.close)

    def _poll_until_loaded():
        """Wait for the background load to clear ``_loading_in_progress``, then finalize."""
        if getattr(main_window, "_loading_in_progress", False):
            QTimer.singleShot(150, _poll_until_loaded)
            return
        _finalize()

    def _on_apply():
        """Apply the new channel/mask/BaSiC selection and re-run quantification."""
        new_channels = [it.text(0) for it in _checked_items(chan_tree)]
        mask_items = _checked_items(mask_tree)
        new_paths = [it.data(0, Qt.UserRole) for it in mask_items]

        if not new_channels:
            QMessageBox.warning(
                dlg, "No channels selected", "Select at least one channel."
            )
            return
        if not new_paths:
            QMessageBox.warning(
                dlg,
                "No masks selected",
                "Select at least one segmentation mask.",
            )
            return

        new_basic_flag = bg_correct_chk.isChecked()
        new_basic_path = _selected_basic_path()

        channels_changed = new_channels != list(main_window.ids_channels)
        masks_changed = new_paths != list(main_window.segmentation_paths)
        basic_changed = (
            new_basic_flag != initial_basic_flag
            or new_basic_path != initial_basic_path
        )
        if not channels_changed and not masks_changed and not basic_changed:
            status_label.setText("No changes to apply.")
            return

        apply_all = apply_all_chk.isChecked()
        basic_state = "on" if new_basic_flag else "off"
        msg = (
            f"This will re-run quantification for "
            f"{'all positions' if apply_all else 'the current position only'} "
            f"with {len(new_channels)} channel(s), {len(new_paths)} mask(s), "
            f"and BaSiC background correction {basic_state}.\n\n"
        )

        lost_columns = _columns_lost_by(
            main_window,
            new_channels,
            basic=new_basic_flag,
            n_masks=len(new_paths),
        )
        doomed_metrics = _calculated_metric_names(main_window)
        if lost_columns:
            msg += (
                f"{len(lost_columns)} measurement column(s) of the previous "
                "configuration will be removed.\n\n"
            )
        if doomed_metrics:
            msg += (
                f"All {len(doomed_metrics)} calculated metric(s) will be "
                "deleted and have to be re-created afterwards:\n  "
                + "\n  ".join(doomed_metrics)
                + "\n\n"
            )

        msg += (
            "Outlier detection will be cleared: the flags are reset and the "
            "saved rules are discarded, because they were built on the "
            "previous measurements.\n\n"
        )

        if apply_all:
            msg += (
                "Cached per position measurement CSVs will be deleted and "
                "recomputed from the images.\n\n"
            )
        elif masks_changed or basic_changed:
            msg += (
                "Warning: only the current position will be reprocessed. "
                "Other positions keep their previous (now stale) values "
                "until reprocessed too.\n\n"
            )
        msg += "This can take a while. Continue?"

        reply = QMessageBox.question(
            dlg,
            "Apply channel/mask changes",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        apply_btn.setEnabled(False)
        close_btn.setEnabled(False)
        status_label.setText("Applying changes…")
        progress_bar.setVisible(True)
        progress["set"](0)

        main_window.ids_channels = new_channels
        main_window.n_channels = len(new_channels)
        for i, ch in enumerate(new_channels):
            setattr(main_window, f"FL_identifiers_{i + 1}", ch)
        main_window.available_channels = list(all_channels)

        main_window.segmentation_paths = new_paths
        main_window.n_masks = len(new_paths)

        main_window.basic.flag = new_basic_flag
        main_window.use_background_correction = new_basic_flag
        main_window.background_correction_path = new_basic_path
        _sync_loader_bg_widgets(main_window, new_basic_flag, new_basic_path)

        _clear_outlier_state(main_window)
        _clear_calculated_metrics(main_window)

        main_window.progress_bar = progress_bar
        main_window.set_progress = progress["set"]
        main_window.bump_progress = progress["bump"]
        main_window.finish_progress = progress["finish"]
        main_window.set_progress_step = progress["step"]
        main_window.progress_msg_label = status_label
        main_window.global_progress_bar = progress_bar

        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()

        try:
            cleanup_memmaps(main_window)

            main_window.update_channel_mask_dropdowns()
            main_window.add_plot_widgets()

            if apply_all:
                _invalidate_all_position_caches(main_window)
                run_all_positions(main_window)
                _finalize()
            else:
                _invalidate_current_position_cache(main_window)
                _load_data(main_window)
                QTimer.singleShot(150, _poll_until_loaded)
        except (
            RuntimeError,
            AttributeError,
            TypeError,
            ValueError,
            OSError,
            KeyError,
        ) as e:
            QApplication.restoreOverrideCursor()
            apply_btn.setEnabled(True)
            close_btn.setEnabled(True)
            status_label.setText(f"Error: {e}")

    apply_btn.clicked.connect(_on_apply)

    main_window._channel_mask_manager_window = dlg
    dlg.show()
