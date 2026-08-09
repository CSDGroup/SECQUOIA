"""Switching, selecting, and navigating between experiment positions."""

from __future__ import annotations

import contextlib
import logging
import os
import re

import qtawesome as qta
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from SECQUOIA.config import TOOLTIPSTEXT
from SECQUOIA.core.gap_filling import add_missing_rows
from SECQUOIA.core.outlier_detection import (
    RulesPack,
    run_outlier_pipeline,
    update_outlier_detection_in_track_df,
    update_unique_outliers_ids,
)
from SECQUOIA.core.project_state import save_track_df
from SECQUOIA.core.segmentation.mask_io import save_masks_incremental
from SECQUOIA.core.segmentation.mask_selection import ensure_current_df_subset
from SECQUOIA.core.tracking.clt_io import CLTParser
from SECQUOIA.core.tracking.track_data import update_track_df
from SECQUOIA.gui.common.messages import show_folder_warning
from SECQUOIA.gui.common.ui_utils import add_progress_bar, widen_to_hint
from SECQUOIA.gui.loading_pipeline import UiProgressBridge, update_images
from SECQUOIA.gui.outlier.markers import update_outlier_marker
from SECQUOIA.gui.outlier.outlier_list import (
    _outliers_is_empty,
    _update_outlier_list,
    auto_select_first_item,
)
from SECQUOIA.utils.plotting import update_plot
from SECQUOIA.utils.positions import position_number_from_folder

LOG = logging.getLogger(__name__)

__all__ = [
    "PositionSelectDialog",
    "load_position",
    "open_position_window",
    "parse_position_entries",
    "resolve_position_index",
    "set_current_position_from_spinbox",
    "switch_to_position_with_progress",
]

# Matches the ''_p0008'' position number in a folder name.
_POSITION_RE = re.compile(r"_p(\d+)")

_LIST_STYLESHEET = (
    "QListWidget { border: 1px solid #555; border-radius: 4px;"
    " background-color: #2b2b2b; color: #ffffff; }"
    "QListWidget::item { padding: 6px 8px;"
    " background-color: transparent; color: #ffffff; }"
    "QListWidget::item:hover { background-color: #3a3a3a; }"
    "QListWidget::item:selected { background-color: #0a84ff;"
    " color: #ffffff; }"
)


def _create_loading_dialog(
    parent: QWidget, title: str = "Loading position"
) -> QDialog:
    """Build a gui for loading data into secquoia."""
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.setWindowModality(Qt.WindowModal)

    vbox = QVBoxLayout(dlg)
    vbox.setContentsMargins(12, 12, 12, 12)
    vbox.setSpacing(10)

    bar, _ = add_progress_bar(vbox)

    status_lbl = QLabel("Switching position…")
    status_lbl.setWordWrap(True)
    vbox.addWidget(status_lbl)

    bridge = UiProgressBridge(bar, status_lbl, parent=dlg)
    start_btn = QPushButton("Start Curation")
    start_btn.setIcon(
        qta.icon("fa5s.arrow-right", color="#1db954", color_disabled="#5a5a5a")
    )
    start_btn.setEnabled(False)
    start_btn.setVisible(False)
    widen_to_hint(start_btn, extra_px=24, fix_policy=False)
    start_btn.setToolTip("Close this dialog and begin curation")
    start_btn.clicked.connect(dlg.close)

    row = QWidget(dlg)
    row_h = QHBoxLayout(row)
    row_h.setContentsMargins(0, 0, 0, 0)
    row_h.setSpacing(8)
    row_h.addStretch(1)
    row_h.addWidget(start_btn)
    vbox.addWidget(row)

    def _show_start_button(val):
        """Reveal the Start Curation button when progress reaches completion."""
        if val >= 100:
            start_btn.setEnabled(True)
            start_btn.setVisible(True)

    bar.valueChanged.connect(_show_start_button)

    dlg.resize(440, dlg.sizeHint().height())
    dlg.show()
    QApplication.processEvents()

    return dlg, bar, status_lbl, start_btn, bridge


def _make_subprogress_factory(bridge):
    """Return a factory that builds sub-progress callbacks for `bridge`."""

    def make_subprogress(start: int, end: int, on_first=None, on_done=None):
        """Create a callback that maps subtask progress into a progressbar range."""
        start_i = int(max(0, min(100, start)))
        end_i = int(max(0, min(100, end)))
        span = max(1, end_i - start_i)

        fired_first = {"v": False}
        fired_done = {"v": False}

        def cb(frac: float, msg: str = ""):
            """Map fractional subtask progress to the shared progress bar and trigger milestones."""
            f = max(0.0, min(1.0, float(frac)))
            val = min(end_i, start_i + round(f * span))

            if (not fired_first["v"]) and f > 0.0:
                fired_first["v"] = True
                if on_first:
                    QTimer.singleShot(0, on_first)

            if (not fired_done["v"]) and f >= 0.999:
                fired_done["v"] = True
                if on_done:
                    QTimer.singleShot(0, on_done)

            bridge.update.emit(val, msg or f"{int(f*100)}%")

        return cb

    return make_subprogress


def load_position(main_window: QWidget, direction: str) -> None:
    """Load the next or previous position based on the direction, showing a custom pop-up progress dialog."""
    if not hasattr(main_window, "folder_list") or not main_window.folder_list:
        LOG.warning("Please first load CSV file and select folder")
        show_folder_warning(main_window)
        return

    assert direction in ("next", "previous"), "Invalid direction"

    dlg, bar, status_lbl, start_curation_btn, bridge = _create_loading_dialog(
        main_window, title="Loading position"
    )

    make_subprogress = _make_subprogress_factory(bridge)
    bar.setValue(0)
    status_lbl.setText("Switching position…")
    QApplication.processEvents()
    QApplication.setOverrideCursor(Qt.WaitCursor)

    increment = 1 if direction == "next" else -1

    bar.setValue(10)
    status_lbl.setText("Saving current position state…")
    QApplication.processEvents()
    save_masks_incremental(main_window)
    save_track_df(main_window)

    if main_window.tracking_format == "tTt":
        if getattr(main_window, "clt_parser", None) is None:
            main_window.clt_parser = CLTParser(
                getattr(main_window, "xml_path", None)
            )
            main_window.clt_parser.folder_exp = main_window.tracking_path
        main_window.clt_parser.export_quantifications_to_clt(
            df=main_window.filtered_df,
            user=main_window.user,
            output_root=main_window.tracking_path,
        )

    bar.setValue(20)
    status_lbl.setText("Selecting new position…")
    QApplication.processEvents()
    main_window.current_position_index = (
        main_window.current_position_index + increment
    ) % len(main_window.position_folders)

    try:
        new_sel = main_window.position_folders[
            main_window.current_position_index
        ]
        main_window.current_position_number = int(
            os.path.basename(new_sel).split("_p")[-1]
        )
    except (RuntimeError, AttributeError, TypeError, ValueError):
        main_window.current_position_number = None

    fl_cb = make_subprogress(20, 45)
    basic_cb = make_subprogress(45, 70)
    meas_cb = make_subprogress(70, 95)

    progress_bundle = {"fl": fl_cb, "basic": basic_cb, "measure": meas_cb}

    bar.setValue(20)
    status_lbl.setText("Loading images and masks…")
    QApplication.processEvents()
    update_images(main_window, progress_cb=progress_bundle)

    bar.setValue(97)
    status_lbl.setText("Refreshing viewers…")
    QApplication.processEvents()

    main_window.update_napari_viewer()
    add_missing_rows(main_window)
    main_window.ALL.setChecked(True)
    main_window.update_window_title()
    main_window.current_outlier_index = 0

    pack_dict = getattr(main_window, "_last_outlier_rules", None)
    df = getattr(main_window, "filtered_df", None)
    if pack_dict and df is not None and not df.empty:
        try:
            pack = RulesPack.from_dict(pack_dict)
            df = run_outlier_pipeline(df, pack, outcol="Outlier_detection")
            main_window.filtered_df = df

            update_outlier_detection_in_track_df(
                main_window, outcol="Outlier_detection"
            )
            ensure_current_df_subset(main_window)
            update_unique_outliers_ids(main_window, outcol="Outlier_detection")
            _update_outlier_list(main_window, interactive=False)
            if hasattr(main_window, "Outliers") and not _outliers_is_empty(
                getattr(main_window, "unique_outliers_ids", None)
            ):
                main_window.Outliers.setChecked(True)
            update_outlier_marker(main_window)
        except (
            ValueError,
            KeyError,
            TypeError,
            RuntimeError,
            AttributeError,
        ) as e:
            LOG.warning("[Outliers] Auto re-run failed: %s", e)

    auto_select_first_item(main_window)

    bar.setValue(100)
    status_lbl.setText("Done.")
    start_curation_btn.setEnabled(True)
    start_curation_btn.setVisible(True)
    QApplication.processEvents()
    QApplication.restoreOverrideCursor()


def switch_to_position_with_progress(
    main_window,
    pos_number: int,
    *,
    save_current: bool = True,
) -> None:
    """Switch to the given position.

    Optionally saves the current position's state, locates the target
    position folder, loads its images and masks with weighted progress
    reporting, and refreshes the viewers, plots, and outlier list.
    """
    if getattr(main_window, "_switching_position", False):
        main_window._pending_position_switch = (int(pos_number), save_current)
        LOG.info(
            "[switch] A position switch is already in progress; queued "
            "p%04d to run next.",
            int(pos_number),
        )
        return

    main_window._switching_position = True
    main_window._pending_position_switch = None
    try:
        if not getattr(main_window, "folder_list", None):
            LOG.warning("Please first load CSV file and select folder")
            show_folder_warning(main_window)
            return

        _dlg, bar, status_lbl, start_curation_btn, bridge = (
            _create_loading_dialog(
                main_window, title=f"Loading p{int(pos_number):04d}"
            )
        )
        make_subprogress = _make_subprogress_factory(bridge)

        # 0–10%: save current position state
        if save_current:
            bar.setValue(5)
            status_lbl.setText("Saving current position state…")
            with contextlib.suppress(
                RuntimeError, AttributeError, ValueError, TypeError, OSError
            ):
                QApplication.processEvents()
                QApplication.setOverrideCursor(Qt.WaitCursor)

            try:
                save_masks_incremental(main_window)
            except (
                RuntimeError,
                AttributeError,
                ValueError,
                TypeError,
                OSError,
            ) as e:
                LOG.warning("[switch] save_masks_incremental: %s", e)

            try:
                save_track_df(main_window)
            except (
                RuntimeError,
                AttributeError,
                ValueError,
                TypeError,
                OSError,
            ) as e:
                LOG.warning("[switch] save_track_df: %s", e)

            if main_window.tracking_format == "tTt":
                if getattr(main_window, "clt_parser", None) is None:
                    main_window.clt_parser = CLTParser(
                        getattr(main_window, "xml_path", None)
                    )
                    main_window.clt_parser.folder_exp = (
                        main_window.tracking_path
                    )
                main_window.clt_parser.export_quantifications_to_clt(
                    df=main_window.filtered_df,
                    user=main_window.user,
                    output_root=main_window.tracking_path,
                )
        else:
            bar.setValue(5)
            status_lbl.setText("Selecting position…")
            with contextlib.suppress(
                RuntimeError, AttributeError, ValueError, TypeError, OSError
            ):
                QApplication.processEvents()

        # 10–20%: set new index/number
        bar.setValue(15)
        status_lbl.setText("Selecting position…")
        with contextlib.suppress(
            RuntimeError, AttributeError, ValueError, TypeError, OSError
        ):
            QApplication.processEvents()

        pos_number = int(pos_number)
        idx = None
        for i, folder in enumerate(
            getattr(main_window, "position_folders", None) or []
        ):
            if position_number_from_folder(folder) == pos_number:
                idx = i
                break

        if idx is None:
            status_lbl.setText(f"Position p{pos_number:04d} was not found.")
            bar.setValue(100)
            start_curation_btn.setEnabled(True)
            start_curation_btn.setVisible(True)
            return

        main_window.current_position_index = idx
        main_window.current_position_number = pos_number

        # Sub-progress split: FL 20→55, BaSiC 55→80, Measure 80→97
        fl_cb = make_subprogress(20, 55)
        basic_cb = make_subprogress(55, 80)
        meas_cb = make_subprogress(80, 97)
        progress_bundle = {"fl": fl_cb, "basic": basic_cb, "measure": meas_cb}

        # 20–97%: load/measure
        bar.setValue(20)
        status_lbl.setText("Loading images and masks…")
        with contextlib.suppress(
            RuntimeError, AttributeError, ValueError, TypeError, OSError
        ):
            QApplication.processEvents()
        update_images(main_window, progress_cb=progress_bundle)

        # 97–100%: refresh viewers + data + outliers
        bar.setValue(97)
        status_lbl.setText("Refreshing viewers…")
        with contextlib.suppress(
            RuntimeError, AttributeError, ValueError, TypeError, OSError
        ):
            QApplication.processEvents()

        main_window.update_napari_viewer()
        update_track_df(main_window)
        update_plot(main_window)
        bar.setValue(98)
        with contextlib.suppress(RuntimeError, AttributeError, ValueError):
            main_window.fit_all_plots()

        add_missing_rows(main_window)
        if hasattr(main_window, "ALL"):
            with contextlib.suppress(RuntimeError, AttributeError, ValueError):
                main_window.ALL.setChecked(True)
        main_window.update_window_title()
        main_window.current_outlier_index = 0

        pack_dict = getattr(main_window, "_last_outlier_rules", None)
        df = getattr(main_window, "filtered_df", None)
        if pack_dict and df is not None and not getattr(df, "empty", True):
            try:
                pack = RulesPack.from_dict(pack_dict)
                df = run_outlier_pipeline(df, pack, outcol="Outlier_detection")
                main_window.filtered_df = df
                update_outlier_detection_in_track_df(
                    main_window, outcol="Outlier_detection"
                )
                ensure_current_df_subset(main_window)
                update_unique_outliers_ids(
                    main_window, outcol="Outlier_detection"
                )
                _update_outlier_list(main_window, interactive=False)
                if hasattr(main_window, "Outliers") and not _outliers_is_empty(
                    getattr(main_window, "unique_outliers_ids", None)
                ):
                    with contextlib.suppress(
                        RuntimeError,
                        AttributeError,
                        ValueError,
                        TypeError,
                        OSError,
                    ):
                        main_window.Outliers.setChecked(True)
                update_outlier_marker(main_window)
            except (
                RuntimeError,
                AttributeError,
                ValueError,
                TypeError,
                OSError,
                ImportError,
                KeyError,
            ) as e:
                LOG.warning("[Outliers] Auto re-run failed: %s", e)

        with contextlib.suppress(
            RuntimeError, AttributeError, ValueError, TypeError, OSError
        ):
            auto_select_first_item(main_window)

        bar.setValue(100)
        status_lbl.setText("Done.")
        start_curation_btn.setEnabled(True)
        start_curation_btn.setVisible(True)
        with contextlib.suppress(
            RuntimeError, AttributeError, ValueError, TypeError, OSError
        ):
            QApplication.processEvents()
            QApplication.restoreOverrideCursor()
    finally:
        main_window._switching_position = False

    pending = main_window._pending_position_switch
    if pending is not None:
        main_window._pending_position_switch = None
        next_pos, next_save_current = pending
        QTimer.singleShot(
            0,
            lambda: switch_to_position_with_progress(
                main_window, next_pos, save_current=next_save_current
            ),
        )


def _switch_to_start_position(
    main_window, pmin: int, pmax: int, items
) -> None:
    """After a run-all pass, switch the viewer to (or nearest to) the originally requested position."""
    try:
        pstart = getattr(main_window, "position_start_selected", None)
        pstart = int(max(pmin, min(pstart, pmax)))
        processed = {n for (n, _p) in items}
        if pstart in processed:
            switch_to_position_with_progress(
                main_window, pstart, save_current=False
            )
        elif processed:
            nearest = min(processed, key=lambda n: abs(n - pstart))
            switch_to_position_with_progress(
                main_window, nearest, save_current=False
            )
    except (
        RuntimeError,
        AttributeError,
        ValueError,
        TypeError,
        OSError,
        StopIteration,
    ) as e:
        LOG.warning("[run_all_positions] start position switch skipped: %s", e)


def set_current_position_from_spinbox(main_window) -> None:
    """Set the current position index and number from the positions_start_combo spinbox."""
    pos_number = int(main_window.positions_start_combo.value())
    idx = None
    for i, folder in enumerate(main_window.position_folders):
        try:
            num = int(os.path.basename(folder).split("_p")[-1])
            if num == pos_number:
                idx = i
                break
        except (RuntimeError, AttributeError, TypeError, ValueError):
            continue
    if idx is not None:
        main_window.current_position_index = idx
        main_window.current_position_number = pos_number
    else:
        main_window.current_position_index = 0
        main_window.current_position_number = pos_number


def resolve_position_index(main_window, pos_number=None) -> None:
    """Return a factory that builds subprogress callbacks for `bridge`."""
    if pos_number is None:
        pos_number = getattr(main_window, "current_position_number", None)
    if pos_number is None:
        pos_number = (
            getattr(main_window, "position_start_selected", None)
            or getattr(main_window, "position_min_selected", None)
            or getattr(main_window, "position_min", 1)
        )
    pos_number = int(pos_number)

    idx = None
    for i, folder in enumerate(getattr(main_window, "position_folders", [])):
        try:
            if int(os.path.basename(folder).split("_p")[-1]) == pos_number:
                idx = i
                break
        except (ValueError, TypeError, IndexError):
            continue

    if idx is None:
        idx = 0
        if main_window.position_folders:
            pos_number = int(
                os.path.basename(main_window.position_folders[0]).split("_p")[
                    -1
                ]
            )
    main_window.current_position_index = idx
    main_window.current_position_number = pos_number


def parse_position_entries(folder_list) -> list[tuple[int, str]]:
    """Pick the position folders out of a folder list and sort them by number."""
    entries = []
    for folder in folder_list or []:
        if not folder or not folder[0].isdigit():
            continue
        match = _POSITION_RE.search(folder)
        if not match or os.path.splitext(folder)[1]:
            continue
        entries.append((int(match.group(1)), folder))

    entries.sort(key=lambda entry: entry[0])
    return entries


class PositionSelectDialog(QDialog):
    """Filterable list of positions, with a button to load the selected one."""

    def __init__(self, main_window, parent: QWidget | None = None):
        """Build the dialog and fill it with the available positions."""
        super().__init__(self._resolve_parent(main_window, parent))
        self.main_window = main_window

        self.setWindowTitle("Selection of a specific position")
        self.setMinimumSize(340, 460)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        main_window.pos_window = self

        self._build_ui()
        self._populate()
        self._connect_signals()

        self._sync_button_state()
        self.list_widget.setFocus()

    @staticmethod
    def _resolve_parent(main_window, parent: QWidget | None) -> QWidget | None:
        """Choose the widget this dialog should be parented to."""
        if parent is not None:
            return parent
        return main_window if isinstance(main_window, QWidget) else None

    def _build_ui(self) -> None:
        """Create the header, filter box, list and button row."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        layout.addWidget(self._build_title())
        subtitle = QLabel(
            "Select a position, then press <b>Load this Position</b>."
        )
        subtitle.setStyleSheet("color: #ffffff;")
        layout.addWidget(subtitle)

        layout.addWidget(self._build_search_edit())

        self.list_widget = self._build_list_widget()
        layout.addWidget(self.list_widget, 1)

        self.count_lbl = QLabel()
        self.count_lbl.setStyleSheet("color: #ffffff;")
        layout.addWidget(self.count_lbl)

        layout.addLayout(self._build_button_row())

    def _build_title(self) -> QLabel:
        """Build the bold heading above the list."""
        title = QLabel("List of positions")
        font = title.font()
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() + 2)
        title.setFont(font)
        title.setToolTip(TOOLTIPSTEXT.ALL_POS)
        return title

    def _build_search_edit(self) -> QLineEdit:
        """Build the filter box, with a readable placeholder on the dark theme."""
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Filter positions… (e.g. 0008)")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setToolTip(TOOLTIPSTEXT.EDIT_POS)

        palette = self.search_edit.palette()
        palette.setColor(QPalette.PlaceholderText, QColor("#ffffff"))
        self.search_edit.setPalette(palette)
        return self.search_edit

    def _build_list_widget(self) -> QListWidget:
        """Build the position list."""
        widget = QListWidget(self)
        widget.setSelectionMode(QAbstractItemView.SingleSelection)
        widget.setSelectionBehavior(QAbstractItemView.SelectRows)
        widget.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        widget.setUniformItemSizes(True)
        widget.setToolTip(TOOLTIPSTEXT.SELECT_POS)
        widget.setStyleSheet(_LIST_STYLESHEET)
        return widget

    def _build_button_row(self) -> QHBoxLayout:
        """Build the Load / Exit row."""
        self.load_btn = QPushButton("Load this position")
        self.load_btn.setDefault(True)
        self.load_btn.setToolTip(TOOLTIPSTEXT.LOAD_POS)

        self.exit_btn = QPushButton("Exit")
        self.exit_btn.setToolTip(TOOLTIPSTEXT.EXIT_POS)

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.load_btn)
        row.addWidget(self.exit_btn)
        return row

    def _populate(self) -> None:
        """Fill the list, marking and selecting the current position."""
        current_pos = getattr(
            self.main_window, "current_position_number", None
        )

        for pos_number, folder in parse_position_entries(
            self.main_window.folder_list
        ):
            item = QListWidgetItem(folder)
            item.setData(Qt.UserRole, pos_number)

            if current_pos is not None and pos_number == int(current_pos):
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                item.setText(f"{folder}   ● current")
                item.setToolTip(
                    f"Position {pos_number:04d} (currently loaded)"
                )
                self.list_widget.addItem(item)
                self.list_widget.setCurrentItem(item)
                continue

            item.setToolTip(f"Load position {pos_number:04d}")
            self.list_widget.addItem(item)

        self.count_lbl.setText(
            f"{self.list_widget.count()} position(s) available"
        )

    def _connect_signals(self) -> None:
        """Wire the filter box, the list and the two buttons."""
        self.search_edit.textChanged.connect(self._on_filter)
        self.list_widget.itemSelectionChanged.connect(self._sync_button_state)
        self.list_widget.itemDoubleClicked.connect(
            lambda _item: self._on_load_clicked()
        )
        self.load_btn.clicked.connect(self._on_load_clicked)
        self.exit_btn.clicked.connect(self.close)

    def _selected_item(self) -> QListWidgetItem | None:
        """Return the highlighted item, ignoring ones hidden by the filter."""
        items = self.list_widget.selectedItems()
        item = items[0] if items else None
        if item is not None and item.isHidden():
            return None
        return item

    def _sync_button_state(self) -> None:
        """Enable the Load button only when a real position is highlighted."""
        item = self._selected_item()
        self.load_btn.setEnabled(
            item is not None and item.data(Qt.UserRole) is not None
        )

    def _on_filter(self, text: str) -> None:
        """Hide items that do not match the filter text."""
        needle = text.strip().lower()
        visible = 0
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            match = needle in item.text().lower()
            item.setHidden(not match)
            visible += int(match)

        self.count_lbl.setText(
            f"{visible} of {self.list_widget.count()} position(s) shown"
        )
        self._sync_button_state()

    def _on_load_clicked(self) -> None:
        """Close the window and switch to the highlighted position."""
        item = self._selected_item()
        if item is None or item.data(Qt.UserRole) is None:
            return
        pos_number = int(item.data(Qt.UserRole))
        self.close()
        switch_to_position_with_progress(self.main_window, pos_number)


def open_position_window(main_window: QWidget) -> PositionSelectDialog | None:
    """Open the position-selection window, if an experiment is loaded."""
    if not getattr(main_window, "folder_list", None):
        LOG.warning("Please first load CSV file and select folder")
        show_folder_warning(main_window)
        return None

    previous = getattr(main_window, "pos_window", None)
    if previous is not None:
        with contextlib.suppress(RuntimeError, AttributeError):
            previous.close()

    dialog = PositionSelectDialog(main_window)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog
