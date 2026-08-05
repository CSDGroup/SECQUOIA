"""Dialog for fusing two lineage trees at a chosen time point."""

from __future__ import annotations

import contextlib
import logging
from functools import partial

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from SECQUOIA.config import TOOLTIPSTEXT
from SECQUOIA.core.segmentation.mask_selection import show_all_masks
from SECQUOIA.core.tracking import on_fuse_trees_clicked
from SECQUOIA.gui.common.messages import show_folder_warning
from SECQUOIA.utils.helpers import jump_to_identification
from SECQUOIA.utils.timing import current_t_range as _current_t_range

LOG = logging.getLogger(__name__)

__all__ = [
    "TrackFuseDialog",
    "open_track_fuse_window",
    "reset_fuse_dialog_fields",
]


class TrackFuseDialog(QDialog):
    """Dialog for fusing two lineage trees at a chosen time point."""

    def __init__(self, main_window: QWidget):
        """Build the dialog and attach its input widgets to main_window."""
        super().__init__(TrackFuseDialog._resolve_parent(main_window))
        self.main_window = main_window

        self.setWindowTitle("Fuse Trees")
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.setSizeGripEnabled(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(0, 0)
        self.setToolTip(TOOLTIPSTEXT.FUSE_WINDOW)
        main_v = QVBoxLayout(self)

        main_window.fuse_tree_id_field_1 = QLineEdit()
        main_window.fuse_tree_id_field_2 = QLineEdit()
        main_window.fuse_time_spin = QSpinBox()
        main_window.fuse_tree_id_field_1.setReadOnly(True)
        main_window.fuse_tree_id_field_2.setReadOnly(True)
        main_window.fuse_cell_field_1 = QLineEdit()
        main_window.fuse_cell_field_2 = QLineEdit()
        main_window.fuse_cell_field_1.setReadOnly(True)
        main_window.fuse_cell_field_2.setReadOnly(True)
        main_window.fuse_tree_id_field_1.setPlaceholderText("Tree ID 1")
        main_window.fuse_tree_id_field_2.setPlaceholderText("Tree ID 2")
        main_window.fuse_cell_field_1.setPlaceholderText("Cell")
        main_window.fuse_cell_field_2.setPlaceholderText("Cell")

        main_window.fuse_tree_id_field_1.setToolTip(TOOLTIPSTEXT.FUSE_ID_1)
        main_window.fuse_tree_id_field_2.setToolTip(TOOLTIPSTEXT.FUSE_ID_2)
        main_window.fuse_cell_field_1.setToolTip(TOOLTIPSTEXT.CELL_1)
        main_window.fuse_cell_field_2.setToolTip(TOOLTIPSTEXT.CELL_2)
        main_window.fuse_time_spin.setToolTip(TOOLTIPSTEXT.FUSE_TIME)

        # Start alternation on field 1
        main_window._fuse_next_slot = 1

        # Initial spinbox range + value
        t_min, t_max = 0, max(
            int(getattr(main_window, "current_time_index", 0)), 0
        )
        with contextlib.suppress(Exception):
            _, _, t_idx_min, t_idx_max = _current_t_range(main_window)
            t_min, t_max = int(t_idx_min), int(t_idx_max)

        main_window.fuse_time_spin.setRange(t_min, t_max)
        cur_t = int(getattr(main_window, "current_time_index", 0))
        with contextlib.suppress(Exception):
            cur_t = max(t_min, min(t_max, cur_t))
        main_window.fuse_time_spin.setValue(cur_t)

        fuse_btn = QPushButton("Fuse")
        fuse_btn.setToolTip(TOOLTIPSTEXT.FUSE_BTN)
        fuse_btn.clicked.connect(partial(on_fuse_trees_clicked, main_window))

        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 0)

        l_tid1 = QLabel("Tree ID 1:")
        l_cell1 = QLabel("Cell 1:")
        l_tid2 = QLabel("Tree ID 2:")
        l_cell2 = QLabel("Cell 2:")

        l_tid1.setToolTip(TOOLTIPSTEXT.TID1)
        l_cell1.setToolTip(TOOLTIPSTEXT.L_CELL_1)
        l_tid2.setToolTip(TOOLTIPSTEXT.TID2)
        l_cell2.setToolTip(TOOLTIPSTEXT.L_CELL_2)

        # Row 0: Tree ID 1, Cell 1
        grid.addWidget(l_tid1, 0, 0)
        grid.addWidget(main_window.fuse_tree_id_field_1, 0, 1)
        grid.addWidget(l_cell1, 0, 2)
        grid.addWidget(main_window.fuse_cell_field_1, 0, 3)

        # Row 1: Tree ID 2, Cell 2
        grid.addWidget(l_tid2, 1, 0)
        grid.addWidget(main_window.fuse_tree_id_field_2, 1, 1)
        grid.addWidget(l_cell2, 1, 2)
        grid.addWidget(main_window.fuse_cell_field_2, 1, 3)

        main_v.addLayout(grid)
        main_v.addSpacing(6)

        # Fuse controls below the rows
        l_fuse_at = QLabel("Fuse Trees at Timepoint (t):")
        l_fuse_at.setToolTip(TOOLTIPSTEXT.L_FUSE_AT)
        main_v.addWidget(l_fuse_at)
        main_v.addWidget(main_window.fuse_time_spin)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(fuse_btn)
        main_v.addLayout(btn_row)

    @staticmethod
    def _resolve_parent(main_window: QWidget):
        """Return the best Qt parent for the dialog, or None."""
        parent = None
        with contextlib.suppress(Exception):
            if isinstance(main_window, QWidget):
                parent = main_window
            else:
                pw = getattr(main_window, "parentWidget", None)
                if callable(pw):
                    parent = pw()
                if parent is None:
                    p = getattr(main_window, "parent", None)
                    if callable(p):
                        parent = p()
        return parent

    def refresh_time_range(self):
        """Re-sync the time point spinbox range and value with the current data."""
        main_window = self.main_window
        with contextlib.suppress(Exception):
            _, _, t_idx_min, t_idx_max = _current_t_range(main_window)
            main_window.fuse_time_spin.setRange(int(t_idx_min), int(t_idx_max))

        with contextlib.suppress(Exception):
            cur_t = int(getattr(main_window, "current_time_index", 0))
            cur_t = max(
                main_window.fuse_time_spin.minimum(),
                min(main_window.fuse_time_spin.maximum(), cur_t),
            )
        with contextlib.suppress(Exception):
            main_window.fuse_time_spin.setValue(cur_t)


def reset_fuse_dialog_fields(
    main_window: QWidget, new_ident: str, t_fuse: int, g1: int
) -> None:
    """Clear the fuse fields, select the merged tree and close the dialog.

    The fields live on ``main_window`` (set by ``TrackFuseDialog.__init__``)
    rather than on the dialog instance, so this clears them even when no
    dialog is currently open.
    """
    with contextlib.suppress(Exception):
        for name in (
            "fuse_tree_id_field_1",
            "fuse_tree_id_field_2",
            "fuse_cell_field_1",
            "fuse_cell_field_2",
        ):
            widget = getattr(main_window, name, None)
            if widget is not None:
                widget.clear()
        jump_to_identification(
            main_window, ident=new_ident, t=int(t_fuse), tracknumber=int(g1)
        )
        main_window._fuse_next_slot = 1
        dialog = getattr(main_window, "fuse_dialog", None)
        if dialog is not None:
            dialog.close()


def open_track_fuse_window(main_window: QWidget) -> None:
    """Open (or re-show) the track-fuse dialog."""
    if not hasattr(main_window, "folder_list") or not main_window.folder_list:
        LOG.warning("Please first load CSV file and select folder")
        show_folder_warning(main_window)
        return

    show_all_masks(main_window)

    dlg = getattr(main_window, "fuse_dialog", None)
    if dlg is None:
        dlg = TrackFuseDialog(main_window)
        main_window.fuse_dialog = dlg

    # Refresh t-range and current value every time it opens
    dlg.refresh_time_range()

    with contextlib.suppress(Exception):
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
