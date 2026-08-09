"""Window level lifecycle for MainWindow: opening subwindows (exporter, tTt transformer,
help), the time marker toggle, the cellfate button bar, save/close/export, splitter/layout
sizing, the window title, and the Tree-ID filter checkbox.
"""

import contextlib
import logging
from textwrap import dedent

import pyqtgraph as pg
import qtawesome as qta
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import (
    QAbstractButton,
    QApplication,
    QMessageBox,
    QPushButton,
    QWidget,
)

from SECQUOIA.core.memmap_store import cleanup_memmaps
from SECQUOIA.core.project_state import save_project_state, save_track_df
from SECQUOIA.core.segmentation.mask_io import save_masks_incremental
from SECQUOIA.core.tracking.clt_io import CLTParser
from SECQUOIA.gui.cell_inspector.integration import detach_cell_inspector
from SECQUOIA.gui.common.help import HelpDocs, HelpPopup
from SECQUOIA.gui.common.messages import (
    apply_dialog_platform_style,
    show_folder_warning,
)
from SECQUOIA.gui.curation_tree import update_list
from SECQUOIA.gui.exporter import ImageMovieExporter
from SECQUOIA.gui.outlier.outlier_list import _update_outlier_list
from SECQUOIA.gui.ttt_data_format_transformer import TttDataFormatTransformer
from SECQUOIA.utils.helpers import update_time_marker
from SECQUOIA.utils.positions import position_number_at_current_index

LOG = logging.getLogger(__name__)


class WindowLifecycle:
    """Subwindows, the time marker toggle, save/close/export, layout sizing, title, and the tree filter."""

    def open_image_movie_exporter(self):
        """Open the single image and movie exporter for the loaded experiment."""
        if not getattr(self, "folder_list", None):
            LOG.warning("Please first load CSV file and select folder")
            show_folder_warning(self)
            return None

        controller = ImageMovieExporter(self)
        controller.open_image_movie_export_window(self)
        return controller

    def open_ttt_dataformat_transformer(self) -> None:
        """Open the tTt data format transformer GUI."""
        if getattr(self, "_ttt_transformer", None) is None:
            w = TttDataFormatTransformer()
            w.setAttribute(Qt.WA_DeleteOnClose, True)
            w.destroyed.connect(
                lambda *_: setattr(self, "_ttt_transformer", None)
            )
            self._ttt_transformer = w

        self._ttt_transformer.show()
        self._ttt_transformer.raise_()
        self._ttt_transformer.activateWindow()

    def _clear_time_markers(self) -> None:
        """Remove all currently drawn time markers (dynamics plots + lineage tree)."""
        if hasattr(self, "current_time_markers"):
            for marker in list(self.current_time_markers.values()):
                with contextlib.suppress(
                    RuntimeError, AttributeError, TypeError
                ):
                    pw = marker.get("plot_widget", None)
                    item = marker.get("item", None)
                    if isinstance(pw, pg.PlotWidget) and item is not None:
                        pw.removeItem(item)
        self.current_time_markers = {}

        old_line = getattr(self, "_lineage_time_line", None)
        if isinstance(old_line, pg.InfiniteLine):
            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                for attr in ("graph3_plot", "graph3_widget"):
                    pw = getattr(self, attr, None)
                    if isinstance(pw, pg.PlotWidget):
                        plot_item = pw.getPlotItem()
                        plot_item.removeItem(old_line)
                        break
        self._lineage_time_line = None

    def on_toggle_time_marker(self, checked: bool) -> None:
        """Menu toggle handler: show/hide the time marker everywhere."""
        self.show_time_marker = bool(checked)
        if self.show_time_marker:
            update_time_marker(self)
        else:
            self._clear_time_markers()

    def show_help_popup(self, checked: bool = False) -> None:
        """Show the hotkeys and mouse controls help window."""
        if self.help_popup is None:
            self.help_popup = HelpPopup(
                parent=self, title="Hotkeys & Mouse — Help"
            )
            self.help_popup.load_text(
                dedent(HelpDocs.HOTKEYS_MOUSE_MD).strip(), as_markdown=True
            )

            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                self.help_popup.destroyed.connect(
                    lambda *_: setattr(self, "help_popup", None)
                )

        self.help_popup.show()
        self.help_popup.raise_()
        self.help_popup.activateWindow()

    def _fate_buttons(self) -> list[QPushButton]:
        """Return the six cellfate buttons as a list."""
        return [
            self.btn_healthy,
            self.btn_dead,
            self.btn_lost,
            self.btn_oof,
            self.btn_outlier,
            self.btn_lowsignal,
        ]

    def _remove_from_any_layout(self, w: QWidget) -> None:
        """Detach a widget from its parent layout."""
        try:
            pw = w.parentWidget()
            if pw and pw.layout():
                pw.layout().removeWidget(w)
        except (RuntimeError, AttributeError, TypeError):
            pass
        w.setParent(None)

    def _move_fate_buttons_to_bottom_bar(self) -> None:
        """Move the cellfate buttons into the bottom bar and show it."""
        if (
            hasattr(self, "fate_bar_container")
            and self.fate_bar_container
            and self.layout2.indexOf(self.fate_bar_container) == -1
        ):
            self.layout2.addWidget(self.fate_bar_container)
        self.fate_bar_container.setVisible(True)

        for btn in self._fate_buttons():
            self.fate_bar.addWidget(btn)

        if (
            hasattr(self, "_napari_container")
            and self._napari_container
            and self.layout2.indexOf(self._napari_container) == -1
        ):
            self.layout2.insertWidget(0, self._napari_container)

    def _set_splitter_equal(self) -> None:
        """Set the main splitter panes to equal sizes."""
        if not hasattr(self, "splitter") or self.splitter is None:
            return
        if self.splitter.orientation() == Qt.Horizontal:
            total = max(2, self.splitter.size().width())
        else:
            total = max(2, self.splitter.size().height())
        half = total // 2
        self.splitter.setSizes([half, total - half])

    def _set_splitter_equal_soon(self) -> None:
        """Set the main splitter panes to equal sizes after a short delay."""
        QTimer.singleShot(0, self._set_splitter_equal)
        QTimer.singleShot(50, self._set_splitter_equal)

    def on_save_clicked(self) -> None:
        """Save project data, tracking data, and masks."""
        if not hasattr(self, "folder_list") or not self.folder_list:
            LOG.warning("Please first load CSV file and select folder")
            show_folder_warning(self)
            return

        btn = self.save_button
        orig_ss = btn.styleSheet()
        orig_icon = btn.icon()
        btn.setEnabled(False)

        # Turn green while saving
        btn.setStyleSheet(
            orig_ss
            + "QPushButton { background-color: #2ecc71; color: white; }"
        )
        try:
            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                btn.setIcon(qta.icon("fa5s.save", color="white"))

            QApplication.processEvents()
            save_track_df(self)
            save_project_state(self)
            if self.tracking_format == "tTt":
                if getattr(self, "clt_parser", None) is None:
                    self.clt_parser = CLTParser(
                        getattr(self, "xml_path", None)
                    )
                    self.clt_parser.folder_exp = self.tracking_path
                self.clt_parser.export_quantifications_to_clt(
                    df=self.filtered_df,
                    user=self.user,
                    output_root=self.tracking_path,
                )

            save_masks_incremental(self)

        except (RuntimeError, AttributeError, TypeError):
            btn.setStyleSheet(
                orig_ss
                + "QPushButton { background-color: #e74c3c; color: white; }"
            )
            QApplication.processEvents()
            raise
        finally:
            btn.setStyleSheet(orig_ss)
            btn.setIcon(orig_icon)
            btn.setEnabled(True)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Confirm exit with Close/Export data/Cancel.

        Close releases the session resources and quits, Export data saves
        first and then quits, Cancel only dismisses the popup.
        """
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Confirm Exit")
        dlg.setText("Are you sure you want to close SECQUOIA?")
        dlg.setInformativeText("Any unsaved changes will be lost.")
        dlg.setIcon(QMessageBox.Warning)

        btn_yes = dlg.addButton("Close", QMessageBox.AcceptRole)
        btn_export = dlg.addButton("Export data", QMessageBox.ActionRole)
        btn_no = dlg.addButton("Cancel", QMessageBox.RejectRole)
        dlg.setDefaultButton(btn_no)
        dlg.setEscapeButton(btn_no)

        apply_dialog_platform_style(dlg)

        try:
            dlg.exec_()
        except AttributeError:
            dlg.exec()

        clicked = dlg.clickedButton()

        if clicked is not btn_yes and clicked is not btn_export:
            event.ignore()
            return

        if clicked is btn_export:
            try:
                self.on_save_clicked()
            except (
                OSError,
                RuntimeError,
                ValueError,
                KeyError,
                AttributeError,
                TypeError,
            ):
                LOG.exception("[close] save before exit failed")
                QMessageBox.warning(
                    self,
                    "Save failed",
                    "Your data could not be saved, so SECQUOIA stays "
                    "open. See the log for details.",
                )
                event.ignore()
                return

        try:
            detach_cell_inspector(self)
            cleanup_memmaps(self)
        except (RuntimeError, AttributeError, TypeError) as e:
            LOG.warning("[close] cleanup warning: %s", e)

        event.accept()

    def export_all(self) -> None:
        """Export tracking data, project state, and masks."""
        save_track_df(self)
        save_project_state(self)
        if self.tracking_format == "tTt":
            self.clt_parser.export_quantifications_to_clt(
                df=self.track_df,
                user=self.user,
                output_root=self.tracking_path,
            )
        save_masks_incremental(self)

    def _after_show(self) -> None:
        """Maximize the window and finalize layout sizing."""
        self.setWindowState(Qt.WindowMaximized)
        self.main_layout.activate()
        self._set_splitter_equal_soon()

    def _force_max_layout(self, ratio: float = 0.30) -> None:
        """Force Qt to recompute the window layout without changing window state."""
        self.setUpdatesEnabled(False)

        self.main_layout.invalidate()
        self.main_layout.activate()
        QApplication.processEvents()

        self._apply_splitter_sizes(ratio)

    def _apply_splitter_sizes(self, ratio: float = 0.30) -> None:
        """Adjust the splitter sizes according to the current window width."""
        if hasattr(self, "splitter") and self.splitter is not None:
            total_width = max(1, self.width())
            left_width = int(ratio * total_width)
            right_width = max(1, total_width - left_width)
            self.splitter.setSizes([left_width, right_width])

        self.main_layout.activate()
        self.updateGeometry()
        self.setUpdatesEnabled(True)

    def update_window_title(self) -> None:
        """Update the window title according to the current position."""
        experiment_name = getattr(self, "experiment_name", "Unknown")

        position_number = position_number_at_current_index(self)
        if position_number is None:
            self.setWindowTitle(f"SECQUOIA - {experiment_name}")
            return

        self.setWindowTitle(
            f"SECQUOIA - {experiment_name} - Position {position_number}"
        )

    def on_checkbox_toggled(
        self, button: QAbstractButton, checked: bool
    ) -> None:
        """Update the tree list when the filter checkbox changes."""
        if not checked:
            return
        if button is self.ALL:
            update_list(self)
        elif button is self.Outliers:
            _update_outlier_list(self)
