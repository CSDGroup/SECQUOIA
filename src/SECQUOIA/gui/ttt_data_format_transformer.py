"""Copy microscopy experiment images into the tTt DataFormat layout.

This module provides a Qt widget that scans an experiment folder, uses
SECQUOIA.core.ttt_naming to parse image filenames into their date,
initials, setup, position, time point, z-slice, and channel tokens, and
copies the images into

    <output>/240323MA35/
        Analysis/
        240323MA35_p0001/240323MA35_p0001_t00001_z001_w00.png

The source folder is never modified; the renaming happens on the copies.
"""

import os
import shutil
from collections.abc import Iterator

import qtawesome as qta
from qtpy import QtWidgets
from qtpy.QtCore import QRegularExpression, Qt
from qtpy.QtGui import QRegularExpressionValidator

from SECQUOIA.config import LINKS, TOOLTIPSTEXT
from SECQUOIA.core.ttt_naming import (
    active_preset_name,
    channel_is_one_indexed,
    list_preset_names,
    normalize_initials,
    normalize_pos,
    normalize_setup,
    normalize_t,
    normalize_w,
    normalize_z,
    parse_experiment_token_from_filename,
    parse_name,
    reload_patterns,
    resolve_positions,
    set_active_preset,
    yymmdd_compact,
)
from SECQUOIA.gui.common.ui_utils import add_progress_bar, make_help_button
from SECQUOIA.gui.dialogs.ttt_pattern_dialog import TttPatternDialog

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
SETUP_VALIDATOR_RE = QRegularExpression(r"\d{0,3}")
INITIALS_VALIDATOR_RE = QRegularExpression(r"[A-Za-z]{0,4}")


def iter_image_files(root: str) -> Iterator[str]:
    """Yield image files directly inside root. Subfolders are not searched."""
    with os.scandir(root) as it:
        for entry in sorted(it, key=lambda e: e.name):
            if not entry.is_file():
                continue
            if os.path.splitext(entry.name)[1].lower() in IMAGE_EXTS:
                yield entry.path


def no_images_message(folder: str) -> str:
    """Warning text for an Experiment folder holding no images at top level."""
    msg = "No image files found directly in the Experiment folder."
    nested = any(
        os.path.splitext(fn)[1].lower() in IMAGE_EXTS
        for dirpath, _, filenames in os.walk(folder)
        if os.path.abspath(dirpath) != os.path.abspath(folder)
        for fn in filenames
    )
    if nested:
        msg += (
            "\n\nImages were found in subfolders. This tool only reads the "
            "top level. Please select the folder that contains the images "
            "themselves."
        )
    return msg


def find_one_image_file(folder: str) -> str:
    """Return the first image file directly inside a folder."""
    for path in iter_image_files(folder):
        return path
    return ""


def hrow(*widgets) -> QtWidgets.QWidget:
    """Wrap widgets in a QWidget with a tight horizontal layout."""
    w = QtWidgets.QWidget()
    lay = QtWidgets.QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(8)
    for x in widgets:
        lay.addWidget(x)
    return w


class TttDataFormatTransformer(QtWidgets.QWidget):
    """Qt widget that copies experiment images into the tTt folder layout."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(
            "Experiment Folder Renamer - tTt DataFormat Transformer"
        )
        self.in_folder = ""
        self.out_folder = ""

        # Help button
        help_btn = make_help_button(
            self, TOOLTIPSTEXT.HELP_BTN_tTt, LINKS.GITHUB_tTt_FORMAT
        )

        # Pattern settings button
        settings_btn = QtWidgets.QToolButton()
        settings_btn.setAutoRaise(True)
        settings_btn.setCursor(Qt.PointingHandCursor)
        settings_btn.setIcon(qta.icon("mdi.plus", color="white"))
        settings_btn.setToolTip(TOOLTIPSTEXT.PATTERN_SETTINGS_tTt)
        settings_btn.clicked.connect(self.open_pattern_settings)

        # Pattern selector
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.setToolTip(TOOLTIPSTEXT.PRESET_COMBO_tTt)
        self.refresh_preset_combo()
        self.preset_combo.activated.connect(self.on_preset_selected)

        # Input row controls
        self.in_path = QtWidgets.QLineEdit()
        self.in_path.setReadOnly(True)
        btn_in_select = QtWidgets.QPushButton("Select")
        btn_read = QtWidgets.QPushButton("Read File naming")

        # Output row controls
        self.out_path = QtWidgets.QLineEdit()
        self.out_path.setReadOnly(True)
        btn_out_select = QtWidgets.QPushButton("Select")
        btn_run = QtWidgets.QPushButton("Run")

        folder_icn = qta.icon("mdi.folder-open-outline", color="white")
        play_icn = qta.icon("mdi.play", color="white")

        btn_in_select.setIcon(folder_icn)
        btn_out_select.setIcon(folder_icn)
        btn_read.setIcon(play_icn)
        btn_run.setIcon(play_icn)

        # Editable fields
        self.ed_date = QtWidgets.QLineEdit()
        self.ed_date.setPlaceholderText("e.g. 240323 (YYMMDD)")
        self.ed_initials = QtWidgets.QLineEdit()
        self.ed_initials.setPlaceholderText("e.g. MA (2 letters required)")
        self.ed_setup = QtWidgets.QLineEdit()
        self.ed_setup.setPlaceholderText("digits only, e.g. 01 or 30")

        self.ed_date.editingFinished.connect(self._normalize_date_field)

        self.ed_initials.setValidator(
            QRegularExpressionValidator(INITIALS_VALIDATOR_RE, self)
        )
        self.ed_initials.setMaxLength(4)
        self.ed_initials.editingFinished.connect(
            lambda: self.ed_initials.setText(
                normalize_initials(self.ed_initials.text())
            )
        )

        self.ed_setup.setValidator(
            QRegularExpressionValidator(SETUP_VALIDATOR_RE, self)
        )
        self.ed_setup.setMaxLength(3)
        self.ed_setup.editingFinished.connect(
            lambda: self.ed_setup.setText(
                normalize_setup(self.ed_setup.text())
            )
        )

        self.ed_pos = QtWidgets.QLineEdit()
        self.ed_pos.setPlaceholderText("e.g. p0001")
        self.ed_time = QtWidgets.QLineEdit()
        self.ed_time.setPlaceholderText("e.g. t00001")
        self.ed_z = QtWidgets.QLineEdit()
        self.ed_z.setPlaceholderText("e.g. z001 (defaults to z001)")
        self.ed_ch = QtWidgets.QLineEdit()
        self.ed_ch.setPlaceholderText("e.g. w00")
        for w in (
            self.ed_date,
            self.ed_initials,
            self.ed_setup,
            self.ed_pos,
            self.ed_time,
            self.ed_z,
            self.ed_ch,
        ):
            w.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
            )

        # Progress bar under Run row
        self.progress, _ = add_progress_bar(None)

        grid = QtWidgets.QGridLayout(self)
        grid.setColumnStretch(1, 1)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

        r = 0
        # Help row
        grid.addWidget(
            hrow(settings_btn, help_btn), r, 1, alignment=Qt.AlignRight
        )
        r += 1

        # Parsing pattern row
        grid.addWidget(QtWidgets.QLabel("Parsing pattern:"), r, 0)
        grid.addWidget(self.preset_combo, r, 1)
        r += 1

        # Experiment folder row
        grid.addWidget(QtWidgets.QLabel("Experiment folder:"), r, 0)
        self.in_path.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        grid.addWidget(hrow(self.in_path, btn_in_select, btn_read), r, 1)
        r += 1

        # Normal fields: label + full-width text box
        grid.addWidget(QtWidgets.QLabel("Experiment date (YYMMDD)"), r, 0)
        grid.addWidget(self.ed_date, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Initials"), r, 0)
        grid.addWidget(self.ed_initials, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Microscope setup"), r, 0)
        grid.addWidget(self.ed_setup, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Positions"), r, 0)
        grid.addWidget(self.ed_pos, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Time point"), r, 0)
        grid.addWidget(self.ed_time, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Z position"), r, 0)
        grid.addWidget(self.ed_z, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Channel"), r, 0)
        grid.addWidget(self.ed_ch, r, 1)
        r += 1

        # Output folder row
        grid.addWidget(QtWidgets.QLabel("Output folder:"), r, 0)
        self.out_path.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        grid.addWidget(hrow(self.out_path, btn_out_select, btn_run), r, 1)
        r += 1

        grid.addWidget(self.progress, r, 1)
        r += 1

        # Signals
        btn_in_select.clicked.connect(self.select_input_folder)
        btn_read.clicked.connect(self.read_naming)
        btn_out_select.clicked.connect(self.select_output_folder)
        btn_run.clicked.connect(self.run)

        # Tooltips
        self.in_path.setToolTip(TOOLTIPSTEXT.SELECT_BTN_tTt)
        btn_in_select.setToolTip(TOOLTIPSTEXT.INPUT_FOLDER_tTt)
        btn_read.setToolTip(TOOLTIPSTEXT.BTN_READ)
        self.out_path.setToolTip(TOOLTIPSTEXT.OUTPUT_FOLDER_tTt)
        btn_out_select.setToolTip(TOOLTIPSTEXT.SELECT_OUTPUT_FOLDER_tTt)
        btn_run.setToolTip(TOOLTIPSTEXT.RUN_tTt)
        self.ed_date.setToolTip(TOOLTIPSTEXT.DATE_tTt)
        self.ed_initials.setToolTip(TOOLTIPSTEXT.INITIALS_tTt)
        self.ed_setup.setToolTip(TOOLTIPSTEXT.SETUP_tTt)
        self.ed_pos.setToolTip(TOOLTIPSTEXT.POSITION_tTt)
        self.ed_time.setToolTip(TOOLTIPSTEXT.TIME_tTt)
        self.ed_z.setToolTip(TOOLTIPSTEXT.Z_POSITION_tTt)
        self.ed_ch.setToolTip(TOOLTIPSTEXT.CHANNEL_tTt)
        self.progress.setToolTip(TOOLTIPSTEXT.PROGRESS_tTt)

        self.adjustSize()
        self.setMinimumSize(self.minimumSizeHint())

    def open_pattern_settings(self) -> None:
        """Open the dialog for editing filename-parsing patterns."""
        dlg = TttPatternDialog(self)
        if dlg.exec_():
            errors = reload_patterns()
            self.refresh_preset_combo()
            if errors:
                self._warn(
                    "Some patterns were not used",
                    "These patterns did not compile and fall back to the "
                    "default for that field:\n\n"
                    + "\n".join(f"- {f}: {e}" for f, e in errors.items()),
                )

    def refresh_preset_combo(self) -> None:
        """Repopulate the preset dropdown and select the active one."""
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItems(list_preset_names())
        self.preset_combo.setCurrentText(active_preset_name())
        self.preset_combo.blockSignals(False)

    def on_preset_selected(self) -> None:
        """Switch which saved preset is used to parse filenames."""
        errors = set_active_preset(self.preset_combo.currentText())
        if errors:
            self._warn(
                "Some patterns were not used",
                "These patterns did not compile and fall back to the "
                "default for that field:\n\n"
                + "\n".join(f"- {f}: {e}" for f, e in errors.items()),
            )

    def _normalize_date_field(self) -> None:
        """Rewrite whatever the user typed as YYMMDD, if it parses."""
        compact = yymmdd_compact(self.ed_date.text())
        if compact:
            self.ed_date.setText(compact)

    def _warn(self, title: str, text: str) -> None:
        """Show a warning message box with the given title and text."""
        QtWidgets.QMessageBox.warning(self, title, text)

    def select_input_folder(self) -> None:
        """Prompt for the source experiment folder and store the chosen path."""
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Experiment Folder"
        )
        if folder:
            self.in_folder = folder
            self.in_path.setText(folder)

    def select_output_folder(self) -> None:
        """Prompt for the destination folder and store the chosen path."""
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Folder"
        )
        if folder:
            self.out_folder = folder
            self.out_path.setText(folder)

    def read_naming(self) -> None:
        """Autofill the fields from the first image filename in the folder."""
        if not self.in_folder:
            QtWidgets.QMessageBox.warning(
                self, "No folder", "Please select an Experiment folder first."
            )
            return

        one = find_one_image_file(self.in_folder)
        if not one:
            self._warn("No images", no_images_message(self.in_folder))
            return

        info = parse_name(one)
        exp = parse_experiment_token_from_filename(one)

        if info["date"]:
            self.ed_date.setText(yymmdd_compact(info["date"]) or info["date"])
        if exp["initials"]:
            self.ed_initials.setText(exp["initials"])
        if exp["setup"]:
            self.ed_setup.setText(exp["setup"])

        self.ed_pos.setText(normalize_pos(info["pos"]))
        self.ed_time.setText(normalize_t(info["t"]))
        self.ed_z.setText(normalize_z(info["z"]))
        self.ed_ch.setText(
            normalize_w(info["ch"], one_indexed=channel_is_one_indexed())
        )

        missing = []
        if not self.ed_date.text():
            missing.append("Experiment Date")
        if not self.ed_initials.text():
            missing.append("Initials")
        if not self.ed_setup.text():
            missing.append("Microscope setup")
        if not self.ed_pos.text():
            missing.append("Positions")
        if not self.ed_time.text():
            missing.append("Timepoint")
        if not self.ed_ch.text():
            missing.append("Channel")

        if missing:
            QtWidgets.QMessageBox.information(
                self,
                "Read file naming",
                f"Read from:\n{os.path.basename(one)}\n\n"
                "These fields are not part of the filename and have to be "
                "filled in manually:\n- " + "\n- ".join(missing),
            )

    def run(self) -> None:
        """Validate the fields, then copy every image into the tTt layout."""
        if not self.in_folder or not self.out_folder:
            QtWidgets.QMessageBox.warning(
                self, "Missing", "Select both input and output folders."
            )
            return

        date_compact = yymmdd_compact(self.ed_date.text())
        initials = normalize_initials(self.ed_initials.text())
        setup = normalize_setup(self.ed_setup.text())

        if date_compact:
            self.ed_date.setText(date_compact)
        self.ed_initials.setText(initials)
        self.ed_setup.setText(setup)

        req_missing = []
        if not date_compact:
            req_missing.append(
                "Experiment Date (YYMMDD - also accepts YYYYMMDD or YYYY-MM-DD)"
            )
        if len(initials) < 2:
            req_missing.append("Initials (at least 2 letters, e.g. MA)")
        if not setup:
            req_missing.append("Microscope setup (digits only, e.g. 30)")
        if not (self.ed_pos.text() or "").strip():
            req_missing.append("Positions")
        if not (self.ed_time.text() or "").strip():
            req_missing.append("Timepoint")
        if not (self.ed_ch.text() or "").strip():
            req_missing.append("Channel")

        if req_missing:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing required fields",
                "Please fill these fields before running:\n- "
                + "\n- ".join(req_missing),
            )
            return

        exp_name = f"{date_compact}{initials}{setup}"

        out_root = os.path.join(self.out_folder, exp_name)
        analysis_dir = os.path.join(out_root, "Analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        files = list(iter_image_files(self.in_folder))
        if not files:
            self._warn("No images", no_images_message(self.in_folder))
            return

        parsed = [parse_name(src) for src in files]
        position_map = resolve_positions(
            [info["pos"] or self.ed_pos.text() for info in parsed]
        )

        self.progress.setRange(0, len(files))
        self.progress.setValue(0)

        problems = []
        pos_dirs_made = set()

        for i, (src, info) in enumerate(
            zip(files, parsed, strict=True), start=1
        ):
            pos = position_map[info["pos"] or self.ed_pos.text()]
            t = normalize_t(info["t"] or self.ed_time.text())
            z = normalize_z(info["z"] or self.ed_z.text() or "z001")
            w = normalize_w(
                info["ch"] or self.ed_ch.text(),
                one_indexed=channel_is_one_indexed(),
            )

            # Require at least position + time + channel; z defaults to z001 if empty
            if not pos or not t or not w:
                problems.append(os.path.basename(src))
                self.progress.setValue(i)
                QtWidgets.QApplication.processEvents()
                continue

            pos_folder_name = f"{exp_name}_{pos}"
            dst_dir = os.path.join(out_root, pos_folder_name)
            if dst_dir not in pos_dirs_made:
                os.makedirs(dst_dir, exist_ok=True)
                pos_dirs_made.add(dst_dir)

            ext = os.path.splitext(src)[1].lower()
            dst_name = f"{exp_name}_{pos}_{t}_{z}_{w}{ext}"
            dst_path = os.path.join(dst_dir, dst_name)

            if os.path.exists(dst_path):
                base = os.path.splitext(dst_name)[0]
                n = 2
                while True:
                    alt = os.path.join(dst_dir, f"{base}_dup{n}{ext}")
                    if not os.path.exists(alt):
                        dst_path = alt
                        break
                    n += 1

            shutil.copy2(src, dst_path)

            self.progress.setValue(i)
            QtWidgets.QApplication.processEvents()

        if problems:
            QtWidgets.QMessageBox.warning(
                self,
                "Done (with warnings)",
                "Finished, but some files could not be renamed (missing pos/t/ch).\n\n"
                "Examples:\n- "
                + "\n- ".join(problems[:10])
                + (
                    ""
                    if len(problems) <= 10
                    else f"\n... and {len(problems)-10} more"
                ),
            )
        else:
            QtWidgets.QMessageBox.information(
                self, "Run", f"Done.\nCreated: {out_root}"
            )


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    w = TttDataFormatTransformer()
    w.show()
    app.exec()
