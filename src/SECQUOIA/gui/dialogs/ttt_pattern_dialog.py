"""Dialog for editing  and adding new  tTt transformer's filename-parsing patterns."""

from __future__ import annotations

import re

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from SECQUOIA.config import LINKS, TOOLTIPSTEXT
from SECQUOIA.core.ttt_naming import (
    normalize_pos,
    normalize_t,
    normalize_w,
    normalize_z,
    yyyymmdd_from_prefix,
)
from SECQUOIA.core.ttt_patterns import (
    FALLBACK_PRESET_NAME,
    FIELDS,
    REQUIRED_GROUPS,
    PatternSet,
    PresetStore,
    build_pattern_from_example,
    compile_field,
    is_builtin_preset,
    load_presets,
    preset_sort_key,
    save_presets,
)
from SECQUOIA.gui.common.ui_utils import (
    ContentWidthScrollArea,
    fit_to_screen,
    make_help_button,
)

_SIMPLE_FIELDS = tuple(f for f in FIELDS if f != "exp_token")

_FIELD_LABELS = {
    "date": "Date",
    "time": "Time point",
    "pos": "Position",
    "z": "Z-slice",
    "ch": "Channel",
    "exp_token": "Date + initials + setup (advanced)",
}
_FIELD_EXAMPLES = {
    "date": {"before": "", "value": "240323", "after": "_"},
    "time": {"before": "Time", "value": "001", "after": ""},
    "pos": {"before": "Well_", "value": "A1", "after": ""},
    "z": {"before": "z", "value": "1", "after": ""},
    "ch": {"before": "Ch", "value": "2", "after": ""},
}
_EXP_TOKEN_HINT = "e.g. 240323MA35 -> groups 'date', 'initials', 'setup'"
_POS_NUMERIC_RE = re.compile(r"p\d{4}")

_PLAIN_LABEL_STYLE = "background: transparent; color: white;"
_MUTED_LABEL_STYLE = "background: transparent; color: #9e9e9e;"


def _plain_label(text: str) -> QLabel:
    """A QLabel with no background box and plain white text."""
    label = QLabel(text)
    label.setStyleSheet(_PLAIN_LABEL_STYLE)
    return label


class TttPatternDialog(QDialog):
    """Build, by example, the six filename-parsing patterns."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("tTt Parsing Pattern Settings")
        self.setSizeGripEnabled(True)
        self.setStyleSheet("QLabel { background: transparent; color: white; }")

        self._store: PresetStore = load_presets()
        current = self._store.active_patterns()
        self._pattern_edits: dict[str, QLineEdit] = {}
        self._status: dict[str, QLabel] = {}
        self._before: dict[str, QLineEdit] = {}
        self._value: dict[str, QLineEdit] = {}
        self._after: dict[str, QLineEdit] = {}
        self._example_preview: dict[str, QLabel] = {}

        main_v = QVBoxLayout(self)
        main_v.addLayout(self._build_header())
        main_v.addLayout(self._build_preset_row())

        fields_content = QWidget()
        fields_v = QVBoxLayout(fields_content)
        fields_v.setContentsMargins(0, 0, 0, 0)
        fields_v.setSpacing(8)
        for field in _SIMPLE_FIELDS:
            fields_v.addWidget(self._build_simple_field_box(field, current))
        fields_v.addWidget(self._build_exp_token_box(current))
        fields_v.addStretch(1)

        scroll = ContentWidthScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(fields_content)
        main_v.addWidget(scroll, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.RestoreDefaults
            | QDialogButtonBox.Save
            | QDialogButtonBox.Cancel
        )
        buttons.button(QDialogButtonBox.Save).setToolTip(
            TOOLTIPSTEXT.SAVE_BTN_tTt_PATTERNS
        )
        buttons.button(QDialogButtonBox.Cancel).setToolTip(
            TOOLTIPSTEXT.CANCEL_BTN_tTt_PATTERNS
        )
        buttons.button(QDialogButtonBox.RestoreDefaults).setToolTip(
            TOOLTIPSTEXT.RESTORE_DEFAULTS_BTN_tTt_PATTERNS
        )
        buttons.button(QDialogButtonBox.Save).clicked.connect(self._on_save)
        buttons.button(QDialogButtonBox.Cancel).clicked.connect(self.reject)
        buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(
            self._restore_defaults
        )
        main_v.addWidget(buttons)

        for field in FIELDS:
            self._validate_field(field)

        fit_to_screen(self, max_frac=0.85, min_size=(560, 320))

    def _build_header(self) -> QHBoxLayout:
        header = QHBoxLayout()
        header.addStretch(1)
        help_btn = make_help_button(
            self, TOOLTIPSTEXT.HELP_BTN_tTt_PATTERNS, LINKS.GITHUB_tTt_PATTERNS
        )
        header.addWidget(help_btn, 0, Qt.AlignRight | Qt.AlignTop)
        return header

    def _build_preset_row(self) -> QHBoxLayout:
        """A dropdown of saved parsing patterns."""
        row = QHBoxLayout()
        row.addWidget(_plain_label("Parsing pattern:"))

        self._preset_combo = QComboBox()
        self._preset_combo.setEditable(True)
        self._preset_combo.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        self._preset_combo.setToolTip(TOOLTIPSTEXT.PRESET_COMBO_tTt_PATTERNS)
        self._refresh_preset_combo()
        self._preset_combo.activated.connect(self._on_preset_selected)
        self._preset_combo.currentTextChanged.connect(
            self._update_delete_button_state
        )
        row.addWidget(self._preset_combo, 1)

        self._delete_preset_btn = QPushButton("Delete")
        self._delete_preset_btn.setToolTip(
            TOOLTIPSTEXT.DELETE_PRESET_tTt_PATTERNS
        )
        self._delete_preset_btn.clicked.connect(self._on_delete_preset)
        row.addWidget(self._delete_preset_btn)
        self._update_delete_button_state()
        return row

    def _refresh_preset_combo(self) -> None:
        names = sorted(self._store.presets, key=preset_sort_key)
        self._preset_combo.blockSignals(True)
        self._preset_combo.clear()
        self._preset_combo.addItems(names)
        self._preset_combo.setCurrentText(self._store.active)
        self._preset_combo.blockSignals(False)

    def _update_delete_button_state(self) -> None:
        name = self._preset_combo.currentText()
        self._delete_preset_btn.setEnabled(
            name in self._store.presets and not is_builtin_preset(name)
        )

    def _on_preset_selected(self) -> None:
        name = self._preset_combo.currentText()
        patterns = self._store.presets.get(name)
        if patterns is not None:
            self._load_preset_into_fields(patterns)

    def _load_preset_into_fields(self, patterns: PatternSet) -> None:
        """Populate every field editor from ``patterns`` (not yet saved)."""
        for field in _SIMPLE_FIELDS:
            self._value[field].clear()
            self._before[field].clear()
            self._after[field].clear()
        for field in FIELDS:
            self._pattern_edits[field].setText(getattr(patterns, field))
        self._ch_one_indexed.setChecked(patterns.ch_one_indexed)

    def _on_delete_preset(self) -> None:
        name = self._preset_combo.currentText()
        if is_builtin_preset(name) or name not in self._store.presets:
            return
        presets = dict(self._store.presets)
        del presets[name]
        active = (
            FALLBACK_PRESET_NAME
            if self._store.active == name
            else self._store.active
        )
        self._store = PresetStore(active=active, presets=presets)
        save_presets(self._store)
        self._refresh_preset_combo()
        self._load_preset_into_fields(self._store.active_patterns())
        for field in FIELDS:
            self._validate_field(field)

    def _build_simple_field_box(
        self, field: str, current: PatternSet
    ) -> QGroupBox:
        """A "before / example value / after" builder for one simple field."""
        box = QGroupBox(_FIELD_LABELS[field])
        grid = QGridLayout(box)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        grid.setColumnStretch(5, 1)

        example = _FIELD_EXAMPLES[field]
        before = QLineEdit()
        before.setPlaceholderText(example["before"] or "(nothing)")
        before.setToolTip(TOOLTIPSTEXT.BEFORE_tTt_PATTERNS)
        value = QLineEdit()
        value.setPlaceholderText(example["value"])
        value.setToolTip(TOOLTIPSTEXT.VALUE_tTt_PATTERNS)
        after = QLineEdit()
        after.setPlaceholderText(example["after"] or "(nothing)")
        after.setToolTip(TOOLTIPSTEXT.AFTER_tTt_PATTERNS)

        grid.addWidget(_plain_label("Fixed text before:"), 0, 0)
        grid.addWidget(before, 0, 1)
        grid.addWidget(_plain_label("Example value:"), 0, 2)
        grid.addWidget(value, 0, 3)
        grid.addWidget(_plain_label("Fixed text after:"), 0, 4)
        grid.addWidget(after, 0, 5)
        self._before[field] = before
        self._value[field] = value
        self._after[field] = after

        pattern_edit = QLineEdit(getattr(current, field))
        pattern_edit.setToolTip(TOOLTIPSTEXT.PATTERN_FIELD_tTt_PATTERNS)
        pattern_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        status = QLabel()
        pattern_edit.textChanged.connect(
            lambda _t, f=field: self._validate_field(f)
        )
        self._pattern_edits[field] = pattern_edit
        self._status[field] = status

        grid.addWidget(_plain_label("Pattern:"), 1, 0)
        grid.addWidget(pattern_edit, 1, 1, 1, 4)
        grid.addWidget(status, 1, 5)

        next_row = 2
        if field == "ch":
            self._ch_one_indexed = QCheckBox(
                "First channel in my filenames is numbered 1, not 0 "
                "(e.g. 'Ch1' is the first channel)"
            )
            self._ch_one_indexed.setToolTip(
                TOOLTIPSTEXT.CH_ONE_INDEXED_tTt_PATTERNS
            )
            self._ch_one_indexed.setChecked(current.ch_one_indexed)
            self._ch_one_indexed.stateChanged.connect(
                lambda: self._update_example_preview("ch")
            )
            grid.addWidget(self._ch_one_indexed, next_row, 0, 1, 6)
            next_row += 1

        preview = _plain_label("")
        preview.setToolTip(TOOLTIPSTEXT.PREVIEW_tTt_PATTERNS)
        grid.addWidget(preview, next_row, 0, 1, 6)
        self._example_preview[field] = preview

        for edit in (before, value, after):
            edit.textChanged.connect(
                lambda _t, f=field, b=before, v=value, a=after: (
                    self._regenerate(f, b, v, a)
                )
            )
        return box

    def _build_exp_token_box(self, current: PatternSet) -> QGroupBox:
        """The compound 'date+initials+setup' field: raw regex, unchanged."""
        field = "exp_token"
        box = QGroupBox(_FIELD_LABELS[field])
        grid = QGridLayout(box)
        grid.setColumnStretch(1, 1)

        edit = QLineEdit(getattr(current, field))
        edit.setToolTip(TOOLTIPSTEXT.EXP_TOKEN_FIELD_tTt_PATTERNS)
        edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        status = QLabel()
        edit.textChanged.connect(lambda _t, f=field: self._validate_field(f))
        self._pattern_edits[field] = edit
        self._status[field] = status

        grid.addWidget(_plain_label("Pattern:"), 0, 0)
        grid.addWidget(edit, 0, 1)
        grid.addWidget(status, 0, 2)

        hint = QLabel(_EXP_TOKEN_HINT)
        hint.setStyleSheet(_MUTED_LABEL_STYLE)
        grid.addWidget(hint, 1, 0, 1, 3)
        return box

    def _regenerate(
        self, field: str, before: QLineEdit, value: QLineEdit, after: QLineEdit
    ) -> None:
        """Rebuild ``field``'s pattern from the before/value/after boxes."""
        if value.text().strip():
            group = REQUIRED_GROUPS[field][0]
            generated = build_pattern_from_example(
                group, before.text(), value.text(), after.text()
            )
            self._pattern_edits[field].setText(generated)
        self._update_example_preview(field)

    def _validate_field(self, field: str) -> None:
        _, error = compile_field(field, self._pattern_edits[field].text())
        status = self._status[field]
        if error:
            status.setText("invalid")
            status.setStyleSheet("background: transparent; color: #e57373;")
            status.setToolTip(error)
        else:
            status.setText("ok")
            status.setStyleSheet("background: transparent; color: #81c784;")
            status.setToolTip(TOOLTIPSTEXT.STATUS_OK_tTt_PATTERNS)
        if field in _SIMPLE_FIELDS:
            self._update_example_preview(field)

    def _normalized_preview(self, field: str, raw: str) -> str:
        """Return what ``raw`` would become in the tTt destination name."""
        if field == "date":
            return yyyymmdd_from_prefix(raw) or raw
        if field == "time":
            return normalize_t(raw)
        if field == "pos":
            return normalize_pos(raw)
        if field == "z":
            return normalize_z(raw)
        if field == "ch":
            return normalize_w(
                raw, one_indexed=self._ch_one_indexed.isChecked()
            )
        return raw

    def _update_example_preview(self, field: str) -> None:
        """Show what the current before/value/after example resolves to."""
        preview = self._example_preview[field]
        example_text = (
            self._before[field].text()
            + self._value[field].text()
            + self._after[field].text()
        )
        if not self._value[field].text().strip():
            preview.setText("")
            return

        rx, _ = compile_field(field, self._pattern_edits[field].text())
        m = rx.search(example_text)
        if not m:
            preview.setText(f"Example: {example_text!r} -> no match")
            return

        raw = m.group(REQUIRED_GROUPS[field][0])
        normalized = self._normalized_preview(field, raw)
        line = f"Example: {example_text!r} -> {normalized!r}"
        if field == "pos" and not _POS_NUMERIC_RE.fullmatch(normalized):
            line += " (assigned its own p0001, p0002, ... when you Run)"
        preview.setText(line)

    def _restore_defaults(self) -> None:
        """Reset the visible fields to the built-in defaults."""
        self._load_preset_into_fields(PatternSet())
        for field in FIELDS:
            self._validate_field(field)

    def _current_field_values(self) -> PatternSet:
        values = {f: self._pattern_edits[f].text() for f in FIELDS}
        values["ch_one_indexed"] = self._ch_one_indexed.isChecked()
        return PatternSet(**values)

    def _on_save(self) -> None:
        """Save the current field values under the selected pattern name."""
        name = self._preset_combo.currentText().strip()
        if not name:
            QMessageBox.warning(
                self,
                "Name needed",
                "Give this parsing pattern a name before saving.",
            )
            return

        presets = dict(self._store.presets)
        presets[name] = self._current_field_values()
        self._store = PresetStore(active=name, presets=presets)
        save_presets(self._store)
        self.accept()
