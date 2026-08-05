"""Dialog for building a derived metric from two existing features."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from SECQUOIA.config import LINKS, STYLE, TOOLTIPSTEXT
from SECQUOIA.core.normalization import apply_normalization
from SECQUOIA.core.project_state import save_project_state
from SECQUOIA.core.quantification import safe_op_series
from SECQUOIA.gui.common.ui_utils import (
    add_progress_bar,
    fixed_label,
    make_help_button,
)
from SECQUOIA.gui.outlier.markers import update_outlier_marker
from SECQUOIA.utils.plotting import update_plot

__all__ = ["MetricDialog", "build_column_names", "open_metric_dialog"]

# Operators offered, in display order.
OPERATORS = ("/", "*", "+", "-")
PRETTY_OPS = {"+": "+", "-": "-", "*": "×", "/": "÷"}
_DEFAULT_OP = "/"


@dataclass(frozen=True)
class _DerivedFeatureSpec:
    """One fully resolved derived metric."""

    feature_key: str
    col_new: str
    feat_a: str
    feat_b: str
    op_val: str
    has_ch_l: bool
    has_ch_r: bool
    c1v: object
    m1v: int
    c2v: object
    m2v: int
    col_a: str
    col_b: str
    norm_cfg: dict


_STYLESHEET = f"""
QWidget {{
    font-size: {STYLE.FONT_SIZE_outlier}pt;
    font-family: "{STYLE.FONT_outlier}";
}}
QGroupBox {{
    border: 1pt solid #ddd;
    border-radius: 6pt;
    margin-top: 6pt;   /* smaller gap */
    padding: 8pt;
    background: transparent;
}}
"""


def side_tag(has_ch: bool, channel, mask) -> str:
    """Build the channel/mask suffix for one side of a metric."""
    return f"Ch{channel}M{mask}" if has_ch else f"M{mask}"


def build_column_names(
    feature_defs: dict,
    feat_left: str,
    c1v,
    m1v,
    feat_right: str,
    c2v,
    m2v,
) -> tuple[str | None, str | None, bool, bool]:
    """Resolve the two source column names from the feature templates."""
    tmpl_left = feature_defs.get(feat_left, {}).get("template")
    tmpl_right = feature_defs.get(feat_right, {}).get("template")
    if not tmpl_left or not tmpl_right:
        return None, None, False, False

    has_ch_left = bool(feature_defs.get(feat_left, {}).get("has_ch", False))
    has_ch_right = bool(feature_defs.get(feat_right, {}).get("has_ch", False))

    col_a = (
        tmpl_left.format(ch=c1v, m=m1v)
        if has_ch_left
        else tmpl_left.format(m=m1v)
    )
    col_b = (
        tmpl_right.format(ch=c2v, m=m2v)
        if has_ch_right
        else tmpl_right.format(m=m2v)
    )
    return col_a, col_b, has_ch_left, has_ch_right


class MetricDialog(QDialog):
    """Configure and run a two-feature derived metric for one plot row."""

    def __init__(self, main_window, row: int, parent=None):
        """Build the dialog, restoring the last configuration for this row."""
        super().__init__(parent or main_window)
        self.main_window = main_window
        self.row = row

        self.mask_count = main_window.n_masks
        self.channel_count = main_window.n_channels
        self.feature_defs = self._load_feature_defs()
        self.feature_keys = sorted(
            self.feature_defs, key=lambda k: k.lower().replace("_", " ")
        )
        self.last_cfg = getattr(main_window, "last_run_config", {}).get(
            row, {}
        )
        self.t_ui_min, self.t_ui_max = self._dataset_time_range()

        self.setWindowTitle("Run Metric Calculations")
        self.setStyleSheet(_STYLESHEET)

        self._build_ui()
        self._connect_signals()
        self._apply_initial_state()

        self.adjustSize()

    def _load_feature_defs(self) -> dict:
        """Read the feature definitions, refreshing the plot if they're missing."""
        feature_defs = getattr(self.main_window, "_feature_defs", None)
        if not feature_defs:
            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                update_plot(self.main_window)
            feature_defs = getattr(self.main_window, "_feature_defs", {}) or {}
        return feature_defs

    def _dataset_time_range(self) -> tuple[int, int]:
        """Find the first and last time point present in the data."""
        source_df = getattr(self.main_window, "filtered_df", None)
        if isinstance(source_df, pd.DataFrame) and "t" in source_df.columns:
            t_vals = (
                pd.to_numeric(source_df["t"], errors="coerce")
                .dropna()
                .astype(int)
            )
            return int(t_vals.min()), int(t_vals.max())
        return 0, 0

    def _build_ui(self) -> None:
        """Lay out the metric row, the normalisation box and the buttons."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        help_row = QHBoxLayout()
        help_row.setContentsMargins(0, 0, 0, 0)
        help_row.addStretch(1)
        help_row.addWidget(
            make_help_button(
                self, TOOLTIPSTEXT.RHELP, LINKS.GITHUB_DYNAMICS_PLOT
            )
        )
        outer.addLayout(help_row)

        outer.addWidget(fixed_label("Configure derived metric"))
        outer.addWidget(self._build_derived_box())
        outer.addWidget(fixed_label("Normalization"))
        outer.addWidget(self._build_norm_box())

        self.progress_bar, helpers = add_progress_bar(outer)
        self.set_progress = helpers["set"]
        self.finish_progress = helpers["finish"]

        outer.addLayout(self._build_button_row())

    def _build_derived_box(self) -> QGroupBox:
        """Build the ``featureA <op> featureB`` row."""
        default_a = (
            self.last_cfg.get("feature_a")
            or getattr(self.main_window, "selected_feature_by_row", {}).get(
                self.row
            )
            or (self.feature_keys[0] if self.feature_keys else None)
        )
        default_b = self.last_cfg.get("feature_b", default_a)

        left = self._build_feature_side(
            default_a, TOOLTIPSTEXT.RC1, TOOLTIPSTEXT.RM1
        )
        self.featA, self.c1_label, self.c1, self.m1_label, self.m1 = left[1:]

        self.op_combo = self._build_op_combo()
        op_label = fixed_label("Op:")

        right = self._build_feature_side(
            default_b, TOOLTIPSTEXT.RC2, TOOLTIPSTEXT.RM2
        )
        self.featB, self.c2_label, self.c2, self.m2_label, self.m2 = right[1:]

        row = QHBoxLayout()
        row.setContentsMargins(5, 5, 5, 5)
        row.setSpacing(6)
        for widget in (
            left[0],
            self.featA,
            self.c1_label,
            self.c1,
            self.m1_label,
            self.m1,
            op_label,
            self.op_combo,
            right[0],
            self.featB,
            self.c2_label,
            self.c2,
            self.m2_label,
            self.m2,
        ):
            row.addWidget(widget)

        self.derived_box_layout = QVBoxLayout()
        self.derived_box_layout.setContentsMargins(8, 8, 8, 8)
        self.derived_box_layout.setSpacing(8)
        self.derived_box_layout.addLayout(row)

        box = QGroupBox()
        box.setLayout(self.derived_box_layout)
        return box

    def _build_feature_side(
        self, default_feature, channel_tooltip: str, mask_tooltip: str
    ) -> tuple:
        """Build one side of the metric: feature, channel and mask combos."""
        feat_label = fixed_label("Feature:")
        feat_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        feat_combo = QComboBox()
        feat_combo.setToolTip(TOOLTIPSTEXT.RFEATURE)
        feat_combo.setSizeAdjustPolicy(QComboBox.AdjustToContentsOnFirstShow)
        for key in self.feature_keys:
            feat_combo.addItem(key, key)
        if default_feature in self.feature_keys:
            feat_combo.setCurrentIndex(
                self.feature_keys.index(default_feature)
            )

        c_combo = self._make_combo(
            [c[1:] for c in self.main_window.ids_channels]
        )
        c_combo.setToolTip(channel_tooltip)
        m_combo = self._make_combo([i + 1 for i in range(self.mask_count)])
        m_combo.setToolTip(mask_tooltip)

        return (
            feat_label,
            feat_combo,
            fixed_label("C:"),
            c_combo,
            fixed_label("M:"),
            m_combo,
        )

    def _build_op_combo(self) -> QComboBox:
        """Build the operator combo, restoring the last choice."""
        combo = QComboBox()
        combo.setToolTip(TOOLTIPSTEXT.ROP)
        combo.setMinimumWidth(56)
        combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for op in OPERATORS:
            combo.addItem(op, op)

        op_init = self.last_cfg.get("op", _DEFAULT_OP)
        if op_init in OPERATORS:
            combo.setCurrentText(op_init)
        return combo

    def _build_norm_box(self) -> QGroupBox:
        """Build the normalisation controls."""
        self.norm_enable = QCheckBox("Normalize")
        self.norm_enable.setToolTip(TOOLTIPSTEXT.NORM_ENABLE)
        self.norm_enable.setChecked(
            bool(self.last_cfg.get("norm_enabled", False))
        )

        self.norm_method_label = fixed_label("Method:")
        self.norm_method = self._build_data_combo(
            [("z-score", "zscore"), ("x/baseline", "inv")],
            self.last_cfg.get("norm_method", "zscore"),
            TOOLTIPSTEXT.NORM_METHOD,
        )

        self.norm_scope_label = fixed_label("Scope:")
        self.norm_scope_label.setToolTip(TOOLTIPSTEXT.SCOPE)
        self.norm_scope = self._build_data_combo(
            [("All", "all"), ("IDs", "ids")],
            self.last_cfg.get("norm_scope", "all"),
            TOOLTIPSTEXT.SCOPE,
        )

        self.tp_min_label = fixed_label("Min Time point:")
        self.tp_min_label.setToolTip(TOOLTIPSTEXT.TPMIN)
        self.tp_max_label = fixed_label("Max Time point:")
        self.tp_max_label.setToolTip(TOOLTIPSTEXT.TPMAX)

        self.tp_min_combo = QComboBox()
        self.tp_min_combo.setToolTip(TOOLTIPSTEXT.TPMIN)
        self.tp_min_combo.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        self.tp_max_combo = QComboBox()
        self.tp_max_combo.setToolTip(TOOLTIPSTEXT.TPMAX)

        layout = QHBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        for widget in (
            self.norm_enable,
            self.norm_method_label,
            self.norm_method,
            self.norm_scope_label,
            self.norm_scope,
            self.tp_min_label,
            self.tp_min_combo,
            self.tp_max_label,
            self.tp_max_combo,
        ):
            layout.addWidget(widget)
        layout.addStretch(1)

        box = QGroupBox()
        box.setLayout(layout)
        return box

    def _build_button_row(self) -> QHBoxLayout:
        """Build the Run / Exit row."""
        self.btn_run = QPushButton("Run")
        self.btn_run.setToolTip(TOOLTIPSTEXT.RRUN)
        self.btn_exit = QPushButton("Exit")
        self.btn_exit.setToolTip(TOOLTIPSTEXT.REXIT)

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.btn_run)
        row.addWidget(self.btn_exit)
        return row

    @staticmethod
    def _make_combo(item_list: list) -> QComboBox:
        """Create a combo prefilled with items, disabled when there are none."""
        combo = QComboBox()
        combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        combo.setMinimumContentsLength(3)
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        for item in item_list:
            combo.addItem(str(item), item)
        if item_list:
            combo.setCurrentIndex(0)
        else:
            combo.setEnabled(False)
        return combo

    @staticmethod
    def _build_data_combo(
        entries: list[tuple[str, str]], selected, tooltip: str
    ) -> QComboBox:
        """Create a combo of label/value pairs with one pre-selected."""
        combo = QComboBox()
        combo.setToolTip(tooltip)
        combo.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        for label, value in entries:
            combo.addItem(label, value)

        index = combo.findData(selected) if selected else -1
        if index >= 0:
            combo.setCurrentIndex(index)
        return combo

    def _connect_signals(self) -> None:
        """Wire the controls that change other controls' visibility."""
        self.norm_enable.toggled.connect(self._toggle_norm_children)
        self.norm_method.currentIndexChanged.connect(
            lambda _=None: self._apply_norm_method_visibility()
        )
        self.featA.currentTextChanged.connect(
            lambda _=None: self._apply_channel_visibility()
        )
        self.featB.currentTextChanged.connect(
            lambda _=None: self._apply_channel_visibility()
        )
        self.btn_run.clicked.connect(self._on_run)
        self.btn_exit.clicked.connect(self.reject)

    def _apply_initial_state(self) -> None:
        """Populate the time combos and sync every conditional control."""
        self._fill_time_combos(self.t_ui_min, self.t_ui_max)
        self._apply_norm_method_visibility()
        self._toggle_norm_children(self.norm_enable.isChecked())
        self._apply_channel_visibility()

        if not self.feature_keys:
            self.derived_box_layout.addWidget(
                QLabel("No features detected for this dataset.")
            )
            self.btn_run.setEnabled(False)

    def _fill_time_combos(self, tmin: int, tmax: int) -> None:
        """Fill the baseline-window combos, restoring the saved window."""
        self.tp_min_combo.clear()
        self.tp_max_combo.clear()
        if tmax < tmin:
            tmin, tmax = tmax, tmin

        for t in range(int(tmin), int(tmax) + 1):
            self.tp_min_combo.addItem(str(t), t)
            self.tp_max_combo.addItem(str(t), t)

        i_min = self.tp_min_combo.findData(
            int(self.last_cfg.get("norm_t_min", tmin))
        )
        i_max = self.tp_max_combo.findData(
            int(self.last_cfg.get("norm_t_max", tmax))
        )
        self.tp_min_combo.setCurrentIndex(max(0, i_min))
        self.tp_max_combo.setCurrentIndex(
            self.tp_max_combo.count() - 1 if i_max < 0 else i_max
        )

    def _apply_norm_method_visibility(self) -> None:
        """Show the time window only for methods that use it."""
        is_zscore = self.norm_method.currentData() == "zscore"
        for widget in (
            self.tp_min_label,
            self.tp_min_combo,
            self.tp_max_label,
            self.tp_max_combo,
        ):
            widget.setVisible(not is_zscore)

    def _toggle_norm_children(self, on: bool) -> None:
        """Enable or disable every normalisation control."""
        for widget in (
            self.norm_method_label,
            self.norm_method,
            self.norm_scope_label,
            self.norm_scope,
            self.tp_min_label,
            self.tp_min_combo,
            self.tp_max_label,
            self.tp_max_combo,
        ):
            widget.setEnabled(on)
        self._apply_norm_method_visibility()

    def _apply_channel_visibility(self) -> None:
        """Show each channel combo only when its feature has channels."""
        for feat_combo, c_label, c_combo in (
            (self.featA, self.c1_label, self.c1),
            (self.featB, self.c2_label, self.c2),
        ):
            key = self._feat_key(feat_combo)
            has_ch = bool(self.feature_defs.get(key, {}).get("has_ch", False))
            c_label.setVisible(has_ch)
            c_combo.setVisible(has_ch)
            c_combo.setEnabled(not (has_ch and self.channel_count < 1))

    @staticmethod
    def _feat_key(combo: QComboBox):
        """Read the selected feature key from a combo."""
        return combo.currentData() or combo.currentText() or None

    def _step(self, pct: int) -> None:
        """Move the progress bar and let the UI repaint."""
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            self.set_progress(pct)
            QApplication.processEvents()

    def _finish(self) -> None:
        """Complete the progress bar and let the UI repaint."""
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            self.finish_progress()
            QApplication.processEvents()

    def _read_norm_config(self) -> dict:
        """Collect the normalisation settings."""
        return {
            "norm_enabled": bool(self.norm_enable.isChecked()),
            "norm_method": self.norm_method.currentData() or "zscore",
            "norm_scope": self.norm_scope.currentData() or "all",
            "norm_t_min": int(
                self.tp_min_combo.currentData() or self.t_ui_min
            ),
            "norm_t_max": int(
                self.tp_max_combo.currentData() or self.t_ui_max
            ),
        }

    def _on_run(self) -> None:
        """Compute the derived metric and register it on the main window."""
        self._step(0)

        feat_a = self._feat_key(self.featA)
        feat_b = self._feat_key(self.featB)
        op_val = (
            self.op_combo.currentData()
            or self.op_combo.currentText()
            or _DEFAULT_OP
        )
        c1v = self.c1.currentData()
        m1v = int(self.m1.currentData())
        c2v = self.c2.currentData()
        m2v = int(self.m2.currentData())
        norm_cfg = self._read_norm_config()

        self._step(15)

        col_a, col_b, has_ch_l, has_ch_r = build_column_names(
            self.feature_defs, feat_a, c1v, m1v, feat_b, c2v, m2v
        )
        if not col_a or not col_b:
            self._finish()
            self.reject()
            return

        left_tag = side_tag(has_ch_l, c1v, m1v)
        right_tag = side_tag(has_ch_r, c2v, m2v)
        feature_key = (
            f"{feat_a}{left_tag} {PRETTY_OPS[op_val]} {feat_b}{right_tag}"
        )
        col_new = f"{feat_a}{left_tag}{op_val}{feat_b}{right_tag}"

        self._step(35)
        df_f = self._compute_into("filtered_df", col_a, col_b, col_new, op_val)

        self._step(60)
        df_t = self._compute_into("track_df", col_a, col_b, col_new, op_val)

        if norm_cfg["norm_enabled"]:
            for df in (df_f, df_t):
                if isinstance(df, pd.DataFrame) and col_new in df.columns:
                    apply_normalization(
                        df,
                        col_new,
                        norm_cfg["norm_method"],
                        norm_cfg["norm_scope"],
                        int(norm_cfg["norm_t_min"]),
                        int(norm_cfg["norm_t_max"]),
                    )

        self._step(75)

        spec = _DerivedFeatureSpec(
            feature_key=feature_key,
            col_new=col_new,
            feat_a=feat_a,
            feat_b=feat_b,
            op_val=op_val,
            has_ch_l=has_ch_l,
            has_ch_r=has_ch_r,
            c1v=c1v,
            m1v=m1v,
            c2v=c2v,
            m2v=m2v,
            col_a=col_a,
            col_b=col_b,
            norm_cfg=norm_cfg,
        )
        self._register_derived_feature(spec)
        self._remember_row_config(spec)

        update_plot(self.main_window)
        save_project_state(self.main_window)
        self._select_in_row_combo(feature_key)
        update_outlier_marker(self.main_window)

        if hasattr(self.main_window, "fit_plot_row"):
            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                self.main_window.fit_plot_row(self.row)

        self._step(100)
        self._finish()
        self.accept()

    def _compute_into(
        self, attr: str, col_a: str, col_b: str, col_new: str, op_val: str
    ):
        """Add the derived column to one of the main window's dataframes."""
        df = getattr(self.main_window, attr, None)
        if (
            isinstance(df, pd.DataFrame)
            and col_a in df.columns
            and col_b in df.columns
        ):
            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                df[col_new] = safe_op_series(df[col_a], df[col_b], op_val)
        return df

    def _register_derived_feature(self, spec: _DerivedFeatureSpec) -> None:
        """Add the new metric to ``main_window._derived_features``."""
        if not hasattr(self.main_window, "_derived_features"):
            self.main_window._derived_features = {}

        self.main_window._derived_features[spec.feature_key] = {
            "template": spec.col_new,
            "has_ch": False,
            "has_m": False,
            "display": spec.feature_key,
            "formula": {
                "base_feature_left": str(spec.feat_a),
                "base_feature_right": str(spec.feat_b),
                "op": str(spec.op_val),
                "has_ch_left": bool(spec.has_ch_l),
                "has_ch_right": bool(spec.has_ch_r),
                "c1": spec.c1v,
                "m1": int(spec.m1v),
                "c2": spec.c2v,
                "m2": int(spec.m2v),
                "source_col_a": str(spec.col_a),
                "source_col_b": str(spec.col_b),
                "target_col": str(spec.col_new),
                **spec.norm_cfg,
            },
        }

    def _remember_row_config(self, spec: _DerivedFeatureSpec) -> None:
        """Store this row's choices so the dialog reopens prefilled."""
        self.main_window.last_run_config = getattr(
            self.main_window, "last_run_config", {}
        )
        self.main_window.last_run_config[self.row] = {
            "feature_a": spec.feat_a,
            "feature_b": spec.feat_b,
            "op": spec.op_val,
            "has_ch_left": spec.has_ch_l,
            "has_ch_right": spec.has_ch_r,
            "c1": spec.c1v,
            "m1": int(spec.m1v),
            "c2": spec.c2v,
            "m2": int(spec.m2v),
            "feature_key": spec.feature_key,
            "col_name": spec.col_new,
            **spec.norm_cfg,
        }

        self.main_window.selected_feature_by_row = getattr(
            self.main_window, "selected_feature_by_row", {}
        )
        self.main_window.selected_feature_by_row[self.row] = spec.feature_key

    def _select_in_row_combo(self, feature_key: str) -> None:
        """Select the new metric in the plot row's feature combo."""
        tools = getattr(self.main_window, "row_tools", {}).get(self.row, {})
        feat_cb = tools.get("feat")
        if feat_cb is None:
            return

        index = feat_cb.findData(feature_key)
        if index < 0:
            index = feat_cb.findText(feature_key)
        if index >= 0:
            feat_cb.setCurrentIndex(index)


def open_metric_dialog(main_window, row: int) -> None:
    """Open the metric-configuration dialog for one plot row."""
    MetricDialog(main_window, row).exec_()
