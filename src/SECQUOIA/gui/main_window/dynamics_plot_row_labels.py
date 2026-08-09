"""Row summary/ID-time label text for dynamics plot rows, and derived metric deletion."""

import re

import pandas as pd
from qtpy.QtCore import QPoint, Qt
from qtpy.QtWidgets import QComboBox, QLabel, QMenu, QMessageBox, QToolButton

from SECQUOIA.config import TOOLTIPSTEXT, ColorStyle
from SECQUOIA.core.project_state import save_project_state, save_track_df
from SECQUOIA.utils.plotting import update_plot


class DynamicsPlotRowLabels:
    """Build row summary/ID-time labels, and delete derived metrics from row state."""

    def _build_row_igt_text(self, *, font_size: int | None = None) -> str:
        """Build the HTML ID/time label text for a plot row."""
        unique_ids = getattr(self, "unique_ids", None)
        ids = [] if unique_ids is None else list(unique_ids)

        ci = int(getattr(self, "current_ident_index", -1))
        ident_raw = str(
            getattr(
                self,
                "ident",
                (
                    str(ids[ci])
                    if 0 <= ci < len(ids)
                    else (str(ids[0]) if ids else "")
                ),
            )
        )

        digits = re.findall(r"\d+", ident_raw)
        id_suffix = digits[-1][-3:].zfill(3) if digits else "—"

        g = getattr(self, "current_TrackNumber_plot", None)
        track3 = f"{int(g):03d}" if isinstance(g, (int | float)) else "—"

        t = getattr(self, "current_time_index", None)
        t_txt = f"{int(t)}" if isinstance(t, (int | float)) else "—"

        id_display = f"{id_suffix}-Cell-{track3}"

        style = (
            f' style="color:white;'
            f"font-family:{ColorStyle.FONT_FAMILY};"
            f"font-size:{font_size or ColorStyle.FONT_SIZE}px;"
            f'font-weight:{ColorStyle.FONT_WEIGHT};"'
        )

        return (
            f"<span{style}>ID:</span> <span{style}>{id_display}</span>"
            f"&nbsp;&nbsp;&nbsp;<span{style}>T:</span> <span{style}>{t_txt}</span>"
        )

    def _content_tooltip(self, content_html: str, description: str) -> str:
        """Wrap a row label's own enlarged content above its description."""
        if not content_html:
            return description

        return (
            f"<div>{content_html}<br>"
            f'<span style="color:{self._SUMMARY_COLORS["dim"]};'
            f"font-family:{ColorStyle.FONT_FAMILY};"
            f'font-size:{ColorStyle.FONT_SIZE}px;">{description}</span></div>'
        )

    def _update_row_igt_label(self, row: int) -> None:
        """Update the ID/time label for a single plot row."""
        tools = getattr(self, "row_tools", {}).get(row, {})
        lbl = tools.get("igt")
        if isinstance(lbl, QLabel):
            html = self._build_row_igt_text()
            lbl.setTextFormat(Qt.RichText)
            lbl.setText(html)
            lbl.setToolTip(
                self._content_tooltip(
                    self._build_row_igt_text(
                        font_size=ColorStyle.TOOLTIP_FONT_SIZE
                    ),
                    TOOLTIPSTEXT.ID,
                )
            )

    def _refresh_all_row_igt(self) -> None:
        """Refresh ID/time labels for all visible plot rows."""
        for row in range(1, int(getattr(self, "_max_plot_rows", 0)) + 1):
            self._update_row_igt_label(row)

    def _summary_span(
        self,
        txt: str,
        color: str,
        weight: int | None = None,
        *,
        font_size: int | None = None,
    ) -> str:
        """Return an HTML span using the configured summary label font styling."""
        weight_css = (
            f"font-weight:{weight};"
            if weight is not None
            else f"font-weight:{ColorStyle.FONT_WEIGHT};"
        )
        return (
            f'<span style="color:{color};'
            f"font-family:{ColorStyle.FONT_FAMILY};"
            f"font-size:{font_size or ColorStyle.FONT_SIZE}px;"
            f'{weight_css}">{txt}</span>'
        )

    def _summary_term_html(
        self,
        m_val,
        channel,
        has_ch: bool,
        *,
        font_size: int | None = None,
    ) -> str:
        """Return formatted HTML for a mask/channel term in a row summary."""
        c = self._SUMMARY_COLORS
        fs = font_size
        inner: list[str] = []
        if isinstance(m_val, int):
            inner.append(
                self._summary_span(f"M{m_val:02d}", c["m"], font_size=fs)
            )
        if has_ch and isinstance(channel, str):
            if inner:
                inner.append(self._summary_span("_", c["br"], font_size=fs))
            inner.append(
                self._summary_span(f"CH{channel}", c["ch"], font_size=fs)
            )
        core = (
            "".join(inner)
            if inner
            else self._summary_span("—", c["br"], font_size=fs)
        )
        return (
            self._summary_span("[", c["br"], font_size=fs)
            + core
            + self._summary_span("]", c["br"], font_size=fs)
        )

    @staticmethod
    def _to_int_or_none(x) -> int | None:
        """Convert x to int, returning None if it is None or not convertible."""
        try:
            return int(x) if x is not None else None
        except (TypeError, ValueError):
            return None

    def _build_formula_row_summary(
        self,
        feat_key: str,
        fd: dict,
        formula: dict,
        *,
        font_size: int | None = None,
    ) -> str:
        """Build the colorized summary text for a derived (formula-based) feature."""
        c = self._SUMMARY_COLORS
        op_map = {"/": "÷", "*": "×", "+": "+", "-": "−"}
        op_raw = str(formula.get("op", "/"))
        op_disp = op_map.get(op_raw, op_raw)

        left_name = (
            str(formula.get("base_feature_left") or "").strip()
            or str(formula.get("base_feature") or "").strip()
        )
        right_name = str(formula.get("base_feature_right") or "").strip()

        if not left_name:
            left_name = str(fd.get("display", feat_key)).split(" ")[0]

        has_ch_L = bool(
            formula.get("has_ch_left", formula.get("has_ch", False))
        )
        has_ch_R = bool(
            formula.get("has_ch_right", formula.get("has_ch", False))
        )

        m1 = self._to_int_or_none(formula.get("m1"))
        c1 = formula.get("c1")
        m2 = self._to_int_or_none(formula.get("m2"))
        c2 = formula.get("c2")

        parts: list[str] = []
        if left_name:
            parts.append(
                self._summary_span(
                    left_name, c["feature"], weight=600, font_size=font_size
                )
            )
            parts.append(" ")
        parts.append(
            self._summary_term_html(m1, c1, has_ch_L, font_size=font_size)
        )

        # Operator
        parts += [
            " ",
            self._summary_span(
                op_disp, c["op"], weight=700, font_size=font_size
            ),
            " ",
        ]

        if right_name:
            parts.append(
                self._summary_span(
                    right_name, c["feature"], weight=600, font_size=font_size
                )
            )
            parts.append(" ")
        parts.append(
            self._summary_term_html(m2, c2, has_ch_R, font_size=font_size)
        )

        return "".join(parts)

    def _build_simple_row_summary(
        self,
        row: int,
        feat_key: str,
        fd: dict,
        *,
        font_size: int | None = None,
    ) -> str:
        """Build the colorized summary text for a plain (non-derived) feature."""
        c = self._SUMMARY_COLORS
        parts: list[str] = [
            self._summary_span(
                fd.get("display", feat_key),
                c["feature"],
                weight=600,
                font_size=font_size,
            )
        ]

        if bool(fd.get("has_m", True)):
            m_val = (getattr(self, "selected_m_by_channel", {}) or {}).get(row)
            if isinstance(m_val, int):
                parts.append(
                    self._summary_span(
                        f"_M{m_val:02d}", c["m"], font_size=font_size
                    )
                )

        if bool(fd.get("has_ch", False)):
            channel = (getattr(self, "selected_ch_by_channel", {}) or {}).get(
                row
            )
            if isinstance(channel, str):
                parts.append(
                    self._summary_span(
                        f"_CH{channel}", c["ch"], font_size=font_size
                    )
                )

        return "".join(parts)

    def _build_row_summary_text(
        self, row: int, *, font_size: int | None = None
    ) -> str:
        """Build a compact, colorized summary for the row's current selection."""
        feature_defs = getattr(self, "_feature_defs", {}) or {}
        feat_key = (getattr(self, "selected_feature_by_row", {}) or {}).get(
            row
        )
        if not feat_key or feat_key not in feature_defs:
            return ""

        fd = feature_defs[feat_key]
        derived = (getattr(self, "_derived_features", {}) or {}).get(feat_key)
        formula = (derived or {}).get("formula")

        if formula:
            return self._build_formula_row_summary(
                feat_key, fd, formula, font_size=font_size
            )
        return self._build_simple_row_summary(
            row, feat_key, fd, font_size=font_size
        )

    def _update_row_summary_label(self, row: int) -> None:
        """Update the compact feature/mask/channel summary label for a single plot row."""
        tools = getattr(self, "row_tools", {}).get(row, {})
        lbl = tools.get("summary")
        if not isinstance(lbl, QLabel):
            return
        html = self._build_row_summary_text(row)
        lbl.setTextFormat(Qt.RichText)
        lbl.setText(html)
        lbl.setToolTip(
            self._content_tooltip(
                self._build_row_summary_text(
                    row, font_size=ColorStyle.TOOLTIP_FONT_SIZE
                ),
                TOOLTIPSTEXT.SELECTION,
            )
        )
        eye = tools.get("eye")
        if isinstance(eye, QToolButton) and not eye.isChecked():
            lbl.setStyleSheet(
                f"QLabel {{ color: {self._SUMMARY_COLORS['dim']}; }}"
            )
        else:
            lbl.setStyleSheet("")

    def _refresh_all_row_summaries(self) -> None:
        """Refresh compact feature/mask/channel summary labels for all plot rows."""
        for row in range(1, int(getattr(self, "_max_plot_rows", 0)) + 1):
            self._update_row_summary_label(row)

    def _feature_combo_context_menu(
        self, row: int, cb: QComboBox, pos: QPoint
    ) -> None:
        """Show a right-click menu for deleting derived metrics from a feature dropdown."""
        feature_key = cb.currentData() or cb.currentText()
        registry = getattr(self, "_derived_features", {}) or {}

        menu = QMenu(cb)
        delete_action = menu.addAction("Delete selected derived metric")
        delete_action.setEnabled(feature_key in registry)

        action = menu.exec_(cb.mapToGlobal(pos))
        if action == delete_action:
            self._delete_derived_metric(feature_key)

    def _confirm_delete_derived_metric(self, feature_key: str) -> bool:
        """Ask the user to confirm deleting a derived metric; return True to proceed."""
        answer = QMessageBox.question(
            self,
            "Delete metric",
            f"Delete this derived metric?\n\n{feature_key}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def _drop_derived_metric_dataframe_columns(
        self, col_name: str | None
    ) -> None:
        """Drop a derived metric's columns from every in-memory dataframe that has them."""
        for df_name in ("track_df", "filtered_df", "df_subset"):
            df = getattr(self, df_name, None)
            if isinstance(df, pd.DataFrame):
                cols = [
                    c
                    for c in (col_name, f"{col_name}__raw")
                    if c and c in df.columns
                ]
                if cols:
                    setattr(
                        self,
                        df_name,
                        df.drop(columns=cols, errors="ignore").copy(),
                    )

    def _purge_derived_metric_from_run_config(
        self, feature_key: str, col_name: str | None
    ) -> None:
        """Remove any saved metric-run config entries referencing a derived metric."""
        self.last_run_config = getattr(self, "last_run_config", {}) or {}
        for r, cfg in list(self.last_run_config.items()):
            if isinstance(cfg, dict) and (
                cfg.get("feature_key") == feature_key
                or cfg.get("col_name") == col_name
            ):
                self.last_run_config.pop(r, None)

    def _purge_derived_metric_from_selection(self, feature_key: str) -> None:
        """Clear any row's remembered feature selection that points at a derived metric."""
        self.selected_feature_by_row = (
            getattr(self, "selected_feature_by_row", {}) or {}
        )
        for r, selected in list(self.selected_feature_by_row.items()):
            if selected == feature_key:
                self.selected_feature_by_row.pop(r, None)

    def _remove_derived_metric_from_dropdowns(self, feature_key: str) -> None:
        """Remove a derived metric's entry from every visible feature dropdown."""
        for r, tools in (getattr(self, "row_tools", {}) or {}).items():
            cb = tools.get("feat")
            if cb is None:
                continue

            old_block = cb.blockSignals(True)
            try:
                for i in range(cb.count() - 1, -1, -1):
                    if (
                        cb.itemData(i) == feature_key
                        or cb.itemText(i) == feature_key
                    ):
                        cb.removeItem(i)

                if cb.count() > 0:
                    self.selected_feature_by_row[r] = (
                        cb.currentData() or cb.currentText()
                    )
                else:
                    self.selected_feature_by_row.pop(r, None)
            finally:
                cb.blockSignals(old_block)

    def _purge_derived_metric(self, feature_key: str) -> bool:
        """Erase one derived metric from every place it is remembered."""
        registry = getattr(self, "_derived_features", None) or {}
        if feature_key not in registry:
            return False

        desc = registry.pop(feature_key, {}) or {}
        formula = desc.get("formula", {}) if isinstance(desc, dict) else {}
        col_name = formula.get("target_col") or desc.get("template")

        if hasattr(self, "_feature_defs"):
            self._feature_defs.pop(feature_key, None)

        self._drop_derived_metric_dataframe_columns(col_name)
        self._purge_derived_metric_from_run_config(feature_key, col_name)
        self._purge_derived_metric_from_selection(feature_key)
        self._remove_derived_metric_from_dropdowns(feature_key)
        return True

    def delete_all_derived_metrics(self) -> list[str]:
        """Delete every derived metric, returning the names that were removed."""
        registry = getattr(self, "_derived_features", None) or {}
        deleted = [
            key for key in list(registry) if self._purge_derived_metric(key)
        ]
        return sorted(deleted)

    def _delete_derived_metric(self, feature_key: str) -> None:
        """Delete a derived metric from dataframes, dropdowns, plot state, and saved metadata."""
        registry = getattr(self, "_derived_features", {}) or {}
        if feature_key not in registry:
            QMessageBox.information(
                self,
                "Delete metric",
                "Only derived metrics can be deleted.",
            )
            return

        if not self._confirm_delete_derived_metric(feature_key):
            return

        self._purge_derived_metric(feature_key)

        update_plot(self)
        self._refresh_all_row_summaries()
        self._refresh_all_row_igt()
        save_track_df(self)
        save_project_state(self)

        QMessageBox.information(
            self,
            "Metric deleted",
            f"Deleted:\n\n{feature_key}",
        )
