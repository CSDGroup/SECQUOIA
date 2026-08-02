"""Track/centroid resolution and the Generations (lineage) dialog."""

import logging
from contextlib import suppress

import pandas as pd
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)

from SECQUOIA.config import TOOLTIPSTEXT

LOG = logging.getLogger(__name__)


class _Lineage:
    """Track/centroid resolution and the Generations (lineage) dialog."""

    def _track_number_column(self, main_window):
        """Detect the track number column name in the DataFrame."""
        df = getattr(main_window, "track_df", None)
        if df is None:
            return None
        for cand in (
            "TrackNumber",
            "track_number",
            "track",
            "track_id",
            "TrackID",
        ):
            if cand in df.columns:
                return cand
        return None

    def _available_track_numbers_for_ident(self, main_window, ident):
        """List available track numbers for a given Identification."""
        col = self._track_number_column(main_window)
        df = getattr(main_window, "track_df", None)
        if df is None or col is None:
            return []
        with suppress(Exception):
            vals = (
                df.loc[df["Identification"] == ident, col]
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
            vals.sort()
            return vals
        return []

    def _row_at_time(self, sub, col, lineage_path, t_index):
        """Pick the row at exactly `t_index` within `sub` (one Identification's rows), preferring `lineage_path` then the highest TrackNumber."""
        if lineage_path and col and col in sub.columns:
            for n in reversed(list(lineage_path)):
                cand = sub[(sub["t"] == t_index) & (sub[col] == int(n))]
                if not cand.empty:
                    return cand.iloc[0]

        cand = sub[sub["t"] == t_index]
        if cand.empty:
            return None
        if col and col in cand.columns:
            cand = cand.sort_values(col)
            return cand.iloc[-1]
        return cand.iloc[0]

    def _get_track_row_for_time(
        self, main_window, ident, t_index, lineage_path=None
    ):
        """Fetch the track row at time t, preferring a lineage path."""
        df = getattr(main_window, "track_df", None)
        if df is None:
            return None
        sub = df[df["Identification"] == ident]
        if sub.empty:
            return None
        col = self._track_number_column(main_window)
        return self._row_at_time(sub, col, lineage_path, t_index)

    def _centroid_at_time(self, sub, col, lineage_path, t_index):
        """Return the (x, y) centroid at exactly `t_index`, or None if no row exists there or its centroid is unusable."""
        row = self._row_at_time(sub, col, lineage_path, t_index)
        if row is None:
            return None
        with suppress(ValueError):
            return self._extract_centroid(row)
        return None

    def _find_prev_next_centroid_t(self, sub, col, lineage_path, t_index):
        """Nearest previous/next t (within `sub`) with a usable centroid."""
        ts = sorted(int(t) for t in pd.unique(sub["t"].dropna()))
        prev_t = next(
            (
                t
                for t in reversed(ts)
                if t < t_index
                and self._centroid_at_time(sub, col, lineage_path, t)
                is not None
            ),
            None,
        )
        next_t = next(
            (
                t
                for t in ts
                if t > t_index
                and self._centroid_at_time(sub, col, lineage_path, t)
                is not None
            ),
            None,
        )
        return prev_t, next_t

    def _resolve_centroid(
        self, main_window, ident, t_index, lineage_path, policy: str
    ) -> tuple[float, float]:
        """Resolve the (x, y) centroid for `ident`/`lineage_path` at `t_index`."""
        df = getattr(main_window, "track_df", None)
        sub = df[df["Identification"] == ident] if df is not None else None
        if sub is None or sub.empty:
            raise KeyError("MISSING_FRAME")
        col = self._track_number_column(main_window)

        xy = self._centroid_at_time(sub, col, lineage_path, t_index)
        if xy is not None:
            return xy
        if policy == "Strict":
            raise KeyError("MISSING_FRAME")

        prev_t, next_t = self._find_prev_next_centroid_t(
            sub, col, lineage_path, t_index
        )
        if policy == "Hold last":
            chosen_t = prev_t
        elif policy == "Nearest":
            if prev_t is None:
                chosen_t = next_t
            elif next_t is None:
                chosen_t = prev_t
            else:
                chosen_t = (
                    prev_t
                    if (t_index - prev_t) <= (next_t - t_index)
                    else next_t
                )
        else:
            chosen_t = None

        if chosen_t is None:
            raise KeyError("MISSING_FRAME")
        xy = self._centroid_at_time(sub, col, lineage_path, chosen_t)
        if xy is None:
            raise KeyError("MISSING_FRAME")
        return xy

    def _children_for(self, n):
        """Return binary tree child track numbers for lineage indexing."""
        return (2 * n, 2 * n + 1)

    def _open_generations_dialog(self, main_window, panel):
        """Open dialog to select and apply a lineage path for a panel."""
        ident = (panel.get("ident") or "").strip()
        if not ident:
            self._warn(
                main_window, "Generations", "No Identification selected."
            )
            return
        avail = set(
            self._available_track_numbers_for_ident(main_window, ident)
        )
        if not avail:
            self._info(
                main_window,
                "Generations",
                f"No TrackNumber data found for ID '{ident}'.",
            )
            return

        dlg = QDialog(main_window.gif_window)
        dlg.setWindowTitle(f"Generations for ID: {ident}")
        lay = QVBoxLayout(dlg)
        info = QLabel(
            "Pick a lineage path by choosing children per level. Children follow heap indexing (n → 2n, 2n+1). Use '— stop here —' to end the path."
        )
        info.setWordWrap(True)
        lay.addWidget(info)

        level_layouts, combos = [], []

        def _make_level_row(level, options):
            row = QHBoxLayout()
            lbl = QLabel(f"Level {level}:")
            combo = QComboBox()
            combo.addItem("— stop here —", None)
            for v in options:
                combo.addItem(str(v), int(v))
            combo.setToolTip(TOOLTIPSTEXT.GENERATIONS_LEVEL_COMBO)
            row.addWidget(lbl)
            row.addWidget(combo, 1)
            lay.addLayout(row)
            level_layouts.append(row)
            combos.append(combo)
            return combo

        start_opts = [1] if 1 in avail else [min(avail)]
        c0 = _make_level_row(0, start_opts)

        def _rebuild_levels():
            while len(combos) > 1:
                cb = combos.pop()
                row = level_layouts.pop()
                while row.count():
                    item = row.takeAt(0)
                    w = item.widget()
                    if w:
                        w.setParent(None)
                lay.removeItem(row)
            sel = c0.currentData()
            last = sel if sel is not None else None
            level = 1
            while last is not None:
                ch = self._children_for(int(last))
                opts = [x for x in ch if x in avail]
                if not opts:
                    break
                cb = _make_level_row(level, opts)
                cb.setCurrentIndex(1 if cb.count() > 1 else 0)
                last = cb.currentData()
                level += 1
            dlg.adjustSize()

        c0.currentIndexChanged.connect(_rebuild_levels)
        _rebuild_levels()

        apply_all_chk = QCheckBox("Apply to all tiles with this ID")
        apply_all_chk.setToolTip(TOOLTIPSTEXT.GENERATIONS_APPLY_ALL)
        lay.addWidget(apply_all_chk)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        lay.addWidget(btns)

        def _on_ok():
            path = []
            for cb in combos:
                v = cb.currentData()
                if v is None:
                    break
                path.append(int(v))
            panel["lineage_path"] = path
            if apply_all_chk.isChecked():
                for p in getattr(main_window, "panels", []):
                    if (p.get("ident") or "").strip() == ident:
                        p["lineage_path"] = list(path)
                        self._update_single_panel_preview(main_window, p)
            else:
                self._update_single_panel_preview(main_window, panel)
            dlg.accept()

        btns.accepted.connect(_on_ok)
        btns.rejected.connect(dlg.reject)
        dlg.exec_()
