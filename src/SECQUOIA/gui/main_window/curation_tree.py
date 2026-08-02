"""MainWindow's interaction with the curation tree: item lookup, status cycling, context menu.

The tree-building free functions this class calls into (apply_curation_status,
apply_active_state, update_list, ...) live in SECQUOIA.gui.curation_tree, since
core/outlier_detection/, core/tracking/, and others need them independent of MainWindow.
"""

import qtawesome as qta
from qtpy.QtCore import QPoint
from qtpy.QtWidgets import QAction, QMenu

from SECQUOIA.config import CURATIONROLE, CURATIONSTATUS
from SECQUOIA.gui.curation_tree import (
    apply_active_state,
    apply_curation_status,
)


class CurationTree:
    """MainWindow's interaction with the curation tree: item lookup, status cycling, context menu."""

    def _find_tree_item_for(self, ident: str, track: float | None):
        """Find the tree item for Identification and TrackNumber."""
        suffix = str(ident).split("-")[-1]
        for i in range(self.tree_widget.topLevelItemCount()):
            parent = self.tree_widget.topLevelItem(i)
            if parent and parent.text(0) == suffix:
                if track is None:
                    return parent
                tf = float(track)
                for j in range(parent.childCount()):
                    ch = parent.child(j)
                    tr = self._resolve_track_from_item(ch)
                    if tr is not None and float(tr) == tf:
                        return ch
        return None

    def _resolve_ident_from_item(self, it) -> str | None:
        """Resolve full Identification string for the given tree item."""
        df_all = getattr(self, "filtered_df", None)
        if df_all is None or "Identification" not in df_all.columns:
            return None
        suffix = it.text(0) if it.parent() is None else it.parent().text(0)
        last_part = df_all["Identification"].astype(str).str.split("-").str[-1]
        matches = df_all.loc[last_part == suffix, "Identification"].unique()
        return matches[0] if len(matches) else None

    def _resolve_track_from_item(self, it) -> float | None:
        """Resolve TrackNumber from a child item label."""
        if it.parent() is None:
            return None
        txt = it.text(0)
        for prefix in ("Cell", ""):
            s = txt.replace(prefix, "").strip()
            try:
                return float(s)
            except ValueError:
                continue
        return None

    def _set_inspected_for_ident(self, ident, inspected_val: int) -> None:
        """Set inspected (0/1/2) for all rows matching an Identification."""
        if not ident:
            return
        df_all = getattr(self, "filtered_df", None)
        if df_all is not None and {"Identification", "inspected"}.issubset(
            df_all.columns
        ):
            df_all.loc[df_all["Identification"].eq(ident), "inspected"] = (
                inspected_val
            )

        td = getattr(self, "track_df", None)
        if td is not None and {"Identification", "inspected"}.issubset(
            td.columns
        ):
            td.loc[td["Identification"].eq(ident), "inspected"] = inspected_val

    def _set_inspected_for_track(
        self, ident, track, inspected_val: int
    ) -> None:
        """Set inspected (0/1/2) for a specific (Identification, TrackNumber)."""
        if not ident:
            return
        need = {"Identification", "TrackNumber", "inspected"}

        df_all = getattr(self, "filtered_df", None)
        if df_all is not None and need.issubset(df_all.columns):
            df_all.loc[
                df_all["Identification"].eq(ident)
                & df_all["TrackNumber"].eq(track),
                "inspected",
            ] = inspected_val

        td = getattr(self, "track_df", None)
        if td is not None and need.issubset(td.columns):
            td.loc[
                td["Identification"].eq(ident) & td["TrackNumber"].eq(track),
                "inspected",
            ] = inspected_val

    def _aggregate_parent_inspected_from_df(self, ident) -> tuple[int, object]:
        """Compute parent inspected value and status from all rows of a Tree-ID."""
        df_all = getattr(self, "filtered_df", None)
        if df_all is None or {"Identification", "inspected"} - set(
            df_all.columns
        ):
            return 0, CURATIONSTATUS.CURATION_NOT_CHECKED

        df_ident = df_all[df_all["Identification"] == ident]
        if df_ident.empty:
            return 0, CURATIONSTATUS.CURATION_NOT_CHECKED

        inspected = df_ident["inspected"].fillna(0).astype(int)
        if (inspected == 0).any():
            return 0, CURATIONSTATUS.CURATION_NOT_CHECKED
        if (inspected == 1).any():
            return 1, CURATIONSTATUS.CURATION_IN_PROGRESS
        return 2, CURATIONSTATUS.CURATION_CHECKED

    def _sync_children_visuals_from_df(self, parent_item) -> None:
        """Update each child’s icon/tooltips from DataFrame values."""
        ident = self._resolve_ident_from_item(parent_item)
        if not ident:
            return

        need = {"Identification", "TrackNumber", "active", "inspected"}
        df_all = getattr(self, "filtered_df", None)
        if df_all is None or not need.issubset(df_all.columns):
            return

        df_ident = df_all[df_all["Identification"] == ident]
        if df_ident.empty:
            return

        for i in range(parent_item.childCount()):
            ch = parent_item.child(i)
            track = self._resolve_track_from_item(ch)
            if track is None:
                continue

            df_track = df_ident[df_ident["TrackNumber"] == track]
            if df_track.empty:
                continue

            is_active = bool(int(df_track["active"].fillna(0).max()))
            apply_active_state(ch, is_active)

            inspected_col = df_track["inspected"].fillna(0)
            if (inspected_col == 2).all():
                apply_curation_status(ch, CURATIONSTATUS.CURATION_CHECKED)
            elif (inspected_col == 1).any():
                apply_curation_status(ch, CURATIONSTATUS.CURATION_IN_PROGRESS)
            else:
                apply_curation_status(ch, CURATIONSTATUS.CURATION_NOT_CHECKED)

    def _apply_parent_status_from_children(self, parent_item) -> None:
        """Refresh parent icon/status based on its children’s rows in dataframe."""
        ident = self._resolve_ident_from_item(parent_item)
        if not ident:
            return
        inspected_val, status = self._aggregate_parent_inspected_from_df(ident)
        apply_curation_status(parent_item, status)
        if inspected_val == 2:
            self._set_inspected_for_ident(ident, 2)

    def _cycle_curation_status_for_item(self, item) -> None:
        """Cycle: Checked(2) -> In progress(1) -> Not checked(0)."""
        cur_status = item.data(0, CURATIONROLE.CURATION_ROLE)
        cycle = [
            (CURATIONSTATUS.CURATION_CHECKED, 2),
            (CURATIONSTATUS.CURATION_IN_PROGRESS, 1),
            (CURATIONSTATUS.CURATION_NOT_CHECKED, 0),
        ]
        try:
            idx = next(
                i for i, (st, _) in enumerate(cycle) if st == cur_status
            )
        except StopIteration:
            idx = -1
        next_status, inspected_val = cycle[(idx + 1) % len(cycle)]

        apply_curation_status(item, next_status)

        ident = self._resolve_ident_from_item(item)
        track = self._resolve_track_from_item(item)

        if track is None:
            self._set_inspected_for_ident(ident, inspected_val)
            self._sync_children_visuals_from_df(item)
        else:
            self._set_inspected_for_track(ident, track, inspected_val)
            parent = item.parent()
            if parent is not None:
                self._apply_parent_status_from_children(parent)

    def on_tree_context_menu(self, pos: QPoint) -> None:
        """Build and display the context menu for a tree item."""
        item = self.tree_widget.itemAt(pos)
        if item is None:
            return

        ident = self._resolve_ident_from_item(item)
        track = self._resolve_track_from_item(item)

        menu = QMenu(self)

        # Curation submenu
        curation_menu = menu.addMenu("Curation status")
        act_checked = QAction(
            qta.icon("fa5s.check-circle", color="#1db954"), "Checked", self
        )
        act_progress = QAction(
            qta.icon("fa5s.question-circle", color="#ffbf00"),
            "In progress",
            self,
        )
        act_not = QAction(
            qta.icon("fa5s.times-circle", color="#e73ce7"), "Not checked", self
        )

        def _do_parent(status, inspected_val: int):
            """Apply a curation status to an entire Identification and sync children."""
            apply_curation_status(item, status)
            self._set_inspected_for_ident(ident, inspected_val)
            self._sync_children_visuals_from_df(item)

        def _do_child(status, inspected_val: int):
            """Apply a curation status to a specific (Identification, TrackNumber)."""
            apply_curation_status(item, status)
            self._set_inspected_for_track(ident, track, inspected_val)
            self._apply_parent_status_from_children(item.parent())

        if track is None:
            act_checked.triggered.connect(
                lambda: _do_parent(CURATIONSTATUS.CURATION_CHECKED, 2)
            )
            act_progress.triggered.connect(
                lambda: _do_parent(CURATIONSTATUS.CURATION_IN_PROGRESS, 1)
            )
            act_not.triggered.connect(
                lambda: _do_parent(CURATIONSTATUS.CURATION_NOT_CHECKED, 0)
            )
        else:
            act_checked.triggered.connect(
                lambda: _do_child(CURATIONSTATUS.CURATION_CHECKED, 2)
            )
            act_progress.triggered.connect(
                lambda: _do_child(CURATIONSTATUS.CURATION_IN_PROGRESS, 1)
            )
            act_not.triggered.connect(
                lambda: _do_child(CURATIONSTATUS.CURATION_NOT_CHECKED, 0)
            )

        curation_menu.addAction(act_checked)
        curation_menu.addAction(act_progress)
        curation_menu.addAction(act_not)

        cell_menu = menu.addMenu("Cell status")
        act_activate = QAction(qta.icon("fa5s.toggle-on"), "Activate", self)
        act_deactivate = QAction(
            qta.icon("fa5s.toggle-off"), "Deactivate", self
        )

        def _do_active_parent(is_active: bool):
            """Toggle active state for an entire Identification and sync children."""
            apply_active_state(item, is_active)
            val = 1 if is_active else 0

            df_all = getattr(self, "filtered_df", None)
            if df_all is not None and {"Identification", "active"}.issubset(
                df_all.columns
            ):
                df_all.loc[df_all["Identification"].eq(ident), "active"] = val

            td = getattr(self, "track_df", None)
            if td is not None and {"Identification", "active"}.issubset(
                td.columns
            ):
                td.loc[td["Identification"].eq(ident), "active"] = val

            self._sync_children_visuals_from_df(item)

        def _do_active_child(is_active: bool):
            """Toggle active state for a specific (Identification, TrackNumber)."""
            apply_active_state(item, is_active)
            val = 1 if is_active else 0

            df_all = getattr(self, "filtered_df", None)
            need = {"Identification", "TrackNumber", "active"}
            if df_all is not None and need.issubset(df_all.columns):
                df_all.loc[
                    df_all["Identification"].eq(ident)
                    & df_all["TrackNumber"].eq(track),
                    "active",
                ] = val

            td = getattr(self, "track_df", None)
            if td is not None and need.issubset(td.columns):
                td.loc[
                    td["Identification"].eq(ident)
                    & td["TrackNumber"].eq(track),
                    "active",
                ] = val

            self._apply_parent_status_from_children(item.parent())

        if track is None:
            act_activate.triggered.connect(lambda: _do_active_parent(True))
            act_deactivate.triggered.connect(lambda: _do_active_parent(False))
        else:
            act_activate.triggered.connect(lambda: _do_active_child(True))
            act_deactivate.triggered.connect(lambda: _do_active_child(False))

        cell_menu.addAction(act_activate)
        cell_menu.addAction(act_deactivate)

        vp = getattr(self, "_tree_viewport", self.tree_widget.viewport())
        global_pos = vp.mapToGlobal(pos)
        menu.exec_(global_pos)
