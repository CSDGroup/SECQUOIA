"""Dynamics plot grid lifecycle: building, rebuilding, resizing, and row count control."""

import contextlib
import logging

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from qtpy.QtCore import QEvent, Qt, QTimer
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QApplication,
    QGridLayout,
    QSizePolicy,
    QSplitter,
    QWidget,
)

from SECQUOIA.core.tracking.track_data import recompute_derived_for_row
from SECQUOIA.gui.common.messages import show_folder_warning
from SECQUOIA.gui.common.ui_utils import clear_layout
from SECQUOIA.gui.lineage_tree.lineage_tree import lineage_tree
from SECQUOIA.gui.lineage_tree.lineage_zoom import reset_lineage_zoom
from SECQUOIA.utils.helpers import update_time_marker
from SECQUOIA.utils.plotting import fit_row_y_axis, update_plot

LOG = logging.getLogger(__name__)


class DynamicsPlotGrid:
    """Build, rebuild, and resize the Dynamics plot grid and its row count."""

    def init_empty_plot(self) -> None:
        """Initialize an empty black plot (no axes, no ticks, no labels), and make
        it expand to fill the layout space.
        """
        self.graph3_layout.setContentsMargins(0, 0, 0, 0)
        self.graph3_layout.setSpacing(0)
        fig = plt.Figure()
        ax = fig.add_axes([0, 0, 1, 1])
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        ax.set_axis_off()
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.graph3_layout.addWidget(canvas)
        self.canvas = canvas

    def _clear_all_plots_ui(self):
        """Remove all plot, lineage, marker, and row-tool widgets."""
        if hasattr(self, "current_time_markers") and isinstance(
            self.current_time_markers, dict
        ):
            for marker in list(self.current_time_markers.values()):
                with contextlib.suppress(
                    RuntimeError, AttributeError, TypeError
                ):
                    marker["plot_widget"].removeItem(marker["item"])
            self.current_time_markers = {}

        clear_layout(getattr(self, "right_layout", None))

        for idx in range(1, 33):
            if hasattr(self, f"plot_widget_{idx}"):
                setattr(self, f"plot_widget_{idx}", None)
        self.plot_grid_container = None
        self.graph3_widget = None
        self.graph3_plot = None
        self.lineage_plot_widget = None
        self.lineage_container = None
        self.plots_and_lineage_splitter = None
        self.lineage_dynamic_container = None
        self.lineage_dynamic_layout = None
        self._lineage_time_line = None
        self._last_lineage_y_map = None
        self._lineage_y_map = {}
        self._zoom_visible_tracks = None
        if hasattr(self, "row_tools"):
            self.row_tools.clear()

        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            QApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def add_plot_widgets(self):
        """Build the dynamics-plot grid and lineage panel, replacing any existing ones."""
        self._clear_all_plots_ui()

        ch_count = int(getattr(self, "n_channels", 0))
        m_count = int(getattr(self, "n_masks", 0))
        max_plots = self._init_plot_grid(ch_count, m_count)

        for row in range(1, max_plots + 1):
            self._build_plot_row_widgets(row, ch_count, m_count)

        self._build_lineage_container()
        self._assemble_plots_and_lineage_splitter()

        self._sync_dynamics_plot_actions(busy=self._dynamics_plot_busy())

    def _init_plot_grid(self, ch_count: int, m_count: int) -> int:
        """Reset plot-grid state and return the clamped number of plot rows to build."""
        self.plot_grid_container = QWidget()
        grid_layout = QGridLayout(self.plot_grid_container)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setHorizontalSpacing(8)
        grid_layout.setVerticalSpacing(2)
        grid_layout.setColumnStretch(0, 1)
        self._plot_grid_layout = grid_layout

        default_max_plots = self._compute_max_plots(
            ch_count, m_count, hard_cap=4
        )

        if getattr(self, "_dynamics_plot_row_count", None) is None:
            self._dynamics_plot_row_count = default_max_plots

        try:
            max_plots = int(self._dynamics_plot_row_count)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            max_plots = default_max_plots

        max_plots = max(1, min(32, max_plots))

        self._dynamics_plot_row_count = max_plots
        self._max_plot_rows = max_plots
        self._features_defaulted = False

        if not hasattr(self, "selected_ch_by_channel"):
            self.selected_ch_by_channel = {}
        if not hasattr(self, "selected_m_by_channel"):
            self.selected_m_by_channel = {}
        if not hasattr(self, "selected_feature_by_row"):
            self.selected_feature_by_row = {}

        self.row_tools = {}

        # Highlight mode state
        if not hasattr(self, "_hl_active"):
            self._hl_active = False  # highlight mode toggle

        self._hl_palette = {
            "Green": QColor(76, 175, 80),
            "Blue": QColor(66, 133, 244),
            "Orange": QColor(255, 160, 0),
            "Magenta": QColor(194, 24, 91),
            "Cyan": QColor(0, 188, 212),
            "Yellow": QColor(251, 192, 45),
        }

        return max_plots

    def _assemble_plots_and_lineage_splitter(self) -> None:
        """Combine the plot grid and lineage panel into the splitter and draw them."""
        self.plots_and_lineage_splitter = QSplitter(Qt.Vertical)
        self.plots_and_lineage_splitter.setOpaqueResize(False)
        self.plots_and_lineage_splitter.setChildrenCollapsible(False)
        self.plots_and_lineage_splitter.setHandleWidth(6)
        self.plots_and_lineage_splitter.setStyleSheet(
            "QSplitter::handle { background: #666; }"
        )

        self.plots_and_lineage_splitter.addWidget(self.plot_grid_container)
        self.plots_and_lineage_splitter.addWidget(self.lineage_container)
        self.plots_and_lineage_splitter.setStretchFactor(0, 3)
        self.plots_and_lineage_splitter.setStretchFactor(1, 1)

        self.right_layout.addWidget(self.plots_and_lineage_splitter, 1)
        update_plot(self)
        QTimer.singleShot(
            0,
            lambda: (
                lineage_tree(self),
                self._autorange_lineage(),
                self._refresh_all_row_summaries(),
                self._refresh_all_row_igt(),
            ),
        )

    def _current_dynamics_plot_row_count(self) -> int:
        """Return the current user-selected dynamics plot row count."""
        current = getattr(self, "_dynamics_plot_row_count", None)

        if current is None:
            current = getattr(self, "_max_plot_rows", None) or 1

        try:
            return max(1, min(32, int(current)))
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return 1

    def _dynamics_plot_busy(self) -> bool:
        """Return True while a dynamics plot rebuild is pending or running."""
        return bool(
            getattr(self, "_dynamics_plot_rebuild_pending", False)
            or getattr(self, "_dynamics_plot_rebuild_running", False)
        )

    def _dynamics_plots_ready(self) -> bool:
        """Return True once data is loaded and a plot grid actually exists."""
        if not getattr(self, "folder_list", None):
            return False
        return getattr(self, "plot_grid_container", None) is not None

    def _require_dynamics_plots_ready(self) -> bool:
        """Return True if plot rows can be changed, warning the user if not."""
        if self._dynamics_plots_ready():
            return True

        if not getattr(self, "folder_list", None):
            LOG.warning("Please first load CSV file and select folder")
            show_folder_warning(self)
        return False

    def add_dynamics_plot_row(self) -> None:
        """Add one Dynamics plot row."""
        if not self._require_dynamics_plots_ready():
            return

        if self._dynamics_plot_busy():
            return

        count = self._current_dynamics_plot_row_count()

        if count >= 32:
            self._sync_dynamics_plot_actions(busy=False)
            return

        self._dynamics_plot_row_count = count + 1
        self._request_dynamics_plot_rebuild()

    def remove_dynamics_plot_row(self) -> None:
        """Remove the last Dynamics plot row safely."""
        if not self._require_dynamics_plots_ready():
            return

        if self._dynamics_plot_busy():
            return

        old_count = self._current_dynamics_plot_row_count()
        new_count = max(1, old_count - 1)

        if new_count == old_count:
            self._sync_dynamics_plot_actions(busy=False)
            return

        self._dynamics_plot_row_count = new_count

        for attr in (
            "selected_feature_by_row",
            "selected_m_by_channel",
            "selected_ch_by_channel",
        ):
            mapping = getattr(self, attr, None)
            if isinstance(mapping, dict):
                for row in range(new_count + 1, old_count + 1):
                    mapping.pop(row, None)

        self._request_dynamics_plot_rebuild()

    def _sync_dynamics_plot_actions(self, busy: bool = False) -> None:
        """Enable/disable dynamics plot menu actions safely."""
        count = self._current_dynamics_plot_row_count()
        ready = self._dynamics_plots_ready()

        add_action = getattr(self, "action_add_dynamics_plot", None)
        remove_action = getattr(self, "action_remove_dynamics_plot", None)

        if add_action is not None:
            add_action.setEnabled(ready and (not busy) and count < 32)

        if remove_action is not None:
            remove_action.setEnabled(ready and (not busy) and count > 1)

    def _request_dynamics_plot_rebuild(self) -> None:
        """Schedule a debounced dynamics plot rebuild on the next event-loop tick."""

        if self._dynamics_plot_busy():
            return

        self._dynamics_plot_rebuild_pending = True
        self._sync_dynamics_plot_actions(busy=True)

        QTimer.singleShot(0, self._run_dynamics_plot_rebuild)

    def _run_dynamics_plot_rebuild(self) -> None:
        """Actually rebuild the Dynamics plot once it is safe to do so."""
        self._dynamics_plot_rebuild_running = True
        self._dynamics_plot_rebuild_pending = False

        try:
            self.setUpdatesEnabled(False)
            self.add_plot_widgets()
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            LOG.warning("[Dynamics plots] rebuild failed: %r", e)
            self.setUpdatesEnabled(True)
            self._dynamics_plot_rebuild_running = False
            self._sync_dynamics_plot_actions(busy=False)
            return

        QTimer.singleShot(50, self._finish_dynamics_plot_rebuild)

    def _finish_dynamics_plot_rebuild(self) -> None:
        """Finalize dynamics plot rebuild after Qt has settled."""
        try:
            self._autorange_lineage()
            reset_lineage_zoom(self)

            if getattr(self, "show_time_marker", False):
                update_time_marker(self)

        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            LOG.warning("[Dynamics plots] lineage refresh failed: %r", e)

        finally:
            QTimer.singleShot(0, self._unlock_dynamics_plot_rebuild)

    def _unlock_dynamics_plot_rebuild(self) -> None:
        """Re-enable UI updates and clear the rebuild-running flag."""
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            QApplication.sendPostedEvents(None, QEvent.DeferredDelete)

        self.setUpdatesEnabled(True)
        self._dynamics_plot_rebuild_running = False
        self._sync_dynamics_plot_actions(busy=False)

    def _compute_max_plots(
        self, ch_count: int, m_count: int, hard_cap: int = 4
    ) -> int:
        """Compute number of plot rows based on channels x masks."""
        ch = max(int(ch_count or 0), 1)
        m = max(int(m_count or 0), 1)
        return min(ch * m, hard_cap)

    def _defaults_for_row(
        self, row_idx: int, ch_count: int, m_count: int
    ) -> tuple[int | None, int | None]:
        """Return the default (channel, mask) selection for a new plot row.

        The channel defaults to the first configured channel; the mask
        cycles through 1..m_count as row_idx increases, so newly added rows
        spread across the available masks instead of repeating the same one.
        """
        ch = getattr(self, "ids_channels", None)
        if ch:
            ch = ch[0][1:]

        i = row_idx - 1
        cc = max(int(ch_count or 0), 1)
        mm = max(int(m_count or 0), 1)
        m = ((i // cc) % mm) + 1
        return (ch, m if m_count > 0 else None)

    def fit_plot_row(self, plot_row: int) -> None:
        """Auto-zoom the given plot row after the next event-loop tick."""
        pw = getattr(self, f"plot_widget_{plot_row}", None)
        if not pw:
            return
        plot_item = pw.getPlotItem()
        scheduled_for = str(getattr(self, "ident", ""))

        def _fit_and_retick() -> None:
            """Rescale Y to the plotted data, with ticks that match it.

            Deliberately not ``ViewBox.autoRange``: with nothing plotted that
            keeps the range the row already has, so a row with no data for the
            new identification would still show the previous one's scale.
            """
            if str(getattr(self, "ident", "")) != scheduled_for:
                # Held down arrow key: the row now shows another Tree-ID, and
                # that one queued a fit of its own.
                return

            axis_fs = getattr(self, "_plot_params", {}).get(
                "axis_font_size", 10
            )
            fit_row_y_axis(
                plot_item,
                axis_fs,
                min_width=getattr(self, "_shared_left_axis_width", None),
            )

        QTimer.singleShot(0, _fit_and_retick)

    def fit_all_plots(self) -> None:
        """Auto-zoom all existing plot widgets."""
        max_rows = int(getattr(self, "_max_plot_rows", 0) or 0)

        for row in range(1, max_rows + 1):
            if getattr(self, f"plot_widget_{row}", None):
                self.fit_plot_row(row)

    def _recompute_derived_for_row(
        self,
        row_idx: int | None = None,
        *,
        mask_idx: int | None = None,
    ) -> None:
        """Reapply all saved metric rules after base mask/channel measurements change."""
        recompute_derived_for_row(self, row_idx, mask_idx=mask_idx)
