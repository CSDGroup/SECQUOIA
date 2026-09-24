"""Global hotkeys and Qt event filtering for MainWindow."""

import contextlib
import logging
import os

from qtpy.QtCore import QEvent, QObject, QPoint, QRect, QSize, Qt
from qtpy.QtGui import QKeyEvent, QKeySequence, QMouseEvent, QWheelEvent
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QPlainTextEdit,
    QShortcut,
    QSpinBox,
    QTextEdit,
    QToolTip,
)

from SECQUOIA.config import CURATIONSTATUS
from SECQUOIA.core.segmentation.close_mask_view import assigned_mask_tooltip
from SECQUOIA.core.segmentation.mask_selection import (
    ensure_current_df_subset,
    show_all_masks,
    show_current_mask,
)
from SECQUOIA.core.tracking import on_division_clicked
from SECQUOIA.gui.cell_inspector.integration import open_cell_inspector
from SECQUOIA.gui.curation_tree import apply_curation_status
from SECQUOIA.gui.dialogs.lineage_style_dialog import (
    open_lineage_style_dialog,
)
from SECQUOIA.gui.dialogs.plot_params_dialog import open_plot_params_dialog
from SECQUOIA.gui.lineage_tree.lineage_selection import (
    _clear_track_selection,
    _select_all_tracks,
)
from SECQUOIA.gui.lineage_tree.lineage_tree import lineage_tree
from SECQUOIA.gui.lineage_tree.lineage_zoom import reset_lineage_zoom
from SECQUOIA.gui.loading.channel_mask_manager_dialog import (
    open_channel_mask_manager,
)
from SECQUOIA.gui.loading.load_data import load_data_window
from SECQUOIA.gui.loading.loading_dialogs import (
    open_load_previous_project_gui,
)
from SECQUOIA.gui.outlier import (
    open_outlier_detection_window,
    show_current_outlier_parameters,
)
from SECQUOIA.gui.outlier.close_mask_review import review_current_point
from SECQUOIA.gui.outlier.navigation import (
    change_outlier,
    change_to_next_outlier,
)
from SECQUOIA.gui.outlier.reset import reset_outlier_state
from SECQUOIA.gui.position_navigation import (
    load_position,
    open_position_window,
)
from SECQUOIA.utils.helpers import (
    change_cell,
    change_time_point,
    change_time_point_in_jump_channel,
    clear_highlight_paints,
    cycle_highlight_color,
    set_jump_channel,
    toggle_highlight_mode,
)

LOG = logging.getLogger(__name__)
_TOOLTIP_AREA_PX = 12


class KeyBindings:
    """Global hotkeys, canvas key/wheel filtering, and the application event filter."""

    def install_global_hotkeys(self) -> None:
        """Install application-wide hotkeys."""
        if getattr(self, "_global_hotkeys_installed", False):
            return
        self._shortcuts = []

        def add_hotkey(key: str, callback) -> None:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(callback)
            self._shortcuts.append(sc)

        def add_hotkey_both(suffix: str, callback) -> None:
            """Register the same callback on both Ctrl+<suffix> and Meta+<suffix>,
            so the shortcut works on Windows/Linux and macOS.
            """
            add_hotkey(f"Ctrl+{suffix}", callback)
            add_hotkey(f"Meta+{suffix}", callback)

        def _guard_if_typing(cb):
            """Wrap a hotkey callback so it is skipped while a text-entry widget has focus."""

            def wrapper():
                w = QApplication.focusWidget()
                if isinstance(
                    w,
                    QLineEdit
                    | QTextEdit
                    | QPlainTextEdit
                    | QSpinBox
                    | QDoubleSpinBox,
                ):
                    return
                if isinstance(w, QComboBox) and w.isEditable():
                    return
                cb()

            return wrapper

        add_hotkey("Down", lambda: change_cell(self, "next"))
        add_hotkey("Up", lambda: change_cell(self, "previous"))
        add_hotkey("Left", lambda: change_time_point_in_jump_channel(self, -1))
        add_hotkey(
            "Right", lambda: change_time_point_in_jump_channel(self, +1)
        )

        add_hotkey_both(
            "Down", _guard_if_typing(lambda: change_outlier(self, "next"))
        )
        add_hotkey_both(
            "Up", _guard_if_typing(lambda: change_outlier(self, "previous"))
        )

        add_hotkey_both(
            "Right", _guard_if_typing(lambda: change_to_next_outlier(self, +1))
        )
        add_hotkey_both(
            "Left", _guard_if_typing(lambda: change_to_next_outlier(self, -1))
        )

        for n in range(10):
            add_hotkey_both(
                str(n),
                _guard_if_typing(
                    lambda n=n: set_jump_channel(self, f"w{n:02d}")
                ),
            )

        add_hotkey_both(
            "Q", _guard_if_typing(lambda: reset_lineage_zoom(self))
        )
        add_hotkey("C", _guard_if_typing(lambda: review_current_point(self)))
        add_hotkey("0", _guard_if_typing(lambda: check_current_ident(self)))
        add_hotkey("D", lambda: cell_fate(self, "Dead"))
        add_hotkey("H", lambda: cell_fate(self, "Healthy"))
        add_hotkey("K", lambda: cell_fate(self, "LowSignal"))
        add_hotkey("O", lambda: cell_fate(self, "Outlier"))
        add_hotkey("L", lambda: cell_fate(self, "Lost"))
        add_hotkey("F", lambda: cell_fate(self, "OoF"))
        add_hotkey_both("S", self.on_save_clicked)
        add_hotkey_both("N", lambda: load_data_window(self))
        add_hotkey_both("O", lambda: open_load_previous_project_gui(self))
        add_hotkey_both("F", self.open_cytometric_analysis)
        add_hotkey_both("H", lambda: open_outlier_detection_window(self))
        add_hotkey_both("R", lambda: reset_outlier_state(self))
        add_hotkey_both("E", self.open_image_movie_exporter)
        add_hotkey_both("M", lambda: show_all_masks(self))
        add_hotkey_both("Shift+M", lambda: show_current_mask(self))
        add_hotkey_both("Shift+C", lambda: open_channel_mask_manager(self))
        add_hotkey_both("I", lambda: open_cell_inspector(self))
        add_hotkey_both(
            "Shift+Left",
            _guard_if_typing(lambda: load_position(self, "previous")),
        )
        add_hotkey_both(
            "Shift+Right",
            _guard_if_typing(lambda: load_position(self, "next")),
        )
        add_hotkey_both(
            "Shift+O",
            _guard_if_typing(lambda: open_position_window(self)),
        )
        add_hotkey(
            "M",
            lambda: self._create_new_mask_id(
                self._active_viewer_and_tools()[0]
            ),
        )
        add_hotkey_both("D", lambda: on_division_clicked(self))
        add_hotkey_both("T", self.open_ttt_dataformat_transformer)
        add_hotkey("Escape", self.close)
        add_hotkey_both(
            "A", _guard_if_typing(lambda: _select_all_tracks(self))
        )
        add_hotkey_both(
            "Shift+A",
            _guard_if_typing(lambda: _clear_track_selection(self)),
        )

        add_hotkey_both(
            "Shift+I",
            _guard_if_typing(lambda: self.action_tracks_ids.trigger()),
        )
        add_hotkey_both(
            "Shift+K",
            _guard_if_typing(lambda: self.action_tracks_visible.trigger()),
        )
        add_hotkey_both(
            "Shift+T",
            _guard_if_typing(lambda: self.action_show_time_marker.trigger()),
        )
        add_hotkey_both(
            "Shift+B",
            _guard_if_typing(lambda: self.action_tracking_bar.trigger()),
        )
        add_hotkey_both(
            "Shift+H",
            _guard_if_typing(lambda: show_current_outlier_parameters(self)),
        )
        add_hotkey("F1", self.show_help_popup)
        add_hotkey_both(
            "Shift+D",
            _guard_if_typing(lambda: open_plot_params_dialog(self)),
        )
        add_hotkey_both(
            "Shift+=",
            _guard_if_typing(lambda: self.add_dynamics_plot_row()),
        )
        add_hotkey_both(
            "Shift+-",
            _guard_if_typing(lambda: self.remove_dynamics_plot_row()),
        )
        add_hotkey_both(
            "Shift+L",
            _guard_if_typing(lambda: open_lineage_style_dialog(self)),
        )

        def _tool_hotkey(mode: str):
            """Activate a viewer interaction tool from a hotkey."""
            v, ts = self._active_viewer_and_tools()
            if not v or not ts:
                return
            if mode == "pan":
                ts._btn_pan.setChecked(True)
            elif mode == "erase":
                ts._btn_erase.setChecked(True)
            elif mode == "brush":
                ts._btn_brush.setChecked(True)
            self._focus_canvas(v)

        if not hasattr(self, "_tool_toggle_state"):
            self._tool_toggle_state = "erase"

        def toggle_tool():
            """Toggle between brush and erase tools."""
            self._tool_toggle_state = (
                "brush" if self._tool_toggle_state == "erase" else "erase"
            )
            _tool_hotkey(self._tool_toggle_state)

        add_hotkey("Space", _guard_if_typing(toggle_tool))
        add_hotkey("6", lambda: _tool_hotkey("pan"))
        add_hotkey("1", lambda: _tool_hotkey("erase"))
        add_hotkey("2", lambda: _tool_hotkey("brush"))
        add_hotkey_both(
            "P", _guard_if_typing(lambda: toggle_highlight_mode(self))
        )

        add_hotkey_both(
            "Shift+P",
            _guard_if_typing(lambda: cycle_highlight_color(self, True)),
        )

        add_hotkey_both(
            "Shift+R",
            _guard_if_typing(lambda: clear_highlight_paints(self)),
        )

        def _undo_cb():
            """Jump to the last edited time point, then undo via undo_labels_edit()."""
            layer = self._labels_target_for_history()
            if layer is None:
                LOG.warning(
                    "[undo] no Labels layer available (active/last-edited not found)"
                )
                return

            t_last = getattr(self, "_last_labels_layer_t", None)
            if t_last is not None:
                self._goto_time_for_history(t_last)

            self.undo_labels_edit(layer)

        def _redo_cb():
            """Jump to the last edited time point, then redo via redo_labels_edit()."""
            layer = self._labels_target_for_history()
            if layer is None:
                LOG.warning(
                    "[redo] no Labels layer available (active/last-edited not found)"
                )
                return

            t_last = getattr(self, "_last_labels_layer_t", None)
            if t_last is not None:
                self._goto_time_for_history(t_last)

            self.redo_labels_edit(layer)

        add_hotkey_both("Z", _undo_cb)
        add_hotkey_both("Shift+Z", _redo_cb)
        add_hotkey_both("Y", _redo_cb)

        add_hotkey(
            "]", _guard_if_typing(lambda: self._nudge_selected_mask(True))
        )
        add_hotkey(
            "[", _guard_if_typing(lambda: self._nudge_selected_mask(False))
        )

        install_alt_plus_minus_brush_resize(self, step=1, accel=5)
        self._install_arrow_event_filter()
        self._global_hotkeys_installed = True

    def _active_viewer_and_tools(self):
        """Return (viewer, tool_strip) for the viewer whose canvas has focus."""
        try:
            for i, v in enumerate(
                getattr(self, "viewer_fluorescence", []) or []
            ):
                try:
                    qt_view = v.window._qt_viewer
                    if (
                        getattr(qt_view.canvas, "native", None)
                        and qt_view.canvas.native.hasFocus()
                    ):
                        ts = (
                            self.tool_strips[i]
                            if i < len(self.tool_strips)
                            else None
                        )
                        return v, ts
                except (RuntimeError, AttributeError, TypeError):
                    pass
            if getattr(self, "viewer_fluorescence", None):
                v = self.viewer_fluorescence[0]
                ts = self.tool_strips[0] if self.tool_strips else None
                return v, ts
        except (RuntimeError, AttributeError, TypeError):
            pass
        return None, None

    # napari's own 0/3/4/5/7/8/9 canvas shortcuts, suppressed so the keys can
    # be reused as SECQUOIA hotkeys.
    _BLOCKED_CANVAS_KEYS = (
        Qt.Key_0,
        Qt.Key_3,
        Qt.Key_4,
        Qt.Key_5,
        Qt.Key_7,
        Qt.Key_8,
        Qt.Key_9,
    )

    def _is_blocked_canvas_key(self, obj: QObject, event: QEvent) -> bool:
        """True if event is a disabled napari number key shortcut on a viewer canvas."""
        return (
            obj in getattr(self, "_canvas_native_to_viewer", {})
            and isinstance(event, QKeyEvent)
            and event.key() in self._BLOCKED_CANVAS_KEYS
            and not (event.modifiers() & Qt.ControlModifier)
        )

    def _is_brush_resize_wheel(self, obj: QObject, event: QEvent) -> bool:
        """True if event is a Ctrl+wheel brush/eraser-resize gesture on a viewer canvas."""
        return (
            obj in getattr(self, "_canvas_native_to_viewer", {})
            and isinstance(event, QWheelEvent)
            and bool(event.modifiers() & Qt.ControlModifier)
        )

    def _install_arrow_event_filter(self) -> None:
        """Install this window as an application-wide Qt event filter."""
        if getattr(self, "_arrow_filter_installed", False):
            return
        app = QApplication.instance()
        if not app:
            return
        app.installEventFilter(self)
        self._arrow_filter_installed = True

    def _show_canvas_tooltip(self, obj: QObject, event: QEvent) -> bool:
        """Show the tooltip for the mask under the cursor of a napari canvas."""
        viewer = getattr(self, "_canvas_native_to_viewer", {}).get(obj)
        if viewer is None:
            return False
        text = assigned_mask_tooltip(self, viewer)
        if not text:
            return False
        # The tooltip goes away once the cursor leaves the area around it.
        area = QRect(
            event.pos() - QPoint(_TOOLTIP_AREA_PX, _TOOLTIP_AREA_PX),
            QSize(2 * _TOOLTIP_AREA_PX, 2 * _TOOLTIP_AREA_PX),
        )
        QToolTip.showText(event.globalPos(), text, obj, area)
        return True

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """Application wide event filter installed on every canvas/tree widget."""
        if self._handle_tree_ctrl_click(obj, event):
            return True

        et = event.type()
        if et == QEvent.ToolTip and self._show_canvas_tooltip(obj, event):
            return True

        if et == QEvent.Enter:
            viewer = getattr(self, "_canvas_native_to_viewer", {}).get(obj)
            if viewer is not None:
                self._focus_canvas(viewer)

        if et in (
            QEvent.ShortcutOverride,
            QEvent.KeyPress,
        ) and self._is_blocked_canvas_key(obj, event):
            event.accept()
            return True

        if et == QEvent.Wheel and self._is_brush_resize_wheel(obj, event):
            nudge = getattr(self, "_nudge_brush_size", None)
            sign = 1 if event.angleDelta().y() > 0 else -1
            if nudge is not None and nudge(sign):
                return True

        if et == QEvent.ShortcutOverride and self._is_time_jump_key_event(
            event
        ):
            event.accept()
            return True

        if et == QEvent.KeyPress and self._is_time_jump_key_event(event):
            return self._handle_time_jump_keypress(event)

        return super().eventFilter(obj, event)

    def _handle_tree_ctrl_click(self, obj: QObject, event: QEvent) -> bool:
        """Cycle curation status on Ctrl/Meta + left-click on the tree viewport.

        Returns True if the event was handled and should be consumed.
        """
        is_tree_left_click = (
            obj is getattr(self, "_tree_viewport", None)
            and event.type()
            in (
                QEvent.MouseButtonPress,
                QEvent.MouseButtonRelease,
                QEvent.MouseButtonDblClick,
            )
            and isinstance(event, QMouseEvent)
            and event.button() == Qt.LeftButton
        )
        if not is_tree_left_click:
            return False

        mods = QApplication.keyboardModifiers()
        if not (mods & (Qt.ControlModifier | Qt.MetaModifier)):
            return False

        if event.type() == QEvent.MouseButtonPress:
            item = self.tree_widget.itemAt(event.pos())
            if item is not None:
                self._cycle_curation_status_for_item(item)
        event.accept()
        return True

    def _is_time_jump_key_event(self, event: QEvent) -> bool:
        """True if `event` is an unmodified Left/Right arrow key press."""
        key = getattr(event, "key", lambda: None)()
        return (
            key in (Qt.Key_Left, Qt.Key_Right)
            and event.modifiers() == Qt.NoModifier
        )

    def _handle_time_jump_keypress(self, event: QEvent) -> bool:
        """Step the current time point per a Left/Right arrow key press."""
        if hasattr(event, "isAutoRepeat") and event.isAutoRepeat():
            return True
        change_time_point_in_jump_channel(
            self, -1 if event.key() == Qt.Key_Left else +1
        )
        return True


def cell_fate(main_window, fate: str) -> None:
    """Set the cell fate for the current cell."""
    if not hasattr(main_window, "folder_list") or not main_window.folder_list:
        from SECQUOIA.gui.common.messages import show_folder_warning

        LOG.warning("Please first load CSV file and select folder")
        show_folder_warning(main_window)
        return

    valid_fates = (
        "Healthy",
        "Dead",
        "Lost",
        "OoF",
        "LowSignal",
        "Outlier",
    )
    if fate not in valid_fates:
        raise ValueError(
            f"Invalid cell fate: {fate}. Must be one of {valid_fates}."
        )

    main_window.track_df.loc[
        (main_window.track_df["Identification"] == main_window.ident)
        & (main_window.track_df["t"] == main_window.current_time_index)
        & (
            main_window.track_df["TrackNumber"]
            == main_window.current_TrackNumber_plot
        ),
        "Cellfate",
    ] = fate
    main_window.filtered_df = main_window.track_df[
        main_window.track_df["Position"]
        == int(
            os.path.basename(
                main_window.position_folders[
                    main_window.current_position_index
                ]
            ).split("_p")[-1]
        )
    ]
    main_window.df_subset = main_window.track_df[
        main_window.track_df["Identification"] == main_window.ident
    ]
    lineage_tree(main_window)

    track = main_window.current_TrackNumber_plot
    item = main_window._find_tree_item_for(main_window.ident, track)
    if item is not None:
        main_window._set_inspected_for_track(main_window.ident, track, 2)
        apply_curation_status(item, CURATIONSTATUS.CURATION_CHECKED)
        parent = item.parent()
        if parent is not None:
            main_window._apply_parent_status_from_children(parent)

    LOG.info("%s is marked as %s.", main_window.ident, fate)


def check_current_ident(main_window) -> None:
    """Mark the whole current Identification as checked and advance to the next one."""
    if not hasattr(main_window, "folder_list") or not main_window.folder_list:
        from SECQUOIA.gui.common.messages import show_folder_warning

        LOG.warning("Please first load CSV file and select folder")
        show_folder_warning(main_window)
        return

    item = main_window._find_tree_item_for(main_window.ident, None)
    if item is not None:
        apply_curation_status(item, CURATIONSTATUS.CURATION_CHECKED)
        main_window._set_inspected_for_ident(main_window.ident, 2)
        main_window._sync_children_visuals_from_df(item)

    LOG.info("%s is marked as checked.", main_window.ident)
    change_cell(main_window, "next")


def install_alt_plus_minus_brush_resize(
    main_window,
    *,
    step: int = 1,
    accel: int = 5,
    min_size: int = 1,
    max_size: int = 512,
    require_paint_or_erase: bool = True,
):
    """Install Alt+'+' / Alt+'-' shortcuts to grow/shrink the active brush."""

    def _focused_viewer():
        """Return the fluorescence viewer with canvas focus, else viewer_1."""
        for v in getattr(main_window, "viewer_fluorescence", []) or []:
            try:
                if v.window._qt_viewer.canvas.native.hasFocus():
                    return v
            except (RuntimeError, AttributeError, TypeError):
                pass
        return getattr(main_window, "viewer_1", None)

    def _active_labels(v):
        """Return the active Labels layer of viewer v, or None."""
        try:
            return v.layers.selection.active
        except (RuntimeError, AttributeError, TypeError):
            return None

    def _mode_name(layer):
        """Return the layer's current mode name."""
        m = getattr(layer, "mode", None)
        return (getattr(m, "name", m) or "").lower()

    def _set_status(v, text: str):
        """Set the viewer's status line, ignoring viewers that don't have one."""
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            v.status = text

    def _nudge(sign: int) -> bool:
        """Increase/decrease the active brush size. Returns False if it didn't apply."""
        v = _focused_viewer()
        if v is None:
            return False
        layer = _active_labels(v)
        if layer is None:
            return False
        if require_paint_or_erase and _mode_name(layer) not in (
            "paint",
            "erase",
        ):
            return False

        mods = QApplication.keyboardModifiers()
        k = step * (accel if (mods & Qt.ShiftModifier) else 1)

        cur = int(getattr(layer, "brush_size", 10) or 10)
        new_size = max(min_size, min(max_size, cur + sign * k))
        if new_size != cur:
            with contextlib.suppress(RuntimeError, AttributeError, TypeError):
                layer.brush_size = int(new_size)
            _set_status(v, f"Brush size: {new_size}")
        return True

    def _add(seq: str, sign: int):
        sc = QShortcut(QKeySequence(seq), main_window)
        sc.setContext(Qt.ApplicationShortcut)
        sc.activated.connect(lambda s=sign: _nudge(s))
        main_window._brush_shortcuts.append(sc)

    _add("Alt+=", +1)
    _add("Alt++", +1)
    _add("Alt+Plus", +1)
    _add("Alt+Add", +1)
    _add("Alt+-", -1)
    _add("Alt+Minus", -1)
    _add("Alt+Subtract", -1)

    main_window._nudge_brush_size = _nudge


def setup_key_bindings_curation(main_window) -> None:
    """Bind Left/Right on both viewers and log when keys are pressed."""

    def _bind_on_viewer(v, name):
        """Bind Left/Right keys on one viewer."""

        def go_left(viewer, event=None):
            change_time_point(main_window, -1)
            if event is not None:
                event.handled = True

        def go_right(viewer, event=None):
            change_time_point(main_window, +1)
            if event is not None:
                event.handled = True

        v.bind_key("Left", go_left, overwrite=True)
        v.bind_key("Right", go_right, overwrite=True)

        try:
            parent = v.window._qt_window

            QShortcut(QKeySequence("Left"), parent).activated.connect(
                lambda: (
                    LOG.debug("[Qt shortcut] LEFT on %s", name),
                    change_time_point(main_window, -1),
                )
            )
            QShortcut(QKeySequence("Right"), parent).activated.connect(
                lambda: (
                    LOG.debug("[Qt shortcut] RIGHT on %s", name),
                    change_time_point(main_window, +1),
                )
            )
        except (RuntimeError, AttributeError, TypeError) as e:
            LOG.warning("Qt shortcut fallback not installed: %s", e)

    for idx, attr in enumerate(("viewer_1", "viewer_2"), start=1):
        v = getattr(main_window, attr, None)
        if v is not None:
            _bind_on_viewer(v, f"viewer_{idx}")
    ensure_current_df_subset(main_window)
