"""Builds the Image & Movie Exporter window and its sidebar widgets."""

import logging
from contextlib import suppress

from qtpy.QtCore import (
    QRect,
    QRectF,
    QSize,
    Qt,
    QTimer,
)
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from SECQUOIA.config import EXPORT, LINKS, TOOLTIPSTEXT
from SECQUOIA.gui.common.ui_utils import make_help_button
from SECQUOIA.utils.timing import current_t_range as _current_t_range

LOG = logging.getLogger(__name__)


class _Window:
    """Builds the Image & Movie Exporter window and its sidebar widgets."""

    def open_image_movie_export_window(self, main_window=None):
        """Build and display the Image & Movie Exporter window and canvas."""
        main_window = main_window or self.main_window

        if not getattr(main_window, "folder_list", None):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Please first load CSV file and select a folder")
            msg.setWindowTitle("Folder Selection Error")
            msg.exec_()
            return

        main_window.gif_window = QWidget()
        main_window.gif_window.setStyleSheet("""
            QWidget#section_frame {
                border: 1px solid #9a9a9a;
                border-radius: 4px;
            }
            QLabel#section_title {
                color: #dddddd;
                font-weight: 600;
                padding-left: 2px;
                border: none;
                background: transparent;
            }
        """)

        main_window.gif_window.setWindowTitle("Image & Movie Exporter")
        root_v = QVBoxLayout(main_window.gif_window)
        root_v.setSizeConstraint(QLayout.SetMinimumSize)
        root_v.setSpacing(6)

        top_h = QHBoxLayout()
        top_h.setContentsMargins(0, 0, 0, 0)
        top_h.addStretch(1)

        main_window.help_btn = make_help_button(
            main_window.gif_window,
            TOOLTIPSTEXT.HELP_BTN,
            LINKS.GITHUB_EXPORTER,
        )
        top_h.addWidget(main_window.help_btn)
        root_v.addLayout(top_h)
        root_h = QHBoxLayout()
        root_v.addLayout(root_h, 1)

        # LEFT SIDEBAR
        sidebar = QVBoxLayout()
        sidebar.setContentsMargins(6, 6, 6, 6)
        sidebar.setSpacing(14)
        main_window.identification_input = QLineEdit()
        main_window.identification_input.setVisible(False)

        # Timeline min/max
        grp_time, box_time, gb = self._make_section("Time Window", QHBoxLayout)
        gb.setContentsMargins(10, 10, 10, 10)
        gb.setSpacing(10)
        gb.addWidget(QLabel("Start (t):"))
        main_window.start_t_input = QSpinBox()
        main_window.start_t_input.setToolTip(TOOLTIPSTEXT.START_T)
        gb.addWidget(main_window.start_t_input)
        gb.addWidget(QLabel("End (t):"))
        main_window.end_t_input = QSpinBox()
        main_window.end_t_input.setToolTip(TOOLTIPSTEXT.END_T)
        gb.addWidget(main_window.end_t_input)

        gb.addWidget(QLabel("Speed (fps):"))
        main_window.gif_speed_input = QSpinBox()
        main_window.gif_speed_input.setMinimum(
            max(1, getattr(EXPORT, "GIF_SPEED_MIN", 1))
        )
        main_window.gif_speed_input.setMaximum(
            getattr(EXPORT, "GIF_SPEED_MAX", 500)
        )
        main_window.gif_speed_input.setValue(
            getattr(EXPORT, "GIF_SPEED_DEFAULT", 20)
        )
        main_window.gif_speed_input.setToolTip(TOOLTIPSTEXT.GIF_SPEED)
        gb.addWidget(main_window.gif_speed_input)

        main_window.play_btn = QPushButton()
        main_window.play_btn.setToolTip(TOOLTIPSTEXT.PLAY_BUTTON)
        icn_play = self._qta_icon("fa5s.play", "mdi.play", "fa.play")
        if icn_play:
            main_window.play_btn.setIcon(icn_play)
            with suppress(Exception):
                main_window.play_btn.setIconSize(QSize(16, 16))
        else:
            main_window.play_btn.setText("Play")
        gb.addWidget(main_window.play_btn)
        gb.addStretch(1)
        sidebar.addWidget(grp_time)

        # Crop + Tile size
        grp_crop, box_crop, lc = self._make_section(
            "Crop Size (centered on cell)", QGridLayout
        )
        r = 0
        lc.addWidget(QLabel("Crop X (px):"), r, 0)
        main_window.crop_full_x_input = QSpinBox()
        main_window.crop_full_x_input.setRange(
            EXPORT.CROP_FULL_MIN, EXPORT.CROP_FULL_MAX
        )
        main_window.crop_full_x_input.setValue(40)
        main_window.crop_full_x_input.setToolTip(TOOLTIPSTEXT.CROP_W)
        lc.addWidget(main_window.crop_full_x_input, r, 1)
        lc.addWidget(QLabel("Crop Y (px):"), r, 2)
        main_window.crop_full_y_input = QSpinBox()
        main_window.crop_full_y_input.setRange(
            EXPORT.CROP_FULL_MIN, EXPORT.CROP_FULL_MAX
        )
        main_window.crop_full_y_input.setValue(40)
        main_window.crop_full_y_input.setToolTip(TOOLTIPSTEXT.CROP_H)
        lc.addWidget(main_window.crop_full_y_input, r, 3)
        r += 1
        lc.addWidget(QLabel("Tile W (px):"), r, 0)
        main_window.tile_w_input = QSpinBox()
        main_window.tile_w_input.setRange(EXPORT.TILE_MIN, EXPORT.TILE_MAX)
        main_window.tile_w_input.setValue(EXPORT.PREVIEW_W)
        main_window.tile_w_input.setToolTip(TOOLTIPSTEXT.TILE_W)
        lc.addWidget(main_window.tile_w_input, r, 1)
        lc.addWidget(QLabel("Tile H (px):"), r, 2)
        main_window.tile_h_input = QSpinBox()
        main_window.tile_h_input.setRange(EXPORT.TILE_MIN, EXPORT.TILE_MAX)
        main_window.tile_h_input.setValue(EXPORT.PREVIEW_H)
        main_window.tile_h_input.setToolTip(TOOLTIPSTEXT.TILE_H)
        lc.addWidget(main_window.tile_h_input, r, 3)
        sidebar.addWidget(grp_crop)

        # Selection
        grp_sel, box_sel, ls = self._make_section("Selection", QGridLayout)
        ls.setHorizontalSpacing(10)
        ls.setVerticalSpacing(6)
        grp_sel.setToolTip(TOOLTIPSTEXT.SELECTION_GROUP)
        ls.addWidget(QLabel("ID:"), 0, 0)
        main_window.sel_ident_combo = QComboBox()
        self._populate_ident_combo_for_panel(
            main_window, main_window.sel_ident_combo
        )
        main_window.sel_ident_combo.setToolTip(TOOLTIPSTEXT.IDENT_COMBO)
        ls.addWidget(main_window.sel_ident_combo, 0, 1)
        ls.addWidget(QLabel("CH:"), 0, 2)
        main_window.sel_chan_combo = QComboBox()
        self._populate_channel_combo_for_panel(
            main_window, main_window.sel_chan_combo
        )
        main_window.sel_chan_combo.setToolTip(TOOLTIPSTEXT.CHAN_COMBO)
        ls.addWidget(main_window.sel_chan_combo, 0, 3)
        main_window.sel_gen_btn = QPushButton("Generations")
        main_window.sel_gen_btn.setToolTip(TOOLTIPSTEXT.GEN_BUTTON)
        ls.addWidget(main_window.sel_gen_btn, 0, 4)

        ls.addWidget(QLabel("B:"), 1, 0)
        main_window.sel_black_spin = QSpinBox()
        main_window.sel_black_spin.setRange(EXPORT.BW_MIN, EXPORT.BW_MAX - 1)
        main_window.sel_black_spin.setValue(EXPORT.BW_DEFAULT_BLACK)
        main_window.sel_black_spin.setToolTip(TOOLTIPSTEXT.BLACK_SPIN)
        ls.addWidget(main_window.sel_black_spin, 1, 1)
        ls.addWidget(QLabel("W:"), 1, 2)
        main_window.sel_white_spin = QSpinBox()
        main_window.sel_white_spin.setRange(EXPORT.BW_MIN + 1, EXPORT.BW_MAX)
        main_window.sel_white_spin.setValue(EXPORT.BW_DEFAULT_WHITE)
        main_window.sel_white_spin.setToolTip(TOOLTIPSTEXT.WHITE_SPIN)
        ls.addWidget(main_window.sel_white_spin, 1, 3)
        main_window.bw_inspector_btn = QPushButton("B/W Inspector")
        main_window.bw_inspector_btn.setToolTip(
            TOOLTIPSTEXT.BW_INSPECTOR_BUTTON
        )
        ls.addWidget(main_window.bw_inspector_btn, 1, 4)
        sidebar.addWidget(grp_sel)

        # Advanced
        grp_adv, box_adv, la2 = self._make_section(
            "Advanced settings", QGridLayout
        )
        grp_adv.setToolTip(TOOLTIPSTEXT.ADV_GROUP)
        la2.setContentsMargins(10, 10, 10, 10)
        la2.setHorizontalSpacing(6)
        la2.setVerticalSpacing(4)
        for c in range(6):
            la2.setColumnStretch(c, 0)
        r = 0
        la2.addWidget(QLabel("M:"), r, 0, alignment=Qt.AlignLeft)
        main_window.adv_mask_combo = QComboBox()
        self._populate_mask_combo_for_panel(
            main_window, main_window.adv_mask_combo
        )
        self._compact_combo(main_window.adv_mask_combo)
        main_window.adv_mask_combo.setToolTip(TOOLTIPSTEXT.ADV_MASK_COMBO)
        la2.addWidget(main_window.adv_mask_combo, r, 1, alignment=Qt.AlignLeft)
        la2.addWidget(QLabel("M α:"), r, 2, alignment=Qt.AlignLeft)
        main_window.adv_mask_alpha_spin = QDoubleSpinBox()
        main_window.adv_mask_alpha_spin.setRange(0.0, 1.0)
        main_window.adv_mask_alpha_spin.setSingleStep(0.05)
        main_window.adv_mask_alpha_spin.setValue(0.35)
        main_window.adv_mask_alpha_spin.setAccelerated(True)
        main_window.adv_mask_alpha_spin.setToolTip(TOOLTIPSTEXT.ADV_MASK_ALPHA)
        self._compact_spin(main_window.adv_mask_alpha_spin)
        la2.addWidget(
            main_window.adv_mask_alpha_spin, r, 3, alignment=Qt.AlignLeft
        )
        r += 1
        la2.addWidget(QLabel("Select M-Mode:"), r, 0, alignment=Qt.AlignLeft)
        main_window.adv_mask_mode_combo = QComboBox()
        main_window.adv_mask_mode_combo.addItems(["Full", "Contour"])
        main_window.adv_mask_mode_combo.setCurrentIndex(0)
        main_window.adv_mask_mode_combo.setToolTip(TOOLTIPSTEXT.ADV_MASK_MODE)
        self._compact_combo(main_window.adv_mask_mode_combo)
        la2.addWidget(
            main_window.adv_mask_mode_combo, r, 1, alignment=Qt.AlignLeft
        )
        la2.addWidget(QLabel("Thickness:"), r, 2, alignment=Qt.AlignLeft)
        main_window.adv_contour_px_spin = QSpinBox()
        main_window.adv_contour_px_spin.setRange(1, 15)
        main_window.adv_contour_px_spin.setValue(1)
        main_window.adv_contour_px_spin.setAccelerated(True)
        main_window.adv_contour_px_spin.setToolTip(TOOLTIPSTEXT.ADV_CONTOUR_PX)
        self._compact_spin(main_window.adv_contour_px_spin)
        la2.addWidget(
            main_window.adv_contour_px_spin, r, 3, alignment=Qt.AlignLeft
        )
        la2.addWidget(QLabel("Color:"), r, 4, alignment=Qt.AlignLeft)
        main_window.adv_mask_color_btn = QPushButton()
        self._btn_color(main_window.adv_mask_color_btn, (255, 255, 255))
        main_window.adv_mask_color_btn.setFixedWidth(24)
        main_window.adv_mask_color_btn.setToolTip(TOOLTIPSTEXT.ADV_MASK_COLOR)
        la2.addWidget(
            main_window.adv_mask_color_btn, r, 5, alignment=Qt.AlignLeft
        )
        r += 1
        la2.addWidget(
            QLabel("Show Time Indicator:"), r, 0, alignment=Qt.AlignLeft
        )
        main_window.adv_time_bar_chk = QCheckBox()
        main_window.adv_time_bar_chk.setToolTip(TOOLTIPSTEXT.ADV_TIME_BAR_CHK)
        la2.addWidget(
            main_window.adv_time_bar_chk, r, 1, alignment=Qt.AlignLeft
        )
        la2.addWidget(QLabel("Size:"), r, 2, alignment=Qt.AlignLeft)
        main_window.adv_indicator_px_spin = QSpinBox()
        main_window.adv_indicator_px_spin.setRange(4, 40)
        main_window.adv_indicator_px_spin.setValue(10)
        main_window.adv_indicator_px_spin.setAccelerated(True)
        self._compact_spin(main_window.adv_indicator_px_spin)
        main_window.adv_indicator_px_spin.setToolTip(
            TOOLTIPSTEXT.ADV_INDICATOR_PX
        )
        la2.addWidget(
            main_window.adv_indicator_px_spin, r, 3, alignment=Qt.AlignLeft
        )
        la2.addWidget(QLabel("Color:"), r, 4, alignment=Qt.AlignLeft)
        main_window.adv_indicator_color_btn = QPushButton()
        self._btn_color(main_window.adv_indicator_color_btn, (255, 255, 255))
        main_window.adv_indicator_color_btn.setFixedWidth(24)
        main_window.adv_indicator_color_btn.setToolTip(
            TOOLTIPSTEXT.ADV_INDICATOR_COLOR
        )
        la2.addWidget(
            main_window.adv_indicator_color_btn, r, 5, alignment=Qt.AlignLeft
        )
        r += 1
        la2.addWidget(QLabel("Show Time:"), r, 0, alignment=Qt.AlignLeft)
        main_window.adv_time_text_chk = QCheckBox()
        main_window.adv_time_text_chk.setToolTip(
            TOOLTIPSTEXT.ADV_TIME_TEXT_CHK
        )
        la2.addWidget(
            main_window.adv_time_text_chk, r, 1, alignment=Qt.AlignLeft
        )
        la2.addWidget(QLabel("Size:"), r, 2, alignment=Qt.AlignLeft)
        main_window.adv_time_font_spin = QSpinBox()
        main_window.adv_time_font_spin.setRange(8, 72)
        main_window.adv_time_font_spin.setValue(14)
        main_window.adv_time_font_spin.setAccelerated(True)
        self._compact_spin(main_window.adv_time_font_spin)
        main_window.adv_time_font_spin.setToolTip(TOOLTIPSTEXT.ADV_TIME_FONT)
        la2.addWidget(
            main_window.adv_time_font_spin, r, 3, alignment=Qt.AlignLeft
        )
        la2.addWidget(QLabel("Color:"), r, 4, alignment=Qt.AlignLeft)
        main_window.adv_time_color_btn = QPushButton()
        self._btn_color(main_window.adv_time_color_btn, (255, 255, 255))
        main_window.adv_time_color_btn.setFixedWidth(24)
        main_window.adv_time_color_btn.setToolTip(TOOLTIPSTEXT.ADV_TIME_COLOR)
        la2.addWidget(
            main_window.adv_time_color_btn, r, 5, alignment=Qt.AlignLeft
        )
        r += 1
        la2.addWidget(QLabel("Show CH label:"), r, 0, alignment=Qt.AlignLeft)
        main_window.adv_show_ch_lbl_chk = QCheckBox()
        main_window.adv_show_ch_lbl_chk.setToolTip(
            TOOLTIPSTEXT.ADV_SHOW_CH_LBL
        )
        la2.addWidget(
            main_window.adv_show_ch_lbl_chk, r, 1, alignment=Qt.AlignLeft
        )
        la2.addWidget(QLabel("Size:"), r, 2, alignment=Qt.AlignLeft)
        main_window.adv_ch_font_spin = QSpinBox()
        main_window.adv_ch_font_spin.setRange(8, 72)
        main_window.adv_ch_font_spin.setValue(14)
        main_window.adv_ch_font_spin.setAccelerated(True)
        self._compact_spin(main_window.adv_ch_font_spin)
        main_window.adv_ch_font_spin.setToolTip(TOOLTIPSTEXT.ADV_CH_FONT)
        la2.addWidget(
            main_window.adv_ch_font_spin, r, 3, alignment=Qt.AlignLeft
        )
        la2.addWidget(QLabel("Color:"), r, 4, alignment=Qt.AlignLeft)
        main_window.adv_ch_color_btn = QPushButton()
        self._btn_color(main_window.adv_ch_color_btn, (255, 255, 255))
        main_window.adv_ch_color_btn.setFixedWidth(24)
        main_window.adv_ch_color_btn.setToolTip(TOOLTIPSTEXT.ADV_CH_COLOR)
        la2.addWidget(
            main_window.adv_ch_color_btn, r, 5, alignment=Qt.AlignLeft
        )
        r += 1
        la2.addWidget(QLabel("CH label text:"), r, 0, alignment=Qt.AlignLeft)
        main_window.adv_ch_label_edit = QLineEdit()
        main_window.adv_ch_label_edit.setPlaceholderText("e.g. GFP")
        main_window.adv_ch_label_edit.setFixedWidth(220)
        main_window.adv_ch_label_edit.setToolTip(
            TOOLTIPSTEXT.ADV_CH_LABEL_EDIT
        )
        la2.addWidget(
            main_window.adv_ch_label_edit, r, 1, 1, 3, alignment=Qt.AlignLeft
        )

        # Font label
        la2.addWidget(QLabel("Font:"), r, 4, alignment=Qt.AlignLeft)
        main_window.adv_font_combo = QComboBox()
        fonts = list(getattr(EXPORT, "FONT_NAMES", ()))
        if not fonts:
            fonts = ["Arial"]
        for name in fonts:
            main_window.adv_font_combo.addItem(name)
        default_name = getattr(EXPORT, "DEFAULT_FONT_NAME", fonts[0])
        idx = main_window.adv_font_combo.findText(default_name)
        if idx >= 0:
            main_window.adv_font_combo.setCurrentIndex(idx)

        self._compact_combo(main_window.adv_font_combo)
        main_window.adv_font_combo.setToolTip(TOOLTIPSTEXT.ADV_FONT_COMBO)
        la2.addWidget(main_window.adv_font_combo, r, 5, alignment=Qt.AlignLeft)
        sidebar.addWidget(grp_adv)

        # Export row
        grp_export, box_export, le = self._make_section(
            "Export Data", QHBoxLayout
        )
        grp_export.setToolTip(TOOLTIPSTEXT.EXPORT_GROUP)
        main_window.export_type_combo = QComboBox()
        main_window.export_type_combo.addItems(["Single image", "Animation"])
        main_window.export_type_combo.setToolTip(TOOLTIPSTEXT.EXPORT_TYPE)
        le.addWidget(main_window.export_type_combo)
        main_window.export_fmt_combo = QComboBox()
        main_window.export_fmt_combo.setToolTip(TOOLTIPSTEXT.EXPORT_FMT)
        le.addWidget(main_window.export_fmt_combo)
        le.addWidget(QLabel("Missing frames:"))
        main_window.missing_policy_combo = QComboBox()
        main_window.missing_policy_combo.addItems(
            ["Strict", "Hold last", "Nearest"]
        )
        main_window.missing_policy_combo.setToolTip(
            TOOLTIPSTEXT.EXPORT_MISSING
        )
        le.addWidget(main_window.missing_policy_combo)
        le.addWidget(QLabel("Grid spacing (px):"))
        main_window.export_gap_spin = QSpinBox()
        main_window.export_gap_spin.setRange(0, 200)
        main_window.export_gap_spin.setValue(EXPORT.EXPORT_GAP_DEFAULT)
        main_window.export_gap_spin.setToolTip(TOOLTIPSTEXT.EXPORT_GAP)
        le.addWidget(main_window.export_gap_spin)
        main_window.export_btn = QPushButton("Export")
        main_window.export_btn.setToolTip(TOOLTIPSTEXT.EXPORT_BTN)
        le.addWidget(main_window.export_btn)
        sidebar.addWidget(grp_export)

        # Presets
        grp_presets, box_presets, lp = self._make_section(
            "Save and Load settings", QGridLayout
        )
        grp_presets.setToolTip(TOOLTIPSTEXT.PRESETS_GROUP)
        lp.addWidget(QLabel("Name:"), 0, 0)
        main_window.preset_name_edit = QLineEdit()
        main_window.preset_name_edit.setPlaceholderText("e.g. fig2_gridA")
        main_window.preset_name_edit.setToolTip(TOOLTIPSTEXT.PRESET_NAME)
        lp.addWidget(main_window.preset_name_edit, 0, 1)
        main_window.preset_save_btn = QPushButton("Save")
        main_window.preset_save_btn.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        main_window.preset_save_btn.setToolTip(TOOLTIPSTEXT.PRESET_SAVE)
        lp.addWidget(main_window.preset_save_btn, 0, 2)
        lp.addWidget(QLabel("Presets:"), 1, 0)
        main_window.preset_list_combo = QComboBox()
        main_window.preset_list_combo.setToolTip(TOOLTIPSTEXT.PRESET_LIST)
        lp.addWidget(main_window.preset_list_combo, 1, 1)
        main_window.preset_load_btn = QPushButton("Load")
        main_window.preset_load_btn.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        main_window.preset_load_btn.setToolTip(TOOLTIPSTEXT.PRESET_LOAD)
        lp.addWidget(main_window.preset_load_btn, 1, 2)
        sidebar.addWidget(grp_presets)

        sidebar.addStretch(1)

        # RIGHT: Canvas
        right_v = QVBoxLayout()
        right_v.setSpacing(10)
        main_window.scene = QGraphicsScene()
        main_window.view = QGraphicsView(main_window.scene)
        main_window.view.setDragMode(QGraphicsView.RubberBandDrag)
        main_window.view.setViewportUpdateMode(
            QGraphicsView.SmartViewportUpdate
        )
        main_window.view.setTransformationAnchor(
            QGraphicsView.AnchorUnderMouse
        )
        main_window.view.setToolTip(TOOLTIPSTEXT.CANVAS_VIEW)
        with suppress(Exception):
            main_window.view.setCacheMode(QGraphicsView.CacheNone)

        canvas_frame = QFrame()
        canvas_frame.setObjectName("canvas_frame")
        canvas_frame.setFrameShape(QFrame.StyledPanel)
        canvas_v = QVBoxLayout(canvas_frame)
        canvas_v.setContentsMargins(6, 6, 6, 6)
        canvas_v.addWidget(main_window.view, 1)
        right_v.addWidget(canvas_frame, 1)
        main_window.gif_window.setStyleSheet(
            main_window.gif_window.styleSheet() + """
            #canvas_frame { border: 1px solid #9a9a9a; border-radius: 4px; }
        """
        )

        def _canvas_context_menu(pos):
            """Show the canvas context menu on empty space.

            Adds a tile at the nearest free grid cell; does nothing when
            an item is already under the cursor.
            """
            view = main_window.view
            scene_pos = view.mapToScene(pos)
            item_under = main_window.scene.itemAt(scene_pos, view.transform())
            if not item_under:
                menu = QMenu(view)
                act_add = menu.addAction("Add image")
                act_add.setToolTip(TOOLTIPSTEXT.CONTEXT_ADD_NEAR_CURSOR)
                act = menu.exec_(view.mapToGlobal(pos))
                if act is act_add:
                    self._add_panel(
                        main_window,
                        self._nearest_free_point(main_window, scene_pos),
                    )

        main_window.view.setContextMenuPolicy(Qt.CustomContextMenu)
        main_window.view.customContextMenuRequested.connect(
            _canvas_context_menu
        )

        left_wrap = QWidget()
        left_wrap.setLayout(sidebar)
        left_wrap.setMinimumWidth(600)
        right_wrap = QWidget()
        right_wrap.setLayout(right_v)
        root_h.addWidget(left_wrap, 0)
        root_h.addWidget(right_wrap, 1)

        main_window.panels = []
        main_window.active_panel = None

        # Timeline min/max
        lo, hi = 0, 9999
        try:
            t_file_min, t_file_max, _, _ = _current_t_range(main_window)
            n_frames = max(0, int(t_file_max) - int(t_file_min) + 1)
            hi = max(0, n_frames - 1)
        except (AttributeError, TypeError, ValueError):
            pass

        main_window.start_t_input.setRange(lo, hi)
        main_window.end_t_input.setRange(lo, hi)
        main_window.start_t_input.setValue(lo)
        main_window.end_t_input.setValue(hi)

        # first tile
        self._add_panel(main_window)
        main_window.gif_window.show()

        # global updates
        main_window.crop_full_x_input.valueChanged.connect(
            lambda *_: self._update_all_previews(main_window)
        )
        main_window.crop_full_y_input.valueChanged.connect(
            lambda *_: self._update_all_previews(main_window)
        )
        main_window.start_t_input.valueChanged.connect(
            lambda *_: self._update_all_previews(main_window)
        )
        main_window.end_t_input.valueChanged.connect(
            lambda *_: self._update_all_previews(main_window)
        )
        with suppress(Exception):
            main_window.time_input.valueChanged.connect(
                lambda *_: self._update_all_previews(main_window)
            )

        main_window.play_btn.clicked.connect(
            lambda *_: self._toggle_play(main_window)
        )
        main_window._preview_timer = QTimer(main_window.gif_window)
        main_window._preview_timer.timeout.connect(
            lambda *_: self._advance_frame(main_window)
        )
        main_window.gif_speed_input.valueChanged.connect(
            lambda *_: self._update_timer_from_speed(main_window)
        )
        main_window.gif_window.destroyed.connect(
            lambda *_: self._stop_play(main_window)
        )

        # Selection/advanced wiring
        self._wire_sidebar_to_active(main_window)
        self._wire_selection_to_active(main_window)

        # Tile size change wiring
        def _apply_tile_size():
            """Resize all preview tiles while preserving their current grid positions."""
            new_w = int(main_window.tile_w_input.value())
            new_h = int(main_window.tile_h_input.value())
            if new_w == EXPORT.PREVIEW_W and new_h == EXPORT.PREVIEW_H:
                return

            panel_cells = [
                (self._cell_from_point(p["proxy"].pos()), p)
                for p in getattr(main_window, "panels", [])
            ]

            EXPORT.PREVIEW_W, EXPORT.PREVIEW_H = new_w, new_h

            for cell, p in panel_cells:
                with suppress(Exception):
                    prx = p["proxy"]
                    cont = p["container"]
                    lbl = p["label"]

                    prx.prepareGeometryChange()
                    lbl.setFixedSize(new_w, new_h)
                    cont.setFixedSize(new_w, new_h)
                    cont.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                    cont.updateGeometry()

                    prx.resize(new_w, new_h)
                    prx.setMinimumSize(new_w, new_h)
                    prx.setMaximumSize(new_w, new_h)
                    prx.setPreferredSize(new_w, new_h)
                    prx.updateGeometry()

                    prx.setPos(self._point_from_cell(cell))

                    cont.update()
                    prx.update()

            self._refresh_scene_rect(main_window)
            with suppress(Exception):
                main_window.view.resetCachedContent()
                main_window.scene.invalidate(
                    main_window.scene.sceneRect(), QGraphicsScene.AllLayers
                )

            main_window.scene.update()
            main_window.view.viewport().update()

            self._update_all_previews(main_window)

            act = getattr(main_window, "active_panel", None)
            if act:
                self._highlight_panel(act, True)

        main_window.tile_w_input.valueChanged.connect(
            lambda *_: _apply_tile_size()
        )
        main_window.tile_h_input.valueChanged.connect(
            lambda *_: _apply_tile_size()
        )

        def _refresh_export_formats():
            """Update available file formats based on single-image vs animation export."""
            typ = main_window.export_type_combo.currentText()
            with self.blocked(main_window.export_fmt_combo):
                main_window.export_fmt_combo.clear()
                main_window.export_fmt_combo.addItems(
                    ["PNG", "TIF"]
                    if typ == "Single image"
                    else ["GIF", "MP4", "AVI", "TIF stack"]
                )
                main_window.export_fmt_combo.setCurrentIndex(0)

        _refresh_export_formats()
        main_window.export_type_combo.currentIndexChanged.connect(
            lambda *_: _refresh_export_formats()
        )

        def _do_export():
            """Dispatch the export action to single-image or animation export."""
            typ = main_window.export_type_combo.currentText()
            fmt = main_window.export_fmt_combo.currentText()

            if typ == "Single image":
                main_window.single_img_fmt_combo = main_window.export_fmt_combo
                self._export_single_images(main_window)
            elif fmt == "TIF stack":
                self._export_tiff_stack(main_window)
            else:
                main_window.anim_fmt_combo = main_window.export_fmt_combo
                self._export_animation_multi(main_window)

        main_window.export_btn.clicked.connect(lambda *_: _do_export())

        # B/W Inspector
        main_window.bw_inspector_btn.clicked.connect(
            lambda *_: self._open_bw_inspector(main_window)
        )

        # Presets wiring
        self._refresh_presets_list(main_window)
        main_window.preset_save_btn.clicked.connect(
            lambda *_: self._save_current_preset_clicked(main_window)
        )
        main_window.preset_load_btn.clicked.connect(
            lambda *_: self._load_selected_preset_clicked(main_window)
        )

        self._update_all_previews(main_window)
        self._refresh_scene_rect(main_window)
        main_window.gif_window.adjustSize()

    def _scene_items_bbox(self, main_window):
        """Compute bounding box of all panel tiles in the scene."""
        if not getattr(main_window, "panels", []):
            return QRect(0, 0, EXPORT.PREVIEW_W, EXPORT.PREVIEW_H)
        xs, ys, xe, ye = [], [], [], []
        for p in main_window.panels:
            pos = p["proxy"].pos()
            sz = p["proxy"].size()
            xs.append(pos.x())
            ys.append(pos.y())
            xe.append(pos.x() + sz.width())
            ye.append(pos.y() + sz.height())
        x_min, y_min = min(xs) - EXPORT.GRID_GAP, min(ys) - EXPORT.GRID_GAP
        x_max, y_max = max(xe) + EXPORT.GRID_GAP, max(ye) + EXPORT.GRID_GAP
        return QRect(
            int(x_min),
            int(y_min),
            int(max(1, x_max - x_min)),
            int(max(1, y_max - y_min)),
        )

    def _refresh_scene_rect(self, main_window):
        """Update QGraphicsScene rect to fit current content."""
        with suppress(Exception):
            main_window.scene.setSceneRect(
                QRectF(self._scene_items_bbox(main_window))
            )
