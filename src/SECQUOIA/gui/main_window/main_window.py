"""MainWindow class (menus, layouts)"""

import logging
import webbrowser
from functools import partial

import napari.layers.labels._labels_mouse_bindings  # noqa: F401
import pyqtgraph as pg
import qtawesome as qta
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import (
    QColor,
    QFont,
)
from qtpy.QtWidgets import (
    QAction,
    QButtonGroup,
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMenuBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)

from SECQUOIA.config import (
    LINKS,
    STYLE,
    TOOLTIPSTEXT,
    ColorStyle,
    TracksViewConfig,
)
from SECQUOIA.core.basic_correction import DataSetBaSiC
from SECQUOIA.gui.cell_inspector.integration import open_cell_inspector
from SECQUOIA.gui.dialogs.lineage_style_dialog import (
    open_lineage_style_dialog,
)
from SECQUOIA.gui.dialogs.plot_params_dialog import (
    open_plot_params_dialog,
)
from SECQUOIA.gui.loading.channel_mask_manager_dialog import (
    open_channel_mask_manager,
)
from SECQUOIA.gui.loading.experiment_loader_dialog import (
    open_experiment_loader_window,
)
from SECQUOIA.gui.loading.load_data import load_data_window
from SECQUOIA.gui.loading.loading_dialogs import (
    open_load_previous_project_gui,
)
from SECQUOIA.gui.main_window.curation_tree import CurationTree
from SECQUOIA.gui.main_window.dynamics_plot_grid import DynamicsPlotGrid
from SECQUOIA.gui.main_window.dynamics_plot_lineage_panel import (
    DynamicsPlotLineagePanel,
)
from SECQUOIA.gui.main_window.dynamics_plot_row_labels import (
    DynamicsPlotRowLabels,
)
from SECQUOIA.gui.main_window.dynamics_plot_row_widgets import (
    DynamicsPlotRowWidgets,
)
from SECQUOIA.gui.main_window.key_bindings import KeyBindings, cell_fate
from SECQUOIA.gui.main_window.labels_editing import LabelsEditing
from SECQUOIA.gui.main_window.layer_visibility import LayerVisibility
from SECQUOIA.gui.main_window.measurement_picking import MeasurementPicking
from SECQUOIA.gui.main_window.mouse_bindings import (
    MouseBindings,
    on_plot_single_click,
)
from SECQUOIA.gui.main_window.tracks_layer import TracksLayer
from SECQUOIA.gui.main_window.viewer_contrast import ViewerContrast
from SECQUOIA.gui.main_window.viewer_setup import ViewerSetup
from SECQUOIA.gui.main_window.viewer_toolbar import ViewerToolbar
from SECQUOIA.gui.main_window.viewer_tools import ViewerTools
from SECQUOIA.gui.main_window.window_lifecycle import WindowLifecycle
from SECQUOIA.gui.outlier import (
    open_outlier_detection_window,
    show_current_outlier_parameters,
)
from SECQUOIA.gui.outlier.navigation import change_outlier
from SECQUOIA.gui.outlier.reset import reset_outlier_state
from SECQUOIA.gui.position_navigation import (
    load_position,
    open_position_window,
)
from SECQUOIA.gui.track_selection import handle_item_click

LOG = logging.getLogger(__name__)


class MainWindow(
    KeyBindings,
    MouseBindings,
    CurationTree,
    LabelsEditing,
    MeasurementPicking,
    TracksLayer,
    LayerVisibility,
    WindowLifecycle,
    ViewerSetup,
    ViewerContrast,
    ViewerTools,
    ViewerToolbar,
    DynamicsPlotGrid,
    DynamicsPlotRowWidgets,
    DynamicsPlotLineagePanel,
    DynamicsPlotRowLabels,
    QWidget,
):
    """Main application window for SECQUOIA."""

    _SUMMARY_COLORS = (
        ColorStyle.as_dict()
    )  # Colors used for summary displays in the UI

    def __init__(self) -> None:
        """Initialize the main window, state, menus, viewers, and layouts."""
        super().__init__()
        self._init_state()
        self._build_ui()

    def _init_state(self) -> None:
        """Initialize instance state: dataframes, selection/position state, viewer refs, and UI attribute placeholders."""
        if not hasattr(self, "n_channels"):
            self.track_df = (
                None  # pd.dataframe of all information of each position
            )
            self.filtered_df = (
                None  # pd.dataframe track_df filtered on current position
            )
            self.df_subset = (
                None  # pd.dataframe filtered_df filtered on current ID
            )

            self.unique_ids = (
                None  # list of unique IDs in the current position
            )
            self.unique_outliers_ids = (
                None  # list of unique outlier IDs in the current position
            )

            # Selection parameters
            self.current_ident_index = 0  # index in the list of unique IDs
            self.ident = None  # current selected ID
            self.current_time_index = 0  # current time point column (t) index
            self.current_TrackNumber_plot = (
                None  # current TrackNumber in the plot
            )
            self.current_position_index = 0  # 0-based, index into main_window.position_folders (zero-based).
            self.current_outlier_index = (
                0  # index in the list of unique outlier IDs
            )
            self.current_position_number = (
                None  # 1-based, position number parsed from folder name.
            )
            self.threshold = (
                None  # Minimal distance between tracking point and mask.
            )
            self.time_interval = None  # Time interval in seconds between frames, cached from dt_seconds
            self.min_mask_size = None  # Minimum size in pixels for a mask to be considered valid
            self.Time_input = None  # Time interval between frames in seconds
            self.time_min_selected = None  # User-selected minimum time point
            self.time_max_selected = None  # User-selected maximum time point
            self.t_max_detected = 0  # Maximum time point detected in files

            self.position_min = 1  # Minimum position number in experiment
            self.position_max = 1  # Maximum position number in experiment
            self.position_indices = []  # List of available position indices
            self.position_min_selected = None  # User-selected minimum position
            self.position_max_selected = None  # User-selected maximum position
            self.position_start_selected = (
                None  # Starting position for curation
            )
            self.folder_list = []  # List of folders in experiment directory

            self.user = ""  # User name for curation
            self.folder = None  # Experiment folder path
            self.position_folders = []  # list[str] — per position folder names
            self.tracking_path = None  # path to the tracking CSV file
            self.xml_path = None  # path to the XML file
            self.clt_parser = None  # clt_io.CLTParser class
            self.segmentation_paths = (
                []
            )  # list[str] — segmentation folder(s) chosen in Analysis.
            self._memmap_dir = None  # Temporary directory path used to store memory-mapped files (numpy.memmap) for image data.
            self._memmap_files = None  # List of file paths for memory-mapped (numpy.memmap) raw image stacks.
            self.position_selection = (
                None  # Path of the currently selected position folder
            )

            # Background correction settings
            self.use_background_correction = (
                False  # Flag to enable/disable background correction
            )
            self.background_correction_path = (
                None  # Path to BaSiC correction folder
            )
            self.corrected_images = (
                None  # List of corrected image stacks after BaSiC
            )
            self._corrected_u8_memmap_files = (
                []
            )  # List of uint8 memmap files after correction

            self.experiment_name = None  # Experiment name
            self.FL_inputs = (
                []
            )  # list of read-only QLineEdit widgets whose .text() returns the channel name
            self.basic = DataSetBaSiC(flag=False)
            # Image files
            self.images = (
                {}
            )  # dict[str, np.ndarray or memmap] per channel image stacks
            self.labels = []  # list of segmentation
            self.image_present = {}
            self._brush_shortcuts = (
                []
            )  # List of QShortcut objects installed for Alt+Plus/Alt+Minus brush-size adjustments.
            self._time_link_registry = {}  # Registry of time sync callbacks
            self._cam_link_cbs = []  # List of camera sync callbacks
            self._viewer_minimized = [
                False,
                False,
            ]  # Minimized state per viewer
            self._global_hotkeys_installed = (
                False  #  Global hotkeys are installed
            )

            self._edit_cache = {}  # Cache for labels layer edits
            self.last_edit_info = None  # Info about last labels edit
            self._last_labels_layer_ref = (
                None  # Weakref to last edited labels layer
            )
            self._last_labels_layer_name = (
                None  # Name of last edited labels layer
            )
            self._last_labels_layer_t = None  # Time point of last labels edit

            self._corrected_slices = (
                set()
            )  # Set of (mask_idx, t_idx) tuples tracking which mask slices were edited

            self.n_channels = STYLE.CHANNEL_NUMBER  # Number of channels
            self.n_masks = STYLE.MASK_NUMBER  # Number of segmentations
            self.cp_tracking = (
                False  # Cross-position CLT parsing disabled by default
            )
            self.ids_channels: list[str] = []
            self.available_channels = (
                []
            )  # List of detected channel names (w01, w02, etc)
            self.import_rt_df = None  # DataFrame with imported real time data
            self.use_import_rt = False  # Flag to enable real time import
            self.import_rt_path = None  # Path to real time CSV file
            self.dt_seconds = None  # Time delta between frames in seconds
            self._plot_params = {}  # Plot appearance parameters
            self._derived_features = (
                {}
            )  # Dictionary of computed derived features
            self.last_run_config = (
                {}
            )  # Last used configuration for derived metrics
            self.selected_feature_by_row = {}  # Selected features per plot row

            self.channel_comment_map = (
                {}
            )  # Map of channel names to comments from XML
            self.position_comment_map = (
                {}
            )  # Map of position numbers to comments from XML

            self.gif_window = None  # Main export dialog window
            self.help_popup = None  # Reference to a help popup window

            self.fuse_time_spin = (
                None  # QSpinBox for selecting the fuse/split time point
            )
            self.fuse_tree_id_field_1 = (
                None  # QLineEdit for first Identification input in Fuse dialog
            )
            self.fuse_tree_id_field_2 = None  # QLineEdit for second Identification input in Fuse dialog
            self.fuse_cell_field_1 = None  # QLineEdit for TrackNumber/cell input for first ID in Fuse dialog
            self.fuse_cell_field_2 = None  # QLineEdit for TrackNumber/cell input for second ID in Fuse dialog
            self.fuse_dialog = (
                None  # Dialog widget used for fusing lineage trees
            )
            self._fuse_next_slot = (
                0  # Internal counter/state used when reusing fuse dialog slots
            )

            self.mask_no_window = (
                None  # Warning window that no masks are loaded
            )
            self._outlier_window_ref = (
                None  # Reference to outlier detection window
            )
            self._lineage_style_dialog = (
                None  # Reference to lineage style dialog
            )
            self._tabs = None  # Reference to tab widget in load data dialog
            self._loading_in_progress = False  # Boolean re-entrancy guard set while a load operation is running.
            self._viewers_dirty = False  # True while the napari viewers hold real experiment layers; False when they hold only the empty placeholders (or do not exist yet).
            self._experiment_reset_done = True  # True when the app state is fresh (from __init__ or from a reset), so the Run click can skip _reset_loaded_experiment.
            self.viewer_1 = None  # First napari viewer instance
            self.viewer_2 = None  # Second napari viewer instance
            self.viewer_fluorescence = (
                []
            )  # List containing both viewer instances
            self.viewer_fluorescence_windows = (
                []
            )  # List of Qt window widgets for viewers
            self.viewer_wrappers = (
                []
            )  # List of wrapper widgets containing viewers + controls
            self.mask_combos = (
                []
            )  # List of mask selection combo boxes per viewer
            self.channel_combos = (
                []
            )  # List of channel selection combo boxes per viewer
            self.contrast_widgets = (
                []
            )  # List of contrast control widgets per viewer
            self.tool_strips = (
                []
            )  # List of tool strips (paint/erase/pan) per viewer
            self.opacity_widgets = []  # List of opacity sliders per viewer
            self.lineage_tools = {}  # Dict of lineage tree control widgets
            self.plot_grid_container = None  # Container widget for plot grid
            self._plot_grid_layout = None  # Grid layout managing plots
            self.lineage_container = None  # Container for lineage view
            self.graph3_widget = None  # Widget containing lineage plot
            self.graph3_view = (
                None  # LineageTreeView reused across ID switches
            )
            self._lineage_placeholder = (
                None  # Placeholder plot widget for lineage
            )
            self.plots_and_lineage_splitter = (
                None  # Splitter between plots and lineage
            )
            self.row_tools = {}  # Dict of plot row control widgets
            self.ALL = None  # QCheckBox for selecting all tracks
            self._last_outlier_rules = (
                None  # Dictionary storing last used outlier detection rules
            )
            self._sliding_rows_widgets = (
                []
            )  # List of widget dictionaries for multi-row sliding window UI
            self.sliding_window_params = (
                {}
            )  # Dictionary of widgets for single sliding-window UI
            self.Outliers = None  # QCheckBox for toggling outlier display
            self.tree_widget = None  # QTreeWidget showing outlier hierarchy
            self.right_layout = (
                None  # QLayout on the right side that holds plot widgets
            )
            self._feature_defs = (
                {}
            )  # Discovered feature definitions / templates from dataframe columns
            self._features_defaulted = False  # One-shot guard that prevents re-defaulting features on first population
            self.selected_m_by_channel = {}  # Selected mask index per plot-row
            self.selected_ch_by_channel = (
                {}
            )  # Selected channel index per plot-row
            self._shared_time_xmax = (
                1.0  # Shared x-axis max (time) used to align plots and lineage
            )
            self._track_colors = (
                {}
            )  # Per ident mapping of TrackNumber -> pg.Color for plotting
            self._max_plot_rows = None  # Maximum number of plot rows to render
            self._dynamics_plot_row_count = (
                None  # User-controlled number of dynamics plot rows.
            )
            self._dynamics_plot_rebuild_pending = (
                False  # Prevents scheduling duplicate dynamics plot rebuilds.
            )
            self._dynamics_plot_rebuild_running = False  # Prevents rebuilding again while plot widgets are being recreated.
            self._selected_tracks = (
                set()
            )  # Set of selected TrackNumber ints used to limit which tracks are plotted
            self._zoom_visible_tracks = (
                set()
            )  # Set of TrackNumber ints allowed by a "zoom" filter for plotting

            # Highlight & Color State
            self._hl_active_color = QColor(
                76, 175, 80
            )  # Current highlight color
            self._track_highlight_colors = {}  # Per track highlight colors
            self._hl_palette = {}  # Available highlight colors
            self._lineage_mode = "T"  # Current lineage view mode
            self._lineage_collapsed = (
                False  # Whether lineage view is collapsed
            )
            self.current_time_markers = (
                {}
            )  # Dict of vertical time-marker items per plot row {row: {"item": line, "plot_widget": pw}}
            self._lineage_time_line = None  # Plot widget InfiniteLine instance for the lineage/time plot
            self._hl_palette_order = (
                None  # Cached ordered list of highlight colors for cycling
            )
            self._hl_palette_index = (
                0  # Current index in the highlight palette order
            )

            # Overview layout attributes
            self._last_lineage_y_map = None  # A mapping of track numbers to their Y positions in the lineage tree.
            self._in_time_change_cb = (
                False  # Re-entrancy guard while handling time-change callbacks
            )
            self._last_zoom_ident = None  # Last identification processed for zoom level persistence across time points
            self.status_label = (
                None  # QLabel for displaying save/load status messages
            )
            self.seg_layers_by_viewer = None  # Dictionary mapping viewer indices (0,1) to their segmentation layers mapping {mask_idx: layer_name}. Used to track which layer name corresponds to which mask index in each viewer.
            self.canvas = None  # Graph plot interface
            self._lineage_y_map = (
                {}
            )  # Dictionary mapping track numbers to Y-axis positions in the lineage tree
            tracks_cfg_dict = getattr(self, "app_config", {}).get(
                "tracks_view", {}
            )
            self.tracks_cfg = TracksViewConfig.from_dict(
                tracks_cfg_dict
            )  # Configuration for tracks visualization.

            self._outlier_marker_items = (
                {}
            )  # Dictionary storing plot marker items that indicate outlier time points with orange stars.
            self.jump_channel = None  # Selected channel for jump within channel to next time point

    def _build_ui(self) -> None:
        """Build the menu bar, tree/fate-button panel, napari viewers, and assemble the main layout."""
        layout = QVBoxLayout()
        self.layout2 = QVBoxLayout()

        # Define Font Size
        font = QFont()
        font.setPointSize(STYLE.FONT_SIZE)
        font.setBold(True)
        font2 = QFont(STYLE.FONT_FAMILY, STYLE.FONT_SIZE)

        # Add plots
        plot_widget_1 = pg.PlotWidget()
        plot_widget_1.hideAxis("left")
        plot_widget_1.hideAxis("bottom")
        plot_widget_1.hideAxis("right")
        plot_widget_1.hideAxis("top")
        plot_widget_1.setBackground("black")
        plot_widget_1.setMouseEnabled(x=False, y=False)

        # Cell history layout, Initialize an empty plot
        self.graph3_layout = QVBoxLayout()
        self.graph3_layout.setContentsMargins(0, 0, 0, 0)
        self.graph3_layout.setSpacing(0)
        self.graph3_widget = QWidget()
        self.graph3_widget.setLayout(self.graph3_layout)
        self.graph3_widget.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.init_empty_plot()

        # Define Cell fate
        description_label8 = QLabel("Define cell fate")
        description_label8.setFont(font)
        self.btn_healthy = QPushButton("Healthy")
        self.btn_healthy.setToolTip(TOOLTIPSTEXT.HEALTHY)
        self.btn_healthy.setFont(font2)

        self.btn_dead = QPushButton("Dead")
        self.btn_dead.setFont(font2)
        self.btn_dead.setToolTip(TOOLTIPSTEXT.DEAD)

        self.btn_lost = QPushButton("Lost")
        self.btn_lost.setFont(font2)
        self.btn_lost.setToolTip(TOOLTIPSTEXT.LOST)

        self.btn_oof = QPushButton("Focus")
        self.btn_oof.setFont(font2)
        self.btn_oof.setToolTip(TOOLTIPSTEXT.FOCUS)

        self.btn_outlier = QPushButton("Outlier")
        self.btn_outlier.setFont(font2)
        self.btn_outlier.setToolTip(TOOLTIPSTEXT.OUTLIER)

        self.btn_lowsignal = QPushButton("Signal")
        self.btn_lowsignal.setFont(font2)
        self.btn_lowsignal.setToolTip(TOOLTIPSTEXT.SIGNAL)

        hbox = QHBoxLayout()
        hbox.setSpacing(4)
        hbox.setContentsMargins(2, 2, 2, 2)

        hbox.addWidget(self.btn_healthy, 1)
        hbox.addWidget(self.btn_dead, 1)
        hbox.addWidget(self.btn_lost, 1)
        hbox.addWidget(self.btn_oof, 1)
        hbox.addWidget(self.btn_outlier, 1)
        hbox.addWidget(self.btn_lowsignal, 1)

        # Connect buttons
        self.btn_healthy.clicked.connect(partial(cell_fate, self, "Healthy"))
        self.btn_dead.clicked.connect(partial(cell_fate, self, "Dead"))
        self.btn_lost.clicked.connect(partial(cell_fate, self, "Lost"))
        self.btn_oof.clicked.connect(partial(cell_fate, self, "OoF"))
        self.btn_outlier.clicked.connect(partial(cell_fate, self, "Outlier"))
        self.btn_lowsignal.clicked.connect(
            partial(cell_fate, self, "LowSignal")
        )

        for b in (
            self.btn_healthy,
            self.btn_dead,
            self.btn_lost,
            self.btn_oof,
            self.btn_outlier,
            self.btn_lowsignal,
        ):
            b.setFocusPolicy(Qt.NoFocus)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.setMinimumHeight(24)
            b.setStyleSheet(f"""
                QPushButton {{
                    padding: 2px 10px;
                    border-radius: 6px;
                    background: #2b2b2b;
                    color: #e6e6e6;
                    border: 1px solid #444;
                    font-size: {STYLE.FONT_SIZE}pt;
                }}
                QPushButton:hover {{
                    background: #3a3a3a;
                }}
                QPushButton:pressed {{
                    background: #1f1f1f;
                }}
                QPushButton:disabled {{
                    background: #1e1e1e;
                    color: #8a8a8a;
                    border-color: #333;
                }}
            """)

        # Connect the click event to a function
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabel("Tree-ID")
        self.tree_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.tree_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.tree_widget.verticalScrollBar().setStyleSheet("""
            QScrollBar:vertical {
                background: #2b2b2b;
                width: 12px;
                margin: 0px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background: #7a7a7a;
                min-height: 24px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #9a9a9a;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
                background: none;
                border: none;
            }
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: none;
            }
        """)
        self._tree_viewport = self.tree_widget.viewport()
        self._tree_viewport.installEventFilter(self)

        header = self.tree_widget.headerItem()
        font = QFont()
        font.setBold(True)
        header.setFont(0, font)
        self.tree_widget.itemClicked.connect(partial(handle_item_click, self))
        plot_widget_1.scene().sigMouseClicked.connect(
            partial(on_plot_single_click, self)
        )
        # Context menu on tree items
        self.tree_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_widget.customContextMenuRequested.connect(
            self.on_tree_context_menu
        )

        # Initialize napari viewer
        self.create_napari_viewers()

        # Set up the layout
        form_layout = QFormLayout()
        self.setStyleSheet(
            f"QLabel {{ font-size: {STYLE.FONT_SIZE}pt; }} QLineEdit {{ font-size: {STYLE.FONT_SIZE}pt; }}"
        )

        left_layout = QVBoxLayout()
        top_bar = QWidget()
        cbox = QHBoxLayout(top_bar)
        cbox.setContentsMargins(0, 0, 0, 0)
        cbox.setSpacing(20)

        # ALL Tree IDs
        all_widget = QWidget(top_bar)
        all_widget.setToolTip(TOOLTIPSTEXT.ALL)
        all_layout = QVBoxLayout(all_widget)
        all_layout.setContentsMargins(0, 0, 0, 0)
        all_layout.setSpacing(2)
        all_label = QLabel("All")
        all_label.setAlignment(Qt.AlignHCenter)
        self.ALL = QCheckBox()
        all_layout.addWidget(all_label)
        all_layout.addWidget(self.ALL, alignment=Qt.AlignHCenter)

        # OUTLIER ID List
        out_widget = QWidget(top_bar)
        out_widget.setToolTip(TOOLTIPSTEXT.OUT)
        out_layout = QVBoxLayout(out_widget)
        out_layout.setContentsMargins(0, 0, 0, 0)
        out_layout.setSpacing(2)
        out_label = QLabel("Out")
        out_label.setAlignment(Qt.AlignHCenter)
        self.Outliers = QCheckBox()
        out_layout.addWidget(out_label)
        out_layout.addWidget(self.Outliers, alignment=Qt.AlignHCenter)

        cb_group = QButtonGroup(top_bar)
        cb_group.setExclusive(True)
        cb_group.addButton(self.ALL, 0)
        cb_group.addButton(self.Outliers, 1)

        self.ALL.setChecked(True)
        cb_group.buttonToggled.connect(self.on_checkbox_toggled)

        cbox.addWidget(all_widget)
        cbox.addWidget(out_widget)
        cbox.addStretch()

        # Create save button
        self.save_button = QPushButton()
        self.save_button.setIcon(qta.icon("fa5s.save", color="white"))
        self.save_button.setToolTip(TOOLTIPSTEXT.SAVE)
        self.save_button.clicked.connect(self.on_save_clicked)
        left_layout.addWidget(top_bar)
        left_layout.addWidget(
            self.tree_widget,
        )
        left_layout.addWidget(self.save_button)

        self.tree_widget.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        top_bar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.tree_widget.header().setStretchLastSection(False)
        self.tree_widget.header().setSectionResizeMode(0, QHeaderView.Stretch)

        # Right side
        self.right_layout = QVBoxLayout()
        self.right_layout.addWidget(plot_widget_1)
        self.right_layout.addWidget(self.graph3_widget)

        # Assemble main layout
        graph_layout = QHBoxLayout()

        left_container = QWidget()
        left_container.setLayout(left_layout)

        left_container.setFixedWidth(120)
        left_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        graph_layout.addWidget(left_container, 0)
        graph_layout.addLayout(self.right_layout, 1)

        layout.addLayout(graph_layout)

        ##Create the main layout
        self.main_layout = QVBoxLayout()

        ##File menu
        menu_bar = QMenuBar(self)
        menu_bar.setNativeMenuBar(False)

        menu_bar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        f = menu_bar.font()
        f.setPointSize(STYLE.FONT_SIZE)
        menu_bar.setFont(f)
        self.main_layout.setMenuBar(menu_bar)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )

        left_widget = QWidget()
        left_widget.setLayout(layout)

        right_widget = QWidget()
        right_widget.setLayout(self.layout2)

        # Add both widgets to splitter
        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(right_widget)

        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 3)

        # Splitter style
        self.splitter.setStyleSheet("""
        QSplitter::handle:horizontal {
            background: #444;      /* dark handle */
            width: 12px;           /* thickness */
        }
        QSplitter::handle:horizontal:pressed {
            background: #666;
        }
        """)
        # Add splitter to your main layout
        self.main_layout.addWidget(self.splitter, 1)
        self.fate_bar = hbox
        self.fate_bar_container = QWidget()
        self.fate_bar_container.setLayout(hbox)
        self.layout2.addWidget(self.fate_bar_container)
        self.add_viewers_to_layout2()

        # Create the File menu overview
        file_menu = menu_bar.addMenu("File")
        view_menu = menu_bar.addMenu("View")
        position_menu = menu_bar.addMenu("Switch Position")
        Outlier_menu = menu_bar.addMenu("Outlier Detection")
        export_menu = menu_bar.addMenu("Export Data")
        help_menu = menu_bar.addMenu("Help")

        # File
        # Load data
        Load_data_action = QAction("Start a new Project (Ctrl+N)", self)
        Load_data_action.triggered.connect(partial(load_data_window, self))
        file_menu.addAction(Load_data_action)

        Open_data_action = QAction("Open a Project (Ctrl+O)", self)
        Open_data_action.triggered.connect(
            partial(open_load_previous_project_gui, self)
        )
        file_menu.addAction(Open_data_action)

        cytometric_analysis = QAction("Cytometric Analysis (Ctrl+F)", self)
        cytometric_analysis.triggered.connect(
            partial(open_experiment_loader_window, self)
        )
        file_menu.addAction(cytometric_analysis)

        tTt_action = QAction("tTt Data Format Transformer (Ctrl+T)", self)
        tTt_action.setToolTip(TOOLTIPSTEXT.MENUBAR_tTt)
        tTt_action.triggered.connect(self.open_ttt_dataformat_transformer)
        file_menu.addAction(tTt_action)

        Save_data_action = QAction("Save data (Ctrl+S)", self)
        Save_data_action.triggered.connect(self.on_save_clicked)
        file_menu.addAction(Save_data_action)

        # Exit
        exit_action = QAction("Exit (Esc)", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Load Positions
        previous_position_action = QAction(
            "Previous Position (Ctrl+Shift+Left)", self
        )
        previous_position_action.triggered.connect(
            partial(load_position, self, "previous")
        )
        position_menu.addAction(previous_position_action)

        next_position_action = QAction(
            "Next Position (Ctrl+Shift+Right)", self
        )
        next_position_action.triggered.connect(
            partial(load_position, self, "next")
        )
        position_menu.addAction(next_position_action)

        select_position_action = QAction(
            "Select Position (Ctrl+Shift+O)", self
        )
        select_position_action.triggered.connect(
            partial(open_position_window, self)
        )
        position_menu.addAction(select_position_action)

        # Time Marker
        time_marker_menu = QMenu("Time Marker", self)
        view_menu.addMenu(time_marker_menu)

        self.show_time_marker = True

        self.action_show_time_marker = QAction(
            "Show Time Marker (Ctrl+Shift+T)", self, checkable=True
        )
        self.action_show_time_marker.setChecked(True)
        self.action_show_time_marker.setStatusTip(
            "Toggle visibility of the green time marker on all plots and lineage view"
        )

        self.action_show_time_marker.toggled.connect(
            self.on_toggle_time_marker
        )

        time_marker_menu.addAction(self.action_show_time_marker)

        # Tracking submenu
        tracking_menu = QMenu("Tracking", self)
        view_menu.addMenu(tracking_menu)

        # Tracking bar
        self.action_tracking_bar = QAction(
            "Tracking bar (Ctrl+Shift+B)", self, checkable=True
        )
        self.action_tracking_bar.setChecked(False)
        self.action_tracking_bar.toggled.connect(self._toggle_tracking)
        tracking_menu.addAction(self.action_tracking_bar)

        # Show Tracks
        self.action_tracks_visible = QAction(
            "Show Tracks (Ctrl+Shift+K)", self, checkable=True
        )
        self.action_tracks_visible.setChecked(False)
        cfg = getattr(self, "tracks_cfg", None) or TracksViewConfig()
        self.action_tracks_visible.toggled.connect(
            self.on_toggle_tracks_visible
        )
        tracking_menu.addAction(self.action_tracks_visible)

        # Show IDs
        self.action_tracks_ids = QAction(
            "Show IDs (Ctrl+Shift+I)", self, checkable=True
        )
        self.action_tracks_ids.setChecked(bool(cfg.show_track_ids))
        self.action_tracks_ids.toggled.connect(self.on_toggle_tracks_ids)
        tracking_menu.addAction(self.action_tracks_ids)

        # Dynamics plot submenu
        dynamics_plots_menu = QMenu("Dynamics plots", self)
        view_menu.addMenu(dynamics_plots_menu)
        self.action_add_dynamics_plot = QAction(
            "Add Dynamics plot (Ctrl+Shift+=)", self
        )
        self.action_add_dynamics_plot.triggered.connect(
            self.add_dynamics_plot_row
        )
        dynamics_plots_menu.addAction(self.action_add_dynamics_plot)
        self.action_remove_dynamics_plot = QAction(
            "Remove Dynamics plot (Ctrl+Shift+-)", self
        )
        self.action_remove_dynamics_plot.triggered.connect(
            self.remove_dynamics_plot_row
        )
        dynamics_plots_menu.addAction(self.action_remove_dynamics_plot)
        self._sync_dynamics_plot_actions(busy=False)

        # Plot parameters
        plot_params_action = QAction("Plot parameters (Ctrl+Shift+D)", self)
        plot_params_action.triggered.connect(
            lambda: open_plot_params_dialog(self)
        )
        view_menu.addAction(plot_params_action)

        # Lineage / Heat Tree Appearance
        plot_params_action = QAction("Lineage parameters(Ctrl+Shift+L)", self)
        plot_params_action.triggered.connect(
            lambda: open_lineage_style_dialog(self)
        )
        view_menu.addAction(plot_params_action)

        # Cell inspector
        cell_inspector_action = QAction("Cell Inspector (Ctrl+I)", self)
        cell_inspector_action.triggered.connect(
            partial(open_cell_inspector, self)
        )
        view_menu.addAction(cell_inspector_action)

        # Channel / Mask manager
        channel_mask_manager_action = QAction(
            "Channel/Segmentation manager(Ctrl+Shift+C)", self
        )
        channel_mask_manager_action.triggered.connect(
            partial(open_channel_mask_manager, self)
        )
        view_menu.addAction(channel_mask_manager_action)

        # Outlier menu
        Outlier_action = QAction("Outlier Detection (Ctrl+H)", self)
        Outlier_action.triggered.connect(
            partial(open_outlier_detection_window, self)
        )
        Outlier_menu.addAction(Outlier_action)

        Next_Outlier_action = QAction("Jump to next Outlier (Ctrl+Down)", self)
        Next_Outlier_action.triggered.connect(
            partial(change_outlier, self, "next")
        )
        Outlier_menu.addAction(Next_Outlier_action)

        Previous_Outlier_action = QAction(
            "Jump to previous Outlier (Ctrl+Up)", self
        )
        Previous_Outlier_action.triggered.connect(
            partial(change_outlier, self, "previous")
        )
        Outlier_menu.addAction(Previous_Outlier_action)

        Current_Params_action = QAction(
            "Current Parameters (Ctrl+Shift+H)", self
        )
        Current_Params_action.triggered.connect(
            partial(show_current_outlier_parameters, self)
        )
        Outlier_menu.addAction(Current_Params_action)

        Reset_Outlier = QAction("Reset Outliers (Ctrl+R)", self)
        Reset_Outlier.triggered.connect(partial(reset_outlier_state, self))
        Outlier_menu.addAction(Reset_Outlier)

        Export_GIFs_Image_action = QAction(
            "Export Single Images or Movies (Ctrl+E)", self
        )
        Export_GIFs_Image_action.triggered.connect(
            self.open_image_movie_exporter
        )

        export_menu.addAction(Export_GIFs_Image_action)
        # Help
        github_page = QAction("Open Github page", self)
        help_menu.addAction(github_page)
        github_page.triggered.connect(
            lambda: webbrowser.open(LINKS.GITHUB_HELP)
        )

        hotkeys_overview = QAction("Hotkeys & Mouse (F1)", self)
        help_menu.addAction(hotkeys_overview)
        hotkeys_overview.triggered.connect(self.show_help_popup)

        self.setContentsMargins(0, 0, 0, 0)
        for L in (
            self.main_layout,
            layout,
            self.layout2,
            graph_layout,
            left_layout,
            self.right_layout,
            form_layout,
        ):
            L.setContentsMargins(0, 0, 0, 0)
            L.setSpacing(0)

        # Set the main layout to the widget
        self.setLayout(self.main_layout)
        self.setWindowTitle("SECQUOIA")
        QTimer.singleShot(0, self._after_show)
