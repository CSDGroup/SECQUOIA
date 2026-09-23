"""Configuration constants and shared settings dataclasses for SECQUOIA."""

import sys
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from qtpy.QtCore import Qt

EXPERIMENTS_ROOT_KEY: str = "paths/experiments_root"

TRACKING_ROOT_KEYS: dict[str, str] = {
    "tTt": "paths/tracking_root_tTt",
    "CTC": "paths/tracking_root_CTC",
    "btrack": "paths/tracking_root_btrack",
    "Ultrack": "paths/tracking_root_Ultrack",
}

BTRACK_REQUIREMENT: str = "btrack>=0.7,<0.8"


@dataclass(frozen=True)
class IORELOAD:
    """Reload settings for image reads that may hit a transient I/O error,
    e.g. a cloud-synced folder that can't serve a file while the network connection is down.
    """

    MAX_WAIT_S: float = 3600.0
    BASE_DELAY_S: float = 2.0
    MAX_DELAY_S: float = 30.0


@dataclass(frozen=True)
class STYLE:
    """Default font settings and initial channel/mask counts."""

    FONT_SIZE: int = 12
    FONT_FAMILY: str = "Arial"
    CHANNEL_NUMBER: int = 1
    MASK_NUMBER: int = 1
    SYMBOL_SIZE: int = 6
    FONT_SIZE_outlier: int = 12
    FONT_outlier: str = "Arial"
    FONT_SIZE_HELP: int = 12


@dataclass(frozen=True)
class NAPARIPARAMETERS:
    """Default napari viewer display parameters."""

    OPACITY: int = 1
    ZOOMFACTOR: int = 10


@dataclass
class TracksViewConfig:
    """Configuration options for displaying tracking data in napari."""

    enable_tracks_layer: bool = True
    init_hidden_until_data: bool = True
    use_placeholder_row: bool = True
    auto_show_on_first_data: bool = True

    show_track_ids: bool = True
    render_track_lines: bool = True
    show_track_tail: bool = True
    head_length: int = 0
    tail_length: int = 999999
    opacity: float = 1.0

    sort_primary: Literal["track_id", "t"] = "track_id"
    sort_secondary: Literal["track_id", "t"] = "t"

    @classmethod
    def from_dict(cls, d: dict | None):
        if not d:
            return cls()
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in allowed})


def _default_label_font() -> str:
    """Pick a UI font that actually exists on the current OS."""
    if sys.platform.startswith("win"):
        return "Segoe UI"
    elif sys.platform == "darwin":
        return "Helvetica Neue"
    else:
        return "DejaVu Sans"


class ColorStyle:
    """Color and font constants for feature, mask, channel, and formula labels."""

    FEATURE = "#FFD54F"
    MASK = "#4FC3F7"
    CHANNEL = "#81C784"
    OP = "#FFFFFF"
    BR = "#CFD8DC"
    DIM = "#B0BEC5"

    FONT_FAMILY = _default_label_font()
    FONT_SIZE = 12  # px
    FONT_WEIGHT = "bold"
    TOOLTIP_FONT_SIZE = 16

    @classmethod
    def as_dict(cls) -> dict[str, str]:
        """Return the colors keyed by the tags used in summary labels."""
        return {
            "feature": cls.FEATURE,
            "m": cls.MASK,
            "ch": cls.CHANNEL,
            "op": cls.OP,
            "br": cls.BR,
            "dim": cls.DIM,
        }

    @classmethod
    def font_css(cls) -> str:
        """Return a CSS font declaration for a widget stylesheet."""
        return (
            f"font-family: '{cls.FONT_FAMILY}'; "
            f"font-size: {cls.FONT_SIZE}px; "
            f"font-weight: {cls.FONT_WEIGHT};"
        )


@dataclass(frozen=True)
class FEATURES:
    """Feature-name prefixes used to build and parse feature columns."""

    METRIC_PREFIXES: ClassVar[tuple[str, ...]] = (
        "Mean",
        "Min",
        "Max",
        "Sum",
        "Std",
        "CV",
    )
    BASIC_VARIANT: ClassVar[tuple[str, ...]] = (
        "NoBgCorrected",
        "BaSiCBgCorrectedRatioFlat",
        "BaSiCBgCorrectedNoRatioFlat",
    )
    NO_BASIC_VARIANT: ClassVar[tuple[str]] = ("NoBgCorrected",)
    MORPH_PREFIXES: ClassVar[tuple[str, ...]] = (
        "XMorphology",
        "YMorphology",
        "AreaMorphology",
        "PerimeterMorphology",
        "Orientation",
        "Eccentricity",
        "AxisMajorLength",
        "AxisMinorLength",
    )


@dataclass(frozen=True)
class PLOTPARAMETERS:
    """Default colors, marker symbols and font sizes for plots."""

    ORANGE: ClassVar[tuple[int, int, int, int]] = (255, 165, 0, 255)
    PURPLE: ClassVar[tuple[int, int, int, int]] = (148, 0, 211, 255)
    RED: ClassVar[tuple[int, int, int, int]] = (230, 30, 30, 255)
    BLUE: ClassVar[tuple[int, int, int, int]] = (40, 110, 240, 255)
    YELLOW: ClassVar[tuple[int, int, int, int]] = (255, 215, 0, 255)
    OUTLIERSIZE: ClassVar[int] = 10
    DEFAULT_SYMBOL: ClassVar[str] = "o"
    DEFAULT_SYMBOL_SIZE: ClassVar[int] = 6
    DEFAULT_LABEL_FONT_SIZE: ClassVar[int] = 10

    PG_SYMBOLS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("None (no points)", ""),
        ("Circle (o)", "o"),
        ("Square (s)", "s"),
        ("Triangle (t)", "t"),
        ("Diamond (d)", "d"),
        ("Plus (+)", "+"),
        ("Cross (x)", "x"),
        ("Pentagon (p)", "p"),
        ("Hexagon (h)", "h"),
        ("Star (*)", "star"),
    )


@dataclass(frozen=True)
class LINKS:
    """External documentation and help links."""

    GITHUB_HELP: str = "https://github.com/CSDGroup/SECQUOIA"
    GITHUB_LOADING_WINDOW: str = (
        "https://github.com/CSDGroup/SECQUOIA/blob/main/docs/loading-window.md"
    )
    GITHUB_EXPORTER: str = (
        "https://github.com/CSDGroup/SECQUOIA/blob/main/docs/image-and-movie-export.md"
    )
    GITHUB_OUTLIER: str = (
        "https://github.com/CSDGroup/SECQUOIA/blob/main/docs/outlier-detection.md"
    )
    GITHUB_CYTOMETRIC: str = (
        "https://github.com/CSDGroup/SECQUOIA/blob/main/docs/cytometric-analysis.md"
    )
    GITHUB_tTt_FORMAT: str = (
        "https://github.com/CSDGroup/SECQUOIA/blob/main/docs/data-formats.md"
    )
    GITHUB_tTt_PATTERNS: str = (
        "https://github.com/CSDGroup/SECQUOIA/blob/main/docs/"
        "data-formats.md#create-additional-parsing-patterns"
    )
    GITHUB_DYNAMICS_PLOT: str = (
        "https://github.com/CSDGroup/SECQUOIA/blob/main/docs/dynamics-plot.md"
    )


@dataclass(frozen=True)
class TOOLTIPSTEXT:
    """Centralized tooltip text used across the SECQUOIA GUI."""

    # Buttons
    SAVE: str = (
        "Save data (Ctrl+S). Segmentation masks and CSV files will be exported."
    )
    HEALTHY: str = "Mark the cell at this time point as healthy. Hotkey: H"
    DEAD: str = "Mark the cell at this time point as dead/apoptotic. Hotkey: D"
    FOCUS: str = "Mark the cell at this time point as out of focus. Hotkey: F"
    LOST: str = "Mark the cell at this time point as lost Hotkey: L."
    OUTLIER: str = "Mark the cell at this time point as an outlier Hotkey: O."
    SIGNAL: str = (
        "Mark the cell at this time point as having low signal. Hotkey: K"
    )

    # menubar
    MENUBAR_tTt: str = (
        "Open the tool to restructure/rename an experiment into the tTt folder format."
    )
    # Napari toolbar
    BTN_BRUSH: str = (
        "Select a specific mask (not ALL) to enable the Brush tool (or press 2)."
    )
    BTN_ERASE: str = (
        "Select a specific mask (not ALL) to enable the Erase tool (or press 1)."
    )
    BTN_PAN: str = "Pan/Zoom (always available) or press 6."

    # Loading window
    USER: str = "Enter user initials in tTt format."
    EXPFOLDER: str = "Select your experiment folder in tTt format."
    TRACKINGFORMAT: str = "Select the tracking format input."
    IMAGEFORMAT: str = "Select the image file format."
    TIME: str = "Provide the time interval between time points (in seconds)."
    MINTIME: str = "Define the minimum time point (start time point)."
    MAXTIME: str = "Define the maximum time point (end time point)."
    STARTPOSITION: str = "Select the position where curation should start."
    THRESHOLD: str = (
        "Define maximum distance in pixels between tracking point and mask centroid."
    )
    AREA: str = "Define minimum mask size."
    BG: str = "Enable BaSiC background correction for the loaded data."
    BG_COMBO: str = (
        "Select a BaSiC* folder from the Analysis directory to use for background correction."
    )
    BG_NONE: str = (
        "No BaSiC* folder found in the Analysis directory of this experiment."
    )
    CP: str = "If enabled (tTt only), links tracks across different positions."
    TRACKING_TREE: str = (
        "Check a Position to (de)select all its lineage trees. Expand to select/deselect individual trees."
    )
    TRACKING_TREE_INFO: str = (
        "Use the checkboxes to select positions and their lineage trees for loading."
    )
    SELECT_ALL_BTN: str = "Select all lineage trees in all positions."
    UNSELECT_ALL_BTN: str = "Unselect all lineage trees in all positions."
    RUN_BTN: str = (
        "Run loading with the current settings. This scans data and prepares curation."
    )
    CLOSE_BTN_LOADING: str = "Close this dialog without starting curation."
    CURATION_BTN: str = "Close this window and begin curation"
    RUN_ALL: str = "Load all positions."
    RUN_POSITION: str = (
        "Load only the positions you select in the tracking tree."
    )
    RUN_POSITION_BACKGROUND: str = (
        "Load selected positions and BaSiC background data (if available)."
    )
    LOADING_FORMAT: str = (
        "Choose what to load: all positions or only the selected ones."
    )
    SEG_TREE: str = "Select one or more segmentation folders from Analysis."
    SEG_TREE_INFO: str = (
        "Tick one or more segmentation results to use during loading/curation."
    )
    CH_TREE_INFO: str = (
        "Tick the channels to quantify. Comments (if any) are shown in the next column."
    )
    CH_TREE: str = "Select channels to include in analysis."
    LOAD_DATA_BTN: str = (
        "Scan folder for channels/segmentation; enable other tabs."
    )
    HELP_BTN: str = "Open the online help page in your web browser."

    # Channel/Mask manager
    CM_APPLY_ALL: str = (
        "Re-run quantification for every position in the experiment after "
        "applying channel/mask changes."
    )
    CM_APPLY_BTN: str = (
        "Apply the selected channels and masks and re-run quantification."
    )

    # Main window tooltips
    MPLOT: str = "Change the mask used for plotting."
    CPLOT: str = "Change the channel used for plotting."
    BTN_R: str = "Open the metrics window."
    BTN_ARITH: str = "Open the mask arithmetics window."
    CH_VIEWER: str = "Change the displayed channels in the viewer."
    M_VIEWER: str = "Change the displayed masks in the viewer."
    SLEFT: str = "Adjust black and white points"
    RRIGHT: str = "Adjust black and white points"
    OPACITY: str = "Opacity of the selected mask (0–100%)"
    FEATURE: str = "Change the plotted feature."
    OUT: str = (
        "Show only identified outliers and close-mask cases. Available after "
        "running outlier or close-mask detection."
    )
    ALL: str = "Show all Tree-IDs for each position."
    EYEV: str = "Show/Hide this viewer."
    EYEL: str = "Show/Hide this lineage tree."
    EYEP: str = "Show/Hide this plot."
    ID: str = "Current Identification (ID), TrackNumber (G), and Time (T)"
    SELECTION: str = "Current selection of feature, mask and channel."

    # Outlier window
    ADD_BTN: str = "Add another row for outlier detection."
    APPLY_BTN: str = "Run outlier detection."
    EXIT_BTN: str = "Close this window."
    RM_BTN: str = "Remove this rule."
    FEAT_OUT: str = "Select a feature for outlier detection."
    M_OUT: str = "Select the mask used for intensity."
    CH_OUT: str = "Select a channel for outlier detection."
    CMP_OUT: str = "Choose a comparison operator."
    VAL_SPIN: str = "Set the threshold value."
    PREVIEW_BTN: str = "Open interactive preview for all rules."
    LOAD_BTN: str = "Load outlier rules from a .json file."
    NEXT_BTN: str = "Go to sliding window settings."
    FACTOR_SD: str = (
        "Tolerance as ±N standard deviations of the window around the current value. Use 1–3 for typical sensitivity; 0 disables this row."
    )
    TMAX_SP: str = (
        "Window end (Δt after current time). Units follow your 't' column. Example: 2 means look ahead 2 time units from the current point."
    )
    TMIN_SP: str = (
        "Window start (Δt before current time). Units follow your 't' column. Example: 2 means look back 2 time units from the current point."
    )
    CH_CB: str = (
        "Channel(s) whose values are used. Tick one or more, or leave empty to "
        "include all channels."
    )
    MASK_CB: str = (
        "Mask(s) whose values are used. Tick one or more, or leave empty to "
        "include all masks."
    )
    FEAT_CB: str = (
        "Feature (measurement) to screen for outliers. If your data only has "
        "feature or feature+mask columns, the matching channel/mask is applied "
        "automatically."
    )
    OP2_SP: str = (
        "Optional second comparison, combined with the first via Combine "
        "(e.g. '<10 AND >2'). Set to 'None' for a single-threshold rule."
    )
    VAL2_SP: str = (
        "Second threshold value, used with Op2. Enabled only when Op2 is not "
        "'None'."
    )
    COMBINE_COMBO: str = (
        "How the two comparisons combine:\n"
        "• OR  = flag the tails (either condition is true)\n"
        "• AND = flag inside a band (both conditions are true)"
    )
    NEXT_OUT: str = "Review summary and run."
    NEXT_CLOSE: str = "Set up close-mask detection."
    CLOSE_DIST: str = (
        "Flag a tracking point when a second mask lies at most this many "
        "pixels from it. Cannot exceed the tolerance used for matching."
    )
    CLOSE_MASK: str = "Masks whose second candidates are checked."
    OUT_NEXT: str = (
        "Go to the next Tree-ID with an outlier or a flagged close mask."
    )
    OUT_PREVIOUS: str = (
        "Go to the previous Tree-ID with an outlier or a flagged close mask."
    )
    OUT_RESET: str = (
        "Remove all outlier and close-mask flags and the saved rules."
    )
    CLOSE_CHECK: str = (
        "Mark the flagged close-mask case or outlier on screen as reviewed "
        "and jump to the next one (C)."
    )
    CLOSE_FIND: str = "Flag tracking points with a close second mask."
    CLOSE_RESET: str = (
        "Remove all close-mask flags, including reviewed ones, and stop "
        "detecting them until 'Find close masks' is run again."
    )
    REFRESH_SUMMARY: str = "Rebuild summary from the current GUI selections."
    ADD_PLOT_BTN: str = (
        "Add a new histogram panel, pre-filled with a copy of this panel's settings."
    )
    RM_PLOT_BTN: str = "Remove this histogram panel from the preview."
    SUBMIT_BTN: str = (
        "Send every panel's feature, thresholds and mask/channel choices back to the "
        "outlier rule rows in the main window, then close the preview."
    )
    CLOSE_BTN: str = (
        "Close the preview and discard changes made here; the rules in the main window stay unchanged."
    )
    NO_MERGING: str = (
        "When dilating, grow each mask without merging neighbours."
    )
    OUT_HELP: str = "Open outlier detection help."
    # Arithmetic tooltips
    RHELP: str = "Open the dynamics plots help page in your web browser."
    RFEATURE: str = "Select a feature for arithmetic operations."
    ROP: str = "Choose a mathematical operator."
    RLOP: str = (
        "intersection of 2 masks (logical AND), union of 2 masks (logical OR), mutually exclusive regions of each mask (logical XOR), NONE: only the first mask is used."
    )
    RRUN: str = "Create and add the calculated feature to the plots."
    RLRUN: str = "Create and add the calculated mask to the plots."
    REXIT: str = "Close the Arithmetic window."
    RC1: str = "Choose channel 1."
    RM1: str = "Choose a mask for channel 1."
    RC2: str = "Choose channel 2."
    RM2: str = "Choose a mask for channel 2."
    RNOT: str = "Invert the mask (Background <=> foreground, logical NOT)."
    RM: str = "Choose a mask."
    RDIL: str = "x>0 dilates mask by x px; x<0 erodes by |x| px."
    TPMIN: str = "Lower time point bound for baseline window (x/baseline)."
    TPMAX: str = "Upper time point bound for baseline window (x/baseline)."
    SCOPE: str = "Normalize across all rows or per selected IDs."
    NORM_ENABLE: str = "Apply a normalization to the derived metric."
    NORM_METHOD: str = "Pick how to normalize the result."
    SHOW_PLOT_BTN: str = (
        "Open a preview window showing the outlier plots for all configured rules."
    )
    OK_BTN_OUT: str = "Close this dialog and return to the main window."
    SAVE_COMPOSITE_BTN: str = (
        "Save all histogram panels as one composite image "
        "(PNG, PDF or SVG). You choose the file name and location."
    )
    CLOSE_PLOTS_BTN: str = (
        "Close the plot window and return to the parameter overview. "
        "Nothing is saved."
    )
    # Lineage / Heat Tree
    TREE: str = "Toggle between Lineage (T) and Heat Tree (H)."
    E_BTN: str = "Redo selection (Ctrl+Shift+A)."
    TREE_FEATURE: str = "Select the feature to plot in the heat tree."
    TREE_MASK: str = "Select the mask to plot in the heat tree."
    TREE_CH: str = "Select the channel to plot in the heat tree."
    HIGHLIGHT_ALL: str = "Highlight all tracks in the lineage tree (Ctrl+A)."
    PAINT_LINEAGE: str = (
        "Activate Highlight mode (Ctrl+P). Ctrl+click branches to paint with the active color. Change color in menu or Ctrl+Shift+P."
    )

    PLACEHOLDERTEXT: str = "Type to search…"
    EDIT: str = "Enter text. F3: Next entry. Shift+F3: Previous entry."
    BTN_NEXT: str = "Find next (F3)"
    BTN_PREV: str = "Find previous (Shift+F3)"

    # Fusion dialog
    FUSE_WINDOW: str = (
        "Fuse two lineage trees together at a chosen time point."
    )
    FUSE_ID_1: str = (
        "Enter the numeric ID of the first tree to fuse (e.g., 12)."
    )
    FUSE_ID_2: str = (
        "Enter the numeric ID of the second tree to fuse (e.g., 34)."
    )
    CELL_1: str = "Cell index within Tree ID 1 (auto-filled; read-only)."
    CELL_2: str = "Cell index within Tree ID 2 (auto-filled; read-only)."
    FUSE_TIME: str = (
        "Select the time point (t) at which the two trees will be fused."
    )
    FUSE_BTN: str = "Fuse the two specified trees at the selected time point."
    TID1: str = "ID of the first tree to fuse."
    TID2: str = "ID of the second tree to fuse."
    L_CELL_1: str = "Cell index within Tree ID 1 (read-only)."
    L_CELL_2: str = "Cell index within Tree ID 2 (read-only)."
    L_FUSE_AT: str = "Choose the time index t at which to perform the fusion."

    # Tracking bar
    NEW_ID: str = "Assign a new unique Tree-ID to the currently selected cell."
    DIVISION: str = (
        "Mark the selected cell as dividing at the current time point (Ctrl+D)."
    )
    REMOVE_DIVISION: str = (
        "Remove an existing division event from the selected cell."
    )
    SPLIT_TREE: str = (
        "Split the selected lineage tree into two separate branches."
    )
    FUSE_TREE: str = (
        "Fuse two selected lineage trees into a single branch (Select Tree-IDs with Ctrl + Left mouse click)."
    )
    UNDO_TRACKING: str = (
        "Undo the most recent tracking edit (New ID, Division, Remove "
        "Division, Split Tree or Fuse Tree). Separate from mask-editing "
        "undo (Ctrl+Z)."
    )
    REDO_TRACKING: str = (
        "Redo the most recently undone tracking edit. Separate from "
        "mask-editing redo (Ctrl+Y)."
    )

    SELECT_BTN: str = (
        "Select experiment folder for Cytometric analysis data in tTt format."
    )
    LOAD_BTN_CYTOMETRIC: str = "Load Cytometric analysis experiment data."
    BASIC_CYTOMETRIC: str = (
        "Enable BaSiC background correction for Cytometric analysis data."
    )
    BASIC_FOLDER_CYTOMETRIC: str = (
        "Select a BaSiC folder from Analysis for Cytometric analysis data."
    )
    RUN_CYTOMETRIC_BTN: str = "Run Cytometric analysis pipeline."
    HELP_CYTOMETRIC: str = "Open Cytometric Analysis help page."

    T_MODE: str = "Settings for Tree (T) mode."
    HEAT_MODE: str = "Settings for Heatmap (H) mode."
    LABEL_MODE: str = "Controls for lineage/track labels."
    COMBO_MODE: str = "Tree mode."
    LINE_WIDTH: str = "Lineage line width."
    CONNECTOR_WIDTH_SB: str = "Connector line width."
    HEAT_WIDTH_SB: str = "Heat lane width (single-channel)."
    HEAT_WIDTH_MULTI_SB: str = "Heat lane width (multi-channel)."
    LOW_BTN: str = (
        "Choose the minimum (LOW) color for single-channel heat lanes."
    )
    HIGH_BTN: str = (
        "Choose the maximum (HIGH) color for single-channel heat lanes."
    )
    SHOW_LABEL_CB: str = "Toggle track/generation labels on the plot."
    T_G_COMBO: str = "Switch between Track number and Generation."
    FONT_SB: str = "Label font size."
    APPLY_BTN_L: str = "Apply changes without closing the dialog."
    OK_BTN_L: str = "Apply changes and close this dialog."
    CANCEL_BTN_L: str = "Close without applying new changes."

    SYMBOL_COMBO: str = "Select the marker shape used for plotted data points."
    SIZE_SPIN: str = (
        "Set the size of the marker symbols (0 hides markers even if a symbol is selected)."
    )
    LINE_WIDTH_PLOT: str = "Set the thickness of the main plot lines."
    LINE_WIDTH_PLOT_HIGHLIGHT: str = (
        "Set the thickness of highlighted or overlay lines."
    )
    AXIS_SPIN_PLOT: str = "Set the font size for axis labels and tick text."
    APPLY_BTN_PLOT: str = (
        "Apply changes to the plot without closing the dialog."
    )
    OK_BTN_PLOT: str = "Apply changes and close the dialog."
    CANCEL_BTN_PLOT: str = "Discard changes and close the dialog."

    START_T: str = "Start time index (inclusive) for preview/export loop."
    END_T: str = "End time index (inclusive) for preview/export loop."
    GIF_SPEED: str = "Playback/export frames per second."
    PLAY_BUTTON: str = "Play/Pause timeline preview."

    PREVIEW_TILE: str = "Preview tile. Right-click to add or remove an image."
    PREVIEW_LABEL: str = (
        "Rendered preview of the selected ID/channel at current time and crop."
    )
    CANVAS_VIEW: str = "Right-click empty space to add an image."
    CONTEXT_ADD_ADJACENT: str = (
        "Add a new tile adjacent to this tile (side is inferred)."
    )
    CONTEXT_REMOVE_TILE: str = "Remove this tile from the canvas."
    CONTEXT_ADD_NEAR_CURSOR: str = (
        "Add a new tile near the cursor (auto-snapped to the grid)."
    )

    CROP_W: str = "Crop width around the tracked centroid."
    CROP_H: str = "Crop height around the tracked centroid."
    TILE_W: str = "Output width per tile (affects preview and export)."
    TILE_H: str = "Output height per tile (affects preview and export)."

    SELECTION_GROUP: str = (
        "Choose the ID, channel, and black/white levels for the active tile."
    )
    IDENT_COMBO: str = "Choose a Tree-ID."
    CHAN_COMBO: str = (
        "Select which image channel stack to render in this tile."
    )
    GEN_BUTTON: str = (
        "Pick a lineage path (track numbers) to follow over time."
    )
    BLACK_SPIN: str = "Black point (0–255)."
    WHITE_SPIN: str = "White point (0–255)."
    BW_INSPECTOR_BUTTON: str = (
        "Open a histogram view with draggable handles to adjust black/white points."
    )

    # Advanced settings (GIF/export window)
    ADV_GROUP: str = (
        "Advanced overlay options for the active tile: mask overlay, time indicator, "
        "time stamp and channel label."
    )
    ADV_MASK_COMBO: str = (
        "Choose which segmentation mask is overlaid on the active tile. Applies to preview and export."
    )
    ADV_MASK_ALPHA: str = (
        "Transparency of the mask overlay (0.0 = invisible, 1.0 = fully opaque). Default 0.35."
    )
    ADV_MASK_MODE: str = (
        "Draw the mask as a filled area ('Full') or as an outline only ('Contour')."
    )
    ADV_CONTOUR_PX: str = (
        "Outline thickness in pixels. Only has an effect in 'Contour' mode."
    )
    ADV_MASK_COLOR: str = "Pick the color used to draw the mask overlay."
    ADV_TIME_BAR_CHK: str = (
        "Draw a progress indicator on the tile showing how far the current frame is within the time range."
    )
    ADV_INDICATOR_PX: str = "Height of the time indicator in pixels."
    ADV_INDICATOR_COLOR: str = "Pick the color of the time indicator."
    ADV_TIME_TEXT_CHK: str = (
        "Burn the elapsed time (hh:mm:ss) of the current frame into the tile."
    )
    ADV_TIME_FONT: str = "Font size of the burned-in time stamp."
    ADV_TIME_COLOR: str = "Pick the color of the burned-in time stamp."
    ADV_SHOW_CH_LBL: str = (
        "Burn a channel label into the tile. Uses the text below, or the channel name if that field is empty."
    )
    ADV_CH_FONT: str = "Font size of the channel label."
    ADV_CH_COLOR: str = "Pick the color of the channel label."
    ADV_CH_LABEL_EDIT: str = (
        "Custom channel label text for the active tile, e.g. 'GFP'. Leave empty to use the channel name."
    )
    ADV_FONT_COMBO: str = (
        "Font family used for the time stamp and channel label of this tile."
    )

    EXPORT_GROUP: str = (
        "Choose export kind and format, define spacing, then click Export."
    )
    EXPORT_TYPE: str = (
        "Export a single frame per time or an animation over a time range."
    )
    EXPORT_FMT: str = "Output format for the chosen export type."
    EXPORT_GAP: str = (
        "Extra spacing between tiles in the exported composite (does not affect on-screen layout)."
    )
    EXPORT_BTN: str = "Run the export using the current settings."
    EXPORT_MISSING: str = (
        "How to handle missing frames in the selected time range."
    )

    PRESETS_GROUP: str = (
        "Save the layout and per tile settings to JSON; reload them later."
    )
    PRESET_NAME: str = "Enter a preset name to save the current configuration."
    PRESET_SAVE: str = "Save current grid + settings to a JSON preset."
    PRESET_LIST: str = "Choose a previously saved preset to load."
    PRESET_LOAD: str = "Load the selected preset and rebuild the canvas."

    GENERATIONS_LEVEL_COMBO: str = (
        "Choose which child track to follow for this generation level."
    )
    GENERATIONS_APPLY_ALL: str = (
        "Apply the chosen lineage path to every tile using this Identification."
    )

    SELECT_BTN_tTt: str = (
        "Select the original (input) experiment folder containing the images."
    )
    INPUT_FOLDER_tTt: str = "Choose the input folder."
    BTN_READ: str = (
        "Reads the first image filename and auto-fills date/pos/time/z/channel when possible."
    )
    OUTPUT_FOLDER_tTt: str = (
        "Select the destination folder where the tTt experiment folder will be created."
    )
    SELECT_OUTPUT_FOLDER_tTt: str = "Choose the output folder."
    RUN_tTt: str = (
        "Create tTt folder structure, copy images, and rename them to tTt convention."
    )
    DATE_tTt: str = (
        "Experiment date. Format: YYYY-MM-DD (recommended) or YYYYMMDD."
    )
    INITIALS_tTt: str = "Your initials (letters only). Example: MA."
    SETUP_tTt: str = "Microscope setup number. Example: 01 or 30."
    POSITION_tTt: str = (
        "Default position used when not found in filenames. Example: p0001."
    )
    TIME_tTt: str = (
        "Default time point used when not found in filenames. Example: t1 -> becomes t00001."
    )
    Z_POSITION_tTt: str = (
        "Default z-slice used when not found in filenames. Example: z1 -> becomes z001. Leave empty to default to z001."
    )
    CHANNEL_tTt: str = (
        "Default channel used when not found in filenames. Example: c1->w00, c2->w01, w0->w00, w1->w01."
    )
    PROGRESS_tTt: str = "Progress bar showing copy/rename status."
    HELP_BTN_tTt: str = (
        "Open the online help page for tTt Data Format Transformer."
    )
    PATTERN_SETTINGS_tTt: str = (
        "Edit the regular patterns used to find tokens in filenames."
    )
    HELP_BTN_tTt_PATTERNS: str = "Open the online help page."
    PRESET_COMBO_tTt: str = (
        "Which saved parsing pattern to use. "
        "Add or edit parsing patterns from Pattern settings."
    )
    BEFORE_tTt_PATTERNS: str = (
        "Fixed text that appears right before the value in your "
        "filename. Leave empty if there is none."
    )
    VALUE_tTt_PATTERNS: str = (
        "One real example of the value, exactly as it appears in one of "
        "your own filenames (e.g. 'A1')."
    )
    AFTER_tTt_PATTERNS: str = (
        "Fixed text that appears right after the value in your "
        "filename. Leave empty if there is none."
    )
    PATTERN_FIELD_tTt_PATTERNS: str = (
        "The pattern used to search filenames, generated from "
        "the fields above."
    )
    EXP_TOKEN_FIELD_tTt_PATTERNS: str = (
        "Regular pattern matching the whole date+initials+setup "
        "token at once (e.g. '240323MA35')."
    )
    PRESET_COMBO_tTt_PATTERNS: str = (
        "Pick a saved pattern, or type a new name and click "
        "Save to add one."
    )
    DELETE_PRESET_tTt_PATTERNS: str = (
        "Delete this parsing pattern. Built-in ones (Nikon, Leica, "
        "Zeiss/Olympus) can't be deleted."
    )
    PREVIEW_tTt_PATTERNS: str = (
        "What the fixed text + example value above resolves to."
    )
    CH_ONE_INDEXED_tTt_PATTERNS: str = (
        "Leave unticked if the first channel is numbered 0 instead "
        "(e.g. 'Ch0')."
    )
    SAVE_BTN_tTt_PATTERNS: str = (
        "Save the current fields under the parsing pattern name above, "
        "and make it the active one."
    )
    CANCEL_BTN_tTt_PATTERNS: str = "Close without saving any changes."
    RESTORE_DEFAULTS_BTN_tTt_PATTERNS: str = (
        "Reset the fields below to the built-in patterns."
    )
    STATUS_OK_tTt_PATTERNS: str = (
        "This pattern compiles and defines the group it needs."
    )

    PROJECT_PATH: str = "Path to project_metadata.json."
    SELECT_PROJECT_PATH: str = "Browse for project_metadata.json."
    LOAD_PROJECT: str = "Load the selected project."
    PROGRESS_PROJECT: str = "Progress bar showing project loading status."
    MSG_LOADING_PROJECT: str = "Loading status."
    EXIT_PROJECT: str = "Close this window without loading a project."
    HELP_PROJECT: str = "Open the load previous project help page."
    ALL_POS: str = "All positions found in the selected experiment folder."
    EDIT_POS: str = (
        "Type to filter the list below. Only matching positions stay visible."
    )
    SELECT_POS: str = (
        "Available positions. One click highlights a position, "
        "a double-click loads it directly."
    )
    LOAD_POS: str = (
        "Load the highlighted position: the current position is saved first, "
        "then its images, masks and tracks are loaded."
    )
    EXIT_POS: str = "Close this window without changing the position."

    PLAYBACK_DISABLED: str = (
        "Playback is switched off. Step through time with the slider or the "
        "arrow keys instead."
    )

    INSPECTOR_LEVEL_READOUT: str = "Black and white points as raw intensities."


@dataclass(frozen=True)
class CURATIONROLE:
    """Custom Qt item roles used for curation state in tree widgets."""

    CURATION_ROLE: int = Qt.UserRole + 100
    ACTIVE_ROLE: int = Qt.UserRole + 101


@dataclass(frozen=True)
class CURATIONSTATUS:
    """String constants describing curation status values."""

    CURATION_CHECKED: str = "checked"
    CURATION_IN_PROGRESS: str = "in_progress"
    CURATION_NOT_CHECKED: str = "not_checked"


@dataclass(frozen=True)
class EXPORT:
    """Default limits, sizes, spacing, and font settings for image/movie export."""

    BW_MIN: int = 0
    BW_MAX: int = 255
    BW_DEFAULT_BLACK: int = 0
    BW_DEFAULT_WHITE: int = 255
    DEPTH_LOW_PERCENTILE: float = 0.1
    DEPTH_HIGH_PERCENTILE: float = 99.9
    DEPTH_SAMPLE_FRAMES: int = 8
    DEPTH_SAMPLE_STRIDE: int = 4

    CROP_FULL_MIN: int = 20
    CROP_FULL_MAX: int = 4096

    TILE_MIN: int = 40
    TILE_MAX: int = 2048
    PREVIEW_W: int = 200
    PREVIEW_H: int = 200

    GIF_SPEED_MIN: int = 1
    GIF_SPEED_MAX: int = 500
    GIF_SPEED_DEFAULT: int = 20
    GIF_SAFE_MAX_FPS: int = 50

    GRID_GAP: int = 6
    TILE_PAD: int = 12
    CANVAS_MARGIN: int = 0

    EXPORT_GAP_DEFAULT: int = 8

    TIME_INDICATOR_Y: int = 8
    TEXT_PAD_L: int = 8
    TEXT_PAD_R: int = 8
    LABEL_GAP: int = 6

    FONT_NAMES = (
        "Arial",
        "DejaVu Sans",
        "Calibri",
        "Times New Roman",
        "Courier New",
    )

    DEFAULT_FONT_NAME = "Arial"


class TRACKING:
    """Column names and prefixes used in the tracking dataframe."""

    REALTIME_PREFIX: ClassVar[str] = "RealTimeMinutes_Ch"
    SAFE_KEEP_NUMERIC: ClassVar[frozenset[str]] = frozenset(
        {
            "t",
            "TrackNumber",
            "Position",
            "t_idx",
            "t_file",
            "track_id",
            "Calculated_Time",
        }
    )


@dataclass
class Rule:
    """Outlier-detection rule with feature, mask, channel, and threshold settings."""

    feat: str
    masks: list[int]
    channels: list[str]
    op1: str
    val1: float
    op2: str | None = None
    val2: float | None = None
    combine: str = "OR"
    masks_raw: list[int] = field(default_factory=list)
    channels_raw: list[str] = field(default_factory=list)


@dataclass
class CloseMaskSettings:
    """Close-mask detection: flag points with a second mask within `distance` px."""

    distance: float
    masks: list[int] | None = None


@dataclass
class SlidingWindow:
    """Sliding-window settings for local outlier detection over time."""

    feature: str
    mask: int | None
    channel: str | None
    t_min: float
    t_max: float
    sd_factor: float
