"""Dialog for combining two segmentation masks with a bitwise operation."""

from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime

import imageio.v2 as imageio
import numpy as np
from qtpy.QtCore import QCoreApplication, Qt, QTimer
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from SECQUOIA.config import LINKS, NAPARIPARAMETERS, STYLE, TOOLTIPSTEXT
from SECQUOIA.core.gap_filling import add_missing_rows
from SECQUOIA.core.project_state import save_project_state
from SECQUOIA.core.quantification import quantify
from SECQUOIA.core.segmentation.mask_arithmetic import compute_bitwise_mask
from SECQUOIA.core.segmentation.mask_io import (
    fallback_mask_filename,
    load_masks,
    t_file_from_idx,
)
from SECQUOIA.gui.common.ui_utils import (
    add_progress_bar,
    fixed_label,
    make_help_button,
)
from SECQUOIA.gui.curation_tree import update_list
from SECQUOIA.gui.main_window.mouse_bindings import (
    add_mouse_drag_to_segmentation_layer,
)
from SECQUOIA.utils.plotting import update_plot
from SECQUOIA.utils.positions import (
    current_position_number,
    resolve_position_range,
)
from SECQUOIA.utils.timing import current_t_range as _current_t_range

LOG = logging.getLogger(__name__)

__all__ = [
    "MaskArithmeticDialog",
    "open_arithmetic_dialog",
]

# Maps the combo box labels to the tokens core.segmentation.mask_arithmetic expects.
OP_MAP = {
    "intersection": "INT",
    "union": "UNI",
    "mutually exclusive": "ME",
    "NONE": "NONE",
}

# Half-width of the preview crop, in pixels.
_PREVIEW_HALF = 64

# Opacity of the label overlay drawn on the preview, 0-255.
_OVERLAY_ALPHA = 120
_PREVIEW_MIN_SIZE = (256, 256)

# Fraction of the progress bar to fill immediately on run, so it visibly
_STARTUP_PROGRESS_FRACTION = 0.05


def to_pixmap_overlay(img2d: np.ndarray, mask2d: np.ndarray) -> QPixmap:
    """Render a greyscale image with a translucent, per label colour overlay."""
    img = img2d.astype(np.float32)
    if img.max() > 0:
        img = img / img.max()
    img_rgb = (np.stack([img, img, img], axis=-1) * 255).astype(np.uint8)

    overlay = np.zeros_like(img_rgb, dtype=np.uint8)
    labels = mask2d.astype(np.int64)
    for lab in np.unique(labels):
        if lab == 0:
            continue
        rng = np.random.default_rng(int(lab))
        overlay[labels == lab] = rng.integers(80, 256, size=3)

    alpha = ((labels > 0).astype(np.uint8) * _OVERLAY_ALPHA).astype(np.uint8)

    # Alpha blend: out = overlay * a + img * (1 - a)
    a = alpha[..., None].astype(np.float32) / 255.0
    out = (
        overlay.astype(np.float32) * a + img_rgb.astype(np.float32) * (1.0 - a)
    ).astype(np.uint8)

    h, w = out.shape[:2]
    qimg = QImage(out.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def crop_bounds(
    h: int, w: int, cx: int, cy: int, half: int = 128
) -> tuple[int, int, int, int]:
    """Return a square crop around a centre point, clipped to the image."""
    left = int(max(0, cx - half))
    top = int(max(0, cy - half))
    right = int(min(w, cx + half))
    bottom = int(min(h, cy + half))
    return left, top, right, bottom


def _append_mask_and_show(
    main_window, new_mask: np.ndarray, layer_name: str
) -> None:
    """Register a new label stack on the main window and show it in the viewers."""
    container = getattr(main_window, "labels", None)
    container = container + [new_mask]

    main_window.labels = container
    main_window.n_masks = len(container)

    main_window.update_channel_mask_dropdowns()

    for attr in ("viewer_1", "viewer_2"):
        viewer = getattr(main_window, attr, None)
        if viewer is None:
            continue
        try:
            viewer.add_labels(
                new_mask, name=layer_name, opacity=NAPARIPARAMETERS.OPACITY
            )
            with contextlib.suppress(Exception):
                add_mouse_drag_to_segmentation_layer(main_window, layer_name)
        except (ValueError, RuntimeError) as e:
            LOG.warning(
                "[bitwise] Failed adding labels layer '%s': %s", layer_name, e
            )


@dataclass(frozen=True)
class BitwiseOpConfig:
    """The mask arithmetic operation selected in the dialog: which masks, which op, and modifiers."""

    m1_idx_1based: int
    m2_idx_1based: int | None
    op_token: str
    not_a: bool
    not_b: bool
    dil1v: int
    dil2v: int


def _bitwise_ops_tag(cfg: BitwiseOpConfig) -> str:
    """Short tag describing the operation, e.g. 'NM1D2_INT_M2'."""
    return (
        f"{'N' if cfg.not_a else ''}"
        f"M{cfg.m1_idx_1based}"
        f"{'D' + str(cfg.dil1v) if cfg.dil1v != 0 else ''}"
        f"_{cfg.op_token}_"
        f"{'N' if cfg.not_b else ''}"
        f"M{cfg.m2_idx_1based if cfg.m2_idx_1based else ''}"
        f"{'D' + str(cfg.dil2v) if cfg.dil2v != 0 else ''}"
    )


def _prepare_output_dir(main_window, cfg: BitwiseOpConfig) -> str:
    """Create the operation-named output folder and register it on main_window."""
    parent_dir = os.path.dirname(
        os.path.normpath(main_window.segmentation_paths[0])
    )
    ts = datetime.now().strftime("%y%m%d")
    ops = _bitwise_ops_tag(cfg)
    out_seg_dir = os.path.join(parent_dir, f"Segmentation_{ts}_{ops}")
    os.makedirs(out_seg_dir, exist_ok=True)

    if out_seg_dir not in getattr(main_window, "segmentation_paths", []):
        main_window.segmentation_paths.append(out_seg_dir)

    return out_seg_dir


def _resolve_position_masks(
    main_window,
    pos: int,
    current_pos: int,
    pos_name: str,
    seg_paths_selected: list[str],
    cfg: BitwiseOpConfig,
    t_file_min: int,
    t_file_max: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return the (mask_a, mask_b) stacks for one position.

    The current position reuses the stacks already held in memory; every
    other position is read back from disk.
    """
    if pos == current_pos:
        mask_b = (
            main_window.labels[cfg.m2_idx_1based - 1]
            if cfg.op_token != "NONE"
            else None
        )
        return main_window.labels[cfg.m1_idx_1based - 1], mask_b

    stacks = load_masks(
        seg_paths_selected,
        pos_name,
        main_window.image_format,
        t_file_min=t_file_min,
        t_file_max=t_file_max,
    )
    if not isinstance(stacks, list | tuple) or not stacks:
        return None, None

    mask_a = stacks[0] if len(stacks) > 0 else None
    mask_b = stacks[1] if len(stacks) > 1 else None
    return mask_a, mask_b


def _write_combined_stack(
    combined_stack: np.ndarray,
    main_window,
    pos_dir: str,
    pos_name: str,
    t_idx_min: int,
    t_idx_max: int,
) -> None:
    """Write every time slice of a combined stack to disk as a mask image."""
    for t_idx in range(t_idx_min, t_idx_max + 1):
        labels2d = combined_stack[t_idx]
        t_file = t_file_from_idx(main_window, t_idx)
        out_path = fallback_mask_filename(
            dirpath=pos_dir,
            pos_name=pos_name,
            t_file=int(t_file),
            image_format=main_window.image_format,
        )
        imageio.imwrite(out_path, labels2d.astype(np.uint16))


def _process_one_position(
    main_window,
    pos: int,
    current_pos: int,
    out_seg_dir: str,
    exp_name: str,
    seg_paths_selected: list[str],
    cfg: BitwiseOpConfig,
    keep_sep: bool,
    t_file_min: int,
    t_file_max: int,
    t_idx_min: int,
    t_idx_max: int,
) -> tuple[int, np.ndarray | None]:
    """Load, combine and write the masks for a single position.

    Returns the position number together with its combined stack (``None``
    when the source masks were missing), so the caller can hand the
    current position's result to the napari viewers.
    """
    pos_name = f"{exp_name}_p{pos:04d}"
    pos_dir = os.path.join(out_seg_dir, pos_name)
    os.makedirs(pos_dir, exist_ok=True)

    mask_a, mask_b = _resolve_position_masks(
        main_window,
        pos,
        current_pos,
        pos_name,
        seg_paths_selected,
        cfg,
        t_file_min,
        t_file_max,
    )

    if mask_a is None or (cfg.op_token != "NONE" and mask_b is None):
        return pos, None

    combined_stack = compute_bitwise_mask(
        mask_a,
        mask_b,
        cfg.op_token,
        bool(cfg.not_a),
        bool(cfg.not_b),
        int(cfg.dil1v),
        int(cfg.dil2v),
        keep_separate=keep_sep,
    )

    _write_combined_stack(
        combined_stack, main_window, pos_dir, pos_name, t_idx_min, t_idx_max
    )

    return pos, combined_stack


def _save_bitwise_masks_all_positions(
    main_window,
    cfg: BitwiseOpConfig,
    keep_sep: bool = False,
    progress_step=lambda i, n: None,
) -> str:
    """Apply the operation to every selected position and write it to disk."""
    out_seg_dir = _prepare_output_dir(main_window, cfg)

    pos_min, pos_max = resolve_position_range(main_window)
    exp_name = (
        str(getattr(main_window, "experiment_name", "Experiment")).strip()
        or "Experiment"
    )
    t_file_min, t_file_max, t_idx_min, t_idx_max = _current_t_range(
        main_window
    )

    positions = list(range(pos_min, pos_max + 1))
    total_positions = max(1, len(positions))
    current_pos = current_position_number(main_window, positions)

    seg_paths_selected = [
        main_window.segmentation_paths[cfg.m1_idx_1based - 1]
    ]
    if cfg.op_token != "NONE":
        seg_paths_selected.append(
            main_window.segmentation_paths[cfg.m2_idx_1based - 1]
        )

    position_kwargs = {
        "main_window": main_window,
        "current_pos": current_pos,
        "out_seg_dir": out_seg_dir,
        "exp_name": exp_name,
        "seg_paths_selected": seg_paths_selected,
        "cfg": cfg,
        "keep_sep": keep_sep,
        "t_file_min": t_file_min,
        "t_file_max": t_file_max,
        "t_idx_min": t_idx_min,
        "t_idx_max": t_idx_max,
    }

    # Nudge the bar right away so the user sees the run has started, since
    # the first position can take a while to load, combine and write.
    progress_step(
        total_positions * _STARTUP_PROGRESS_FRACTION, total_positions
    )

    for i, pos in enumerate(positions, start=1):
        _, combined_stack = _process_one_position(pos=pos, **position_kwargs)

        if pos == current_pos and combined_stack is not None:
            title = f"Segmentation{main_window.n_masks + 1}"
            _append_mask_and_show(
                main_window, combined_stack, layer_name=title
            )

        progress_step(i, total_positions)

    return out_seg_dir


class MaskArithmeticDialog(QDialog):
    """Modal dialog for combining two masks, with a live preview.

    Attributes:
        main_window: The application main window.
    """

    def __init__(self, main_window, parent: QWidget | None = None):
        """Build the dialog and prepare the preview crop."""
        super().__init__(parent if parent is not None else main_window)
        self.main_window = main_window

        self.setWindowTitle("Bitwise mask")
        self.setStyleSheet(f"""
        QWidget {{
            font-size: {STYLE.FONT_SIZE_outlier}pt;
            font-family: "{STYLE.FONT_outlier}";
        }}
        """)

        self._build_ui()
        self._prepare_preview_crop()
        self._connect_signals()

        self._toggle_m2()
        QTimer.singleShot(0, self.update_preview)

    def _build_ui(self) -> None:
        """Create the preview, the control row, the progress bar and buttons."""
        root = QVBoxLayout(self)

        help_row = QHBoxLayout()
        help_row.setContentsMargins(0, 0, 0, 0)
        help_row.addStretch(1)
        help_row.addWidget(
            make_help_button(
                self, TOOLTIPSTEXT.RHELP, LINKS.GITHUB_DYNAMICS_PLOT
            )
        )
        root.addLayout(help_row)

        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(*_PREVIEW_MIN_SIZE)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet(
            "background:#111; border:1pt solid #333;"
        )
        root.addWidget(self.preview_label)

        root.addLayout(self._build_control_row())

        self.keep_sep_cb = QCheckBox("Keep touching masks separate")
        self.keep_sep_cb.setToolTip(TOOLTIPSTEXT.NO_MERGING)
        root.addWidget(self.keep_sep_cb)

        prog_row = QHBoxLayout()
        self.progress_bar, helpers = add_progress_bar(prog_row)
        self.progress_bar.setMinimumHeight(18)
        self.set_progress = helpers["set"]
        self.finish_progress = helpers["finish"]
        self.set_progress_step = helpers["step"]
        root.addLayout(prog_row)

        root.addLayout(self._build_button_row())

    def _build_control_row(self) -> QHBoxLayout:
        """Build the M1 / operator / M2 control strip."""
        mask_count = max(1, self.main_window.n_masks)

        self.not1 = QCheckBox("invert")
        self.not1.setToolTip(TOOLTIPSTEXT.RNOT)
        self.lbl_dil_m1 = fixed_label("Dilation (px)")
        self.dil_m1 = self._make_dilation_spin()
        self.lbl_m1 = fixed_label("M1:")
        self.m1 = self._make_mask_combo(mask_count)

        self.lbl_op = fixed_label("Op:")
        self.op = QComboBox()
        self.op.setToolTip(TOOLTIPSTEXT.RLOP)
        self.op.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        for label in (
            "NONE (ignore M2)",
            "intersection",
            "union",
            "mutually exclusive",
        ):
            token = "NONE" if label.startswith("NONE") else label
            self.op.addItem(label, token)
        self.op.setCurrentIndex(0)

        self.not2 = QCheckBox("invert")
        self.not2.setToolTip(TOOLTIPSTEXT.RNOT)
        self.lbl_dil_m2 = fixed_label("Dilation (px)")
        self.dil_m2 = self._make_dilation_spin()
        self.lbl_m2 = fixed_label("M2:")
        self.m2 = self._make_mask_combo(mask_count)

        row = QHBoxLayout()
        for widget in (
            self.not1,
            self.lbl_dil_m1,
            self.dil_m1,
            self.lbl_m1,
            self.m1,
            self.lbl_op,
            self.op,
            self.not2,
            self.lbl_dil_m2,
            self.dil_m2,
            self.lbl_m2,
            self.m2,
        ):
            row.addWidget(widget)
        return row

    def _build_button_row(self) -> QHBoxLayout:
        """Build the Run / Exit row."""
        self.run_btn = QPushButton("Run")
        self.run_btn.setToolTip(TOOLTIPSTEXT.RLRUN)
        self.exit_btn = QPushButton("Exit")
        self.exit_btn.setToolTip(TOOLTIPSTEXT.REXIT)

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.run_btn)
        row.addWidget(self.exit_btn)
        return row

    @staticmethod
    def _make_dilation_spin() -> QSpinBox:
        """Create a dilation spin box. Negative values erode.

        Returns:
            The configured spin box.
        """
        spin = QSpinBox()
        spin.setRange(-999, 999)
        spin.setValue(0)
        spin.setToolTip(TOOLTIPSTEXT.RDIL)
        return spin

    @staticmethod
    def _make_mask_combo(mask_count: int) -> QComboBox:
        """Create a combo listing the available masks, numbered from 1."""
        combo = QComboBox()
        combo.setToolTip(TOOLTIPSTEXT.RM)
        for i in range(1, mask_count + 1):
            combo.addItem(str(i), i)
        combo.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        return combo

    def _prepare_preview_crop(self) -> None:
        """Work out the crop window used by every preview refresh."""
        self.t = int(getattr(self.main_window, "current_time_index", 0))
        img2d = self.main_window.images[self.main_window.ids_channels[0]][
            self.t
        ]

        cx, cy = self._current_center_xy(self.t, img2d)
        h, w = img2d.shape
        self.left, self.top, self.right, self.bottom = crop_bounds(
            h, w, cx, cy, half=_PREVIEW_HALF
        )
        self.img_crop = img2d[self.top : self.bottom, self.left : self.right]

    def _connect_signals(self) -> None:
        """Refresh the preview whenever any control changes."""
        self.op.currentIndexChanged.connect(self._toggle_m2)

        for signal in (
            self.not1.stateChanged,
            self.not2.stateChanged,
            self.dil_m1.valueChanged,
            self.dil_m2.valueChanged,
            self.m1.currentIndexChanged,
            self.m2.currentIndexChanged,
            self.op.currentIndexChanged,
            self.keep_sep_cb.stateChanged,
        ):
            signal.connect(self.update_preview)

        self.run_btn.clicked.connect(self._on_run)
        self.exit_btn.clicked.connect(self.reject)

    def _current_center_xy(self, t: int, img2d: np.ndarray) -> tuple[int, int]:
        """Find the tracked cell's centre at a time point."""
        try:
            df = getattr(self.main_window, "df_subset", None)
            if df is None or df.empty:
                df = getattr(self.main_window, "filtered_df", None)
            if df is not None and not df.empty:
                sub = df[df["t"] == int(t)]
                if not sub.empty:
                    x, y = sub[["XMorphology", "YMorphology"]].iloc[0]
                    return x, y
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
        h, w = img2d.shape[:2]
        return w // 2, h // 2

    def _read_controls(self) -> dict:
        """Collect the current control values."""
        return {
            "dil1v": int(self.dil_m1.value() or 0),
            "dil2v": int(self.dil_m2.value() or 0),
            "m1v": int(self.m1.currentData() or 1),
            "m2v": int(self.m2.currentData() or 1),
            "op": OP_MAP[self.op.currentData() or "NONE"],
            "not_a": bool(self.not1.isChecked()),
            "not_b": bool(self.not2.isChecked()),
            "keep_sep": bool(self.keep_sep_cb.isChecked()),
        }

    def update_preview(self) -> None:
        """Recompute and redraw the preview for the current settings."""
        try:
            cfg = self._read_controls()
            labels = self.main_window.labels

            slice_a = labels[cfg["m1v"] - 1][self.t]
            slice_b = (
                None if cfg["op"] == "NONE" else labels[cfg["m2v"] - 1][self.t]
            )

            combined = compute_bitwise_mask(
                slice_a[None, ...],
                None if slice_b is None else slice_b[None, ...],
                cfg["op"],
                cfg["not_a"],
                cfg["not_b"],
                cfg["dil1v"],
                cfg["dil2v"],
                keep_separate=cfg["keep_sep"],
            )[0]

            mask_crop = combined[
                self.top : self.bottom, self.left : self.right
            ]
            pixmap = to_pixmap_overlay(self.img_crop, mask_crop)

            self.preview_label.setPixmap(
                pixmap.scaled(
                    self.preview_label.width(),
                    self.preview_label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        except (
            AttributeError,
            IndexError,
            KeyError,
            TypeError,
            ValueError,
        ) as e:
            self.preview_label.setText(f"Preview unavailable\n{e}")

    def _toggle_m2(self) -> None:
        """Enable the second-mask controls only when the operator needs them."""
        use_m2 = self.op.currentData() != "NONE"
        for widget in (
            self.not2,
            self.lbl_dil_m2,
            self.dil_m2,
            self.lbl_m2,
            self.m2,
        ):
            widget.setEnabled(use_m2)

    def _on_run(self) -> None:
        """Apply the operation across all positions, then refresh the app."""
        self.run_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.progress_bar.show()
            self.set_progress(0)
            QApplication.processEvents()

            ui_cfg = self._read_controls()
            op_cfg = BitwiseOpConfig(
                m1_idx_1based=ui_cfg["m1v"],
                m2_idx_1based=(
                    None if ui_cfg["op"] == "NONE" else ui_cfg["m2v"]
                ),
                op_token=ui_cfg["op"],
                not_a=ui_cfg["not_a"],
                not_b=ui_cfg["not_b"],
                dil1v=ui_cfg["dil1v"],
                dil2v=ui_cfg["dil2v"],
            )

            def _tick(idx, total):
                """Advance the progress bar and keep the UI responsive."""
                self.set_progress_step(idx, total)
                QCoreApplication.processEvents()

            out_dir = _save_bitwise_masks_all_positions(
                main_window=self.main_window,
                cfg=op_cfg,
                keep_sep=ui_cfg["keep_sep"],
                progress_step=_tick,
            )
            LOG.info("[bitwise] Saved new segmentation to: %s", out_dir)

            quantify(self.main_window)
            add_missing_rows(self.main_window)
            update_list(self.main_window)
            update_plot(self.main_window)
            self.main_window.zoom_in()
            save_project_state(self.main_window)

            self.finish_progress()
            QApplication.processEvents()
            self.accept()
        except ValueError as e:
            LOG.error("[bitwise] Failed to compute mask: %s", e)
            self.finish_progress()
            self.reject()
        finally:
            self.run_btn.setEnabled(True)
            QApplication.restoreOverrideCursor()


def open_arithmetic_dialog(main_window: QWidget) -> None:
    """Open the bitwise mask arithmetic dialog modally."""
    MaskArithmeticDialog(main_window).exec_()
