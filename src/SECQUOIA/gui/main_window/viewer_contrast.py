"""Per viewer contrast and mask opacity slider controls."""

import contextlib
import logging

import numpy as np
from napari.layers import Image
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QSlider, QWidget

from SECQUOIA.config import TOOLTIPSTEXT

LOG = logging.getLogger(__name__)

_SLIDER_STEPS = 1000


def _to_slider(val: float, rmin: float, rmax: float) -> int:
    """Map a real value in [rmin, rmax] to a slider position in [0, _SLIDER_STEPS]."""
    if rmax == rmin:
        return 0
    val = max(rmin, min(rmax, val))
    return int(round((val - rmin) / (rmax - rmin) * _SLIDER_STEPS))


def _to_real(idx: int, rmin: float, rmax: float) -> float:
    """Map a slider position in [0, _SLIDER_STEPS] to a real value in [rmin, rmax]."""
    if rmax == rmin:
        return rmin
    return rmin + (rmax - rmin) * (float(idx) / _SLIDER_STEPS)


class ViewerContrast:
    """Contrast (black/white point) and mask-opacity slider controls."""

    def reset_all_contrast(self, reset_ui: bool = True) -> None:
        """Set all channel-image layers' contrast to the full range."""
        viewers = getattr(self, "viewer_fluorescence", []) or []
        for vi, viewer in enumerate(viewers):
            if viewer is None:
                continue
            for layer in getattr(viewer, "layers", []):
                try:
                    if not isinstance(layer, Image):
                        continue
                except (RuntimeError, AttributeError, TypeError):
                    if not hasattr(layer, "contrast_limits"):
                        continue

                try:
                    rmin, rmax = map(
                        float,
                        getattr(
                            layer,
                            "contrast_limits_range",
                            layer.contrast_limits,
                        ),
                    )
                except (RuntimeError, AttributeError, TypeError):
                    rmin, rmax = 0.0, 1.0
                if not (rmax > rmin):
                    try:
                        dmin = float(np.nanmin(layer.data))
                        dmax = float(np.nanmax(layer.data))
                        if dmax > dmin:
                            rmin, rmax = dmin, dmax
                        else:
                            rmin, rmax = 0.0, 1.0
                    except (RuntimeError, AttributeError, TypeError):
                        rmin, rmax = 0.0, 1.0

                try:
                    layer.contrast_limits = (rmin, rmax)
                except (RuntimeError, AttributeError, TypeError) as e:
                    LOG.warning(
                        "[reset contrast] failed on %s: %s",
                        getattr(layer, "name", "?"),
                        e,
                    )

            if (
                reset_ui
                and hasattr(self, "contrast_widgets")
                and vi < len(self.contrast_widgets)
            ):
                with contextlib.suppress(
                    RuntimeError, AttributeError, TypeError
                ):
                    self._sync_contrast_ui_from_layer(
                        viewer, self.contrast_widgets[vi]
                    )

    def _style_half_slider(
        self,
        s,
        *,
        handle_color="#2d8cff",
        groove_h=2,
        handle_w=8,
        overlap_px=0,
    ):
        """Slim horizontal slider styling."""
        s.setOrientation(Qt.Horizontal)
        s.setRange(0, _SLIDER_STEPS)
        s.setSingleStep(1)
        s.setPageStep(25)
        s.setFixedHeight(max(groove_h + 8, 12))
        s.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: {groove_h}px;
                background: #ffffff;
                border-radius: {groove_h//2}px;
            }}
            QSlider::sub-page:horizontal {{ background: #ffffff; border-radius: {groove_h//2}px; }}
            QSlider::add-page:horizontal {{ background: #ffffff; border-radius: {groove_h//2}px; }}
            QSlider::handle:horizontal {{
                background: {handle_color};
                border: 1px solid rgba(0,0,0,120);
                width: {handle_w}px;
                margin: -6px -{int(overlap_px)}px;  /* allow handles to meet */
                border-radius: 2px;
            }}
            """)

    def _build_contrast_controls(self, viewer):
        """Create and wire the per viewer black/white contrast slider controls."""
        container = QWidget()
        outer = QHBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        rail = QWidget(container)
        rail_l = QHBoxLayout(rail)
        rail_l.setContentsMargins(0, 0, 0, 0)
        rail_l.setSpacing(0)
        rail.setMinimumHeight(14)

        s_left = QSlider(Qt.Horizontal, rail)
        s_left.setToolTip(TOOLTIPSTEXT.SLEFT)
        s_right = QSlider(Qt.Horizontal, rail)
        s_right.setToolTip(TOOLTIPSTEXT.RRIGHT)

        self._style_half_slider(s_left, groove_h=2, handle_w=8)
        self._style_half_slider(s_right, groove_h=2, handle_w=8)

        SLIDER_WIDTH = 56
        s_left.setMinimumWidth(SLIDER_WIDTH)
        s_right.setMinimumWidth(SLIDER_WIDTH)
        s_left.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        s_right.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        rail_l.addWidget(s_left, 0)
        rail_l.addWidget(s_right, 0)
        outer.addWidget(rail, 0)
        outer.addStretch(1)

        container._viewer_ref = viewer
        container._s_left = s_left
        container._s_right = s_right
        container._armed = False
        container._range = (0.0, 1.0)

        s_left.valueChanged.connect(
            lambda _=None, cont=container: self._apply_contrast_from_ui(
                cont, who="left"
            )
        )
        s_right.valueChanged.connect(
            lambda _=None, cont=container: self._apply_contrast_from_ui(
                cont, who="right"
            )
        )

        s_left.blockSignals(True)
        s_right.blockSignals(True)
        s_left.setValue(100)
        s_right.setValue(900)
        s_left.blockSignals(False)
        s_right.blockSignals(False)

        container.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        return container

    def _sync_contrast_ui_from_layer(self, viewer, contrast_container) -> None:
        """Update the contrast sliders to match the active channel layer's limits."""
        if viewer is None or contrast_container is None:
            return
        layer = self._get_active_channel_layer(viewer)
        s_left = contrast_container._s_left
        s_right = contrast_container._s_right

        if layer is None or not hasattr(layer, "contrast_limits"):
            s_left.setEnabled(False)
            s_right.setEnabled(False)
            return

        s_left.setEnabled(True)
        s_right.setEnabled(True)

        try:
            lo, hi = map(float, layer.contrast_limits)
        except (RuntimeError, AttributeError, TypeError):
            lo, hi = 0.0, 1.0

        try:
            rmin, rmax = map(
                float, getattr(layer, "contrast_limits_range", (lo, hi))
            )
        except (RuntimeError, AttributeError, TypeError):
            rmin, rmax = lo, hi

        if rmin == rmax:
            rmin, rmax = 0.0, max(1.0, rmax)

        contrast_container._range = (rmin, rmax)

        li = _to_slider(lo, rmin, rmax)
        ri = _to_slider(hi, rmin, rmax)

        if ri <= li:
            ri = min(_SLIDER_STEPS, li + 1)
            if ri == li:
                li = max(0, li - 1)

        s_left.blockSignals(True)
        s_right.blockSignals(True)
        s_left.setValue(li)
        s_right.setValue(ri)
        s_left.blockSignals(False)
        s_right.blockSignals(False)

    def _apply_contrast_from_ui(self, contrast_container, who: str):
        """Read side-by-side sliders and set layer.contrast_limits."""
        if not getattr(contrast_container, "_armed", False):
            return

        viewer = contrast_container._viewer_ref
        if viewer is None:
            return
        layer = self._get_active_channel_layer(viewer)
        if layer is None or not hasattr(layer, "contrast_limits"):
            return

        s_left = contrast_container._s_left
        s_right = contrast_container._s_right
        rmin, rmax = getattr(contrast_container, "_range", (0.0, 1.0))

        li = s_left.value()
        ri = s_right.value()

        if who == "left" and li >= ri:
            ri = min(_SLIDER_STEPS, li + 1)
            s_right.blockSignals(True)
            s_right.setValue(ri)
            s_right.blockSignals(False)
        elif who == "right" and ri <= li:
            li = max(0, ri - 1)
            s_left.blockSignals(True)
            s_left.setValue(li)
            s_left.blockSignals(False)
        else:
            if li == ri:
                if ri < _SLIDER_STEPS:
                    ri += 1
                    s_right.blockSignals(True)
                    s_right.setValue(ri)
                    s_right.blockSignals(False)
                else:
                    li = max(0, li - 1)
                    s_left.blockSignals(True)
                    s_left.setValue(li)
                    s_left.blockSignals(False)

        lo = float(_to_real(s_left.value(), rmin, rmax))
        hi = float(_to_real(s_right.value(), rmin, rmax))

        if not (lo < hi):
            eps = max((rmax - rmin) / _SLIDER_STEPS, np.finfo(float).eps)
            hi = min(rmax, lo + eps)
            if not (lo < hi):
                lo = max(rmin, hi - eps)

            s_left.blockSignals(True)
            s_left.setValue(_to_slider(lo, rmin, rmax))
            s_left.blockSignals(False)
            s_right.blockSignals(True)
            s_right.setValue(_to_slider(hi, rmin, rmax))
            s_right.blockSignals(False)

        try:
            layer.contrast_limits = (lo, hi)
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            LOG.warning("[contrast] could not set contrast_limits: %s", e)

    def _get_active_channel_layer(self, viewer) -> Image:
        """Return the active channel image layer for *viewer*, or None."""
        if viewer is None:
            return None
        try:
            row_i = self.viewer_fluorescence.index(viewer)
            ch_combo = self.channel_combos[row_i]
            channel = ch_combo.currentData()
            if channel is not None:
                lyr = self._find_layer_by_name(
                    viewer, self._ch_layer_name_for(f"w{channel}")
                )
                if lyr is not None and self._is_channel_layer(lyr):
                    return lyr
        except (RuntimeError, AttributeError, TypeError, ValueError):
            pass
        for lyr in getattr(viewer, "layers", []):
            if self._is_channel_layer(lyr) and getattr(lyr, "visible", False):
                return lyr
        return None

    def _build_opacity_control(self, viewer):
        """Horizontal slider (0-100%) that controls the opacity of the currently
        selected mask in this viewer.
        """
        container = QWidget()
        outer = QHBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        lbl = QLabel("α:")
        f = lbl.font()
        lbl.setFont(f)
        lbl.setToolTip("Opacity of the selected mask")
        lbl.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        outer.addWidget(lbl, 0)

        s = QSlider(Qt.Horizontal, container)
        s.setToolTip(TOOLTIPSTEXT.OPACITY)
        self._style_half_slider(s, groove_h=2, handle_w=8)
        s.setRange(0, 100)
        s.setSingleStep(1)
        s.setPageStep(5)
        s.setMinimumWidth(56)
        s.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        outer.addWidget(s, 0)
        outer.addStretch(1)
        container.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)

        container._viewer_ref = viewer
        container._slider = s

        s.valueChanged.connect(
            lambda _=None, cont=container: self._apply_opacity_from_ui(cont)
        )

        s.setValue(100)
        return container

    def _apply_opacity_from_ui(self, opacity_container):
        """Read slider (0-100) and set opacity on the selected mask layer."""
        viewer = getattr(opacity_container, "_viewer_ref", None)
        if viewer is None:
            return

        s = opacity_container._slider
        new_alpha = float(s.value()) / 100.0

        m_val = self._mask_selection_for_viewer(viewer)

        def _set_mask_opacity(mask_idx: int):
            """Set the opacity for a single segmentation mask layer if it exists."""
            lyr = self._get_seg_layer(viewer, mask_idx)
            if lyr is not None and hasattr(lyr, "opacity"):
                try:
                    lyr.opacity = new_alpha
                except (RuntimeError, AttributeError, TypeError) as e:
                    LOG.warning(
                        "[opacity] could not set opacity for mask %s: %s",
                        mask_idx,
                        e,
                    )

        if m_val == 0:
            # ALL masks
            m_n = max(0, int(getattr(self, "n_masks", 0)))
            for i in range(1, m_n + 1):
                _set_mask_opacity(i)
        else:
            # Single selected mask
            _set_mask_opacity(int(m_val))

    def _sync_opacity_ui_from_mask(self, viewer, opacity_container):
        """Enable the opacity slider and set its value from the segmentation layer(s)
        when at least one mask exists, else disable and reset it to 100.
        """
        if viewer is None or opacity_container is None:
            return

        s = opacity_container._slider

        m_n = max(0, int(getattr(self, "n_masks", 0)))
        if m_n == 0:
            s.setEnabled(False)
            s.blockSignals(True)
            s.setValue(100)
            s.blockSignals(False)
            return

        s.setEnabled(True)

        m_val = self._mask_selection_for_viewer(viewer)

        if m_val == 0:
            vals = []
            for i in range(1, m_n + 1):
                lyr = self._get_seg_layer(viewer, i)
                if lyr is not None and hasattr(lyr, "opacity"):
                    with contextlib.suppress(
                        RuntimeError, AttributeError, TypeError
                    ):
                        vals.append(float(getattr(lyr, "opacity", 1.0)))
            mean_alpha = (sum(vals) / len(vals)) if vals else 1.0
            val = int(round(mean_alpha * 100.0))
            val = max(0, min(100, val))
            s.blockSignals(True)
            s.setValue(val)
            s.blockSignals(False)
            return

        # Single mask selected
        lyr = self._get_seg_layer(viewer, int(m_val))
        if lyr is None or not hasattr(lyr, "opacity"):
            s.blockSignals(True)
            s.setValue(100)
            s.blockSignals(False)
            return

        val = int(round(float(getattr(lyr, "opacity", 1.0)) * 100.0))
        val = max(0, min(100, val))
        s.blockSignals(True)
        s.setValue(val)
        s.blockSignals(False)
