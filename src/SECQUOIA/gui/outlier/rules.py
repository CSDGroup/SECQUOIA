"""Turning the outlier dialog's widgets into a saveable rules pack."""

from __future__ import annotations

import logging

from SECQUOIA.config import Rule, SlidingWindow
from SECQUOIA.core.outlier_detection import RulesPack, rule_is_active

LOG = logging.getLogger(__name__)

__all__ = ["snapshot_outlier_ui_to_pack"]


def snapshot_outlier_ui_to_pack(main_window) -> RulesPack:
    """Convert the current outlier UI settings into a RulesPack object."""
    m_n = int(getattr(main_window, "n_masks", 0) or 0)
    ch_n = int(getattr(main_window, "n_channels", 0) or 0)

    rules = []
    for i, w in enumerate(getattr(main_window, "_rows_widgets", []), 1):
        masks_raw = list(w["m_multi"].selected_values())
        chans_raw = list(w["ch_multi"].selected_values())
        masks = (
            list(range(1, m_n + 1))
            if (masks_raw == [] and m_n > 0)
            else masks_raw
        )
        channels = (
            [ch[1:] for ch in main_window.ids_channels]
            if (chans_raw == [] and ch_n > 0)
            else chans_raw
        )

        rule = Rule(
            feat=str(w["feat"].currentText()),
            masks=[int(x) for x in masks],
            channels=channels,
            masks_raw=[int(x) for x in masks_raw],
            channels_raw=list(chans_raw),
            op1=str(w["op1"].currentText()),
            val1=float(w["val1"].value()),
            op2=(
                w["op2"].currentData()
                if w["op2"].currentData() is not None
                else None
            ),
            val2=(
                float(w["val2"].value())
                if w["op2"].currentData() is not None
                else None
            ),
            combine=str(w["combine"].currentText()).upper(),
        )
        if not rule_is_active(rule):
            LOG.info(
                "[Rules] Threshold row %d skipped: no threshold value set.",
                i,
            )
            continue
        rules.append(rule)

    sliding = []
    for i, sw in enumerate(
        getattr(main_window, "_sliding_rows_widgets", []), 1
    ):
        sd_factor = float(sw["factor"].value())
        t_min = float(sw["tmin"].value())
        t_max = float(sw["tmax"].value())

        if sd_factor <= 0:
            LOG.warning(
                "[Rules] Sliding-window row %d skipped: SD factor must be > 0 (got %s).",
                i,
                sd_factor,
            )
            continue
        if t_max < t_min:
            LOG.warning(
                "[Rules] Sliding-window row %d skipped: t_max (%s) must be >= t_min (%s).",
                i,
                t_max,
                t_min,
            )
            continue

        sliding.append(
            SlidingWindow(
                feature=str(sw["feat"].currentText()),
                mask=sw["mask"].currentData(),
                channel=sw["ch"].currentData(),
                t_min=t_min,
                t_max=t_max,
                sd_factor=sd_factor,
            )
        )

    close_tab = getattr(main_window, "_close_mask_tab", None)
    close_masks = close_tab.settings_for_rules() if close_tab else None

    return RulesPack(
        version=1,
        m_n=m_n,
        ch_n=ch_n,
        rules=rules,
        sliding_windows=sliding,
        close_masks=close_masks,
    )
