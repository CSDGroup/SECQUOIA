"""Builds the HTML summary shown on the Loading tab of the Load Data dialog."""

from __future__ import annotations

import os

from qtpy.QtCore import Qt

__all__ = ["_update_summary"]


def _widget_value(main_window, attr_name):
    """Return `widget.value()` for a main_window widget attr, or None if absent."""
    widget = getattr(main_window, attr_name, None)
    return widget.value() if widget else None


def _summary_project_name(main_window):
    return getattr(main_window, "project_name", "") or "(n/a)"


def _summary_experiment_name(main_window):
    return getattr(main_window, "experiment_name", "") or "(n/a)"


def _summary_folder(main_window):
    return getattr(main_window, "folder", "") or "(not selected)"


def _summary_image_format(main_window):
    return (
        getattr(main_window, "image_format", None)
        or main_window.image_format_combo.currentText()
    )


def _summary_tracking_info(main_window):
    tformat = (
        getattr(main_window, "tracking_format", None)
        or main_window.tracking_format_combo.currentText()
    )
    tpath = getattr(main_window, "tracking_path", None)
    tpath_disp = os.path.basename(tpath) if tpath else "(not selected)"
    return f"{tformat} &mdash; {tpath_disp}"


def _summary_time_range(main_window):
    tmin = getattr(main_window, "time_min_selected", None) or _widget_value(
        main_window, "time_min_spin"
    )
    tmax = getattr(main_window, "time_max_selected", None) or _widget_value(
        main_window, "time_max_spin"
    )
    return f"{tmin} – {tmax}" if (tmin and tmax) else "(n/a)"


def _summary_dt(main_window):
    dt_sec = _widget_value(main_window, "time_input")
    return f"{dt_sec} s" if dt_sec is not None else "(n/a)"


def _summary_channels(main_window):
    sel_channels = []
    if getattr(main_window, "chan_tree", None):
        for i in range(main_window.chan_tree.topLevelItemCount()):
            it = main_window.chan_tree.topLevelItem(i)
            if it.checkState(0) == Qt.Checked:
                sel_channels.append(it.text(0))
    return ", ".join(sel_channels) if sel_channels else "(none selected)"


def _summary_segmentations(main_window):
    segs = getattr(main_window, "segmentation_paths", []) or []
    return (
        ", ".join(os.path.basename(s) for s in segs)
        if segs
        else "(none selected)"
    )


def _summary_background_correction(main_window):
    use_bg = bool(getattr(main_window, "use_background_correction", False))
    bg_sel = getattr(main_window, "background_correction_path", None)
    return os.path.basename(bg_sel) if (use_bg and bg_sel) else "(off)"


def _summary_import_realtime(main_window):
    use_rt = bool(getattr(main_window, "use_import_rt", False))
    rt_sel = getattr(main_window, "import_rt_path", None)
    return os.path.basename(rt_sel) if (use_rt and rt_sel) else "(off)"


def _summary_threshold(main_window):
    thr = getattr(main_window, "threshold", None)
    return f"{thr}" if thr is not None else "(n/a)"


def _summary_min_mask_size(main_window):
    min_sz = getattr(main_window, "min_mask_size", None)
    return f"{min_sz} px" if min_sz is not None else "(n/a)"


def _summary_loading_format(main_window):
    return getattr(main_window, "loading_format", "Positions")


def _summary_position_bounds(main_window):
    pmin = getattr(main_window, "position_min_selected", None)
    pmax = getattr(main_window, "position_max_selected", None)
    return (
        f"{pmin} – {pmax}"
        if (pmin is not None and pmax is not None)
        else "(n/a)"
    )


def _summary_position_start(main_window):
    pstart = getattr(main_window, "position_start_selected", None)
    return f"{pstart}" if pstart is not None else "(n/a)"


def _summary_positions(main_window):
    positions_line = ""
    if getattr(main_window, "tracking_counts_label", None):
        positions_line = main_window.tracking_counts_label.text()
    return positions_line or "(no positions/identifications yet)"


def _update_summary(main_window):
    """Refresh the HTML summary of the current loading configuration."""
    rows = [
        ("Project name", _summary_project_name(main_window)),
        ("Experiment name", _summary_experiment_name(main_window)),
        ("Experiment folder", _summary_folder(main_window)),
        ("Image format", _summary_image_format(main_window)),
        ("Tracking information", _summary_tracking_info(main_window)),
        ("Time point range", _summary_time_range(main_window)),
        ("Time [s] between time points", _summary_dt(main_window)),
        ("Channels", _summary_channels(main_window)),
        ("Segmentations", _summary_segmentations(main_window)),
        (
            "Background correction",
            _summary_background_correction(main_window),
        ),
        ("Import real time [ms]", _summary_import_realtime(main_window)),
        (
            "Max distance of track point from segmentation mask",
            _summary_threshold(main_window),
        ),
        ("Min cell size", _summary_min_mask_size(main_window)),
        ("Loading format", _summary_loading_format(main_window)),
        ("Position bounds (auto)", _summary_position_bounds(main_window)),
        (
            "Start position of curation",
            _summary_position_start(main_window),
        ),
        (
            "Positions &amp; Cell Lineage Trees",
            _summary_positions(main_window),
        ),
    ]

    rows_html = "\n".join(
        f"  <b>{label}:</b> {value}<br>" for label, value in rows
    )
    html = f"<div>\n{rows_html}\n</div>"

    if getattr(main_window, "summary_label", None):
        main_window.summary_label.setText(html)
