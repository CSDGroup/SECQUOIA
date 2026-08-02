"""Cytometric Analysis batch quantification and mask matching."""

from __future__ import annotations

import glob
import logging
import os
import re
from contextlib import suppress
from pathlib import Path

import numpy as np
import pandas as pd
from qtpy.QtCore import QCoreApplication
from qtpy.QtWidgets import QMessageBox
from skimage.measure import regionprops_table
from skimage.segmentation import clear_border

from SECQUOIA.core.basic_correction import apply_basic_correction
from SECQUOIA.core.fluorescence_loading import load_fl_channels
from SECQUOIA.core.memmap_store import cleanup_memmaps
from SECQUOIA.core.quantification import sum_intensity
from SECQUOIA.utils.io_reload import to_csv_with_reload

LOG = logging.getLogger(__name__)


def run_cytometric_analysis(main_window) -> None:
    """Iterate over all *_pXXXX folders inside main_window.folder, load images and masks.

    Run region-based quantification, and save CSV per position.
    """
    from SECQUOIA.core.segmentation.mask_io import load_masks

    exp_dir = getattr(main_window, "folder", None)
    if not exp_dir or not os.path.isdir(exp_dir):
        QMessageBox.warning(
            getattr(main_window, "experiment_loader_window", main_window),
            "Select Experiment Folder",
            "Please select a valid experiment folder first.",
        )
        return

    seg_roots = list(getattr(main_window, "segmentation_paths", None) or [])
    if not seg_roots:
        QMessageBox.warning(
            getattr(main_window, "experiment_loader_window", main_window),
            "Select Segmentation Folder",
            "Please load experiment data and select at least one segmentation folder first.",
        )
        return

    invalid_seg_roots = [p for p in seg_roots if not os.path.isdir(p)]
    if invalid_seg_roots:
        QMessageBox.warning(
            getattr(main_window, "experiment_loader_window", main_window),
            "Invalid Segmentation Folder",
            "One or more selected segmentation folders are invalid:\n\n"
            + "\n".join(invalid_seg_roots),
        )
        return

    img_fmt = (
        str(getattr(main_window, "image_format", "png")).lstrip(".").lower()
    )
    seg_fmt = (
        str(getattr(main_window, "seg_format", img_fmt)).lstrip(".").lower()
    )

    out_dir = os.path.join(exp_dir, "Analysis", "Cytometric_Analysis")
    os.makedirs(out_dir, exist_ok=True)

    p_re = re.compile(r"_p(\d{4})")
    pos_dirs = []
    for name in os.listdir(exp_dir):
        full = os.path.join(exp_dir, name)
        if os.path.isdir(full) and p_re.search(name):
            pos_dirs.append(full)

    def _pnum(path: str) -> int:
        m = p_re.search(os.path.basename(path))
        return int(m.group(1)) if m else 10**9

    pos_dirs = sorted(pos_dirs, key=_pnum)

    if not pos_dirs:
        QMessageBox.information(
            getattr(main_window, "experiment_loader_window", main_window),
            "No Positions Found",
            "No position folders matching *_p#### were found inside the experiment folder.",
        )
        return

    dlg = getattr(main_window, "experiment_loader_window", None)
    bar = getattr(dlg, "progress_bar", None) if dlg is not None else None
    if bar is not None:
        bar.setValue(5)
        bar.setFormat("%p%")
        QCoreApplication.processEvents()

    threshold_px = (
        dlg.threshold_spin.value()
        if dlg is not None and hasattr(dlg, "threshold_spin")
        else 20
    )

    t_re = re.compile(r"_t(\d+)")

    def get_t(stem: str) -> int:
        m = t_re.search(stem)
        if not m:
            raise ValueError(f"No _t##### found in: {stem}")
        return int(m.group(1))

    ch_re = re.compile(r"(?:_)?(w\d\d)$")

    def quantify_one_position(pos_path: str, out_csv: str) -> None:
        pos_name = os.path.basename(pos_path)
        img_dir = Path(pos_path)
        img_files = sorted(img_dir.glob(f"*.{img_fmt}"))
        if not img_files:
            to_csv_with_reload(pd.DataFrame(), out_csv, index=False)
            return

        channels = sorted(
            {m.group(1) for f in img_files if (m := ch_re.search(f.stem))}
        )
        if not channels:
            to_csv_with_reload(pd.DataFrame(), out_csv, index=False)
            return

        img_t_vals = set()
        for f in img_files:
            if ch_re.search(f.stem) and t_re.search(f.stem):
                with suppress(ValueError):
                    img_t_vals.add(get_t(f.stem))
        if not img_t_vals:
            to_csv_with_reload(pd.DataFrame(), out_csv, index=False)
            return

        seg_t_vals = []
        for seg_root in seg_roots:
            seg_pos_dir = os.path.join(seg_root, pos_name)
            seg_files = sorted(
                glob.glob(os.path.join(seg_pos_dir, f"*.{seg_fmt}"))
            )
            for fp in seg_files:
                with suppress(ValueError):
                    seg_t_vals.append(get_t(Path(fp).stem))
        seg_t_vals = sorted(set(seg_t_vals))

        if not seg_t_vals:
            to_csv_with_reload(pd.DataFrame(), out_csv, index=False)
            return

        t_min, t_max = seg_t_vals[0], seg_t_vals[-1]

        mask_stacks = load_masks(
            seg_roots,
            pos_selection=pos_path,
            image_format=seg_fmt,
            t_file_min=t_min,
            t_file_max=t_max,
        )

        if mask_stacks is None:
            mask_stacks = [None] * len(seg_roots)

        if not isinstance(mask_stacks, (list | tuple)):
            mask_stacks = [mask_stacks]

        mask_stacks = list(mask_stacks)
        if len(mask_stacks) < len(seg_roots):
            mask_stacks += [None] * (len(seg_roots) - len(mask_stacks))

        basic_checked = bool(
            getattr(getattr(main_window, "basic", None), "flag", False)
        )
        basic_path = getattr(main_window, "background_correction_path", None)

        with suppress(OSError, RuntimeError, ValueError):
            cleanup_memmaps(main_window, remove_viewer_image_layers=False)

        mpos = p_re.search(pos_name)
        pos_num = int(mpos.group(1)) if mpos else 0
        main_window.current_position_number = pos_num
        main_window.position_selection = pos_path
        main_window.image_format = img_fmt
        main_window.time_min_selected, main_window.time_max_selected = (
            t_min,
            t_max,
        )
        main_window.ids_channels, main_window.n_channels = channels, len(
            channels
        )
        for ci, ch in enumerate(channels):
            setattr(main_window, f"FL_identifiers_{ci + 1}", ch)

        try:
            load_fl_channels(main_window)
        except (OSError, ValueError, RuntimeError) as e:
            LOG.error("load_fl_channels failed: %s", e)
            to_csv_with_reload(pd.DataFrame(), out_csv, index=False)
            return

        raw_stacks = getattr(main_window, "images", None) or {}
        present = getattr(main_window, "image_present", None) or {}

        T = max(0, t_max - t_min + 1)
        any_present = np.zeros(T, dtype=bool)
        for ch in channels:
            pr = present.get(ch)
            if pr is not None and len(pr) == T:
                any_present |= pr.astype(bool)
        img_ts_in_range = {t_min + i for i in range(T) if any_present[i]}

        corr_ratio = corr_norf = None
        if basic_checked and basic_path:
            try:
                exp_name = getattr(
                    main_window, "experiment_name", os.path.basename(exp_dir)
                )
                main_window.basic.add_path_basic(basic_path)
                main_window.basic.add_exp_name(exp_name)
                main_window.basic.add_t_range((t_min, t_max))
                main_window.basic.add_channels_valid(channels)
                main_window.basic.update_all(position=pos_num)
                apply_basic_correction(main_window)
                corr_ratio = (
                    getattr(main_window, "corrected_images_ratioflat", None)
                    or {}
                )
                corr_norf = (
                    getattr(main_window, "corrected_images_noratioflat", None)
                    or {}
                )
            except (OSError, ValueError, RuntimeError) as e:
                LOG.warning("BaSiC skipped: %s", e)
                corr_ratio = corr_norf = None

        stem, ext = os.path.splitext(out_csv)
        mask_tables = {}

        for mask_idx, stack in enumerate(mask_stacks, start=1):
            suffix = f"M{mask_idx}"
            out_csv_mask = f"{stem}_{suffix}{ext}"

            if stack is None:
                to_csv_with_reload(pd.DataFrame(), out_csv_mask, index=False)
                continue

            stack = np.asarray(stack)
            if stack.ndim == 2:
                stack = stack[None, ...]

            seg_map = {}
            for t in seg_t_vals:
                idx = t - t_min
                if 0 <= idx < stack.shape[0]:
                    seg_map[t] = stack[idx]

            t_values = sorted(set(seg_map) & img_ts_in_range)
            if not t_values:
                to_csv_with_reload(pd.DataFrame(), out_csv_mask, index=False)
                continue

            rows = []
            for t in t_values:
                idx = t - t_min
                mask = seg_map[t]

                geom = pd.DataFrame(
                    regionprops_table(
                        mask,
                        properties=(
                            "label",
                            "area",
                            "centroid",
                            "eccentricity",
                            "orientation",
                            "axis_major_length",
                            "axis_minor_length",
                        ),
                    )
                )
                if geom.empty:
                    continue

                geom = geom.rename(
                    columns={
                        "centroid-0": "YMorphology",
                        "centroid-1": "XMorphology",
                    }
                )
                geom.insert(1, "t", t)

                interior_labels = set(np.unique(clear_border(mask)))
                geom["TouchingBorder"] = (
                    ~geom["label"].isin(interior_labels)
                ).astype(int)

                for ch in channels:
                    mcol_raw = f"MeanNoBgCorrectedCh{ch[1:]}{suffix}"
                    stdcol_raw = f"StdNoBgCorrectedCh{ch[1:]}{suffix}"
                    cvcol_raw = f"CVNoBgCorrectedCh{ch[1:]}{suffix}"
                    scol_raw = f"SumNoBgCorrectedCh{ch[1:]}{suffix}"

                    stack_ch = raw_stacks.get(ch)
                    pr = present.get(ch)
                    ok = (
                        stack_ch is not None
                        and 0 <= idx < getattr(stack_ch, "shape", (0,))[0]
                        and (pr is None or (idx < len(pr) and bool(pr[idx])))
                    )
                    if not ok:
                        geom[mcol_raw] = np.nan
                        geom[stdcol_raw] = np.nan
                        geom[cvcol_raw] = np.nan
                        geom[scol_raw] = np.nan
                        continue

                    img = stack_ch[idx]
                    tmp = pd.DataFrame(
                        regionprops_table(
                            mask,
                            intensity_image=img,
                            properties=(
                                "label",
                                "mean_intensity",
                                "intensity_std",
                            ),
                            extra_properties=(sum_intensity,),
                        )
                    ).rename(
                        columns={
                            "mean_intensity": mcol_raw,
                            "intensity_std": stdcol_raw,
                            "sum_intensity": scol_raw,
                        }
                    )

                    geom = geom.merge(tmp, on="label", how="left")

                    geom[cvcol_raw] = np.divide(
                        geom[stdcol_raw],
                        geom[mcol_raw],
                        out=np.full(len(geom), np.nan, dtype=float),
                        where=geom[mcol_raw].to_numpy(dtype=float) != 0,
                    )

                if basic_checked:
                    for ch in channels:
                        for variant, stacks in (
                            ("BaSiCBgCorrectedRatioFlat", corr_ratio),
                            ("BaSiCBgCorrectedNoRatioFlat", corr_norf),
                        ):
                            mcol = f"Mean{variant}Ch{ch[1:]}{suffix}"
                            stdcol = f"Std{variant}Ch{ch[1:]}{suffix}"
                            cvcol = f"CV{variant}Ch{ch[1:]}{suffix}"
                            scol = f"Sum{variant}Ch{ch[1:]}{suffix}"

                            frame = None
                            if stacks is not None:
                                s = stacks.get(ch)
                                pr = present.get(ch)
                                if (
                                    s is not None
                                    and 0 <= idx < len(s)
                                    and (
                                        pr is None
                                        or (idx < len(pr) and bool(pr[idx]))
                                    )
                                ):
                                    frame = s[idx]

                            if frame is None:
                                geom[mcol] = np.nan
                                geom[stdcol] = np.nan
                                geom[cvcol] = np.nan
                                geom[scol] = np.nan
                                continue

                            tmp = pd.DataFrame(
                                regionprops_table(
                                    mask,
                                    intensity_image=frame,
                                    properties=(
                                        "label",
                                        "mean_intensity",
                                        "intensity_std",
                                    ),
                                    extra_properties=(sum_intensity,),
                                )
                            ).rename(
                                columns={
                                    "mean_intensity": mcol,
                                    "intensity_std": stdcol,
                                    "sum_intensity": scol,
                                }
                            )

                            geom = geom.merge(tmp, on="label", how="left")

                            geom[cvcol] = np.divide(
                                geom[stdcol],
                                geom[mcol],
                                out=np.full(len(geom), np.nan, dtype=float),
                                where=geom[mcol].to_numpy(dtype=float) != 0,
                            )

                geom = geom.rename(
                    columns={
                        "label": f"label{suffix}",
                        "t": f"t{suffix}",
                        "area": f"AreaMorphology{suffix}",
                        "XMorphology": f"XMorphology{suffix}",
                        "YMorphology": f"YMorphology{suffix}",
                        "eccentricity": f"Eccentricity{suffix}",
                        "orientation": f"Orientation{suffix}",
                        "axis_major_length": f"AxisMajorLength{suffix}",
                        "axis_minor_length": f"AxisMinorLength{suffix}",
                        "TouchingBorder": f"TouchingBorder{suffix}",
                    }
                )

                col_order = [
                    f"label{suffix}",
                    f"t{suffix}",
                    f"AreaMorphology{suffix}",
                    f"XMorphology{suffix}",
                    f"YMorphology{suffix}",
                    f"Eccentricity{suffix}",
                    f"Orientation{suffix}",
                    f"AxisMajorLength{suffix}",
                    f"AxisMinorLength{suffix}",
                    f"TouchingBorder{suffix}",
                ]
                for ch in channels:
                    col_order += [
                        f"MeanNoBgCorrectedCh{ch[1:]}{suffix}",
                        f"StdNoBgCorrectedCh{ch[1:]}{suffix}",
                        f"CVNoBgCorrectedCh{ch[1:]}{suffix}",
                        f"SumNoBgCorrectedCh{ch[1:]}{suffix}",
                    ]
                if basic_checked:
                    for ch in channels:
                        for variant in (
                            "BaSiCBgCorrectedRatioFlat",
                            "BaSiCBgCorrectedNoRatioFlat",
                        ):
                            col_order += [
                                f"Mean{variant}Ch{ch[1:]}{suffix}",
                                f"Std{variant}Ch{ch[1:]}{suffix}",
                                f"CV{variant}Ch{ch[1:]}{suffix}",
                                f"Sum{variant}Ch{ch[1:]}{suffix}",
                            ]

                geom = geom[col_order]
                rows.append(geom)

            out = (
                pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
            )
            to_csv_with_reload(out, out_csv_mask, index=False)
            mask_tables[suffix] = out

        if len(mask_tables) > 1:
            matched_csv = f"{stem}_matched{ext}"
            match_mask_dataframes(
                mask_tables=mask_tables,
                threshold_px=threshold_px,
                out_csv=matched_csv,
                position_value=pos_name,
            )

        with suppress(OSError, RuntimeError, ValueError):
            cleanup_memmaps(main_window, remove_viewer_image_layers=False)
        return

    n = len(pos_dirs)
    for i, pos_path in enumerate(pos_dirs, start=1):
        pos_base = os.path.basename(pos_path)
        m = p_re.search(pos_base)
        p_token = f"p{m.group(1)}" if m else pos_base

        out_csv = os.path.join(out_dir, f"Cytometric_Analysis_{p_token}.csv")

        if bar is not None:
            bar.setValue(max(5, int(round((i - 1) * 100 / max(1, n)))))
            bar.setFormat(f"{p_token} (%p%)")
            QCoreApplication.processEvents()

        try:
            quantify_one_position(pos_path, out_csv)
        except (OSError, ValueError, RuntimeError) as e:
            LOG.error(
                "Giving up on position %s after reloading (%r); skipping it "
                "and continuing with the rest.",
                p_token,
                e,
            )

        if bar is not None:
            bar.setValue(max(5, int(round(i * 100 / max(1, n)))))
            bar.setFormat(f"{p_token} (%p%)")
            QCoreApplication.processEvents()

    if bar is not None:
        bar.setValue(100)
        bar.setFormat("Done (%p%)")
        QCoreApplication.processEvents()

    QMessageBox.information(
        getattr(main_window, "experiment_loader_window", main_window),
        "Cytometric Analysis Finished",
        f"Saved {len(pos_dirs) * len(seg_roots)} CSV file(s) to:\n{out_dir}",
    )


def _mask_sort_key(mask_label: str) -> int:
    """Return the numeric suffix of a mask label such as 'M1' or 'M2'."""
    m = re.search(r"(\d+)$", str(mask_label))
    return int(m.group(1)) if m else 10**9


def _compute_anchor_xy(
    df: pd.DataFrame, mask_order: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute x/y coordinates across available mask coordinates."""
    x_cols = [
        f"XMorphology{m}"
        for m in mask_order
        if f"XMorphology{m}" in df.columns
    ]
    y_cols = [
        f"YMorphology{m}"
        for m in mask_order
        if f"YMorphology{m}" in df.columns
    ]

    if not x_cols or not y_cols:
        n = len(df)
        return np.full(n, np.nan), np.full(n, np.nan)

    ax = df[x_cols].mean(axis=1, skipna=True).to_numpy(dtype=float)
    ay = df[y_cols].mean(axis=1, skipna=True).to_numpy(dtype=float)
    return ax, ay


def _match_two_mask_tables(
    current_df: pd.DataFrame,
    next_df: pd.DataFrame,
    next_label: str,
    threshold_px: float,
    mask_order: list[str],
    position_value,
    time_value,
) -> pd.DataFrame:
    """Match rows in current_df to rows in next_df using nearest-neighbor matching.

    Matching is one-to-one and constrained to a distance threshold.
    """
    current_df = current_df.copy().reset_index(drop=True)
    next_df = next_df.copy().reset_index(drop=True)

    current_cols = list(current_df.columns)
    next_cols = [c for c in next_df.columns if c not in current_cols]

    anchor_x, anchor_y = _compute_anchor_xy(current_df, mask_order)

    x_col_next = f"XMorphology{next_label}"
    y_col_next = f"YMorphology{next_label}"

    if x_col_next not in next_df.columns or y_col_next not in next_df.columns:
        rows = []
        all_cols = current_cols + next_cols
        for _, row in current_df.iterrows():
            d = dict.fromkeys(all_cols, pd.NA)
            d.update(row.to_dict())
            d["Position"] = position_value
            d["timepoint"] = time_value
            rows.append(d)

        for _, row in next_df.iterrows():
            d = dict.fromkeys(all_cols, pd.NA)
            d.update(row.to_dict())
            d["Position"] = position_value
            d["timepoint"] = time_value
            rows.append(d)

        out = pd.DataFrame(rows)
        ordered_cols = ["Position", "timepoint"] + [
            c for c in all_cols if c not in ("Position", "timepoint")
        ]
        return out.reindex(columns=ordered_cols)

    next_x = pd.to_numeric(next_df[x_col_next], errors="coerce").to_numpy(
        dtype=float
    )
    next_y = pd.to_numeric(next_df[y_col_next], errors="coerce").to_numpy(
        dtype=float
    )

    # Build all candidate pairs within threshold
    candidates = []
    for i in range(len(current_df)):
        if np.isnan(anchor_x[i]) or np.isnan(anchor_y[i]):
            continue

        dist = np.hypot(next_x - anchor_x[i], next_y - anchor_y[i])
        valid_j = np.where(np.isfinite(dist) & (dist <= float(threshold_px)))[
            0
        ]

        for j in valid_j:
            candidates.append((float(dist[j]), i, j))

    candidates.sort(key=lambda x: x[0])

    used_left = set()
    used_right = set()
    matched_pairs = []

    for dist, i, j in candidates:
        if i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)
        matched_pairs.append((i, j, dist))

    rows = []
    all_cols = current_cols + next_cols

    # matched pairs
    for i, j, _dist in matched_pairs:
        d = dict.fromkeys(all_cols, pd.NA)
        d.update(current_df.loc[i].to_dict())
        d.update(next_df.loc[j].to_dict())
        d["Position"] = position_value
        d["timepoint"] = time_value
        rows.append(d)

    # unmatched current rows
    for i in range(len(current_df)):
        if i in used_left:
            continue
        d = dict.fromkeys(all_cols, pd.NA)
        d.update(current_df.loc[i].to_dict())
        d["Position"] = position_value
        d["timepoint"] = time_value
        rows.append(d)

    # unmatched next rows
    for j in range(len(next_df)):
        if j in used_right:
            continue
        d = dict.fromkeys(all_cols, pd.NA)
        d.update(next_df.loc[j].to_dict())
        d["Position"] = position_value
        d["timepoint"] = time_value
        rows.append(d)

    out = pd.DataFrame(rows)
    ordered_cols = ["Position", "timepoint"] + [
        c for c in all_cols if c not in ("Position", "timepoint")
    ]
    return out.reindex(columns=ordered_cols)


def match_mask_dataframes(
    mask_tables: dict[str, pd.DataFrame],
    threshold_px: float,
    out_csv: str | None = None,
    position_value=None,
    mask_order: list[str] | None = None,
) -> pd.DataFrame:
    """Sequentially match masks across mask labels (M1, M2, M3, ...) per time point and per position."""
    if not mask_tables:
        out = pd.DataFrame()
        if out_csv is not None:
            to_csv_with_reload(out, out_csv, index=False)
        return out

    if mask_order is None:
        mask_order = sorted(mask_tables.keys(), key=_mask_sort_key)

    mask_tables = {
        k: (v.copy() if v is not None else pd.DataFrame())
        for k, v in mask_tables.items()
    }

    # Collect all time points across all mask labels
    all_timepoints = set()
    for mask_label in mask_order:
        df = mask_tables.get(mask_label, pd.DataFrame())
        t_col = f"t{mask_label}"
        if df.empty or t_col not in df.columns:
            continue
        vals = (
            pd.to_numeric(df[t_col], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        )
        all_timepoints.update(vals.tolist())

    all_timepoints = sorted(all_timepoints)

    matched_chunks = []

    for t in all_timepoints:
        available = []
        for mask_label in mask_order:
            df = mask_tables.get(mask_label, pd.DataFrame())
            t_col = f"t{mask_label}"
            if df.empty or t_col not in df.columns:
                continue

            sub = df.loc[pd.to_numeric(df[t_col], errors="coerce") == t].copy()
            if not sub.empty:
                available.append((mask_label, sub.reset_index(drop=True)))

        if not available:
            continue

        # Start with the first available mask label at this time point
        current = available[0][1].copy().reset_index(drop=True)
        current["Position"] = position_value
        current["timepoint"] = t

        # Sequentially match to the remaining available labels
        for next_label, next_df in available[1:]:
            current = _match_two_mask_tables(
                current_df=current,
                next_df=next_df,
                next_label=next_label,
                threshold_px=threshold_px,
                mask_order=mask_order,
                position_value=position_value,
                time_value=t,
            )

        matched_chunks.append(current)

    if matched_chunks:
        out = pd.concat(matched_chunks, ignore_index=True, sort=False)
    else:
        out = pd.DataFrame()

    ordered_cols = ["Position", "timepoint"]
    for mask_label in mask_order:
        preferred = [
            f"label{mask_label}",
            f"t{mask_label}",
            f"AreaMorphology{mask_label}",
            f"XMorphology{mask_label}",
            f"YMorphology{mask_label}",
            f"Eccentricity{mask_label}",
            f"Orientation{mask_label}",
            f"AxisMajorLength{mask_label}",
            f"AxisMinorLength{mask_label}",
            f"TouchingBorder{mask_label}",
        ]
        existing = [c for c in preferred if c in out.columns]
        extra = [
            c
            for c in out.columns
            if c.endswith(mask_label)
            and c not in ordered_cols
            and c not in existing
        ]
        ordered_cols.extend(existing + extra)

    ordered_cols.extend([c for c in out.columns if c not in ordered_cols])

    out = out.reindex(columns=ordered_cols)

    if out_csv is not None:
        to_csv_with_reload(out, out_csv, index=False)

    return out
