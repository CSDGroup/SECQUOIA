"""Loading, filtering, and caching of the tracking/measurement DataFrame."""

from __future__ import annotations

import contextlib
import logging
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from skimage.io import imread
from skimage.measure import regionprops_table

from SECQUOIA.config import BTRACK_REQUIREMENT, FEATURES, NAPARIPARAMETERS
from SECQUOIA.core.normalization import apply_normalization
from SECQUOIA.core.quantification import (
    FeatureNaming,
    drop_stale_measurement_columns,
    intensity_column,
    safe_op_series,
)
from SECQUOIA.core.segmentation.mask_selection import (
    rebuild_segmentation_layer_index,
    set_active_layers_from_header_buttons,
)
from SECQUOIA.core.tracking.clt_io import CLTParser
from SECQUOIA.core.tracking.numbering import _binary_tracknumbers
from SECQUOIA.gui.curation_tree import update_list
from SECQUOIA.utils.io_reload import read_with_reload, to_csv_with_reload
from SECQUOIA.utils.positions import (
    current_position_number,
    position_index_from_number,
)
from SECQUOIA.utils.timing import apply_realtime, build_realtime_lookup

LOG = logging.getLogger(__name__)

__all__ = [
    "BTRACK_MISSING_MESSAGE",
    "apply_derived_features",
    "assign_track_numbers",
    "btrack_is_available",
    "collect_positions_to_track_df_parallel",
    "ctc_to_long",
    "filter_track_df_to_selected",
    "import_btrack",
    "load_tracks_from_folder",
    "load_ultrack_with_tracknumber",
    "recompute_derived_for_row",
    "tracks_to_dataframe",
    "update_track_df",
]


def apply_derived_features(
    main_window,
    df: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """Replay every saved arithmetic metric onto a dataframe."""
    reg = getattr(main_window, "_derived_features", {}) or {}
    if not reg:
        return df

    inplace_on_track_df = df is None
    if inplace_on_track_df:
        df = getattr(main_window, "track_df", None)

    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    for desc in reg.values():
        if not isinstance(desc, dict):
            continue
        fml = desc.get("formula")
        if not isinstance(fml, dict):
            continue

        col_a = fml.get("source_col_a")
        col_b = fml.get("source_col_b")
        col_t = fml.get("target_col")
        op = str(fml.get("op", "/"))
        if not col_t:
            continue

        if col_a in df.columns and col_b in df.columns:
            try:
                df[col_t] = safe_op_series(df[col_a], df[col_b], op)
            except (TypeError, ValueError, KeyError) as err:
                LOG.warning("[derived] compute failed for %s: %s", col_t, err)
                if col_t not in df.columns:
                    df[col_t] = np.nan
        else:
            if col_t in df.columns:
                LOG.info(
                    "[derived] Sources of %s are missing (%s, %s); "
                    "clearing its stale values.",
                    col_t,
                    col_a,
                    col_b,
                )
            df[col_t] = np.nan

    if inplace_on_track_df:
        main_window.track_df = df

    return df


def _formula_uses_changed_mask(mask_idx: int | None, *cols: str) -> bool:
    """Return whether any column name refers to the mask index that just changed."""
    if mask_idx is None:
        return True

    pattern = re.compile(rf"M0*{int(mask_idx)}(?!\d)", re.IGNORECASE)
    return any(pattern.search(str(c)) for c in cols if c)


def _recompute_derived_column(
    df: pd.DataFrame,
    desc: dict,
    row_idx: int | None,
    row_exists: bool,
    mask_idx: int | None,
) -> None:
    """Recompute one derived metric's column in ``df``, in place."""
    fml = desc.get("formula")
    if not isinstance(fml, dict):
        return

    col_a = fml.get("source_col_a")
    col_b = fml.get("source_col_b")
    col_t = fml.get("target_col")
    op = fml.get("op", "/")

    if not col_a or not col_b or not col_t:
        return

    if not _formula_uses_changed_mask(mask_idx, col_a, col_b, col_t):
        return

    if col_a not in df.columns or col_b not in df.columns:
        return

    if col_t not in df.columns:
        df[col_t] = np.nan

    norm_enabled = bool(fml.get("norm_enabled", False))

    if norm_enabled or not row_exists:
        df[col_t] = safe_op_series(df[col_a], df[col_b], op)

        if norm_enabled:
            apply_normalization(
                df,
                col_t,
                fml.get("norm_method", "zscore"),
                fml.get("norm_scope", "all"),
                int(
                    fml.get(
                        "norm_t_min",
                        df["t"].min() if "t" in df.columns else 0,
                    )
                ),
                int(
                    fml.get(
                        "norm_t_max",
                        df["t"].max() if "t" in df.columns else 0,
                    )
                ),
            )
    else:
        df.loc[[row_idx], col_t] = safe_op_series(
            df.loc[[row_idx], col_a],
            df.loc[[row_idx], col_b],
            op,
        )


def recompute_derived_for_row(
    main_window,
    row_idx: int | None = None,
    *,
    mask_idx: int | None = None,
) -> None:
    """Reapply the saved metric rules after measurements changed."""
    df = getattr(main_window, "track_df", None)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return

    registry = getattr(main_window, "_derived_features", {}) or {}
    if not isinstance(registry, dict):
        return

    row_exists = row_idx is not None and row_idx in df.index

    for desc in registry.values():
        if isinstance(desc, dict):
            _recompute_derived_column(df, desc, row_idx, row_exists, mask_idx)

    main_window.track_df = df


def _refresh_filtered_df(main_window, position=None) -> None:
    """Rebuild ``main_window.filtered_df`` from ``track_df`` for one position."""
    df = getattr(main_window, "track_df", None)
    if not isinstance(df, pd.DataFrame) or "Position" not in df.columns:
        return
    if position is None:
        position = getattr(main_window, "current_position_number", None)
    if position is None:
        return
    with contextlib.suppress(
        RuntimeError, AttributeError, TypeError, KeyError
    ):
        main_window.filtered_df = df[df["Position"] == position].copy()


def _load_tracking_data(main_window) -> None:
    """Load tracking data and validate that the required columns are present."""
    if (
        main_window.tracking_format == "CTC"
        or main_window.tracking_format == "btrack"
        or main_window.tracking_format == "Ultrack"
    ):
        if not os.path.isdir(main_window.tracking_path):
            raise ValueError(
                f"CTC or btrack or Ultrack: tracking_path is not a directory: {main_window.tracking_path}"
            )

        main_window.track_df = collect_positions_to_track_df_parallel(
            main_window.tracking_path,
            tracking_format=main_window.tracking_format,
        )
        required_parameters = [
            "track_id",
            "x",
            "y",
            "t",
            "Position",
            "TrackNumber",
        ]
        missing_parameters = [
            p
            for p in required_parameters
            if p not in main_window.track_df.columns
        ]

        if missing_parameters:
            LOG.error("Missing parameters: %s", ", ".join(missing_parameters))
            return

        LOG.debug("All required parameters are present.")

        main_window.track_df = main_window.track_df[
            [
                p
                for p in required_parameters
                if p in main_window.track_df.columns
            ]
        ]
        main_window.track_df.rename(
            columns={"x": "XMorphology", "y": "YMorphology"}, inplace=True
        )

        main_window.track_df["Identification"] = main_window.track_df.apply(
            lambda row: (
                f"{main_window.experiment_name}-p{int(row['Position']):04d}-{int(row['track_id']):03d}"
                if pd.notnull(row["track_id"]) and not pd.isna(row["track_id"])
                else None
            ),
            axis=1,
        )
        main_window.track_df["TrackNumber"] = main_window.track_df[
            "TrackNumber"
        ].astype(float)
        main_window.threshold = 10

    elif main_window.tracking_format == "tTt":
        main_window.clt_parser = CLTParser(main_window.xml_path)

        main_window.track_df = main_window.clt_parser.load_folder(
            folder_exp=main_window.tracking_path,
            folder_seg="",
            cp_tracking=bool(main_window.cp_tracking),
            progress_fun=None,
            dry_run=True,
        )
        main_window.track_df["Identification"] = main_window.track_df[
            "Identification"
        ].str.replace(r"^(.*)_(p\d{4}-\d{3})$", r"\1-\2", regex=True)
        main_window.track_df["TrackNumber"] = main_window.track_df[
            "TrackNumber"
        ].astype(float)
        main_window.threshold = 10
    else:
        raise ValueError(
            f"Unsupported tracking format {main_window.tracking_format}"
        )


def ctc_to_long(folder: str) -> pd.DataFrame:
    """Convert a CTC result folder to a long DataFrame."""
    folder = Path(folder)

    track_path = folder / "res_track.txt"
    if not track_path.exists():
        raise FileNotFoundError(f"Missing {track_path}")

    with track_path.open("r") as f:
        first = next(
            line
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        )
    has_header = any(c.isalpha() for c in first)

    tracks = pd.read_csv(
        track_path,
        sep=r"\s+",
        engine="python",
        header=0 if has_header else None,
        comment="#",
    )
    if not has_header:
        tracks.columns = ["L", "B", "E", "P"]
    tracks = tracks[["L", "B", "E", "P"]].astype(int)

    parent = dict(zip(tracks["L"], tracks["P"], strict=False))
    root_cache: dict[int, int] = {}

    def find_root(label_id: int) -> int:
        if label_id in root_cache:
            return root_cache[label_id]
        p = parent.get(label_id, 0)
        r = label_id if p == 0 else find_root(p)
        root_cache[label_id] = r
        return r

    tracks["root"] = tracks["L"].map(find_root)
    tracks["track_id"] = tracks["root"]
    roots = sorted(tracks.loc[tracks["P"] == 0, "L"].unique())

    children: dict[int, list[int]] = {}
    for _, row in tracks.iterrows():
        children.setdefault(int(row["P"]), []).append(int(row["L"]))

    b_map = dict(zip(tracks["L"], tracks["B"], strict=False))
    for p in children:
        children[p].sort(
            key=lambda label_id: (b_map.get(label_id, 10**9), label_id)
        )

    tracknum = _binary_tracknumbers(roots, children)

    tracks["TrackNumber"] = tracks["L"].map(tracknum).astype(int)

    mask_files = sorted(folder.glob("mask*.tif"))
    if not mask_files:
        raise FileNotFoundError(f"No mask*.tif files found in {folder}")

    rows = []
    for t, fp in enumerate(mask_files):
        lab = imread(fp)
        rp = regionprops_table(lab, properties=("label", "centroid"))
        if len(rp["label"]) == 0:
            continue
        rows.append(
            pd.DataFrame(
                {
                    "L": rp["label"],
                    "t": t,
                    "y": rp["centroid-0"],
                    "x": rp["centroid-1"],
                }
            )
        )

    coords = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["L", "t", "y", "x"])
    )
    coords["L"] = coords["L"].astype(int)

    out = coords.merge(
        tracks[["L", "track_id", "TrackNumber", "B", "E"]], on="L", how="inner"
    )
    out = out[(out["t"] >= out["B"]) & (out["t"] <= out["E"])].copy()

    out = out[["track_id", "t", "y", "x", "TrackNumber"]].sort_values(
        ["track_id", "t", "TrackNumber"]
    )
    out.reset_index(drop=True, inplace=True)
    return out


def collect_positions_to_track_df_parallel(
    parent_folder: str,
    tracking_format: str,
    workers: int = 4,
    obj_type: str = "obj_type_1",
) -> pd.DataFrame:
    """Load every position folder in parallel and concatenate them into one track_df."""
    parent = Path(parent_folder)
    pos_re = re.compile(r"_p(\d+)$")

    subs = []
    for sub in parent.iterdir():
        if sub.is_dir():
            m = pos_re.search(sub.name)
            if m:
                subs.append((sub, int(m.group(1))))

    def run_one(item: tuple[Path, int]) -> pd.DataFrame:
        sub, position = item

        if tracking_format == "CTC":
            df = ctc_to_long(str(sub))
        elif tracking_format == "Ultrack":
            df = load_ultrack_with_tracknumber(str(sub))
        else:
            tracks = load_tracks_from_folder(sub, obj_type=obj_type)
            df = tracks_to_dataframe(tracks)

        df["Position"] = position
        return df

    with ThreadPoolExecutor(max_workers=workers) as ex:
        dfs = list(ex.map(run_one, sorted(subs, key=lambda x: x[1])))

    dfs = [d for d in dfs if d is not None and not d.empty]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_ultrack_with_tracknumber(folder: str | Path) -> pd.DataFrame:
    """Read Ultrack CSVs in a folder and add a TrackNumber per lineage.

    Uses the same tree numbering (root=1, daughters=2*parent[+1])
    produced by ``ctc_to_long``, so the division/split/fuse editing tools
    work the same regardless of import format.
    """
    files = sorted(Path(folder).glob("*.csv"))
    if not files:
        return pd.DataFrame()

    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    tcol = next((c for c in ("t", "frame", "time") if c in df.columns), None)
    if tcol is None:
        return pd.DataFrame()

    df["track_id"] = df["track_id"].astype(int)
    df["parent_track_id"] = df["parent_track_id"].fillna(-1).astype(int)

    meta = (
        df.sort_values(tcol)
        .groupby("track_id", as_index=False)
        .agg(parent=("parent_track_id", "first"), birth=(tcol, "min"))
    )
    parent = dict(zip(meta["track_id"], meta["parent"], strict=False))

    @lru_cache(None)
    def root(t: int) -> int:
        p = parent.get(t, -1)
        return t if p == -1 or p == t or p not in parent else root(p)

    meta["LineageID"] = meta["track_id"].map(root)

    birth = dict(zip(meta["track_id"], meta["birth"], strict=False))
    children: dict[int, list[int]] = {}
    for track_id, p in parent.items():
        if p != -1 and p != track_id and p in parent:
            children.setdefault(p, []).append(track_id)
    for p, kids in children.items():
        children[p] = sorted(
            kids, key=lambda tid: (birth.get(tid, 10**9), tid)
        )

    roots = sorted(meta.loc[meta["track_id"] == meta["LineageID"], "track_id"])
    tracknum = _binary_tracknumbers(roots, children)
    meta["TrackNumber"] = meta["track_id"].map(tracknum).astype(int)

    out = df.merge(
        meta[["track_id", "LineageID", "TrackNumber"]],
        on="track_id",
        how="left",
    )
    out = out.drop(columns="track_id").rename(
        columns={"LineageID": "track_id"}
    )
    out = out.sort_values(["track_id", tcol, "TrackNumber"]).reset_index(
        drop=True
    )
    return out


BTRACK_MISSING_MESSAGE = (
    "SECQUOIA does not automatically install btrack in this enviorment,\n\n"
    "Install it yourself with:\n\n"
    f'    pip install "{BTRACK_REQUIREMENT}"\n\n'
    "Then restart SECQUOIA and load the data again."
)


def import_btrack():
    """Return the ``btrack`` module, importing it on first use."""
    try:
        import btrack
    except ImportError as exc:
        raise ImportError(BTRACK_MISSING_MESSAGE) from exc
    return btrack


def btrack_is_available() -> bool:
    """Whether btrack can be imported here."""
    try:
        import_btrack()
    except ImportError:
        return False
    return True


def load_tracks_from_folder(
    folder: Path, obj_type: str = "obj_type_1"
) -> list:
    """Read all btrack HDF5 files in a folder and return their combined tracks."""
    folder = Path(folder)
    h5_files = sorted(folder.glob("*.h5"))
    if not h5_files:
        return []
    btrack = import_btrack()
    tracks = []
    for f in h5_files:
        with btrack.io.HDF5FileHandler(f, "r", obj_type=obj_type) as reader:
            tracks.extend(reader.tracks)
    return tracks


def assign_track_numbers(tracks: list) -> dict[int, int]:
    """Assign a per lineage TrackNumber to each track.

    Uses the same tree numbering (root=1, daughters=2*parent[+1])
    as ``ctc_to_long``, so the division/split/fuse editing tools behave
    consistently regardless of whether tracks were imported via CTC,
    Ultrack, or btrack.
    """
    by_id = {tr.ID: tr for tr in tracks}
    by_root = defaultdict(list)
    for tr in tracks:
        root_id = tr.root if tr.root is not None else tr.ID
        by_root[root_id].append(tr)

    tracknum: dict[int, int] = {}

    for root_id, lineage in by_root.items():
        root_track = next(
            (
                tr
                for tr in lineage
                if getattr(tr, "is_root", False) or root_id == tr.ID
            ),
            min(lineage, key=lambda tr: (tr.start, tr.ID)),
        )

        children: dict[int, list[int]] = {}
        for parent in lineage:
            child_ids = [c for c in (parent.children or []) if c in by_id]
            if child_ids:
                children[parent.ID] = sorted(
                    child_ids, key=lambda c: (by_id[c].start, c)
                )

        tracknum.update(_binary_tracknumbers([root_track.ID], children))

        for tr in sorted(
            (t for t in lineage if t.ID not in tracknum),
            key=lambda tr: (tr.start, tr.ID),
        ):
            tracknum[tr.ID] = max(tracknum.values(), default=0) + 1

    return tracknum


def tracks_to_dataframe(tracks: list) -> pd.DataFrame:
    """Flatten btrack tracks into a long DataFrame with track_id, t, TrackNumber, x, y."""
    if not tracks:
        return pd.DataFrame(columns=["track_id", "t", "TrackNumber", "x", "y"])

    tn = assign_track_numbers(tracks)

    rows = []
    for tr in tracks:
        lineage_track_id = tr.root if tr.root is not None else tr.ID
        for t, x, y in zip(tr.t, tr.x, tr.y, strict=False):
            rows.append(
                {
                    "track_id": int(lineage_track_id),
                    "t": int(t),
                    "TrackNumber": int(tn[tr.ID]),
                    "x": float(x),
                    "y": float(y),
                }
            )

    df = pd.DataFrame(rows)
    return df.sort_values(
        ["track_id", "t", "TrackNumber"], kind="mergesort", ignore_index=True
    )


def filter_track_df_to_selected(main_window) -> None:
    """Keep only (Position, Identification) rows that are checked in the Tracking tree."""
    df = getattr(main_window, "track_df", None)
    tree = getattr(main_window, "tracking_tree", None)
    if df is None or df.empty or tree is None:
        return

    keep = set()
    for i in range(tree.topLevelItemCount()):
        parent = tree.topLevelItem(i)
        try:
            pos = int(str(parent.text(0)).split()[-1])
        except (RuntimeError, AttributeError, TypeError):
            pos = parent.text(0)

        for j in range(parent.childCount()):
            ch = parent.child(j)
            if ch.checkState(0) == Qt.Checked:
                keep.add((pos, ch.text(0)))

    if not keep:
        return

    row_keys = pd.MultiIndex.from_arrays(
        [df["Position"], df["Identification"].astype(str)]
    )
    mask = row_keys.isin(keep)

    track_df = df if mask.all() else df.loc[mask].copy()
    main_window.track_df = track_df

    current_pos = getattr(main_window, "current_position_number", None)
    if current_pos is None or "Position" not in track_df.columns:
        return

    positions = (
        pd.to_numeric(track_df["Position"], errors="coerce")
        .dropna()
        .astype(int)
        .to_numpy()
    )
    if positions.size == 0:
        return

    available = np.unique(positions)
    current_pos = int(current_pos)
    if current_pos in available:
        return

    insert_at = int(np.searchsorted(available, current_pos, side="right"))
    next_pos = int(
        available[insert_at] if insert_at < available.size else available[0]
    )

    main_window.current_position_number = next_pos
    main_window.position_start_selected = next_pos
    main_window.current_position_index = position_index_from_number(
        main_window, next_pos
    )


def update_track_df(main_window) -> None:
    """Update the track DataFrame based on the current position index.

    Filters it and updates both viewers' segmentation layer selections
    dynamically for all masks.
    """
    if getattr(main_window, "track_df", None) is None:
        LOG.warning("No track_df found.")
        return

    position_number = current_position_number(main_window)
    if position_number is None:
        LOG.warning("Could not parse position number.")
        return

    filtered_df = main_window.track_df[
        main_window.track_df["Position"] == position_number
    ].copy()
    main_window.filtered_df = filtered_df

    if "Identification" in filtered_df.columns:
        main_window.unique_ids = filtered_df["Identification"].unique()
    else:
        main_window.unique_ids = None

    main_window.current_ident_index = 0
    main_window.current_TrackNumber_plot = 1

    mask_indices: list[int] = []
    try:
        mc = getattr(main_window, "n_masks", 0)
        if mc > 0:
            mask_indices = list(range(1, mc + 1))
    except (RuntimeError, AttributeError, TypeError):
        pass
    if not mask_indices:
        pat = re.compile(r"^label_id_m(\d+)$")
        indices = []
        for c in filtered_df.columns:
            m = pat.match(c)
            if m:
                with contextlib.suppress(
                    RuntimeError, AttributeError, TypeError, ValueError
                ):
                    indices.append(int(m.group(1)))
        mask_indices = sorted(set(indices)) if indices else []

    mask_ids_dict = {}
    for m in mask_indices:
        col = f"label_id_m{m}"
        if col in filtered_df.columns:
            try:
                vals = filtered_df[col].dropna().astype("Int64").to_numpy()
            except (RuntimeError, AttributeError, TypeError):
                vals = filtered_df[col].dropna().to_numpy()
            mask_ids_dict[m] = np.unique(vals)

    first_row = filtered_df.iloc[0] if not filtered_df.empty else None

    def _apply_selection_to_viewer(viewer, row, mask_idxs):
        """Point one viewer's segmentation layers at ``row``'s labels.

        A ``label_id_m{m}`` above 0 selects that label on
        ``Segmentation{m}`` and hides the rest; 0, NaN or a missing
        column shows all labels for that layer. Finally centres and zooms
        the camera on the row.
        """
        if viewer is None or not hasattr(viewer, "layers") or row is None:
            return

        for m in mask_idxs:
            layer_name = f"Segmentation{m}"
            if layer_name not in viewer.layers:
                continue

            layer = viewer.layers[layer_name]
            lab_col = f"label_id_m{m}"
            val = row.get(lab_col, 0)

            try:
                if pd.isna(val):
                    val = 0
            except (RuntimeError, AttributeError, TypeError):
                pass

            try:
                val_int = int(val)
            except (RuntimeError, AttributeError, TypeError):
                val_int = 0

            if val_int > 0:
                with contextlib.suppress(
                    RuntimeError, AttributeError, TypeError
                ):
                    layer.show_selected_label = True
                    layer.selected_label = val_int
            else:
                with contextlib.suppress(
                    RuntimeError, AttributeError, TypeError
                ):
                    layer.show_selected_label = False

        with contextlib.suppress(Exception):
            viewer.camera.center = (
                first_row["YMorphology"],
                first_row["XMorphology"],
            )
            viewer.camera.zoom = NAPARIPARAMETERS.ZOOMFACTOR

    # Apply to both viewers
    _apply_selection_to_viewer(
        getattr(main_window, "viewer_1", None),
        first_row,
        mask_indices,
    )
    _apply_selection_to_viewer(
        getattr(main_window, "viewer_2", None),
        first_row,
        mask_indices,
    )

    first_labels = {}
    for m in mask_indices:
        lab_col = f"label_id_m{m}"
        if first_row is None:
            first_labels[m] = 0
            continue
        try:
            v = first_row.get(lab_col, 0)
            v = 0 if pd.isna(v) else int(v)
        except (RuntimeError, AttributeError, TypeError):
            v = 0
        first_labels[m] = v

    rebuild_segmentation_layer_index(main_window)
    set_active_layers_from_header_buttons(main_window)
    main_window.update_display_track_id()
    update_list(main_window)


def _apply_import_realtime_to_track_df(main_window) -> None:
    """Merge per frame real-time acquisition timestamps from the import CSV into track_df."""
    wide = build_realtime_lookup(main_window, force=True)
    if wide is None:
        return
    main_window.track_df = apply_realtime(main_window.track_df, wide)


def _position_csv_path(
    main_window, position: int, t_file_min: int, t_file_max: int
) -> str:
    """Build the cached-measurements CSV path for one position."""
    experiment_root = getattr(main_window, "folder", None)
    save_folder = os.path.join(
        experiment_root,
        "Analysis",
        f"SECQUOIA_files_{main_window.tracking_format}",
        (getattr(main_window, "project_name", "") or "Project_1").strip(),
    )
    os.makedirs(save_folder, exist_ok=True)

    fname = (
        f"SECQUOIA_p{position:04d}"
        f"_t{t_file_min:05d}-{t_file_max:05d}"
        f"_m{main_window.n_masks}_ch{main_window.n_channels}.csv"
    )
    return os.path.join(save_folder, fname)


def _cache_position_measurements(
    main_window,
    position: int,
    t_file_min: int,
    t_file_max: int,
    *,
    log_prefix: str = "csv-cache",
) -> None:
    """Load one position's cached measurements CSV, if it exists."""
    df = getattr(main_window, "track_df", None)
    if not isinstance(df, pd.DataFrame):
        return

    df_pos = df[df.get("Position", -1) == position].copy()
    if df_pos.empty:
        LOG.warning(
            "[%s] No rows to save for Position %s.", log_prefix, position
        )
        return

    path = _position_csv_path(main_window, position, t_file_min, t_file_max)
    try:
        to_csv_with_reload(df_pos, path, index=False)
    except (OSError, ValueError, RuntimeError) as e:
        LOG.error(
            "[%s] Failed to save position CSV after reloading: %r",
            log_prefix,
            e,
        )
        return
    LOG.info("[%s] Saved position CSV: %s", log_prefix, path)


def _required_measurement_columns(main_window) -> list[str]:
    """The base measurement columns every cached position CSV should carry.

    Intensity names are built through `intensity_column` so they match the
    channel token `quantify` writes, rather than a bare channel index.
    """
    channels = list(getattr(main_window, "ids_channels", None) or [])
    masks = range(1, int(main_window.n_masks) + 1)

    required = [
        intensity_column(f"{base}NoBgCorrected", channel, mask)
        for channel in channels
        for mask in masks
        for base in FEATURES.METRIC_PREFIXES
    ]
    for mask in masks:
        required += [f"{prefix}M{mask}" for prefix in FEATURES.MORPH_PREFIXES]
        required += [f"label_id_m{mask}", f"nn_dist_px_m{mask}"]
    return required


def _prune_stale_measurement_columns(
    main_window, df: pd.DataFrame, log_prefix: str
) -> pd.DataFrame:
    """Drop measurement columns the current configuration no longer writes."""
    try:
        naming = FeatureNaming.from_main_window(main_window)
        mask_indices = range(1, int(main_window.n_masks) + 1)
    except (AttributeError, TypeError, ValueError) as err:
        LOG.warning(
            "[%s] Cannot determine the measured configuration, keeping all "
            "columns: %s",
            log_prefix,
            err,
        )
        return df
    return drop_stale_measurement_columns(
        df, naming, mask_indices, log_prefix=log_prefix
    )


def _load_cached_position_measurements(
    main_window,
    position: int,
    t_file_min: int,
    t_file_max: int,
    *,
    ensure_columns: bool = False,
    log_prefix: str = "csv-cache",
) -> pd.DataFrame | None:
    """Load a cached measurements CSV for one position, if present, merging it into ``track_df``.

    Replaces any existing rows for ``position`` in ``track_df`` with the cached
    rows and sets ``main_window.filtered_df`` to them. Returns the loaded
    per position DataFrame, or ``None`` if no cache existed or the load failed.
    """
    path = _position_csv_path(main_window, position, t_file_min, t_file_max)
    if not os.path.isfile(path):
        return None

    try:
        df_pos = read_with_reload(pd.read_csv, path)

        required_cols: list[str] = []
        try:
            required_cols = _required_measurement_columns(main_window)
        except (AttributeError, TypeError, ValueError) as err:
            LOG.warning("[%s] base col names warning: %s", log_prefix, err)

        df_pos = _prune_stale_measurement_columns(
            main_window, df_pos, log_prefix
        )

        if ensure_columns:
            if "Position" in df_pos.columns:
                with contextlib.suppress(
                    RuntimeError, AttributeError, TypeError
                ):
                    df_pos["Position"] = df_pos["Position"].astype(
                        int, errors="ignore"
                    )
            for col in required_cols:
                if col not in df_pos.columns:
                    df_pos[col] = np.nan

        base_df = getattr(main_window, "track_df", pd.DataFrame())
        base_df = base_df[base_df.get("Position", -1) != position]
        main_window.track_df = _prune_stale_measurement_columns(
            main_window,
            pd.concat([base_df, df_pos], ignore_index=True),
            log_prefix,
        )
        main_window.filtered_df = df_pos
        LOG.info("[%s] Loaded cached measurements: %s", log_prefix, path)
        return df_pos
    except (RuntimeError, AttributeError, TypeError, OSError, ValueError) as e:
        LOG.warning(
            "[%s] Failed to load cached CSV (%s); will re-measure. Error: %s",
            log_prefix,
            path,
            e,
        )
        return None
