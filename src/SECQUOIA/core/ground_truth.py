"""Ground truth scoring of the mask-to-track matching (developer mode)."""

from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import datetime

import numpy as np
import pandas as pd

from SECQUOIA.core.quantification.naming import (
    alt_distance_column,
    alt_label_column,
    centroid_columns,
    distance_column,
    label_column,
)
from SECQUOIA.utils.paths import project_analysis_dir

__all__ = [
    "CATEGORIES",
    "CORRECT",
    "GT_ANNOTATION_ENV",
    "GT_COLUMNS",
    "INCORRECT",
    "MISSED",
    "AnnotationStore",
    "annotation_path",
    "annotation_store",
    "current_row",
    "default_records",
    "gt_annotation_mode",
    "measured_masks",
    "next_open_point",
    "open_masks",
    "record_all_defaults",
    "record_correction",
    "record_judgment",
    "record_key",
    "record_note",
    "snapshot",
]

GT_ANNOTATION_ENV = "SECQUOIA_GT_ANNOTATION"
GT_FILENAME = "ground_truth_annotations.csv"

CORRECT = "Correct"
INCORRECT = "Incorrect"
MISSED = "Missed"
CATEGORIES = (CORRECT, INCORRECT, MISSED)

KEY_COLUMNS = ["Identification", "TrackNumber", "t", "mask_idx"]
GT_COLUMNS = [
    "Identification",
    "Position",
    "TrackNumber",
    "t",
    "mask_idx",
    "assigned_label_id",
    "distance_px",
    "checked_label_id",
    "alt_label_id",
    "alt_distance_px",
    "tracking_x",
    "tracking_y",
    "mask_centroid_x",
    "mask_centroid_y",
    "mask_area_px",
    "category",
    "notes",
    "threshold_px",
    "timestamp",
]


def gt_annotation_mode() -> int:
    """0 without the developer switch, 1 for one shared bar, 2 for a bar per viewer."""
    value = os.environ.get(GT_ANNOTATION_ENV, "").strip().lower()
    if value == "2":
        return 2
    return 1 if value in ("1", "true", "yes", "on") else 0


def _number(value) -> float | None:
    """`value` as a float, or None when it is missing."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _timestamp() -> str:
    """Now, in the format every ``timestamp`` value uses."""
    return datetime.now().isoformat(timespec="seconds")


def _category_for(assigned_label_id: int, checked_label_id: int) -> str:
    """``Correct``/``Incorrect``/``Missed`` from comparing the two label ids."""
    if checked_label_id == assigned_label_id:
        return CORRECT
    return MISSED if assigned_label_id == 0 else INCORRECT


def current_row(main_window) -> pd.Series | None:
    """The ``filtered_df`` row of the current cell, track and time point, or None."""
    df = getattr(main_window, "filtered_df", None)
    if df is None or df.empty:
        return None
    try:
        keep = (
            (df["Identification"].astype(str) == str(main_window.ident))
            & (df["TrackNumber"] == main_window.current_TrackNumber_plot)
            & (df["t"] == main_window.current_time_index)
        )
    except (AttributeError, KeyError, TypeError):
        return None
    rows = df[keep]
    return None if rows.empty else rows.iloc[0]


def record_key(row: pd.Series, mask_idx: int) -> tuple[str, int, int, int]:
    """``(Identification, TrackNumber, t, mask_idx)``, the identity of one judgment."""
    return (
        str(row["Identification"]),
        int(row["TrackNumber"]),
        int(row["t"]),
        int(mask_idx),
    )


def _snapshot_frame(df: pd.DataFrame, mask_idx: int) -> pd.DataFrame:
    """`snapshot` for every row of `df` at once.

    The measured mask values are left empty where no mask is assigned: the
    columns then hold placeholders, not a mask. The second candidate columns
    (the nearest other mask within tolerance, already computed by matching for
    the close-mask flags) are independent of whether the first one is assigned.
    """

    def column(name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        return pd.Series(np.nan, index=df.index)

    label = column(label_column(mask_idx))
    assigned = label > 0
    alt_label = column(alt_label_column(mask_idx)).fillna(0)
    has_alt = alt_label > 0
    x_col, y_col = centroid_columns(mask_idx)
    return pd.DataFrame(
        {
            "assigned_label_id": label.where(assigned, 0).astype(int),
            "distance_px": column(distance_column(mask_idx)).where(assigned),
            "alt_label_id": alt_label.astype(int),
            "alt_distance_px": column(alt_distance_column(mask_idx)).where(
                has_alt
            ),
            "tracking_x": column("XMorphology"),
            "tracking_y": column("YMorphology"),
            "mask_centroid_x": column(x_col).where(assigned),
            "mask_centroid_y": column(y_col).where(assigned),
            "mask_area_px": column(f"AreaMorphologyM{mask_idx}").where(
                assigned
            ),
        }
    )


def snapshot(row: pd.Series, mask_idx: int) -> dict:
    values = _snapshot_frame(pd.DataFrame([row]), mask_idx).iloc[0].to_dict()
    label = int(values.pop("assigned_label_id"))
    alt_label = int(values.pop("alt_label_id"))
    return {
        "assigned_label_id": label,
        "alt_label_id": alt_label,
        **{
            name: None if pd.isna(value) else float(value)
            for name, value in values.items()
        },
    }


def default_records(
    df: pd.DataFrame, masks: Iterable[int], threshold: float | None
) -> pd.DataFrame:
    """A row for every tracked point and mask of `df`, before anyone judged it."""
    if not {"Identification", "TrackNumber", "t"} <= set(df.columns):
        return pd.DataFrame(columns=GT_COLUMNS)
    df = df.reset_index(drop=True)
    track = pd.to_numeric(df["TrackNumber"], errors="coerce")
    time = pd.to_numeric(df["t"], errors="coerce")
    usable = (track.notna() & time.notna()).to_numpy()
    position = (
        pd.to_numeric(df["Position"], errors="coerce")
        if "Position" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    parts = []
    for mask_idx in masks:
        if label_column(mask_idx) not in df.columns:
            continue
        part = pd.DataFrame(
            {
                "Identification": df["Identification"].astype(str),
                "Position": position,
                "TrackNumber": track,
                "t": time,
                "mask_idx": int(mask_idx),
            }
        ).join(_snapshot_frame(df, mask_idx))
        part["checked_label_id"] = part["assigned_label_id"]
        part["category"] = np.where(
            part["assigned_label_id"] > 0, CORRECT, MISSED
        )
        parts.append(part[usable])
    if not parts:
        return pd.DataFrame(columns=GT_COLUMNS)

    records = pd.concat(parts, ignore_index=True)
    records[["TrackNumber", "t"]] = records[["TrackNumber", "t"]].astype(int)
    records["notes"] = ""
    records["threshold_px"] = _number(threshold)
    records["timestamp"] = None
    return records[GT_COLUMNS]


def _checked_label_for(category: str, assigned_label_id: int) -> int:
    return 0 if category == INCORRECT else assigned_label_id


def _build_record(
    row: pd.Series,
    mask_idx: int,
    category: str,
    notes: str,
    threshold: float | None,
) -> dict:
    """One judgment: the key, the snapshot and what the expert decided."""
    if category not in CATEGORIES:
        raise ValueError(
            f"Invalid category: {category}. Must be one of {CATEGORIES}."
        )
    identification, track, t, mask = record_key(row, mask_idx)
    position = _number(row.get("Position"))
    snap = snapshot(row, mask_idx)
    return {
        "Identification": identification,
        "Position": None if position is None else int(position),
        "TrackNumber": track,
        "t": t,
        "mask_idx": mask,
        **snap,
        "checked_label_id": _checked_label_for(
            category, snap["assigned_label_id"]
        ),
        "category": category,
        "notes": notes.strip(),
        "threshold_px": _number(threshold),
        "timestamp": _timestamp(),
    }


def _keys(df: pd.DataFrame) -> pd.MultiIndex:
    """The identity of every judgment in `df`."""
    return pd.MultiIndex.from_frame(df[KEY_COLUMNS])


class AnnotationStore:
    """The judgments of one project, held in memory and saved on every change."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.df = self._load()

    def _load(self) -> pd.DataFrame:
        """The judgments already in the file, so a session can be resumed."""
        if not os.path.exists(self.path):
            return pd.DataFrame(columns=GT_COLUMNS)
        df = pd.read_csv(self.path, dtype={"Identification": str})
        missing = [
            column
            for column in (*KEY_COLUMNS, "category", "notes", "timestamp")
            if column not in df.columns
        ]
        if missing:
            raise ValueError(
                f"{self.path} is not a ground-truth file; missing {missing}."
            )
        for column in ("category", "notes"):
            df[column] = df[column].fillna("").astype(str)
        return df

    def _index_of(self, key: tuple) -> int | None:
        """Row label of the judgment with this key, or None."""
        if self.df.empty:
            return None
        identification, track, t, mask = key
        hit = self.df[
            (self.df["Identification"] == identification)
            & (self.df["TrackNumber"] == track)
            & (self.df["t"] == t)
            & (self.df["mask_idx"] == mask)
        ]
        return None if hit.empty else hit.index[0]

    def get(self, key: tuple) -> dict | None:
        """The saved judgment with this key, or None."""
        index = self._index_of(key)
        return None if index is None else self.df.loc[index].to_dict()

    def upsert(self, record: dict) -> None:
        """Add `record`, replacing the judgment with the same key, and save."""
        index = self._index_of(
            (
                record["Identification"],
                record["TrackNumber"],
                record["t"],
                record["mask_idx"],
            )
        )
        kept = self.df if index is None else self.df.drop(index=index)
        new = pd.DataFrame([record])
        self.df = (
            new if kept.empty else pd.concat([kept, new], ignore_index=True)
        )
        self.save()

    def sync_defaults(
        self, df: pd.DataFrame, masks: Iterable[int], threshold: float | None
    ) -> None:
        """Give every tracked point and mask of `df` a row, and save."""
        fresh = default_records(df, masks, threshold)
        clicked = self.df[self.df["timestamp"].notna()]
        untouched = self.df[
            self.df["timestamp"].isna() & ~_keys(self.df).isin(_keys(fresh))
        ]
        added = fresh[~_keys(fresh).isin(_keys(clicked))]
        pieces = [p for p in (clicked, untouched, added) if not p.empty]
        self.df = (
            pd.concat(pieces, ignore_index=True)
            if pieces
            else pd.DataFrame(columns=GT_COLUMNS)
        )
        self.save()

    def save(self) -> None:
        """Write the table in the canonical column order, sorted by key."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = f"{self.path}.tmp"
        self.df[GT_COLUMNS].sort_values(KEY_COLUMNS).to_csv(tmp, index=False)
        os.replace(tmp, self.path)


def annotation_path(main_window) -> str | None:
    """Where this project's judgments live, or None before one is loaded."""
    if not getattr(main_window, "folder", None):
        return None
    try:
        return os.path.join(project_analysis_dir(main_window), GT_FILENAME)
    except ValueError:  # the tracking format is not known yet
        return None


def annotation_store(main_window) -> AnnotationStore | None:
    """The store of the loaded project, reopened when another one is loaded."""
    path = annotation_path(main_window)
    if path is None:
        return None
    store = getattr(main_window, "_gt_store", None)
    if store is None or store.path != path:
        store = AnnotationStore(path)
        main_window._gt_store = store
    return store


def record_judgment(
    main_window, mask_idx: int, category: str, notes: str = ""
) -> dict | None:
    """Store the judgment of the mask assigned to the current row."""
    row = current_row(main_window)
    store = annotation_store(main_window)
    if row is None or store is None:
        return None
    record = _build_record(
        row,
        mask_idx,
        category,
        notes,
        getattr(main_window, "threshold", None),
    )
    store.upsert(record)
    return record


def record_note(main_window, mask_idx: int, notes: str) -> dict | None:
    """Save `notes` for the current row, changing nothing else about it."""
    row = current_row(main_window)
    store = annotation_store(main_window)
    if row is None or store is None:
        return None
    saved = store.get(record_key(row, mask_idx))
    if saved is None or saved["category"] not in CATEGORIES:
        return None
    saved["notes"] = notes.strip()
    saved["timestamp"] = _timestamp()
    store.upsert(saved)
    return saved


def record_correction(main_window, target) -> dict | None:
    """Record a right-click or paint correction as the checked label."""
    store = annotation_store(main_window)
    if store is None:
        return None
    key = (
        str(target.ident),
        int(target.track_no),
        int(target.t),
        int(target.mask_idx),
    )
    saved = store.get(key)
    if saved is None:
        return None
    assigned = int(saved["assigned_label_id"])
    checked = int(target.label_id)
    saved["checked_label_id"] = checked
    saved["category"] = _category_for(assigned, checked)
    saved["timestamp"] = _timestamp()
    store.upsert(saved)
    return saved


def measured_masks(df: pd.DataFrame, masks: Iterable[int]) -> list[int]:
    """The masks of `masks` that `df` has a label column for."""
    return [
        mask_idx for mask_idx in masks if label_column(mask_idx) in df.columns
    ]


def open_masks(main_window, masks: Iterable[int]) -> list[int] | None:
    """The masks of the current point that nobody has judged yet."""
    row = current_row(main_window)
    store = annotation_store(main_window)
    if row is None or store is None:
        return None
    measured = measured_masks(main_window.filtered_df, masks)
    if not measured:
        return None
    unjudged = []
    for mask_idx in measured:
        saved = store.get(record_key(row, mask_idx))
        if saved is None or pd.isna(saved["timestamp"]):
            unjudged.append(mask_idx)
    return unjudged


def record_all_defaults(main_window, masks: Iterable[int]) -> None:
    """Accept every open mask of the current point exactly as it already stands."""
    row = current_row(main_window)
    store = annotation_store(main_window)
    if row is None or store is None:
        return
    for mask_idx in open_masks(main_window, masks) or []:
        saved = store.get(record_key(row, mask_idx))
        if saved is not None and saved["category"] in CATEGORIES:
            saved["timestamp"] = _timestamp()
            store.upsert(saved)


def _points(df: pd.DataFrame) -> pd.DataFrame:
    points = pd.DataFrame(
        {
            "Identification": df["Identification"].astype(str),
            "TrackNumber": pd.to_numeric(df["TrackNumber"], errors="coerce"),
            "t": pd.to_numeric(df["t"], errors="coerce"),
        }
    ).dropna()
    return points.astype({"TrackNumber": int, "t": int}).drop_duplicates()


def next_open_point(main_window) -> tuple[str, int, int] | None:
    """The next ``(Identification, TrackNumber, t)`` with a mask nobody judged."""
    df = getattr(main_window, "filtered_df", None)
    store = annotation_store(main_window)
    if df is None or store is None or store.df.empty:
        return None
    unjudged = store.df[store.df["timestamp"].isna()]
    open_points = _points(unjudged).merge(_points(df))
    rank = {
        str(ident): n
        for n, ident in enumerate(getattr(main_window, "unique_ids", []))
    }

    def order(point: tuple[str, int, int]) -> tuple[int, str, int, int]:
        ident, track, t = point
        return (rank.get(ident, len(rank)), ident, track, t)

    points = sorted(
        (
            (str(ident), int(track), int(t))
            for ident, track, t in open_points.itertuples(
                index=False, name=None
            )
        ),
        key=order,
    )
    if not points:
        return None
    try:
        current = order(
            (
                str(main_window.ident),
                int(main_window.current_TrackNumber_plot),
                int(main_window.current_time_index),
            )
        )
    except (AttributeError, TypeError, ValueError):
        return points[0]
    return next((p for p in points if order(p) > current), points[0])
