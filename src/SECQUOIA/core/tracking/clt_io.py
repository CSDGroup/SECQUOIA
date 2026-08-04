"""Reading and writing tTt CLT tracking files.

Parses ``%% TrackingData`` blocks into a track_df and writes SECQUOIA
measurements back out as ``%% Quantification`` blocks. TAT metadata,
when supplied, converts between global microscope coordinates and
pixels.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Iterator
from typing import NamedTuple

import numpy as np
import pandas as pd

from SECQUOIA.core.tatexp_xml import TATParser
from SECQUOIA.utils.io_reload import (
    read_text_with_reload,
    write_text_with_reload,
)

__all__ = [
    "CLTParser",
]

LOG = logging.getLogger(__name__)


def _as_scalar(val, default=None):
    """Collapse an accidental series/array to a single value."""
    if isinstance(val, pd.Series):
        return val.iloc[0] if not val.empty else default
    if isinstance(val, np.ndarray):
        return val.flat[0] if val.size else default
    return val


class _ColumnSchema(NamedTuple):
    """Resolved column layout for one '%% TrackingData' chunk."""

    total_cols: int
    i_t: int | None
    i_pos: int | None
    i_x: int
    i_y: int


class CLTParser:
    """Parse, load, and export CLT tracking and quantification files.

    The parser can optionally use TAT metadata to convert global microscope
    coordinates into pixel coordinates and to resolve cross-position tracking.
    """

    def __init__(self, tat_xml_path: str | None = None):
        """Set up an empty parser, optionally backed by TAT metadata."""
        self.tat: TATParser | None = None
        if tat_xml_path and os.path.exists(tat_xml_path):
            self.tat = TATParser.from_xml(tat_xml_path)

        self.folder_exp: str | None = None
        self.folder_seg: str | None = None
        self.cp_tracking: bool | None = None
        self.dry_run: bool | None = None
        self.clt_paths: list[str] = []
        self.clt_paths_by_ident: dict[str, str] = {}

    @staticmethod
    def _identification_from_path(path: str) -> str:
        """Derive an Identification from a CLT file name."""
        stem = os.path.splitext(os.path.basename(path))[0]
        ident = re.sub(r"[A-Za-z]+$", "", stem)
        return ident

    @staticmethod
    def _track_id_from_ident(ident: str) -> int:
        m = re.search(r"-(\d+)[^\d]*$", ident)
        return int(m.group(1)) if m else 1

    def _apply_pixel_conversion(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert global offsets (x, y) to pixel coordinates using TAT metadata.

        x_px = (x_global - posX) / um_per_px
        y_px = (y_global - posY) / um_per_px
        """
        if self.tat is None:
            return df
        pos_map = self.tat.pos_map
        um_per_px = self.tat.um_per_px

        posx = df["Position"].map(
            lambda p: pos_map.get(int(p), (np.nan, np.nan))[0]
        )
        posy = df["Position"].map(
            lambda p: pos_map.get(int(p), (np.nan, np.nan))[1]
        )

        out = df.copy()
        out["XMorphology"] = (df["XMorphology"] - posx) / um_per_px
        out["YMorphology"] = (df["YMorphology"] - posy) / um_per_px
        out[["XMorphology", "YMorphology"]] = np.rint(
            out[["XMorphology", "YMorphology"]]
        ).astype(int)
        return out

    _QUANT_BLOCK_RE = re.compile(
        r'(?ms)^\s*%%\s*Quantification\b(?=[^\n]*Software="SECQUOIA").*?(?=^\s*%%\s|\Z)'
    )

    def _remove_secquoia_quant_blocks(
        self, text: str, *, normalize_newline: bool = False
    ) -> str:
        """Remove every ``%% Quantification`` block SECQUOIA wrote."""
        out = self._QUANT_BLOCK_RE.sub("", text)
        return out.rstrip("\n") + "\n" if normalize_newline else out

    def _build_quantification_block_from_df(
        self,
        df_ident: pd.DataFrame,
        user: str,
        quant_name: str,
        software: str,
        remark: str | None,
        imaging_channel: int,
        default_zindex: int,
    ) -> str:
        """Build a %% Quantification block from a per Identification DataFrame."""
        required_cols = {"t", "Position", "TrackNumber"}
        missing = required_cols - set(df_ident.columns)
        if missing:
            raise ValueError(
                f"_build_quantification_block_from_df: missing required columns "
                f"{sorted(missing)}"
            )

        # Work on a copy and drop rows without track/time
        df_local = df_ident.copy()

        if df_local.columns.has_duplicates:
            dupes = (
                df_local.columns[df_local.columns.duplicated()]
                .unique()
                .tolist()
            )
            LOG.warning("Duplicate columns dropped before export: %s", dupes)
            df_local = df_local.loc[:, ~df_local.columns.duplicated()]

        df_local = df_local[df_local["TrackNumber"].notna()]
        df_local = df_local[df_local["t"].notna()]
        if df_local.empty:
            return ""

        # Meta columns that should NOT become quantification data columns
        meta_cols = {
            "t",
            "Position",
            "TrackNumber",
            "Identification",
            "track_id",
            "ZIndex",
            "ImagingChannel",
        }

        data_cols = [c for c in df_local.columns if c not in meta_cols]
        lines: list[str] = []
        header_parts = [
            f'%% Quantification Name="{quant_name}"',
            f'User="{user}"',
            f'Software="{software}"',
        ]
        if remark:
            header_parts.append(f'Remarks="{remark}"')
        lines.append(" ".join(header_parts))

        #  Column description (Frame + Data)
        lines.append("\t% Columns")
        lines.append("\t\t% Frame")
        lines.append("\t\t\tTimePoint")
        lines.append("\t\t\tFieldOfView")
        lines.append("\t\t\tZIndex")
        lines.append("\t\t\tImagingChannel")
        lines.append("\t\t% Data")
        for name in data_cols:
            lines.append(f'\t\t\tdouble "{name}"')

        #  One % Cell block per TrackNumber
        for track_number, df_track in df_local.groupby(
            "TrackNumber", sort=True
        ):
            try:
                tn_int = int(track_number)
            except (TypeError, ValueError):
                continue

            lines.append(f"\t% Cell id={tn_int}")
            df_track_sorted = df_track.sort_values(["t", "Position"])

            for _, row in df_track_sorted.iterrows():
                # Time point (1-based)
                try:
                    timepoint = int(_as_scalar(row["t"])) + 1
                except (TypeError, ValueError):
                    continue

                # FieldOfView
                try:
                    field_of_view = int(_as_scalar(row["Position"]))
                except (TypeError, ValueError):
                    field_of_view = 0

                #  Here default_zindex & imaging_channel are used
                z_raw = (
                    _as_scalar(row["ZIndex"])
                    if "ZIndex" in row.index
                    else None
                )
                if z_raw is not None and not pd.isna(z_raw):
                    z_val = int(z_raw)
                else:
                    z_val = default_zindex

                ch_raw = (
                    _as_scalar(row["ImagingChannel"])
                    if "ImagingChannel" in row.index
                    else None
                )
                if ch_raw is not None and not pd.isna(ch_raw):
                    im_ch = int(ch_raw)
                else:
                    im_ch = imaging_channel

                values = [
                    str(timepoint),
                    str(field_of_view),
                    str(z_val),
                    str(im_ch),
                ]

                for name in data_cols:
                    val = _as_scalar(row.get(name, np.nan), np.nan)
                    if pd.isna(val):
                        values.append("nan")
                    elif isinstance(val, int | np.integer):
                        values.append(str(int(val)))
                    else:
                        try:
                            v_float = float(val)
                            values.append(f"{v_float:.6g}")
                        except (TypeError, ValueError):
                            values.append(str(val))

                lines.append("\t\t" + ";".join(values))

        return "\n".join(lines)

    def _global_xy(self, position, x_px, y_px):
        """Inverse of _apply_pixel_conversion: pixel coords -> global 'LocalPixel'."""
        if self.tat is None:
            return x_px, y_px
        posx, posy = self.tat.pos_map.get(int(position), (0.0, 0.0))
        um = self.tat.um_per_px
        return x_px * um + posx, y_px * um + posy

    def _build_trackingdata_block_from_df(self, df_ident: pd.DataFrame) -> str:
        """Build a '%% TrackingData' block (one % Cell per TrackNumber).

        Row format: TimePoint;FieldOfView;ZIndex;ImagingChannel;x;y
        - TimePoint is 1-based in the file.
        - x/y are converted from internal pixels back to global coords via TAT.
        """
        df = df_ident.copy()
        df = df[df["TrackNumber"].notna() & df["t"].notna()]
        if df.empty:
            return ""

        lines = [
            '%% TrackingData Units="LocalPixel"',
            "\t% Columns",
            "\t\t% Frame",
            "\t\t\tTimePoint",
            "\t\t\tFieldOfView",
            "\t\t\tZIndex",
            "\t\t\tImagingChannel",
            "\t\t% Data",
            "\t\t\tdouble x",
            "\t\t\tdouble y",
        ]

        for track_number, df_track in df.groupby("TrackNumber", sort=True):
            try:
                tn = int(track_number)
            except (TypeError, ValueError):
                continue

            df_track = df_track.sort_values(["t", "Position"])

            fate = ""
            if "Cellfate" in df_track.columns:
                is_dead = (
                    df_track["Cellfate"].astype(str).isin(("Dead", "Death"))
                )
                if is_dead.any():
                    fate = "Death"
                    t_death = df_track.loc[is_dead, "t"].min()
                    df_track = df_track[df_track["t"] <= t_death]
            lines.append(f'\t% Cell id={tn} cellFate="{fate}"')

            for _, row in df_track.iterrows():
                try:
                    tp = int(row["t"]) + 1  # file is 1-based
                except (TypeError, ValueError):
                    continue

                pos = row.get("Position", 0)
                pos = 0 if pd.isna(pos) else int(pos)

                x_px = row.get("XMorphology", 0)
                y_px = row.get("YMorphology", 0)
                x_px = 0.0 if pd.isna(x_px) else float(x_px)
                y_px = 0.0 if pd.isna(y_px) else float(y_px)

                xg, yg = self._global_xy(pos, x_px, y_px)
                lines.append(f"\t\t{tp};{pos};1;0;{xg:.6g};{yg:.6g}")

        return "\n".join(lines)

    def _new_clt_text(self, df_ident: pd.DataFrame, user: str) -> str:
        """Minimal valid CLT skeleton (header + TrackingData) for a new file."""
        header = "\n".join(
            [
                (
                    '%% CellLineageTree Version="0.0.5" Encoding="UTF-8" '
                    'MinParserVersion="0.0.5"'
                ),
                "",
                "%% Changelog",
                "\t% Columns",
                "\t\tstring user",
                "\t\tdatetime dateTimeUTC",
                "\t\tstring comment",
                "\t% Entries",
                "",
                "%% FrameIndex",
                "\tTimePoint",
                "\tFieldOfView",
                "\tZIndex",
                "\tImagingChannel",
                "",
            ]
        )
        tracking = self._build_trackingdata_block_from_df(df_ident)
        return header + "\n" + tracking + "\n"

    _ROW_COLS = [
        "track_id",
        "XMorphology",
        "YMorphology",
        "t",
        "Position",
        "TrackNumber",
        "Cellfate",
        "Identification",
    ]

    _TRACKINGDATA_RE = re.compile(
        r"(?s)%%\s*TrackingData\b.*?(?=^\s*%%\s|\Z)", re.MULTILINE
    )
    _COLUMNS_RE = re.compile(
        r"(?s)^\s*%+\s*Columns\b.*?"
        r"(?=^\s*%+\s*[A-Z]|^\s*%+\s*Cell\b|^\s*%%|\Z)",
        re.MULTILINE,
    )
    _FRAME_RE = re.compile(
        r"(?s)^\s*%+\s*Frame\b(.*?)(?=^\s*%+|\Z)", re.MULTILINE
    )
    _DATA_RE = re.compile(
        r"(?s)^\s*%+\s*Data\b(.*?)(?=^\s*%+|\Z)", re.MULTILINE
    )
    _CELL_BLOCK_RE = re.compile(
        r"(?ms)^\s*%+\s*Cell\b([^\n]*)\n(.*?)(?=^\s*%+\s*Cell\b|^\s*%%|\Z)"
    )
    _CELL_ID_RE = re.compile(r"\bid\s*=\s*(\d+)")
    _CELL_FATE_RE = re.compile(r'\bcellFate\s*=\s*"([^"]*)"')

    @staticmethod
    def _names_from_columns_line(text: str, *, last_token: bool) -> list[str]:
        """Pull one name per non-comment line of a Frame/Data sub-block."""
        names: list[str] = []
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            parts = s.split()
            if not parts:
                continue
            name = parts[-1] if last_token else parts[0]
            names.append(name.strip('"') if last_token else name)
        return names

    def _declared_column_names(
        self, chunk: str
    ) -> tuple[list[str], list[str]]:
        """Read the '% Columns' > '% Frame'/'% Data' names, if declared."""
        mcol = self._COLUMNS_RE.search(chunk)
        if not mcol:
            return [], []

        colblk = mcol.group(0)
        frame_names: list[str] = []
        data_names: list[str] = []

        mframe = self._FRAME_RE.search(colblk)
        if mframe:
            frame_names = self._names_from_columns_line(
                mframe.group(1), last_token=False
            )
        mdata = self._DATA_RE.search(colblk)
        if mdata:
            data_names = self._names_from_columns_line(
                mdata.group(1), last_token=True
            )
        return frame_names, data_names

    def _resolve_column_schema(self, chunk: str, path: str) -> _ColumnSchema:
        """Work out which semicolon-separated field holds t/Position/x/y.

        Reads the declared '% Columns' > '% Frame'/'% Data' names when
        present, otherwise falls back to the standard CLT layout.
        """
        frame_names, data_names = self._declared_column_names(chunk)
        if not frame_names:
            frame_names = [
                "TimePoint",
                "FieldOfView",
                "ZIndex",
                "ImagingChannel",
            ]
        if not data_names:
            data_names = ["XMorphology", "YMorphology"]

        if "XMorphology" not in data_names or "YMorphology" not in data_names:
            raise ValueError(
                f"{os.path.basename(path)}: TrackingData Data columns must "
                f"declare XMorphology and YMorphology, got {data_names}"
            )

        n_frame = len(frame_names)
        all_names = frame_names + data_names
        i_t = (
            frame_names.index("TimePoint")
            if "TimePoint" in frame_names
            else None
        )
        if "FieldOfView" in frame_names:
            i_pos = frame_names.index("FieldOfView")
        elif "Position" in all_names:
            i_pos = all_names.index("Position")
        else:
            i_pos = None

        return _ColumnSchema(
            total_cols=n_frame + len(data_names),
            i_t=i_t,
            i_pos=i_pos,
            i_x=n_frame + data_names.index("XMorphology"),
            i_y=n_frame + data_names.index("YMorphology"),
        )

    @staticmethod
    def _normalize_cell_fate(raw: str) -> str:
        """Map a raw cellFate="..." attribute to the app's fate vocabulary."""
        fate = raw.strip()
        if not fate:
            return "Healthy"
        if fate == "Division":
            return "Healthy"
        if fate == "Death":
            return "Dead"
        return fate

    def _iter_cell_blocks(self, chunk: str) -> Iterator[tuple[int, str, str]]:
        """Yield (cell_id, normalized_cell_fate, body_text) per '% Cell' block.

        Cell blocks without a parseable ``id=`` attribute are skipped.
        """
        for cm in self._CELL_BLOCK_RE.finditer(chunk):
            m_id = self._CELL_ID_RE.search(cm.group(1))
            if not m_id:
                continue
            cell_id = int(m_id.group(1))
            m_fate = self._CELL_FATE_RE.search(cm.group(1))
            raw_fate = m_fate.group(1) if m_fate else ""
            yield cell_id, self._normalize_cell_fate(raw_fate), cm.group(2)

    @staticmethod
    def _parse_tracking_line(
        line: str, schema: _ColumnSchema
    ) -> tuple[float, float, float, float] | None:
        """Parse one ';'-separated row, or None if it should be skipped."""
        s = line.strip()
        if not s or s.startswith("%") or ";" not in s:
            return None
        parts = s.split(";")
        if len(parts) < schema.total_cols:
            return None
        try:
            t_val = (
                int(float(parts[schema.i_t])) - 1
                if schema.i_t is not None
                else np.nan
            )
            pos_val = (
                int(float(parts[schema.i_pos]))
                if schema.i_pos is not None
                else np.nan
            )
            x_val = float(parts[schema.i_x])
            y_val = float(parts[schema.i_y])
        except ValueError:
            return None
        return t_val, pos_val, x_val, y_val

    def _rows_from_file(self, path: str) -> list[tuple]:
        """Parse one CLT file into a list of plain tuples."""
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        text = self._remove_secquoia_quant_blocks(text)

        m = self._TRACKINGDATA_RE.search(text)
        if not m:
            return []
        chunk = m.group(0)

        schema = self._resolve_column_schema(chunk, path)
        ident = self._identification_from_path(path)
        tid = self._track_id_from_ident(ident)

        rows: list[tuple] = []
        for cell_id, cell_fate, body in self._iter_cell_blocks(chunk):
            for line in body.splitlines():
                parsed = self._parse_tracking_line(line, schema)
                if parsed is None:
                    continue
                t_val, pos_val, x_val, y_val = parsed
                rows.append(
                    (
                        tid,
                        x_val,
                        y_val,
                        t_val,
                        pos_val,
                        cell_id,
                        cell_fate,
                        ident,
                    )
                )
        return rows

    def _finalize_rows(
        self,
        rows: list[tuple],
        src: list[int] | None,
        cp_tracking: bool,
    ) -> pd.DataFrame:
        """Turn accumulated row tuples into the standard track DataFrame."""
        if not rows:
            return pd.DataFrame(columns=self._ROW_COLS)

        df = pd.DataFrame(rows, columns=self._ROW_COLS)

        # Use TAT to determine Position from the original (global) offsets
        # TODO:cross-position tracking
        if self.tat is not None and cp_tracking:
            df["Position"] = [
                self.tat.query_index(x, y)
                for x, y in zip(
                    df["XMorphology"], df["YMorphology"], strict=False
                )
            ]

        df = self._apply_pixel_conversion(df)

        keep = df["Cellfate"]
        df["Cellfate"] = "Healthy"
        if src is None:
            grp = [df["Identification"], df["TrackNumber"]]
        else:
            grp = [pd.Series(src, index=df.index), df["TrackNumber"]]
        idx_last = df.groupby(grp, sort=False)["t"].idxmax()
        df.loc[idx_last, "Cellfate"] = keep.loc[idx_last]

        return df[self._ROW_COLS]

    def load_folder(
        self,
        folder_exp: str,
        folder_seg: str,
        cp_tracking: bool,
        progress_fun: Callable[[int, int], None] | None = None,
        dry_run: bool = False,
    ) -> pd.DataFrame:
        """Load all .clt files in folder_exp and parse them into a track_df."""
        self.folder_exp = folder_exp
        self.folder_seg = folder_seg
        self.cp_tracking = cp_tracking
        self.dry_run = dry_run

        self._index_existing_clt_files()

        total = len(self.clt_paths)
        if progress_fun:
            progress_fun(0, total)

        all_rows: list[tuple] = []
        src: list[int] = []
        for idx, p in enumerate(self.clt_paths, start=1):
            try:
                r = self._rows_from_file(p)
                all_rows.extend(r)
                src.extend([idx] * len(r))
            except (
                OSError,
                UnicodeDecodeError,
                ValueError,
                TypeError,
                KeyError,
            ) as e:
                LOG.warning("Failed to parse %s: %s", p, e)
            finally:
                if progress_fun:
                    progress_fun(idx, total)

        return self._finalize_rows(all_rows, src, cp_tracking)

    def _index_existing_clt_files(self) -> None:
        """Build self.clt_paths and self.clt_paths_by_ident from folder_exp."""
        self.clt_paths = []
        self.clt_paths_by_ident = {}

        if not self.folder_exp:
            return

        for root, _, files in os.walk(self.folder_exp):
            for f in files:
                if f.lower().endswith(".clt") and not f.startswith("._"):
                    path = os.path.join(root, f)
                    self.clt_paths.append(path)

                    ident = self._identification_from_path(path)
                    self.clt_paths_by_ident.setdefault(ident, path)

    @staticmethod
    def _group_by_identification(
        df: pd.DataFrame,
    ) -> tuple[list[str], dict[str, pd.DataFrame]]:
        """Split df into per Identification frames in one pass."""
        ident_col = df["Identification"]
        valid = df[ident_col.notna()]
        stripped = ident_col[ident_col.notna()].astype(str).str.strip()

        idents = stripped.unique().tolist()
        groups = dict(tuple(valid.groupby(stripped, sort=False)))
        return idents, groups

    def _index_pos_tag_dirs(self, root: str) -> dict[str, str]:
        """Map each 'pNNNN' position-tag suffix to its first matching directory.

        Walks the tree only once, mapping each suffix to the first
        directory under root whose name ends with it, instead of once per
        identification that needs a new output directory.
        """
        pos_tag_dirs: dict[str, str] = {}
        for walk_root, dirs, _ in os.walk(root):
            for d in dirs:
                m = re.search(r"(p\d{4})$", d)
                if m:
                    pos_tag_dirs.setdefault(
                        m.group(1), os.path.join(walk_root, d)
                    )
        return pos_tag_dirs

    def _merge_quant_block(
        self,
        existing_text: str,
        quant_block: str,
        *,
        overwrite_existing_quant: bool,
    ) -> str:
        """Append quant_block to existing_text, stripping prior SECQUOIA quantification
        blocks first when overwrite_existing_quant is set."""
        text = existing_text
        if overwrite_existing_quant:
            text = self._remove_secquoia_quant_blocks(
                text, normalize_newline=True
            )
        if not text.endswith("\n"):
            text += "\n"
        return text + "\n" + quant_block

    @staticmethod
    def _resolve_new_file_dir(
        ident: str,
        df_ident: pd.DataFrame,
        existing_path: str | None,
        new_root: str,
        pos_tag_dirs: dict[str, str],
    ) -> str:
        if existing_path:
            return os.path.dirname(existing_path)

        pos = int(
            pd.to_numeric(df_ident["Position"], errors="coerce")
            .dropna()
            .iloc[0]
        )
        pos_tag = f"p{pos:04d}"
        out_dir = pos_tag_dirs.get(pos_tag)
        if out_dir is None:
            base = re.split(r"[-_]p\d{4}", ident)[0]
            out_dir = os.path.join(new_root, f"{base}_{pos_tag}")
        return out_dir

    def _export_quant_for_ident(
        self,
        *,
        ident: str,
        df_ident: pd.DataFrame,
        user: str,
        remark: str,
        overwrite_existing_quant: bool,
        create_missing: bool,
        new_root: str,
        pos_tag_dirs: dict[str, str],
    ) -> None:
        """Write one Identification's quantification into its CLT file."""
        # ensure consistent Identification format
        if "-p" in ident and "_p" not in ident:
            ident = ident.replace("-p", "_p", 1)

        quant_block = self._build_quantification_block_from_df(
            df_ident=df_ident,
            user=user,
            quant_name="SECQUOIAQuantification",
            software="SECQUOIA",
            remark=remark,
            imaging_channel=0,
            default_zindex=1,
        )
        if not quant_block:
            return

        existing_path = self.clt_paths_by_ident.get(ident)

        if existing_path and os.path.exists(existing_path):
            existing_text = read_text_with_reload(existing_path)
            text = self._merge_quant_block(
                existing_text,
                quant_block,
                overwrite_existing_quant=overwrite_existing_quant,
            )
            out_path = existing_path
        else:
            if not create_missing:
                return

            out_dir = self._resolve_new_file_dir(
                ident, df_ident, existing_path, new_root, pos_tag_dirs
            )
            out_path = os.path.join(out_dir, f"{ident}{user}.clt")

            if os.path.exists(out_path):
                existing_text = read_text_with_reload(out_path)
                text = self._merge_quant_block(
                    existing_text,
                    quant_block,
                    overwrite_existing_quant=overwrite_existing_quant,
                )
            else:
                text = self._new_clt_text(df_ident, user) + "\n" + quant_block

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        write_text_with_reload(out_path, text)

        if ident not in self.clt_paths_by_ident:
            self.clt_paths_by_ident[ident] = out_path
        if out_path not in self.clt_paths:
            self.clt_paths.append(out_path)

    def export_quantifications_to_clt(
        self,
        df: pd.DataFrame,
        user: str,
        remark: str = "",
        progress_fun: Callable[[int, int], None] | None = None,
        overwrite_existing_quant: bool = True,
        create_missing: bool = True,
        output_root: str | None = None,
    ) -> None:
        """Export quantifications from a track DataFrame into CLT files."""
        if "Identification" not in df.columns:
            raise ValueError(
                "Quantification export requires a DataFrame with an 'Identification' column."
            )

        if self.folder_exp is None:
            raise RuntimeError(
                "CLTParser.folder_exp is not set. Call load_folder(...) "
                "before export_quantifications_to_clt, or set folder_exp manually."
            )

        new_root = output_root or self.folder_exp

        if not self.clt_paths_by_ident:
            self._index_existing_clt_files()

        idents, groups = self._group_by_identification(df)

        total = len(idents)
        if progress_fun:
            progress_fun(0, total)

        pos_tag_dirs = (
            self._index_pos_tag_dirs(new_root) if create_missing else {}
        )

        for idx, ident in enumerate(idents, start=1):
            df_ident = groups.get(ident)
            if ident and df_ident is not None and not df_ident.empty:
                try:
                    self._export_quant_for_ident(
                        ident=ident,
                        df_ident=df_ident,
                        user=user,
                        remark=remark,
                        overwrite_existing_quant=overwrite_existing_quant,
                        create_missing=create_missing,
                        new_root=new_root,
                        pos_tag_dirs=pos_tag_dirs,
                    )
                except (OSError, ValueError, RuntimeError) as e:
                    LOG.error("[CLT export] Skipping '%s': %r", ident, e)

            if progress_fun:
                progress_fun(idx, total)
