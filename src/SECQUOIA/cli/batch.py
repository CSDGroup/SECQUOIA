"""Headless batch quantification.

Runs what the Loading window does with the ``All`` loading format, without
opening any window: load the tracking data, then load, background-correct,
mask and quantify every selected position, and write the per position CSVs,
``Updated_CSV_File.csv`` and ``project_metadata.json``. The project can be
opened in the GUI afterwards with ``File → Open a project``.

Each Loading window field maps to one argument; see ``docs/batch-mode.md``.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

# No window is ever created, but Qt is still imported by the shared pipeline
# modules; make sure it never looks for a display (clusters, SSH sessions).
os.environ.setdefault("QT_API", "pyside6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

LOG = logging.getLogger("SECQUOIA.cli.batch")

TRACKING_FORMATS = ("btrack", "CTC", "tTt", "Ultrack")
IMAGE_FORMATS = ("png", "tif", "jpg")

# Same defaults as the Loading window's widgets.
DEFAULT_PROJECT_NAME = "Project_1"
DEFAULT_MAX_DISTANCE = 20
DEFAULT_MIN_CELL_SIZE = 5
DEFAULT_INTERVAL_S = 300

# Marks an optional-value flag given without a value ("pick what is detected").
_AUTO = object()

EXIT_OK = 0
EXIT_POSITIONS_FAILED = 1
EXIT_USAGE = 2


class BatchError(Exception):
    """A problem with the arguments or the experiment folder."""


# Argument parsing


def parse_position_spec(spec: str) -> set[int]:
    """Parse ``"1-3,5"`` into ``{1, 2, 3, 5}``."""
    numbers: set[int] = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        try:
            if sep:
                a, b = int(lo), int(hi)
                if b < a:
                    a, b = b, a
                numbers.update(range(a, b + 1))
            else:
                numbers.add(int(lo))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"invalid position {part!r}; use numbers and ranges like 1-3,5"
            ) from None
    if not numbers:
        raise argparse.ArgumentTypeError("no positions given")
    return numbers


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, grouped like the Loading window's tabs."""
    p = argparse.ArgumentParser(
        prog="SECQUOIA-batch",
        description=(
            "Quantify tracked positions without the GUI. Does the same as "
            "the Loading window with loading format 'All', then writes the "
            "CSVs and project_metadata.json so the project can be opened in "
            "SECQUOIA later."
        ),
        epilog="Full guide: docs/batch-mode.md",
    )

    tracking = p.add_argument_group("Tracking tab")
    tracking.add_argument(
        "experiment",
        help="Experiment folder (tTt layout, e.g. .../250615MA40).",
    )
    tracking.add_argument(
        "--tracking-folder",
        required=True,
        metavar="PATH",
        help="Tracking folder. A relative path is looked up from the current "
        "directory first, then from the experiment folder.",
    )
    tracking.add_argument(
        "--tracking-format",
        choices=TRACKING_FORMATS,
        default="tTt",
        help="Tracking input format (default: %(default)s).",
    )
    tracking.add_argument(
        "--image-format",
        choices=IMAGE_FORMATS,
        default="png",
        help="Format of the images and masks (default: %(default)s).",
    )
    tracking.add_argument(
        "--project-name",
        default=DEFAULT_PROJECT_NAME,
        help="Project name; results go to Analysis/SECQUOIA_files_<format>/"
        "<project name>/ (default: %(default)s).",
    )
    tracking.add_argument(
        "--user", default="", help="User initials stored with the project."
    )
    tracking.add_argument(
        "--positions",
        type=parse_position_spec,
        metavar="SPEC",
        help="Positions to quantify, e.g. '1-3,5' (default: every position "
        "with tracking data).",
    )
    tracking.add_argument(
        "--tree-ids",
        nargs="+",
        metavar="ID",
        help="Only these Tree-IDs (Identification values, e.g. "
        "250615MA40-p0001-001). Use --list-tree-ids to see them.",
    )
    tracking.add_argument(
        "--tree-ids-file",
        metavar="FILE",
        help="Text file with one Tree-ID per line; combined with --tree-ids.",
    )

    seg = p.add_argument_group("Segmentation tab")
    seg.add_argument(
        "--segmentation",
        nargs="+",
        metavar="NAME",
        help="Segmentation folder names in Analysis/ (default: all "
        "'Segmentation*' folders). Masks are numbered M1, M2, ... in "
        "alphabetical folder order, as in the GUI.",
    )
    seg.add_argument(
        "--basic",
        nargs="?",
        const=_AUTO,
        default=None,
        metavar="NAME",
        help="Enable BaSiC background correction, optionally naming the "
        "'BaSiC*' folder in Analysis/ (default when given without a name: "
        "the first one found). Off when omitted.",
    )
    seg.add_argument(
        "--real-time",
        nargs="?",
        const=_AUTO,
        default=None,
        metavar="CSV",
        help="Import real acquisition times from an 'images*.csv' in the "
        "experiment folder (default when given without a name: the first "
        "one found). Off when omitted.",
    )
    seg.add_argument(
        "--min-cell-size",
        type=int,
        default=DEFAULT_MIN_CELL_SIZE,
        metavar="PX",
        help="Ignore masks smaller than this many pixels "
        "(default: %(default)s).",
    )
    seg.add_argument(
        "--max-distance",
        type=int,
        default=DEFAULT_MAX_DISTANCE,
        metavar="PX",
        help="Maximum distance between a tracking point and a mask for them "
        "to be matched (default: %(default)s).",
    )

    chan = p.add_argument_group("Channel Selection tab")
    chan.add_argument(
        "--channels",
        nargs="+",
        metavar="wNN",
        help="Channels to quantify, e.g. w00 w02 (default: all detected).",
    )
    chan.add_argument(
        "--t-min",
        type=int,
        default=1,
        metavar="T",
        help="First time point (default: %(default)s).",
    )
    chan.add_argument(
        "--t-max",
        type=int,
        metavar="T",
        help="Last time point (default: the last one found in the images).",
    )
    chan.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL_S,
        metavar="SECONDS",
        help="Seconds between time points (default: %(default)s).",
    )
    chan.add_argument(
        "--start-position",
        type=int,
        metavar="N",
        help="Position the GUI opens on when the project is reopened "
        "(default: the first quantified position).",
    )

    run = p.add_argument_group("Loading tab and run control")
    run.add_argument(
        "--overwrite",
        action="store_true",
        help="If the project folder already exists, delete it and start "
        "over (the GUI asks the same question). Without this the run stops.",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Load the tracking data, print the summary and exit without "
        "quantifying or writing anything.",
    )
    run.add_argument(
        "--list-tree-ids",
        action="store_true",
        help="Print every selected Position and Tree-ID, then exit.",
    )
    verbosity = run.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Also log every processing step.",
    )
    verbosity.add_argument(
        "-q", "--quiet", action="store_true", help="Only log warnings."
    )
    return p


# Resolving the arguments against what is on disk


@dataclass
class BatchConfig:
    """Everything the run needs, with the experiment folder already scanned."""

    folder: str
    experiment_name: str
    analysis_dir: str
    tracking_path: str
    tracking_format: str
    image_format: str
    project_name: str
    user: str
    position_folders: list[str]
    position_numbers: list[int]
    segmentation_paths: list[str]
    basic_path: str | None
    realtime_path: str | None
    available_channels: list[str]
    channels: list[str]
    t_min: int
    t_max: int
    t_max_detected: int
    interval_s: int
    min_cell_size: int
    max_distance: int
    xml_path: str | None
    positions: set[int] | None = None
    tree_ids: set[str] | None = None
    start_position: int | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def project_dir(self) -> str:
        """``<experiment>/Analysis/SECQUOIA_files_<format>/<project>``."""
        return os.path.join(
            self.analysis_dir,
            f"SECQUOIA_files_{self.tracking_format}",
            self.project_name,
        )


def _resolve_tracking_path(raw: str, folder: str) -> str:
    """Absolute tracking folder; relative paths try cwd, then the experiment."""
    candidates = [os.path.abspath(raw)]
    if not os.path.isabs(raw):
        candidates.append(os.path.abspath(os.path.join(folder, raw)))
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise BatchError(f"Tracking folder not found: {raw}")


def _pick_named(
    requested, available: list[str], what: str, where: str
) -> str | None:
    """Resolve an optional-value flag (None = off, _AUTO = first found)."""
    if requested is None:
        return None
    if not available:
        raise BatchError(f"No {what} found in {where}.")
    if requested is _AUTO:
        return available[0]
    name = os.path.basename(os.path.normpath(str(requested)))
    if name not in available:
        raise BatchError(
            f"{what} {requested!r} not found in {where}. "
            f"Found: {', '.join(available)}"
        )
    return name


def _read_tree_ids(args) -> set[str] | None:
    """Tree-IDs from --tree-ids and --tree-ids-file, or None if neither."""
    ids: set[str] = set(args.tree_ids or [])
    if args.tree_ids_file:
        try:
            with open(args.tree_ids_file, encoding="utf-8") as f:
                ids.update(line.strip() for line in f if line.strip())
        except OSError as e:
            raise BatchError(f"Cannot read --tree-ids-file: {e}") from e
    if args.tree_ids is None and not args.tree_ids_file:
        return None
    if not ids:
        raise BatchError("--tree-ids-file contains no Tree-IDs.")
    return ids


def resolve_config(args) -> BatchConfig:
    """Check the arguments against the experiment folder; nothing is written."""
    from SECQUOIA.core.experiment_layout import (
        detect_time_max,
        detect_unique_w_channels,
        find_analysis_dir,
        find_basic_folders,
        find_example_position_folder,
        find_images_csvs,
    )
    from SECQUOIA.gui.loading.experiment_paths import _scan_position_folders

    folder = os.path.abspath(args.experiment)
    if not os.path.isdir(folder):
        raise BatchError(f"Experiment folder not found: {args.experiment}")
    experiment_name = os.path.basename(folder)

    analysis_dir = find_analysis_dir(folder)
    if not analysis_dir:
        raise BatchError(f"No 'Analysis' folder in {folder}.")

    position_folders, position_numbers = _scan_position_folders(
        folder, experiment_name
    )
    if not position_folders:
        raise BatchError(
            f"No position folders named '{experiment_name}_pNNNN' in {folder}."
        )

    tracking_path = _resolve_tracking_path(args.tracking_folder, folder)

    xml_path = None
    if args.tracking_format == "tTt":
        xmls = sorted(glob.glob(os.path.join(folder, "*_TATexp.xml")))
        if not xmls:
            raise BatchError(
                "tTt tracking needs a TAT experiment XML (*_TATexp.xml) in "
                "the experiment folder."
            )
        xml_path = xmls[0]

    # Segmentation tab
    seg_available = sorted(
        e.name
        for e in os.scandir(analysis_dir)
        if e.is_dir() and e.name.startswith("Segmentation")
    )
    if not seg_available:
        raise BatchError(f"No 'Segmentation*' folders in {analysis_dir}.")
    if args.segmentation:
        wanted = [
            os.path.basename(os.path.normpath(s)) for s in args.segmentation
        ]
        unknown = [s for s in wanted if s not in seg_available]
        if unknown:
            raise BatchError(
                f"Segmentation folder(s) not found: {', '.join(unknown)}. "
                f"Found: {', '.join(seg_available)}"
            )
        seg_names = [s for s in seg_available if s in set(wanted)]
    else:
        seg_names = seg_available

    basic_name = _pick_named(
        args.basic, find_basic_folders(folder), "BaSiC folder", analysis_dir
    )
    rt_name = _pick_named(
        args.real_time, find_images_csvs(folder), "images*.csv file", folder
    )

    # Channel Selection tab
    example = find_example_position_folder(
        folder, experiment_name, position_folders
    )
    available_channels = detect_unique_w_channels(example, 1)
    if args.channels:
        unknown = [c for c in args.channels if c not in available_channels]
        if unknown:
            raise BatchError(
                f"Channel(s) not found: {', '.join(unknown)}. "
                f"Found: {', '.join(available_channels)}"
            )
        channels = [c for c in available_channels if c in set(args.channels)]
    else:
        channels = list(available_channels)

    t_max_detected = max(1, detect_time_max(example))
    t_min = args.t_min
    t_max = args.t_max if args.t_max is not None else t_max_detected
    if not 1 <= t_min <= t_max_detected:
        raise BatchError(
            f"--t-min must be between 1 and {t_max_detected}, got {t_min}."
        )
    if not t_min <= t_max <= t_max_detected:
        raise BatchError(
            f"--t-max must be between {t_min} and {t_max_detected}, "
            f"got {t_max}."
        )
    for name in ("interval", "min_cell_size", "max_distance"):
        if getattr(args, name) < 0:
            flag = "--" + name.replace("_", "-")
            raise BatchError(f"{flag} cannot be negative.")

    project_name = (args.project_name or "").strip() or DEFAULT_PROJECT_NAME

    config = BatchConfig(
        folder=folder,
        experiment_name=experiment_name,
        analysis_dir=analysis_dir,
        tracking_path=tracking_path,
        tracking_format=args.tracking_format,
        image_format=args.image_format,
        project_name=project_name,
        user=(args.user or "").strip(),
        position_folders=position_folders,
        position_numbers=position_numbers,
        segmentation_paths=[os.path.join(analysis_dir, n) for n in seg_names],
        basic_path=(
            os.path.join(analysis_dir, basic_name) if basic_name else None
        ),
        realtime_path=os.path.join(folder, rt_name) if rt_name else None,
        available_channels=available_channels,
        channels=channels,
        t_min=t_min,
        t_max=t_max,
        t_max_detected=t_max_detected,
        interval_s=args.interval,
        min_cell_size=args.min_cell_size,
        max_distance=args.max_distance,
        xml_path=xml_path,
        positions=args.positions,
        tree_ids=_read_tree_ids(args),
        start_position=args.start_position,
    )

    if config.positions:
        missing = sorted(config.positions - set(position_numbers))
        if missing:
            config.warnings.append(
                "No position folder for requested position(s): "
                + ", ".join(str(n) for n in missing)
            )
    return config


# The stand-in for MainWindow


def new_session(config: BatchConfig) -> SimpleNamespace:
    """Build the object the pipeline reads instead of ``MainWindow``.

    The attributes are the ones the Loading window sets when its fields are
    filled in and ``Load Experiment`` / ``Run`` are clicked.
    """
    from SECQUOIA.core.basic_correction import DataSetBaSiC

    use_basic = config.basic_path is not None
    s = SimpleNamespace(
        # Tracking tab
        project_name=config.project_name,
        user=config.user,
        folder=config.folder,
        experiment_name=config.experiment_name,
        folder_list=[
            f
            for f in os.listdir(config.folder)
            if os.path.isdir(os.path.join(config.folder, f))
        ],
        position_folders=list(config.position_folders),
        position_indices=list(config.position_numbers),
        position_min=config.position_numbers[0],
        position_max=config.position_numbers[-1],
        image_format=config.image_format,
        tracking_format=config.tracking_format,
        tracking_path=config.tracking_path,
        xml_path=config.xml_path,
        clt_parser=None,
        cp_tracking=False,
        # Segmentation tab
        segmentation_paths=list(config.segmentation_paths),
        n_masks=len(config.segmentation_paths),
        basic=DataSetBaSiC(flag=use_basic),
        use_background_correction=use_basic,
        background_correction_path=config.basic_path,
        use_import_rt=config.realtime_path is not None,
        import_rt_path=config.realtime_path,
        import_rt_df=None,
        _rt_wide=None,
        min_mask_size=config.min_cell_size,
        threshold=config.max_distance,
        # Channel Selection tab
        available_channels=list(config.available_channels),
        ids_channels=list(config.channels),
        n_channels=len(config.channels),
        FL_inputs=[],
        t_max_detected=config.t_max_detected,
        time_min_selected=config.t_min,
        time_max_selected=config.t_max,
        dt_seconds=config.interval_s,
        time_interval=float(config.interval_s),
        position_min_selected=config.position_numbers[0],
        position_max_selected=config.position_numbers[-1],
        position_start_selected=config.position_numbers[0],
        current_position_number=config.position_numbers[0],
        current_position_index=0,
        # Loading tab
        loading_format="All",
        # Loaded data and caches (as reset before a GUI run)
        track_df=None,
        filtered_df=None,
        images={},
        labels=[],
        image_present={},
        corrected_images=None,
        _memmap_dir=None,
        _memmap_files=None,
        _memmap_lock=None,
        _corrected_u8_memmap_files=[],
        _derived_features={},
        _feature_defs={},
        last_run_config={},
        selected_feature_by_row={},
        jump_channel=config.channels[0] if config.channels else None,
    )
    for i, channel in enumerate(config.channels):
        setattr(s, f"FL_identifiers_{i + 1}", channel)
    return s


def _set_current_position(session, number: int) -> None:
    """Point the session at position `number` (the GUI's start position)."""
    from SECQUOIA.utils.positions import position_index_from_number

    session.current_position_number = int(number)
    session.position_start_selected = int(number)
    session.current_position_index = position_index_from_number(
        session, number
    )


def load_tracking(session, config: BatchConfig) -> None:
    """``Load Experiment`` plus the Position/Tree-ID selection."""
    import pandas as pd

    from SECQUOIA.core.tracking.track_data import (
        BTRACK_MISSING_MESSAGE,
        _load_tracking_data,
        btrack_is_available,
    )

    if config.tracking_format == "btrack" and not btrack_is_available():
        raise BatchError(BTRACK_MISSING_MESSAGE)

    try:
        _load_tracking_data(session)
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        raise BatchError(f"Loading the tracking data failed: {e}") from e
    # _load_tracking_data resets the threshold; Run reads it from the field.
    session.threshold = config.max_distance

    df = session.track_df
    if df is None or df.empty:
        raise BatchError(f"No tracking data found in {config.tracking_path}.")
    if "Position" not in df.columns or "Identification" not in df.columns:
        raise BatchError(
            "The tracking data has no Position/Identification columns; "
            "check --tracking-format and --tracking-folder."
        )

    keep = pd.Series(True, index=df.index)
    if config.positions:
        pos = pd.to_numeric(df["Position"], errors="coerce")
        keep &= pos.isin(config.positions)
    if config.tree_ids:
        ids = df["Identification"].astype(str)
        unknown = sorted(config.tree_ids - set(ids.unique()))
        if unknown:
            shown = ", ".join(unknown[:10])
            more = f" (+{len(unknown) - 10} more)" if len(unknown) > 10 else ""
            config.warnings.append(f"Unknown Tree-ID(s): {shown}{more}")
        keep &= ids.isin(config.tree_ids)
    if not keep.all():
        session.track_df = df.loc[keep].copy()
    if session.track_df.empty:
        raise BatchError("No tracking data left after the selection.")


def positions_to_run(session, config: BatchConfig) -> list[tuple[int, str]]:
    """``(number, folder)`` of every selected position that has tracking."""
    from SECQUOIA.gui.position_navigation import tree_counts
    from SECQUOIA.utils.positions import position_number_from_folder

    counts = tree_counts(session) or {}
    items, skipped = [], []
    for path in session.position_folders:
        n = position_number_from_folder(path)
        if n is None or (config.positions and n not in config.positions):
            continue
        if counts.get(n, 0) > 0:
            items.append((n, path))
        else:
            skipped.append(n)
    if skipped:
        msg = "Skipped position(s) without tracking points: " + ", ".join(
            f"p{n:04d}" for n in skipped
        )
        if config.positions:
            config.warnings.append(msg)
        else:
            LOG.info("%s", msg)
    return sorted(items)


# Reporting


def summary_lines(session, config: BatchConfig, items) -> list[str]:
    """The Loading tab's summary, as plain text."""
    from SECQUOIA.gui.position_navigation import tree_counts

    counts = tree_counts(session) or {}
    n_trees = sum(counts.get(n, 0) for n, _ in items)
    start = session.position_start_selected
    rows = [
        ("Project name", config.project_name),
        ("Experiment name", config.experiment_name),
        ("Experiment folder", config.folder),
        ("Image format", config.image_format),
        ("Tracking", f"{config.tracking_format} — {config.tracking_path}"),
        ("Time point range", f"{config.t_min} – {config.t_max}"),
        ("Time [s] between time points", f"{config.interval_s} s"),
        ("Channels", ", ".join(config.channels)),
        (
            "Segmentations",
            ", ".join(
                f"M{i}={os.path.basename(p)}"
                for i, p in enumerate(config.segmentation_paths, 1)
            ),
        ),
        (
            "Background correction",
            (
                os.path.basename(config.basic_path)
                if config.basic_path
                else "(off)"
            ),
        ),
        (
            "Import real time [ms]",
            (
                os.path.basename(config.realtime_path)
                if config.realtime_path
                else "(off)"
            ),
        ),
        (
            "Max distance of track point from segmentation mask",
            f"{config.max_distance}",
        ),
        ("Min cell size", f"{config.min_cell_size} px"),
        ("Loading format", "All"),
        ("Start position of curation", f"{start}"),
        (
            "Positions & Cell Lineage Trees",
            f"{len(items)} positions, {n_trees} Tree-IDs",
        ),
        ("Output folder", config.project_dir),
    ]
    width = max(len(label) for label, _ in rows)
    lines = [f"{label:<{width}}  {value}" for label, value in rows]
    if items:
        per_pos = ", ".join(f"p{n:04d}:{counts.get(n, 0)}" for n, _ in items)
        lines.append(f"{'Tree-IDs per position':<{width}}  {per_pos}")
    return lines


def _print_tree_ids(session, items) -> None:
    """Print ``Position<TAB>Tree-ID`` for every selected tree."""
    numbers = {n for n, _ in items}
    df = session.track_df
    pairs = (
        df[["Position", "Identification"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["Position", "Identification"])
    )
    for pos, ident in pairs.itertuples(index=False):
        if int(pos) in numbers:
            print(f"{int(pos)}\t{ident}")


# Running


def _project_exists_message(config: BatchConfig) -> str:
    return (
        f"Project folder already exists: {config.project_dir}\n"
        "Use another --project-name, or --overwrite to delete it and start "
        "over."
    )


def _prepare_project_dir(config: BatchConfig, overwrite: bool) -> None:
    """Create the project folder, deleting an old one only with --overwrite."""
    project_dir = config.project_dir
    if os.path.isdir(project_dir):
        if not overwrite:
            raise BatchError(_project_exists_message(config))
        LOG.warning("Deleting existing project folder %s", project_dir)
        try:
            shutil.rmtree(project_dir)
        except OSError as e:
            raise BatchError(f"Could not delete {project_dir}: {e}") from e
    os.makedirs(project_dir, exist_ok=True)


def quantify_positions(session, items) -> list[int]:
    """Quantify each position as the ``All`` run does; returns failed ones."""
    from SECQUOIA.core.basic_correction import _prepare_basic_correction_inputs
    from SECQUOIA.core.memmap_store import cleanup_memmaps
    from SECQUOIA.gui.loading_pipeline import (
        _LOAD_ERRORS,
        _build_position_index_map,
        _NextPositionPrefetch,
        _process_one_position,
    )

    last_msg = [None]

    def setp(_pct):
        return None

    def setmsg(msg):
        if msg and msg != last_msg[0]:
            last_msg[0] = msg
            LOG.debug("%s", msg)

    t_file_min, t_file_max = _prepare_basic_correction_inputs(session)
    idx_map = _build_position_index_map(session)
    prefetch = _NextPositionPrefetch(session, t_file_min, t_file_max)

    failed: list[int] = []
    total = len(items)
    try:
        for i, (pnum, ppath) in enumerate(items, 1):
            LOG.info("p%04d (%d/%d): quantifying…", pnum, i, total)
            started = time.perf_counter()
            try:
                _process_one_position(
                    session,
                    pnum,
                    ppath,
                    idx_map,
                    t_file_min,
                    t_file_max,
                    setp,
                    setmsg,
                    prefetch=prefetch,
                    next_item=items[i] if i < total else None,
                )
            except _LOAD_ERRORS as e:
                LOG.error("p%04d failed: %s", pnum, e)
                failed.append(pnum)
            else:
                LOG.info(
                    "p%04d (%d/%d): done in %.1f s",
                    pnum,
                    i,
                    total,
                    time.perf_counter() - started,
                )
            finally:
                cleanup_memmaps(session, remove_viewer_image_layers=False)
    finally:
        prefetch.close()
    return failed


def write_outputs(session, quantified: list[int]) -> None:
    """Write ``Updated_CSV_File.csv``, the per position CSVs and the metadata.

    The GUI only rewrites the CSV of the position it ends on; here every
    quantified position's CSV is rewritten from the final table, so all of
    them carry the gap-filled rows and real-time columns. Failed positions
    get no CSV, so a later GUI run does not mistake them for cached results.
    """
    from SECQUOIA.core.project_state import save_project_state, save_track_df
    from SECQUOIA.core.tracking.track_data import (
        _apply_import_realtime_to_track_df,
        _cache_position_measurements,
    )
    from SECQUOIA.utils.timing import current_t_range

    _apply_import_realtime_to_track_df(session)

    if quantified:
        start = session.position_start_selected
        if start not in quantified:
            start = quantified[0]
        _set_current_position(session, start)
        t_file_min, t_file_max, _, _ = current_t_range(session)
        for pnum in quantified:
            _cache_position_measurements(
                session, pnum, t_file_min, t_file_max, log_prefix="batch"
            )
        save_track_df(session)
    else:
        LOG.error("No position was quantified; writing no CSV files.")
    save_project_state(session)


class _ConsoleFilter(logging.Filter):
    """Let warnings through, plus this module's progress unless quiet."""

    def __init__(self, quiet: bool) -> None:
        super().__init__()
        self.quiet = quiet

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        return not self.quiet and record.name == LOG.name


def _configure_console(configure_logging, args) -> None:
    """Console: batch progress and warnings (all INFO with -v, only warnings
    with -q). The experiment's SECQUOIA.log always gets the full INFO log."""
    logger = configure_logging(logging.DEBUG if args.verbose else logging.INFO)
    for handler in logger.handlers:
        if type(handler) is not logging.StreamHandler:
            continue  # FileHandler subclasses StreamHandler
        for old in [
            f for f in handler.filters if isinstance(f, _ConsoleFilter)
        ]:
            handler.removeFilter(old)
        if not args.verbose:
            handler.addFilter(_ConsoleFilter(args.quiet))


def run(args) -> int:
    """Run the batch quantification for parsed `args`; returns the exit code."""
    from SECQUOIA.core.logging_setup import (
        attach_experiment_log,
        configure_logging,
    )

    _configure_console(configure_logging, args)

    from SECQUOIA.core.memmap_store import reap_stale_memmap_dirs
    from SECQUOIA.gui.loading.load_data.load_data_realtime_import import (
        _maybe_import_realtime,
    )
    from SECQUOIA.utils.timing import rebase_track_df_to_t_range

    config = resolve_config(args)
    writes = not (args.dry_run or args.list_tree_ids)
    if writes and not args.overwrite and os.path.isdir(config.project_dir):
        # Fail before the slow tracking load; --overwrite deletes it later.
        raise BatchError(_project_exists_message(config))

    session = new_session(config)
    LOG.info("Loading tracking data from %s", config.tracking_path)
    load_tracking(session, config)
    items = positions_to_run(session, config)
    if not items:
        for w in config.warnings:
            LOG.warning("%s", w)
        raise BatchError("No selected position has tracking data.")

    start = config.start_position
    if start is None or start not in {n for n, _ in items}:
        if start is not None:
            config.warnings.append(
                f"--start-position {start} is not quantified; using "
                f"{items[0][0]}."
            )
        start = items[0][0]
    _set_current_position(session, start)

    for w in config.warnings:
        LOG.warning("%s", w)

    if args.list_tree_ids:
        _print_tree_ids(session, items)
        return EXIT_OK

    if args.dry_run or not args.quiet:
        print("\n".join(summary_lines(session, config, items)), flush=True)
    if args.dry_run:
        print("\nDry run: nothing was quantified or written.")
        return EXIT_OK

    _prepare_project_dir(config, args.overwrite)
    attach_experiment_log(config.folder)
    try:
        reap_stale_memmap_dirs()
    except OSError as e:
        LOG.debug("Could not reap stale memmap folders: %s", e)

    _maybe_import_realtime(session)
    rebase_track_df_to_t_range(session)

    started = time.perf_counter()
    failed = quantify_positions(session, items)
    write_outputs(session, [n for n, _ in items if n not in failed])

    elapsed = time.perf_counter() - started
    done = len(items) - len(failed)
    LOG.info(
        "Quantified %d/%d positions in %.1f s. Results: %s",
        done,
        len(items),
        elapsed,
        config.project_dir,
    )
    if failed:
        LOG.error(
            "Failed positions: %s",
            ", ".join(f"p{n:04d}" for n in failed),
        )
        return EXIT_POSITIONS_FAILED
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``SECQUOIA-batch``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except BatchError as e:
        print(f"SECQUOIA-batch: error: {e}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        print("SECQUOIA-batch: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
