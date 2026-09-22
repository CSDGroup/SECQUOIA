"""Runtime/memory measurement for the loading pipeline.

Disabled by default and only activated when ``SECQUOIA_MEASURE`` is set.

``PositionRun.finish``/``_record_edit_row`` catch ``Exception`` broadly on
purpose: they're the outer boundary wrapping a wide, heterogeneous mix of
calls (pandas, numpy, file I/O, arbitrary ``main_window`` state) where an
exhaustive exception list isn't practical, and logging with
``exc_info=True`` there means nothing is silently lost.
"""

from __future__ import annotations

import contextvars
import csv
import hashlib
import logging
import os
import platform
import sys
import threading
import time
import uuid
from contextlib import contextmanager, suppress
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)

__all__ = [
    "CSV_COLUMNS",
    "PositionRun",
    "active_groups",
    "active_stage",
    "is_enabled",
    "mask_edit_measurement",
]

_ALL_GROUPS = frozenset({"time", "memory", "disk", "cpu"})

_MEMORY_STAGES = (
    "load_channels",
    "load_masks",
    "bg_correction",
    "measure",
    "matching",
    "quantify",
)

_TIME_STAGES = (*_MEMORY_STAGES, "save")


CSV_COLUMNS: tuple[str, ...] = (
    # Identity
    "run_id",
    "timestamp",
    "tag",
    "position",
    "mode",
    # Scaling factors
    "n_timepoints",
    "n_channels",
    "n_mask_sets",
    "n_tracked_cells",
    "n_objects_total",
    "image_h",
    "image_w",
    "dtype",
    "crop_size",
    "crop_offset_px",
    "t_total",
    "t_load_channels",
    "t_load_masks",
    "t_bg_correction",
    "t_measure",
    "t_matching",
    "t_quantify",
    "t_save",
    # Memory (MB).
    "rss_before",
    "rss_peak_load_channels",
    "rss_peak_load_masks",
    "rss_peak_bg_correction",
    "rss_peak_measure",
    "rss_peak_matching",
    "rss_peak_quantify",
    "rss_after",
    "rss_delta",
    # Disk (MB)
    "memmap_bytes",
    "analysis_dir_bytes",
    "input_bytes_read",
    # Derived
    "theoretical_bytes",
    "ram_peak_to_theoretical_ratio",
    "throughput_mb_s",
    "sec_per_timepoint",
    # CPU
    "cpu_time_s",
    "cpu_util_pct",
    "n_workers",
    # Environment
    "os",
    "os_version",
    "cpu_model",
    "cores",
    "ram_total_gb",
    "python_version",
    "numpy_version",
    "napari_version",
    "qt_version",
    "secquoia_version",
)

_SAMPLE_INTERVAL_S = 0.05


def _env_flag_set(value: str | None) -> set[str] | None:
    """Parse ``SECQUOIA_MEASURE``. ``None`` means disabled."""
    if not value:
        return None
    v = value.strip().lower()
    if v in ("0", "false", "off", "no"):
        return None
    if v in ("1", "true", "all", "yes"):
        return set(_ALL_GROUPS)
    groups = {g.strip().lower() for g in v.split(",") if g.strip()}
    groups &= _ALL_GROUPS
    return groups or None


def active_groups() -> set[str] | None:
    """The measurement groups ``SECQUOIA_MEASURE`` enables, or ``None``."""
    return _env_flag_set(os.environ.get("SECQUOIA_MEASURE"))


def is_enabled() -> bool:
    """Whether any measurement group is enabled."""
    return active_groups() is not None


_ACTIVE_RUN: contextvars.ContextVar[PositionRun | None] = (
    contextvars.ContextVar("_active_run", default=None)
)


@contextmanager
def active_stage(name: str):
    """Time/RAM-sample stage ``name`` of whichever ``PositionRun`` is active."""
    run = _ACTIVE_RUN.get()
    if run is None:
        yield
        return
    with run.stage(name):
        yield


@contextmanager
def mask_edit_measurement(main_window, position):
    """Time one interactive mask edit recompute; appends its own CSV row."""
    if not is_enabled():
        yield
        return
    t0 = time.perf_counter()
    process = _psutil().Process()
    rss_before = process.memory_info().rss
    cpu0 = process.cpu_times()
    try:
        yield
    finally:
        elapsed_s = time.perf_counter() - t0
        rss_after = process.memory_info().rss
        cpu1 = process.cpu_times()
        cpu_time_s = (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)
        _record_edit_row(
            main_window, position, elapsed_s, rss_before, rss_after, cpu_time_s
        )


def _psutil():
    """Import psutil lazily, so it's never touched when measuring is off."""
    import psutil

    return psutil


class _RssSampler:
    """Samples this process's RSS on a background thread; tracks the peak."""

    def __init__(self, process) -> None:
        self._process = process
        self._psutil = _psutil()
        self._peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rss = self._process.memory_info().rss
            except (self._psutil.Error, OSError):
                rss = 0
            self._peak = max(self._peak, rss)
            self._stop.wait(_SAMPLE_INTERVAL_S)

    def start(self) -> _RssSampler:
        try:
            self._peak = self._process.memory_info().rss
        except (self._psutil.Error, OSError):
            self._peak = 0
        self._thread.start()
        return self

    def stop_and_peak(self) -> int:
        self._stop.set()
        self._thread.join(timeout=1.0)
        return self._peak


def _version_of(module_name: str) -> str | None:
    """The installed ``__version__`` of a module, or ``None`` if unavailable."""
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", None)
    except ImportError:
        return None


def _environment_record() -> dict:
    """Static facts about this machine/install; recomputed for every row."""
    try:
        psutil = _psutil()
        cores = psutil.cpu_count(logical=True) or None
        ram_total_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    except (ImportError, OSError):
        cores, ram_total_gb = None, None

    try:
        from SECQUOIA import _version as _sq_version

        secquoia_version = getattr(_sq_version, "version", None)
    except ImportError:
        secquoia_version = None

    return {
        "os": platform.system(),
        "os_version": platform.platform(),
        "cpu_model": platform.processor() or None,
        "cores": cores,
        "ram_total_gb": ram_total_gb,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "napari_version": _version_of("napari"),
        "qt_version": _version_of("PySide6") or _version_of("PyQt5"),
        "secquoia_version": secquoia_version,
    }


def _dir_size_bytes(path: str | None) -> int:
    """Best effort recursive size of a directory; 0 if it can't be read."""
    if not path or not os.path.isdir(path):
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


def _glob_bytes(directory: str | None, patterns: list[str]) -> int:
    """Sum the sizes of files in ``directory`` matching any of ``patterns``."""
    if not directory or not os.path.isdir(directory):
        return 0
    total = 0
    for pattern in patterns:
        for file_path in Path(directory).glob(pattern):
            try:
                total += file_path.stat().st_size
            except OSError:
                continue
    return total


def _count_objects(labels) -> int | None:
    """Best effort total segmented object count across one position's masks."""
    if labels is None:
        return None
    try:
        if isinstance(labels, dict):
            stacks = list(labels.values())
        elif isinstance(labels, (list | tuple)):
            stacks = list(labels)
        else:
            stacks = [labels]

        total = 0
        for stack in stacks:
            arr = np.asarray(stack)
            if arr.ndim < 2:
                continue
            frames = arr if arr.ndim > 2 else arr[np.newaxis, ...]
            for frame in frames:
                if frame.max() > 0:
                    total += len(np.unique(frame)) - 1
        return total
    except (TypeError, ValueError, AttributeError):
        return None


def _worker_count(main_window) -> int | None:
    """The FL loader's thread count for this run, reusing its own heuristic."""
    try:
        from SECQUOIA.core.fluorescence_loading import (
            _worker_count as _fl_worker_count,
        )

        return _fl_worker_count(main_window)
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
        return None


class _NullStage:
    """Context manager returned by a no-op ``PositionRun`` stage."""

    def __enter__(self) -> _NullStage:
        return self

    def __exit__(self, *exc_info) -> bool:
        return False


class _NullPositionRun:
    """Stand-in used when ``SECQUOIA_MEASURE`` is unset: does nothing."""

    def stage(self, _name: str) -> _NullStage:
        return _NullStage()

    def stage_group(self, _names: tuple[str, ...]) -> _NullStage:
        return _NullStage()

    def wrap_timed(self, _name, func):
        return func

    def finish(self, *_args, **_kwargs) -> None:
        return None


class PositionRun:
    """Times and samples the RAM of one position's load and quantify pass."""

    def __init__(self, mode: str, groups: set[str]) -> None:
        self._groups = groups
        self._process = _psutil().Process()
        self._run_id = uuid.uuid4().hex[:12]
        self._mode = mode
        self._t0 = time.perf_counter()
        self._cpu0 = self._process.cpu_times()
        self._rss_before = self._process.memory_info().rss
        self._stage_times: dict[str, float] = {}
        self._stage_peaks: dict[str, int] = {}
        self._active_token: contextvars.Token | None = None

    @classmethod
    def start(
        cls, position, mode: str = "Positions"
    ) -> PositionRun | _NullPositionRun:
        """Begin measuring one position, or return a no-op if disabled."""
        groups = active_groups()
        if groups is None:
            return _NullPositionRun()
        try:
            run = cls(mode, groups)
        except (ImportError, OSError):
            LOG.warning(
                "[profiling] could not start measurement for position %s",
                position,
                exc_info=True,
            )
            return _NullPositionRun()
        run._active_token = _ACTIVE_RUN.set(run)
        return run

    @contextmanager
    def stage(self, name: str):
        """Time one named, non-overlapping stage and sample its RAM peak."""
        with self.stage_group((name,)):
            t0 = time.perf_counter()
            try:
                yield self
            finally:
                self._add_time(name, time.perf_counter() - t0)

    @contextmanager
    def stage_group(self, names: tuple[str, ...]):
        """Sample one RAM peak shared by several concurrently running stages."""
        sampler = (
            _RssSampler(self._process).start()
            if "memory" in self._groups
            else None
        )
        try:
            yield self
        finally:
            if sampler is not None:
                peak = sampler.stop_and_peak()
                for name in names:
                    self._stage_peaks[name] = max(
                        self._stage_peaks.get(name, 0), peak
                    )

    def wrap_timed(self, name: str, func):
        """Wrap ``func`` so its own call duration is added to stage ``name``."""

        def _wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                self._add_time(name, time.perf_counter() - t0)

        return _wrapped

    def _add_time(self, name: str, elapsed_s: float) -> None:
        self._stage_times[name] = self._stage_times.get(name, 0.0) + elapsed_s

    def finish(self, main_window, position) -> None:
        """Compute derived columns and append one row to the measurement CSV."""
        try:
            self._finish(main_window, position)
        except Exception:  # noqa: BLE001 - outer boundary, see docstring
            LOG.warning(
                "[profiling] failed to record measurement for position %s",
                position,
                exc_info=True,
            )
        finally:
            if self._active_token is not None:
                _ACTIVE_RUN.reset(self._active_token)
                self._active_token = None

    def _finish(self, main_window, position) -> None:
        t_total = time.perf_counter() - self._t0
        rss_after = self._process.memory_info().rss
        cpu1 = self._process.cpu_times()
        cpu_time_s = (cpu1.user - self._cpu0.user) + (
            cpu1.system - self._cpu0.system
        )

        row = dict.fromkeys(CSV_COLUMNS)
        row.update(_environment_record())
        row["run_id"] = self._run_id
        row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        row["tag"] = os.environ.get("SECQUOIA_MEASURE_TAG", "")
        row["position"] = position
        row["mode"] = self._mode
        row["crop_size"] = os.environ.get("SECQUOIA_MEASURE_CROP_SIZE", "")
        row["crop_offset_px"] = os.environ.get(
            "SECQUOIA_MEASURE_CROP_OFFSET_PX", ""
        )

        scaling = _scaling_factors(main_window, position)
        row.update(scaling)

        if "time" in self._groups:
            row["t_total"] = round(t_total, 4)
            for name in _TIME_STAGES:
                if name in self._stage_times:
                    row[f"t_{name}"] = round(self._stage_times[name], 4)

        if "memory" in self._groups:
            mb = 1024**2
            row["rss_before"] = round(self._rss_before / mb, 2)
            row["rss_after"] = round(rss_after / mb, 2)
            row["rss_delta"] = round((rss_after - self._rss_before) / mb, 2)
            for name in _MEMORY_STAGES:
                peak = self._stage_peaks.get(name)
                if peak:
                    row[f"rss_peak_{name}"] = round(peak / mb, 2)

        if "disk" in self._groups:
            row["memmap_bytes"] = _dir_size_bytes(
                getattr(main_window, "_memmap_dir", None)
            )
            row["analysis_dir_bytes"] = _dir_size_bytes(
                _find_analysis_dir(main_window)
            )
            row["input_bytes_read"] = _estimate_input_bytes(main_window)

        if "cpu" in self._groups:
            row["cpu_time_s"] = round(cpu_time_s, 4)
            row["cpu_util_pct"] = (
                round(100.0 * cpu_time_s / t_total, 1) if t_total > 0 else None
            )
            row["n_workers"] = _worker_count(main_window)

        row["theoretical_bytes"] = _theoretical_bytes(scaling)
        _add_derived_rates(row, scaling, self._stage_peaks, self._stage_times)

        _append_row(_csv_path(main_window), row)


def _add_derived_rates(
    row: dict,
    scaling: dict,
    stage_peaks: dict[str, int],
    stage_times: dict[str, float],
) -> None:
    """Fill the ratio/rate columns that combine two other measurements."""
    peak_total = max((v for v in stage_peaks.values() if v), default=None)
    theoretical_bytes = row.get("theoretical_bytes")
    if peak_total and theoretical_bytes:
        row["ram_peak_to_theoretical_ratio"] = round(
            peak_total / theoretical_bytes, 3
        )

    load_bytes = row.get("input_bytes_read")
    t_load = stage_times.get("load_channels", 0.0) + stage_times.get(
        "load_masks", 0.0
    )
    if load_bytes and t_load:
        row["throughput_mb_s"] = round(load_bytes / (1024**2) / t_load, 2)

    n_timepoints = scaling.get("n_timepoints")
    t_total = row.get("t_total")
    if n_timepoints and t_total:
        row["sec_per_timepoint"] = round(t_total / n_timepoints, 4)


def _find_analysis_dir(main_window) -> str | None:
    try:
        from SECQUOIA.core.experiment_layout import (
            find_analysis_dir,
        )

        return find_analysis_dir(getattr(main_window, "folder", None))
    except (ImportError, OSError):
        return None


def _csv_path(main_window) -> str:
    override = os.environ.get("SECQUOIA_MEASURE_DIR")
    if override:
        return os.path.join(override, "SECQUOIA_measure.csv")
    analysis_dir = _find_analysis_dir(main_window)
    if analysis_dir:
        return os.path.join(analysis_dir, "SECQUOIA_measure.csv")
    return os.path.join(
        getattr(main_window, "folder", ".") or ".", "SECQUOIA_measure.csv"
    )


def _append_row(path: str, row: dict) -> None:
    path = _path_for_current_schema(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _path_for_current_schema(path: str) -> str:
    """Redirect to a sibling file if ``path`` exists with an older schema."""
    if not os.path.exists(path):
        return path
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            existing_header = next(csv.reader(fh), [])
    except OSError:
        return path
    if tuple(existing_header) == CSV_COLUMNS:
        return path

    digest = hashlib.sha1(
        ",".join(CSV_COLUMNS).encode(), usedforsecurity=False
    ).hexdigest()[:6]
    stem, ext = os.path.splitext(path)
    return f"{stem}_{digest}{ext}"


def _scaling_factors(
    main_window, position, *, include_object_count: bool = True
) -> dict:
    """The per position values the CSV's scaling factor columns record."""
    factors: dict = dict.fromkeys(
        (
            "n_timepoints",
            "n_channels",
            "n_mask_sets",
            "n_tracked_cells",
            "n_objects_total",
            "image_h",
            "image_w",
            "dtype",
        )
    )

    try:
        from SECQUOIA.utils.timing import current_t_range

        _, _, t_idx_min, t_idx_max = current_t_range(main_window)
        factors["n_timepoints"] = max(0, t_idx_max - t_idx_min + 1)
    except (ImportError, TypeError, ValueError):
        pass

    n_channels = getattr(main_window, "n_channels", None)
    if n_channels is not None:
        with suppress(TypeError, ValueError):
            factors["n_channels"] = int(n_channels)

    seg_paths = getattr(main_window, "segmentation_paths", None) or []
    with suppress(TypeError):
        factors["n_mask_sets"] = len(seg_paths)

    factors["n_tracked_cells"] = _n_tracked_cells(main_window, position)
    if include_object_count:
        factors["n_objects_total"] = _count_objects(
            getattr(main_window, "labels", None)
        )

    image_h, image_w, dtype = _first_image_shape(
        getattr(main_window, "images", None) or {}
    )
    factors["image_h"] = image_h
    factors["image_w"] = image_w
    factors["dtype"] = dtype

    return factors


def _n_tracked_cells(main_window, position) -> int | None:
    """Unique tracked lineages for this position."""
    track_df = getattr(main_window, "track_df", None)
    if track_df is None or "Position" not in getattr(track_df, "columns", []):
        return None
    try:
        subset = track_df[track_df["Position"] == position]
        if "Identification" not in subset.columns:
            return None
        return int(subset["Identification"].nunique())
    except (KeyError, TypeError, ValueError):
        return None


def _first_image_shape(
    images: dict,
) -> tuple[int | None, int | None, str | None]:
    """Height, width and dtype of the first loaded (non-empty) FL channel."""
    for arr in images.values():
        if arr is None:
            continue
        try:
            a = np.asarray(arr)
        except (TypeError, ValueError):
            continue
        if a.ndim >= 2:
            return int(a.shape[-2]), int(a.shape[-1]), str(a.dtype)
    return None, None, None


def _theoretical_bytes(scaling: dict) -> int | None:
    """T x C x H x W x itemsize: the raw FL data volume, ignoring masks."""
    n_timepoints = scaling.get("n_timepoints")
    n_channels = scaling.get("n_channels")
    height = scaling.get("image_h")
    width = scaling.get("image_w")
    dtype_str = scaling.get("dtype")
    if not all([n_timepoints, n_channels, height, width, dtype_str]):
        return None
    try:
        itemsize = np.dtype(dtype_str).itemsize
    except TypeError:
        return None
    return n_timepoints * n_channels * height * width * itemsize


def _estimate_input_bytes(main_window) -> int:
    """Best effort bytes read for this position: FL + mask files on disk."""
    total = 0
    position_dir = getattr(main_window, "position_selection", None)
    image_format = getattr(main_window, "image_format", None)
    if position_dir and image_format:
        try:
            n_channels = int(getattr(main_window, "n_channels", 0) or 0)
        except (TypeError, ValueError):
            n_channels = 0
        for i in range(n_channels):
            identifier = getattr(main_window, f"FL_identifiers_{i + 1}", None)
            if identifier:
                total += _glob_bytes(
                    position_dir, [f"*{identifier}.{image_format}"]
                )

    if not position_dir or not image_format:
        return total

    pos_name = os.path.basename(str(position_dir))
    seg_paths = getattr(main_window, "segmentation_paths", None) or []
    if not isinstance(seg_paths, (list | tuple)):
        seg_paths = [seg_paths]
    for seg_root in seg_paths:
        seg_dir = os.path.join(str(seg_root), pos_name)
        total += _glob_bytes(seg_dir, [f"*.{image_format}"])

    return total


def _record_edit_row(
    main_window,
    position,
    elapsed_s: float,
    rss_before: int,
    rss_after: int,
    cpu_time_s: float,
) -> None:
    """Build and append one ``mode="Edit"`` row for :func:`mask_edit_measurement`."""
    try:
        groups = active_groups() or set()

        row = dict.fromkeys(CSV_COLUMNS)
        row.update(_environment_record())
        row["run_id"] = uuid.uuid4().hex[:12]
        row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        row["tag"] = os.environ.get("SECQUOIA_MEASURE_TAG", "")
        row["position"] = position
        row["mode"] = "Edit"

        row.update(
            _scaling_factors(main_window, position, include_object_count=False)
        )

        if "time" in groups:
            row["t_total"] = round(elapsed_s, 4)

        if "memory" in groups:
            mb = 1024**2
            row["rss_before"] = round(rss_before / mb, 2)
            row["rss_after"] = round(rss_after / mb, 2)
            row["rss_delta"] = round((rss_after - rss_before) / mb, 2)

        if "cpu" in groups:
            row["cpu_time_s"] = round(cpu_time_s, 4)
            row["cpu_util_pct"] = (
                round(100.0 * cpu_time_s / elapsed_s, 1)
                if elapsed_s > 0
                else None
            )

        _append_row(_csv_path(main_window), row)
    except Exception:  # noqa: BLE001 - outer boundary, see docstring
        LOG.warning(
            "[profiling] failed to record an edit measurement for position %s",
            position,
            exc_info=True,
        )
