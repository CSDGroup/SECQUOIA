"""Loading fluorescence images into per-channel numpy.memmap stacks."""

from __future__ import annotations

import concurrent.futures
import logging
import os
import tempfile
import threading
from glob import glob as _glob
from typing import NamedTuple

import imageio.v2 as imageio
import numpy as np

from SECQUOIA.core.memmap_store import acquire_dir_lock
from SECQUOIA.utils.io_reload import read_with_reload
from SECQUOIA.utils.timing import current_t_range as _current_t_range
from SECQUOIA.utils.timing import (
    filename_t_file_number as _filename_t_file_number,
)

LOG = logging.getLogger(__name__)

__all__ = ["load_fl_channels"]


def load_fl_channels(main_window, progress_cb=None) -> None:
    """Load FL images into per channel np.memmap stacks (uint8) over the selected t range.

    Missing frames remain 0 (black) but are also tracked via main_window.image_present.

    Frames are decoded in parallel across all channels by a bounded thread pool;
    each frame owns its own row of its channel's memmap.
    """
    progress = _resolve_progress_callback(progress_cb)
    t_file_min, t_file_max, _, _ = _current_t_range(main_window)
    n_channels = int(main_window.n_channels)

    _reset_load_state(main_window)
    read_fn, is_bgr = _reader_for_format(main_window.image_format)
    T = max(0, t_file_max - t_file_min + 1)

    opened: list[_OpenedChannel] = []
    tasks: list[tuple] = []
    channels_done = 0

    for ci, channel in enumerate(main_window.ids_channels):
        entry = _prepare_channel(
            main_window,
            ci,
            channel,
            t_file_min,
            t_file_max,
            T,
            read_fn,
            is_bgr,
        )
        if entry is None:
            channels_done += 1
            continue
        opened.append(entry)
        tasks.extend(entry.tasks)

    total_frames = len(tasks)
    max_workers = _worker_count(main_window)
    done_frames = _decode_frames(tasks, read_fn, is_bgr, max_workers, progress)

    for entry in opened:
        entry.out.flush()
        channels_done += 1
        LOG.info(
            "FL channel %s (%s) loaded via memmap: %s frames mapped into T=%s "
            "frames (t files %s..%s), dtype=%s, stack file=%s",
            entry.channel_num,
            entry.identifier,
            entry.present.sum(),
            T,
            t_file_min,
            t_file_max,
            entry.dtype,
            entry.mm_path,
        )

    # Ensure the step reaches 100% (no matter what)
    if callable(progress):
        if total_frames > 0:
            progress(1.0, f"Loading FL… {done_frames}/{total_frames}")
        else:
            progress(1.0, f"Loading FL… {channels_done}/{n_channels} channels")


def _resolve_progress_callback(progress_cb):
    """Pull the FL-phase progress callback out of a dict or bare callable."""
    if isinstance(progress_cb, dict):
        return progress_cb.get("fl", None)
    if callable(progress_cb):
        return progress_cb
    return None


def _reset_load_state(main_window) -> None:
    """Clear/initialize the per load bookkeeping a fresh FL load writes into."""
    if not hasattr(main_window, "images") or main_window.images is None:
        main_window.images = {}
    else:
        main_window.images.clear()

    if (not hasattr(main_window, "image_present")) or (
        main_window.image_present is None
    ):
        main_window.image_present = {}
    else:
        main_window.image_present.clear()

    if not hasattr(main_window, "_memmap_dir") or not main_window._memmap_dir:
        main_window._memmap_dir = tempfile.mkdtemp(prefix="SECQUOIA_memmaps_")
        main_window._memmap_lock = acquire_dir_lock(main_window._memmap_dir)

    if (not hasattr(main_window, "_memmap_files")) or (
        main_window._memmap_files is None
    ):
        main_window._memmap_files = []
    else:
        main_window._memmap_files.clear()


def _mark_channel_empty(main_window, channel) -> None:
    """Record that a channel has no frames to load for this pass."""
    main_window.images[channel] = None
    main_window.image_present[channel] = np.zeros(0, dtype=bool)


def _files_for_channel(
    position_dir: str,
    fl_identifier: str,
    image_format: str,
    t_file_min: int,
    t_file_max: int,
) -> dict[int, str]:
    """Map t-file numbers to file paths for one channel, within the selected range."""
    all_files = sorted(
        _glob(f"{position_dir}/*{fl_identifier}.{image_format}")
    )

    file_by_t = {}
    for fp in all_files:
        t_file = _filename_t_file_number(fp)
        if t_file is None:
            continue
        if t_file_min <= t_file <= t_file_max:
            file_by_t[t_file] = fp
    return file_by_t


class _OpenedChannel(NamedTuple):
    """Everything needed to finalize and log one successfully opened FL channel."""

    channel_num: int
    identifier: str
    out: np.memmap
    present: np.ndarray
    mm_path: str
    dtype: np.dtype
    tasks: list


def _prepare_channel(
    main_window, ci, channel, t_file_min, t_file_max, T, read_fn, is_bgr
) -> _OpenedChannel | None:
    """Set up one channel's memmap and decode tasks, or mark it empty."""
    fl_identifier = getattr(main_window, f"FL_identifiers_{ci + 1}", None)
    if not fl_identifier:
        LOG.warning("FL identifier for channel %d not found.", ci + 1)
        _mark_channel_empty(main_window, channel)
        return None

    if T == 0:
        LOG.warning("No t-range selected; channel %d will be empty.", ci + 1)
        _mark_channel_empty(main_window, channel)
        return None

    file_by_t = _files_for_channel(
        main_window.position_selection,
        fl_identifier,
        main_window.image_format,
        t_file_min,
        t_file_max,
    )
    if not file_by_t:
        LOG.warning(
            "No images found for FL channel %d (%s) in selected t range.",
            ci + 1,
            fl_identifier,
        )
        _mark_channel_empty(main_window, channel)
        return None

    # Read one image to get (H, W) and dtype (8-bit)
    first_fp = next(iter(file_by_t.values()))
    try:
        first_arr = _to_2d(read_with_reload(read_fn, first_fp), is_bgr)
    except (OSError, ValueError, RuntimeError) as e:
        LOG.error(
            "FL channel %d (%s): giving up reading %s after reloading (%r); "
            "channel will be empty.",
            ci + 1,
            fl_identifier,
            first_fp,
            e,
        )
        _mark_channel_empty(main_window, channel)
        return None
    H, W = first_arr.shape[:2]
    dtype = np.uint8 if first_arr.dtype == np.uint8 else first_arr.dtype

    mm_name = f"p{getattr(main_window, 'current_position_number', 0):04d}_ch{ci + 1}_t{t_file_min:05d}-{t_file_max:05d}.dat"
    mm_path = os.path.join(main_window._memmap_dir, mm_name)
    out = np.memmap(mm_path, dtype=dtype, mode="w+", shape=(T, H, W))

    present = np.zeros(T, dtype=bool)
    for tf in file_by_t:
        present[tf - t_file_min] = True

    tasks = [
        (out, tf - t_file_min, file_by_t[tf], dtype, present)
        for tf in sorted(file_by_t)
    ]

    main_window.images[channel] = out
    main_window.image_present[channel] = present
    main_window._memmap_files.append(mm_path)

    return _OpenedChannel(
        ci + 1, fl_identifier, out, present, mm_path, dtype, tasks
    )


def _decode_frames(tasks, read_fn, is_bgr, max_workers, progress) -> int:
    """Decode all queued frames in parallel, each writing into its own memmap row."""
    total_frames = len(tasks)
    if not total_frames:
        return 0

    def _load_one(task):
        out, row, fp, dtype, present = task
        try:
            arr = _to_2d(read_with_reload(read_fn, fp), is_bgr)
        except (OSError, ValueError, RuntimeError) as e:
            LOG.error(
                "Giving up on FL frame after reloading: %s (%r). Leaving "
                "frame blank/missing.",
                fp,
                e,
            )
            present[row] = False
            return
        if arr.dtype != dtype:
            arr = arr.astype(dtype, copy=False)
        out[row] = arr  # distinct row per task -> no lock needed here

    progress_lock = threading.Lock()
    progress_step = max(1, total_frames // 100)
    done_frames = 0

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="fl_load"
    ) as executor:
        for _ in executor.map(_load_one, tasks):
            with progress_lock:
                done_frames += 1
                n_done = done_frames
                emit = callable(progress) and (
                    total_frames <= 100
                    or n_done == total_frames
                    or n_done % progress_step == 0
                )
            if emit:
                progress(
                    n_done / total_frames,
                    f"Loading FL… {n_done}/{total_frames}",
                )

    return done_frames


def _reader_for_format(image_format: str):
    """Return (read_fn, is_bgr) for the given image format."""
    fmt = str(image_format or "").lower().lstrip(".")

    if fmt in ("tif", "tiff"):
        try:
            import tifffile

            return tifffile.imread, False
        except ImportError:
            pass

    return imageio.imread, False


def _to_2d(arr, is_bgr: bool):
    """Collapse a possibly-multichannel image to a single 2-D plane."""
    if arr.ndim <= 2:
        return arr
    if is_bgr and arr.shape[-1] >= 3:
        return arr[..., 2]
    return arr[..., 0]


def _worker_count(main_window=None) -> int:
    """Number of reader threads for FL loading.

    Scales with the machine but never exceeds 8, and never drops below 1.
    An explicit ``main_window.fl_max_workers`` overrides the heuristic."""
    override = getattr(main_window, "fl_max_workers", None)
    if override:
        try:
            return max(1, min(8, int(override)))
        except (TypeError, ValueError):
            pass

    n = None
    _process_cpu_count = getattr(os, "process_cpu_count", None)
    if _process_cpu_count is not None:
        n = _process_cpu_count()
    else:
        _sched_getaffinity = getattr(
            os, "sched_getaffinity", None
        )  # Linux/BSD
        if _sched_getaffinity is not None:
            n = len(_sched_getaffinity(0))
    if not n:
        n = os.cpu_count()
    if not n:
        n = 1

    return max(1, min(8, max(2, n - 1)))
