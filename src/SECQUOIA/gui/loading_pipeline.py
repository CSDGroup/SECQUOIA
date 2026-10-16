"""Loading an experiment and showing it in the viewers."""

from __future__ import annotations

import concurrent.futures
import contextlib
import copy
import logging
import os
import shutil
import tempfile
import threading
from types import SimpleNamespace

from qtpy.QtCore import QObject, Qt, QTimer, Signal, Slot
from qtpy.QtWidgets import QApplication, QLabel, QProgressBar

from SECQUOIA.core.basic_correction import (
    _prepare_basic_correction_inputs,
    apply_basic_correction,
)
from SECQUOIA.core.fluorescence_loading import load_fl_channels
from SECQUOIA.core.gap_filling import add_missing_rows
from SECQUOIA.core.memmap_store import cleanup_memmaps
from SECQUOIA.core.project_state import save_track_df
from SECQUOIA.core.quantification import quantify
from SECQUOIA.core.segmentation.mask_io import load_masks
from SECQUOIA.core.tracking.track_data import (
    _apply_import_realtime_to_track_df,
    _cache_position_measurements,
    _load_cached_position_measurements,
    _refresh_filtered_df,
    apply_derived_features,
)
from SECQUOIA.gui.curation_tree import update_list
from SECQUOIA.utils.positions import (
    detected_position_numbers,
    position_number_from_folder,
    resolve_position_range,
)
from SECQUOIA.utils.profiling import PositionRun
from SECQUOIA.utils.timing import (
    current_t_range as _current_t_range,
)
from SECQUOIA.utils.timing import (
    rebase_track_df_to_t_range,
)

LOG = logging.getLogger(__name__)

__all__ = [
    "SharedProgress",
    "UiProgressBridge",
    "run_all_positions",
    "run_loading",
    "update_images",
]


class SharedProgress(QObject):
    """Combine progress from multiple weighted processing stages into one signal.
    Each named processing part, such as fluorescence loading, BaSiC correction,
    measurement, or UI refresh, contributes a weighted fraction to the total
    progress."""

    progressed = Signal(int, str)

    def __init__(self, weights: dict[str, float], parent=None):
        """Initialize weighted progress tracking."""
        super().__init__(parent)
        self._lock = threading.Lock()
        w_sum = sum(weights.values()) or 1.0
        self._weights = {k: (v / w_sum) for k, v in weights.items()}
        self._parts = dict.fromkeys(self._weights, 0.0)

    def update(self, part: str, frac: float, msg: str | None = None):
        """Update one progress part and emit the combined total progress."""
        frac = max(0.0, min(1.0, float(frac)))
        with self._lock:
            prev = self._parts.get(part, 0.0)
            frac = max(prev, frac)
            self._parts[part] = frac
            total = sum(
                self._weights[k] * self._parts[k] for k in self._weights
            )
        pct = int(round(total * 100))
        pct = 0 if pct < 0 else 100 if pct > 100 else pct
        self.progressed.emit(pct, msg or f"{part}: {int(frac * 100)}%")


class UiProgressBridge(QObject):
    """Thread safe bridge to update a QProgressBar and a QLabel from worker threads."""

    update = Signal(int, str)

    def __init__(self, bar: QProgressBar, label: QLabel, parent=None):
        """Store the target widgets and queue updates onto the GUI thread."""
        super().__init__(parent)
        self._bar = bar
        self._label = label
        self.update.connect(self._on_update, Qt.QueuedConnection)

    @Slot(int, str)
    def _on_update(self, val: int, msg: str):
        """Update the progress bar and label on the GUI thread."""
        self._bar.setValue(int(max(0, min(100, val))))
        if msg:
            self._label.setText(msg)
        QApplication.processEvents()


def _set_loading_status(main_window, text, kind="info") -> None:
    """Update the Loading tab's live status label with optional error styling."""
    label = getattr(main_window, "progress_msg_label", None)
    if label is None:
        return
    with contextlib.suppress(RuntimeError, AttributeError, TypeError):
        if kind == "error":
            label.setStyleSheet("color: #ff4d4f; font-weight: 600;")
        else:
            label.setStyleSheet("")
        label.setText(text)


def run_loading(main_window) -> None:
    """Entry point for the Loading tab's "Run" button.

    Validates the selected folders and channel identifiers, then dispatches to the
    appropriate loader based on the chosen loading mode.
    """
    _set_loading_status(main_window, "Preparing to run…", kind="info")

    if not getattr(main_window, "folder", None) or not getattr(
        main_window, "segmentation_paths", []
    ):
        _set_loading_status(
            main_window,
            "Error: Please select an Experiment folder and at least one Segmentation folder.",
            kind="error",
        )
        return

    if not getattr(main_window, "FL_inputs", None) or len(
        main_window.FL_inputs
    ) < int(getattr(main_window, "n_channels", 0)):
        _set_loading_status(
            main_window,
            "Error: Please select channels to quantify (FL identifiers are missing).",
            kind="error",
        )
        return

    # Dynamically set FL identifiers as attributes (e.g., FL_identifiers_1, _2, ...)
    for i in range(main_window.n_channels):
        setattr(
            main_window,
            f"FL_identifiers_{i + 1}",
            main_window.FL_inputs[i].text(),
        )

    # Ensure all identifiers are properly set
    for i in range(main_window.n_channels):
        identifier = getattr(main_window, f"FL_identifiers_{i + 1}")
        if not identifier:
            _set_loading_status(
                main_window,
                f"Error: Please provide a valid identifier for FL Channel {i + 1}.",
                kind="error",
            )
            return

    mode = getattr(main_window, "loading_format", "Positions")
    _set_loading_status(
        main_window, f"Loading mode selected → {mode}", kind="info"
    )

    if mode == "All":
        _set_loading_status(
            main_window,
            "Running measurements for ALL selected positions…",
            kind="info",
        )

        pos_min, pos_max = resolve_position_range(main_window)

        # Build the grid from detected position numbers in range
        nums = detected_position_numbers(main_window, pos_min, pos_max)

        if hasattr(main_window, "_init_position_status_grid"):
            main_window._init_position_status_grid(
                nums or range(pos_min, pos_max + 1)
            )
        run_all_positions(main_window)
        return

    if mode == "Positions":
        _set_loading_status(
            main_window, "Loading selected positions…", kind="info"
        )
        _load_data(main_window)
        return

    if mode == "Positions+Background loading":
        _set_loading_status(
            main_window,
            "Note: 'Positions+Background loading' is not implemented yet.",
            kind="error",
        )
        LOG.warning(
            "Status: 'Positions+Background loading' is not implemented yet."
        )
        return

    _set_loading_status(
        main_window, "Loading selected positions…", kind="info"
    )
    _load_data(main_window)


def _rebase_to_t_range(main_window) -> None:
    """Filter ``track_df`` to the selected t range and rebase ``t`` to 0."""
    try:
        rebase_track_df_to_t_range(main_window)
    except (RuntimeError, AttributeError, TypeError, ValueError) as e:
        LOG.warning("Could not filter/rebase CSV to t range: %s", e)


def _load_data(main_window) -> None:
    """Load fluorescence images and masks for the current position, then
    measure it (or reuse cached measurements) and refresh the viewer."""
    if getattr(main_window, "_loading_in_progress", False):
        return
    main_window._loading_in_progress = True

    _set_loading_status(main_window, "")

    weights = {"fl": 0.35, "basic": 0.25, "measure": 0.35, "ui": 0.05}
    shared = SharedProgress(
        weights, parent=getattr(main_window, "mask_no_window", None)
    )

    def _on_progress(pct: int, msg: str):
        with contextlib.suppress(AttributeError, RuntimeError, TypeError):
            main_window.set_progress(pct)

        _set_loading_status(main_window, msg or "")
        QApplication.processEvents()

    shared.progressed.connect(_on_progress)

    if not main_window.position_folders:
        _set_loading_status(main_window, "No position folders found.")

        if hasattr(main_window, "finish_progress"):
            main_window.finish_progress()
            QApplication.processEvents()
        main_window._loading_in_progress = False
        return

    main_window.position_selection = main_window.position_folders[
        main_window.current_position_index
    ]

    position = position_number_from_folder(main_window.position_selection)
    run = PositionRun.start(position, mode="Positions")

    # Load BaSiC metrics (if enabled + available)
    t_file_min, t_file_max = _prepare_basic_correction_inputs(main_window)

    seg_input = getattr(main_window, "segmentation_paths", None)

    main_window.images = {}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
    load_stages = run.stage_group(("load_channels", "load_masks"))
    load_stages.__enter__()

    shared.update("fl", 0.0, "Starting: Load fluorescence")

    future_fl = executor.submit(
        run.wrap_timed("load_channels", load_fl_channels),
        main_window,
        lambda f, m=None: shared.update("fl", f, m),
    )

    future_segmentation = executor.submit(
        run.wrap_timed("load_masks", load_masks),
        seg_input,
        main_window.position_selection,
        main_window.image_format,
        t_file_min=t_file_min,
        t_file_max=t_file_max,
    )

    stack = [future_fl, future_segmentation]

    def _finish_loading_steps():
        _rebase_to_t_range(main_window)

        skip_measure = False
        try:
            position = int(
                os.path.basename(main_window.position_selection).split("_p")[
                    -1
                ]
            )
            cached_df = _load_cached_position_measurements(
                main_window,
                position,
                t_file_min,
                t_file_max,
                log_prefix="load_data",
            )
            skip_measure = cached_df is not None
        except (RuntimeError, AttributeError, TypeError) as e:
            LOG.warning(
                "[load_data] Could not check/load precomputed CSV: %s", e
            )

        try:
            # Run Basic
            shared.update("basic", 0.0, "Starting: Background correction")
            with run.stage("bg_correction"):
                apply_basic_correction(
                    main_window,
                    progress_cb=lambda f, m=None: shared.update("basic", f, m),
                )

            if not skip_measure:
                shared.update("measure", 0.0, "Starting: Measurements")
                with run.stage("quantify"):
                    quantify(
                        main_window,
                        progress_cb=lambda f, m=None: shared.update(
                            "measure", f, m
                        ),
                    )

            apply_derived_features(main_window)

            shared.update("ui", 0.00, "Updating viewer…")
            main_window.update_napari_viewer()
            QApplication.processEvents()

            shared.update("ui", 0.33, "Adding labels…")
            add_missing_rows(main_window)
            _apply_import_realtime_to_track_df(main_window)
            main_window.filtered_df = main_window.track_df[
                main_window.track_df["Position"] == position
            ].copy()
            QApplication.processEvents()

            shared.update("ui", 1.00, "Finalizing…")
            main_window.update_window_title()
            QApplication.processEvents()

            try:
                with run.stage("save"):
                    save_track_df(main_window)
            except (RuntimeError, AttributeError, TypeError, OSError) as e:
                LOG.warning("[load_data] Could not save updated CSV: %s", e)

            btn = getattr(main_window, "start_curation_btn", None)
            if btn is not None:
                with contextlib.suppress(
                    RuntimeError, AttributeError, TypeError
                ):
                    btn.setEnabled(True)
                    btn.show()

            QTimer.singleShot(
                0, lambda: main_window._force_max_layout(ratio=0.30)
            )
            shared.update("fl", 1.0)
            shared.update("basic", 1.0)
            shared.update("measure", 1.0, "All done")
        finally:
            run.finish(main_window, position)

    def _finish_when_ready():
        if any(not f.done() for f in stack):
            QTimer.singleShot(15, _finish_when_ready)
            return

        try:
            main_window.labels = future_segmentation.result()
        finally:
            executor.shutdown(wait=False)
            load_stages.__exit__(None, None, None)

        for i in range(len(stack)):
            if stack[i].exception():
                LOG.error(
                    "Error loading stack %d: %s", i, stack[i].exception()
                )

        try:
            _finish_loading_steps()
        except (
            RuntimeError,
            AttributeError,
            TypeError,
            ValueError,
            OSError,
            KeyError,
        ) as e:
            LOG.error("[load_data] Failed to finish loading: %s", e)
            _set_loading_status(main_window, f"Error: {e}", kind="error")
        finally:
            if hasattr(main_window, "finish_progress"):
                with contextlib.suppress(
                    RuntimeError, AttributeError, TypeError
                ):
                    main_window.finish_progress()
                QApplication.processEvents()
            main_window._loading_in_progress = False

    QTimer.singleShot(0, _finish_when_ready)


_RUN_ALL_PHASE_WEIGHTS = {"fl": 0.35, "basic": 0.25, "meas": 0.40}
_RUN_ALL_PHASE_OFFSETS = {
    "fl": 0.0,
    "basic": _RUN_ALL_PHASE_WEIGHTS["fl"],
    "meas": _RUN_ALL_PHASE_WEIGHTS["fl"] + _RUN_ALL_PHASE_WEIGHTS["basic"],
}


def _normalize_folder_path(path: str) -> str:
    """Normalize a folder path so it can be matched case/format-insensitively."""
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def _collect_positions_to_run(
    main_window, pmin: int, pmax: int
) -> list[tuple[int, str]]:
    """Return (position_number, folder_path) pairs within [pmin, pmax], sorted by number."""
    in_range = set(detected_position_numbers(main_window, pmin, pmax))
    numbered = (
        (position_number_from_folder(p), p)
        for p in getattr(main_window, "position_folders", None) or []
    )
    items = [(n, p) for n, p in numbered if n in in_range]
    items.sort(key=lambda item: item[0])
    return items


def _build_position_index_map(main_window) -> dict[str, int]:
    """Map each normalized position folder path to its index in main_window.position_folders."""
    return {
        _normalize_folder_path(p): i
        for i, p in enumerate(getattr(main_window, "position_folders", []))
    }


def _init_run_all_progress_bar(gb, total: int) -> None:
    """Configure the global progress bar for a run-all-positions pass."""
    with contextlib.suppress(
        RuntimeError, AttributeError, ValueError, TypeError, OSError
    ):
        gb.setRange(0, total)
        gb.setValue(0)
        gb.setFormat("Position %v/%m")
        gb.setVisible(True)
        QApplication.processEvents()


def _teardown_run_all_progress_bar(gb) -> None:
    """Hide and reset the global progress bar after a run-all-positions pass."""
    with contextlib.suppress(
        RuntimeError, AttributeError, ValueError, TypeError, OSError
    ):
        gb.setVisible(False)
        gb.reset()
        QApplication.processEvents()


def _make_progress_setters(main_window):
    """Build (set_percent, set_message) callables that update the loading UI safely."""
    sp = getattr(main_window, "set_progress", None)

    def setp(x):
        if sp is None:
            return
        with contextlib.suppress(
            RuntimeError, AttributeError, ValueError, TypeError, OSError
        ):
            sp(int(max(0, min(100, x))))

    def setmsg(s):
        label = getattr(main_window, "progress_msg_label", None)
        if label:
            with contextlib.suppress(
                RuntimeError, AttributeError, ValueError, TypeError, OSError
            ):
                label.setText(s or "")

    return setp, setmsg


def _phase_progress_callback(phase: str, setp, setmsg):
    """Build a progress_cb that maps one sub-phase's 0..1 fraction onto its slice of 0..100%."""
    weight = _RUN_ALL_PHASE_WEIGHTS[phase]
    offset = _RUN_ALL_PHASE_OFFSETS[phase]

    def inner(frac, msg=None):
        setp((offset + max(0.0, min(1.0, float(frac))) * weight) * 100)
        if msg is not None:
            setmsg(msg or "")

    return inner


def _set_position_name_label(main_window, pnum: int) -> None:
    """Update the on-screen 'pNNNN' label for the position currently being processed."""
    name_label = getattr(main_window, "position_name_label", None)
    if name_label:
        with contextlib.suppress(
            RuntimeError, AttributeError, ValueError, TypeError, OSError
        ):
            name_label.setText(f"p{pnum:04d}")


_PREFETCH_ATTRS = (
    "n_channels",
    "ids_channels",
    "image_format",
    "fl_max_workers",
    "basic_max_workers",
    "time_min_selected",
    "time_max_selected",
)

_LOADED_INPUT_ATTRS = (
    "images",
    "image_present",
    "_memmap_files",
    "_memmap_dir",
    "_memmap_lock",
    "_export_display_range",
    "labels",
)

_BASIC_OUTPUT_ATTRS = (
    "corrected_images_ratioflat",
    "corrected_images_noratioflat",
    "_ratioflat_memmap_files",
    "_noratioflat_memmap_files",
    "basic_valid_frames",
)

_PREFETCH_MARGIN = 1.2
_PREFETCH_RESERVE_BYTES = 2 * 1024**3

_LOAD_ERRORS = (
    RuntimeError,
    AttributeError,
    ValueError,
    TypeError,
    OSError,
    KeyError,
)


def _nbytes(obj) -> int:
    """Total bytes of the arrays in `obj`."""
    if isinstance(obj, dict):
        return sum(_nbytes(v) for v in obj.values())
    if isinstance(obj, list | tuple):
        return sum(_nbytes(v) for v in obj)
    return int(getattr(obj, "nbytes", 0) or 0)


def _prefetch_needs(main_window) -> tuple[int, int, int]:
    """Estimated (raw disk, BaSiC disk, RAM) bytes to prefetch one position."""
    images = getattr(main_window, "images", None) or {}
    basic = getattr(main_window, "basic", None)
    basic_disk = 0
    if getattr(basic, "flag", False):
        for channel in getattr(basic, "darkfield", None) or {}:
            stack = images.get(channel)
            basic_disk += 2 * 4 * int(getattr(stack, "size", 0) or 0)
    ram = _nbytes(getattr(main_window, "labels", None))
    return _nbytes(images), basic_disk, ram


def _free_resources() -> tuple[int, int]:
    """(available RAM, free scratch disk) in bytes."""
    import psutil

    ram = int(psutil.virtual_memory().available)
    disk = int(shutil.disk_usage(tempfile.gettempdir()).free)
    return ram, disk


def _prefetch_plan(main_window) -> str | None:
    """How much of the next position may be prefetched right now."""
    raw_disk, basic_disk, ram_need = _prefetch_needs(main_window)
    try:
        ram_free, disk_free = _free_resources()
    except (ImportError, OSError) as e:
        LOG.warning("[prefetch] free RAM/disk unknown (%s); no prefetch.", e)
        return None

    def fits(need: int, free: int) -> bool:
        return need * _PREFETCH_MARGIN + _PREFETCH_RESERVE_BYTES <= free

    if not (fits(ram_need, ram_free) and fits(raw_disk, disk_free)):
        return None
    if basic_disk and not fits(raw_disk + basic_disk, disk_free):
        return "images"
    return "full"


def _load_position_inputs(staging, segmentation_paths, tmin: int, tmax: int):
    """Load one position's masks and FL images into `staging`, then BaSiC."""
    try:
        staging.labels = load_masks(
            segmentation_paths,
            staging.position_selection,
            staging.image_format,
            t_file_min=tmin,
            t_file_max=tmax,
        )
        load_fl_channels(staging)
        staging.basic_done = False
        if getattr(staging.basic, "flag", False):
            staging.basic.update_all(position=staging.current_position_number)
            apply_basic_correction(staging)
            staging.basic_done = True
    except BaseException:
        cleanup_memmaps(staging, remove_viewer_image_layers=False)
        raise
    return staging


class _NextPositionPrefetch:
    """Loads the next position's images, masks and BaSiC in the background."""

    def __init__(self, main_window, tmin: int, tmax: int) -> None:
        self._main_window = main_window
        self._tmin, self._tmax = tmin, tmax
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="prefetch"
        )
        self._pending: tuple[int, concurrent.futures.Future] | None = None

    def start(self, pnum: int, ppath: str) -> str | None:
        """Begin loading `pnum` if resources allow; returns the plan used."""
        self._discard()
        mw = self._main_window
        plan = _prefetch_plan(mw)
        if plan is None:
            LOG.warning(
                "[prefetch] p%04d not prefetched: not enough free RAM or "
                "scratch disk; it loads after the current position.",
                pnum,
            )
            return None
        if plan == "images":
            LOG.info(
                "[prefetch] p%04d: not enough scratch disk to prefetch BaSiC "
                "too; it runs after the current position.",
                pnum,
            )
        basic = getattr(mw, "basic", None)
        staging = SimpleNamespace(
            **{attr: getattr(mw, attr, None) for attr in _PREFETCH_ATTRS},
            position_selection=ppath,
            current_position_number=int(pnum),
            _memmap_dir=None,
            basic=copy.copy(basic) if plan == "full" else None,
        )
        for ci in range(int(mw.n_channels)):
            name = f"FL_identifiers_{ci + 1}"
            setattr(staging, name, getattr(mw, name, None))
        future = self._executor.submit(
            _load_position_inputs,
            staging,
            getattr(mw, "segmentation_paths", []),
            self._tmin,
            self._tmax,
        )
        self._pending = (int(pnum), future)
        return plan

    def take(self, pnum: int) -> concurrent.futures.Future | None:
        """The prefetch for `pnum`, or None when none was started for it."""
        if self._pending is None or self._pending[0] != int(pnum):
            self._discard()
            return None
        future = self._pending[1]
        self._pending = None
        return future

    def close(self) -> None:
        """Drop an unused prefetch and stop the background thread."""
        self._discard()
        self._executor.shutdown(wait=True)

    def _discard(self) -> None:
        """Wait for an unused prefetch and delete its scratch files."""
        if self._pending is None:
            return
        _pnum, future = self._pending
        self._pending = None
        try:
            staging = future.result()
        except _LOAD_ERRORS:
            return
        cleanup_memmaps(staging, remove_viewer_image_layers=False)


def _adopt_prefetched_inputs(main_window, future, pnum: int):
    """Hand a prefetched position to `main_window`."""
    try:
        staging = future.result()
    except _LOAD_ERRORS as e:
        LOG.warning(
            "[run_all_positions] prefetch of p%04d failed (%s); loading it now.",
            pnum,
            e,
        )
        return None
    attrs = _LOADED_INPUT_ATTRS
    if staging.basic_done:
        attrs += _BASIC_OUTPUT_ATTRS
    for attr in attrs:
        setattr(main_window, attr, getattr(staging, attr, None))
    return staging


def _process_one_position(
    main_window,
    pnum: int,
    ppath: str,
    idx_map,
    tmin: int,
    tmax: int,
    setp,
    setmsg,
    prefetch: _NextPositionPrefetch | None = None,
    next_item: tuple[int, str] | None = None,
) -> None:
    """Select, load, correct, mask, and quantify a single position."""
    main_window.position_selection = ppath
    j = idx_map.get(_normalize_folder_path(ppath))
    if j is not None:
        main_window.current_position_index = j
    main_window.current_position_number = int(pnum)

    _set_position_name_label(main_window, pnum)
    setp(0)
    setmsg(f"p{pnum:04d}: preparing…")

    def _start_next_prefetch():
        if prefetch is None or next_item is None:
            return
        if prefetch.start(*next_item) is None:
            setmsg(f"p{next_item[0]:04d}: not prefetched (low RAM or disk)")

    prefetched = prefetch.take(pnum) if prefetch is not None else None
    run = PositionRun.start(pnum, mode="All")
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="mask_load"
    )
    try:
        future_masks = None
        staging = None
        if prefetched is not None:
            setmsg(f"p{pnum:04d}: waiting for prefetched images…")
            with run.stage("load_channels"):
                staging = _adopt_prefetched_inputs(
                    main_window, prefetched, pnum
                )

        if staging is None:
            future_masks = executor.submit(
                load_masks,
                getattr(main_window, "segmentation_paths", []),
                ppath,
                main_window.image_format,
                t_file_min=tmin,
                t_file_max=tmax,
            )

            with run.stage("load_channels"):
                load_fl_channels(
                    main_window,
                    progress_cb=_phase_progress_callback("fl", setp, setmsg),
                )

        main_window.basic.update_all(position=pnum)
        if staging is None or not staging.basic_done:
            with run.stage("bg_correction"):
                apply_basic_correction(
                    main_window,
                    progress_cb=_phase_progress_callback(
                        "basic", setp, setmsg
                    ),
                )

        if future_masks is not None:
            with run.stage("load_masks"):
                main_window.labels = future_masks.result()

        _start_next_prefetch()

        with run.stage("quantify"):
            quantify(
                main_window,
                progress_cb=_phase_progress_callback("meas", setp, setmsg),
            )
        apply_derived_features(main_window)
        add_missing_rows(main_window)
    finally:
        executor.shutdown(wait=True)
        run.finish(main_window, pnum)


def _finish_position_iteration(
    main_window, run_all_mode, gb, i, total, setp
) -> None:
    """Advance the global progress bar and reset per position UI state after one iteration."""
    if run_all_mode and gb:
        with contextlib.suppress(
            RuntimeError, AttributeError, ValueError, TypeError, OSError
        ):
            gb.setValue(i)
    with contextlib.suppress(
        RuntimeError, AttributeError, ValueError, TypeError, OSError
    ):
        QApplication.processEvents()
    cleanup_memmaps(main_window)
    if i < total:
        setp(0)


def _close_mask_window(main_window) -> None:
    """Close and release the modal 'loading masks' window, if one is open."""
    try:
        win = getattr(main_window, "mask_no_window", None)
        if win is not None:
            with contextlib.suppress(
                RuntimeError, AttributeError, ValueError, TypeError, OSError
            ):
                win.close()
            with contextlib.suppress(
                RuntimeError, AttributeError, ValueError, TypeError, OSError
            ):
                win.deleteLater()
            main_window.mask_no_window = None
        with contextlib.suppress(
            RuntimeError, AttributeError, ValueError, TypeError, OSError
        ):
            QApplication.processEvents()
    except (RuntimeError, AttributeError, ValueError, TypeError, OSError) as e:
        LOG.warning(
            "[run_all_positions] could not close loading window: %s", e
        )


def _split_untracked_positions(
    main_window, items: list[tuple[int, str]]
) -> tuple[list[tuple[int, str]], list[int]]:
    """Split `items` into positions with tracking points and the numbers without."""
    from SECQUOIA.gui.position_navigation import tree_counts

    counts = tree_counts(main_window)
    if counts is None:
        return items, []
    kept = [(n, p) for n, p in items if counts.get(n, 0) > 0]
    skipped = [n for n, _p in items if counts.get(n, 0) == 0]
    return kept, skipped


def run_all_positions(main_window) -> None:
    """Import and measure every position in the selected range.

    Each position runs 0→100% on the per position bar with live status
    messages; the global bar advances one step per finished position.
    """
    pmin, pmax = resolve_position_range(main_window)
    items = _collect_positions_to_run(main_window, pmin, pmax)
    if not items:
        _set_loading_status(
            main_window,
            "No position folders found in the selected range.",
            kind="error",
        )
        return

    _rebase_to_t_range(main_window)
    items, skipped = _split_untracked_positions(main_window, items)
    skipped_note = ""
    if skipped:
        skipped_note = (
            f" Skipped {len(skipped)} position(s) without tracking points: "
            + ", ".join(f"p{n:04d}" for n in skipped)
        )
        LOG.info("[run_all_positions]%s", skipped_note)
        _set_loading_status(main_window, skipped_note.strip(), kind="info")
    if not items:
        _set_loading_status(
            main_window,
            "No positions with tracking points in the selected range.",
            kind="error",
        )
        return

    run_all_mode = getattr(main_window, "loading_format", "Positions") == "All"
    gb = getattr(main_window, "global_progress_bar", None)
    if run_all_mode and gb:
        _init_run_all_progress_bar(gb, len(items))

    setp, setmsg = _make_progress_setters(main_window)
    t_file_min, t_file_max = _prepare_basic_correction_inputs(main_window)
    idx_map = _build_position_index_map(main_window)

    total = len(items)
    prefetch = _NextPositionPrefetch(main_window, t_file_min, t_file_max)
    try:
        for i, (pnum, ppath) in enumerate(items, 1):
            try:
                _process_one_position(
                    main_window,
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
                setp(100)
                setmsg(f"p{pnum:04d} done  ({i}/{total})")
            except _LOAD_ERRORS as e:
                LOG.error("[run_all_positions] error p%04d: %s", pnum, e)
                setmsg(f"Error in p{pnum:04d}: {e}")
                setp(100)
            finally:
                _finish_position_iteration(
                    main_window, run_all_mode, gb, i, total, setp
                )
    finally:
        prefetch.close()

    if hasattr(main_window, "finish_progress"):
        main_window.finish_progress()
    _set_loading_status(
        main_window, "All positions finished." + skipped_note, kind="info"
    )

    if run_all_mode and gb:
        _teardown_run_all_progress_bar(gb)

    _close_mask_window(main_window)
    _apply_import_realtime_to_track_df(main_window)
    try:
        save_track_df(main_window)
    except (RuntimeError, AttributeError, TypeError, OSError) as e:
        LOG.warning("[run_all_positions] Could not save updated CSV: %s", e)

    from SECQUOIA.gui.position_navigation import _switch_to_start_position

    _switch_to_start_position(main_window, pmin, pmax, items)


def _save_previous_position_state(main_window) -> None:
    """Persist the previous position's measurements to its cached CSV before switching away."""
    try:
        prev_sel = getattr(main_window, "position_selection", None)
        if not prev_sel or not hasattr(main_window, "track_df"):
            return

        try:
            prev_pos = int(os.path.basename(prev_sel).split("_p")[-1])
        except (RuntimeError, AttributeError, TypeError):
            return

        t_file_min, t_file_max, _, _ = _current_t_range(main_window)
        _cache_position_measurements(
            main_window,
            prev_pos,
            t_file_min,
            t_file_max,
            log_prefix="update_images",
        )
    except (RuntimeError, AttributeError, TypeError) as e:
        LOG.warning(
            "[update_images] Warning while saving previous position CSV: %s", e
        )


def _resolve_target_position(main_window) -> int | None:
    """Select the folder at current_position_index and record its number.

    Returns the parsed position number, or None when it cannot be read
    from the folder name.
    """
    main_window.position_selection = main_window.position_folders[
        main_window.current_position_index
    ]

    try:
        position = int(
            os.path.basename(main_window.position_selection).split("_p")[-1]
        )
    except (RuntimeError, AttributeError, TypeError):
        position = None
    main_window.current_position_number = position
    return position


def _load_images_and_masks(
    main_window, seg_input, t_file_min, t_file_max, cb_fl, run
):
    """Load FL images and segmentation masks for the current position.

    Both run in parallel. Blocks, while pumping the Qt event loop, until
    they finish. Returns the segmentation result and also stores it on
    main_window.labels.
    """
    main_window.images = {}

    with (
        run.stage_group(("load_channels", "load_masks")),
        concurrent.futures.ThreadPoolExecutor() as executor,
    ):
        future_fl = executor.submit(
            run.wrap_timed("load_channels", load_fl_channels),
            main_window,
            cb_fl,
        )
        future_seg = executor.submit(
            run.wrap_timed("load_masks", load_masks),
            seg_input,
            main_window.position_selection,
            main_window.image_format,
            t_file_min=t_file_min,
            t_file_max=t_file_max,
        )
        pending = {future_fl, future_seg}
        while pending:
            QApplication.processEvents()
            _done, pending = concurrent.futures.wait(pending, timeout=0.01)

    if future_fl.exception():
        LOG.error("Error loading FL channel: %s", future_fl.exception())
    if future_seg.exception():
        LOG.error(
            "Error loading segmentation masks: %s", future_seg.exception()
        )

    try:
        seg_result = future_seg.result()
    except (RuntimeError, AttributeError, TypeError) as e:
        LOG.error("Segmentation result unavailable: %s", e)
        seg_result = None

    main_window.labels = seg_result
    return seg_result


def _measure_current_position(
    main_window, position, t_file_min, t_file_max, cb_basic, cb_measure, run
) -> None:
    """Populate measurements for the current position: from cache if available, else by measuring."""
    skip_measure = False
    try:
        cached_df = _load_cached_position_measurements(
            main_window,
            position,
            t_file_min,
            t_file_max,
            ensure_columns=True,
            log_prefix="update_images",
        )
        skip_measure = cached_df is not None
    except (RuntimeError, AttributeError, TypeError) as e:
        LOG.warning(
            "[update_images] Warning while checking/loading CSV for new position: %s",
            e,
        )

    main_window.basic.update_all(position=main_window.current_position_number)
    with run.stage("bg_correction"):
        apply_basic_correction(main_window, progress_cb=cb_basic)

    try:
        if not skip_measure:
            with run.stage("quantify"):
                quantify(main_window, progress_cb=cb_measure)
    except (RuntimeError, AttributeError, TypeError) as e:
        LOG.error("Fluorescence measurement error: %s", e)

    apply_derived_features(main_window)
    _refresh_filtered_df(main_window)


def update_images(main_window, progress_cb=None) -> None:
    """Load a new position: images, masks, and measurements.

    Caches the previous position's measurements, loads the new position's
    FL images and masks, measures it or reuses a cached CSV, applies the
    derived features, and repopulates the curation tree.
    """
    if getattr(main_window, "_updating_images", False):
        main_window._update_images_requested = True
        LOG.info(
            "[update_images] A position load is already in progress; "
            "queued to load the latest requested position when it finishes."
        )
        return

    main_window._updating_images = True
    main_window._update_images_requested = False
    try:
        cleanup_memmaps(main_window)
        _save_previous_position_state(main_window)

        position = _resolve_target_position(main_window)
        seg_input = getattr(main_window, "segmentation_paths", None)
        t_file_min, t_file_max, _, _ = _current_t_range(main_window)

        if isinstance(progress_cb, dict):
            cb_fl = progress_cb.get("fl")
            cb_basic = progress_cb.get("basic")
            cb_measure = progress_cb.get("measure")
        else:
            cb_fl = cb_basic = cb_measure = progress_cb

        run = PositionRun.start(position, mode="Switch")
        try:
            _load_images_and_masks(
                main_window, seg_input, t_file_min, t_file_max, cb_fl, run
            )

            _measure_current_position(
                main_window,
                position,
                t_file_min,
                t_file_max,
                cb_basic,
                cb_measure,
                run,
            )
        finally:
            run.finish(main_window, position)

        update_list(main_window)
        main_window.current_time_index = 0
    finally:
        main_window._updating_images = False

    if main_window._update_images_requested:
        main_window._update_images_requested = False
        QTimer.singleShot(0, lambda: update_images(main_window))
