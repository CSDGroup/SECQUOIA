"""BaSiC flatfield/darkfield background correction of fluorescence stacks."""

from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import os
import re
import threading
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from qtpy.QtWidgets import QApplication

from SECQUOIA.core.experiment_layout import (
    find_analysis_dir,
    find_basic_folders,
)
from SECQUOIA.core.fluorescence_loading import (
    _worker_count as _fl_worker_count,
)
from SECQUOIA.utils.timing import current_t_range as _current_t_range

LOG = logging.getLogger(__name__)

__all__ = ["DataSetBaSiC", "apply_basic_correction"]


class DataSetBaSiC:
    """Manage BaSiC background correction files for one experiment position.

    The class loads darkfield, flatfield, ratio-flat, and baseline fluorescence
    files for the selected experiment, time range, position, and channel list.
    """

    def __init__(
        self,
        flag: bool,
        exp_name: str | None = None,
        t_range: tuple[int, int] | None = None,
        channels_valid: list[str] | None = None,
        path_basic: str | None = None,
        position: int | None = None,
    ):
        """
        Store BaSiC data set configuration for a given Position.

        Inputs:
        flag: whether BaSiC correction is enabled.
        exp_name: Experiment name.
        t_range: Time range (t_min, t_max).
        channels_valid: List of valid channel names.
        path_basic: path to the BaSiC folder.
        position: Position index (1-based).
        """
        self.flag = flag
        self.exp_name = exp_name
        self.t_min = t_range[0] if t_range else None
        self.t_max = t_range[1] if t_range else None
        self.channels_valid = channels_valid
        self.path_basic = path_basic
        self.position = position
        self.darkfield: dict[str, np.ndarray] | None = None
        self.flatfield: dict[str, np.ndarray] | None = None
        self.amp: dict[str, np.ndarray] | None = None
        self.base: dict[str, np.ndarray] | None = None

        if self.flag:
            self.update_all(position)

    def add_exp_name(self, exp_name: str) -> None:
        """Set the experiment name, if BaSiC correction is enabled."""
        if self.flag:
            self.exp_name = exp_name

    def add_t_range(self, t_range: tuple[int, int]) -> None:
        """Set ``(t_min, t_max)``, swapping the pair if given reversed.

        No-op if BaSiC correction is disabled.
        """
        if not self.flag:
            return

        t_min = t_range[0]
        t_max = t_range[1]

        if t_min > t_max:
            t_min, t_max = t_max, t_min

        self.t_min = t_min
        self.t_max = t_max

    def add_channels_valid(self, channels_valid: list[str]) -> None:
        """Set the list of valid channel names, if BaSiC is enabled."""
        if self.flag:
            self.channels_valid = channels_valid

    def add_path_basic(self, path_basic: str) -> None:
        """Set the BaSiC folder path after validating it.

        Silently ignored (with a log message) if BaSiC correction is
        disabled, if ``path_basic`` is not an existing directory, or if
        its name does not contain the literal substring ``"BaSiC"``.
        """
        if not self.flag:
            LOG.debug("BaSiC checkbox off; ignoring path_basic=%s", path_basic)
            return

        if not os.path.isdir(path_basic) or "BaSiC" not in path_basic:
            LOG.warning(
                "path_basic is not a valid BaSiC directory: %s", path_basic
            )
            return

        self.path_basic = path_basic

    def update_position(self, position: int) -> None:
        """Set the position index, if BaSiC correction is enabled."""
        if self.flag:
            self.position = position

    def _get_channel(self, fn: str) -> str:
        m = re.search(r"_(w\d+)", fn)
        return m.group(1) if m else "1"

    def _files(self, pattern: str) -> list[Path]:
        root = Path(self.path_basic) / f"{self.exp_name}_p{self.position:04d}"
        return sorted(root.glob(pattern))

    def _read_stack(self, files: list[Path]) -> dict[str, np.ndarray]:
        vals = {}
        for f in files:
            channel = self._get_channel(f.name)
            vals[channel] = imageio.imread(str(f))

        return vals

    def _read_array(
        self, files: list[Path]
    ) -> dict[str, dict[str, np.ndarray]]:
        vals = {}
        for f in files:
            channel = self._get_channel(f.name)
            times, vs = np.loadtxt(
                f,
                skiprows=1,
                delimiter=":",
                usecols=(0, 1),
                unpack=True,
                converters={0: lambda s: int(s[1:])},
            )

            times = np.asarray(times, dtype=np.int32)
            vs = np.asarray(vs, dtype=np.float32)

            mask = (times >= self.t_min) & (times <= self.t_max)

            vals[channel] = {
                "times": times[mask],
                "values": vs[mask],
            }

        return vals

    def _read_data(
        self, attr: str, files: list[Path]
    ) -> dict[str, np.ndarray] | dict[str, dict[str, np.ndarray]]:
        if attr in ["darkfield", "flatfield"]:
            return self._read_stack(files)
        else:
            return self._read_array(files)

    def _filter_valid_channels(self, data: list[str]) -> list[str]:
        return [
            f for f in data if any(ch in f.name for ch in self.channels_valid)
        ]

    def _update_component(self, attr: str, pattern: str) -> None:
        if not self.flag:
            return
        files = self._files(pattern)
        files = self._filter_valid_channels(files)
        setattr(self, attr, self._read_data(attr, files))

    def update_df(self) -> None:
        """Reload darkfield images for the current position/channels."""
        self._update_component("darkfield", "darkfield_w*.tif")

    def update_ff(self) -> None:
        """Reload flatfield images for the current position/channels."""
        self._update_component("flatfield", "flatfield_w*.tif")

    def update_amp(self) -> None:
        """Reload ratio-flat amplitude series for the current position/channels."""
        self._update_component("amp", "ratioflat_w*.txt")

    def update_base(self) -> None:
        """Reload baseline fluorescence series for the current position/channels."""
        self._update_component("base", "basefluor_w*.txt")

    def update_all(self, position: int) -> None:
        """Set the position and reload darkfield, flatfield, amp, and base."""
        self.update_position(position)
        self.update_df()
        self.update_ff()
        self.update_amp()
        self.update_base()


def _ensure_background_path(main_window) -> None:
    """Resolve and register the selected BaSiC background-correction folder."""
    if not main_window.basic.flag:
        LOG.debug("[BaSiC] checkbox off")
        return None

    stored = getattr(main_window, "background_correction_path", None)
    if stored and os.path.isdir(stored):
        LOG.debug("[BaSiC] using stored background path: %s", stored)
        main_window.basic.add_path_basic(stored)
        return None

    analysis = find_analysis_dir(getattr(main_window, "folder", None))
    LOG.debug("[BaSiC] analysis=%s", analysis)
    if not analysis or not os.path.isdir(analysis):
        LOG.warning("[BaSiC] no analysis dir")
        return None

    combo = getattr(main_window, "bg_correct_combo", None)
    sel = None
    with contextlib.suppress(RuntimeError, AttributeError, TypeError):
        sel = combo.currentText() if combo is not None else None
    if not sel or sel.startswith("("):
        names = find_basic_folders(getattr(main_window, "folder", None))
        LOG.debug("[BaSiC] basic folders found: %s", names)
        sel = names[0] if names else None
    LOG.debug("[BaSiC] selected BaSiC folder name: %s", sel)
    if not sel:
        return None

    path = os.path.join(analysis, sel)
    main_window.basic.add_path_basic(path)
    main_window.background_correction_path = path


def _prepare_basic_correction_inputs(main_window) -> tuple[int, int]:
    """Feed BaSiC the active channels, experiment name, and time range; return that time range."""
    t_file_min, t_file_max, _t_idx_min, _t_idx_max = _current_t_range(
        main_window
    )
    main_window.basic.add_channels_valid(main_window.ids_channels)
    main_window.basic.add_exp_name(main_window.experiment_name)
    main_window.basic.add_t_range((t_file_min, t_file_max))
    _ensure_background_path(main_window)
    main_window.basic.update_all(position=main_window.current_position_number)
    return t_file_min, t_file_max


class _BasicPlan:
    """Everything needed to correct one channel, precomputed once."""

    __slots__ = (
        "channel",
        "raw_stack",
        "dark2d",
        "flat2d",
        "flat_minus_1",
        "flat_nz",
        "base_1d",
        "amp_1d",
        "out_ratio",
        "out_norf",
        "rows",
    )


def apply_basic_correction(main_window, progress_cb=None) -> None:
    """
    Compute two BaSiC-style corrections per channel over the selected t-range.
      NoRatioFlat -- the standard BaSiC correction;
        I_NRF(t) = (I(t) - D) / F - Base(t)
        -> main_window.corrected_images_noratioflat

      RatioFlat -- F attenuated per time point by Amp(t) (the "ratio flat"):
        I_RF(t)  = (I(t) - D) / (1 + (F - 1) * Amp(t)) - Base(t)
        -> main_window.corrected_images_ratioflat
    """
    if not main_window.basic.flag:
        return

    t_file_min, t_file_max, _, _ = _current_t_range(main_window)
    T = t_file_max - t_file_min + 1

    main_window.corrected_images_ratioflat = {}
    main_window.corrected_images_noratioflat = {}
    main_window._ratioflat_memmap_files = []
    main_window._noratioflat_memmap_files = []
    main_window.basic_valid_frames = {}

    n_channels_basic = len(main_window.basic.darkfield.keys())

    def _skip(channel, reason=None, *, set_valid=None):
        if reason:
            LOG.warning("[BaSiC] %s: %s", channel, reason)
        main_window.corrected_images_ratioflat[channel] = None
        main_window.corrected_images_noratioflat[channel] = None
        main_window._ratioflat_memmap_files.append(None)
        main_window._noratioflat_memmap_files.append(None)
        if set_valid is not None:
            main_window.basic_valid_frames[channel] = set_valid

    plans: list[_BasicPlan] = []

    for ci, channel in enumerate(main_window.basic.darkfield):
        raw_stack = main_window.images[channel]
        pos_num = main_window.current_position_number
        ident = getattr(main_window, f"FL_identifiers_{ci + 1}", None)

        dark = (
            main_window.basic.darkfield.get(channel)
            if main_window.basic.darkfield
            else None
        )
        flat = (
            main_window.basic.flatfield.get(channel)
            if main_window.basic.flatfield
            else None
        )
        base_info = (
            main_window.basic.base.get(channel)
            if main_window.basic.base
            else None
        )
        amp_info = (
            main_window.basic.amp.get(channel)
            if main_window.basic.amp
            else None
        )

        if raw_stack is None or ident is None:
            _skip(channel, "missing raw/identifier — skip.", set_valid=None)
            continue

        if (
            dark is None
            or flat is None
            or base_info is None
            or amp_info is None
        ):
            _skip(
                channel, "missing dark/flat/base/amp — skip.", set_valid=None
            )
            continue

        dark2d = np.asarray(dark, dtype=np.float32)
        flat2d = np.asarray(flat, dtype=np.float32)

        _, H, W = raw_stack.shape
        if dark2d.shape != (H, W) or flat2d.shape != (H, W):
            LOG.warning(
                "[BaSiC] %s: dark/flat shape mismatch; expected %s, got %s/%s. Skip.",
                channel,
                (1, H, W),
                (1, *dark2d.shape),
                (1, *flat2d.shape),
            )
            _skip(channel, None, set_valid=None)
            continue

        base_1d, has_base = _series_to_dense(
            base_info, t_file_min, t_file_max, fill_value=0.0
        )
        amp_1d, has_amp = _series_to_dense(
            amp_info, t_file_min, t_file_max, fill_value=1.0
        )

        image_present = main_window.image_present.get(channel, None)
        if image_present is None or len(image_present) != T:
            image_present = np.ones(T, dtype=bool)
        else:
            image_present = np.asarray(image_present, dtype=bool)

        valid_t = image_present & has_base & has_amp
        main_window.basic_valid_frames[channel] = valid_t

        if not np.any(valid_t):
            _skip(
                channel,
                "no overlapping valid frames between images and text entries — skip.",
            )
            continue

        missing_txt = image_present & ~(has_base & has_amp)
        if np.any(missing_txt):
            LOG.warning(
                "[BaSiC] %s: %d present image frames have no matching base/amp "
                "entry; zeroing those frames.",
                channel,
                missing_txt.sum(),
            )

        mm_ratio = (
            f"p{pos_num:04d}_{channel}_RatioFlat_corrected_t"
            f"{t_file_min:05d}-{t_file_max:05d}.dat"
        )
        mm_norf = (
            f"p{pos_num:04d}_{channel}_NoRatioFlat_corrected_t"
            f"{t_file_min:05d}-{t_file_max:05d}.dat"
        )
        path_ratio = os.path.join(main_window._memmap_dir, mm_ratio)
        path_norf = os.path.join(main_window._memmap_dir, mm_norf)
        out_ratio = np.memmap(
            path_ratio, dtype=np.float32, mode="w+", shape=(T, H, W)
        )
        out_norf = np.memmap(
            path_norf, dtype=np.float32, mode="w+", shape=(T, H, W)
        )

        plan = _BasicPlan()
        plan.channel = channel
        plan.raw_stack = raw_stack
        plan.dark2d = dark2d
        plan.flat2d = flat2d
        plan.flat_minus_1 = np.subtract(flat2d, 1.0, dtype=np.float32)
        plan.flat_nz = flat2d != 0.0
        plan.base_1d = base_1d
        plan.amp_1d = amp_1d
        plan.out_ratio = out_ratio
        plan.out_norf = out_norf
        plan.rows = np.flatnonzero(valid_t)
        plans.append(plan)

        main_window.corrected_images_ratioflat[channel] = out_ratio
        main_window.corrected_images_noratioflat[channel] = out_norf
        main_window._ratioflat_memmap_files.append(path_ratio)
        main_window._noratioflat_memmap_files.append(path_norf)

        LOG.debug("[BaSiC] %s: RatioFlat  -> %s", channel, path_ratio)
        LOG.debug("[BaSiC] %s: NoRatioFlat -> %s", channel, path_norf)

    tasks = [(plan, int(t)) for plan in plans for t in plan.rows]
    total = len(tasks)

    local = threading.local()

    def _buffers(shape):
        bufs = getattr(local, "bufs", None)
        if bufs is None or bufs[0].shape != shape:
            bufs = (
                np.empty(shape, dtype=np.float32),  # work  = I - D
                np.empty(shape, dtype=np.float32),  # norf
                np.empty(shape, dtype=np.float32),  # denom
                np.empty(shape, dtype=np.float32),  # ratio
            )
            local.bufs = bufs
        return bufs

    def _correct_one(task):
        plan, t = task
        work, norf, denom, ratio = _buffers(plan.dark2d.shape)

        base_t = np.float32(plan.base_1d[t])
        amp_t = np.float32(plan.amp_1d[t])

        np.subtract(plan.raw_stack[t], plan.dark2d, out=work)
        norf.fill(0.0)
        np.divide(work, plan.flat2d, out=norf, where=plan.flat_nz)
        np.subtract(norf, base_t, out=norf)

        np.multiply(plan.flat_minus_1, amp_t, out=denom)
        np.add(denom, 1, out=denom)

        ratio.fill(0.0)
        np.divide(work, denom, out=ratio, where=denom != 0.0)
        np.subtract(ratio, base_t, out=ratio)

        plan.out_norf[t] = norf
        plan.out_ratio[t] = ratio

    if total:
        done = 0
        step = max(1, total // 100)
        workers = min(_worker_count(main_window), total)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="basic_corr"
        ) as executor:
            for _ in executor.map(_correct_one, tasks):
                done += 1
                if callable(progress_cb) and (
                    total <= 100 or done == total or done % step == 0
                ):
                    progress_cb(done / total, f"BaSiC {done}/{total}")
                    QApplication.processEvents()

    for plan in plans:
        plan.out_ratio.flush()
        plan.out_norf.flush()

    if callable(progress_cb):
        progress_cb(
            1.0,
            (
                f"BaSiC {len(plans)}/{n_channels_basic}"
                if not total
                else f"BaSiC {total}/{total}"
            ),
        )
        QApplication.processEvents()

    LOG.info("[BaSiC] Correction complete.")


def _worker_count(main_window=None) -> int:
    """Number of worker threads for BaSiC correction"""
    override = getattr(main_window, "basic_max_workers", None)
    if override:
        try:
            return max(1, min(8, int(override)))
        except (TypeError, ValueError):
            pass
    return _fl_worker_count(main_window)


def _series_to_dense(
    series: dict[str, np.ndarray], t_min: int, t_max: int, fill_value: float
) -> tuple[np.ndarray, np.ndarray]:
    """Expand sparse text-file values onto the dense image timeline [t_min, t_max]."""
    T = t_max - t_min + 1
    dense = np.full(T, fill_value, dtype=np.float32)
    has_value = np.zeros(T, dtype=bool)

    if series is None:
        return dense, has_value

    times = np.asarray(series["times"], dtype=np.int32)
    values = np.asarray(series["values"], dtype=np.float32)

    idx = times - t_min
    ok = (idx >= 0) & (idx < T)

    idx = idx[ok]
    dense[idx] = values[ok]
    has_value[idx] = True

    return dense, has_value
