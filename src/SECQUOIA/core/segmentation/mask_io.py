"""Disk I/O for segmentation mask files: loading, saving, and path resolution."""

import concurrent.futures
import glob
import logging
import os
import re

import imageio.v2 as imageio
import numpy as np
from skimage.measure import label

from SECQUOIA.utils.io_reload import read_with_reload
from SECQUOIA.utils.timing import current_t_range as _current_t_range
from SECQUOIA.utils.timing import (
    filename_t_file_number as _filename_t_file_number,
)

LOG = logging.getLogger(__name__)


def _mask_worker_count(max_workers: int | None = None) -> int:
    """Return the number of reader threads for mask loading."""
    if max_workers:
        try:
            return max(1, min(8, int(max_workers)))
        except (TypeError, ValueError):
            pass

    n = None
    _process_cpu_count = getattr(os, "process_cpu_count", None)
    if _process_cpu_count is not None:
        n = _process_cpu_count()
    else:
        _sched_getaffinity = getattr(os, "sched_getaffinity", None)
        if _sched_getaffinity is not None:
            n = len(_sched_getaffinity(0))
    if not n:
        n = os.cpu_count()
    if not n:
        n = 1
    return max(1, min(8, max(2, n - 1)))


def mask_reader_for_format(image_format: str):
    """Return (read_fn, is_bgr) for the given image format.

    tif/tiff       -> tifffile (fast, keeps 16-bit)
    anything else  -> imageio
    fallback       -> imageio, if tifffile is unavailable
    """
    fmt = str(image_format or "").lower().lstrip(".")

    if fmt in ("tif", "tiff"):
        try:
            import tifffile

            return tifffile.imread, False
        except ImportError:
            pass
    return imageio.imread, False


def _mask_to_2d(arr: np.ndarray, is_bgr: bool) -> np.ndarray:
    """Collapse a possibly-multichannel image to a single 2-D plane."""
    if arr.ndim <= 2:
        return arr
    if is_bgr and arr.shape[-1] >= 3:
        return arr[..., 2]
    return arr[..., 0]


def _n_distinct_values(img: np.ndarray) -> int:
    """Exact count of distinct values, without np.unique's full sort.

    For non-negative integer images this is O(n + max) via bincount instead of
    O(n log n). Falls back to np.unique for anything else.
    """
    if img.dtype.kind not in ("u", "i"):
        return int(np.unique(img).size)

    flat = np.ravel(img)
    if flat.size == 0:
        return 0
    mn = int(flat.min())
    if mn < 0:
        return int(np.unique(img).size)
    mx = int(flat.max())
    if mx == 0:
        return 1
    if mx > 1 << 24:
        return int(np.unique(img).size)
    return int(np.count_nonzero(np.bincount(flat, minlength=mx + 1)))


def _plan_single_mask_stack(
    path_seg: str,
    pos_selection: str,
    image_format: str,
    dtype: np.dtype,
    recursive: bool,
    t_file_min: int | None,
    t_file_max: int | None,
    read_fn,
    is_bgr: bool,
) -> tuple[np.ndarray | None, list]:
    """Allocate the output stack and build the per frame read tasks."""
    if not path_seg:
        LOG.warning("Segmentation path not set.")
        return None, []

    base = os.path.join(path_seg, os.path.basename(pos_selection))
    pattern = f"**/*.{image_format}" if recursive else f"*.{image_format}"
    search_glob = os.path.join(base, pattern)

    image_files = sorted(glob.glob(search_glob, recursive=recursive))

    use_range = t_file_min is not None and t_file_max is not None

    if use_range:
        file_by_t: dict[int, str] = {}
        for fp in image_files:
            n = _filename_t_file_number(fp)
            if n is not None and t_file_min <= n <= t_file_max:
                file_by_t[n] = fp
        image_files = [file_by_t[n] for n in sorted(file_by_t)]
    else:
        file_by_t = {}

    if not image_files:
        LOG.warning(
            "[%s] No segmentation masks found for position '%s'%s.",
            path_seg,
            os.path.basename(pos_selection),
            " in selected t range" if use_range else "",
        )
        return None, []

    try:
        probe = _mask_to_2d(read_with_reload(read_fn, image_files[0]), is_bgr)
    except (OSError, ValueError, RuntimeError) as e:
        LOG.error(
            "[%s] Giving up reading %s after reloading (%r); skipping this "
            "segmentation folder for this position.",
            path_seg,
            image_files[0],
            e,
        )
        return None, []
    H, W = probe.shape[:2]

    if use_range:
        T = max(0, t_file_max - t_file_min + 1)
        stack = np.zeros((T, H, W), dtype=dtype)
        tasks = []
        for n in sorted(file_by_t):
            idx = n - t_file_min
            if 0 <= idx < T:
                tasks.append((stack, idx, file_by_t[n]))
        LOG.info(
            "[%s] Segmentation masks loaded into full T=%d frames.",
            path_seg,
            T,
        )
    else:
        T = len(image_files)
        stack = np.zeros((T, H, W), dtype=dtype)
        tasks = [(stack, i, fp) for i, fp in enumerate(image_files)]
        LOG.info("[%s] Segmentation masks loaded: %d images.", path_seg, T)

    return stack, tasks


def _read_mask_into(task, read_fn, is_bgr: bool, dtype: np.dtype) -> None:
    """Decode one mask frame and write it into its own row of the stack."""
    stack, row, fp = task
    try:
        img = _mask_to_2d(read_with_reload(read_fn, fp), is_bgr)
    except (OSError, ValueError, RuntimeError) as e:
        LOG.error(
            "Giving up on mask frame after reloading: %s (%r). Leaving this "
            "frame's mask empty.",
            fp,
            e,
        )
        return

    if _n_distinct_values(img) == 2:
        img = label(img > 0).astype(dtype, copy=False)
    elif img.dtype != dtype:
        img = img.astype(dtype, copy=False)

    stack[row] = img


def load_masks(
    path_seg: str | list | tuple | dict,
    pos_selection: str,
    image_format: str,
    dtype: np.dtype = np.uint16,
    recursive: bool = False,
    t_file_min: int | None = None,
    t_file_max: int | None = None,
    max_workers: int | None = None,
) -> np.ndarray | list | dict | None:
    """Load segmentation masks for a position, optionally over a t-file range.

    All mask folders and all frames are decoded by a single bounded thread pool.
    The returned structure (array / list / dict), the frame order and the
    16-bit label values are identical to the serial version.
    """
    read_fn, is_bgr = mask_reader_for_format(image_format)

    if isinstance(path_seg, (str | bytes)):
        paths, kind = [path_seg], "single"
    elif isinstance(path_seg, (list | tuple)):
        paths, kind = list(path_seg), "list"
    elif isinstance(path_seg, dict):
        paths, kind = list(path_seg.values()), "dict"
    else:
        LOG.error(
            "Unsupported type for path_seg. Provide a str, a list/tuple of str, "
            "or a dict[str, str]."
        )
        return None

    stacks: list = []
    all_tasks: list = []
    for p in paths:
        stack, tasks = _plan_single_mask_stack(
            p,
            pos_selection,
            image_format=image_format,
            dtype=dtype,
            recursive=recursive,
            t_file_min=t_file_min,
            t_file_max=t_file_max,
            read_fn=read_fn,
            is_bgr=is_bgr,
        )
        stacks.append(stack)
        all_tasks.extend(tasks)

    if all_tasks:
        workers = min(_mask_worker_count(max_workers), len(all_tasks))
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="mask_load"
        ) as ex:
            list(
                ex.map(
                    lambda t: _read_mask_into(t, read_fn, is_bgr, dtype),
                    all_tasks,
                )
            )

    if kind == "single":
        return stacks[0]
    if kind == "list":
        return stacks
    return dict(zip(path_seg.keys(), stacks, strict=False))


def t_file_from_idx(main_window, t_idx: int) -> int:
    """Map a rebased time index back to the original on disk t-file number."""
    t_file_min, t_file_max, t_idx_min, t_idx_max = _current_t_range(
        main_window
    )
    return int(t_file_min + (t_idx - t_idx_min))


def position_name(main_window) -> str:
    """Return the position name derived from main_window.position_selection (e.g. '250615MA40_p0007')."""
    return (
        os.path.basename(
            os.path.normpath(getattr(main_window, "position_selection", ""))
        )
        or "pos"
    )


def position_dir_for_mask(mask_dir: str, pos_name: str) -> str:
    """Return the subfolder of `mask_dir` holding the given position."""
    return os.path.join(mask_dir, pos_name)


def labels2d_from_container(
    labels_container, mask_idx_1based: int, t_idx: int
) -> np.ndarray | None:
    """Extract a 2D labels slice for (mask_idx, t_idx) from a container."""
    if labels_container is None:
        return None

    try:
        if isinstance(labels_container, np.ndarray):
            arr = labels_container
            if arr.ndim == 2:
                return arr if t_idx == 0 else None
            if arr.ndim >= 3 and 0 <= t_idx < arr.shape[0]:
                return arr[t_idx]
            return None

        if isinstance(labels_container, (list | tuple)):
            k0 = int(mask_idx_1based) - 1
            if 0 <= k0 < len(labels_container):
                arr = np.asarray(labels_container[k0])
                if arr.ndim == 2:
                    return arr if t_idx == 0 else None
                if arr.ndim >= 3 and 0 <= t_idx < arr.shape[0]:
                    return arr[t_idx]
            return None

        if isinstance(labels_container, dict):
            key = str(mask_idx_1based)
            if key in labels_container:
                arr = np.asarray(labels_container[key])
            else:
                keys = sorted(labels_container.keys())
                k0 = int(mask_idx_1based) - 1
                if 0 <= k0 < len(keys):
                    arr = np.asarray(labels_container[keys[k0]])
                else:
                    return None
            if arr.ndim == 2:
                return arr if t_idx == 0 else None
            if arr.ndim >= 3 and 0 <= t_idx < arr.shape[0]:
                return arr[t_idx]
            return None
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return None

    return None


def find_exact_mask(
    dirpath: str,
    pos_name: str,
    t_file: int,
    image_format: str,
) -> str | None:
    """Locate the exact original mask file (of ``image_format``) matching pos_name and t_file.

    Matches names that:
    - start with '<pos_name>_'
    - contain '_tNNNNN_'
    """
    patt_mid = f"_t{t_file:05d}_"
    try:
        for name in os.listdir(dirpath):
            if not name.lower().endswith(f".{image_format}"):
                continue
            if not name.startswith(pos_name + "_"):
                continue
            if patt_mid in name:
                return os.path.join(dirpath, name)
    except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
        pass
    return None


def infer_w_token(dirpath: str) -> str | None:
    """Infer a 'wNN' token from any file in the directory, e.g. 'w04'."""
    try:
        for name in os.listdir(dirpath):
            m = re.search(r"_w(\d+)", name)
            if m:
                return f"w{m.group(1)}"
    except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
        pass
    return None


def fallback_mask_filename(
    dirpath: str,
    pos_name: str,
    t_file: int,
    image_format: str,
    z_token: str = "z001",
    m_token: str = "m00",
    default_w: str = "w01",
) -> str:
    """Build a filename when no exact original match exists: <pos_name>_tNNNNN_z###_wNN_m## _mask."""
    wtok = infer_w_token(dirpath) or default_w
    return os.path.join(
        dirpath,
        f"{pos_name}_t{t_file:05d}_{z_token}_{wtok}_{m_token}_mask.{image_format}",
    )


def save_masks_incremental(main_window) -> None:
    """Save only corrected mask slices into the copied segmentation folders under Analysis."""
    seg_dirs = getattr(main_window, "segmentation_paths", None)
    if not seg_dirs:
        LOG.warning(
            "No segmentation paths set. Did you copy them into Analysis?"
        )
        return

    corrected = sorted(
        set(getattr(main_window, "_corrected_slices", []) or [])
    )
    if not corrected:
        LOG.info("Nothing to save (no corrections tracked).")
        return

    pos_sel = getattr(main_window, "position_selection", None)
    if not pos_sel:
        LOG.warning("position_selection not set; load data first.")
        return

    pos = position_name(main_window)
    labels_container = getattr(main_window, "labels", None)
    n_masks = int(getattr(main_window, "n_masks", len(seg_dirs) or 1))

    saved_ok = set()
    saved = 0
    errors = 0

    for mask_idx, t_idx in corrected:
        # Validate mask index
        if mask_idx < 1 or mask_idx > n_masks:
            LOG.warning(
                "Skip invalid mask index %s (n_masks=%s).", mask_idx, n_masks
            )
            continue

        mask_root = seg_dirs[mask_idx - 1]
        if not mask_root:
            LOG.warning("Skip: no segmentation dir for mask %s.", mask_idx)
            continue

        pos_dir = position_dir_for_mask(mask_root, pos)
        if not os.path.isdir(pos_dir):
            try:
                os.makedirs(pos_dir, exist_ok=True)
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                LOG.error("Cannot create position dir '%s': %s", pos_dir, e)
                continue

        # Labels slice
        labels2d = labels2d_from_container(labels_container, mask_idx, t_idx)
        if labels2d is None:
            LOG.warning("Skip: no labels for mask %s, t=%s.", mask_idx, t_idx)
            continue

        # Map to t-file number
        try:
            t_file = t_file_from_idx(main_window, t_idx)
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            LOG.warning("t-file mapping failed for t_idx=%s: %s", t_idx, e)
            continue

        image_format = str(getattr(main_window, "image_format", "png")).lstrip(
            "."
        )
        out_path = find_exact_mask(pos_dir, pos, t_file, image_format)
        if out_path is None:
            out_path = fallback_mask_filename(
                pos_dir, pos, t_file, image_format
            )

        # Write image as uint16 labels
        try:
            arr = np.asarray(labels2d)
            if arr.dtype != np.uint16:
                arr = arr.astype(np.uint16, copy=False)
            read_with_reload(
                lambda p, _arr=arr: imageio.imwrite(p, _arr), out_path
            )
            LOG.debug(
                "Wrote corrected mask %s, t_idx=%s (t_file=%s) -> %s",
                mask_idx,
                t_idx,
                t_file,
                out_path,
            )
            saved_ok.add((mask_idx, t_idx))
            saved += 1
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            LOG.error(
                "Error writing mask %s, t_idx=%s: %s", mask_idx, t_idx, e
            )
            errors += 1

    if hasattr(main_window, "_corrected_slices"):
        main_window._corrected_slices = set(corrected) - saved_ok

    if hasattr(main_window, "status_label") and main_window.status_label:
        main_window.status_label.setText(
            f"Status: Saved {saved} corrected slice(s), {errors} error(s)."
        )
    LOG.info(
        "Done. Saved=%d, Errors=%d, Remaining corrected=%d",
        saved,
        errors,
        len(getattr(main_window, "_corrected_slices", [])),
    )
