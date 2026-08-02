"""Lifecycle management for the temporary numpy.memmap image stacks used during loading."""

from __future__ import annotations

import contextlib
import gc
import glob
import logging
import os
import shutil
import stat
import tempfile
import time

import numpy as np

try:
    import fcntl  # POSIX (macOS/Linux)
except ImportError:
    fcntl = None

try:
    import msvcrt  # Windows
except ImportError:
    msvcrt = None

LOG = logging.getLogger(__name__)

__all__ = [
    "cleanup_memmaps",
    "acquire_dir_lock",
    "release_dir_lock",
    "reap_stale_memmap_dirs",
]

_LOCK_FILENAME = ".owner.lock"
_MEMMAP_DIR_GLOB = "SECQUOIA_memmaps_*"


def acquire_dir_lock(dir_path: str):
    """Take a non-blocking exclusive lock proving this process owns ``dir_path``.

    The OS releases the lock the instant the owning process's file handles
    close, including on a crash — unlike a PID or marker file, it can never
    outlive the process that created it. Returns the open file object
    holding the lock (keep it referenced for as long as ownership should
    last), or None if another live process already holds it.
    """
    lock_path = os.path.join(dir_path, _LOCK_FILENAME)
    try:
        # Deliberately left open on success: the caller holds this handle for
        # as long as it owns the lock, so a `with` block would close it here.
        fh = open(lock_path, "a+b")  # noqa: SIM115
    except OSError:
        return None

    try:
        if fcntl is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        elif msvcrt is not None:
            fh.seek(0, os.SEEK_END)
            if fh.tell() == 0:
                fh.write(b"\0")
                fh.flush()
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        return fh
    except OSError:
        fh.close()
        return None


def release_dir_lock(lock_file) -> None:
    """Release and close a lock previously returned by ``acquire_dir_lock``."""
    if lock_file is None:
        return
    with contextlib.suppress(OSError, ValueError):
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        elif msvcrt is not None:
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
    with contextlib.suppress(OSError, ValueError):
        lock_file.close()


def reap_stale_memmap_dirs() -> None:
    """Delete leftover memmap temp dirs from runs that never shut down cleanly."""
    pattern = os.path.join(tempfile.gettempdir(), _MEMMAP_DIR_GLOB)
    for path in glob.glob(pattern):
        if not os.path.isdir(path):
            continue

        lock_file = acquire_dir_lock(path)
        if lock_file is None:
            continue

        release_dir_lock(lock_file)
        if _rmtree_retry(path, retries=3, sleep_s=0.2):
            LOG.info("[memmap cleanup] reaped stale temp dir: %s", path)
        else:
            LOG.warning(
                "[memmap cleanup] failed to reap stale temp dir: %s", path
            )


def _detach_viewer_images(viewer) -> None:
    """Drop napari Image layer data and remove those layers from a viewer."""
    if viewer is None:
        return
    with contextlib.suppress(AttributeError, RuntimeError, TypeError):
        for lyr in list(viewer.layers):
            if getattr(lyr, "__class__", None).__name__ == "Image":
                with contextlib.suppress(
                    AttributeError, RuntimeError, TypeError
                ):
                    lyr.data = None
                with contextlib.suppress(
                    AttributeError, RuntimeError, TypeError
                ):
                    viewer.layers.remove(lyr)


def _close_memmap(arr) -> None:
    """Flush and close one numpy.memmap array, if it is one."""
    if isinstance(arr, np.memmap):
        with contextlib.suppress(OSError, ValueError):
            arr.flush()
        mm = getattr(arr, "_mmap", None)
        if mm is not None:
            with contextlib.suppress(OSError, ValueError):
                mm.close()


def _flush_close_attr(main_window, attr_name: str) -> None:
    """Close every memmap held in a dict/list attribute on main_window, then clear it."""
    obj = getattr(main_window, attr_name, None)

    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            _close_memmap(v)
            obj[k] = None
        obj.clear()
        setattr(main_window, attr_name, {})

    elif isinstance(obj, list):
        for i, v in enumerate(list(obj)):
            _close_memmap(v)
            obj[i] = None
        obj.clear()
        setattr(main_window, attr_name, [])


def _delete_file_list(
    main_window, attr_name: str, retries: int, sleep_s: float
) -> None:
    """Delete every file path tracked in a list attribute on main_window, then clear it."""
    files = getattr(main_window, attr_name, None)
    if not isinstance(files, list):
        return

    for p in list(files):
        if not p:
            continue

        for _ in range(retries):
            try:
                if os.path.isfile(p):
                    with contextlib.suppress(OSError):
                        os.chmod(p, stat.S_IWRITE)
                    os.remove(p)
                break
            except FileNotFoundError:
                break
            except PermissionError:
                time.sleep(sleep_s)
            except OSError:
                time.sleep(sleep_s)

    files.clear()
    setattr(main_window, attr_name, [])


def _rmtree_retry(path: str, retries: int, sleep_s: float) -> bool:
    """Remove a directory tree, retrying on transient (typically Windows) file locks."""
    if not path or not os.path.isdir(path):
        return True

    def _make_writable(p):
        with contextlib.suppress(OSError):
            os.chmod(p, stat.S_IWRITE)

    def _onexc(func, p, exc_info):
        _make_writable(p)
        with contextlib.suppress(FileNotFoundError, PermissionError, OSError):
            func(p)

    for _ in range(retries):
        try:
            try:
                shutil.rmtree(path, onexc=_onexc)
            except TypeError:
                shutil.rmtree(path, onerror=_onexc)
            return True
        except FileNotFoundError:
            return True
        except PermissionError:
            time.sleep(sleep_s)
        except OSError:
            time.sleep(sleep_s)

    return False


def cleanup_memmaps(
    main_window,
    *,
    remove_viewer_image_layers: bool = True,
    retries: int = 8,
    sleep_s: float = 0.2,
) -> None:
    """Release and delete all temporary numpy.memmap stacks.

    Optionally detaches napari image layers, flushes and closes memmaps held by
    ``main_window`` containers, deletes the tracked memmap files, and removes the
    memmap directory.
    """
    if remove_viewer_image_layers:
        _detach_viewer_images(getattr(main_window, "viewer_1", None))
        _detach_viewer_images(getattr(main_window, "viewer_2", None))

    _flush_close_attr(main_window, "images")
    _flush_close_attr(main_window, "_memmap_arrays")
    _flush_close_attr(main_window, "corrected_images")
    _flush_close_attr(main_window, "corrected_images_ratioflat")
    _flush_close_attr(main_window, "corrected_images_noratioflat")

    # Force ref drop
    gc.collect()

    _delete_file_list(main_window, "_memmap_files", retries, sleep_s)
    _delete_file_list(main_window, "_corrected_memmap_files", retries, sleep_s)
    _delete_file_list(main_window, "_ratioflat_memmap_files", retries, sleep_s)
    _delete_file_list(
        main_window, "_noratioflat_memmap_files", retries, sleep_s
    )

    release_dir_lock(getattr(main_window, "_memmap_lock", None))
    main_window._memmap_lock = None

    # Remove directory last
    mm_dir = getattr(main_window, "_memmap_dir", "")
    if mm_dir and os.path.isdir(mm_dir):
        ok = _rmtree_retry(mm_dir, retries, sleep_s)
        if ok:
            main_window._memmap_dir = ""
        else:
            LOG.warning(
                "[memmap cleanup] could not remove dir after retries: %s",
                mm_dir,
            )
