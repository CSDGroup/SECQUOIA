"""Discovery of files and folders inside an experiment directory."""

from __future__ import annotations

import contextlib
import os
import re

__all__ = [
    "detect_time_max",
    "detect_unique_w_channels",
    "find_analysis_dir",
    "find_basic_folders",
    "find_example_position_folder",
    "find_images_csvs",
]


_CHANNEL_RE = re.compile(r"_w(\d+)")
_TIMEPOINT_RE = re.compile(r"_t(\d+)_")
_ANALYSIS_DIRNAME = "analysis"
_BASIC_PREFIX = "BaSiC"


def find_analysis_dir(folder: str | None) -> str | None:
    """Return the ``Analysis`` subdirectory of an experiment folder."""
    if not folder or not os.path.isdir(folder):
        return None

    with contextlib.suppress(OSError), os.scandir(folder) as entries:
        for entry in entries:
            if entry.is_dir() and entry.name.lower() == _ANALYSIS_DIRNAME:
                return entry.path
    return None


def find_basic_folders(folder: str | None) -> list[str]:
    """Return the names of BaSiC correction folders under ``Analysis``."""
    analysis = find_analysis_dir(folder)
    if not analysis:
        return []

    names = []
    with contextlib.suppress(OSError), os.scandir(analysis) as entries:
        for entry in entries:
            if entry.is_dir() and entry.name.startswith(_BASIC_PREFIX):
                names.append(entry.name)
    return sorted(names)


def find_images_csvs(folder: str | None) -> list[str]:
    """Return the names of ``images*.csv`` files in the experiment folder."""
    if not folder or not os.path.isdir(folder):
        return []

    names = []
    with contextlib.suppress(OSError), os.scandir(folder) as entries:
        for entry in entries:
            lowered = entry.name.lower()
            if (
                entry.is_file()
                and lowered.startswith("images")
                and lowered.endswith(".csv")
            ):
                names.append(entry.name)
    return sorted(names)


def find_example_position_folder(
    folder: str | None,
    experiment_name: str | None,
    position_folders: list[str] | None = None,
) -> str | None:
    """Pick one position folder to probe for channels and time points."""
    if not folder or not experiment_name:
        return None

    preferred = [
        os.path.join(folder, f"{experiment_name}_p0001"),
        os.path.join(folder, f"{experiment_name}_p0002"),
    ]
    for path in preferred:
        if os.path.isdir(path):
            return path

    return (position_folders or [None])[0] if position_folders else None


def detect_unique_w_channels(
    position_folder: str | None, n_channels_fallback: int = 1
) -> list[str]:
    """Extract the distinct ``wNN`` channel identifiers from a position folder."""
    channels = set()
    if position_folder:
        with contextlib.suppress(OSError):
            for name in os.listdir(position_folder):
                match = _CHANNEL_RE.search(name)
                if match:
                    channels.add("w" + match.group(1))

    if channels:
        return sorted(
            channels, key=lambda s: int(re.search(r"\d+", s).group(0))
        )

    count = int(n_channels_fallback or 0)
    return [f"w{i + 1:02d}" for i in range(count)]


def detect_time_max(position_folder: str | None) -> int:
    """Determine the highest time point index present in a position folder."""
    if not position_folder:
        return 1

    t_max = 1
    with contextlib.suppress(OSError):
        for name in os.listdir(position_folder):
            match = _TIMEPOINT_RE.search(name)
            if match:
                t_max = max(t_max, int(match.group(1)))
    return max(1, t_max)
