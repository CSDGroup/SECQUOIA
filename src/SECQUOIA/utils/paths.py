"""Shared filesystem path helpers for the per project Analysis output folder."""

from __future__ import annotations

import os

__all__ = ["project_analysis_dir"]


def project_analysis_dir(
    main_window,
    experiment_root: str | None = None,
    tracking_format: str | None = None,
) -> str:
    """Return ``<experiment_root>/Analysis/SECQUOIA_files_<tracking_format>/<project_name>``."""
    if experiment_root is None:
        experiment_root = getattr(main_window, "folder", None)
    if tracking_format is None:
        tracking_format = getattr(main_window, "tracking_format", "unknown")
    project = (getattr(main_window, "project_name", "") or "Project_1").strip()
    return os.path.join(
        experiment_root,
        "Analysis",
        f"SECQUOIA_files_{tracking_format}",
        project,
    )
