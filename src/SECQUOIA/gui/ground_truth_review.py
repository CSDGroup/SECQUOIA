"""Moving through the points to judge, and the X key (ground truth developer mode)."""

from __future__ import annotations

import contextlib
import logging

from SECQUOIA.core.ground_truth import next_open_point, open_masks
from SECQUOIA.core.segmentation.mask_selection import get_mask_indices
from SECQUOIA.gui.outlier.close_mask_review import _jump_to_time
from SECQUOIA.gui.outlier.navigation import go_to_ident
from SECQUOIA.gui.outlier.outlier_list import _show_info_dialog
from SECQUOIA.utils.plotting import update_plot

LOG = logging.getLogger(__name__)

__all__ = ["accept_current_point", "advance_if_complete"]


def _show_point(main_window, ident: str, track: int, t: int) -> bool:
    """Show the point ``(ident, track, t)`` in both viewers."""
    if str(main_window.ident) != ident and not go_to_ident(main_window, ident):
        return False
    if main_window.current_TrackNumber_plot != track:
        main_window.current_TrackNumber_plot = track
        update_plot(main_window)
    _jump_to_time(main_window, t)
    return True


def advance_if_complete(main_window) -> None:
    """Jump to the next open point once every mask of the current one is judged."""
    if open_masks(main_window, get_mask_indices(main_window)) != []:
        return
    point = next_open_point(main_window)
    if point is None:
        LOG.info("[ground truth] All points checked.")
        with contextlib.suppress(RuntimeError, AttributeError, TypeError):
            _show_info_dialog(
                main_window, "Ground truth", "All points checked"
            )
        return
    try:
        shown = _show_point(main_window, *point)
    except (
        RuntimeError,
        AttributeError,
        TypeError,
        ValueError,
        KeyError,
    ) as e:
        LOG.warning("[ground truth] Could not show the point %s: %s", point, e)
        return
    if not shown:
        LOG.warning("[ground truth] Could not show the point %s.", point)


def accept_current_point(main_window) -> None:
    """The X key: accept every open mask of the current point as it stands."""
    bars = getattr(main_window, "gt_bars", None)
    if bars:
        bars[0].accept_all()
