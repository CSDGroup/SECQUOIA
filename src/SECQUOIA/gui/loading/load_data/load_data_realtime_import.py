"""TTT XML comment loading and real-time (ms) CSV import for the Load Data dialog."""

from __future__ import annotations

import logging
import os
import re

import pandas as pd

from SECQUOIA.core.tatexp_xml import (
    extract_channel_comments_from_xml,
    extract_position_comments_from_xml,
)

LOG = logging.getLogger(__name__)

__all__ = [
    "_maybe_load_ttt_channel_comments",
    "_maybe_load_ttt_position_comments",
    "_clear_import_realtime",
    "_maybe_import_realtime",
]


def _maybe_load_ttt_channel_comments(main_window):
    """If tTt and xml_path present, parse channel comments; else {}."""
    cf = {}
    if getattr(main_window, "tracking_format", "") == "tTt":
        cf = extract_channel_comments_from_xml(
            getattr(main_window, "xml_path", None)
        )
    main_window.channel_comment_map = cf


def _maybe_load_ttt_position_comments(main_window):
    """If tTt and xml_path present, parse position comments; else {}."""
    pf = {}
    if getattr(main_window, "tracking_format", "") == "tTt":
        pf = extract_position_comments_from_xml(
            getattr(main_window, "xml_path", None)
        )
    main_window.position_comment_map = pf


def _clear_import_realtime(main_window):
    """Drop any previously imported real-time data and its cached lookup."""
    main_window.import_rt_df = None
    main_window._rt_wide = None


def _maybe_import_realtime(main_window):
    """Import measurement time data from the selected images CSV when enabled."""
    if not getattr(main_window, "use_import_rt", False):
        _clear_import_realtime(main_window)
        return

    path = getattr(main_window, "import_rt_path", None)
    if not (path and os.path.isfile(path)):
        _clear_import_realtime(main_window)
        return

    try:
        df = pd.read_csv(
            path,
            sep=";",
            engine="python",
            encoding="utf-8-sig",
            on_bad_lines="skip",
        )
    except (
        pd.errors.ParserError,
        OSError,
        ValueError,
        UnicodeDecodeError,
    ) as e:
        LOG.warning("[Import Real time] read failed: %s", e)
        _clear_import_realtime(main_window)
        return

    def _norm_header(s: str) -> str:
        """Normalize a CSV header for column matching."""
        return re.sub(r"[^a-z0-9]+", "", str(s).lower())

    nmap = {_norm_header(c): c for c in df.columns}

    tcol = nmap.get("measurementtimems")
    icol = nmap.get("imagefile")
    if not (tcol and icol):
        _clear_import_realtime(main_window)
        return

    out = df[[tcol, icol]].copy()
    out.columns = ["Measurement Time (ms)", "Image File"]
    main_window.import_rt_df = out
    main_window._rt_wide = None
