"""Image & Movie Exporter: GIF/Movie/TIF/PNG export window and canvas."""

from __future__ import annotations

from SECQUOIA.gui.exporter.bw_inspector import _BWInspector
from SECQUOIA.gui.exporter.common import _Common
from SECQUOIA.gui.exporter.export_jobs import _ExportJobs
from SECQUOIA.gui.exporter.lineage import _Lineage
from SECQUOIA.gui.exporter.panels import _Panels
from SECQUOIA.gui.exporter.playback import _Playback
from SECQUOIA.gui.exporter.presets import _Presets
from SECQUOIA.gui.exporter.rendering import _Rendering
from SECQUOIA.gui.exporter.sidebar import _Sidebar
from SECQUOIA.gui.exporter.window import _Window

__all__ = [
    "ImageMovieExporter",
]


class ImageMovieExporter(
    _Common,
    _Panels,
    _Window,
    _Playback,
    _Rendering,
    _ExportJobs,
    _Lineage,
    _BWInspector,
    _Sidebar,
    _Presets,
):
    """Image and Movie Exporter tool."""

    def __init__(self, main_window):
        """Store the host application's main window reference."""
        self.main_window = main_window
