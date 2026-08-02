"""Image & Movie Exporter: GIF/Movie/TIFF/PNG export window and canvas.

``ImageMovieExporter`` itself only owns ``__init__``; every other method is
contributed by one of the classes below, each covering one concern
(panel/grid layout, rendering, playback, the file-writing exporters, lineage
resolution, the B/W inspector dialog, sidebar wiring, and presets).
"""

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
