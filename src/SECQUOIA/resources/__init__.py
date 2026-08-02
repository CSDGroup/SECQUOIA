"""SECQUOIA's bundled asset library (icons, splash art) and its loaders"""

from functools import cache
from importlib.resources import as_file, files

from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon, QPixmap


@cache
def app_icon() -> QIcon:
    """Return the SECQUOIA application icon."""
    with as_file(files(__package__) / "icons" / "secquoia.png") as path:
        pix = QPixmap(str(path))
    if pix.isNull():
        return QIcon()

    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(
            pix.scaled(
                size,
                size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
    return icon


def splash_pixmap() -> QPixmap | None:
    """Return the splash banner pixmap, or None if it isn't available."""
    with as_file(files(__package__) / "images" / "splash.png") as path:
        pix = QPixmap(str(path))
    return None if pix.isNull() else pix
