"""Tell the user at startup when a newer SECQUOIA version is tagged on GitHub."""

from __future__ import annotations

import http.client
import json
import logging
import threading
import urllib.request

LOG = logging.getLogger(__name__)

_TAGS_URL = "https://api.github.com/repos/CSDGroup/SECQUOIA/tags"
_TIMEOUT_S = 3


def newest_version(tag_names, current: str) -> str | None:
    """The highest tag version above `current`, or None when there is none."""
    from packaging.version import InvalidVersion, Version

    try:
        installed = Version(current)
    except InvalidVersion:
        return None
    versions = []
    for name in tag_names:
        try:
            versions.append(Version(str(name).lstrip("v")))
        except InvalidVersion:
            continue
    newest = max(versions, default=None)
    return str(newest) if newest is not None and newest > installed else None


def _fetch_tag_names() -> list[str]:
    """Names of the repository's tags on GitHub."""
    with urllib.request.urlopen(_TAGS_URL, timeout=_TIMEOUT_S) as r:
        return [t["name"] for t in json.load(r)]


def check_for_update(parent) -> None:
    """Check GitHub in the background and show a notice if SECQUOIA is outdated."""
    from qtpy.QtCore import QTimer

    from SECQUOIA import __version__

    result: dict[str, str | None] = {}

    def _worker():
        try:
            result["newer"] = newest_version(_fetch_tag_names(), __version__)
        except (
            OSError,
            http.client.HTTPException,
            ValueError,
            KeyError,
            TypeError,
            ImportError,
        ) as e:
            LOG.info("[update check] skipped: %s", e)
        finally:
            result.setdefault("newer", None)

    def _when_done():
        if "newer" not in result:
            QTimer.singleShot(200, _when_done)
            return
        if result["newer"]:
            try:
                _show_notice(parent, result["newer"], __version__)
            except (RuntimeError, AttributeError, TypeError, ImportError) as e:
                LOG.warning("[update check] could not show notice: %s", e)

    threading.Thread(target=_worker, name="update_check", daemon=True).start()
    QTimer.singleShot(200, _when_done)


def _show_notice(parent, newer: str, installed: str) -> None:
    """Popup saying which version is available, with a link to the update steps."""
    from qtpy.QtCore import QUrl
    from qtpy.QtGui import QDesktopServices
    from qtpy.QtWidgets import QMessageBox

    from SECQUOIA.config import LINKS, TOOLTIPSTEXT
    from SECQUOIA.gui.common.messages import apply_dialog_platform_style

    dlg = QMessageBox(parent)
    dlg.setWindowTitle("Update available")
    dlg.setIcon(QMessageBox.Information)
    dlg.setText(f"SECQUOIA {newer} is available.")
    dlg.setInformativeText(
        f"You are currently using version {installed}. Please update to "
        "the newest version from GitHub."
    )
    btn_how = dlg.addButton("Open GitHub", QMessageBox.AcceptRole)
    btn_how.setToolTip(TOOLTIPSTEXT.UPDATE_OPEN_GITHUB)
    btn_later = dlg.addButton("Later", QMessageBox.RejectRole)
    btn_later.setToolTip(TOOLTIPSTEXT.UPDATE_LATER)
    apply_dialog_platform_style(dlg)
    dlg.exec()

    if dlg.clickedButton() is btn_how:
        QDesktopServices.openUrl(QUrl(LINKS.GITHUB_UPDATING))
