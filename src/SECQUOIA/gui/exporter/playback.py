"""Play/pause/timer-driven preview playback."""

import logging
from contextlib import suppress

from qtpy.QtCore import (
    QSize,
)

from SECQUOIA.gui.common.ui_utils import is_widget_alive

LOG = logging.getLogger(__name__)


class _Playback:
    """Play/pause/timer-driven preview playback."""

    def _set_play_button_state(self, main_window, playing: bool):
        """Update play/pause button icon, text and style."""
        btn = getattr(main_window, "play_btn", None)
        if not btn:
            return
        icn = (
            self._qta_icon("fa5s.pause", "mdi.pause", "fa.pause")
            if playing
            else self._qta_icon("fa5s.play", "mdi.play", "fa.play")
        )
        if icn:
            btn.setIcon(icn)
            with suppress(Exception):
                btn.setIconSize(QSize(16, 16))
            btn.setText("")
        else:
            btn.setText("Pause" if playing else "Play")
        btn.setStyleSheet(
            "QPushButton { background-color: #2ecc71; color: white; }"
            if playing
            else ""
        )

    def _update_timer_from_speed(self, main_window):
        """Apply preview timer interval based on FPS control."""
        if not hasattr(main_window, "_preview_timer"):
            return
        fps = max(1, int(main_window.gif_speed_input.value()))
        interval = max(10, int(1000 / fps))
        if main_window._preview_timer.isActive():
            main_window._preview_timer.start(interval)

    def _toggle_play(self, main_window):
        """Toggle preview playback on/off."""
        if not hasattr(main_window, "_preview_timer"):
            return
        if main_window._preview_timer.isActive():
            self._stop_play(main_window)
        else:
            self._start_play(main_window)

    def _start_play(self, main_window):
        """Start preview playback from current start time."""
        fps = max(1, int(main_window.gif_speed_input.value()))
        interval = max(10, int(1000 / fps))
        main_window._loop_start_t = int(main_window.start_t_input.value())
        main_window._preview_timer.start(interval)
        self._set_play_button_state(main_window, True)

    def _stop_play(self, main_window):
        """Stop preview playback and update button state."""
        if hasattr(main_window, "_preview_timer"):
            main_window._preview_timer.stop()

        with suppress(Exception):
            if hasattr(main_window, "_loop_start_t"):
                main_window.start_t_input.setValue(
                    int(main_window._loop_start_t)
                )

        self._set_play_button_state(main_window, False)

    def _advance_frame(self, main_window):
        """Advance start frame or loop within selected time window."""
        with suppress(Exception):
            t = int(main_window.start_t_input.value())
            t_end = int(main_window.end_t_input.value())
            loop_start = int(getattr(main_window, "_loop_start_t", t))
            main_window.start_t_input.setValue(
                t + 1 if t < t_end else loop_start
            )

    def _update_all_previews(self, main_window):
        """Re-render previews for all panels."""
        if not is_widget_alive(getattr(main_window, "gif_window", None)):
            return
        for p in getattr(main_window, "panels", []):
            self._update_single_panel_preview(main_window, p)
