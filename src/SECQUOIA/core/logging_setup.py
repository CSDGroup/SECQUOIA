"""Central logging configuration for SECQUOIA.

At startup only a console handler is attached (no file is written yet,
since no experiment is loaded). Once an experiment folder is selected,
`attach_experiment_log` moves the file output to that experiment's
``Analysis`` folder, so the log lives with the data it describes.
"""

from __future__ import annotations

import logging
from datetime import datetime

from SECQUOIA.core.experiment_layout import find_analysis_dir

LOGGER_NAME = "SECQUOIA"
LOG_FILENAME = "SECQUOIA.log"
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

_file_handler: logging.Handler | None = None


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Set up the SECQUOIA logger with a console handler only.

    Call once at application startup, before any experiment folder is known.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter(_LOG_FORMAT, "%H:%M:%S")
        )
        logger.addHandler(console_handler)

    return logger


def attach_experiment_log(folder: str | None) -> None:
    """(Re)attach the file handler to ``<folder>/Analysis/SECQUOIA.log``.

    Any previously attached file handler is removed first, so switching
    experiments within one session doesn't leave multiple open log files.
    """
    global _file_handler
    logger = logging.getLogger(LOGGER_NAME)

    if _file_handler is not None:
        logger.removeHandler(_file_handler)
        _file_handler.close()
        _file_handler = None

    analysis_dir = find_analysis_dir(folder)
    if not analysis_dir:
        return

    log_path = f"{analysis_dir}/{LOG_FILENAME}"
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    _file_handler = handler

    logger.info(
        "--- SECQUOIA session started %s ---",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    logger.info("Logging to %s", log_path)
