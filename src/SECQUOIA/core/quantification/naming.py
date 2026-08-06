"""Measurement column naming helpers shared across the quantify pipeline."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from SECQUOIA.config import FEATURES


def safe_op_series(a: pd.Series, b: pd.Series, op: str) -> pd.Series:
    """Elementwise ``a op b`` for one of ``"/", "*", "+", "-"``.

    Both series are coerced to numeric (non-numeric values become NaN),
    and division by zero yields NaN rather than raising or returning inf.
    An unrecognized ``op`` yields a series of NaN the same length as ``a``.
    """
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    if op == "/":
        b = b.replace(0, np.nan)
        return a / b
    if op == "*":
        return a * b
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    return pd.Series(np.nan, index=a.index)


def label_column(mask_idx: int) -> str:
    """Column holding the matched label id of mask `mask_idx`."""
    return f"label_id_m{mask_idx}"


def distance_column(mask_idx: int) -> str:
    """Column holding the track-to-object distance (px) for mask `mask_idx`."""
    return f"nn_dist_px_m{mask_idx}"


def step_distance_column(mask_idx: int) -> str:
    """Column holding the per frame lineage displacement (px) for mask `mask_idx`."""
    return f"step_disp_px_m{mask_idx}"


def centroid_columns(mask_idx: int) -> tuple[str, str]:
    """Per mask centroid columns (x, y)."""
    return f"XMorphologyM{mask_idx}", f"YMorphologyM{mask_idx}"


def intensity_column(prefix: str, channel: str, mask_idx: int) -> str:
    """Column name for an intensity metric, e.g. `MeanNoBgCorrectedCh01M2`."""
    return f"{prefix}Ch{channel[1:]}M{mask_idx}"


@dataclass(frozen=True)
class FeatureNaming:
    """Builds the measurement column names used throughout quantification."""

    metric_prefixes: tuple[str, ...]
    shape_prefixes: tuple[str, ...]
    channels: tuple[str, ...]

    @classmethod
    def for_config(cls, channels, *, basic: bool) -> "FeatureNaming":
        """Build a `FeatureNaming` for an explicit channel list and BaSiC flag."""
        variants = (
            FEATURES.BASIC_VARIANT if basic else FEATURES.NO_BASIC_VARIANT
        )
        return cls(
            metric_prefixes=tuple(
                f"{metric}{variant}"
                for variant in variants
                for metric in FEATURES.METRIC_PREFIXES
            ),
            shape_prefixes=tuple(FEATURES.MORPH_PREFIXES),
            channels=tuple(channels),
        )

    @classmethod
    def from_main_window(cls, main_window) -> "FeatureNaming":
        """Build a `FeatureNaming` from the current GUI state."""
        return cls.for_config(
            main_window.ids_channels, basic=main_window.basic.flag
        )

    def intensity_column(
        self, prefix: str, channel: str, mask_idx: int
    ) -> str:
        """Column name for an intensity metric.

        `prefix` carries both the metric and the BaSiC variant, so a
        full name looks like ``MeanNoBgCorrectedCh01M2``.
        """
        return intensity_column(prefix, channel, mask_idx)

    def shape_column(self, prefix: str, mask_idx: int) -> str:
        """Column name for a shape metric, e.g. `AreaMorphologyM2`."""
        return f"{prefix}M{mask_idx}"

    def intensity_columns(self, channel: str, mask_idx: int) -> list[str]:
        """All intensity metric column names for one channel and mask."""
        return [
            self.intensity_column(prefix, channel, mask_idx)
            for prefix in self.metric_prefixes
        ]
