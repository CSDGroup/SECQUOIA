"""Tests for pruning measurement columns a configuration change orphaned.

Switching BaSiC off (or dropping a channel or a mask) changes which column
families ``quantify`` writes. Nothing used to remove the columns of the
previous configuration, so ``track_df`` and with it ``filtered_df`` and the
feature dropdowns.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from SECQUOIA.core.quantification import (
    FeatureNaming,
    _configured_mask_indices,
    _write_empty_measurement_columns,
    drop_stale_measurement_columns,
    expected_measurement_columns,
    stale_measurement_columns,
)
from SECQUOIA.core.tracking import track_data
from SECQUOIA.core.tracking.track_data import (
    _load_cached_position_measurements,
    apply_derived_features,
)

CHANNELS = ["C00", "C01"]
POSITION = 1

BASIC_VARIANTS = (
    "NoBgCorrected",
    "BaSiCBgCorrectedRatioFlat",
    "BaSiCBgCorrectedNoRatioFlat",
)


def naming_for(channels=CHANNELS, *, basic: bool) -> FeatureNaming:
    """Build the naming a main window with this configuration would produce."""
    return FeatureNaming.from_main_window(
        SimpleNamespace(
            ids_channels=list(channels),
            basic=SimpleNamespace(flag=basic),
        )
    )


def measured_frame(
    channels=CHANNELS, *, variants=BASIC_VARIANTS, masks=(1,)
) -> pd.DataFrame:
    """A track_df as it looks after a run with the given configuration."""
    data: dict[str, list[float]] = {}
    for mask in masks:
        for variant in variants:
            for channel in channels:
                for metric in ("Mean", "Std"):
                    data[f"{metric}{variant}Ch{channel[1:]}M{mask}"] = [
                        1.0,
                        2.0,
                    ]
        data[f"AreaMorphologyM{mask}"] = [10.0, 11.0]
        data[f"XMorphologyM{mask}"] = [3.0, 4.0]
        data[f"YMorphologyM{mask}"] = [5.0, 6.0]
        data[f"label_id_m{mask}"] = [1, 2]
        data[f"nn_dist_px_m{mask}"] = [0.5, 0.6]
        data[f"step_disp_px_m{mask}"] = [0.0, 1.5]
    return pd.DataFrame(
        {"Position": [POSITION, POSITION], "t": [0, 1], **data}
    )


def test_expected_columns_cover_every_written_family():
    """A frame measured with the same config has nothing to prune."""
    naming = naming_for(basic=True)
    frame = measured_frame()

    expected = expected_measurement_columns(naming, [1])
    measurement_cols = set(frame.columns) - {"Position", "t"}

    assert measurement_cols <= expected
    assert stale_measurement_columns(frame.columns, naming, [1]) == []


def test_switching_basic_off_drops_only_the_basic_variants():
    """The common case: BaSiC run first, then the same data without BaSiC."""
    frame = measured_frame()

    pruned = drop_stale_measurement_columns(
        frame, naming_for(basic=False), [1]
    )

    assert "MeanBaSiCBgCorrectedRatioFlatCh00M1" not in pruned.columns
    assert "MeanBaSiCBgCorrectedNoRatioFlatCh01M1" not in pruned.columns
    assert "StdBaSiCBgCorrectedRatioFlatCh01M1" not in pruned.columns
    assert pruned["MeanNoBgCorrectedCh00M1"].tolist() == [1.0, 2.0]
    assert pruned["StdNoBgCorrectedCh01M1"].tolist() == [1.0, 2.0]
    assert pruned["AreaMorphologyM1"].tolist() == [10.0, 11.0]
    assert pruned["label_id_m1"].tolist() == [1, 2]


def test_switching_basic_on_drops_nothing():
    """Enabling BaSiC only adds families; the existing ones stay valid."""
    frame = measured_frame(variants=("NoBgCorrected",))

    pruned = drop_stale_measurement_columns(frame, naming_for(basic=True), [1])

    assert list(pruned.columns) == list(frame.columns)


def test_removing_a_channel_drops_its_intensity_columns():
    frame = measured_frame(channels=["C00", "C01", "C02"])

    pruned = drop_stale_measurement_columns(
        frame, naming_for(["C00", "C01"], basic=True), [1]
    )

    assert "MeanNoBgCorrectedCh02M1" not in pruned.columns
    assert "MeanBaSiCBgCorrectedRatioFlatCh02M1" not in pruned.columns
    assert "MeanNoBgCorrectedCh01M1" in pruned.columns


def test_removing_a_mask_drops_its_whole_column_family():
    frame = measured_frame(masks=(1, 2))

    pruned = drop_stale_measurement_columns(frame, naming_for(basic=True), [1])

    for column in (
        "MeanNoBgCorrectedCh00M2",
        "AreaMorphologyM2",
        "XMorphologyM2",
        "YMorphologyM2",
        "label_id_m2",
        "nn_dist_px_m2",
        "step_disp_px_m2",
    ):
        assert column not in pruned.columns
    for column in (
        "MeanNoBgCorrectedCh00M1",
        "AreaMorphologyM1",
        "label_id_m1",
        "nn_dist_px_m1",
        "step_disp_px_m1",
    ):
        assert column in pruned.columns


def test_an_unpadded_channel_token_is_stale():
    frame = measured_frame()
    frame["MeanNoBgCorrectedCh1M1"] = [7.0, 8.0]

    pruned = drop_stale_measurement_columns(frame, naming_for(basic=True), [1])

    assert "MeanNoBgCorrectedCh1M1" not in pruned.columns
    assert pruned["MeanNoBgCorrectedCh01M1"].tolist() == [1.0, 2.0]


def test_columns_outside_the_measurement_families_are_never_pruned():
    """Curation, outlier, timing and user-defined columns must survive."""
    frame = measured_frame()
    survivors = {
        "Identification": ["a", "b"],
        "TrackNumber": [1, 1],
        "ID": [1, 2],
        "Cellfate": ["Healthy", "Healthy"],
        "active": [1, 1],
        "inspected": [0, 0],
        "Outlier_detection": [False, True],
        "realtime": [0.0, 300.0],
        "XMorphology": [3, 4],
        "YMorphology": [5, 6],
        "MeanNoBgCorrectedCh00M1/AreaMorphologyM1": [0.1, 0.2],
        "MeanBaSiCBgCorrectedRatioFlatCh00M1/AreaMorphologyM1": [0.3, 0.4],
    }
    for name, values in survivors.items():
        frame[name] = values

    pruned = drop_stale_measurement_columns(
        frame, naming_for(basic=False), [1]
    )

    for name, values in survivors.items():
        assert name in pruned.columns, name
        assert pruned[name].tolist() == values


def test_pruning_applies_to_every_position_at_once():
    """The configuration is global, so a stale column is stale everywhere."""
    other = measured_frame()
    other["Position"] = [2, 2]
    frame = pd.concat([measured_frame(), other], ignore_index=True)

    pruned = drop_stale_measurement_columns(
        frame, naming_for(basic=False), [1]
    )

    assert not [c for c in pruned.columns if "BaSiC" in c]
    assert len(pruned) == 4


def test_a_mask_that_failed_to_load_keeps_its_columns():
    """Pruning follows the configured mask count, not what loaded this run."""
    main_window = SimpleNamespace(n_masks=2)

    # Only mask 1 produced label stacks this time.
    assert list(_configured_mask_indices(main_window, [1])) == [1, 2]


def test_pruning_follows_a_mask_removed_from_the_configuration():
    main_window = SimpleNamespace(n_masks=1)

    assert list(_configured_mask_indices(main_window, [1])) == [1]


def test_quantify_prunes_when_no_objects_were_measured():
    """The 'nothing measured' branch must clean up the columns too."""
    main_window = SimpleNamespace(
        n_masks=1,
        track_df=measured_frame(),
        ids_channels=list(CHANNELS),
        basic=SimpleNamespace(flag=False),
    )

    _write_empty_measurement_columns(
        main_window, POSITION, [1], naming_for(basic=False), "ID"
    )

    assert not [c for c in main_window.track_df.columns if "BaSiC" in c]
    assert "MeanNoBgCorrectedCh00M1" in main_window.track_df.columns


@pytest.fixture
def cached_csv(tmp_path, monkeypatch):
    """Point the cache loader at a CSV this test controls."""
    path = tmp_path / "cache.csv"
    monkeypatch.setattr(
        track_data, "_position_csv_path", lambda *_a, **_k: str(path)
    )
    return path


def main_window_for(*, basic: bool) -> SimpleNamespace:
    """A stand-in exposing what the cache loader and naming read."""
    return SimpleNamespace(
        ids_channels=list(CHANNELS),
        n_channels=len(CHANNELS),
        n_masks=1,
        basic=SimpleNamespace(flag=basic),
        track_df=pd.DataFrame({"Position": [2], "t": [0]}),
    )


def test_a_basic_cache_cannot_reintroduce_the_columns(cached_csv):
    """A CSV written under BaSiC must not resurrect it after switching off."""
    measured_frame().to_csv(cached_csv, index=False)
    main_window = main_window_for(basic=False)

    loaded = _load_cached_position_measurements(
        main_window, POSITION, 0, 1, ensure_columns=True
    )

    assert not [c for c in loaded.columns if "BaSiC" in c]
    assert not [c for c in main_window.track_df.columns if "BaSiC" in c]
    assert loaded["MeanNoBgCorrectedCh00M1"].tolist() == [1.0, 2.0]


def test_a_basic_cache_is_kept_while_basic_is_on(cached_csv):
    measured_frame().to_csv(cached_csv, index=False)
    main_window = main_window_for(basic=True)

    loaded = _load_cached_position_measurements(
        main_window, POSITION, 0, 1, ensure_columns=True
    )

    assert loaded["MeanBaSiCBgCorrectedRatioFlatCh00M1"].tolist() == [1.0, 2.0]


def test_a_cache_is_pruned_without_ensure_columns(cached_csv):
    """The initial load path does not backfill."""
    frame = measured_frame()
    frame["MeanNoBgCorrectedCh1M1"] = [np.nan, np.nan]
    frame.to_csv(cached_csv, index=False)
    main_window = main_window_for(basic=False)

    loaded = _load_cached_position_measurements(main_window, POSITION, 0, 1)

    assert not [c for c in loaded.columns if "BaSiC" in c]
    assert "MeanNoBgCorrectedCh1M1" not in loaded.columns
    assert loaded["MeanNoBgCorrectedCh01M1"].tolist() == [1.0, 2.0]
    assert "SumNoBgCorrectedCh01M1" not in loaded.columns


def test_an_unknown_configuration_keeps_every_column(cached_csv):
    """Without a readable config nothing is deleted."""
    measured_frame().to_csv(cached_csv, index=False)
    main_window = main_window_for(basic=False)
    del main_window.basic

    loaded = _load_cached_position_measurements(
        main_window, POSITION, 0, 1, ensure_columns=True
    )

    assert "MeanBaSiCBgCorrectedRatioFlatCh00M1" in loaded.columns


def test_a_derived_metric_is_cleared_when_its_source_disappears():
    """An arithmetic metric over a dropped BaSiC variant must not keep values."""
    target = "MeanBaSiCBgCorrectedRatioFlatCh00M1/AreaMorphologyM1"
    main_window = SimpleNamespace(
        track_df=pd.DataFrame(
            {
                "AreaMorphologyM1": [10.0, 20.0],
                target: [0.3, 0.4],
            }
        ),
        _derived_features={
            "ratio": {
                "formula": {
                    "source_col_a": "MeanBaSiCBgCorrectedRatioFlatCh00M1",
                    "source_col_b": "AreaMorphologyM1",
                    "target_col": target,
                    "op": "/",
                }
            }
        },
    )

    out = apply_derived_features(main_window)

    assert out[target].isna().all()


def test_a_derived_metric_is_recomputed_while_its_sources_exist():
    target = "MeanNoBgCorrectedCh00M1/AreaMorphologyM1"
    main_window = SimpleNamespace(
        track_df=pd.DataFrame(
            {
                "MeanNoBgCorrectedCh00M1": [4.0, 9.0],
                "AreaMorphologyM1": [2.0, 3.0],
                target: [np.nan, np.nan],
            }
        ),
        _derived_features={
            "ratio": {
                "formula": {
                    "source_col_a": "MeanNoBgCorrectedCh00M1",
                    "source_col_b": "AreaMorphologyM1",
                    "target_col": target,
                    "op": "/",
                }
            }
        },
    )

    out = apply_derived_features(main_window)

    assert out[target].tolist() == [2.0, 3.0]
