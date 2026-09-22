"""Tests for SECQUOIA quantification.

These tests share a tiny artificial dataset:

- one position
- one cell
- two time points
- one fluorescence channel: w01
- two segmentation masks:
    - mask 1 is bigger: 3 x 3 pixels, area 9
    - mask 2 is smaller: 2 x 2 pixels, area 4

Expected values:
- mask 1:
    - area: 9 at both time points
    - mean intensity: 10 at t=0, 20 at t=1
    - sum intensity: 90 at t=0, 180 at t=1
- mask 2:
    - area: 4 at both time points
    - mean intensity: 10 at t=0, 20 at t=1
    - sum intensity: 40 at t=0, 80 at t=1
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from conftest import (
    assert_column_values,
    find_module_name,
    make_fake_images,
    make_fake_labels,
    make_fake_main_window,
)

from SECQUOIA.core.normalization import apply_normalization
from SECQUOIA.core.quantification import quantify, safe_op_series
from SECQUOIA.core.quantification.columns import init_mask_channel_columns
from SECQUOIA.core.quantification.matching import (
    COMPARE_MATCHING_ENV,
    DEFAULT_MATCH_STRATEGY,
    MATCH_STRATEGIES,
    MATCH_STRATEGY_ENV,
    assign_objects_to_tracks,
    compare_matching_enabled,
    greedy_nearest_assign,
    resolve_match_strategy,
    second_nearest_within,
)
from SECQUOIA.core.quantification.naming import FeatureNaming


def test_quantify_one_cell_two_timepoints_two_masks(tmp_path, no_progress):
    main_window = make_fake_main_window(tmp_path)

    quantify(
        main_window,
        progress_cb=no_progress,
        max_pixel_distance=5,
    )

    after = main_window.track_df.copy()
    after = after.sort_values(["Identification", "TrackNumber", "t"])
    after = after.reset_index(drop=True)

    assert len(after) == 2
    assert after["t"].tolist() == [0, 1]
    assert after["Identification"].tolist() == ["fake-001", "fake-001"]
    assert after["TrackNumber"].tolist() == [1, 1]
    assert_column_values(after, "label_id_m1", [1, 1])
    assert_column_values(after, "label_id_m2", [1, 1])
    assert_column_values(after, "AreaMorphologyM1", [9, 9])
    assert_column_values(after, "AreaMorphologyM2", [4, 4])
    assert_column_values(after, "MeanNoBgCorrectedCh01M1", [10, 20])
    assert_column_values(after, "SumNoBgCorrectedCh01M1", [90, 180])
    assert_column_values(after, "MeanNoBgCorrectedCh01M2", [10, 20])
    assert_column_values(after, "SumNoBgCorrectedCh01M2", [40, 80])


@pytest.mark.smoke
def test_quantify_imports():
    """Basic smoke test: the function can be imported."""
    assert callable(quantify)


SQRT_HALF = 0.7071067811865475


def metric_dialog_module():
    """Import the module holding MetricDialog, build_column_names, side_tag."""
    return importlib.import_module(find_module_name("metric_dialog.py"))


def make_feature_defs() -> dict:
    """Feature definitions as ``update_plot`` would derive them."""
    return {
        "AreaMorphology": {
            "template": "AreaMorphologyM{m}",
            "has_ch": False,
            "has_m": True,
            "display": "AreaMorphology",
        },
        "SumNoBgCorrected": {
            "template": "SumNoBgCorrectedCh{ch}M{m}",
            "has_ch": True,
            "has_m": True,
            "display": "SumNoBgCorrected",
        },
        "MeanNoBgCorrected": {
            "template": "MeanNoBgCorrectedCh{ch}M{m}",
            "has_ch": True,
            "has_m": True,
            "display": "MeanNoBgCorrected",
        },
    }


def run_metric(
    df: pd.DataFrame,
    feature_defs: dict,
    feat_a: str,
    c1v,
    m1v: int,
    op_val: str,
    feat_b: str,
    c2v,
    m2v: int,
    norm_cfg: dict | None = None,
) -> str:
    """Reproduce the data part of ``MetricDialog._on_run`` on one dataframe."""
    dialog = metric_dialog_module()

    col_a, col_b, has_ch_l, has_ch_r = dialog.build_column_names(
        feature_defs, feat_a, c1v, m1v, feat_b, c2v, m2v
    )
    assert (
        col_a is not None and col_b is not None
    ), f"Could not resolve column names for {feat_a!r} and {feat_b!r}."

    left_tag = dialog.side_tag(has_ch_l, c1v, m1v)
    right_tag = dialog.side_tag(has_ch_r, c2v, m2v)
    col_new = f"{feat_a}{left_tag}{op_val}{feat_b}{right_tag}"

    assert col_a in df.columns, f"{col_a!r} missing from quantify output."
    assert col_b in df.columns, f"{col_b!r} missing from quantify output."

    df[col_new] = safe_op_series(df[col_a], df[col_b], op_val)

    if norm_cfg and norm_cfg.get("norm_enabled"):
        apply_normalization(
            df,
            col_new,
            norm_cfg["norm_method"],
            norm_cfg["norm_scope"],
            int(norm_cfg["norm_t_min"]),
            int(norm_cfg["norm_t_max"]),
        )

    return col_new


@pytest.fixture
def quantified_df(tmp_path, no_progress) -> pd.DataFrame:
    """Run the quantify pipeline once and hand back its sorted track_df."""
    main_window = make_fake_main_window(tmp_path)
    quantify(main_window, progress_cb=no_progress, max_pixel_distance=5)

    df = main_window.track_df.copy()
    df = df.sort_values(["Identification", "TrackNumber", "t"])
    return df.reset_index(drop=True)


def test_metric_ratio_without_normalization(quantified_df):
    """Sum / Area on mask 1 gives back the mean intensity."""
    col = run_metric(
        quantified_df,
        make_feature_defs(),
        "SumNoBgCorrected",
        "01",
        1,
        "/",
        "AreaMorphology",
        None,
        1,
    )

    assert col == "SumNoBgCorrectedCh01M1/AreaMorphologyM1"
    assert_column_values(quantified_df, col, [10, 20])


def test_metric_ratio_mask_2(quantified_df):
    """The same metric on the smaller mask: 40/4 and 80/4."""
    col = run_metric(
        quantified_df,
        make_feature_defs(),
        "SumNoBgCorrected",
        "01",
        2,
        "/",
        "AreaMorphology",
        None,
        2,
    )

    assert col == "SumNoBgCorrectedCh01M2/AreaMorphologyM2"
    assert_column_values(quantified_df, col, [10, 20])


def test_metric_ratio_between_masks(quantified_df):
    """Area of mask 1 over area of mask 2: 9/4 at both time points."""
    col = run_metric(
        quantified_df,
        make_feature_defs(),
        "AreaMorphology",
        None,
        1,
        "/",
        "AreaMorphology",
        None,
        2,
    )

    assert_column_values(quantified_df, col, [2.25, 2.25])


@pytest.mark.parametrize("scope", ["all", "ids"])
def test_metric_then_zscore(quantified_df, scope):
    """Z-score of [10, 20] is symmetric around zero.

    With a single cell in the dataset, per-ID and global scope have to agree.
    """
    col = run_metric(
        quantified_df,
        make_feature_defs(),
        "SumNoBgCorrected",
        "01",
        1,
        "/",
        "AreaMorphology",
        None,
        1,
        norm_cfg={
            "norm_enabled": True,
            "norm_method": "zscore",
            "norm_scope": scope,
            "norm_t_min": 0,
            "norm_t_max": 1,
        },
    )

    assert_column_values(quantified_df, col, [-SQRT_HALF, SQRT_HALF])


@pytest.mark.parametrize(
    ("tmax", "expected"),
    [
        (0, [1.0, 2.0]),
        (1, [10 / 15, 20 / 15]),
    ],
)
@pytest.mark.parametrize("scope", ["all", "ids"])
def test_metric_then_inverse(quantified_df, scope, tmax, expected):
    """Inverse normalisation divides by the mean inside the time window."""
    col = run_metric(
        quantified_df,
        make_feature_defs(),
        "SumNoBgCorrected",
        "01",
        1,
        "/",
        "AreaMorphology",
        None,
        1,
        norm_cfg={
            "norm_enabled": True,
            "norm_method": "inv",
            "norm_scope": scope,
            "norm_t_min": 0,
            "norm_t_max": tmax,
        },
    )

    assert_column_values(quantified_df, col, expected)


def test_normalization_disabled_leaves_values_untouched(quantified_df):
    """``norm_enabled=False`` must not modify the derived column."""
    col = run_metric(
        quantified_df,
        make_feature_defs(),
        "SumNoBgCorrected",
        "01",
        1,
        "/",
        "AreaMorphology",
        None,
        1,
        norm_cfg={
            "norm_enabled": False,
            "norm_method": "zscore",
            "norm_scope": "all",
            "norm_t_min": 0,
            "norm_t_max": 1,
        },
    )

    assert_column_values(quantified_df, col, [10, 20])


def make_norm_df(values: list[float], ids: list[str] | None = None):
    """Small dataframe shaped like track_df, for normalisation unit tests."""
    n = len(values)
    return pd.DataFrame(
        {
            "Identification": ids or [f"cell-{i:03d}" for i in range(n)],
            "t": list(range(n)),
            "value": values,
        }
    )


def test_safe_op_series_operators():
    a = pd.Series([10.0, 20.0])
    b = pd.Series([2.0, 4.0])

    assert safe_op_series(a, b, "+").tolist() == [12.0, 24.0]
    assert safe_op_series(a, b, "-").tolist() == [8.0, 16.0]
    assert safe_op_series(a, b, "*").tolist() == [20.0, 80.0]
    assert safe_op_series(a, b, "/").tolist() == [5.0, 5.0]


def test_safe_op_series_division_by_zero_is_nan():
    """A zero denominator must not raise and must not produce inf."""
    result = safe_op_series(pd.Series([10.0]), pd.Series([0.0]), "/")

    assert result.isna().all()


def test_safe_op_series_unknown_operator_is_nan():
    result = safe_op_series(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), "^")

    assert result.isna().all()


def test_zscore_constant_column_becomes_zero():
    """Zero standard deviation would divide by zero; the result must be 0."""
    df = make_norm_df([5.0, 5.0, 5.0], ids=["a", "a", "a"])
    apply_normalization(df, "value", "zscore", "all", 0, 2)

    assert_column_values(df, "value", [0.0, 0.0, 0.0])


def test_zscore_per_id_is_independent_per_cell():
    """Two cells with different offsets must z-score to the same shape."""
    df = pd.DataFrame(
        {
            "Identification": ["a", "a", "b", "b"],
            "t": [0, 1, 0, 1],
            "value": [10.0, 20.0, 110.0, 120.0],
        }
    )
    apply_normalization(df, "value", "zscore", "ids", 0, 1)

    assert_column_values(
        df, "value", [-SQRT_HALF, SQRT_HALF, -SQRT_HALF, SQRT_HALF]
    )


def test_zscore_ignores_the_time_window():
    df = make_norm_df([10.0, 20.0, 30.0], ids=["a", "a", "a"])
    apply_normalization(df, "value", "zscore", "all", 0, 0)

    assert_column_values(df, "value", [-1.0, 0.0, 1.0])


def test_inverse_per_id_falls_back_to_global_baseline():
    """A cell with no rows in the window uses the global window mean."""
    df = pd.DataFrame(
        {
            "Identification": ["a", "a", "b"],
            "t": [0, 1, 5],
            "value": [10.0, 20.0, 40.0],
        }
    )
    apply_normalization(df, "value", "inv", "ids", 0, 1)
    assert_column_values(df, "value", [10 / 15, 20 / 15, 40 / 15])


def test_inverse_zero_baseline_is_nan():
    """A baseline of zero must produce NaN rather than inf."""
    df = make_norm_df([0.0, 5.0], ids=["a", "a"])
    apply_normalization(df, "value", "inv", "all", 0, 0)

    assert df["value"].isna().all()


def test_apply_normalization_unknown_method_is_a_no_op():
    df = make_norm_df([10.0, 20.0], ids=["a", "a"])
    apply_normalization(df, "value", "minmax", "all", 0, 1)

    assert_column_values(df, "value", [10.0, 20.0])


def test_apply_normalization_missing_column_is_a_no_op():
    df = make_norm_df([10.0, 20.0], ids=["a", "a"])
    apply_normalization(df, "does_not_exist", "zscore", "all", 0, 1)

    assert df.columns.tolist() == ["Identification", "t", "value"]


@pytest.mark.gui
@pytest.mark.smoke
def test_metric_dialog_imports():
    """The metric dialog pulls in Qt, so it is marked as a GUI test."""
    dialog = metric_dialog_module()
    assert callable(dialog.build_column_names)
    assert callable(dialog.side_tag)


@pytest.mark.gui
def test_build_column_names_channel_and_mask_templates():
    (
        col_a,
        col_b,
        has_ch_l,
        has_ch_r,
    ) = metric_dialog_module().build_column_names(
        make_feature_defs(),
        "SumNoBgCorrected",
        "01",
        1,
        "AreaMorphology",
        None,
        2,
    )

    assert col_a == "SumNoBgCorrectedCh01M1"
    assert col_b == "AreaMorphologyM2"
    assert has_ch_l is True
    assert has_ch_r is False


@pytest.mark.gui
def test_build_column_names_unknown_feature_returns_none():
    col_a, col_b, _, _ = metric_dialog_module().build_column_names(
        make_feature_defs(), "NotAFeature", "01", 1, "AreaMorphology", None, 1
    )

    assert col_a is None
    assert col_b is None


def _make_cytometric_files(main_window) -> None:
    """Create the on-disk image + mask files."""
    pos = Path(main_window.folder) / "fake_p0001"
    for t in (0, 1):
        (pos / f"fake_t{t:05d}_w01.png").touch()
        for seg_root in main_window.segmentation_paths:
            seg_pos = Path(seg_root) / "fake_p0001"
            seg_pos.mkdir(parents=True, exist_ok=True)
            (seg_pos / f"mask_t{t:05d}.png").touch()


def _install_cytometric_stubs(monkeypatch) -> None:
    """Replace the disk/GUI helpers so the same in-memory arrays used by the
    quantify test flow through run_cytometric_analysis, and Qt dialogs stay
    silent instead of blocking the test run."""
    images = make_fake_images()
    masks = make_fake_labels()

    def fake_fl_channel(mw, progress_cb=None) -> None:
        mw.images = images
        mw.image_present = {"w01": np.array([True, True])}

    fake_widgets = ModuleType("SECQUOIA.gui.widgets")
    fake_widgets._FL_channel = fake_fl_channel
    fake_widgets.apply_basic_correction = lambda *a, **k: None
    fake_widgets.add_mouse_drag_to_segmentation_layer = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "SECQUOIA.gui.widgets", fake_widgets)

    monkeypatch.setattr(
        "SECQUOIA.core.segmentation.mask_io.load_masks",
        lambda *a, **k: list(masks),
        raising=False,
    )

    import SECQUOIA.core.cytometric_analysis as cyto

    monkeypatch.setattr(cyto, "load_fl_channels", fake_fl_channel)
    monkeypatch.setattr(cyto, "apply_basic_correction", lambda *a, **k: None)
    monkeypatch.setattr(cyto, "cleanup_memmaps", lambda *a, **k: None)
    monkeypatch.setattr(
        cyto,
        "QMessageBox",
        SimpleNamespace(
            information=lambda *a, **k: None,
            warning=lambda *a, **k: None,
        ),
    )
    monkeypatch.setattr(
        cyto,
        "QCoreApplication",
        SimpleNamespace(processEvents=lambda *a, **k: None),
    )


def test_run_cytometric_analysis_one_position_two_masks(tmp_path, monkeypatch):
    main_window = make_fake_main_window(tmp_path)
    _make_cytometric_files(main_window)
    _install_cytometric_stubs(monkeypatch)

    from SECQUOIA.core.cytometric_analysis import run_cytometric_analysis

    run_cytometric_analysis(main_window)

    out_dir = Path(main_window.folder) / "Analysis" / "Cytometric_Analysis"
    m1 = pd.read_csv(out_dir / "Cytometric_Analysis_p0001_M1.csv").sort_values(
        "tM1"
    )
    m2 = pd.read_csv(out_dir / "Cytometric_Analysis_p0001_M2.csv").sort_values(
        "tM2"
    )

    assert_column_values(m1, "AreaMorphologyM1", [9, 9])
    assert_column_values(m1, "MeanNoBgCorrectedCh01M1", [10, 20])
    assert_column_values(m1, "SumNoBgCorrectedCh01M1", [90, 180])

    assert_column_values(m2, "AreaMorphologyM2", [4, 4])
    assert_column_values(m2, "MeanNoBgCorrectedCh01M2", [10, 20])
    assert_column_values(m2, "SumNoBgCorrectedCh01M2", [40, 80])

    assert (out_dir / "Cytometric_Analysis_p0001_matched.csv").exists()


@pytest.mark.smoke
def test_run_cytometric_analysis_imports():
    """Basic smoke test: the function can be imported."""
    from SECQUOIA.core.cytometric_analysis import run_cytometric_analysis

    assert callable(run_cytometric_analysis)


# Second potential mask (alt_label_id_m*, alt_dist_px_m*)
def xy(*points) -> np.ndarray:
    return np.array(points, dtype=float).reshape(-1, 2)


def test_second_nearest_has_no_candidate_with_a_single_object():
    idx, dist = second_nearest_within(
        xy((0, 0)), xy((1, 0)), np.array([0]), max_dist=10
    )

    assert idx.tolist() == [-1]
    assert np.isinf(dist).all()


def test_second_nearest_finds_the_nearest_other_object():
    objects = xy((1, 0), (4, 0), (2, 0))

    idx, dist = second_nearest_within(
        xy((0, 0)), objects, np.array([0]), max_dist=10
    )

    assert idx.tolist() == [2]
    assert dist.tolist() == [2.0]


def test_second_nearest_is_inclusive_at_the_threshold():
    objects = xy((1, 0), (5, 0))
    tracks, assigned = xy((0, 0)), np.array([0])

    at_limit, _ = second_nearest_within(tracks, objects, assigned, 5.0)
    below_limit, _ = second_nearest_within(tracks, objects, assigned, 4.99)

    assert at_limit.tolist() == [1]
    assert below_limit.tolist() == [-1]


def test_second_nearest_counts_objects_claimed_by_another_track():
    """A track that fell back to a farther object still sees the taken one."""
    tracks = xy((0, 0), (2, 0))
    objects = xy((1, 0), (9, 0))
    # Track 0 owns object 0; track 1 was pushed to object 1.
    assigned = np.array([0, 1])

    idx, dist = second_nearest_within(tracks, objects, assigned, max_dist=10)

    assert idx.tolist() == [1, 0]
    assert dist.tolist() == [9.0, 1.0]


def test_second_nearest_for_an_unmatched_track_excludes_nothing():
    idx, dist = second_nearest_within(
        xy((0, 0)), xy((3, 0)), np.array([-1]), max_dist=5
    )

    assert idx.tolist() == [0]
    assert dist.tolist() == [3.0]


def test_second_nearest_ties_go_to_the_lower_object_index():
    objects = xy((0, 0), (2, 0), (0, 2))

    idx, _ = second_nearest_within(
        xy((1, 1)), objects, np.array([-1]), max_dist=5
    )

    assert idx.tolist() == [0]


def test_second_nearest_handles_empty_inputs():
    empty = np.empty((0, 2))

    no_tracks = second_nearest_within(empty, xy((1, 1)), np.array([]), 5)
    no_objects = second_nearest_within(xy((1, 1)), empty, np.array([-1]), 5)

    assert no_tracks[0].tolist() == []
    assert no_objects[0].tolist() == [-1]
    assert np.isinf(no_objects[1]).all()


def assert_column_with_nan(df: pd.DataFrame, column: str, expected) -> None:
    """Like ``assert_column_values`` but a NaN matches a NaN."""
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    np.testing.assert_allclose(values, np.asarray(expected, dtype=float))


def two_object_labels() -> list[np.ndarray]:
    """Mask 1 gets a second object 4 px from the tracking point at t=0 only."""
    mask_1, mask_2 = make_fake_labels()
    mask_1[0, 2:5, 6:9] = 2
    return [mask_1, mask_2]


def test_quantify_records_the_second_mask_within_the_threshold(
    tmp_path, no_progress
):
    main_window = make_fake_main_window(tmp_path)
    main_window.labels = two_object_labels()

    quantify(main_window, progress_cb=no_progress, max_pixel_distance=5)

    df = main_window.track_df.sort_values("t").reset_index(drop=True)
    assert_column_values(df, "label_id_m1", [1, 1])
    assert_column_values(df, "alt_label_id_m1", [2, 0])
    assert_column_with_nan(df, "alt_dist_px_m1", [4.0, np.nan])
    assert_column_values(df, "alt_label_id_m2", [0, 0])
    assert_column_with_nan(df, "alt_dist_px_m2", [np.nan, np.nan])


def test_quantify_ignores_a_second_mask_beyond_the_threshold(
    tmp_path, no_progress
):
    main_window = make_fake_main_window(tmp_path)
    main_window.labels = two_object_labels()

    quantify(main_window, progress_cb=no_progress, max_pixel_distance=3)

    df = main_window.track_df.sort_values("t").reset_index(drop=True)
    assert_column_values(df, "alt_label_id_m1", [0, 0])
    assert_column_with_nan(df, "alt_dist_px_m1", [np.nan, np.nan])


def test_quantify_leaves_the_assignment_unchanged_by_the_candidate_search(
    tmp_path, no_progress
):
    plain = make_fake_main_window(tmp_path / "plain")
    crowded = make_fake_main_window(tmp_path / "crowded")
    crowded.labels = two_object_labels()

    quantify(plain, progress_cb=no_progress, max_pixel_distance=5)
    quantify(crowded, progress_cb=no_progress, max_pixel_distance=5)

    for column in ("label_id_m1", "nn_dist_px_m1", "AreaMorphologyM1"):
        assert (
            plain.track_df[column].tolist()
            == crowded.track_df[column].tolist()
        )


def test_quantify_saves_the_candidate_columns_in_the_csv(
    tmp_path, no_progress
):
    main_window = make_fake_main_window(tmp_path)
    main_window.labels = two_object_labels()

    quantify(main_window, progress_cb=no_progress, max_pixel_distance=5)

    csvs = list(Path(main_window.folder).rglob("*.csv"))
    saved = [pd.read_csv(path) for path in csvs]
    with_alt = [df for df in saved if "alt_label_id_m1" in df.columns]
    assert with_alt, "no saved CSV carries the candidate columns"
    assert sorted(with_alt[0]["alt_label_id_m1"].tolist()) == [0, 2]


# Assignment strategies: "legacy" and "sorted_pairs"
#
# The chain of conflicts, threshold 10:
#   A is 3 px from X, B is 5 px from X and 9 px from Y, C is 7 px from Y.
# Legacy: A->X, then B falls back to Y and C is left without a mask.
# Sorted pairs: A->X, C->Y, and B (the longest pair) is left without a mask.
CHAIN_OBJECTS = xy((0, 0), (-5, 9))  # X, Y
CHAIN_TRACKS = xy((3, 0), (-5, 0), (-5, 16))  # A, B, C
CHAIN_THRESHOLD = 10.0


def test_the_default_strategy_is_sorted_pairs():
    default = greedy_nearest_assign(CHAIN_TRACKS, CHAIN_OBJECTS, 10.0)
    sorted_pairs = greedy_nearest_assign(
        CHAIN_TRACKS, CHAIN_OBJECTS, 10.0, strategy="sorted_pairs"
    )

    assert default[0].tolist() == sorted_pairs[0].tolist()
    assert DEFAULT_MATCH_STRATEGY == "sorted_pairs"


def test_legacy_gives_the_contested_object_to_the_wrong_track():
    assigned, dist = greedy_nearest_assign(
        CHAIN_TRACKS, CHAIN_OBJECTS, CHAIN_THRESHOLD, strategy="legacy"
    )

    assert assigned.tolist() == [0, 1, -1]  # A->X, B->Y, C unmatched
    assert dist[1] == 9.0


def test_sorted_pairs_resolves_the_chain_of_conflicts():
    assigned, dist = greedy_nearest_assign(
        CHAIN_TRACKS, CHAIN_OBJECTS, CHAIN_THRESHOLD, strategy="sorted_pairs"
    )

    assert assigned.tolist() == [0, -1, 1]  # A->X, B unmatched, C->Y
    assert dist[0] == 3.0
    assert dist[2] == 7.0
    assert np.isinf(dist[1])


def test_every_strategy_agrees_when_nothing_competes():
    tracks = xy((0, 0), (20, 0), (40, 0))
    objects = xy((1, 0), (21, 0), (80, 0))

    results = [
        greedy_nearest_assign(tracks, objects, 5.0, strategy=name)[0].tolist()
        for name in MATCH_STRATEGIES
    ]

    assert results == [[0, 1, -1]] * len(MATCH_STRATEGIES)


def test_sorted_pairs_breaks_ties_by_track_then_object_index():
    tracks = xy((0, 0), (2, 0))
    objects = xy((1, 0))  # 1 px from both tracks

    assigned, _ = greedy_nearest_assign(
        tracks, objects, 5.0, strategy="sorted_pairs"
    )

    assert assigned.tolist() == [0, -1]

    tracks = xy((1, 1))
    objects = xy((0, 0), (2, 0), (0, 2))  # all equally far
    assigned, _ = greedy_nearest_assign(
        tracks, objects, 5.0, strategy="sorted_pairs"
    )

    assert assigned.tolist() == [0]


def test_sorted_pairs_never_reuses_an_object_and_respects_the_threshold():
    rng = np.random.default_rng(7)
    for _ in range(50):
        tracks = rng.uniform(0, 30, size=(rng.integers(1, 9), 2))
        objects = rng.uniform(0, 30, size=(rng.integers(1, 9), 2))

        assigned, dist = greedy_nearest_assign(
            tracks, objects, 10.0, strategy="sorted_pairs"
        )

        used = assigned[assigned >= 0]
        assert len(used) == len(set(used.tolist()))
        assert (dist[assigned >= 0] <= 10.0).all()
        assert np.isinf(dist[assigned < 0]).all()


def test_sorted_pairs_is_no_worse_than_legacy_on_total_matches():
    """Sorted pairs never matches fewer tracks than legacy on the chain."""
    counts = {
        name: int(
            (
                greedy_nearest_assign(
                    CHAIN_TRACKS, CHAIN_OBJECTS, CHAIN_THRESHOLD, strategy=name
                )[0]
                >= 0
            ).sum()
        )
        for name in MATCH_STRATEGIES
    }

    assert counts["sorted_pairs"] >= counts["legacy"]


@pytest.mark.parametrize("strategy", MATCH_STRATEGIES)
def test_strategy_has_no_effect_without_one_to_one(strategy):
    assigned, _ = greedy_nearest_assign(
        CHAIN_TRACKS,
        CHAIN_OBJECTS,
        CHAIN_THRESHOLD,
        one_to_one=False,
        strategy=strategy,
    )

    assert assigned.tolist() == [0, 0, 1]


def test_an_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="Unknown match strategy"):
        greedy_nearest_assign(
            CHAIN_TRACKS, CHAIN_OBJECTS, 10.0, strategy="nearest"
        )


def chain_frames():
    """The chain of conflicts as the dataframes `assign_objects_to_tracks` uses."""
    naming = FeatureNaming.for_config(["C01"], basic=False)
    tracks = pd.DataFrame(
        {
            "t": [0, 0, 0],
            "XMorphology": CHAIN_TRACKS[:, 0],
            "YMorphology": CHAIN_TRACKS[:, 1],
        }
    )
    tracks = init_mask_channel_columns(tracks, [1], naming)
    objects = pd.DataFrame(
        {
            "t": [0, 0],
            "__mask_idx__": [1, 1],
            "XMorphology": CHAIN_OBJECTS[:, 0],
            "YMorphology": CHAIN_OBJECTS[:, 1],
            "label_id_m1": [1, 2],
        }
    )
    return tracks, objects, naming


def run_chain(strategy: str) -> pd.DataFrame:
    tracks, objects, naming = chain_frames()
    return assign_objects_to_tracks(
        tracks,
        objects,
        [1],
        naming,
        max_pixel_distance=CHAIN_THRESHOLD,
        one_to_one=True,
        strategy=strategy,
    )


def test_assign_objects_to_tracks_writes_the_labels_of_each_strategy():
    legacy = run_chain("legacy")
    sorted_pairs = run_chain("sorted_pairs")

    assert legacy["label_id_m1"].tolist() == [1, 2, 0]
    assert sorted_pairs["label_id_m1"].tolist() == [1, 0, 2]
    assert sorted_pairs["nn_dist_px_m1"].tolist()[0] == 3.0
    assert sorted_pairs["nn_dist_px_m1"].tolist()[2] == 7.0
    assert np.isnan(sorted_pairs["nn_dist_px_m1"].tolist()[1])


def test_the_candidate_search_follows_the_assignment_of_either_strategy():
    """B is 5 px from X and 9 px from Y, so it always sees both objects.

    Legacy assigns B to Y, so its candidate is X. Sorted pairs leaves B
    unmatched; X is still its nearest object within the threshold, even
    though A claimed it.
    """
    legacy = run_chain("legacy").iloc[1]
    sorted_pairs = run_chain("sorted_pairs").iloc[1]

    assert legacy["label_id_m1"] == 2
    assert legacy["alt_label_id_m1"] == 1
    assert legacy["alt_dist_px_m1"] == 5.0

    assert sorted_pairs["label_id_m1"] == 0
    assert sorted_pairs["alt_label_id_m1"] == 1
    assert sorted_pairs["alt_dist_px_m1"] == 5.0


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        ("legacy", "2 under sorted_pairs, 2 under optimal"),
        ("sorted_pairs", "2 under legacy, 0 under optimal"),
        ("optimal", "2 under legacy, 0 under sorted_pairs"),
    ],
)
def test_assignment_logs_how_many_tracks_each_other_strategy_would_change(
    strategy, expected, caplog, monkeypatch
):
    monkeypatch.setenv(COMPARE_MATCHING_ENV, "1")
    with caplog.at_level("INFO", logger="SECQUOIA.core.quantification"):
        run_chain(strategy)

    messages = [r.getMessage() for r in caplog.records]
    assert any(
        f"[matching] strategy={strategy}: track assignments that would "
        f"differ: {expected}" in message
        for message in messages
    ), messages


def test_the_strategies_are_not_compared_unless_the_switch_is_on(
    caplog, monkeypatch
):
    monkeypatch.delenv(COMPARE_MATCHING_ENV, raising=False)
    with caplog.at_level("INFO", logger="SECQUOIA.core.quantification"):
        run_chain("sorted_pairs")

    assert "would differ" not in caplog.text


@pytest.mark.parametrize("value", ["1", "true", " Yes ", "ON"])
def test_the_compare_switch_accepts_common_spellings(value, monkeypatch):
    monkeypatch.setenv(COMPARE_MATCHING_ENV, value)

    assert compare_matching_enabled()


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
def test_the_compare_switch_is_off_for_anything_else(value, monkeypatch):
    monkeypatch.setenv(COMPARE_MATCHING_ENV, value)

    assert not compare_matching_enabled()


def test_quantify_defaults_to_sorted_pairs_and_logs_the_strategy(
    tmp_path, no_progress, caplog, monkeypatch
):
    monkeypatch.delenv(MATCH_STRATEGY_ENV, raising=False)
    main_window = make_fake_main_window(tmp_path)

    with caplog.at_level("INFO", logger="SECQUOIA.core.quantification"):
        quantify(main_window, progress_cb=no_progress, max_pixel_distance=5)

    assert "match_strategy=sorted_pairs" in caplog.text


def test_quantify_accepts_sorted_pairs_and_matches_the_simple_dataset(
    tmp_path, no_progress, caplog
):
    legacy = make_fake_main_window(tmp_path / "legacy")
    sorted_pairs = make_fake_main_window(tmp_path / "sorted_pairs")

    quantify(
        legacy,
        progress_cb=no_progress,
        max_pixel_distance=5,
        match_strategy="legacy",
    )
    with caplog.at_level("INFO", logger="SECQUOIA.core.quantification"):
        quantify(
            sorted_pairs,
            progress_cb=no_progress,
            max_pixel_distance=5,
            match_strategy="sorted_pairs",
        )

    assert "match_strategy=sorted_pairs" in caplog.text
    for column in ("label_id_m1", "nn_dist_px_m1", "AreaMorphologyM1"):
        assert (
            legacy.track_df[column].tolist()
            == sorted_pairs.track_df[column].tolist()
        )


# Choosing the strategy for a run of the app
def test_the_strategy_defaults_to_sorted_pairs(monkeypatch):
    monkeypatch.delenv(MATCH_STRATEGY_ENV, raising=False)

    assert resolve_match_strategy() == "sorted_pairs"


def test_the_environment_variable_picks_the_strategy(monkeypatch):
    monkeypatch.setenv(MATCH_STRATEGY_ENV, "legacy")

    assert resolve_match_strategy() == "legacy"


def test_the_environment_variable_is_forgiving_about_case_and_spaces(
    monkeypatch,
):
    monkeypatch.setenv(MATCH_STRATEGY_ENV, "  Optimal ")

    assert resolve_match_strategy() == "optimal"


def test_an_empty_environment_variable_means_the_default(monkeypatch):
    monkeypatch.setenv(MATCH_STRATEGY_ENV, "")

    assert resolve_match_strategy() == "sorted_pairs"


def test_an_explicit_strategy_beats_the_environment(monkeypatch):
    monkeypatch.setenv(MATCH_STRATEGY_ENV, "optimal")

    assert resolve_match_strategy("legacy") == "legacy"


def test_a_mistyped_strategy_is_an_error_not_a_silent_default(monkeypatch):
    monkeypatch.setenv(MATCH_STRATEGY_ENV, "sorted-pairs")

    with pytest.raises(ValueError, match="Unknown match strategy"):
        resolve_match_strategy()


def test_quantify_follows_the_environment_variable(
    tmp_path, no_progress, caplog, monkeypatch
):
    monkeypatch.setenv(MATCH_STRATEGY_ENV, "legacy")
    main_window = make_fake_main_window(tmp_path)

    with caplog.at_level("INFO", logger="SECQUOIA.core.quantification"):
        quantify(main_window, progress_cb=no_progress, max_pixel_distance=5)

    assert "match_strategy=legacy" in caplog.text


def test_quantify_stops_early_on_a_mistyped_strategy(
    tmp_path, no_progress, monkeypatch
):
    monkeypatch.setenv(MATCH_STRATEGY_ENV, "nearest")
    main_window = make_fake_main_window(tmp_path)

    with pytest.raises(ValueError, match="Unknown match strategy"):
        quantify(main_window, progress_cb=no_progress, max_pixel_distance=5)

    assert "label_id_m1" not in main_window.track_df.columns


# The optimal strategy (Hungarian method)
#
# X=(0,0) and Y=(9,0), tolerance 6. A=(4,0) is 4 from X and 5 from Y; B=(-5,0)
# is 5 from X and 14 from Y, so only X is within reach of B. Taking the shortest
# pair first (A-X) leaves B with nothing; giving A the farther Y matches both.
OPT_OBJECTS = xy((0, 0), (9, 0))
OPT_TRACKS = xy((4, 0), (-5, 0))
OPT_THRESHOLD = 6.0


def test_optimal_matches_more_tracks_than_the_greedy_strategies():
    results = {
        name: greedy_nearest_assign(
            OPT_TRACKS, OPT_OBJECTS, OPT_THRESHOLD, strategy=name
        )[0].tolist()
        for name in MATCH_STRATEGIES
    }

    assert results["legacy"] == [0, -1]
    assert results["sorted_pairs"] == [0, -1]
    assert results["optimal"] == [1, 0]


def test_optimal_resolves_the_chain_of_conflicts_like_sorted_pairs():
    assigned, dist = greedy_nearest_assign(
        CHAIN_TRACKS, CHAIN_OBJECTS, CHAIN_THRESHOLD, strategy="optimal"
    )

    assert assigned.tolist() == [0, -1, 1]  # A->X, B unmatched, C->Y
    assert dist[0] == 3.0
    assert dist[2] == 7.0


def test_optimal_never_matches_beyond_the_tolerance():
    """The user's tolerance is a hard limit, not a preference."""
    tracks = xy((0, 0), (0, 100))
    objects = xy((0, 6), (0, 130))  # the second object is 30 px from track two

    assigned, dist = greedy_nearest_assign(
        tracks, objects, 8.0, strategy="optimal"
    )

    assert assigned.tolist() == [0, -1]
    assert np.isinf(dist[1])


def test_optimal_includes_a_pair_exactly_at_the_tolerance():
    assigned, dist = greedy_nearest_assign(
        xy((0, 0)), xy((5, 0)), 5.0, strategy="optimal"
    )

    assert assigned.tolist() == [0]
    assert dist.tolist() == [5.0]


def test_optimal_reports_the_distance_of_each_assigned_pair():
    assigned, dist = greedy_nearest_assign(
        OPT_TRACKS, OPT_OBJECTS, OPT_THRESHOLD, strategy="optimal"
    )

    assert dist.tolist() == [5.0, 5.0]
    assert assigned.tolist() == [1, 0]


def test_optimal_leaves_a_track_with_a_missing_position_unmatched():
    tracks = np.array([[np.nan, np.nan], [0.0, 0.0]])

    assigned, _ = greedy_nearest_assign(
        tracks, xy((1, 0), (50, 50)), 5.0, strategy="optimal"
    )

    assert assigned.tolist() == [-1, 0]


def test_optimal_handles_empty_inputs_and_unequal_counts():
    empty = np.empty((0, 2))

    assert (
        greedy_nearest_assign(empty, xy((1, 1)), 5, strategy="optimal")[
            0
        ].tolist()
        == []
    )
    assert greedy_nearest_assign(xy((1, 1)), empty, 5, strategy="optimal")[
        0
    ].tolist() == [-1]
    more_tracks = greedy_nearest_assign(
        xy((0, 0), (1, 0), (2, 0)), xy((0, 0)), 5, strategy="optimal"
    )[0]
    assert (more_tracks >= 0).sum() == 1
    more_objects = greedy_nearest_assign(
        xy((0, 0)), xy((0, 1), (0, 2), (0, 0.5)), 5, strategy="optimal"
    )[0]
    assert more_objects.tolist() == [2]  # the nearest of the three


def brute_force_best(distances: np.ndarray, threshold: float):
    """The most matches, then the smallest total distance, by trying everything."""
    import itertools

    n_tracks, n_objects = distances.shape
    best = (0, 0.0)
    options = [*range(n_objects), -1]

    for choice in itertools.product(options, repeat=n_tracks):
        used = [j for j in choice if j >= 0]
        if len(used) != len(set(used)):
            continue
        pairs = [(i, j) for i, j in enumerate(choice) if j >= 0]
        if any(distances[i, j] > threshold for i, j in pairs):
            continue
        candidate = (len(pairs), -sum(distances[i, j] for i, j in pairs))
        if candidate > (best[0], -best[1]):
            best = (candidate[0], -candidate[1])
    return best


@pytest.mark.parametrize("seed", range(25))
def test_optimal_finds_the_best_assignment_on_small_random_cases(seed):
    rng = np.random.default_rng(seed)
    tracks = rng.uniform(0, 30, (rng.integers(1, 6), 2))
    objects = rng.uniform(0, 30, (rng.integers(1, 6), 2))
    threshold = 12.0

    assigned, dist = greedy_nearest_assign(
        tracks, objects, threshold, strategy="optimal"
    )

    matches = int((assigned >= 0).sum())
    total = float(dist[assigned >= 0].sum())
    distances = np.hypot(
        tracks[:, [0]] - objects[:, 0][None, :],
        tracks[:, [1]] - objects[:, 1][None, :],
    )
    best_matches, best_total = brute_force_best(distances, threshold)
    assert matches == best_matches
    assert total == pytest.approx(best_total)


@pytest.mark.parametrize("seed", range(20))
def test_optimal_is_never_worse_than_the_greedy_strategies(seed):
    rng = np.random.default_rng(100 + seed)
    tracks = rng.uniform(0, 60, (rng.integers(5, 40), 2))
    objects = rng.uniform(0, 60, (rng.integers(5, 40), 2))
    stats = {}

    for name in MATCH_STRATEGIES:
        assigned, dist = greedy_nearest_assign(
            tracks, objects, 10.0, strategy=name
        )
        used = assigned[assigned >= 0]
        assert len(used) == len(set(used.tolist()))
        assert (dist[assigned >= 0] <= 10.0).all()
        stats[name] = (
            int((assigned >= 0).sum()),
            float(dist[assigned >= 0].sum()),
        )

    assert stats["optimal"][0] >= stats["sorted_pairs"][0]
    assert stats["optimal"][0] >= stats["legacy"][0]


def test_optimal_gives_the_same_answer_every_time():
    rng = np.random.default_rng(7)
    tracks, objects = rng.uniform(0, 50, (30, 2)), rng.uniform(0, 50, (30, 2))

    first = greedy_nearest_assign(tracks, objects, 10.0, strategy="optimal")
    again = greedy_nearest_assign(tracks, objects, 10.0, strategy="optimal")

    assert first[0].tolist() == again[0].tolist()


def test_assign_objects_to_tracks_writes_the_optimal_labels():
    optimal = run_chain("optimal")

    assert optimal["label_id_m1"].tolist() == [1, 0, 2]
    assert optimal["nn_dist_px_m1"].tolist()[0] == 3.0
    assert optimal["nn_dist_px_m1"].tolist()[2] == 7.0


def test_the_candidate_columns_follow_the_optimal_assignment_too():
    """B is left without a mask, so its nearest mask (X) is its candidate."""
    row = run_chain("optimal").iloc[1]

    assert row["label_id_m1"] == 0
    assert row["alt_label_id_m1"] == 1
    assert row["alt_dist_px_m1"] == 5.0


def test_optimal_can_be_chosen_for_a_run_of_the_app(monkeypatch):
    monkeypatch.setenv(MATCH_STRATEGY_ENV, "optimal")

    assert resolve_match_strategy() == "optimal"


def test_quantify_runs_with_the_optimal_strategy(
    tmp_path, no_progress, caplog
):
    legacy = make_fake_main_window(tmp_path / "legacy")
    optimal = make_fake_main_window(tmp_path / "optimal")

    quantify(legacy, progress_cb=no_progress, max_pixel_distance=5)
    with caplog.at_level("INFO", logger="SECQUOIA.core.quantification"):
        quantify(
            optimal,
            progress_cb=no_progress,
            max_pixel_distance=5,
            match_strategy="optimal",
        )

    assert "match_strategy=optimal" in caplog.text
    for column in ("label_id_m1", "nn_dist_px_m1", "AreaMorphologyM1"):
        assert (
            legacy.track_df[column].tolist()
            == optimal.track_df[column].tolist()
        )
