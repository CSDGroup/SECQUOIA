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
