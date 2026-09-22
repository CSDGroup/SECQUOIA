"""Characterisation tests for ``update_mask_measurements_for_current_track``.

Three groups:

``Phase A`` - golden values and guard paths.
    What the method writes for a known synthetic edit, and that every early
    return leaves ``track_df`` untouched.

``Phase B`` - equivalence with the batch pipeline.
    ``quantify()`` and this method are two implementations of the same
    measurement. On identical input they must produce identical columns.

``Phase C`` - parity in the awkward cases.
    Sparse acquisition, all-zero frames, centroid updates: places where the
    interactive path and ``quantify()`` could plausibly drift apart.

The artificial dataset from ``test_quantify.py``:
one position, one cell, two time points, channel ``w01``, and two masks
(mask 1 is 3x3 -> area 9, mask 2 is 2x2 -> area 4).
"""

from __future__ import annotations

import importlib
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from conftest import find_module_name, make_fake_labels, make_fake_main_window

from SECQUOIA.core.outlier_detection import apply_close_mask_detection
from SECQUOIA.core.outlier_detection.close_masks import (
    CLOSE_MASK_FLAG,
    FLAG_FLAGGED,
    FLAG_OK,
    FLAG_REVIEWED,
)
from SECQUOIA.core.quantification import quantify

ANCHOR_X = 3
ANCHOR_Y = 3


def main_window_module():
    """Import the module holding ``MainWindow`` (pulls in napari and Qt)."""
    return importlib.import_module(find_module_name("main_window.py"))


def measurement_picking_module():
    """Import the module holding the measurement-picking collaborators."""
    return importlib.import_module(find_module_name("measurement_picking.py"))


class FakeLabelsLayer(SimpleNamespace):
    """Stand-in for a napari Labels layer: only ``name`` and ``data`` are read."""


def make_layer(name: str, data: np.ndarray) -> FakeLabelsLayer:
    return FakeLabelsLayer(name=name, data=data)


def make_edit_main_window(tmp_path) -> SimpleNamespace:
    """``make_fake_main_window`` plus the attributes the edit path needs.

    ``update_mask_measurements_for_current_track`` reads a wider slice of
    MainWindow state than ``quantify`` does. Everything added here is either
    plain data or a recording stub for a Qt-side effect.
    """
    main_window = make_fake_main_window(tmp_path)

    main_window.current_TrackNumber_plot = 1
    main_window.current_ident_index = 0
    main_window.unique_ids = ["fake-001"]
    main_window.image_present = {"w01": np.array([True, True])}

    main_window.calls = []
    main_window._recompute_derived_for_row = (
        lambda *a, **k: main_window.calls.append(("recompute_derived", a, k))
    )
    main_window.update_tracks_for_ids = lambda ids: main_window.calls.append(
        ("update_tracks_for_ids", tuple(ids))
    )
    main_window._set_inspected_for_ident = (
        lambda ident, v: main_window.calls.append(("set_inspected", ident, v))
    )
    main_window._find_tree_item_for = lambda ident, _: None
    main_window._sync_children_visuals_from_df = lambda item: None

    return main_window


@pytest.fixture
def edit_main_window(tmp_path):
    return make_edit_main_window(tmp_path)


@pytest.fixture
def patch_collaborators(monkeypatch):
    module = main_window_module()
    mp_module = measurement_picking_module()
    monkeypatch.setattr(
        mp_module, "resolve_current_ident", lambda mw: "fake-001", raising=True
    )
    monkeypatch.setattr(
        mp_module,
        "apply_outlier_selection_for_current_point",
        lambda mw: None,
        raising=True,
    )
    monkeypatch.setattr(
        mp_module, "update_plot", lambda mw: None, raising=True
    )
    return module


def run_update(main_window, layer, module, *, source: str = "test") -> None:
    """Call the method unbound, with the fake standing in for ``self``."""
    if not hasattr(main_window, "_refresh_after_measurement_update"):
        main_window._refresh_after_measurement_update = (
            module.MainWindow._refresh_after_measurement_update.__get__(
                main_window
            )
        )

    module.MainWindow.update_mask_measurements_for_current_track(
        main_window, layer, source=source
    )


def values_equal(a, b) -> bool:
    """NA-safe scalar comparison.

    Nullable-integer columns (``label_id_m*`` is Int64) yield ``pd.NA`` from
    ``==``, and ``bool(pd.NA)`` raises. Treat NA/NaN as equal to itself.
    """
    a_na, b_na = pd.isna(a), pd.isna(b)
    if a_na or b_na:
        return bool(a_na and b_na)
    return bool(a == b)


# Phase A
@pytest.mark.gui
def test_update_writes_expected_measurements_for_mask_1(
    edit_main_window, patch_collaborators
):
    """A no-op 'edit' on mask 1 at t=0 writes area 9, mean 10, sum 90.

    These are the same numbers ``test_quantify`` asserts for mask 1 at t=0,
    which is the point: the interactive path must agree with the pipeline.
    """
    mask_1, _mask_2 = make_fake_labels()
    layer = make_layer("Segmentation1", mask_1)
    edit_main_window.current_time_index = 0

    run_update(edit_main_window, layer, patch_collaborators)

    df = edit_main_window.track_df
    row = df.index[df["t"] == 0][0]

    assert int(df.at[row, "label_id_m1"]) == 1
    assert float(df.at[row, "AreaMorphologyM1"]) == pytest.approx(9.0)
    assert float(df.at[row, "MeanNoBgCorrectedCh01M1"]) == pytest.approx(10.0)
    assert float(df.at[row, "SumNoBgCorrectedCh01M1"]) == pytest.approx(90.0)


@pytest.mark.gui
def test_update_writes_expected_measurements_for_mask_2(
    edit_main_window, patch_collaborators
):
    """The smaller mask: area 4 and sum 40, same mean."""
    _mask_1, mask_2 = make_fake_labels()
    layer = make_layer("Segmentation2", mask_2)
    edit_main_window.current_time_index = 0

    run_update(edit_main_window, layer, patch_collaborators)

    df = edit_main_window.track_df
    row = df.index[df["t"] == 0][0]

    assert float(df.at[row, "AreaMorphologyM2"]) == pytest.approx(4.0)
    assert float(df.at[row, "SumNoBgCorrectedCh01M2"]) == pytest.approx(40.0)


@pytest.mark.gui
def test_update_only_touches_the_current_timepoint(
    edit_main_window, patch_collaborators
):
    """t=1 must be left as NaN when the edit happened at t=0."""
    mask_1, _ = make_fake_labels()
    edit_main_window.current_time_index = 0

    run_update(
        edit_main_window,
        make_layer("Segmentation1", mask_1),
        patch_collaborators,
    )

    df = edit_main_window.track_df
    other = df.index[df["t"] == 1][0]
    assert pd.isna(df.at[other, "AreaMorphologyM1"])


@pytest.mark.gui
def test_update_touches_no_other_mask(edit_main_window, patch_collaborators):
    """Editing mask 1 must not create or write mask 2 columns."""
    mask_1, _ = make_fake_labels()
    edit_main_window.current_time_index = 0

    run_update(
        edit_main_window,
        make_layer("Segmentation1", mask_1),
        patch_collaborators,
    )

    df = edit_main_window.track_df
    assert "AreaMorphologyM2" not in df.columns
    assert "SumNoBgCorrectedCh01M2" not in df.columns


@pytest.mark.gui
def test_update_is_idempotent(edit_main_window, patch_collaborators):
    """Running twice on unchanged labels must not change any value."""
    mask_1, _ = make_fake_labels()
    layer = make_layer("Segmentation1", mask_1)
    edit_main_window.current_time_index = 0

    run_update(edit_main_window, layer, patch_collaborators)
    first = edit_main_window.track_df.copy()
    run_update(edit_main_window, layer, patch_collaborators)
    second = edit_main_window.track_df

    pd.testing.assert_frame_equal(first, second, check_dtype=True)


@pytest.mark.gui
def test_update_reports_changed_columns(
    edit_main_window, patch_collaborators, caplog
):
    """The diff log is the operator's only feedback; pin its shape."""
    mask_1, _ = make_fake_labels()
    edit_main_window.current_time_index = 0

    with caplog.at_level(logging.INFO, logger="SECQUOIA"):
        run_update(
            edit_main_window,
            make_layer("Segmentation1", mask_1),
            patch_collaborators,
            source="paint",
        )

    out = caplog.text
    assert "(paint)" in out
    assert "Identification='fake-001'" in out
    assert "TrackNumber=1" in out
    assert "m=1" in out
    assert "label=1" in out
    assert "AreaMorphologyM1" in out


@pytest.mark.gui
def test_update_refreshes_downstream_state(
    edit_main_window, patch_collaborators
):
    """Downstream state must be refreshed, not just track_df."""
    mask_1, _ = make_fake_labels()
    edit_main_window.current_time_index = 0

    run_update(
        edit_main_window,
        make_layer("Segmentation1", mask_1),
        patch_collaborators,
    )

    names = [c[0] for c in edit_main_window.calls]
    assert "recompute_derived" in names
    assert "update_tracks_for_ids" in names
    assert ("set_inspected", "fake-001", 2) in edit_main_window.calls
    assert (edit_main_window.filtered_df["Position"] == 1).all()
    assert edit_main_window.df_subset is not None


# Phase A - guard paths
@pytest.mark.gui
@pytest.mark.parametrize(
    ("mutate", "expected_log"),
    [
        pytest.param(
            lambda mw: setattr(mw, "track_df", pd.DataFrame()),
            "track_df missing or empty",
            id="empty-track-df",
        ),
        pytest.param(
            lambda mw: setattr(mw, "current_time_index", 99),
            "No 2D labels",
            id="time-index-out-of-range",
        ),
    ],
)
def test_guard_paths_log_and_return(
    edit_main_window, patch_collaborators, caplog, mutate, expected_log
):
    mask_1, _ = make_fake_labels()
    mutate(edit_main_window)
    before = edit_main_window.track_df.copy()

    with caplog.at_level(logging.WARNING, logger="SECQUOIA"):
        run_update(
            edit_main_window,
            make_layer("Segmentation1", mask_1),
            patch_collaborators,
        )

    assert expected_log in caplog.text
    pd.testing.assert_frame_equal(before, edit_main_window.track_df)


@pytest.mark.gui
def test_unparseable_layer_name_aborts(
    edit_main_window, patch_collaborators, caplog
):
    """A layer name with no digits yields no mask index."""
    mask_1, _ = make_fake_labels()
    before = edit_main_window.track_df.copy()

    with caplog.at_level(logging.WARNING, logger="SECQUOIA"):
        run_update(
            edit_main_window,
            make_layer("Nucleus", mask_1),
            patch_collaborators,
        )

    assert "Could not infer mask index" in caplog.text
    pd.testing.assert_frame_equal(before, edit_main_window.track_df)


@pytest.mark.gui
def test_anchor_outside_the_frame_aborts(
    edit_main_window, patch_collaborators, caplog
):
    """An anchor beyond the label bounds must not index out of range."""
    mask_1, _ = make_fake_labels()
    edit_main_window.track_df.loc[:, "XMorphology"] = 999.0
    before = edit_main_window.track_df.copy()

    with caplog.at_level(logging.WARNING, logger="SECQUOIA"):
        run_update(
            edit_main_window,
            make_layer("Segmentation1", mask_1),
            patch_collaborators,
        )

    assert "Invalid anchor" in caplog.text
    pd.testing.assert_frame_equal(before, edit_main_window.track_df)


@pytest.mark.gui
def test_missing_anchor_falls_back_to_per_mask_column(
    edit_main_window, patch_collaborators
):
    """With XMorphology NaN, the XMorphologyM1 column must be used instead."""
    df = edit_main_window.track_df
    df["XMorphologyM1"] = float(ANCHOR_X)
    df["YMorphologyM1"] = float(ANCHOR_Y)
    df.loc[:, "XMorphology"] = np.nan
    df.loc[:, "YMorphology"] = np.nan

    mask_1, _ = make_fake_labels()
    edit_main_window.current_time_index = 0

    run_update(
        edit_main_window,
        make_layer("Segmentation1", mask_1),
        patch_collaborators,
    )

    row = edit_main_window.track_df.index[edit_main_window.track_df["t"] == 0][
        0
    ]
    assert float(
        edit_main_window.track_df.at[row, "AreaMorphologyM1"]
    ) == pytest.approx(9.0)


@pytest.mark.gui
def test_anchor_on_background_writes_nothing_measurable(
    edit_main_window, patch_collaborators
):
    """label_id == 0 makes measure_single_object return None.

    The method still writes the (empty) result, so the columns appear as NaN
    rather than being left absent.
    """
    mask_1, _ = make_fake_labels()
    edit_main_window.track_df.loc[:, "XMorphology"] = 0.0
    edit_main_window.track_df.loc[:, "YMorphology"] = 0.0
    edit_main_window.current_time_index = 0

    run_update(
        edit_main_window,
        make_layer("Segmentation1", mask_1),
        patch_collaborators,
    )

    row = edit_main_window.track_df.index[edit_main_window.track_df["t"] == 0][
        0
    ]
    assert int(edit_main_window.track_df.at[row, "label_id_m1"]) == 0
    assert pd.isna(edit_main_window.track_df.at[row, "AreaMorphologyM1"])


@pytest.mark.gui
def test_ambiguous_track_number_is_inferred_from_the_dataframe(
    edit_main_window, patch_collaborators
):
    """With current_TrackNumber_plot unset, a unique match is inferred."""
    edit_main_window.current_TrackNumber_plot = None
    mask_1, _ = make_fake_labels()
    edit_main_window.current_time_index = 0

    run_update(
        edit_main_window,
        make_layer("Segmentation1", mask_1),
        patch_collaborators,
    )

    row = edit_main_window.track_df.index[edit_main_window.track_df["t"] == 0][
        0
    ]
    assert float(
        edit_main_window.track_df.at[row, "AreaMorphologyM1"]
    ) == pytest.approx(9.0)


# Phase B - equivalence with the batch pipeline
@pytest.mark.gui
def test_update_matches_quantify_on_unmodified_labels(
    tmp_path, no_progress, patch_collaborators
):
    """Run ``quantify`` over the dataset, then run the interactive update on an
    *unmodified* frame.
    """
    main_window = make_edit_main_window(tmp_path)
    quantify(main_window, progress_cb=no_progress, max_pixel_distance=5)

    df = main_window.track_df
    row = df.index[df["t"] == 0][0]
    measured_cols = [
        c for c in df.columns if c.endswith("M1") or c == "label_id_m1"
    ]
    before = df.loc[row, measured_cols].copy()

    mask_1, _ = make_fake_labels()
    main_window.current_time_index = 0
    run_update(
        main_window, make_layer("Segmentation1", mask_1), patch_collaborators
    )

    after = main_window.track_df.loc[row, measured_cols]

    mismatches = {
        col: (before[col], after[col])
        for col in measured_cols
        if not values_equal(before[col], after[col])
    }
    assert not mismatches, (
        "quantify() and update_mask_measurements_for_current_track disagree "
        f"on {len(mismatches)} column(s):\n"
        + "\n".join(
            f"  {c}: quantify={b!r} update={a!r}"
            for c, (b, a) in mismatches.items()
        )
    )


@pytest.mark.gui
def test_update_matches_quantify_after_a_real_edit(
    tmp_path, no_progress, patch_collaborators
):
    """Shrink the object by one row of pixels, then compare against a re-quantify."""
    edited_mask, _ = make_fake_labels()
    edited_mask[0, 4, :] = 0  # 3x3 -> 2x3, area 9 -> 6

    # Reference: full pipeline on the edited labels.
    reference = make_edit_main_window(tmp_path / "reference")
    reference.labels = [edited_mask.copy(), make_fake_labels()[1]]
    reference.masks = reference.labels
    quantify(reference, progress_cb=no_progress, max_pixel_distance=5)
    ref_df = reference.track_df
    ref_row = ref_df.index[ref_df["t"] == 0][0]

    # Interactive: quantify on the original, then update after the edit.
    incremental = make_edit_main_window(tmp_path / "incremental")
    quantify(incremental, progress_cb=no_progress, max_pixel_distance=5)
    incremental.current_time_index = 0
    run_update(
        incremental,
        make_layer("Segmentation1", edited_mask),
        patch_collaborators,
    )
    inc_df = incremental.track_df
    inc_row = inc_df.index[inc_df["t"] == 0][0]

    for col in (
        "AreaMorphologyM1",
        "MeanNoBgCorrectedCh01M1",
        "SumNoBgCorrectedCh01M1",
    ):
        assert float(inc_df.at[inc_row, col]) == pytest.approx(
            float(ref_df.at[ref_row, col])
        ), f"{col} diverged after an incremental update"

    # Sanity: the edit really did change the area.
    assert float(inc_df.at[inc_row, "AreaMorphologyM1"]) == pytest.approx(6.0)


@pytest.mark.gui
def test_label_id_column_keeps_its_integer_dtype(
    tmp_path, no_progress, patch_collaborators
):
    """``quantify`` leaves label_id_m1 as Int64; an update must not widen it."""
    main_window = make_edit_main_window(tmp_path)
    quantify(main_window, progress_cb=no_progress, max_pixel_distance=5)
    dtype_before = main_window.track_df["label_id_m1"].dtype

    mask_1, _ = make_fake_labels()
    main_window.current_time_index = 0
    run_update(
        main_window, make_layer("Segmentation1", mask_1), patch_collaborators
    )

    assert main_window.track_df["label_id_m1"].dtype == dtype_before


# Phase C - Parity with the batch pipeline
@pytest.mark.gui
def test_all_zero_frame_flagged_present_is_measured_like_the_pipeline(
    tmp_path, no_progress, patch_collaborators
):
    """A dark frame flagged as acquired is measured by both paths.

    With a usable ``image_present`` flag array, ``_frame_missing`` honours it
    and never reaches its all-zero fallback  which only fires when
    ``image_present`` is absent.
    """
    main_window = make_edit_main_window(tmp_path)
    main_window.images["w01"][0, :, :] = 0
    main_window.image_present = {"w01": np.array([True, True])}

    quantify(main_window, progress_cb=no_progress, max_pixel_distance=5)
    df = main_window.track_df
    row = df.index[df["t"] == 0][0]
    expected = df.at[row, "MeanNoBgCorrectedCh01M1"]

    mask_1, _ = make_fake_labels()
    main_window.current_time_index = 0
    run_update(
        main_window, make_layer("Segmentation1", mask_1), patch_collaborators
    )

    assert float(
        main_window.track_df.at[row, "MeanNoBgCorrectedCh01M1"]
    ) == pytest.approx(float(expected))


@pytest.mark.gui
def test_update_does_not_erase_the_per_mask_centroid(
    tmp_path, no_progress, patch_collaborators
):
    """The per mask centroid columns must survive an edit."""
    main_window = make_edit_main_window(tmp_path)
    quantify(main_window, progress_cb=no_progress, max_pixel_distance=5)

    df = main_window.track_df
    row = df.index[df["t"] == 0][0]
    assert not pd.isna(
        df.at[row, "XMorphologyM1"]
    ), "precondition: quantify sets it"

    mask_1, _ = make_fake_labels()
    main_window.current_time_index = 0
    run_update(
        main_window, make_layer("Segmentation1", mask_1), patch_collaborators
    )

    assert not pd.isna(main_window.track_df.at[row, "XMorphologyM1"])
    assert not pd.isna(main_window.track_df.at[row, "YMorphologyM1"])


@pytest.mark.gui
def test_update_writes_the_new_centroid_after_a_move(
    tmp_path, no_progress, patch_collaborators
):
    """Painting the object one pixel down must move its recorded centroid."""
    main_window = make_edit_main_window(tmp_path)
    quantify(main_window, progress_cb=no_progress, max_pixel_distance=5)

    moved, _ = make_fake_labels()
    moved[0] = 0
    moved[0, 3:6, 2:5] = 1  # centroid y: 3 -> 4, anchor (3, 3) still inside

    row = main_window.track_df.index[main_window.track_df["t"] == 0][0]
    main_window.current_time_index = 0
    run_update(
        main_window, make_layer("Segmentation1", moved), patch_collaborators
    )

    assert float(
        main_window.track_df.at[row, "YMorphologyM1"]
    ) == pytest.approx(4.0)


@pytest.mark.gui
def test_non_finite_metrics_are_zero_filled_like_the_pipeline(
    tmp_path, no_progress, patch_collaborators
):
    """An all-zero frame makes CV = std/mean non-finite."""
    main_window = make_edit_main_window(tmp_path)
    main_window.images["w01"][0, :, :] = 0
    main_window.image_present = {"w01": np.array([True, True])}

    quantify(main_window, progress_cb=no_progress, max_pixel_distance=5)
    df = main_window.track_df
    row = df.index[df["t"] == 0][0]
    expected = df.at[row, "CVNoBgCorrectedCh01M1"]
    assert float(expected) == pytest.approx(
        0.0
    ), "precondition: pipeline zero-filled"

    mask_1, _ = make_fake_labels()
    main_window.current_time_index = 0
    run_update(
        main_window, make_layer("Segmentation1", mask_1), patch_collaborators
    )

    assert float(
        main_window.track_df.at[row, "CVNoBgCorrectedCh01M1"]
    ) == pytest.approx(0.0)


@pytest.mark.gui
def test_intensity_is_nan_when_the_anchor_hits_no_object(
    tmp_path, no_progress, patch_collaborators
):
    """Clicking outside any mask must write NaN, never a zero-filled number."""
    main_window = make_edit_main_window(tmp_path)
    quantify(main_window, progress_cb=no_progress, max_pixel_distance=5)

    df = main_window.track_df
    row = df.index[df["t"] == 0][0]
    df.loc[:, "XMorphology"] = 0.0  # background pixel
    df.loc[:, "YMorphology"] = 0.0

    mask_1, _ = make_fake_labels()
    main_window.current_time_index = 0
    run_update(
        main_window, make_layer("Segmentation1", mask_1), patch_collaborators
    )

    after = main_window.track_df
    assert int(after.at[row, "label_id_m1"]) == 0
    for col in (
        "MeanNoBgCorrectedCh01M1",
        "SumNoBgCorrectedCh01M1",
        "CVNoBgCorrectedCh01M1",
    ):
        assert pd.isna(
            after.at[row, col]
        ), f"{col} should be NaN, not zero-filled"


@pytest.mark.gui
def test_unacquired_frame_is_nan_in_every_metric(
    edit_main_window, patch_collaborators
):
    """Sparse acquisition: a channel not imaged at t must be NaN, not 0."""

    edit_main_window.image_present = {"w01": np.array([True, False])}
    edit_main_window.current_time_index = 1  # not acquired for w01

    mask_1, _ = make_fake_labels()
    run_update(
        edit_main_window,
        make_layer("Segmentation1", mask_1),
        patch_collaborators,
    )

    df = edit_main_window.track_df
    row = df.index[df["t"] == 1][0]

    for metric in ("Mean", "Min", "Max", "Sum", "Std", "CV"):
        column = f"{metric}NoBgCorrectedCh01M1"
        assert pd.isna(
            df.at[row, column]
        ), f"{column} must be NaN on an unacquired frame, got {df.at[row, column]!r}"

    # Morphology is channel-independent: the mask exists, so it is still measured.
    assert float(df.at[row, "AreaMorphologyM1"]) == pytest.approx(9.0)
    assert not pd.isna(df.at[row, "XMorphologyM1"])


@pytest.mark.gui
def test_acquired_frame_still_records_zero_intensity(
    edit_main_window, patch_collaborators
):
    edit_main_window.images["w01"][0, :, :] = 0
    edit_main_window.image_present = {"w01": np.array([True, True])}
    edit_main_window.current_time_index = 0

    mask_1, _ = make_fake_labels()
    run_update(
        edit_main_window,
        make_layer("Segmentation1", mask_1),
        patch_collaborators,
    )

    df = edit_main_window.track_df
    row = df.index[df["t"] == 0][0]
    assert float(df.at[row, "MeanNoBgCorrectedCh01M1"]) == pytest.approx(0.0)
    assert float(df.at[row, "SumNoBgCorrectedCh01M1"]) == pytest.approx(0.0)


@pytest.mark.gui
def test_mask_index_resolution_is_shared_between_call_sites(
    edit_main_window, patch_collaborators
):
    """One layer name, one mask index, wherever it is resolved."""
    from SECQUOIA.core.segmentation import mask_selection

    assert hasattr(
        mask_selection, "mask_index_from_layer_name"
    ), "Expected a single shared helper for mask-index resolution."
    resolve = mask_selection.mask_index_from_layer_name

    assert resolve("Segmentation1") == 1
    assert resolve("Segmentation_02") == 2
    assert resolve("seg 3") == 3
    assert resolve("Mask 1") == 1
    assert resolve("Nucleus") is None
    assert resolve(None) is None

    # And the method resolves the same layer name through that helper.
    mask_1, _ = make_fake_labels()
    edit_main_window.current_time_index = 0
    run_update(
        edit_main_window, make_layer("Mask 1", mask_1), patch_collaborators
    )

    row = edit_main_window.track_df.index[edit_main_window.track_df["t"] == 0][
        0
    ]
    assert float(
        edit_main_window.track_df.at[row, "AreaMorphologyM1"]
    ) == pytest.approx(9.0)


# Distances and second candidates after an edit
#
# Frame t=0 of mask 1 holds two neighbouring objects and two tracking points:
#
#   label 1: x 2..6, y 2..4  centroid (4, 3)   <- track fake-001 at (4, 3)
#   label 2: x 7..9, y 2..4  centroid (8, 3)   <- track fake-002 at (8, 3)
#
# The matching threshold is 5 px, so each track sees the other object 4 px away.
ROW_A = 0
ROW_B = 2


def crowded_labels() -> list[np.ndarray]:
    mask_1, mask_2 = make_fake_labels()
    mask_1[0] = 0
    mask_1[0, 2:5, 2:7] = 1
    mask_1[0, 2:5, 7:10] = 2
    return [mask_1, mask_2]


def make_crowded_main_window(tmp_path, no_progress) -> SimpleNamespace:
    """Quantify two tracks over the crowded frame, ready for an edit."""
    main_window = make_edit_main_window(tmp_path)
    main_window.labels = crowded_labels()
    tracks = main_window.track_df
    tracks.loc[tracks["t"] == 0, ["XMorphology", "YMorphology"]] = [4.0, 3.0]
    second = tracks[tracks["t"] == 0].assign(
        Identification="fake-002", XMorphology=8.0, track_id=2
    )
    main_window.track_df = pd.concat([tracks, second], ignore_index=True)
    main_window.filtered_df = main_window.track_df.copy()

    quantify(main_window, progress_cb=no_progress, max_pixel_distance=5)
    main_window.track_df = main_window.track_df.reset_index(drop=True)
    return main_window


def row_at(main_window, ident: str, t: int = 0):
    df = main_window.track_df
    return df[(df["Identification"] == ident) & (df["t"] == t)].iloc[0]


def edit(main_window, module, labels_t0: np.ndarray) -> None:
    """Run the update on mask 1, frame 0, as if `labels_t0` had just been drawn."""
    data = crowded_labels()[0]
    data[0] = labels_t0
    main_window.current_time_index = 0
    run_update(main_window, make_layer("Segmentation1", data), module)


@pytest.mark.gui
def test_quantify_and_edit_agree_on_the_crowded_frame(
    tmp_path, no_progress, patch_collaborators
):
    main_window = make_crowded_main_window(tmp_path, no_progress)
    before = row_at(main_window, "fake-001")
    assert before["label_id_m1"] == 1
    assert before["alt_label_id_m1"] == 2
    assert before["alt_dist_px_m1"] == 4.0

    edit(main_window, patch_collaborators, crowded_labels()[0][0])

    after = row_at(main_window, "fake-001")
    for column in ("nn_dist_px_m1", "alt_label_id_m1", "alt_dist_px_m1"):
        assert values_equal(before[column], after[column]), column


@pytest.mark.gui
def test_painting_updates_the_distance_to_the_assigned_mask(
    tmp_path, no_progress, patch_collaborators
):
    main_window = make_crowded_main_window(tmp_path, no_progress)
    assert row_at(main_window, "fake-001")["nn_dist_px_m1"] == 0.0

    # Grow label 1 by two columns: its centroid moves from x=4 to x=5.
    painted = crowded_labels()[0][0].copy()
    painted[2:5, 7:9] = 1
    painted[2:5, 9] = 2
    edit(main_window, patch_collaborators, painted)

    row = row_at(main_window, "fake-001")
    assert row["label_id_m1"] == 1
    assert row["nn_dist_px_m1"] == 1.0


@pytest.mark.gui
def test_painting_moves_the_candidate_distance(
    tmp_path, no_progress, patch_collaborators
):
    main_window = make_crowded_main_window(tmp_path, no_progress)

    # Label 2 shrinks to x 9 only: its centroid moves from x=8 to x=9.
    labels = crowded_labels()[0][0].copy()
    labels[2:5, 7:9] = 0
    edit(main_window, patch_collaborators, labels)

    row = row_at(main_window, "fake-001")
    assert row["alt_label_id_m1"] == 2
    assert row["alt_dist_px_m1"] == 5.0


@pytest.mark.gui
def test_erasing_the_candidate_clears_the_alt_columns(
    tmp_path, no_progress, patch_collaborators
):
    main_window = make_crowded_main_window(tmp_path, no_progress)

    labels = crowded_labels()[0][0].copy()
    labels[labels == 2] = 0
    edit(main_window, patch_collaborators, labels)

    row = row_at(main_window, "fake-001")
    assert row["alt_label_id_m1"] == 0
    assert pd.isna(row["alt_dist_px_m1"])


@pytest.mark.gui
def test_a_candidate_beyond_the_threshold_is_dropped(
    tmp_path, no_progress, patch_collaborators
):
    main_window = make_crowded_main_window(tmp_path, no_progress)

    labels = crowded_labels()[0][0].copy()
    labels[2:5, 7:9] = 0
    labels[2:5, 9] = 0
    labels[8:10, 9] = 2  # centroid (9, 8.5 -> 8): far from (4, 3)
    edit(main_window, patch_collaborators, labels)

    row = row_at(main_window, "fake-001")
    assert row["alt_label_id_m1"] == 0
    assert pd.isna(row["alt_dist_px_m1"])


@pytest.mark.gui
def test_a_candidate_below_the_minimum_mask_size_is_ignored(
    tmp_path, no_progress, patch_collaborators
):
    main_window = make_crowded_main_window(tmp_path, no_progress)
    main_window.min_mask_size = 10  # label 2 has 9 px

    edit(main_window, patch_collaborators, crowded_labels()[0][0])

    row = row_at(main_window, "fake-001")
    assert row["alt_label_id_m1"] == 0
    assert pd.isna(row["alt_dist_px_m1"])


@pytest.mark.gui
def test_right_click_swaps_assigned_and_candidate(
    tmp_path, no_progress, patch_collaborators
):
    """Moving the tracking point onto label 2 makes label 1 the candidate."""
    main_window = make_crowded_main_window(tmp_path, no_progress)
    df = main_window.track_df
    idx = df.index[(df["Identification"] == "fake-001") & (df["t"] == 0)][0]
    df.loc[idx, ["XMorphology", "YMorphology"]] = [8.0, 3.0]

    edit(main_window, patch_collaborators, crowded_labels()[0][0])

    row = row_at(main_window, "fake-001")
    assert row["label_id_m1"] == 2
    assert row["nn_dist_px_m1"] == 0.0
    assert row["alt_label_id_m1"] == 1
    assert row["alt_dist_px_m1"] == 4.0


@pytest.mark.gui
def test_edit_updates_the_candidates_of_other_tracks_at_the_same_time(
    tmp_path, no_progress, patch_collaborators
):
    main_window = make_crowded_main_window(tmp_path, no_progress)
    other_before = row_at(main_window, "fake-002")
    assert other_before["label_id_m1"] == 2
    assert other_before["alt_label_id_m1"] == 1
    assert other_before["alt_dist_px_m1"] == 4.0

    # Erase x 5..6 of label 1: its centroid moves from x=4 to x=3.
    labels = crowded_labels()[0][0].copy()
    labels[2:5, 5:7] = 0
    edit(main_window, patch_collaborators, labels)

    other = row_at(main_window, "fake-002")
    assert other["label_id_m1"] == 2, "other rows keep their assigned label"
    assert other["alt_label_id_m1"] == 1
    assert other["alt_dist_px_m1"] == 5.0


@pytest.mark.gui
def test_edit_leaves_other_frames_and_masks_untouched(
    tmp_path, no_progress, patch_collaborators
):
    main_window = make_crowded_main_window(tmp_path, no_progress)
    columns = [
        c
        for c in main_window.track_df.columns
        if "m2" in c or c.endswith("M2")
    ]
    before = main_window.track_df[columns].copy()
    later_before = row_at(main_window, "fake-001", t=1).copy()

    labels = crowded_labels()[0][0].copy()
    labels[labels == 2] = 0
    edit(main_window, patch_collaborators, labels)

    pd.testing.assert_frame_equal(main_window.track_df[columns], before)
    later_after = row_at(main_window, "fake-001", t=1)
    for column in ("nn_dist_px_m1", "alt_label_id_m1", "alt_dist_px_m1"):
        assert values_equal(later_before[column], later_after[column])


@pytest.mark.gui
def test_edit_creates_missing_alt_columns_in_an_old_table(
    edit_main_window, patch_collaborators
):
    """A CSV written before the candidate columns existed still updates."""
    mask_1, _ = make_fake_labels()
    edit_main_window.track_df = edit_main_window.track_df.drop(
        columns=[
            c
            for c in edit_main_window.track_df.columns
            if c.startswith("alt_")
        ],
        errors="ignore",
    )

    run_update(
        edit_main_window,
        make_layer("Segmentation1", mask_1),
        patch_collaborators,
    )

    row = edit_main_window.track_df.iloc[0]
    assert row["alt_label_id_m1"] == 0
    assert pd.isna(row["alt_dist_px_m1"])
    assert row["nn_dist_px_m1"] == 0.0


# Close-mask flags after an edit
def flag_crowded(main_window, threshold: float) -> None:
    """Run the close-mask detection over the quantified crowded frame."""
    main_window.filtered_df = main_window.track_df.copy()
    apply_close_mask_detection(main_window, threshold)


def flag_of(main_window, ident: str, t: int = 0, table: str = "track_df"):
    df = getattr(main_window, table)
    return df[(df["Identification"] == ident) & (df["t"] == t)][
        CLOSE_MASK_FLAG
    ].iloc[0]


@pytest.mark.gui
def test_editing_a_flagged_row_marks_it_reviewed(
    tmp_path, no_progress, patch_collaborators
):
    main_window = make_crowded_main_window(tmp_path, no_progress)
    flag_crowded(main_window, 5.0)
    assert flag_of(main_window, "fake-001") == FLAG_FLAGGED
    assert flag_of(main_window, "fake-002") == FLAG_FLAGGED

    edit(main_window, patch_collaborators, crowded_labels()[0][0])

    assert flag_of(main_window, "fake-001") == FLAG_REVIEWED
    assert flag_of(main_window, "fake-002") == FLAG_FLAGGED
    assert (
        flag_of(main_window, "fake-001", table="filtered_df") == FLAG_REVIEWED
    )


@pytest.mark.gui
def test_a_right_click_correction_swaps_the_roles_and_reviews_the_row(
    tmp_path, no_progress, patch_collaborators
):
    main_window = make_crowded_main_window(tmp_path, no_progress)
    flag_crowded(main_window, 5.0)
    df = main_window.track_df
    idx = df.index[(df["Identification"] == "fake-001") & (df["t"] == 0)][0]
    df.loc[idx, ["XMorphology", "YMorphology"]] = [8.0, 3.0]

    edit(main_window, patch_collaborators, crowded_labels()[0][0])

    row = row_at(main_window, "fake-001")
    assert (row["label_id_m1"], row["alt_label_id_m1"]) == (2, 1)
    assert row[CLOSE_MASK_FLAG] == FLAG_REVIEWED
    assert main_window.close_mask_pinned_row == ("fake-001", 1, 0)


@pytest.mark.gui
def test_a_row_that_comes_within_the_distance_after_an_edit_is_flagged(
    tmp_path, no_progress, patch_collaborators
):
    main_window = make_crowded_main_window(tmp_path, no_progress)
    flag_crowded(main_window, 3.0)
    assert flag_of(main_window, "fake-001") == FLAG_OK

    # Label 2 grows two columns to the left: its centroid moves from x=8 to x=7.
    labels = crowded_labels()[0][0].copy()
    labels[2:5, 5:7] = 2
    edit(main_window, patch_collaborators, labels)

    row = row_at(main_window, "fake-001")
    assert row["alt_dist_px_m1"] == 3.0
    assert row[CLOSE_MASK_FLAG] == FLAG_FLAGGED


@pytest.mark.gui
def test_another_row_is_unflagged_when_an_edit_moves_its_candidate_away(
    tmp_path, no_progress, patch_collaborators
):
    main_window = make_crowded_main_window(tmp_path, no_progress)
    flag_crowded(main_window, 4.0)
    assert flag_of(main_window, "fake-002") == FLAG_FLAGGED

    # Label 1 loses x 5..6: its centroid moves from x=4 to x=3, 5 px from fake-002.
    labels = crowded_labels()[0][0].copy()
    labels[2:5, 5:7] = 0
    edit(main_window, patch_collaborators, labels)

    assert row_at(main_window, "fake-002")["alt_dist_px_m1"] == 5.0
    assert flag_of(main_window, "fake-002") == FLAG_OK
    assert flag_of(main_window, "fake-001") == FLAG_REVIEWED


@pytest.mark.gui
def test_reviewed_rows_stay_reviewed_through_an_edit_at_their_frame(
    tmp_path, no_progress, patch_collaborators
):
    main_window = make_crowded_main_window(tmp_path, no_progress)
    flag_crowded(main_window, 5.0)
    df = main_window.track_df
    other = df.index[(df["Identification"] == "fake-002") & (df["t"] == 0)][0]
    df.loc[other, CLOSE_MASK_FLAG] = FLAG_REVIEWED

    edit(main_window, patch_collaborators, crowded_labels()[0][0])

    assert flag_of(main_window, "fake-002") == FLAG_REVIEWED


@pytest.mark.gui
def test_an_edit_before_any_detection_creates_no_flags(
    tmp_path, no_progress, patch_collaborators
):
    main_window = make_crowded_main_window(tmp_path, no_progress)

    edit(main_window, patch_collaborators, crowded_labels()[0][0])

    assert CLOSE_MASK_FLAG not in main_window.track_df.columns


@pytest.mark.gui
def test_an_edit_refreshes_the_napari_view(
    tmp_path, no_progress, patch_collaborators, monkeypatch
):
    main_window = make_crowded_main_window(tmp_path, no_progress)
    flag_crowded(main_window, 5.0)
    refreshed = []
    monkeypatch.setattr(
        measurement_picking_module(),
        "refresh_close_mask_view_at_current",
        lambda window: refreshed.append(window),
    )

    edit(main_window, patch_collaborators, crowded_labels()[0][0])

    assert refreshed == [main_window]
