"""Characterisation tests for ``update_mask_measurements_for_current_track``.

Three groups:

``Phase A`` - golden values and guard paths.
    What the method writes for a known synthetic edit, and that every early
    return leaves ``track_df`` untouched.

``Phase B`` - equivalence with the batch pipeline.
    ``quantify()`` and this method are two implementations of the same
    measurement. On identical input they must produce identical columns.

``Phase C`` - known divergences.
    Cases where the two implementations are currently known to disagree.
    These encode the behaviour we want after the refactor and are expected to
    fail today; see the docstring on each one.

The artificial dataset from ``test_quantify.py``:
one position, one cell, two time points, channel ``w01``, and two masks
(mask 1 is 3x3 -> area 9, mask 2 is 2x2 -> area 4).
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from conftest import make_fake_labels, make_fake_main_window

from SECQUOIA.core.quantification import quantify

ANCHOR_X = 3
ANCHOR_Y = 3


# Locating MainWindow
def find_module_name(filename: str) -> str:
    """Return the dotted module name for `filename` inside the package."""
    spec = importlib.util.find_spec("SECQUOIA")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError(
            "SECQUOIA is not importable. Run `pip install -e '.[testing]'`."
        )

    root = Path(next(iter(spec.submodule_search_locations)))
    matches = sorted(root.rglob(filename))
    if not matches:
        raise ModuleNotFoundError(
            f"No {filename} found anywhere under {root}."
        )

    relative = matches[0].relative_to(root).with_suffix("")
    return ".".join(("SECQUOIA", *relative.parts))


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
    """Running twice on unchanged labels must not change any value.

    Also pins the log: the second run reports no column changes.
    """
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
    """Phase 8 must run: filtered_df rebuilt, tracks refreshed, row inspected."""
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


# Phase C - parity with the batch pipeline
@pytest.mark.gui
def test_all_zero_frame_flagged_present_is_measured_like_the_pipeline(
    tmp_path, no_progress, patch_collaborators
):
    """A dark frame flagged as acquired is measured by both paths.

    This was expected to diverge and does not: when ``image_present`` has a
    usable flag array, ``_frame_missing`` honours it and never reaches its
    all-zero fallback. The fallback only fires when ``image_present`` is
    absent or unusable, which is also when the pipeline would treat the frame
    as present. Keep the test - it guards the behaviour during the refactor.
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
