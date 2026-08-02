"""Characterisation tests for ``right_mouse_click_viewer``.

These pin the observable behaviour of the right-click anchor path: what
lands in ``track_df``, which measurement refreshes are triggered, and which
guard paths do nothing. They assert on outputs only, never on internal
structure, so they stay green across the extraction of the helper methods.

Groups:

``TestGuards``           - events that must be ignored entirely.
``TestPicking``          - which layer wins, coordinate handling, label 0.
``TestAnchorWrite``      - dataframe mutation, column creation, dtype coercion.
``TestMeasurementRefresh`` - single-mask vs ALL dispatch.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


# Locating MainWindow (mirrors test_mask_measurement_update.py)
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


@pytest.fixture(scope="module")
def mw_module():
    """The imported main_window module."""
    return importlib.import_module(find_module_name("main_window.py"))


@pytest.fixture(scope="module")
def MainWindow(mw_module):
    return mw_module.MainWindow


@pytest.fixture(scope="module")
def mb_module():
    """The imported mouse_bindings module, where the Ctrl/right-click guards live."""
    return importlib.import_module(find_module_name("mouse_bindings.py"))


# Methods copied onto the fake MainWindow.
BOUND_METHODS = (
    "right_mouse_click_viewer",
    "_is_right_click_suppressed",
    "_viewer_display_name",
    "_pick_segmentation_at_cursor",
    "_frame_shape",
    "_mask_index_from_binding",
    "_binding_from_layer_name",
    "_resolve_current_ident_str",
    "_infer_track_number",
    "_resolve_track_row",
    "_ensure_float_column",
    "_write_morphology_anchor",
    "_remeasure_all_masks",
    "_remeasure_single_layer",
    "_recompute_measurements_after_pick",
)


class FakeLayer:

    def __init__(self, name, data, value_at=1, visible=True):
        self.name = name
        self.data = np.asarray(data)
        self.visible = visible
        self._value_at = value_at

    def get_value(self, position, **kwargs):
        return self._value_at

    def world_to_data(self, position):
        return np.asarray(position, dtype=float)


class FakeViewer:
    def __init__(self, layers, title="viewer_1"):
        self.layers = list(layers)
        self.title = title


def make_event(
    kind="mouse_press", button=2, position=(0, 3.4, 5.6)
) -> SimpleNamespace:
    return SimpleNamespace(
        type=kind,
        button=button,
        position=position,
        view_direction=None,
        dims_displayed=None,
    )


def make_track_df() -> pd.DataFrame:
    """Two time points of one track, with no morphology columns yet."""
    return pd.DataFrame(
        {
            "Identification": ["fake-001", "fake-001"],
            "TrackNumber": [1, 1],
            "t": [0, 1],
            "Area": [9.0, 9.0],
        }
    )


@pytest.fixture
def main_window(MainWindow, mw_module, mb_module, monkeypatch):
    """A fake MainWindow with the right-click methods bound onto it."""
    mw = SimpleNamespace()

    for name in BOUND_METHODS:
        setattr(mw, name, getattr(MainWindow, name).__get__(mw, MainWindow))

    mw.current_time_index = 0
    mw.current_TrackNumber_plot = 1
    mw.ident = "fake-001"
    mw.unique_ids = ["fake-001"]
    mw.current_ident_index = 0
    mw.n_masks = 2
    mw.track_df = make_track_df()
    mw.BASE_SEG_NAME = "Segmentation"
    mw._is_labels_layer = lambda layer: True

    mw.calls = []
    mw.mask_selection = 1  # 0 would mean "ALL"
    mw._mask_selection_for_viewer = lambda viewer: mw.mask_selection
    mw._get_seg_layer = lambda viewer, m_idx: next(
        (lyr for lyr in viewer.layers if lyr.name == f"Segmentation{m_idx}"),
        None,
    )
    mw.update_mask_measurements_for_current_track = (
        lambda layer, *, source="": mw.calls.append(
            ("measure", layer.name, source)
        )
    )
    mw._refresh_lineage_after_edit = lambda: mw.calls.append(("refresh",))

    # Ctrl is not held unless a test says otherwise.
    monkeypatch.setattr(
        mb_module,
        "QApplication",
        SimpleNamespace(keyboardModifiers=lambda: mw_module.Qt.NoModifier),
        raising=True,
    )
    return mw


def run(main_window, viewer, event):
    """Drive the callback generator to exhaustion and return the yields."""
    return list(main_window.right_mouse_click_viewer(viewer, event))


@pytest.fixture
def viewer():
    """One 4x4 segmentation layer, label 7 everywhere the cursor lands."""
    return FakeViewer([FakeLayer("Segmentation1", np.ones((2, 4, 4)), 7)])


# Guards
class TestGuards:
    def test_ctrl_held_is_ignored(
        self, main_window, mw_module, mb_module, viewer, monkeypatch
    ):
        monkeypatch.setattr(
            mb_module,
            "QApplication",
            SimpleNamespace(
                keyboardModifiers=lambda: mw_module.Qt.ControlModifier
            ),
            raising=True,
        )
        before = main_window.track_df.copy()

        assert run(main_window, viewer, make_event()) == []
        assert main_window.calls == []
        pd.testing.assert_frame_equal(main_window.track_df, before)

    def test_left_click_is_ignored(self, main_window, viewer):
        assert run(main_window, viewer, make_event(button=1)) == []
        assert main_window.calls == []

    def test_non_press_event_is_ignored(self, main_window, viewer):
        assert run(main_window, viewer, make_event(kind="mouse_move")) == []
        assert main_window.calls == []

    def test_press_yields_once_and_drains(self, main_window, viewer):
        # A press that is not followed by mouse_move yields exactly once.
        assert run(main_window, viewer, make_event()) == [None]


# Picking
class TestPicking:
    def test_topmost_visible_layer_wins(self, main_window):
        bottom = FakeLayer("Segmentation1", np.ones((2, 4, 4)), 1)
        top = FakeLayer("Segmentation2", np.ones((2, 4, 4)), 2)
        v = FakeViewer([bottom, top])

        pick = main_window._pick_segmentation_at_cursor(v, make_event())
        assert pick["tag"] == "Segmentation2"
        assert pick["m_idx"] == 2

    def test_invisible_layers_are_skipped(self, main_window):
        top = FakeLayer("Segmentation2", np.ones((2, 4, 4)), 2, visible=False)
        bottom = FakeLayer("Segmentation1", np.ones((2, 4, 4)), 1)
        v = FakeViewer([bottom, top])

        assert (
            main_window._pick_segmentation_at_cursor(v, make_event())["tag"]
            == "Segmentation1"
        )

    def test_non_segmentation_names_are_skipped(self, main_window):
        v = FakeViewer([FakeLayer("Channel_w01", np.ones((2, 4, 4)), 5)])
        assert (
            main_window._pick_segmentation_at_cursor(v, make_event()) is None
        )

    def test_coordinates_are_rounded(self, main_window, viewer):
        pick = main_window._pick_segmentation_at_cursor(
            viewer, make_event(position=(0, 1.6, 2.4))
        )
        assert (pick["YMorphology"], pick["XMorphology"]) == (2, 2)

    def test_coordinates_are_clamped_to_the_frame(self, main_window, viewer):
        pick = main_window._pick_segmentation_at_cursor(
            viewer, make_event(position=(0, 99.0, -5.0))
        )
        # Frame is 4x4, so the valid range is 0..3.
        assert (pick["YMorphology"], pick["XMorphology"]) == (3, 0)

    def test_background_excluded_by_default(self, main_window):
        v = FakeViewer([FakeLayer("Segmentation1", np.zeros((2, 4, 4)), 0)])
        assert (
            main_window._pick_segmentation_at_cursor(v, make_event()) is None
        )

    def test_background_included_when_requested(self, main_window):
        """The right-click path opts in: clicking background is meaningful."""
        v = FakeViewer([FakeLayer("Segmentation1", np.zeros((2, 4, 4)), 0)])
        pick = main_window._pick_segmentation_at_cursor(
            v, make_event(), include_background=True
        )
        assert pick is not None and pick["label"] == 0

    def test_right_click_writes_anchor_on_background(self, main_window):
        v = FakeViewer([FakeLayer("Segmentation1", np.zeros((2, 4, 4)), 0)])
        run(main_window, v, make_event(position=(0, 2.0, 1.0)))
        row = main_window.track_df.iloc[0]
        assert (row["YMorphology"], row["XMorphology"]) == (2.0, 1.0)

    @pytest.mark.parametrize(
        ("layer_name", "expected"),
        [
            ("Segmentation", 1),
            ("Segmentation2", 2),
            ("Segmentation_3", 3),
        ],
    )
    def test_mask_index_resolution(self, main_window, layer_name, expected):
        v = FakeViewer([FakeLayer(layer_name, np.ones((2, 4, 4)), 1)])
        pick = main_window._pick_segmentation_at_cursor(v, make_event())
        assert pick["m_idx"] == expected

    def test_get_value_failure_falls_through(self, main_window):
        broken = FakeLayer("Segmentation1", np.ones((2, 4, 4)), 1)
        broken.get_value = lambda *a, **k: (_ for _ in ()).throw(RuntimeError)
        v = FakeViewer([broken])
        assert (
            main_window._pick_segmentation_at_cursor(v, make_event()) is None
        )


# Anchor write
class TestAnchorWrite:
    def test_writes_x_and_y_to_the_current_row(self, main_window, viewer):
        run(main_window, viewer, make_event(position=(0, 2.0, 3.0)))

        row = main_window.track_df.loc[main_window.track_df["t"] == 0].iloc[0]
        assert row["XMorphology"] == 3.0
        assert row["YMorphology"] == 2.0

    def test_other_timepoints_untouched(self, main_window, viewer):
        run(main_window, viewer, make_event(position=(0, 2.0, 3.0)))

        other = main_window.track_df.loc[main_window.track_df["t"] == 1].iloc[
            0
        ]
        assert pd.isna(other["XMorphology"])

    def test_writes_at_the_current_time_index(self, main_window, viewer):
        main_window.current_time_index = 1
        run(main_window, viewer, make_event(position=(0, 1.0, 1.0)))

        assert main_window.track_df.loc[1, "XMorphology"] == 1.0
        assert pd.isna(main_window.track_df.loc[0, "XMorphology"])

    def test_integer_columns_are_coerced_to_float(self, main_window, viewer):
        df = main_window.track_df
        df["XMorphology"] = np.array([0, 0], dtype=np.int64)
        df["YMorphology"] = np.array([0, 0], dtype=np.int64)

        run(main_window, viewer, make_event(position=(0, 2.0, 3.0)))

        assert main_window.track_df["XMorphology"].dtype == float
        assert main_window.track_df.loc[0, "XMorphology"] == 3.0

    def test_empty_dataframe_aborts_without_raising(self, main_window, viewer):
        main_window.track_df = pd.DataFrame()
        run(main_window, viewer, make_event())
        assert main_window.track_df.empty

    def test_missing_dataframe_aborts_without_raising(
        self, main_window, viewer
    ):
        main_window.track_df = None
        run(main_window, viewer, make_event())
        # The lineage refresh still runs: the guard is about the write only.
        assert ("refresh",) in main_window.calls

    def test_track_number_inferred_when_plot_has_none(
        self, main_window, viewer
    ):
        main_window.current_TrackNumber_plot = None
        run(main_window, viewer, make_event(position=(0, 2.0, 3.0)))
        assert main_window.track_df.loc[0, "XMorphology"] == 3.0

    def test_ambiguous_track_number_aborts(self, main_window, viewer):
        main_window.current_TrackNumber_plot = None
        main_window.track_df = pd.DataFrame(
            {
                "Identification": ["fake-001", "fake-001"],
                "TrackNumber": [1, 2],
                "t": [0, 0],
                "Area": [9.0, 9.0],
            }
        )
        run(main_window, viewer, make_event())
        assert "XMorphology" not in main_window.track_df.columns

    def test_unknown_identification_aborts(self, main_window, viewer):
        main_window.ident = "does-not-exist"
        run(main_window, viewer, make_event())
        # The columns are created lazily, so an aborted write leaves the
        # dataframe exactly as it was.
        assert "XMorphology" not in main_window.track_df.columns

    def test_ident_falls_back_to_unique_ids(self, main_window, viewer):
        del main_window.ident
        run(main_window, viewer, make_event(position=(0, 2.0, 3.0)))
        assert main_window.track_df.loc[0, "XMorphology"] == 3.0


# Measurement refresh
class TestMeasurementRefresh:
    def test_single_mask_refreshes_only_the_picked_layer(
        self, main_window, viewer
    ):
        main_window.mask_selection = 1
        run(main_window, viewer, make_event())

        measures = [c for c in main_window.calls if c[0] == "measure"]
        assert measures == [("measure", "Segmentation1", "right_click")]

    def test_all_selection_refreshes_every_mask(self, main_window):
        main_window.mask_selection = 0
        main_window.n_masks = 2
        v = FakeViewer(
            [
                FakeLayer("Segmentation1", np.ones((2, 4, 4)), 1),
                FakeLayer("Segmentation2", np.ones((2, 4, 4)), 1),
            ]
        )
        run(main_window, v, make_event())

        measures = [c for c in main_window.calls if c[0] == "measure"]
        assert measures == [
            ("measure", "Segmentation1", "right_click_all"),
            ("measure", "Segmentation2", "right_click_all"),
        ]

    def test_all_selection_with_no_masks_is_a_no_op(self, main_window, viewer):
        main_window.mask_selection = 0
        main_window.n_masks = 0
        run(main_window, viewer, make_event())

        assert [c for c in main_window.calls if c[0] == "measure"] == []

    def test_failed_write_suppresses_the_single_refresh(
        self, main_window, viewer
    ):
        """No row to anchor means the measurement would use stale coordinates."""
        main_window.ident = "does-not-exist"
        main_window.mask_selection = 1
        run(main_window, viewer, make_event())

        assert [c for c in main_window.calls if c[0] == "measure"] == []

    def test_miss_still_refreshes_all_when_selection_is_all(self, main_window):
        """ALL mode is not conditional on hitting a layer."""
        main_window.mask_selection = 0
        v = FakeViewer([FakeLayer("Channel_w01", np.ones((2, 4, 4)), 5)])
        v.layers.append(FakeLayer("Segmentation1", np.ones((2, 4, 4)), 1))
        main_window.n_masks = 1
        run(main_window, v, make_event())

        assert (
            "measure",
            "Segmentation1",
            "right_click_all",
        ) in main_window.calls

    def test_measurement_errors_are_contained(self, main_window, viewer):
        def boom(layer, *, source=""):
            raise ValueError("measurement exploded")

        main_window.update_mask_measurements_for_current_track = boom
        run(main_window, viewer, make_event())

        # The lineage refresh must still happen.
        assert ("refresh",) in main_window.calls

    def test_lineage_refresh_runs_exactly_once(self, main_window, viewer):
        run(main_window, viewer, make_event())
        assert main_window.calls.count(("refresh",)) == 1

    def test_lineage_refresh_runs_on_a_miss(self, main_window):
        v = FakeViewer([FakeLayer("Channel_w01", np.ones((2, 4, 4)), 5)])
        run(main_window, v, make_event())
        assert main_window.calls.count(("refresh",)) == 1
