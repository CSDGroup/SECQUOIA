"""Tests for SECQUOIA mask selection.

These keep the napari Labels layers in step with the selected row. The
viewers are stand-ins holding two Segmentation layers of four frames at
8 x 8 pixels, and the rows carry label ids and coordinates, so both the
highlighted label and the camera position can be checked.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from napari.layers import Labels

from SECQUOIA.core.segmentation.mask_selection import (
    _is_valid_center,
    _label_value_from_row,
    _resolve_t_index,
    _resolve_zoom_center,
    apply_all_mask_selections,
    ensure_current_df_subset,
    get_mask_indices,
    mask_index_from_layer_name,
    rebuild_segmentation_layer_index,
    set_active_layers_from_header_buttons,
    show_all_masks,
    show_current_mask,
)


# Fakes
class FakeLayerList(dict):
    """Stands in for a napari LayerList.

    The production code iterates it for names, indexes it by name, and tests
    membership with ``in`` -- all of which a dict already does. ``selection``
    records which layer was made active.
    """

    def __init__(self, layers: dict):
        super().__init__(layers)
        self.selection = SimpleNamespace(active=None)


def labels_layer(name: str) -> Labels:
    return Labels(np.zeros((4, 8, 8), dtype=np.uint16), name=name)


def fake_viewer(**layers):
    return SimpleNamespace(
        layers=FakeLayerList(layers),
        camera=SimpleNamespace(center=(0.0, 0.0), zoom=1.0),
    )


def segmentation_viewer(n_masks: int = 2):
    return fake_viewer(
        **{
            f"Segmentation{m}": labels_layer(f"Segmentation{m}")
            for m in range(1, n_masks + 1)
        }
    )


# Layer naming
class TestMaskIndexFromLayerName:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("Segmentation1", 1),
            ("Segmentation10", 10),
            ("segmentation_2", 2),
            ("seg 3", 3),
            ("SEG_4", 4),
            ("Masks 7", 7),
        ],
    )
    def test_reads_the_mask_index(self, name, expected):
        assert mask_index_from_layer_name(name) == expected

    def test_prefers_the_segmentation_token_over_a_trailing_number(self):
        assert mask_index_from_layer_name("Segmentation2 copy 9") == 2

    @pytest.mark.parametrize("name", ["Image", "", None])
    def test_returns_none_without_a_number(self, name):
        assert mask_index_from_layer_name(name) is None


class TestGetMaskIndices:
    def test_counts_up_from_n_masks(self, fake_main_window):
        assert get_mask_indices(fake_main_window(n_masks=3)) == [1, 2, 3]

    def test_falls_back_to_the_label_id_columns(self, fake_main_window):
        """A reloaded project may have gaps: masks 1 and 3 but no 2."""
        df = pd.DataFrame({"label_id_m1": [1], "label_id_m3": [1], "t": [0]})
        main_window = fake_main_window(n_masks=0, filtered_df=df)

        assert get_mask_indices(main_window) == [1, 3]

    def test_defaults_to_a_single_mask(self, fake_main_window):
        main_window = fake_main_window(n_masks=0, filtered_df=None)

        assert get_mask_indices(main_window) == [1]

    def test_defaults_to_a_single_mask_without_matching_columns(
        self, fake_main_window
    ):
        main_window = fake_main_window(
            n_masks=0, filtered_df=pd.DataFrame({"t": [0]})
        )

        assert get_mask_indices(main_window) == [1]


# Row inspection
class TestResolveZoomCenter:
    def test_prefers_the_unsuffixed_coordinates(self):
        row = pd.Series(
            {
                "YMorphology": 10.0,
                "XMorphology": 20.0,
                "YMorphologyM1": 99.0,
                "XMorphologyM1": 99.0,
            }
        )

        assert _resolve_zoom_center(row, [1]) == (10.0, 20.0)

    def test_falls_back_to_the_first_per_mask_pair(self):
        row = pd.Series(
            {
                "YMorphology": np.nan,
                "XMorphology": np.nan,
                "YMorphologyM2": 30.0,
                "XMorphologyM2": 40.0,
            }
        )

        assert _resolve_zoom_center(row, [1, 2]) == (30.0, 40.0)

    def test_returns_nan_when_nothing_is_usable(self):
        row = pd.Series({"YMorphology": np.nan, "XMorphology": np.nan})

        cy, cx = _resolve_zoom_center(row, [1])

        assert np.isnan(cy) and np.isnan(cx)


class TestIsValidCenter:

    def test_accepts_a_real_point(self):
        assert _is_valid_center(10.0, 20.0)

    def test_rejects_the_origin(self):
        """(0, 0) is the "no position" sentinel, not a place to look at."""
        assert not _is_valid_center(0.0, 0.0)

    def test_rejects_a_non_finite_point(self):
        assert not _is_valid_center(np.nan, 5.0)
        assert not _is_valid_center(np.inf, 5.0)

    def test_accepts_a_point_just_off_the_origin(self):
        assert _is_valid_center(0.0, 0.5)


class TestLabelValueFromRow:
    def test_reads_the_label_for_the_requested_mask(self):
        row = pd.Series({"label_id_m1": 7, "label_id_m2": 9})

        assert _label_value_from_row(row, 2) == 9

    def test_returns_zero_for_a_missing_column(self):
        assert _label_value_from_row(pd.Series({"t": 0}), 1) == 0

    def test_returns_zero_for_a_missing_value(self):
        row = pd.Series({"label_id_m1": np.nan})

        assert _label_value_from_row(row, 1) == 0

    def test_truncates_a_float_label(self):
        row = pd.Series({"label_id_m1": 7.0})

        assert _label_value_from_row(row, 1) == 7


class TestResolveTIndex:
    def test_prefers_the_rows_own_time(self, fake_main_window):
        main_window = fake_main_window(current_time_index=5)

        assert _resolve_t_index(main_window, pd.Series({"t": 2})) == 2

    def test_falls_back_to_the_current_index(self, fake_main_window):
        main_window = fake_main_window(current_time_index=5)

        assert _resolve_t_index(main_window, pd.Series({"x": 1})) == 5

    def test_falls_back_when_the_time_is_unparsable(self, fake_main_window):
        main_window = fake_main_window(current_time_index=5)

        assert _resolve_t_index(main_window, pd.Series({"t": "n/a"})) == 5


# Dataframe subset
def two_ident_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Identification": ["a", "a", "b"],
            "TrackNumber": [1, 1, 1],
            "t": [0, 1, 0],
            "label_id_m1": [5, 6, 7],
            "label_id_m2": [15, 16, 17],
            "XMorphology": [10.0, 11.0, 12.0],
            "YMorphology": [20.0, 21.0, 22.0],
        }
    )


class TestEnsureCurrentDfSubset:
    def test_selects_the_rows_of_the_current_identification(
        self, fake_main_window
    ):
        main_window = fake_main_window(
            filtered_df=two_ident_frame(),
            unique_ids=["a", "b"],
            current_ident_index=1,
        )

        assert ensure_current_df_subset(main_window) is True
        assert main_window.df_subset["t"].tolist() == [0]
        assert main_window.df_subset["label_id_m1"].tolist() == [7]

    def test_falls_back_to_the_first_row_for_an_out_of_range_index(
        self, fake_main_window
    ):
        main_window = fake_main_window(
            filtered_df=two_ident_frame(),
            unique_ids=["a", "b"],
            current_ident_index=99,
        )

        ensure_current_df_subset(main_window)

        assert main_window.df_subset["Identification"].unique().tolist() == [
            "a"
        ]

    def test_clears_the_subset_without_a_frame(self, fake_main_window):
        main_window = fake_main_window(filtered_df=None)

        assert ensure_current_df_subset(main_window) is False
        assert main_window.df_subset is None

    def test_clears_the_subset_for_an_empty_frame(self, fake_main_window):
        main_window = fake_main_window(filtered_df=pd.DataFrame())

        assert ensure_current_df_subset(main_window) is False

    def test_reports_false_when_the_identification_matches_nothing(
        self, fake_main_window
    ):
        main_window = fake_main_window(
            filtered_df=two_ident_frame(),
            unique_ids=["ghost"],
            current_ident_index=0,
        )

        assert ensure_current_df_subset(main_window) is False


# Layer index
class TestRebuildSegmentationLayerIndex:
    def test_maps_mask_indices_to_layer_names(self, fake_main_window):
        main_window = fake_main_window(
            viewer_1=segmentation_viewer(2), viewer_2=None
        )

        rebuild_segmentation_layer_index(main_window)

        assert main_window.seg_layers_by_viewer[0] == {
            1: "Segmentation1",
            2: "Segmentation2",
        }

    def test_indexes_both_viewers_separately(self, fake_main_window):
        main_window = fake_main_window(
            viewer_1=segmentation_viewer(1), viewer_2=segmentation_viewer(2)
        )

        rebuild_segmentation_layer_index(main_window)

        assert len(main_window.seg_layers_by_viewer[0]) == 1
        assert len(main_window.seg_layers_by_viewer[1]) == 2

    def test_ignores_image_layers(self, fake_main_window):
        viewer = fake_viewer(
            Segmentation1=labels_layer("Segmentation1"),
            Brightfield=SimpleNamespace(name="Brightfield"),
        )
        main_window = fake_main_window(viewer_1=viewer, viewer_2=None)

        rebuild_segmentation_layer_index(main_window)

        assert main_window.seg_layers_by_viewer[0] == {1: "Segmentation1"}

    def test_assigns_a_free_index_to_an_unnumbered_layer(
        self, fake_main_window
    ):
        viewer = fake_viewer(
            Segmentation1=labels_layer("Segmentation1"),
            Nuclei=labels_layer("Nuclei"),
        )
        main_window = fake_main_window(viewer_1=viewer, viewer_2=None)

        rebuild_segmentation_layer_index(main_window)

        mapping = main_window.seg_layers_by_viewer[0]
        assert mapping[1] == "Segmentation1"
        assert "Nuclei" in mapping.values()

    def test_a_missing_viewer_gets_an_empty_mapping(self, fake_main_window):
        main_window = fake_main_window(viewer_1=None, viewer_2=None)

        rebuild_segmentation_layer_index(main_window)

        assert main_window.seg_layers_by_viewer == {0: {}, 1: {}}


class TestSetActiveLayersFromHeaderButtons:
    def test_activates_the_remembered_mask(self, fake_main_window):
        viewer = segmentation_viewer(2)
        main_window = fake_main_window(
            viewer_1=viewer,
            viewer_2=None,
            active_mask_index_by_viewer={0: 2},
        )

        set_active_layers_from_header_buttons(main_window)

        assert viewer.layers.selection.active is viewer.layers["Segmentation2"]

    def test_defaults_to_the_first_mask(self, fake_main_window):
        viewer = segmentation_viewer(2)
        main_window = fake_main_window(viewer_1=viewer, viewer_2=None)

        set_active_layers_from_header_buttons(main_window)

        assert viewer.layers.selection.active is viewer.layers["Segmentation1"]

    def test_remembers_the_choice_per_viewer(self, fake_main_window):
        main_window = fake_main_window(
            viewer_1=segmentation_viewer(2), viewer_2=segmentation_viewer(2)
        )

        set_active_layers_from_header_buttons(main_window)

        assert main_window.active_mask_index_by_viewer == {0: 1, 1: 1}

    def test_a_negative_index_is_clamped_to_the_first_mask(
        self, fake_main_window
    ):
        viewer = segmentation_viewer(2)
        main_window = fake_main_window(
            viewer_1=viewer,
            viewer_2=None,
            active_mask_index_by_viewer={0: -5},
        )

        set_active_layers_from_header_buttons(main_window)

        assert main_window.active_mask_index_by_viewer[0] == 1

    def test_an_unmapped_mask_leaves_the_selection_alone(
        self, fake_main_window
    ):
        viewer = segmentation_viewer(1)
        main_window = fake_main_window(
            viewer_1=viewer,
            viewer_2=None,
            active_mask_index_by_viewer={0: 9},
        )

        set_active_layers_from_header_buttons(main_window)

        assert viewer.layers.selection.active is None

    def test_builds_the_index_when_it_is_missing(self, fake_main_window):
        main_window = fake_main_window(
            viewer_1=segmentation_viewer(1), viewer_2=None
        )

        set_active_layers_from_header_buttons(main_window)

        assert hasattr(main_window, "seg_layers_by_viewer")


# Applying a selection
class TestApplyAllMaskSelections:
    def test_highlights_the_label_of_each_mask(self, fake_main_window):
        viewer = segmentation_viewer(2)
        main_window = fake_main_window(
            n_masks=2, viewer_1=viewer, viewer_2=None
        )
        row = two_ident_frame().iloc[0]

        apply_all_mask_selections(main_window, row)

        assert viewer.layers["Segmentation1"].selected_label == 5
        assert viewer.layers["Segmentation2"].selected_label == 15
        assert viewer.layers["Segmentation1"].show_selected_label is True

    def test_centres_the_camera_on_the_row(self, fake_main_window):
        viewer = segmentation_viewer(1)
        main_window = fake_main_window(
            n_masks=1, viewer_1=viewer, viewer_2=None
        )
        row = two_ident_frame().iloc[0]

        apply_all_mask_selections(main_window, row)

        assert viewer.camera.center == (20.0, 10.0)

    def test_zooms_only_when_the_identification_changes(
        self, fake_main_window
    ):
        viewer = segmentation_viewer(1)
        main_window = fake_main_window(
            n_masks=1,
            viewer_1=viewer,
            viewer_2=None,
            _last_zoom_ident="a",  # same ident as the row
        )
        viewer.camera.zoom = 3.0
        row = two_ident_frame().iloc[0]

        apply_all_mask_selections(main_window, row, zoom_level=99)

        assert viewer.camera.zoom == 3.0

    def test_zooms_on_a_new_identification(self, fake_main_window):
        viewer = segmentation_viewer(1)
        main_window = fake_main_window(
            n_masks=1, viewer_1=viewer, viewer_2=None, _last_zoom_ident="other"
        )
        row = two_ident_frame().iloc[0]

        apply_all_mask_selections(main_window, row, zoom_level=99)

        assert viewer.camera.zoom == 99

    def test_remembers_the_identification_it_zoomed_to(self, fake_main_window):
        main_window = fake_main_window(
            n_masks=1, viewer_1=segmentation_viewer(1), viewer_2=None
        )
        row = two_ident_frame().iloc[0]

        apply_all_mask_selections(main_window, row)

        assert main_window._last_zoom_ident == "a"

    def test_a_row_with_no_label_hides_the_selection(self, fake_main_window):
        """label 0 means this mask has nothing tracked at this frame."""
        viewer = segmentation_viewer(1)
        main_window = fake_main_window(
            n_masks=1, viewer_1=viewer, viewer_2=None, current_time_index=0
        )
        row = two_ident_frame().iloc[0].copy()
        row["label_id_m1"] = 0

        apply_all_mask_selections(main_window, row)

        assert viewer.layers["Segmentation1"].show_selected_label is False

    def test_a_row_with_no_label_offers_a_fresh_label_to_paint(
        self, fake_main_window
    ):
        viewer = segmentation_viewer(1)
        layer = viewer.layers["Segmentation1"]
        layer.data[0, 2:4, 2:4] = 6
        main_window = fake_main_window(
            n_masks=1, viewer_1=viewer, viewer_2=None, current_time_index=0
        )
        row = two_ident_frame().iloc[0].copy()
        row["label_id_m1"] = 0

        apply_all_mask_selections(main_window, row)

        assert layer.selected_label == 7

    def test_camera_is_not_moved_to_the_origin(self, fake_main_window):
        viewer = segmentation_viewer(1)
        viewer.camera.center = (5.0, 5.0)
        main_window = fake_main_window(
            n_masks=1, viewer_1=viewer, viewer_2=None
        )
        row = two_ident_frame().iloc[0].copy()
        row["XMorphology"] = 0.0
        row["YMorphology"] = 0.0

        apply_all_mask_selections(main_window, row)

        assert viewer.camera.center == (5.0, 5.0)

    def test_centring_can_be_switched_off(self, fake_main_window):
        viewer = segmentation_viewer(1)
        viewer.camera.center = (5.0, 5.0)
        main_window = fake_main_window(
            n_masks=1, viewer_1=viewer, viewer_2=None
        )
        row = two_ident_frame().iloc[0]

        apply_all_mask_selections(main_window, row, center_camera=False)

        assert viewer.camera.center == (5.0, 5.0)

    def test_a_missing_viewer_is_skipped(self, fake_main_window):
        main_window = fake_main_window(n_masks=1, viewer_1=None, viewer_2=None)

        apply_all_mask_selections(main_window, two_ident_frame().iloc[0])

    def test_both_viewers_are_updated(self, fake_main_window):
        viewer_1 = segmentation_viewer(1)
        viewer_2 = segmentation_viewer(1)
        main_window = fake_main_window(
            n_masks=1, viewer_1=viewer_1, viewer_2=viewer_2
        )

        apply_all_mask_selections(main_window, two_ident_frame().iloc[0])

        assert viewer_1.layers["Segmentation1"].selected_label == 5
        assert viewer_2.layers["Segmentation1"].selected_label == 5


# Show all / show current
class TestShowAllMasks:
    def test_turns_off_isolation_on_every_layer(self, fake_main_window):
        viewer = segmentation_viewer(2)
        for layer in viewer.layers.values():
            layer.show_selected_label = True
        main_window = fake_main_window(
            n_masks=2, viewer_1=viewer, viewer_2=None
        )

        show_all_masks(main_window)

        assert all(
            layer.show_selected_label is False
            for layer in viewer.layers.values()
        )

    def test_a_missing_layer_is_skipped(self, fake_main_window):
        viewer = segmentation_viewer(1)
        main_window = fake_main_window(
            n_masks=5, viewer_1=viewer, viewer_2=None
        )

        show_all_masks(main_window)

        assert viewer.layers["Segmentation1"].show_selected_label is False

    def test_a_missing_viewer_is_skipped(self, fake_main_window):
        show_all_masks(
            fake_main_window(n_masks=1, viewer_1=None, viewer_2=None)
        )


class TestShowCurrentMask:
    def window(self, fake_main_window, viewer, df=None):
        df = two_ident_frame() if df is None else df
        return fake_main_window(
            n_masks=2,
            viewer_1=viewer,
            viewer_2=None,
            filtered_df=df,
            _current_zoom_identification=lambda frame: "a",
            _filtered_zoom_subset=lambda frame: frame.iloc[0:1],
        )

    def test_isolates_the_label_of_the_current_row(self, fake_main_window):
        viewer = segmentation_viewer(2)
        main_window = self.window(fake_main_window, viewer)

        show_current_mask(main_window)

        assert viewer.layers["Segmentation1"].selected_label == 5
        assert viewer.layers["Segmentation2"].selected_label == 15

    def test_does_nothing_without_a_frame(self, fake_main_window):
        viewer = segmentation_viewer(1)
        main_window = self.window(fake_main_window, viewer, df=pd.DataFrame())

        show_current_mask(main_window)

        assert viewer.layers["Segmentation1"].selected_label == 1

    def test_falls_back_to_the_identification_rows_when_the_subset_is_empty(
        self, fake_main_window
    ):
        viewer = segmentation_viewer(1)
        main_window = self.window(fake_main_window, viewer)
        main_window._filtered_zoom_subset = lambda frame: frame.iloc[0:0]

        show_current_mask(main_window)

        assert viewer.layers["Segmentation1"].selected_label == 5

    def test_a_window_without_the_zoom_helpers_is_not_an_error(
        self, fake_main_window
    ):
        main_window = fake_main_window(
            n_masks=1,
            viewer_1=segmentation_viewer(1),
            viewer_2=None,
            filtered_df=two_ident_frame(),
        )

        show_current_mask(main_window)
