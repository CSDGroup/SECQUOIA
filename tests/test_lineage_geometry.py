"""Characterization tests for the pure lineage geometry layer."""

import numpy as np
import pandas as pd
import pytest

from SECQUOIA.gui.lineage_tree.lineage_geometry import (
    FeatureCatalog,
    LineageGeometry,
    assign_y_tidy,
    build_edges,
    build_track_table,
    build_value_map,
    generation,
    left,
    normalize,
    parent,
    plan_heat_segments,
    right,
    segments_to_polyline,
    valid_times_by_track,
    value_range,
)

# Track numbering


@pytest.mark.parametrize(
    ("n", "expected"), [(1, 0), (2, 1), (3, 1), (4, 2), (7, 2), (8, 3)]
)
def test_generation_matches_binary_depth(n, expected):
    assert generation(n) == expected


def test_generation_of_zero_is_clamped_not_negative_infinity():
    # log2(0) would be -inf; the max(1, n) guard is load-bearing.
    assert generation(0) == 0


def test_parent_child_round_trip():
    for n in range(1, 64):
        assert parent(left(n)) == n
        assert parent(right(n)) == n


# Track / edge tables


def test_build_track_table_one_row_per_track(lineage_df, ident):
    tracks = build_track_table(lineage_df, ident)
    assert list(tracks["TrackNumber"]) == [1, 2, 3, 4, 5]
    assert list(tracks.columns) == ["TrackNumber", "t_start", "t_end", "fate"]


def test_build_track_table_start_and_end_are_observed_extremes(
    lineage_df, ident
):
    tracks = build_track_table(lineage_df, ident).set_index("TrackNumber")
    assert tracks.loc[1, "t_start"] == 0.0
    assert tracks.loc[1, "t_end"] == 10.0
    assert tracks.loc[5, "t_start"] == 20.0
    assert tracks.loc[5, "t_end"] == 28.0


def test_build_track_table_defaults_missing_fate_to_healthy(lineage_df, ident):
    tracks = build_track_table(lineage_df, ident).set_index("TrackNumber")
    assert tracks.loc[1, "fate"] == "Healthy"
    assert tracks.loc[5, "fate"] == "Dead"


def test_build_track_table_rejects_unknown_ident(lineage_df):
    with pytest.raises(ValueError, match="No rows for Identification"):
        build_track_table(lineage_df, "does_not_exist")


def test_build_edges_links_parents_to_present_children(lineage_df, ident):
    edges = build_edges(build_track_table(lineage_df, ident))
    assert set(map(tuple, edges[["parent", "child"]].to_numpy())) == {
        (1, 2),
        (1, 3),
        (2, 4),
        (2, 5),
    }


def test_build_edges_uses_child_start_as_division_time(lineage_df, ident):
    edges = build_edges(build_track_table(lineage_df, ident))
    div = dict(zip(edges["child"], edges["t_div"], strict=False))
    assert div[2] == 10.0
    assert div[3] == 10.0
    assert div[4] == 20.0


def test_build_edges_on_single_track_is_empty(lineage_df, ident):
    single = build_track_table(lineage_df, ident).head(1)
    edges = build_edges(single)
    assert edges.empty
    assert list(edges.columns) == ["parent", "child", "t_div"]


# Layout


def test_assign_y_tidy_gives_every_track_a_row(lineage_df, ident):
    y_map = assign_y_tidy(build_track_table(lineage_df, ident))
    assert set(y_map) == {1, 2, 3, 4, 5}


def test_assign_y_tidy_centres_parents_between_children(lineage_df, ident):
    y_map = assign_y_tidy(build_track_table(lineage_df, ident))
    assert y_map[2] == pytest.approx(0.5 * (y_map[4] + y_map[5]))
    assert y_map[1] == pytest.approx(0.5 * (y_map[2] + y_map[3]))


def test_assign_y_tidy_leaves_occupy_consecutive_integer_rows(
    lineage_df, ident
):
    y_map = assign_y_tidy(build_track_table(lineage_df, ident))
    leaves = sorted(y_map[t] for t in (4, 5, 3))
    assert leaves == [0.0, 1.0, 2.0]


def test_assign_y_tidy_is_deterministic(lineage_df, ident):
    tracks = build_track_table(lineage_df, ident)
    assert assign_y_tidy(tracks) == assign_y_tidy(tracks)


# Valid times
def test_valid_times_covers_all_frames_when_labels_are_nonzero(
    lineage_df, ident
):
    sub = lineage_df[lineage_df["Identification"] == ident]
    vt = valid_times_by_track(sub, 1)
    assert vt[1] == {float(t) for t in range(11)}


def test_valid_times_drops_frames_with_zero_label(lineage_df_with_gaps, ident):
    sub = lineage_df_with_gaps[lineage_df_with_gaps["Identification"] == ident]
    vt = valid_times_by_track(sub, 1)
    assert not ({15.0, 16.0, 17.0, 18.0, 19.0} & vt[3])
    assert 14.0 in vt[3] and 20.0 in vt[3]


def test_valid_times_is_empty_when_label_column_is_absent(lineage_df, ident):
    sub = lineage_df[lineage_df["Identification"] == ident].drop(
        columns=["label_id_m1"]
    )
    vt = valid_times_by_track(sub, 1)
    assert set(vt) == {1, 2, 3, 4, 5}
    assert all(v == set() for v in vt.values())


# Polyline construction
def test_segments_to_polyline_of_contiguous_run_has_no_gaps():
    xs = segments_to_polyline([0.0, 1.0, 2.0])
    assert not np.isnan(xs).any()
    assert xs.tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 3.0]


def test_segments_to_polyline_inserts_nan_between_runs():
    xs = segments_to_polyline([0.0, 1.0, 5.0, 6.0])
    assert np.isnan(xs).sum() == 1
    # The NaN separates the two runs rather than terminating the array.
    assert not np.isnan(xs[0]) and not np.isnan(xs[-1])


def test_segments_to_polyline_is_empty_for_no_input():
    assert segments_to_polyline([]).size == 0


def test_segments_to_polyline_applies_the_time_mapping():
    xs = segments_to_polyline([0.0, 1.0], xmap=lambda t: 3.0 * t)
    assert xs.tolist() == [0.0, 3.0, 3.0, 6.0]


# LineageGeometry
@pytest.fixture
def geometry(lineage_df, ident):
    tracks = build_track_table(lineage_df, ident)
    return LineageGeometry.from_tracks(tracks, build_edges(tracks))


def test_geometry_matches_the_maps_it_replaces(geometry):
    assert geometry.t_start[1] == 0.0
    assert geometry.t_end[3] == 30.0
    # Track 1 divides at 10, track 2 at 20; leaves never divide.
    assert geometry.parent_div == {1: 10.0, 2: 20.0}


def test_division_span_stops_at_the_division(geometry):
    assert geometry.division_span(1) == (0.0, 10.0)


def test_division_span_of_a_leaf_runs_to_its_end(geometry):
    assert geometry.division_span(3) == (10.0, 30.0)


def test_division_span_is_none_for_unknown_track(geometry):
    assert geometry.division_span(99) is None


def test_clamped_span_never_exceeds_the_track_end():
    """A division recorded after the parent's last frame.

    The plain tree draws past the end, the heatmap clamps. Both are kept.
    """
    tracks = pd.DataFrame(
        {"TrackNumber": [1], "t_start": [0.0], "t_end": [5.0]}
    )
    edges = pd.DataFrame({"parent": [1], "child": [2], "t_div": [9.0]})
    geom = LineageGeometry.from_tracks(tracks, edges)
    assert geom.division_span(1) == (0.0, 9.0)
    assert geom.clamped_span(1) == (0.0, 5.0)


def test_renderers_use_the_unclamped_span():
    """A child normally starts on the frame after the parent's last one."""
    tracks = pd.DataFrame(
        {"TrackNumber": [1], "t_start": [0.0], "t_end": [13.0]}
    )
    edges = pd.DataFrame({"parent": [1], "child": [2], "t_div": [14.0]})
    geom = LineageGeometry.from_tracks(tracks, edges)

    values = {float(t): 1.0 for t in range(14)}
    unclamped = plan_heat_segments(values, *geom.division_span(1), None, 0.12)
    clamped = plan_heat_segments(values, *geom.clamped_span(1), None, 0.12)

    assert unclamped[-1].x1 - unclamped[-1].x0 == pytest.approx(1.24)
    assert clamped[-1].x1 - clamped[-1].x0 == pytest.approx(0.12)


def test_geometry_from_empty_edges(lineage_df, ident):
    tracks = build_track_table(lineage_df, ident)
    geom = LineageGeometry.from_tracks(tracks, pd.DataFrame())
    assert geom.parent_div == {}
    assert geom.division_span(1) == (0.0, 10.0)


# Heat segment planning


def _reference_segments(tm, t0_raw, t1_raw, xmap, pad):
    """The pre-refactor algorithm, transcribed verbatim for comparison."""
    out = []
    if (
        (not np.isfinite(t0_raw))
        or (not np.isfinite(t1_raw))
        or (t1_raw <= t0_raw)
    ):
        return out

    first_full = int(np.ceil(t0_raw))
    last_full = int(np.floor(t1_raw))

    key0 = float(int(np.floor(t0_raw)))
    if key0 in tm:
        v0 = tm[key0]
        seg_s = xmap(float(t0_raw))
        seg_e = xmap(float(min(t1_raw, first_full))) + pad
        if seg_e > seg_s:
            out.append((seg_s, seg_e, v0))

    for t in range(first_full, last_full):
        key = float(t)
        if key not in tm:
            continue
        v = tm[key]
        seg_s = xmap(float(t)) - pad
        seg_e = xmap(float(t + 1)) + pad
        if seg_e > seg_s:
            out.append((seg_s, seg_e, v))

    key1 = float(last_full)
    if key1 in tm:
        v1 = tm[key1]
        seg_s = xmap(float(max(t0_raw, last_full))) - pad
        seg_e = xmap(float(t1_raw))
        if seg_e > seg_s:
            out.append((seg_s, seg_e, v1))

    return out


@pytest.mark.parametrize("pad", [0.0, 0.12])
@pytest.mark.parametrize(
    "xmap",
    [None, lambda t: 2.5 * t, lambda t: t**1.5],
    ids=["identity", "linear", "nonlinear"],
)
def test_plan_heat_segments_matches_the_original_algorithm(pad, xmap):
    rng = np.random.default_rng(1234)
    effective = (lambda t: float(t)) if xmap is None else xmap

    for _ in range(300):
        t0 = float(rng.uniform(0, 20))
        t1 = t0 + float(rng.uniform(-1, 15))
        frames = range(int(np.floor(t0)) - 1, int(np.ceil(t1)) + 2)
        tm = {
            float(f): float(rng.normal())
            for f in frames
            if rng.random() > 0.25
        }

        got = plan_heat_segments(tm, t0, t1, xmap, pad)
        want = _reference_segments(tm, t0, t1, effective, pad)

        assert len(got) == len(want)
        for g, w in zip(got, want, strict=True):
            assert g.x0 == pytest.approx(w[0])
            assert g.x1 == pytest.approx(w[1])
            assert g.value == pytest.approx(w[2])


def test_plan_heat_segments_empty_for_inverted_interval():
    assert plan_heat_segments({0.0: 1.0}, 5.0, 5.0) == []
    assert plan_heat_segments({0.0: 1.0}, 5.0, 4.0) == []


def test_plan_heat_segments_empty_for_non_finite_bounds():
    assert plan_heat_segments({0.0: 1.0}, np.nan, 4.0) == []
    assert plan_heat_segments({0.0: 1.0}, 0.0, np.inf) == []


def test_plan_heat_segments_skips_frames_with_no_value():
    segments = plan_heat_segments({0.0: 1.0, 2.0: 3.0}, 0.0, 3.0)
    assert [s.value for s in segments] == [1.0, 3.0]


def test_plan_heat_segments_covers_whole_frames_once():
    segments = plan_heat_segments({0.0: 10.0, 1.0: 11.0, 2.0: 12.0}, 0.0, 3.0)
    assert [(s.x0, s.x1) for s in segments] == [
        (0.0, 1.0),
        (1.0, 2.0),
        (2.0, 3.0),
    ]


# Value maps and scaling
def test_build_value_map_keys_by_track_then_time(lineage_df, ident):
    sub = lineage_df[lineage_df["Identification"] == ident]
    vm = build_value_map(sub, "MeanNoBgCorrectedCh1M1")
    assert set(vm) == {1, 2, 3, 4, 5}
    assert vm[1][0.0] == pytest.approx(1000.0 + 0.0 + 1)


def test_build_value_map_honours_valid_times(lineage_df, ident):
    sub = lineage_df[lineage_df["Identification"] == ident]
    vm = build_value_map(
        sub, "MeanNoBgCorrectedCh1M1", valid_times={1: {0.0, 1.0}}
    )
    assert set(vm[1]) == {0.0, 1.0}
    # Tracks absent from valid_times are left unfiltered.
    assert len(vm[2]) == 11


def test_build_value_map_drops_non_finite_values(lineage_df, ident):
    sub = lineage_df[lineage_df["Identification"] == ident].copy()
    sub.loc[sub["t"] == 5, "MeanNoBgCorrectedCh1M1"] = np.nan
    vm = build_value_map(sub, "MeanNoBgCorrectedCh1M1")
    assert 5.0 not in vm[1]


def test_build_value_map_returns_empty_for_missing_column(lineage_df, ident):
    sub = lineage_df[lineage_df["Identification"] == ident]
    assert build_value_map(sub, "NoSuchColumn") == {}


def test_value_range_falls_back_on_flat_input():
    assert value_range([3.0, 3.0, 3.0]) == (0.0, 1.0)


def test_value_range_falls_back_on_all_nan():
    assert value_range([np.nan, np.nan]) == (0.0, 1.0)


def test_value_range_ignores_nan_among_real_values():
    assert value_range([1.0, np.nan, 9.0]) == (1.0, 9.0)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, 0.0), (5.0, 0.5), (10.0, 1.0), (-3.0, 0.0), (99.0, 1.0)],
)
def test_normalize_clamps_to_unit_interval(value, expected):
    assert normalize(value, 0.0, 10.0) == pytest.approx(expected)


def test_normalize_maps_non_finite_to_zero():
    assert normalize(np.nan, 0.0, 10.0) == 0.0


# Feature catalog
def test_catalog_finds_channel_and_mask_features(lineage_df):
    catalog = FeatureCatalog.from_columns(list(lineage_df.columns))
    assert "MeanNoBgCorrected" in catalog.features
    assert catalog.has_channel["MeanNoBgCorrected"] is True
    assert catalog.channels_for("MeanNoBgCorrected") == [1, 2]
    assert catalog.masks_for("MeanNoBgCorrected") == [1]


def test_catalog_marks_maskonly_features_as_channelless(lineage_df):
    catalog = FeatureCatalog.from_columns(list(lineage_df.columns))
    assert catalog.has_channel["AreaMorphology"] is False
    assert catalog.resolve("AreaMorphology", 1, None) == ["AreaMorphologyM1"]


def test_catalog_resolve_single_channel(lineage_df):
    catalog = FeatureCatalog.from_columns(list(lineage_df.columns))
    assert catalog.resolve("MeanNoBgCorrected", 1, 2) == [
        "MeanNoBgCorrectedCh2M1"
    ]


def test_catalog_resolve_all_channels_is_ordered(lineage_df):
    catalog = FeatureCatalog.from_columns(list(lineage_df.columns))
    assert catalog.resolve("MeanNoBgCorrected", 1, -1) == [
        "MeanNoBgCorrectedCh1M1",
        "MeanNoBgCorrectedCh2M1",
    ]


def test_catalog_resolve_returns_empty_for_unknown_selection(lineage_df):
    catalog = FeatureCatalog.from_columns(list(lineage_df.columns))
    assert catalog.resolve("MeanNoBgCorrected", 9, 1) == []
    assert catalog.resolve("NoSuchFeature", 1, 1) == []
    assert catalog.resolve("MeanNoBgCorrected", None, 1) == []


def test_catalog_excludes_position_columns_from_the_dropdown(lineage_df):
    catalog = FeatureCatalog.from_columns(list(lineage_df.columns))
    selectable = catalog.selectable_features()
    assert "XMorphology" not in selectable
    assert "YMorphology" not in selectable
    assert "MeanNoBgCorrected" in selectable


def test_catalog_heat_gate_accepts_any_per_channel_column():
    assert FeatureCatalog.from_columns(["MeanRawCh1M1"]).heat_columns_present
    assert FeatureCatalog.from_columns(
        ["AreaMorphologyM1", "SumNoBgCorrectedCh1M1"]
    ).heat_columns_present


def test_catalog_heat_gate_needs_a_channel_dimension():
    assert not FeatureCatalog.from_columns(
        ["AreaMorphologyM1", "PerimeterMorphologyM2"]
    ).heat_columns_present


def test_catalog_sees_lowercase_mask_suffixes():
    catalog = FeatureCatalog.from_columns(["step_disp_px_m1"])
    assert "step_disp_px_" in catalog.features
    assert catalog.resolve("step_disp_px_", 1, None) == ["step_disp_px_m1"]


def test_catalog_excludes_label_id_under_either_spelling():
    catalog = FeatureCatalog.from_columns(["label_id_m1", "nn_dist_px_m2"])
    assert catalog.selectable_features() == []


_DERIVED_COL = "MeanNoBgCorrectedCh1M1/MeanNoBgCorrectedCh2M1"
_DERIVED_KEY = "MeanNoBgCorrectedCh1M1 ÷ MeanNoBgCorrectedCh2M1"
_REGISTRY = {
    _DERIVED_KEY: {"template": _DERIVED_COL, "has_ch": False, "has_m": False}
}


def test_derived_metric_is_not_mangled_by_the_regex():
    catalog = FeatureCatalog.from_columns([_DERIVED_COL], _REGISTRY)
    assert catalog.features == {_DERIVED_KEY}


def test_derived_metric_resolves_without_a_mask():
    catalog = FeatureCatalog.from_columns([_DERIVED_COL], _REGISTRY)
    assert catalog.has_mask[_DERIVED_KEY] is False
    assert catalog.masks_for(_DERIVED_KEY) == []
    assert catalog.resolve(_DERIVED_KEY, None, None) == [_DERIVED_COL]


def test_derived_metric_does_not_disturb_normal_features(lineage_df):
    columns = [*lineage_df.columns, _DERIVED_COL]
    catalog = FeatureCatalog.from_columns(columns, _REGISTRY)
    assert _DERIVED_KEY in catalog.features
    assert catalog.resolve("MeanNoBgCorrected", 1, 2) == [
        "MeanNoBgCorrectedCh2M1"
    ]


def test_unregistered_operator_column_is_ignored():
    catalog = FeatureCatalog.from_columns([_DERIVED_COL])
    assert catalog.features == set()


def test_catalog_is_falsy_when_no_features_found():
    assert not FeatureCatalog.from_columns(["t", "TrackNumber"])
