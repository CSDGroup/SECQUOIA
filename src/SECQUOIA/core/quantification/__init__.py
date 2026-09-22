"""Cell quantification pipeline.

``quantify`` is the public entry point. For the labels of one position it
measures per (mask, channel, frame) regionprops (``measure``), matches the
resulting objects onto existing tracks by nearest neighbour (``matching``),
and attaches lineage displacement metrics (``lineage``).
"""

import logging

import pandas as pd

from SECQUOIA.core.project_state import save_position_measurements
from SECQUOIA.core.quantification.columns import (
    drop_stale_measurement_columns,
    expected_measurement_columns,
    init_mask_channel_columns,
    sort_measurement_columns,
    stale_measurement_columns,
)
from SECQUOIA.core.quantification.image_layers import (
    _collect_label_stacks,
    _count_frame_steps,
)
from SECQUOIA.core.quantification.lineage import (
    add_lineage_step_distance,
    compute_step_distance_for_row,
)
from SECQUOIA.core.quantification.matching import (
    _prepare_track_columns,
    assign_objects_to_tracks,
    resolve_match_strategy,
    zero_fill_present_frames,
)
from SECQUOIA.core.quantification.measure import (
    measure_objects,
    measure_single_object,
    sum_intensity,
)
from SECQUOIA.core.quantification.naming import (
    CLOSE_MASK_FLAG,
    FeatureNaming,
    intensity_column,
    label_column,
    safe_op_series,
)
from SECQUOIA.core.quantification.progress import ProgressReporter
from SECQUOIA.utils.profiling import active_stage
from SECQUOIA.utils.timing import calculate_time

# Public API of this package
__all__ = [
    "FeatureNaming",
    "compute_step_distance_for_row",
    "drop_stale_measurement_columns",
    "expected_measurement_columns",
    "intensity_column",
    "label_column",
    "measure_single_object",
    "quantify",
    "safe_op_series",
    "stale_measurement_columns",
    "sum_intensity",
]

LOG = logging.getLogger(__name__)

# Number of post processing stages reported to `progress_cb` after measuring.
POST_PROCESSING_STEPS = 8


def _position_has_identifications(main_window, position, id_col: str) -> bool:
    """Return True if the tracks at `position` carry an identification.

    Only the first of `id_col` / ``"Identification"`` that exists as a
    column is consulted; an empty one is not retried against the other.
    """
    pos_rows = main_window.track_df.loc[
        main_window.track_df.get("Position", -1) == position
    ]
    for candidate in (id_col, "Identification"):
        if candidate in pos_rows.columns:
            return bool(
                pos_rows[candidate]
                .dropna()
                .astype(str)
                .str.strip()
                .ne("")
                .any()
            )
    return False


def _configured_mask_indices(main_window, measured_indices) -> range:
    """Mask indices the current configuration defines, for column pruning.

    Pruning must not punish a mask that merely failed to load this time:
    ``n_masks`` is the configured count, so a mask missing from
    `measured_indices` keeps its columns unless the configuration itself
    dropped it.
    """
    configured = int(getattr(main_window, "n_masks", 0) or 0)
    return range(1, max(configured, max(measured_indices, default=0)) + 1)


def _write_empty_measurement_columns(
    main_window, position, mask_indices, naming: FeatureNaming, id_col: str
) -> None:
    track_df = main_window.track_df
    pos_rows = track_df[track_df["Position"] == position]
    if pos_rows.empty:
        return

    pos_rows = init_mask_channel_columns(pos_rows.copy(), mask_indices, naming)
    main_window.track_df = pd.concat(
        [track_df[track_df["Position"] != position], pos_rows],
        ignore_index=True,
    )
    main_window.track_df = drop_stale_measurement_columns(
        main_window.track_df,
        naming,
        _configured_mask_indices(main_window, mask_indices),
        log_prefix="quantify",
    )
    main_window.track_df = sort_measurement_columns(
        main_window.track_df, mask_indices, naming, id_col
    )


def _recompute_derived_metrics(main_window, position) -> None:
    """Run the main window's derived-column hook for every row of `position`."""
    recompute_row = getattr(main_window, "_recompute_derived_for_row", None)
    if not callable(recompute_row):
        return
    try:
        df = main_window.track_df
        for idx in df.index[df["Position"] == position]:
            recompute_row(idx, mask_idx=None)
    except (
        RuntimeError,
        AttributeError,
        TypeError,
        KeyError,
        ValueError,
    ) as err:
        LOG.error(
            "[derived] recompute (position=%s) failed: %s", position, err
        )


def _merge_objects_into_tracks(
    main_window,
    objects_df: pd.DataFrame,
    mask_indices,
    naming: FeatureNaming,
    *,
    position,
    max_pixel_distance: float,
    one_to_one: bool,
    match_strategy: str,
    progress: ProgressReporter,
) -> pd.DataFrame:
    """Return the rows of `position` with the object measurements attached."""
    main_window.track_df = _ensure_consistent_types(main_window.track_df)
    objects_df = _ensure_consistent_types(objects_df)

    existing_rows = main_window.track_df[
        main_window.track_df["Position"] == position
    ].copy()

    progress.stage("Merging with tracks…")

    if existing_rows.empty:
        return init_mask_channel_columns(
            objects_df.copy(), mask_indices, naming
        )

    merged_df = init_mask_channel_columns(existing_rows, mask_indices, naming)
    merged_df = _prepare_track_columns(
        merged_df, objects_df, mask_indices, naming
    )
    merged_df = assign_objects_to_tracks(
        merged_df,
        objects_df,
        mask_indices,
        naming,
        max_pixel_distance=max_pixel_distance,
        one_to_one=one_to_one,
        strategy=match_strategy,
    )
    merged_df = _resolve_column_conflicts(merged_df)
    merged_df = merged_df.drop(columns=[CLOSE_MASK_FLAG], errors="ignore")

    progress.stage("Post-pass fill…")
    return zero_fill_present_frames(
        main_window, merged_df, mask_indices, naming
    )


def _replace_position_rows(
    main_window, merged_df: pd.DataFrame, position
) -> None:
    """Swap the rows of `position` in ``track_df`` for `merged_df`."""
    merged_df = merged_df.copy().assign(active=1, inspected=0)
    if main_window.tracking_format != "tTt":
        merged_df = merged_df.assign(Cellfate="Healthy")

    others = main_window.track_df[main_window.track_df["Position"] != position]
    main_window.track_df = pd.concat([others, merged_df], ignore_index=True)
    calculate_time(main_window)


def quantify(
    main_window,
    *,
    label_src_attr: str = "labels",
    max_pixel_distance: float | None = None,
    one_to_one: bool = True,
    match_strategy: str | None = None,
    id_col: str = "ID",
    progress_cb=None,
) -> None:
    """Measure per mask, per channel intensities into ``track_df``.

    Pipeline:

    1. `measure_objects` - regionprops per (mask, channel, frame),
       aggregated by label into one row per segmented object.
    2. `_merge_objects_into_tracks` - nearest-neighbour matching of the
       objects onto the existing tracks at the same time point `t`.
       `match_strategy` (``"sorted_pairs"``, ``"legacy"`` or ``"optimal"``)
       decides how tracks competing for one object are resolved; without
       one, the ``SECQUOIA_MATCH_STRATEGY`` environment variable is used,
       else ``"sorted_pairs"``.
    3. Derived metrics, stale-column pruning, column ordering, lineage
       displacement and the per position CSV export.
    """
    naming = FeatureNaming.from_main_window(main_window)
    match_strategy = resolve_match_strategy(match_strategy)

    label_entries = _collect_label_stacks(
        getattr(main_window, label_src_attr, None)
    )
    if not label_entries:
        LOG.error("No label stacks found in `%s`. Aborting.", label_src_attr)
        return

    mask_indices = sorted(mask_idx for mask_idx, _ in label_entries)
    if max_pixel_distance is None:
        max_pixel_distance = main_window.threshold
    min_area_pixels = main_window.min_mask_size
    position = main_window.current_position_number

    progress = ProgressReporter(
        progress_cb,
        _count_frame_steps(main_window, label_entries) + POST_PROCESSING_STEPS,
    )

    # 1 Measure
    with active_stage("measure"):
        object_rows = measure_objects(
            main_window,
            label_entries,
            naming,
            position=position,
            min_area_pixels=min_area_pixels,
            progress=progress,
        )
    if not object_rows:
        LOG.warning(
            "No measurements computed (no labeled regions after size filtering?)."
        )
        _write_empty_measurement_columns(
            main_window, position, mask_indices, naming, id_col
        )
        return

    objects_df = pd.DataFrame(object_rows)
    progress.stage("Aggregating objects…")

    if not _position_has_identifications(main_window, position, id_col):
        LOG.warning(
            "Skip merge/write for Position %s: no Identification present.",
            position,
        )
        return

    # 2 Match objects to tracks
    with active_stage("matching"):
        merged_df = _merge_objects_into_tracks(
            main_window,
            objects_df,
            mask_indices,
            naming,
            position=position,
            max_pixel_distance=max_pixel_distance,
            one_to_one=one_to_one,
            match_strategy=match_strategy,
            progress=progress,
        )

    # 3 Finalise
    _replace_position_rows(main_window, merged_df, position)
    main_window.track_df = drop_stale_measurement_columns(
        main_window.track_df,
        naming,
        _configured_mask_indices(main_window, mask_indices),
        log_prefix="quantify",
    )
    progress.stage("Finalizing table…")

    _recompute_derived_metrics(main_window, position)
    progress.stage("Recomputing derived metrics…")

    measured_masks = sorted(objects_df["__mask_idx__"].unique().tolist())
    main_window.track_df = sort_measurement_columns(
        main_window.track_df, measured_masks, naming, id_col
    )
    progress.stage("Sorting columns…")

    add_lineage_step_distance(main_window, measured_masks)
    progress.stage("Computing displacements…")

    saved = save_position_measurements(
        main_window, position, len(label_entries)
    )
    if saved:
        progress.stage("Saving results…")

    LOG.info(
        "Fluorescence measurement done for Position %s | labels=%d | channels=%s | "
        "min_area=%spx | tolerance=%spx | one_to_one=%s | match_strategy=%s",
        position,
        len(label_entries),
        main_window.n_channels,
        min_area_pixels,
        max_pixel_distance,
        one_to_one,
        match_strategy,
    )
    progress.finish()


def _ensure_consistent_types(df) -> pd.DataFrame:
    """Coerce `Position` to int so tracks and objects compare equal."""
    df["Position"] = df["Position"].astype(int, errors="ignore")
    return df


def _resolve_column_conflicts(df, suffix="_new") -> pd.DataFrame:
    """Fold any ``<base><suffix>`` column back into ``<base>``.

    A safeguard rather than a step of the current pipeline: nothing in
    `quantify` writes suffixed columns any more.
    """
    for col in df.columns:
        if col.endswith(suffix):
            base_col = col[: -len(suffix)]
            if base_col in df.columns:
                df[base_col] = df[col].combine_first(df[base_col])
            df.drop(columns=[col], inplace=True)
    return df
