"""Outlier detection: rules and detection algorithms, dataframe sync, and single-point recheck."""

from __future__ import annotations

from SECQUOIA.core.outlier_detection.close_masks import (
    apply_close_mask_detection,
    find_close_mask_cases,
    sync_close_mask_flag_to_track_df,
    update_unique_close_mask_ids,
)
from SECQUOIA.core.outlier_detection.detection import (
    RulesPack,
    _outlier_rules_dir,
    load_outlier_rules_from_disk,
    resolve_feature_columns,
    rule_is_active,
    run_outlier_pipeline,
    run_sliding_windows,
    run_threshold_rules,
    save_outlier_rules_to_disk,
    threshold_values_set,
)
from SECQUOIA.core.outlier_detection.recheck import (
    _normalize_unique_ids,
    apply_outlier_selection_for_current_point,
    resolve_current_ident,
)
from SECQUOIA.core.outlier_detection.track_sync import (
    update_outlier_detection_in_track_df,
    update_unique_outliers_ids,
)

__all__ = [
    "RulesPack",
    "_normalize_unique_ids",
    "_outlier_rules_dir",
    "apply_close_mask_detection",
    "apply_outlier_selection_for_current_point",
    "find_close_mask_cases",
    "load_outlier_rules_from_disk",
    "resolve_current_ident",
    "resolve_feature_columns",
    "rule_is_active",
    "run_outlier_pipeline",
    "run_sliding_windows",
    "run_threshold_rules",
    "save_outlier_rules_to_disk",
    "sync_close_mask_flag_to_track_df",
    "threshold_values_set",
    "update_outlier_detection_in_track_df",
    "update_unique_close_mask_ids",
    "update_unique_outliers_ids",
]
