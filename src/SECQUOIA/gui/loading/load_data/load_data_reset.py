"""Clears main_window state before (re)loading an experiment or dataset."""

from __future__ import annotations

import contextlib

__all__ = ["_reset_dataset_state", "_reset_loaded_experiment"]


def _reset_dataset_state(main_window):
    """Clear channel/mask selection state before (re)loading an experiment."""
    for w in getattr(main_window, "FL_inputs", []):
        with contextlib.suppress(AttributeError, RuntimeError):
            w.deleteLater()
    main_window.FL_inputs = []
    main_window.available_channels = []
    main_window.ids_channels = []
    main_window.n_channels = 0
    main_window.segmentation_paths = []
    main_window.n_masks = 0


def _reset_loaded_experiment(main_window):
    """Discard everything produced by a previous load before loading a new experiment."""
    # image data
    main_window.images = {}
    main_window.corrected_images = None
    main_window.labels = []
    main_window.image_present = {}
    main_window._memmap_dir = None
    main_window._memmap_files = None
    main_window._corrected_u8_memmap_files = []

    # edit state
    main_window._edit_cache = {}
    main_window._corrected_slices = set()
    main_window.last_edit_info = None
    main_window._last_labels_layer_ref = None
    main_window._last_labels_layer_name = None
    main_window._last_labels_layer_t = None

    if getattr(main_window, "_viewers_dirty", False):
        for v in (
            getattr(main_window, "viewer_1", None),
            getattr(main_window, "viewer_2", None),
        ):
            if v is not None:
                with contextlib.suppress(RuntimeError, AttributeError):
                    v.layers.clear()
                    main_window.add_empty_segmentation_image_layer(v)
        main_window._viewers_dirty = False

    main_window._rt_wide = None

    # plot/feature caches
    main_window._feature_defs = {}
    main_window._derived_features = {}
    main_window.selected_feature_by_row = {}
    main_window.selected_m_by_channel = {}
    main_window.selected_ch_by_channel = {}
    main_window._features_defaulted = False
    main_window._loading_in_progress = False
