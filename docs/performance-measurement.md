# Performance measurement

SECQUOIA allows developers to record how long loading takes and how much memory it uses, so that runtime and RAM can be reported for growing datasets (more time points, channels, segmentation masks, tracked cells, or larger images) using [psutil](https://github.com/giampaolo/psutil).

## How to activate performance measurement

Set `SECQUOIA_MEASURE` before starting SECQUOIA:

```bash
SECQUOIA_MEASURE=ALL SECQUOIA_MEASURE_TAG=rep2 SECQUOIA
```

- `ALL` (or `1`) records everything.
- A comma-separated subset records only those groups, e.g. `SECQUOIA_MEASURE=time,memory`.
- `SECQUOIA_MEASURE_TAG`: free text copied into every row, e.g. a replicate number (`rep1`, `rep2`, …).
- `SECQUOIA_MEASURE_DIR`: where the CSV is written. Defaults to `<experiment>/Analysis/SECQUOIA_measure.csv`.
- `SECQUOIA_MEASURE_CROP_SIZE` & `SECQUOIA_MEASURE_CROP_OFFSET_PX`: metadata for the image dimension.

## What gets measured

One row is appended to the CSV for every position that finishes loading and quantification (`mode` is `"Positions"`, `"All"`, or `"Switch"`), and one further row for every interactive mask edit (`mode="Edit"`, see below). Columns:

| Group | Columns |
| --- | --- |
| Identity | `run_id`, `timestamp`, `tag`, `position`, `mode` |
| Scaling factors | `n_timepoints`, `n_channels`, `n_mask_sets`, `n_tracked_cells`, `n_objects_total`, `image_h`, `image_w`, `dtype`, `crop_size`, `crop_offset_px` |
| Time (s) | `t_total`, `t_load_channels`, `t_load_masks`, `t_bg_correction`, `t_measure`, `t_matching`, `t_quantify`, `t_save` |
| Memory (MB) | `rss_before`, `rss_peak_load_channels`, `rss_peak_load_masks`, `rss_peak_bg_correction`, `rss_peak_measure`, `rss_peak_matching`, `rss_peak_quantify`, `rss_after`, `rss_delta` |
| Disk (MB) | `memmap_bytes`, `analysis_dir_bytes`, `input_bytes_read` |
| Derived | `theoretical_bytes`, `ram_peak_to_theoretical_ratio`, `throughput_mb_s`, `sec_per_timepoint` |
| CPU | `cpu_time_s`, `cpu_util_pct`, `n_workers` |
| Environment | `os`, `os_version`, `cpu_model`, `cores`, `ram_total_gb`, `python_version`, `numpy_version`, `napari_version`, `qt_version`, `secquoia_version` |

### Mode

- `mode == "Positions"`: Single position loading.
- `mode == "All"`: Quantify all positions.
- `mode == "Switch"`: Moving to another position.
- `mode == "Edit"`: Interactive mask edits.
