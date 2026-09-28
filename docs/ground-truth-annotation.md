# Ground truth annotation

Developers can annotate by hand whether each tracked point is linked to the right mask.

## How to activate ground truth annotation

Set `SECQUOIA_GT_ANNOTATION` before starting SECQUOIA:

```bash
SECQUOIA_GT_ANNOTATION=1 SECQUOIA
```

- `1`: one annotation bar above the cell fate buttons.
- `2`: one annotation bar under each viewer.

## Annotating

- Every tracked point and mask is listed in the CSV for the current position.
- A mask with an assigned label starts as `Correct`, and a mask without one starts as `Missed`.
- Choose the `Mask` in the dropdown, then click:
    - `Correct`: the assigned mask is right.
    - `Incorrect`: the assigned mask is wrong.
    - `Missed`: no mask is assigned.
- `X` accepts all open masks of the current time point of the current Tree-ID as they are and jumps to the next open time point.
- Enter a comment in the `Notes` field and press `Enter` to save it.
- Right-click, paint, erase, undo and redo record the label under the tracking point as `checked_label_id`. If it differs from the assigned label, the row becomes `Incorrect`.

## Output

`<experiment>/Analysis/SECQUOIA_files_<format>/<Project name>/ground_truth_annotations.csv`

- One file per project, shared by all positions.

| Column | Description |
| --- | --- |
| `assigned_label_id` | Label assigned by SECQUOIA (`0` = none) |
| `checked_label_id` | Label you confirmed (`0` = none) |
| `alt_label_id`, `alt_distance_px` | Second nearest mask and its distance |
| `category` | `Correct`, `Incorrect` or `Missed` |
| `notes`, `timestamp` | Your note; time of the last review (empty if never reviewed) |
