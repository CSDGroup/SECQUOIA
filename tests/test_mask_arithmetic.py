"""Tests for SECQUOIA mask arithmetic.

These tests use the same tiny artificial geometry as ``test_quantify.py``:

- 10 x 10 frames, one time point unless a test needs more
- mask 1 is bigger: 3 x 3 pixels at [2:5, 2:5], area 9
- mask 2 is smaller: 2 x 2 pixels at [2:4, 2:4], area 4
- mask 2 sits fully inside mask 1

Because mask 2 is nested inside mask 1, the bitwise operators have results
that can be written down by hand:

    INT  ->  the 2 x 2 region            area 4
    UNI  ->  the 3 x 3 region            area 9
    ME   ->  3 x 3 minus 2 x 2, L-shape  area 5

Dilation uses ``skimage.morphology.disk``, which for radius 1 is a plus
shape, not a 3 x 3 square:

    0 1 0
    1 1 1
    0 1 0

So dilating the 3 x 3 block by 1 gives a 5 x 5 block with the four corners
missing: area 21, not 25. The expected arrays below are written out in full
so this is visible rather than implied.
"""

from __future__ import annotations

import numpy as np
import pytest

from SECQUOIA.core.segmentation.mask_arithmetic import (
    compute_bitwise_mask,
    connected_components_stack,
    relabel_separated,
)

FRAME = (10, 10)


def make_stack(*regions: tuple[slice, slice], label_value: int = 1):
    """Build a single time point label stack with the given regions filled."""
    stack = np.zeros((1, *FRAME), dtype=np.uint16)
    for rows, cols in regions:
        stack[0, rows, cols] = label_value
    return stack


def big_mask():
    """The 3 x 3 mask, area 9."""
    return make_stack((slice(2, 5), slice(2, 5)))


def small_mask():
    """The 2 x 2 mask, area 4, nested inside the 3 x 3 mask."""
    return make_stack((slice(2, 4), slice(2, 4)))


def binary_of(stack) -> np.ndarray:
    """Reduce a label stack to booleans, for shape comparisons."""
    return stack[0] > 0


def filled_coordinates(stack) -> set[tuple[int, int]]:
    """Coordinates of every non-zero pixel in the first frame."""
    return {(int(r), int(c)) for r, c in np.argwhere(stack[0] > 0)}


def assert_regions_equal(
    stack, expected_coords: set, message: str = ""
) -> None:
    """Assert the filled region matches an explicit coordinate set."""
    observed = filled_coordinates(stack)

    assert observed == expected_coords, (
        f"{message}\n"
        f"Missing:   {sorted(expected_coords - observed)}\n"
        f"Unexpected:{sorted(observed - expected_coords)}\n"
        f"Frame:\n{stack[0][0:8, 0:8]}"
    )


def block(r0: int, r1: int, c0: int, c1: int) -> set[tuple[int, int]]:
    """Coordinate set for a rectangular block."""
    return {(r, c) for r in range(r0, r1) for c in range(c0, c1)}


def test_intersection_of_nested_masks_is_the_small_mask():
    result = compute_bitwise_mask(
        big_mask(), small_mask(), "INT", False, False, 0, 0
    )

    assert_regions_equal(
        result, block(2, 4, 2, 4), "INT of nested masks must be the 2 x 2."
    )
    assert int((result > 0).sum()) == 4


def test_union_of_nested_masks_is_the_big_mask():
    result = compute_bitwise_mask(
        big_mask(), small_mask(), "UNI", False, False, 0, 0
    )

    assert_regions_equal(
        result, block(2, 5, 2, 5), "UNI of nested masks must be the 3 x 3."
    )
    assert int((result > 0).sum()) == 9


def test_exclusive_or_of_nested_masks_is_an_l_shape():
    """3 x 3 minus the nested 2 x 2 leaves five pixels in an L."""
    result = compute_bitwise_mask(
        big_mask(), small_mask(), "ME", False, False, 0, 0
    )

    expected = block(2, 5, 2, 5) - block(2, 4, 2, 4)
    assert_regions_equal(result, expected, "ME must leave the L-shape.")
    assert int((result > 0).sum()) == 5

    expected_frame = np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
        ]
    )
    observed_frame = (result[0][2:5, 2:5] > 0).astype(int)

    assert np.array_equal(
        observed_frame, expected_frame
    ), f"Expected L-shape:\n{expected_frame}\nObserved:\n{observed_frame}"


def test_op_none_returns_the_first_mask_relabelled():
    result = compute_bitwise_mask(big_mask(), None, "NONE", False, False, 0, 0)
    assert_regions_equal(result, block(2, 5, 2, 5))
    assert sorted(np.unique(result).tolist()) == [0, 1]


def test_not_a_inverts_the_first_mask():
    """Inverting the 3 x 3 leaves the rest of the 10 x 10 frame: 100 - 9."""
    result = compute_bitwise_mask(big_mask(), None, "NONE", True, False, 0, 0)

    assert int((result > 0).sum()) == 91
    assert filled_coordinates(result).isdisjoint(block(2, 5, 2, 5))


def test_intersection_with_inverted_small_mask_is_the_l_shape():
    """big AND NOT small is the same L-shape that ME produces here."""
    result = compute_bitwise_mask(
        big_mask(), small_mask(), "INT", False, True, 0, 0
    )

    expected = block(2, 5, 2, 5) - block(2, 4, 2, 4)
    assert_regions_equal(result, expected)


def test_dilation_by_one_uses_a_disk_footprint():
    """The 3 x 3 grows to a 5 x 5 minus its four corners: area 21."""
    result = compute_bitwise_mask(big_mask(), None, "NONE", False, False, 1, 0)

    expected = block(1, 6, 1, 6) - {(1, 1), (1, 5), (5, 1), (5, 5)}
    assert_regions_equal(
        result, expected, "Dilation by 1 must use the plus-shaped disk(1)."
    )
    assert int((result > 0).sum()) == 21

    # Rows 1-5, columns 1-5, written out:
    expected_frame = np.array(
        [
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ]
    )
    observed_frame = (result[0][1:6, 1:6] > 0).astype(int)

    assert np.array_equal(
        observed_frame, expected_frame
    ), f"Expected:\n{expected_frame}\nObserved:\n{observed_frame}"


def test_erosion_by_one_leaves_the_centre_pixel():
    """Eroding the 3 x 3 with disk(1) leaves only the centre: area 1."""
    result = compute_bitwise_mask(
        big_mask(), None, "NONE", False, False, -1, 0
    )
    assert_regions_equal(result, {(3, 3)}, "Erosion must leave the centre.")


def test_zero_dilation_is_a_no_op():
    result = compute_bitwise_mask(big_mask(), None, "NONE", False, False, 0, 0)
    assert_regions_equal(result, block(2, 5, 2, 5))


def test_dilation_applies_to_each_side_independently():
    """dil1v grows mask A only; dil2v grows mask B only.

    Dilating only the small mask makes it overflow the big one, so the union
    becomes larger than the big mask alone.
    """
    result = compute_bitwise_mask(
        big_mask(), small_mask(), "UNI", False, False, 0, 1
    )

    # small (2 x 2 at [2:4, 2:4]) dilated by disk(1), unioned with the 3 x 3.
    dilated_small = block(1, 5, 2, 4) | block(2, 4, 1, 5)
    expected = block(2, 5, 2, 5) | dilated_small

    assert_regions_equal(result, expected)


def test_dilation_over_multiple_timepoints():
    """Each time slice is dilated independently."""
    stack = np.zeros((3, *FRAME), dtype=np.uint16)
    stack[0, 2:5, 2:5] = 1  # 3 x 3
    stack[1] = 0  # empty frame
    stack[2, 5:7, 5:7] = 1  # 2 x 2 elsewhere

    result = compute_bitwise_mask(stack, None, "NONE", False, False, 1, 0)

    assert int((result[0] > 0).sum()) == 21
    assert int((result[1] > 0).sum()) == 0
    assert int((result[2] > 0).sum()) == 12


def adjacent_pair():
    """Two 2 x 2 blocks side by side, touching along one edge."""
    left = make_stack((slice(2, 4), slice(2, 4)))
    right = make_stack((slice(2, 4), slice(4, 6)))
    return left, right


def test_union_of_touching_masks_merges_without_keep_separate():
    left, right = adjacent_pair()

    result = compute_bitwise_mask(
        left, right, "UNI", False, False, 0, 0, keep_separate=False
    )

    labels = sorted(int(v) for v in np.unique(result) if v)
    assert labels == [
        1
    ], f"Touching objects must merge into one label, got {labels}."
    assert int((result > 0).sum()) == 8


def test_union_of_touching_masks_stays_split_with_keep_separate():
    left, right = adjacent_pair()

    result = compute_bitwise_mask(
        left, right, "UNI", False, False, 0, 0, keep_separate=True
    )

    labels = sorted(int(v) for v in np.unique(result) if v)
    assert labels == [1, 2], (
        f"keep_separate must preserve two objects, got {labels}.\n"
        f"Frame:\n{result[0][1:5, 1:7]}"
    )

    # Each keeps its own four pixels; the seed offset makes the right block 2.
    assert filled_coordinates(result) == block(2, 4, 2, 6)
    assert int((result == 1).sum()) == 4
    assert int((result == 2).sum()) == 4
    assert set(zip(*np.where(result[0] == 1), strict=False)) == block(
        2, 4, 2, 4
    )
    assert set(zip(*np.where(result[0] == 2), strict=False)) == block(
        2, 4, 4, 6
    )


def test_keep_separate_dilation_fast_path_preserves_input_labels():
    """keep_separate + NONE + dil > 0 takes the expand_labels fast path.

    That path skips connected_components_stack, so the original label numbers
    survive instead of being renumbered from scratch.
    """
    stack = np.zeros((1, *FRAME), dtype=np.uint16)
    stack[0, 2:4, 2:4] = 1
    stack[0, 2:4, 6:8] = 2

    result = compute_bitwise_mask(
        stack, None, "NONE", False, False, 1, 0, keep_separate=True
    )

    assert sorted(int(v) for v in np.unique(result) if v) == [1, 2]
    # Each 2 x 2 grows by disk(1) into 12 pixels.
    assert int((result == 1).sum()) == 12
    assert int((result == 2).sum()) == 12


def test_keep_separate_falls_back_when_no_seeds_survive():
    """With both inputs inverted there are no seed stacks, so plain labelling
    is used and the result is still a valid single component."""
    empty = np.zeros((1, *FRAME), dtype=np.uint16)

    result = relabel_separated((empty > 0) | (big_mask() > 0), [])

    assert sorted(int(v) for v in np.unique(result) if v) == [1]


def test_connected_components_labels_each_slice_independently():
    """Labels restart per frame; they are not tracked across time."""
    stack = np.zeros((2, *FRAME), dtype=bool)
    stack[0, 2:4, 2:4] = True
    stack[1, 6:8, 6:8] = True

    result = connected_components_stack(stack)

    assert sorted(int(v) for v in np.unique(result[0]) if v) == [1]
    assert sorted(int(v) for v in np.unique(result[1]) if v) == [1]


def test_connected_components_separates_disjoint_objects():
    frame = np.zeros(FRAME, dtype=bool)
    frame[1:3, 1:3] = True
    frame[6:8, 6:8] = True

    result = connected_components_stack(frame)

    assert sorted(int(v) for v in np.unique(result) if v) == [1, 2]


def test_connected_components_accepts_2d_input():
    frame = np.zeros(FRAME, dtype=bool)
    frame[2:5, 2:5] = True

    result = connected_components_stack(frame)

    assert result.shape == FRAME
    assert int((result > 0).sum()) == 9


def test_connected_components_rejects_4d_input():
    with pytest.raises(ValueError, match="2D or 3D"):
        connected_components_stack(np.zeros((2, 2, 10, 10), dtype=bool))


def test_missing_second_mask_raises():
    with pytest.raises(ValueError, match="Second mask is required"):
        compute_bitwise_mask(big_mask(), None, "INT", False, False, 0, 0)


def test_unsupported_operator_raises():
    with pytest.raises(ValueError, match="Unsupported op"):
        compute_bitwise_mask(
            big_mask(), small_mask(), "NAND", False, False, 0, 0
        )


def test_operator_is_case_insensitive():
    lower = compute_bitwise_mask(
        big_mask(), small_mask(), "int", False, False, 0, 0
    )
    upper = compute_bitwise_mask(
        big_mask(), small_mask(), "INT", False, False, 0, 0
    )

    assert np.array_equal(lower, upper)


@pytest.mark.smoke
def test_mask_arithmetic_imports():
    assert callable(compute_bitwise_mask)
    assert callable(connected_components_stack)
    assert callable(relabel_separated)
