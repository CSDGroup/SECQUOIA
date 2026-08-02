"""Bitwise combination and re-labelling of segmentation mask stacks."""

from __future__ import annotations

import numpy as np
from skimage.measure import label
from skimage.morphology import binary_dilation, binary_erosion, disk
from skimage.segmentation import expand_labels

__all__ = [
    "compute_bitwise_mask",
    "connected_components_stack",
    "relabel_separated",
]

_UNBOUNDED_DISTANCE = 1e9


def apply_mask_morphology(mask: np.ndarray, x: int) -> np.ndarray:
    """Dilate, erode, or no-op a mask depending on the sign of `x`.

    x > 0 → dilate by x pixels
    x < 0 → erode  by |x| pixels
    x = 0 → no-op
    """
    if x == 0:
        return mask.copy()

    selem = disk(abs(x))
    op_morph = binary_dilation if x > 0 else binary_erosion

    if mask.ndim == 2:
        out = op_morph(mask, footprint=selem)
    elif mask.ndim == 3:
        out = np.empty_like(mask, dtype=bool)
        for i in range(mask.shape[0]):
            out[i] = op_morph(mask[i], footprint=selem)
    else:
        raise ValueError("Unsupported mask ndim (only 2D or 3D supported).")

    return out


def connected_components_stack(binary_stack: np.ndarray) -> np.ndarray:
    """Label connected components in a 2D mask, or per time slice of a 3D ``(T, H, W)`` stack.

    Raises ``ValueError`` if `binary_stack` is neither 2D nor 3D.
    """
    if binary_stack.ndim == 2:
        return label(binary_stack)

    if binary_stack.ndim == 3:
        out = np.zeros_like(binary_stack, dtype=np.int32)
        for t in range(binary_stack.shape[0]):
            out[t] = label(binary_stack[t])
        return out

    raise ValueError("Input stack must be 2D or 3D.")


def relabel_separated(
    binary_stack: np.ndarray, seed_stacks: list[np.ndarray]
) -> np.ndarray:
    """Label a boolean result so that touching source objects stay separate.

    Assumes `binary_stack` and every array in `seed_stacks` are 3D
    ``(T, H, W)`` stacks sharing the same shape.
    """
    out = np.zeros_like(binary_stack, dtype=np.int32)

    for t in range(binary_stack.shape[0]):
        seeds = np.zeros(binary_stack.shape[1:], dtype=np.int32)
        offset = 0
        for stack in seed_stacks:
            frame = stack[t]
            occupied = frame > 0
            seeds[occupied] = frame[occupied].astype(np.int32) + offset
            offset += int(frame.max())

        if seeds.max() == 0:
            out[t] = label(binary_stack[t])
            continue

        grown = expand_labels(seeds, distance=_UNBOUNDED_DISTANCE)
        out[t] = grown * binary_stack[t]
    return out


def compute_bitwise_mask(
    a: np.ndarray,
    b: np.ndarray | None,
    op: str,
    not_a: bool,
    not_b: bool,
    dil1v: int,
    dil2v: int,
    keep_separate: bool = False,
) -> np.ndarray:
    """Combine two label stacks with a bitwise operator."""
    op = str(op).upper()

    # Fast path: a pure dilation of a non-inverted mask keeps its own labels.
    if keep_separate and op == "NONE" and dil1v > 0 and not not_a:
        return np.stack([expand_labels(sl, distance=dil1v) for sl in a])

    mask_a = a > 0
    if not_a:
        mask_a = ~mask_a
    mask_a = apply_mask_morphology(mask_a, dil1v)

    if op == "NONE":
        return connected_components_stack(mask_a)

    if b is None:
        raise ValueError("Second mask is required unless op is 'NONE'.")

    mask_b = b > 0
    if not_b:
        mask_b = ~mask_b
    mask_b = apply_mask_morphology(mask_b, dil2v)

    if op == "INT":
        result = mask_a & mask_b
    elif op == "UNI":
        result = mask_a | mask_b
    elif op == "ME":
        result = mask_a ^ mask_b
    else:
        raise ValueError(f"Unsupported op: {op}")

    if keep_separate:
        seed_stacks = [
            stack
            for stack, inverted in ((a, not_a), (b, not_b))
            if stack is not None and not inverted
        ]
        return relabel_separated(result, seed_stacks)

    return connected_components_stack(result)
