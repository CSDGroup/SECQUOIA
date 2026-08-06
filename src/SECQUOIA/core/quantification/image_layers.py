"""Collecting label stacks and image layers to be measured."""

import logging
from dataclasses import dataclass

import numpy as np

LOG = logging.getLogger(__name__)


def _collect_label_stacks(source) -> list[tuple[int, np.ndarray]]:
    """Normalise `main_window.labels` into ``[(mask_idx >= 1, stack[T, H, W]), ...]``."""
    if source is None:
        return []
    if isinstance(source, np.ndarray):
        return [(1, source)]
    if isinstance(source, (list | tuple)):
        stacks = source
    elif isinstance(source, dict):
        stacks = list(source.values())
    else:
        return []
    return [
        (idx, stack)
        for idx, stack in enumerate(stacks, start=1)
        if stack is not None
    ]


def _ensure_int_labels(stack: np.ndarray) -> np.ndarray:
    """Return `stack` with an integer dtype, casting only when needed."""
    if np.issubdtype(stack.dtype, np.integer):
        return stack
    return stack.astype(np.int32, copy=False)


def _presence_flags(main_window, channel):
    """Per frame acquisition flags for `channel`, or None when unknown."""
    image_present = getattr(main_window, "image_present", None)
    if isinstance(image_present, dict):
        return image_present.get(channel)
    return None


def _is_frame_present(present_flags, t: int, n_frames: int) -> bool:
    """Whether `channel` was acquired at frame `t`.

    Returns True when the flags are unknown or do not cover `n_frames`,
    so a mismatched flag list disables presence filtering rather than
    dropping every frame.
    """
    if present_flags is None or len(present_flags) != n_frames:
        return True
    return bool(present_flags[t])


@dataclass(frozen=True)
class ImageLayer:
    """One (variant, channel) image stack to be measured against the labels."""

    variant: str
    channel: str
    stack: object


def _image_layers(main_window) -> list[ImageLayer]:
    """Flatten every (variant, channel) combination into a list of layers."""
    variants = [("NoBgCorrected", getattr(main_window, "images", None))]
    if main_window.basic.flag:
        variants += [
            (
                "BaSiCBgCorrectedRatioFlat",
                getattr(main_window, "corrected_images_ratioflat", None),
            ),
            (
                "BaSiCBgCorrectedNoRatioFlat",
                getattr(main_window, "corrected_images_noratioflat", None),
            ),
        ]

    layers: list[ImageLayer] = []
    for variant, image_stack in variants:
        if not image_stack:
            continue
        for channel in image_stack:
            stack = image_stack[channel]
            if stack is None:
                LOG.warning(
                    "images[%s] not found. Skipping this channel.", channel
                )
                continue
            layers.append(
                ImageLayer(variant=variant, channel=channel, stack=stack)
            )
    return layers


def _frames_per_mask(label_entries, layers: list[ImageLayer]) -> list[int]:
    """Return the number of frames measured for each mask."""
    return [
        max(
            (min(len(layer.stack), len(label_stack)) for layer in layers),
            default=0,
        )
        for _mask_idx, label_stack in label_entries
    ]


def _count_frame_steps(main_window, label_entries) -> int:
    """Total frames measured across all masks, for the progress bar."""
    try:
        return sum(_frames_per_mask(label_entries, _image_layers(main_window)))
    except (
        RuntimeError,
        AttributeError,
        TypeError,
        ValueError,
    ):
        return 0
