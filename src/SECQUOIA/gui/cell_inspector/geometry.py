"""Works out which pixels to cut out of a frame."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rect:
    """A pixel rectangle. Left and top are included, right and bottom are not."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        """Width in pixels."""
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        """Height in pixels."""
        return max(0, self.bottom - self.top)

    def slices(self) -> tuple[slice, slice]:
        """The (rows, cols) slices to index an (H, W) array with."""
        return slice(self.top, self.bottom), slice(self.left, self.right)


@dataclass(frozen=True)
class CropWindow:
    """A crop request and the part of it that lies inside the image."""

    requested: Rect
    valid: Rect
    inset: tuple[int, int]

    @property
    def needs_padding(self) -> bool:
        """True if part of the crop falls outside the image."""
        return (
            self.valid.width != self.requested.width
            or self.valid.height != self.requested.height
        )


def crop_window(
    center_x: float,
    center_y: float,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
) -> CropWindow:
    """Build a width x height crop centred on (center_x, center_y)."""
    width = max(1, int(width))
    height = max(1, int(height))

    left = int(round(float(center_x))) - width // 2
    top = int(round(float(center_y))) - height // 2
    requested = Rect(left, top, left + width, top + height)

    image_width = max(0, int(image_width))
    image_height = max(0, int(image_height))

    def _clamp(value: int, limit: int) -> int:
        """Keep value between 0 and limit."""
        return min(max(value, 0), limit)

    valid = Rect(
        left=_clamp(requested.left, image_width),
        top=_clamp(requested.top, image_height),
        right=_clamp(requested.right, image_width),
        bottom=_clamp(requested.bottom, image_height),
    )
    return CropWindow(
        requested=requested,
        valid=valid,
        inset=(valid.top - requested.top, valid.left - requested.left),
    )
