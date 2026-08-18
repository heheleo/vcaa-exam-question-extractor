"""Image manipulation: crop, trim blank space, merge cross-page questions."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.models import Bbox


def crop_question(page_image: Path, bbox: Bbox) -> Image.Image:
    """Crop a question region from a page image.

    The bbox is clamped to the image bounds before cropping.
    """
    img = Image.open(page_image)
    x2 = min(bbox.x + bbox.w, img.width)
    y2 = min(bbox.y + bbox.h, img.height)
    x1 = max(bbox.x, 0)
    y1 = max(bbox.y, 0)
    return img.crop((x1, y1, x2, y2))


def trim_blank_bottom(image: Image.Image) -> Image.Image:
    """Remove trailing blank (white) space from the bottom of an image.

    Scans upward from the bottom until it finds a non-white row of pixels.
    Working lines and drawn content are preserved -- only pure white rows
    (pixel value >= 250) are trimmed. Adds 10px padding below the last
    content row.
    """
    gray = image.convert("L")
    width, height = gray.size
    pixels = gray.load()

    last_content = height - 1
    for y in range(height - 1, -1, -1):
        for x in range(width):
            if pixels[x, y] < 250:
                last_content = y
                break
        else:
            continue
        break

    # If entirely blank, return as-is
    if last_content == height - 1:
        all_white = all(pixels[0, y] >= 250 for y in range(height))
        if all_white:
            return image

    crop_to = min(last_content + 10, height)
    return image.crop((0, 0, width, crop_to))


def merge_vertical(images: list[Image.Image]) -> Image.Image:
    """Vertically concatenate images with no gap between them.

    Images are center-aligned horizontally if widths differ.
    Returns a 1x1 blank image if the list is empty.
    """
    if not images:
        return Image.new("RGB", (1, 1), "white")

    if len(images) == 1:
        return images[0]

    max_width = max(im.width for im in images)
    total_height = sum(im.height for im in images)

    merged = Image.new("RGB", (max_width, total_height), "white")
    y_offset = 0
    for im in images:
        x_offset = (max_width - im.width) // 2
        merged.paste(im, (x_offset, y_offset))
        y_offset += im.height

    return merged
