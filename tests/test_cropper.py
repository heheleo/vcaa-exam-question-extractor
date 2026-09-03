from pathlib import Path

import pytest
from PIL import Image

from src.cropper import crop_question, merge_vertical, trim_blank_bottom
from src.models import Bbox


@pytest.fixture
def sample_page_image(tmp_path: Path) -> Path:
    path = tmp_path / "test_page.png"
    img = Image.new("RGB", (400, 600), color="white")
    for x in range(50, 250):
        for y in range(100, 250):
            img.putpixel((x, y), (50, 50, 50))
    img.save(path)
    return path


def test_crop_simple(sample_page_image):
    bbox = Bbox(x=50, y=100, w=200, h=150)
    cropped = crop_question(sample_page_image, bbox)
    assert cropped.size == (200, 150)


def test_crop_respects_bounds(sample_page_image):
    bbox = Bbox(x=-10, y=-10, w=9999, h=9999)
    cropped = crop_question(sample_page_image, bbox)
    img = Image.open(sample_page_image)
    assert cropped.width <= img.width
    assert cropped.height <= img.height


def test_trim_removes_white_bottom():
    img = Image.new("RGB", (200, 300), color="white")
    for x in range(200):
        for y in range(100):
            img.putpixel((x, y), (0, 0, 0))
    trimmed = trim_blank_bottom(img)
    assert trimmed.height < 300
    assert 100 <= trimmed.height <= 115


def test_trim_all_white_unchanged():
    img = Image.new("RGB", (100, 100), color="white")
    trimmed = trim_blank_bottom(img)
    assert trimmed.size == (100, 100)


def test_trim_preserves_ruled_lines():
    img = Image.new("RGB", (200, 200), color="white")
    for x in range(50, 150):
        img.putpixel((x, 190), (0, 0, 0))  # Dark line near bottom
    trimmed = trim_blank_bottom(img)
    assert trimmed.height >= 190


def test_merge_two_images():
    img1 = Image.new("RGB", (100, 50), color=(255, 0, 0))
    img2 = Image.new("RGB", (100, 50), color=(0, 0, 255))
    merged = merge_vertical([img1, img2])
    assert merged.size == (100, 100)


def test_merge_different_widths():
    img1 = Image.new("RGB", (80, 50), color=(255, 0, 0))
    img2 = Image.new("RGB", (100, 50), color=(0, 0, 255))
    merged = merge_vertical([img1, img2])
    assert merged.width == 100
    assert merged.height == 100
    # Left-aligned: narrower image hugs x=0, right side filled with white
    assert merged.getpixel((0, 0)) == (255, 0, 0)
    assert merged.getpixel((99, 0)) == (255, 255, 255)
    assert merged.getpixel((0, 75)) == (0, 0, 255)


def test_merge_single_image():
    img = Image.new("RGB", (100, 50), color=(255, 0, 0))
    merged = merge_vertical([img])
    assert merged.size == (100, 50)


def test_merge_empty_returns_blank():
    merged = merge_vertical([])
    assert merged.size == (1, 1)
