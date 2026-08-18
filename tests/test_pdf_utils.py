from pathlib import Path

import fitz
import pytest
from PIL import Image

from src.pdf_utils import render_pages, get_page_count


@pytest.fixture
def single_page_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 50), "Question 1", fontsize=14)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def multi_page_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "multi.pdf"
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 50), f"Question {i+1}", fontsize=14)
    doc.save(str(path))
    doc.close()
    return path


def test_get_page_count_single(single_page_pdf):
    assert get_page_count(single_page_pdf) == 1


def test_get_page_count_multi(multi_page_pdf):
    assert get_page_count(multi_page_pdf) == 3


def test_render_pages_creates_images(single_page_pdf, tmp_path):
    out_dir = tmp_path / "rendered"
    images = render_pages(single_page_pdf, out_dir, dpi=150)
    assert len(images) == 1
    assert images[0].exists()
    assert images[0].suffix == ".png"
    img = Image.open(images[0])
    assert img.width > 0 and img.height > 0


def test_render_pages_multi(multi_page_pdf, tmp_path):
    out_dir = tmp_path / "rendered"
    images = render_pages(multi_page_pdf, out_dir, dpi=150)
    assert len(images) == 3
    for i, p in enumerate(images):
        assert p.name == f"page_{i+1:03d}.png"


def test_render_pages_creates_output_dir(single_page_pdf, tmp_path):
    out_dir = tmp_path / "auto_created"
    images = render_pages(single_page_pdf, out_dir, dpi=150)
    assert out_dir.exists()
    assert len(images) == 1
