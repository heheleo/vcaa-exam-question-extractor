# tests/test_integration.py
import json
from pathlib import Path
from unittest.mock import MagicMock

import fitz
import pytest
from PIL import Image

from src.models import Bbox, QuestionBbox, PaperMeta
from src.pipeline import process_paper


def create_test_pdf(path: Path, pages: int = 2):
    doc = fitz.open()
    for p in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 40), f"Question {p+1}", fontsize=14)
        page.insert_text((50, 70), "(a) Part a", fontsize=11)
        page.insert_text((50, 90), "(b) Part b", fontsize=11)
        page.insert_text((50, 110), "[5 marks]", fontsize=10)
    doc.save(str(path))
    doc.close()


@pytest.fixture
def test_pdf(tmp_path):
    path = tmp_path / "2024-vcaa-exam-1.pdf"
    create_test_pdf(path, pages=2)
    return path


def test_full_pipeline_with_mock(test_pdf, tmp_path):
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    output_dir.mkdir()

    paper = PaperMeta(path=test_pdf, year=2024, source="vcaa", exam_type="exam1")

    mock = MagicMock()
    mock.detect.side_effect = [
        [QuestionBbox(question_number="1", bbox=Bbox(40, 30, 500, 150), marks=5)],
        [QuestionBbox(question_number="2", bbox=Bbox(40, 30, 500, 150), marks=5)],
    ]

    result = process_paper(paper, output_dir, mock, temp_dir, dpi=72)

    assert result.key == "2024-vcaa-exam1"
    assert len(result.questions) == 2
    assert result.questions[0].number == "1"
    assert result.questions[1].number == "2"

    exam_dir = output_dir / "2024-vcaa-exam1"
    assert (exam_dir / "q01.png").is_file()
    assert (exam_dir / "q02.png").is_file()
    assert (exam_dir / "mapping.json").is_file()

    with open(exam_dir / "mapping.json") as f:
        mapping = json.load(f)
    assert mapping["year"] == 2024
    assert mapping["total_questions"] == 2

    for q in result.questions:
        img = Image.open(exam_dir / q.image)
        assert img.width > 0 and img.height > 0


def test_cross_page_merge(test_pdf, tmp_path):
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    output_dir.mkdir()

    paper = PaperMeta(path=test_pdf, year=2024, source="vcaa", exam_type="exam1")

    mock = MagicMock()
    mock.detect.side_effect = [
        [QuestionBbox(question_number="1", bbox=Bbox(40, 30, 500, 700), continued=True)],
        [QuestionBbox(question_number="1", bbox=Bbox(40, 30, 500, 400), marks=8,
                      continued_from=True)],
    ]

    result = process_paper(paper, output_dir, mock, temp_dir, dpi=72)

    assert len(result.questions) == 1
    assert result.questions[0].cross_page is True
    assert result.questions[0].pages == [1, 2]
