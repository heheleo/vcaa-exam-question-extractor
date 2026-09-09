# tests/test_integration.py
import json
from pathlib import Path
from unittest.mock import MagicMock

import fitz
import pytest
from PIL import Image

from src.models import Bbox, PaperMeta, QuestionBbox
from src.pipeline import process_paper


def create_test_pdf(path: Path, pages: int = 2):
    doc = fitz.open()
    for p in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 40), f"Question {p + 1}", fontsize=14)
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
        [
            QuestionBbox(
                label="1", bbox=Bbox(40, 30, 500, 150), marks=5, text="Find x."
            )
        ],
        [QuestionBbox(label="2", bbox=Bbox(40, 30, 500, 150), marks=5)],
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
    assert mapping["questions"][0]["text"] == "Find x."
    assert mapping["questions"][1]["text"] == ""

    for q in result.questions:
        img = Image.open(exam_dir / q.image)
        assert img.width > 0 and img.height > 0


def test_exam2_mcq_and_short_answer_naming(tmp_path):
    # Section A (MCQ, page 1) then Section B numbering restarts (page 2).
    path = tmp_path / "2024-vcaa-exam-2.pdf"
    create_test_pdf(path, pages=2)

    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    output_dir.mkdir()

    paper = PaperMeta(path=path, year=2024, source="vcaa", exam_type="exam2")

    mock = MagicMock()
    mock.detect.side_effect = [
        [
            QuestionBbox(label="1", bbox=Bbox(40, 30, 500, 150), marks=1),
            QuestionBbox(label="2", bbox=Bbox(40, 250, 500, 150), marks=1),
        ],
        [
            QuestionBbox(label="1", bbox=Bbox(40, 30, 500, 150), marks=5),
            QuestionBbox(label="a", bbox=Bbox(40, 250, 500, 100), marks=1),
        ],
    ]

    result = process_paper(paper, output_dir, mock, temp_dir, dpi=72)

    assert [q.image for q in result.questions] == ["mc01.png", "mc02.png", "q01.png"]
    assert [q.number for q in result.questions] == ["1", "2", "1"]
    assert [q.kind for q in result.questions] == ["mcq", "mcq", "short"]
    assert [q.has_subquestions for q in result.questions] == [False, False, True]

    exam_dir = output_dir / "2024-vcaa-exam2"
    for f in ("mc01.png", "mc02.png", "q01.png", "mapping.json"):
        assert (exam_dir / f).is_file()

    with open(exam_dir / "mapping.json") as f:
        mapping = json.load(f)
    assert [q["type"] for q in mapping["questions"]] == ["mcq", "mcq", "short"]


def test_exam1_questions_are_short_kind(tmp_path):
    path = tmp_path / "2024-vcaa-exam-1.pdf"
    create_test_pdf(path, pages=1)

    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    output_dir.mkdir()

    paper = PaperMeta(path=path, year=2024, source="vcaa", exam_type="exam1")

    mock = MagicMock()
    mock.detect.return_value = [
        QuestionBbox(label="1", bbox=Bbox(40, 30, 500, 150), marks=5)
    ]

    result = process_paper(paper, output_dir, mock, temp_dir, dpi=72)

    assert [q.image for q in result.questions] == ["q01.png"]
    assert [q.kind for q in result.questions] == ["short"]
    assert result.questions[0].has_subquestions is True


def test_raster_only_pdf_still_processed(tmp_path):
    # Detection is entirely AI-driven, so the page content doesn't matter.
    path = tmp_path / "2024-vcaa-exam-1.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.draw_rect(fitz.Rect(50, 50, 500, 700), color=(0, 0, 0))
    doc.save(str(path))
    doc.close()

    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    output_dir.mkdir()

    paper = PaperMeta(path=path, year=2024, source="vcaa", exam_type="exam1")

    mock = MagicMock()
    mock.detect.return_value = [
        QuestionBbox(label="1", bbox=Bbox(40, 30, 500, 150), marks=5)
    ]

    result = process_paper(paper, output_dir, mock, temp_dir, dpi=72)

    assert len(result.questions) == 1
    assert result.questions[0].number == "1"
    mock.detect.assert_called_once()


def test_marks_use_header_total(tmp_path):
    # The header's marks are the question total; a part's marks don't add.
    path = tmp_path / "2024-vcaa-exam-1.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 40), "Question 1", fontsize=14)
    doc.save(str(path))
    doc.close()

    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    output_dir.mkdir()

    paper = PaperMeta(path=path, year=2024, source="vcaa", exam_type="exam1")

    mock = MagicMock()
    mock.detect.return_value = [
        QuestionBbox(label="1", bbox=Bbox(40, 30, 500, 150), marks=5),
        QuestionBbox(label="a", bbox=Bbox(40, 200, 500, 100), marks=1),
    ]

    result = process_paper(paper, output_dir, mock, temp_dir, dpi=72)

    assert len(result.questions) == 1
    assert result.questions[0].marks == 5


def test_cross_page_merge(tmp_path):
    # Page 1: Question 1. Page 2: its continuation part.
    path = tmp_path / "2024-vcaa-exam-1.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 40), "Question 1", fontsize=14)
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 40), "(b) Part b", fontsize=11)
    doc.save(str(path))
    doc.close()

    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    output_dir.mkdir()

    paper = PaperMeta(path=path, year=2024, source="vcaa", exam_type="exam1")

    mock = MagicMock()
    mock.detect.side_effect = [
        [QuestionBbox(label="1", bbox=Bbox(40, 30, 500, 700))],
        [QuestionBbox(label="b", bbox=Bbox(40, 30, 500, 400))],
    ]

    result = process_paper(paper, output_dir, mock, temp_dir, dpi=72)

    assert len(result.questions) == 1
    assert result.questions[0].number == "1"
    assert result.questions[0].cross_page is True
    assert result.questions[0].pages == [1, 2]
