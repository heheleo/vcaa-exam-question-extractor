from pathlib import Path

from src.models import Bbox, QuestionBbox
from src.pipeline import (
    extract_number,
    group_questions,
    scan_input_dir,
)


def make_q(label: str, page: int, y: int, x: int = 50, marks: int | None = None):
    return (
        page,
        QuestionBbox(label=label, bbox=Bbox(x=x, y=y, w=700, h=100), marks=marks),
    )


def test_scan_finds_valid_files(tmp_path: Path):
    (tmp_path / "2024-vcaa-exam-1.pdf").touch()
    (tmp_path / "2024-vcaa-exam-2.pdf").touch()
    (tmp_path / "2025-heffernan-exam-1.pdf").touch()
    (tmp_path / "random-file.pdf").touch()
    (tmp_path / "notes.txt").touch()

    papers = scan_input_dir(tmp_path)
    assert len(papers) == 3
    keys = {p.key for p in papers}
    assert "2024-vcaa-exam1" in keys
    assert "2024-vcaa-exam2" in keys
    assert "2025-heffernan-exam1" in keys


def test_scan_empty_dir(tmp_path: Path):
    assert scan_input_dir(tmp_path) == []


def test_scan_recursive(tmp_path: Path):
    (tmp_path / "subdir").mkdir()
    (tmp_path / "2024-vcaa-exam-1.pdf").touch()
    (tmp_path / "subdir" / "2023-vcaa-exam-2.pdf").touch()
    assert len(scan_input_dir(tmp_path)) == 2


def test_group_single_questions():
    qs = [make_q("1", 1, 100), make_q("2", 1, 300), make_q("3", 1, 500)]
    groups = group_questions(qs)
    assert len(groups) == 3
    assert all(len(g) == 1 for g in groups)


def test_group_parts_attach():
    qs = [make_q("2", 1, 100), make_q("a.", 1, 300), make_q("b.", 1, 500)]
    groups = group_questions(qs)
    assert len(groups) == 1
    assert len(groups[0]) == 3


def test_group_continuation_across_pages():
    """Part labels on page 2 attach to the question started on page 1."""
    qs = [
        make_q("2", 1, 100),
        make_q("a.", 1, 300),
        make_q("ii.", 2, 100),
        make_q("c.", 2, 300),
    ]
    groups = group_questions(qs)
    assert len(groups) == 1
    assert [page for page, _ in groups[0]] == [1, 1, 2, 2]


def test_group_continued_header_same_number():
    """A reprinted 'Question 3 (continued)' header attaches to question 3."""
    qs = [make_q("Question 3", 1, 100), make_q("Question 3 (continued)", 2, 100)]
    groups = group_questions(qs)
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_group_number_reset_new_question():
    """MCQ '1' then ER '1' (exam2): same number, different question."""
    qs = [make_q("1", 2, 100), make_q("2", 2, 300), make_q("1", 15, 100)]
    groups = group_questions(qs)
    assert len(groups) == 3
    assert groups[2][0][1].label == "1"


def test_group_unrecognized_label_dropped():
    qs = [make_q("1", 1, 100), make_q("TURN OVER", 1, 300), make_q("2", 2, 100)]
    groups = group_questions(qs)
    assert len(groups) == 2


def test_group_orphan_part_dropped():
    """A part label with no preceding question has nothing to attach to."""
    assert group_questions([make_q("a.", 1, 100)]) == []


def test_group_duplicate_overlap_dropped():
    qs = [make_q("1", 1, 100), make_q("1", 1, 100)]  # same box detected twice
    groups = group_questions(qs)
    assert len(groups) == 1
    assert len(groups[0]) == 1


def test_group_contained_part_box_dropped():
    """Model boxed the whole question AND a sub-part inside it."""
    whole = QuestionBbox(label="1", bbox=Bbox(x=50, y=100, w=700, h=500))
    part = QuestionBbox(label="a.", bbox=Bbox(x=50, y=200, w=700, h=50))
    groups = group_questions([(1, whole), (1, part)])
    assert len(groups) == 1
    assert len(groups[0]) == 1


def test_extract_number():
    assert extract_number("3") == "3"
    assert extract_number("3.") == "3"
    assert extract_number("Question 3") == "3"
    assert extract_number("Question 3 (continued)") == "3"
    assert extract_number("a.") is None
    assert extract_number("TURN OVER") is None
