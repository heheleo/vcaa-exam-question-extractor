from pathlib import Path

from src.models import Bbox, QuestionBbox
from src.pipeline import (
    match_continued_questions,
    question_sort_key,
    scan_input_dir,
)


def make_q(num: str, page: int, y: int, continued=False, continued_from=False):
    return (
        page,
        QuestionBbox(
            question_number=num,
            bbox=Bbox(x=50, y=y, w=700, h=100),
            continued=continued,
            continued_from=continued_from,
        ),
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


def test_match_no_continuations():
    qs = [make_q("1", 1, 100), make_q("2", 1, 300), make_q("3", 1, 500)]
    groups = match_continued_questions(qs)
    assert len(groups) == 3
    assert all(len(g) == 1 for g in groups)


def test_match_continued_pair():
    qs = [
        make_q("1", 1, 100, continued=True),
        make_q("1", 2, 10, continued_from=True),
    ]
    groups = match_continued_questions(qs)
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_match_orphaned_continued_from():
    groups = match_continued_questions([make_q("1", 1, 10, continued_from=True)])
    assert len(groups) == 1


def test_match_orphaned_continued():
    groups = match_continued_questions([make_q("1", 1, 100, continued=True)])
    assert len(groups) == 1


def test_match_mismatched_continued_numbers():
    """continued_from with different question number starts a new group."""
    qs = [
        make_q("1", 1, 100, continued=True),
        make_q("2", 2, 10, continued_from=True),  # Different number!
    ]
    groups = match_continued_questions(qs)
    assert len(groups) == 2  # Should be 2 separate questions, not merged


def test_question_sort_key():
    q1 = make_q("1", 1, 500)
    q2 = make_q("2", 1, 100)
    sorted_qs = sorted([q1, q2], key=question_sort_key)
    assert sorted_qs[0][1].question_number == "2"
    assert sorted_qs[1][1].question_number == "1"
