from pathlib import Path

from src.models import Bbox, QuestionBbox
from src.post_ocr import (
    extract_number,
    group_questions,
    is_part_label,
    question_sort_key,
    split_mcq_short_answer,
)


def make_q(label: str, page: int, y: int, x: int = 50, marks: int | None = None):
    return (
        page,
        QuestionBbox(label=label, bbox=Bbox(x=x, y=y, w=700, h=100), marks=marks),
    )


def test_is_part_label():
    assert is_part_label("a.")
    assert is_part_label("ii.")
    assert is_part_label("a.ii")
    assert is_part_label("A.")
    assert not is_part_label("TURN OVER")
    assert not is_part_label("3")


def test_question_sort_key_orders_by_page_y_x():
    q1 = make_q("1", 1, 100, x=600)
    q2 = make_q("2", 1, 100, x=50)
    sorted_qs = sorted([q1, q2], key=question_sort_key)
    assert sorted_qs[0][1].label == "2"


def _numbered_groups(labels: list[str]):
    """Build one single-block group per label on successive pages."""
    qs = [make_q(label, page, 100) for page, label in enumerate(labels, start=1)]
    return group_questions(qs)


def test_split_detects_numbering_reset():
    groups = _numbered_groups(["1", "2", "3", "1", "2"])
    mcq, short = split_mcq_short_answer(groups)
    assert [g[0][1].label for g in mcq] == ["1", "2", "3"]
    assert [g[0][1].label for g in short] == ["1", "2"]


def test_split_full_exam2_sections():
    groups = _numbered_groups([str(i) for i in range(1, 21)] + ["1", "2"])
    mcq, short = split_mcq_short_answer(groups)
    assert len(mcq) == 20
    assert len(short) == 2


def test_split_falls_back_to_20_without_reset():
    groups = _numbered_groups([str(i) for i in range(1, 23)])
    mcq, short = split_mcq_short_answer(groups)
    assert len(mcq) == 20
    assert len(short) == 2


def test_split_all_mcq_when_short_section_missing():
    groups = _numbered_groups(["1", "2", "3"])
    mcq, short = split_mcq_short_answer(groups)
    assert len(mcq) == 3
    assert short == []


def test_split_empty():
    assert split_mcq_short_answer([]) == ([], [])
