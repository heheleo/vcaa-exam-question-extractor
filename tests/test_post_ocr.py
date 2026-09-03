from pathlib import Path

from src.models import Bbox, QuestionBbox
from src.post_ocr import (
    extract_number,
    group_questions,
    is_part_label,
    question_sort_key,
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
