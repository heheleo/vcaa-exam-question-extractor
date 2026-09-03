from __future__ import annotations

import logging
import re

from src.models import Bbox, QuestionBbox

logger = logging.getLogger(__name__)
_NUMBER_RE = re.compile(r"(?:question\s+)?(\d+)", re.IGNORECASE)
_PART_RE = re.compile(r"(?:[a-z](?:\.\s*[ivx]+)?|[ivx]+)\.?$", re.IGNORECASE)


def extract_number(label: str) -> str | None:
    """Extract the leading top-level question number from a label.

    "3", "3.", "Question 3", "Question 3 (continued)" -> "3"
    "a.", "ii.", "TURN OVER" -> None
    """
    m = _NUMBER_RE.match(label.strip())
    return m.group(1) if m else None


def is_part_label(label: str) -> bool:
    """True if the label is a sub-part label: "a.", "ii.", "a.ii", "A."..."""
    return bool(_PART_RE.fullmatch(label.strip()))


def _overlap_ratio(a: Bbox, b: Bbox) -> float:
    """Overlap area / area of the smaller box. 1.0 = one contains the other."""
    ix = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = ix * iy
    if inter == 0:
        return 0.0
    return inter / min(a.w * a.h, b.w * b.h)


def question_sort_key(item: tuple[int, QuestionBbox]) -> tuple[int, int, int]:
    """Sort blocks by page, then vertical, then horizontal position
    (left-to-right for two-column MCQ pages)."""
    page_num, q = item
    return (page_num, q.bbox.y, q.bbox.x)


def group_questions(
    raw_questions: list[tuple[int, QuestionBbox]],
) -> list[list[tuple[int, QuestionBbox]]]:
    """Chain detected blocks into logical questions by label sequence.

    A numeric label ("1", "Question 3") starts a new question. Part labels
    ("a.", "ii.") attach to the current question — including across pages,
    which is how continuations chain. Unrecognized labels (e.g. "TURN
    OVER") and boxes largely contained in the previous box on the same
    page (duplicate detections) are dropped.
    """
    if not raw_questions:
        return []

    sorted_qs = sorted(raw_questions, key=question_sort_key)
    groups: list[list[tuple[int, QuestionBbox]]] = []
    current: list[tuple[int, QuestionBbox]] | None = None
    current_num: str | None = None
    prev: tuple[int, QuestionBbox] | None = None

    for page, q in sorted_qs:
        # Duplicate or contained box? (e.g. the model boxed the question
        # AND a sub-part inside it)
        if (
            prev is not None
            and prev[0] == page
            and _overlap_ratio(prev[1].bbox, q.bbox) > 0.5
        ):
            logger.info("Dropping duplicate box labeled %r on page %d", q.label, page)
            continue

        num = extract_number(q.label)
        if num is None:
            if not is_part_label(q.label) or current is None:
                logger.warning(
                    "Dropping unrecognized label %r on page %d", q.label, page
                )
                continue
        elif current is None or num != current_num:
            current = []
            groups.append(current)
            current_num = num

        current.append((page, q))
        prev = (page, q)

    return groups
