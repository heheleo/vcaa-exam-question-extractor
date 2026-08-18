"""PDF rendering, detection, and cropping pipeline."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from src.cropper import crop_question, merge_vertical, trim_blank_bottom
from src.detector import Detector
from src.models import (
    Bbox,
    ExamResult,
    PaperMeta,
    QuestionBbox,
    QuestionResult,
    parse_filename,
)
from src.pdf_utils import render_pages

logger = logging.getLogger(__name__)


def scan_input_dir(input_dir: Path) -> list[PaperMeta]:
    """Recursively find all valid exam PDFs in a directory."""
    papers: list[PaperMeta] = []
    for pdf_path in sorted(input_dir.rglob("*.pdf")):
        meta = parse_filename(pdf_path)
        if meta is not None:
            papers.append(meta)
            logger.info("Found: %s -> %s", pdf_path.name, meta.key)
        else:
            logger.debug("Skipping (no match): %s", pdf_path.name)
    return papers


def question_sort_key(item: tuple[int, QuestionBbox]) -> tuple[int, int, int]:
    """Sort blocks by page, then vertical, then horizontal position
    (left-to-right for two-column MCQ pages)."""
    page_num, q = item
    return (page_num, q.bbox.y, q.bbox.x)


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


def process_paper(
    paper: PaperMeta,
    output_dir: Path,
    detector: Detector,
    temp_dir: Path,
    dpi: int = 300,
) -> ExamResult:
    """Run the full extraction pipeline for a single paper.

    Stage 1: Render pages to PNGs.
    Stage 2: Detect question blocks via Qwen V3.
    Stage 3: Chain blocks into questions, crop, trim, merge, save.
    """
    exam_output_dir = output_dir / paper.key
    exam_output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Render
    logger.info("[%s] Rendering pages...", paper.key)
    page_images = render_pages(paper.path, temp_dir, dpi=dpi)
    logger.info("[%s] Rendered %d pages", paper.key, len(page_images))

    # Stage 2: Detect
    raw_questions: list[tuple[int, QuestionBbox]] = []
    for i, page_path in enumerate(page_images, start=1):
        for q in detector.detect(page_path, paper.exam_type, page_number=i):
            raw_questions.append((i, q))

    if not raw_questions:
        logger.warning("[%s] No questions detected!", paper.key)
        return ExamResult(
            key=paper.key,
            year=paper.year,
            source=paper.source,
            exam_type=paper.exam_type,
            questions=[],
            path=str(paper.path),
        )

    # Stage 3: Chain, crop, trim, merge
    question_groups = group_questions(raw_questions)
    logger.info(
        "[%s] Grouped %d detections into %d logical questions",
        paper.key,
        len(raw_questions),
        len(question_groups),
    )

    results: list[QuestionResult] = []
    for group_idx, group in enumerate(question_groups, start=1):
        parts = []
        pages_involved = []
        for page_num, q_bbox in group:
            page_path = page_images[page_num - 1]
            cropped = crop_question(page_path, q_bbox.bbox)
            trimmed = trim_blank_bottom(cropped)
            parts.append(trimmed)
            pages_involved.append(page_num)

        final_image = merge_vertical(parts)
        first_label = group[0][1].label
        question_number = extract_number(first_label) or first_label

        # The header's marks are the question total; part marks ("1 mark"
        # inside a part's answer space) only matter when the header has none.
        header_marks = group[0][1].marks
        if header_marks is not None:
            total_marks = header_marks
        else:
            marks_values = [q.marks for _, q in group if q.marks is not None]
            total_marks = sum(marks_values) if marks_values else None

        image_filename = f"q{group_idx:02d}.png"
        image_path = exam_output_dir / image_filename
        final_image.save(str(image_path))

        has_subquestions = paper.exam_type == "exam1" or (
            paper.exam_type == "exam2" and group_idx > 20
        )

        results.append(
            QuestionResult(
                number=question_number,
                image=image_filename,
                marks=total_marks,
                pages=pages_involved,
                has_subquestions=has_subquestions,
                cross_page=len(group) > 1,
            )
        )

    # Write mapping.json
    exam_result = ExamResult(
        key=paper.key,
        year=paper.year,
        source=paper.source,
        exam_type=paper.exam_type,
        questions=results,
        path=str(paper.path),
    )
    mapping_path = exam_output_dir / "mapping.json"
    mapping_path.write_text(
        json.dumps(exam_result.to_mapping(), indent=2, ensure_ascii=False)
    )
    logger.info(
        "[%s] Wrote %d questions to %s", paper.key, len(results), exam_output_dir
    )

    return exam_result
