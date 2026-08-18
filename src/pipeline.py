"""Pipeline orchestrator: ties together PDF rendering, detection, and cropping."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.cropper import crop_question, merge_vertical, trim_blank_bottom
from src.detector import Detector
from src.models import (
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


def question_sort_key(item: tuple[int, QuestionBbox]) -> tuple[int, int]:
    """Sort questions by page number, then vertical position."""
    page_num, q = item
    return (page_num, q.bbox.y)


def match_continued_questions(
    raw_questions: list[tuple[int, QuestionBbox]],
) -> list[list[tuple[int, QuestionBbox]]]:
    """Group questions that span multiple pages into logical units.

    Uses the continued/continued_from flags. Returns a list of groups;
    each group is one logical question that may span 1+ pages.
    """
    if not raw_questions:
        return []

    sorted_qs = sorted(raw_questions, key=question_sort_key)
    groups: list[list[tuple[int, QuestionBbox]]] = []
    current_group: list[tuple[int, QuestionBbox]] = []

    for i, (page, q) in enumerate(sorted_qs):
        if q.continued_from and current_group:
            # Only merge if question numbers match
            if current_group[-1][1].question_number == q.question_number:
                current_group.append((page, q))
            else:
                # Different question number on new page — close old group, start fresh
                groups.append(current_group)
                current_group = [(page, q)]
        elif q.continued_from and not current_group:
            current_group = [(page, q)]
        elif current_group:
            _prev_page, prev_q = current_group[-1]
            if prev_q.continued:
                # Previous expected continuation; merge if same number
                if prev_q.question_number == q.question_number:
                    current_group.append((page, q))
                else:
                    groups.append(current_group)
                    current_group = [(page, q)]
            else:
                groups.append(current_group)
                current_group = [(page, q)]
        else:
            current_group = [(page, q)]

        if not q.continued and current_group:
            # Check if next item continues this group
            if i + 1 < len(sorted_qs):
                _, next_q = sorted_qs[i + 1]
                if next_q.continued_from:
                    continue
            groups.append(current_group)
            current_group = []

    if current_group:
        groups.append(current_group)

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
    Stage 2: Detect question bounding boxes via Qwen V3.
    Stage 3: Match cross-page continuations, crop, trim, merge, save.
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
        questions = detector.detect(page_path, paper.exam_type, page_number=i)
        for q in questions:
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

    # Stage 3: Match, crop, trim, merge
    question_groups = match_continued_questions(raw_questions)
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
        question_number = group[0][1].question_number

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
