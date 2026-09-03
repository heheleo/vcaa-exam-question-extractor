"""PDF rendering, detection, and cropping pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.post_ocr import group_questions, extract_number
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
        for q in detector.detect(page_path, page_number=i):
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
        question_text = "\n".join(q.text for _, q in group if q.text)

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
                text=question_text,
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
