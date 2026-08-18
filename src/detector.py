"""Qwen V3 vision model integration for question detection."""

from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path

from openai import OpenAI

from src.models import Bbox, QuestionBbox

logger = logging.getLogger(__name__)

EXAM1_SYSTEM_PROMPT = """\
You are an expert at analyzing VCE mathematics exam layouts.

This is page {page_number} of a VCE Exam 1 (Short Answer Questions).
Image dimensions: {width}px wide x {height}px tall. Use these exact pixel coordinates.

Identify every TOP-LEVEL question on this page. A top-level question is the \
highest-numbered unit (e.g., "Question 1", "1."). Sub-parts labeled (a), (b), \
i., ii. belong to their parent question — do NOT split them.

For each top-level question, return a bounding box that:
- Starts at the question number/title
- Extends downward to include ALL sub-parts AND all answer/working lines
- Extends horizontally across the full question width
- Excludes page headers, footers, and watermarks

Also determine:
- marks: total marks shown. Look for "[X marks]" or "X marks" near the question. \
Use null if not visible.
- continued: true ONLY if the question is clearly cut off at the page bottom and \
appears incomplete. False if it ends naturally on this page.
- continued_from: true ONLY if this question began on a previous page (starts \
mid-sentence or shows continuation indicator).

If this page contains no exam questions — for example, it is a cover page, \
a formula sheet, a blank page, instructions only, or a "TURN OVER" / \
"END OF SECTION" page — return {{"page": {page_number}, "questions": []}}.

Return ONLY a JSON object (no markdown, no explanation):
{{"page": {page_number}, "questions": [{{"question_number": "...", \
"bbox": {{"x": 0, "y": 0, "w": 0, "h": 0}}, "marks": null, \
"continued": false, "continued_from": false}}]}}"""

EXAM2_SYSTEM_PROMPT = """\
You are an expert at analyzing VCE mathematics exam layouts.

This is page {page_number} of a VCE Exam 2 (Multiple Choice + Extended Response).
Image dimensions: {width}px wide x {height}px tall. Use these exact pixel coordinates.

This exam contains TWO question formats on different pages/sections:
1. Multiple Choice Questions (MCQ): Short numbered questions (1-20) with \
options A-E. Often in 2-column layout.
2. Extended Response (ER): Longer numbered questions (1-4) with sub-parts \
(a), (b), i., ii. Often full-width.

Identify every TOP-LEVEL question on this page:
- For MCQ: Each numbered multiple-choice block (question + options A-E) is \
ONE top-level question.
- For ER: Each numbered question with ALL its sub-parts is ONE top-level question.
- Sub-parts are NEVER separate questions.

For each question, return a bounding box that:
- Starts at the question number
- Extends downward to include all options (MCQ) or all sub-parts + working lines (ER)
- Extends horizontally across the full question
- Excludes page headers, footers, and watermarks

Also determine:
- marks: total marks shown. MCQ = usually 1. ER = look for "[X marks]". \
Use null if not visible.
- continued: true ONLY if question is cut off at the bottom of this page.
- continued_from: true ONLY if question began on a previous page.

If this page contains no exam questions — for example, it is a cover page, \
a formula sheet, a blank page, instructions only, or a "TURN OVER" / \
"END OF SECTION" page — return {{"page": {page_number}, "questions": []}}.

Return ONLY a JSON object (no markdown, no explanation):
{{"page": {page_number}, "questions": [{{"question_number": "...", \
"bbox": {{"x": 0, "y": 0, "w": 0, "h": 0}}, "marks": null, \
"continued": false, "continued_from": false}}]}}"""


def _fix_malformed_bbox(text: str) -> str:
    """Fix common VLM bbox formatting errors before JSON parsing.

    Handles:
    - {"x": 138, 66, 941, 398} → {"x": 138, "y": 66, "w": 941, "h": 398}
    - [138, 66, 941, 398] as bbox value → {"x": 138, ...}
    """
    # Fix bbox objects missing keys: {"x": N, N, N, N} → {"x": N, "y": N, "w": N, "h": N}
    text = re.sub(
        r'"bbox"\s*:\s*\{\s*"x"\s*:\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\}',
        r'"bbox": {"x": \1, "y": \2, "w": \3, "h": \4}',
        text,
    )
    # Fix bbox as 4-element array: [N, N, N, N] → {"x": N, "y": N, "w": N, "h": N}
    text = re.sub(
        r'"bbox"\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]',
        r'"bbox": {"x": \1, "y": \2, "w": \3, "h": \4}',
        text,
    )
    return text

def _cast_into_int(value) -> int | None:
    """Casts a value into int if possible
    Prevents strings from crossing into typed code
    """
    if value is None:
        return None
    try:
        return int(value)
    except:
        return None

def parse_detection_response(
    raw_json: str,
    page_width: int,
    page_height: int,
) -> list[QuestionBbox]:
    """Parse Qwen V3 JSON response into QuestionBbox objects.

    Handles invalid JSON, missing keys, out-of-bounds bboxes, zero-area
    bboxes, and markdown-wrapped JSON. Never raises — returns empty list
    on failure.
    """
    text = raw_json.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove opening fence (``` or ```json)
        lines = lines[1:]
        # Remove closing fence if present
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Fix common VLM bbox formatting errors before parsing
    text = _fix_malformed_bbox(text)

    # Try multiple parsing strategies
    data = None
    for attempt in range(3):
        try:
            parsed = json.loads(text)
            # If result is a string, it's double-encoded JSON — parse again
            if isinstance(parsed, str):
                text = _fix_malformed_bbox(parsed.strip())
                continue
            data = parsed
            break
        except json.JSONDecodeError:
            # If text starts with a quote, try stripping surrounding quotes
            if attempt == 0 and text.startswith('"') and text.endswith('"'):
                try:
                    text = _fix_malformed_bbox(json.loads(text))
                except json.JSONDecodeError:
                    pass
            elif attempt == 1:
                # Last resort: extract first { ... } block
                start = text.find("{")
                end = text.rfind("}")
                if start >= 0 and end > start:
                    text = _fix_malformed_bbox(text[start:end + 1])
            else:
                logger.warning("Failed to parse JSON after 3 attempts: %.200s...", raw_json)
                return []

    if not isinstance(data, dict) or "questions" not in data:
        return []

    questions: list[QuestionBbox] = []
    for entry in data["questions"]:
        try:
            bbox_raw = entry["bbox"]
            # Handle array format: [x, y, w, h]
            if isinstance(bbox_raw, list) and len(bbox_raw) == 4:
                bbox = Bbox(
                    x=int(bbox_raw[0]), y=int(bbox_raw[1]),
                    w=int(bbox_raw[2]), h=int(bbox_raw[3]),
                )
            else:
                bbox = Bbox(
                    x=int(bbox_raw.get("x", 0)),
                    y=int(bbox_raw.get("y", 0)),
                    w=int(bbox_raw.get("w", 0)),
                    h=int(bbox_raw.get("h", 0)),
                )
            bbox = bbox.clamp(page_width, page_height)

            if not bbox.is_valid(page_width, page_height):
                logger.warning(
                    "Skipping question '%s': bbox has zero or negative area",
                    entry.get("question_number", "?"),
                )
                continue

            questions.append(QuestionBbox(
                question_number=str(entry.get("question_number", "")),
                bbox=bbox,
                marks=_cast_into_int(entry.get("marks")),
                continued=bool(entry.get("continued", False)),
                continued_from=bool(entry.get("continued_from", False)),
            ))
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("Skipping malformed question entry: %s", e)
            continue

    return questions


class Detector:
    """Calls Qwen V3 API to detect question bounding boxes on exam pages."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 120):
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model

    def detect(
        self,
        page_image: Path,
        exam_type: str,
        page_number: int,
    ) -> list[QuestionBbox]:
        """Detect top-level question bounding boxes on a single page image.

        Args:
            page_image: Path to the rendered page PNG.
            exam_type: "exam1" (SAQ) or "exam2" (MCQ+ER).
            page_number: 1-based page number (used in the prompt).

        Returns:
            List of detected questions with bounding boxes.
            Empty list if nothing detected or on error.
        """
        from PIL import Image as PILImage

        with PILImage.open(page_image) as img:
            width, height = img.size

        template = EXAM1_SYSTEM_PROMPT if exam_type == "exam1" else EXAM2_SYSTEM_PROMPT
        prompt = template.format(page_number=page_number, width=width, height=height)
        image_b64 = self._encode_image(page_image)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
                temperature=0.0,
                max_tokens=4096,
            )
        except Exception as e:
            logger.error(
                "API call failed for page %d of exam type '%s': %s",
                page_number, exam_type, e,
            )
            return []

        raw_text = response.choices[0].message.content or ""
        logger.debug("Raw detector response for page %d: %s", page_number, raw_text[:200])

        questions = parse_detection_response(raw_text, width, height)
        logger.info("Page %d: detected %d questions", page_number, len(questions))
        return questions

    def _encode_image(self, image_path: Path) -> str:
        """Read image file and return base64-encoded string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
