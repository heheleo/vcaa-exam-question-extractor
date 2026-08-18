"""Vision model integration for question detection."""

from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path

from openai import OpenAI

from src.models import Bbox, QuestionBbox

logger = logging.getLogger(__name__)

_EXAM1_INTRO = """\
You are reading a page of a VCE mathematics exam.

This is page {page_number} of a VCE Exam 1 (short answer questions).
Image dimensions: {width}px wide x {height}px tall. Use these exact pixel coordinates."""

_EXAM2_INTRO = """\
You are reading a page of a VCE mathematics exam.

This is page {page_number} of a VCE Exam 2 (Multiple Choice + Extended Response).
Image dimensions: {width}px wide x {height}px tall. Use these exact pixel coordinates.

This exam has two sections: Multiple Choice questions (numbered 1-20, with \
options A-E, often in a 2-column layout) and Extended Response questions \
(numbered 1-4, with sub-parts such as a., b., i., ii.)."""

_BLOCK_INSTRUCTIONS = """\

Find every BLOCK of question content on this page. A block begins at a printed label:
- a top-level question label, such as "Question 3" or "3."
- or a part label, such as "a.", "b.", "i.", "ii.", "c."

A block extends downward from its label until the next block's label or the end \
of the question content (include answer and working lines). Include the full \
width of the block. Exclude page headers, footers, and watermarks.

For each block, return:
- label: the label at the top of the block, copied VERBATIM (e.g. "Question 3", \
"3.", "a.", "ii."). Never guess or renumber.
- bbox: the smallest box covering the entire block.
- marks: total marks shown for the block ("[X marks]" or "X marks"), or null if \
none visible. Marks usually appear at the top-level question only.

If this page contains no question content — for example, a cover page, a \
formula sheet, a blank page, instructions only, or a "TURN OVER" / \
"END OF SECTION" page — return {{"page": {page_number}, "blocks": []}}.

Return ONLY a JSON object (no markdown, no explanation):
{{"page": {page_number}, "blocks": [{{"label": "...", \
"bbox": {{"x": 0, "y": 0, "w": 0, "h": 0}}, "marks": null}}]}}"""

EXAM1_SYSTEM_PROMPT = _EXAM1_INTRO + _BLOCK_INSTRUCTIONS
EXAM2_SYSTEM_PROMPT = _EXAM2_INTRO + _BLOCK_INSTRUCTIONS


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
    except (ValueError, TypeError):
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
                    text = _fix_malformed_bbox(text[start : end + 1])
            else:
                logger.warning(
                    "Failed to parse JSON after 3 attempts: %.200s...", raw_json
                )
                return []

    if not isinstance(data, dict) or "blocks" not in data:
        return []

    blocks: list[QuestionBbox] = []
    for entry in data["blocks"]:
        try:
            label = str(entry.get("label", "")).strip()
            if not label:
                logger.warning("Skipping block with empty label")
                continue
            bbox_raw = entry["bbox"]
            # Handle array format: [x, y, w, h]
            if isinstance(bbox_raw, list) and len(bbox_raw) == 4:
                bbox = Bbox(
                    x=int(bbox_raw[0]),
                    y=int(bbox_raw[1]),
                    w=int(bbox_raw[2]),
                    h=int(bbox_raw[3]),
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
                    "Skipping block '%s': bbox has zero or negative area",
                    label,
                )
                continue

            blocks.append(
                QuestionBbox(
                    label=label,
                    bbox=bbox,
                    marks=_cast_into_int(entry.get("marks")),
                )
            )
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            logger.warning("Skipping malformed block entry: %s", e)
            continue

    return blocks


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
        """Detect question blocks on a single page image.

        Args:
            page_image: Path to the rendered page PNG.
            exam_type: "exam1" (SAQ) or "exam2" (MCQ+ER).
            page_number: 1-based page number (used in the prompt).

        Returns:
            List of detected blocks with labels and bounding boxes.
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
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                temperature=0.0,
                max_tokens=4096,
            )
        except Exception as e:
            logger.error(
                "API call failed for page %d of exam type '%s': %s",
                page_number,
                exam_type,
                e,
            )
            return []

        raw_text = response.choices[0].message.content or ""
        logger.debug(
            "Raw detector response for page %d: %s", page_number, raw_text[:200]
        )

        blocks = parse_detection_response(raw_text, width, height)
        logger.info("Page %d: detected %d blocks", page_number, len(blocks))
        return blocks

    def _encode_image(self, image_path: Path) -> str:
        """Read image file and return base64-encoded string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
