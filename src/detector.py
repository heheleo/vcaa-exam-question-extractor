"""Vision model integration for question detection."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

from openai import OpenAI

from src.models import Bbox, QuestionBbox

logger = logging.getLogger(__name__)

DETECTION_PROMPT = """\
You are reading page {page_number} of a VCE mathematics exam. The image is {width}x{height} pixels.

Find every printed question or sub-part label on this page: things like \
"Question 3", "3.", "4", "a.", "b.", "i.", "ii.". Ignore page headers, \
footers, watermarks, and multiple-choice option letters (A., B., C., D., E.).

For each label return:
- label: the exact printed text, nothing added, e.g. 'a.', '3', 'ii.'
- box_2d: [ymin, xmin, ymax, xmax] in the built-in bounding box format, \
values normalized to 0-1000. The box is the rectangle from this label down \
to the next label, covering the full width of the question (one column on \
two-column pages) and its working and answer lines
- marks: the marks printed with the label (e.g. 3 for "(3 marks)"), or null
- text: all printed text in this block, transcribed exactly

If the page has no question content (cover page, formula sheet, blank page), \
return {{"page": {page_number}, "blocks": []}}.

Return ONLY JSON, nothing else:
{{"page": {page_number}, "blocks": [{{"label": "3", "box_2d": [80, 60, 400, 940], "marks": 3, "text": "Question 3\nFind the derivative of ..."}}]}}
"""


def _box_2d_to_bbox(box: list, page_width: int, page_height: int) -> Bbox:
    """Convert Gemini's box_2d [ymin, xmin, ymax, xmax] (0-1000) to pixels."""
    ymin, xmin, ymax, xmax = [float(v) for v in box]
    x1 = round(xmin / 1000 * page_width)
    y1 = round(ymin / 1000 * page_height)
    x2 = round(xmax / 1000 * page_width)
    y2 = round(ymax / 1000 * page_height)
    return Bbox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)


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
    """Parse JSON response into QuestionBbox objects.

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

    # Try multiple parsing strategies
    data = None
    for attempt in range(3):
        try:
            parsed = json.loads(text)
            # If result is a string, it's double-encoded JSON — parse again
            if isinstance(parsed, str):
                text = parsed.strip()
                continue
            data = parsed
            break
        except json.JSONDecodeError:
            # If text starts with a quote, try stripping surrounding quotes
            if attempt == 0 and text.startswith('"') and text.endswith('"'):
                try:
                    text = json.loads(text)
                except json.JSONDecodeError:
                    pass
            elif attempt == 1:
                # Last resort: extract first { ... } block
                start = text.find("{")
                end = text.rfind("}")
                if start >= 0 and end > start:
                    text = text[start : end + 1]
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
            box_raw = entry["box_2d"]
            if not isinstance(box_raw, list) or len(box_raw) != 4:
                logger.warning("Skipping block '%s': malformed box_2d", label)
                continue
            bbox = _box_2d_to_bbox(box_raw, page_width, page_height).clamp(
                page_width, page_height
            )

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
                    text=str(entry.get("text", "")).strip(),
                )
            )
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            logger.warning("Skipping malformed block entry: %s", e)
            continue

    return blocks


class Detector:
    """Calls a vision model API to detect question bounding boxes on exam pages."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 120):
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model

    def detect(
        self,
        page_image: Path,
        page_number: int,
    ) -> list[QuestionBbox]:
        """Detect question blocks on a single page image.

        Args:
            page_image: Path to the rendered page PNG.
            page_number: 1-based page number (used in the prompt).

        Returns:
            List of detected blocks with labels and bounding boxes.
            Empty list if nothing detected or on error.
        """
        from PIL import Image as PILImage

        with PILImage.open(page_image) as img:
            width, height = img.size

        prompt = DETECTION_PROMPT.format(
            page_number=page_number, width=width, height=height
        )
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
                max_tokens=8192,
            )
        except Exception as e:
            logger.error("API call failed for page %d: %s", page_number, e)
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
