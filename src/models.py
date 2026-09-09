"""Shared data models for the question extraction pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Bbox:
    """Pixel-coordinate bounding box."""

    x: int
    y: int
    w: int
    h: int

    def clamp(self, width: int, height: int) -> Bbox:
        """Clamp coordinates to stay within [0, width) x [0, height)."""
        x = max(0, min(self.x, width - 1))
        y = max(0, min(self.y, height - 1))
        w = min(self.w, width - x)
        h = min(self.h, height - y)
        return Bbox(x=x, y=y, w=w, h=h)

    def is_valid(self, width: int, height: int) -> bool:
        """Check bbox is fully in bounds and has positive area."""
        return (
            0 <= self.x < width
            and 0 <= self.y < height
            and self.w > 0
            and self.h > 0
            and self.x + self.w <= width
            and self.y + self.h <= height
        )


@dataclass
class QuestionBbox:
    """A detected question block on a single page."""

    label: str
    bbox: Bbox
    marks: int | None = None
    text: str = ""  # OCRed content of the block, for later label classification


@dataclass
class PaperMeta:
    """Parsed metadata for an input paper."""

    path: Path
    year: int
    source: str
    exam_type: str

    @property
    def key(self) -> str:
        return f"{self.year}-{self.source}-{self.exam_type}"


@dataclass
class QuestionResult:
    """Final output record for one extracted question."""

    number: str
    image: str  # relative filename, e.g. "q01.png" or "mc01.png"
    marks: int | None
    pages: list[int]
    has_subquestions: bool
    cross_page: bool
    text: str = ""  # concatenated OCRed text of the question's blocks
    kind: str = "short"  # "mcq" (exam2 Section A) or "short"


@dataclass
class ExamResult:
    """Output record for one processed exam."""

    key: str
    year: int
    source: str
    exam_type: str
    questions: list[QuestionResult] = field(default_factory=list)
    path: str = ""  # original PDF path for index.json

    def to_mapping(self) -> dict:
        return {
            "year": self.year,
            "source": self.source,
            "exam_type": self.exam_type,
            "total_questions": len(self.questions),
            "questions": [
                {
                    "number": q.number,
                    "image": q.image,
                    "type": q.kind,
                    "marks": q.marks,
                    "pages": q.pages,
                    "has_subquestions": q.has_subquestions,
                    "cross_page": q.cross_page,
                    "text": q.text,
                }
                for q in self.questions
            ],
        }


# Filename pattern: {year}-{source}-exam-{n}.pdf
_FILENAME_RE = re.compile(
    r"(?P<year>\d{4})-(?P<source>[a-zA-Z0-9_-]+?)-exam-(?P<exam_n>[12])\.pdf$"
)


def parse_filename(path: str | Path) -> PaperMeta | None:
    """Parse paper metadata from a filename.

    Expected format: {year}-{source}-exam-{n}.pdf
    Example: 2024-vcaa-exam-1.pdf -> year=2024, source=vcaa, exam_type=exam1

    Returns None if the filename doesn't match.
    """
    name = Path(path).name
    m = _FILENAME_RE.match(name)
    if m is None:
        return None
    return PaperMeta(
        path=Path(path),
        year=int(m.group("year")),
        source=m.group("source").lower(),
        exam_type=f"exam{m.group('exam_n')}",
    )


def build_index(exams: list[ExamResult]) -> dict:
    """Build the aggregate index.json structure from processed exams."""
    entries = []
    for e in sorted(exams, key=lambda e: (e.year, e.source, e.exam_type)):
        entries.append(
            {
                "key": e.key,
                "year": e.year,
                "source": e.source,
                "type": e.exam_type,
                "questions": len(e.questions),
                "input_path": e.path,
            }
        )
    return {
        "exams": entries,
        "total_exams": len(entries),
        "total_questions": sum(len(e.questions) for e in exams),
    }
