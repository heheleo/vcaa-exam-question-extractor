"""PDF rendering utilities using PyMuPDF."""

from __future__ import annotations

from pathlib import Path

import fitz


def render_pages(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 300,
) -> list[Path]:
    """Render every page of a PDF as a PNG image.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory to write page images into (created if missing).
        dpi: Rendering resolution in dots per inch.

    Returns:
        List of paths to the rendered PNG images, sorted by page number.
        Filenames are page_001.png, page_002.png, etc.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    try:
        paths: list[Path] = []
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        for i in range(doc.page_count):
            page = doc[i]
            pix = page.get_pixmap(matrix=mat)
            out_path = output_dir / f"page_{i + 1:03d}.png"
            pix.save(str(out_path))
            paths.append(out_path)

        return paths
    finally:
        doc.close()
