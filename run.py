from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import yaml

from src.detector import Detector
from src.models import build_index
from src.pipeline import process_paper, scan_input_dir


def main():
    parser = argparse.ArgumentParser(
        description="Extract exam questions as images from VCE math PDFs.",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Directory containing exam PDFs (recursively scanned).",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Directory to write extracted question images and mappings.",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config.yaml",
        help="Path to YAML config (default: config.yaml).",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Page rendering DPI (default: 300)."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.verbose:
        # Only enable debug for our own code, not third-party libs
        logging.getLogger("src").setLevel(logging.DEBUG)
        logging.getLogger("pipeline").setLevel(logging.DEBUG)
    logger = logging.getLogger("pipeline")

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    api = config.get("api", {})
    base_url = api.get("base_url", "")
    api_key = api.get("api_key", "")
    model = api.get("model", "qwen-vl-max")

    if not base_url or not api_key:
        logger.error("API base_url and api_key must be set in config.yaml")
        sys.exit(1)

    # Discover papers
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        logger.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    papers = scan_input_dir(input_dir)
    if not papers:
        logger.error("No valid exam PDFs found in %s", input_dir)
        logger.info(
            "Expected filename format: {year}-{source}-exam-{1,2}.pdf "
            "(e.g., 2024-vcaa-exam-1.pdf)"
        )
        sys.exit(1)

    logger.info("Found %d papers to process", len(papers))

    detector = Detector(base_url=base_url, api_key=api_key, model=model)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    exam_results = []

    with tempfile.TemporaryDirectory(prefix="vce_pipeline_") as tmp:
        temp_dir = Path(tmp)

        for i, paper in enumerate(papers, start=1):
            logger.info("=" * 60)
            logger.info(
                "[%d/%d] Processing %s (type=%s, source=%s)",
                i,
                len(papers),
                paper.key,
                paper.exam_type,
                paper.source,
            )
            try:
                result = process_paper(
                    paper=paper,
                    output_dir=output_dir,
                    detector=detector,
                    temp_dir=temp_dir / paper.key,
                    dpi=args.dpi,
                )
                exam_results.append(result)
                logger.info(
                    "[%s] ✓ Extracted %d questions", paper.key, len(result.questions)
                )
            except Exception as e:
                logger.exception("[%s] ✗ Failed: %s", paper.key, e)

    # Write aggregate index
    index = build_index(exam_results)
    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))

    logger.info("=" * 60)
    logger.info(
        "DONE — %d/%d papers processed",
        sum(1 for r in exam_results if r.questions),
        len(papers),
    )
    logger.info("Total questions extracted: %d", index["total_questions"])
    logger.info("Output: %s", output_dir)
    logger.info("Index:  %s", index_path)


if __name__ == "__main__":
    main()
