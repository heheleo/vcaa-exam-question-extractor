import json

from PIL import Image

from src.detector import (
    EXAM1_SYSTEM_PROMPT,
    EXAM2_SYSTEM_PROMPT,
    Detector,
    _fix_malformed_bbox,
    parse_detection_response,
)


def test_parse_valid_response():
    response = json.dumps(
        {
            "page": 1,
            "blocks": [
                {
                    "label": "1",
                    "bbox": {"x": 50, "y": 100, "w": 700, "h": 200},
                    "marks": 5,
                },
                {
                    "label": "2",
                    "bbox": {"x": 50, "y": 310, "w": 700, "h": 300},
                    "marks": 8,
                },
            ],
        }
    )
    blocks = parse_detection_response(response, page_width=800, page_height=1000)
    assert len(blocks) == 2
    assert blocks[0].label == "1"
    assert blocks[0].marks == 5
    assert blocks[1].label == "2"


def test_parse_marks_string_bleed():
    """String marks from the VLM must not escape as str into typed code."""
    response = json.dumps(
        {
            "page": 1,
            "blocks": [
                {
                    "label": "1",
                    "bbox": {"x": 0, "y": 0, "w": 100, "h": 50},
                    "marks": "2",
                },
                {
                    "label": "2",
                    "bbox": {"x": 0, "y": 60, "w": 100, "h": 50},
                    "marks": "2.5",
                },
            ],
        }
    )
    blocks = parse_detection_response(response, 800, 600)
    assert blocks[0].marks == 2
    assert blocks[1].marks is None


def test_parse_response_with_nulls():
    response = json.dumps(
        {
            "page": 3,
            "blocks": [
                {
                    "label": "5",
                    "bbox": {"x": 0, "y": 0, "w": 100, "h": 50},
                    "marks": None,
                },
            ],
        }
    )
    blocks = parse_detection_response(response, page_width=200, page_height=200)
    assert len(blocks) == 1
    assert blocks[0].marks is None


def test_parse_bbox_clamped_to_bounds():
    response = json.dumps(
        {
            "page": 1,
            "blocks": [
                {
                    "label": "1",
                    "bbox": {"x": -10, "y": -5, "w": 9999, "h": 9999},
                    "marks": None,
                },
            ],
        }
    )
    blocks = parse_detection_response(response, page_width=800, page_height=600)
    assert len(blocks) == 1
    b = blocks[0].bbox
    assert b.x == 0
    assert b.y == 0
    assert b.x + b.w <= 800
    assert b.y + b.h <= 600


def test_parse_invalid_json():
    assert parse_detection_response("not json", 800, 600) == []


def test_parse_double_encoded_json():
    """JSON wrapped as a JSON string should be handled."""
    inner = '{"page": 1, "blocks": [{"label": "1", "bbox": {"x": 0, "y": 0, "w": 100, "h": 50}, "marks": null}]}'
    response = json.dumps(inner)  # Double-encode: a JSON string
    blocks = parse_detection_response(response, 800, 600)
    assert len(blocks) == 1
    assert blocks[0].label == "1"


def test_parse_quoted_json():
    """JSON wrapped in regular quotes should be handled."""
    inner = '{"page": 1, "blocks": [{"label": "2", "bbox": {"x": 10, "y": 20, "w": 100, "h": 50}, "marks": null}]}'
    response = '"' + inner + '"'  # Wrapped in quotes
    blocks = parse_detection_response(response, 800, 600)
    assert len(blocks) == 1
    assert blocks[0].label == "2"


def test_parse_markdown_wrapped_json():
    """JSON inside ```json ... ``` fences should be extracted."""
    inner = '{"page": 1, "blocks": [{"label": "3", "bbox": {"x": 0, "y": 0, "w": 50, "h": 30}, "marks": 2}]}'
    response = "```json\n" + inner + "\n```"
    blocks = parse_detection_response(response, 800, 600)
    assert len(blocks) == 1
    assert blocks[0].label == "3"


def test_parse_bbox_array_format():
    """Bbox as [x, y, w, h] array should be handled."""
    response = '{"page": 1, "blocks": [{"label": "4", "bbox": [10, 20, 100, 200], "marks": 3}]}'
    blocks = parse_detection_response(response, 800, 600)
    assert len(blocks) == 1
    assert blocks[0].bbox.x == 10
    assert blocks[0].bbox.y == 20
    assert blocks[0].bbox.w == 100
    assert blocks[0].bbox.h == 200


def test_parse_bbox_missing_keys():
    """Bbox with missing y/w/h keys should be fixed before parsing."""
    response = '{"page": 1, "blocks": [{"label": "5", "bbox": {"x": 50, 100, 300, 200}, "marks": 2}]}'
    blocks = parse_detection_response(response, 800, 600)
    assert len(blocks) == 1
    assert blocks[0].bbox.x == 50
    assert blocks[0].bbox.y == 100
    assert blocks[0].bbox.w == 300
    assert blocks[0].bbox.h == 200


def test_parse_skips_malformed_bbox_shapes():
    """One malformed bbox must not lose the rest of the page's blocks."""
    response = json.dumps(
        {
            "page": 1,
            "blocks": [
                {
                    "label": "bad1",
                    "bbox": [10, 20, 100, 200, 0.9, 1],
                    "marks": None,
                },
                {
                    "label": "bad2",
                    "bbox": 5,
                    "marks": None,
                },
                {
                    "label": "good",
                    "bbox": {"x": 10, "y": 10, "w": 100, "h": 50},
                    "marks": None,
                },
            ],
        }
    )
    blocks = parse_detection_response(response, 800, 600)
    assert len(blocks) == 1
    assert blocks[0].label == "good"


def test_parse_skips_empty_label():
    response = json.dumps(
        {
            "page": 1,
            "blocks": [
                {
                    "label": "",
                    "bbox": {"x": 0, "y": 0, "w": 100, "h": 50},
                    "marks": None,
                },
            ],
        }
    )
    assert parse_detection_response(response, 800, 600) == []


def test_fix_malformed_bbox_missing_keys():
    result = _fix_malformed_bbox('{"bbox": {"x": 1, 2, 3, 4}}')
    assert '"y": 2' in result
    assert '"w": 3' in result
    assert '"h": 4' in result


def test_fix_malformed_bbox_array():
    result = _fix_malformed_bbox('{"bbox": [10, 20, 30, 40]}')
    assert '"x": 10' in result
    assert '"y": 20' in result
    assert '"w": 30' in result
    assert '"h": 40' in result


def test_fix_malformed_bbox_noop():
    """Already-valid bbox should be unchanged."""
    valid = '{"bbox": {"x": 1, "y": 2, "w": 3, "h": 4}}'
    result = _fix_malformed_bbox(valid)
    assert result == valid


def test_parse_missing_blocks_key():
    assert parse_detection_response('{"something": "else"}', 800, 600) == []


def test_parse_empty_blocks():
    assert parse_detection_response('{"page": 1, "blocks": []}', 800, 600) == []


def test_parse_skips_zero_area_bbox():
    response = json.dumps(
        {
            "page": 1,
            "blocks": [
                {
                    "label": "bad",
                    "bbox": {"x": 0, "y": 0, "w": 0, "h": 0},
                    "marks": None,
                },
                {
                    "label": "good",
                    "bbox": {"x": 10, "y": 10, "w": 100, "h": 50},
                    "marks": None,
                },
            ],
        }
    )
    blocks = parse_detection_response(response, 800, 600)
    assert len(blocks) == 1
    assert blocks[0].label == "good"


def test_exam1_prompt_contains_key_instructions():
    assert "Exam 1" in EXAM1_SYSTEM_PROMPT
    assert "label" in EXAM1_SYSTEM_PROMPT.lower()
    assert "verbatim" in EXAM1_SYSTEM_PROMPT.lower()
    assert "blocks" in EXAM1_SYSTEM_PROMPT.lower()
    assert "formula sheet" in EXAM1_SYSTEM_PROMPT.lower()


def test_exam2_prompt_contains_key_instructions():
    assert "Exam 2" in EXAM2_SYSTEM_PROMPT
    assert "Multiple Choice" in EXAM2_SYSTEM_PROMPT
    assert "Extended Response" in EXAM2_SYSTEM_PROMPT
    assert "label" in EXAM2_SYSTEM_PROMPT.lower()
    assert "formula sheet" in EXAM2_SYSTEM_PROMPT.lower()


def test_encode_image_base64(tmp_path):
    img_path = tmp_path / "test.png"
    Image.new("RGB", (100, 50), color="white").save(img_path)
    detector = Detector(base_url="http://fake", api_key="fake", model="fake")
    b64 = detector._encode_image(img_path)
    assert isinstance(b64, str)
    assert len(b64) > 0
