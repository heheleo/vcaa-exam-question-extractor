import json
from pathlib import Path

import pytest
from PIL import Image

from src.detector import (
    Detector,
    parse_detection_response,
    _fix_malformed_bbox,
    EXAM1_SYSTEM_PROMPT,
    EXAM2_SYSTEM_PROMPT,
)
from src.models import Bbox


def test_parse_valid_response():
    response = json.dumps({
        "page": 1,
        "questions": [
            {"question_number": "1", "bbox": {"x": 50, "y": 100, "w": 700, "h": 200},
             "marks": 5, "continued": False, "continued_from": False},
            {"question_number": "2", "bbox": {"x": 50, "y": 310, "w": 700, "h": 300},
             "marks": 8, "continued": True, "continued_from": False},
        ]
    })
    questions = parse_detection_response(response, page_width=800, page_height=1000)
    assert len(questions) == 2
    assert questions[0].question_number == "1"
    assert questions[0].marks == 5
    assert questions[0].continued is False
    assert questions[1].continued is True


def test_parse_response_with_nulls():
    response = json.dumps({
        "page": 3,
        "questions": [
            {"question_number": "5", "bbox": {"x": 0, "y": 0, "w": 100, "h": 50},
             "marks": None, "continued": False, "continued_from": True},
        ]
    })
    questions = parse_detection_response(response, page_width=200, page_height=200)
    assert len(questions) == 1
    assert questions[0].marks is None
    assert questions[0].continued_from is True


def test_parse_bbox_clamped_to_bounds():
    response = json.dumps({
        "page": 1,
        "questions": [
            {"question_number": "1", "bbox": {"x": -10, "y": -5, "w": 9999, "h": 9999},
             "marks": None, "continued": False, "continued_from": False},
        ]
    })
    questions = parse_detection_response(response, page_width=800, page_height=600)
    assert len(questions) == 1
    b = questions[0].bbox
    assert b.x == 0
    assert b.y == 0
    assert b.x + b.w <= 800
    assert b.y + b.h <= 600


def test_parse_invalid_json():
    assert parse_detection_response("not json", 800, 600) == []


def test_parse_double_encoded_json():
    """JSON wrapped as a JSON string should be handled."""
    inner = '{"page": 1, "questions": [{"question_number": "1", "bbox": {"x": 0, "y": 0, "w": 100, "h": 50}, "marks": null, "continued": false, "continued_from": false}]}'
    response = json.dumps(inner)  # Double-encode: a JSON string
    questions = parse_detection_response(response, 800, 600)
    assert len(questions) == 1
    assert questions[0].question_number == "1"


def test_parse_quoted_json():
    """JSON wrapped in regular quotes should be handled."""
    inner = '{"page": 1, "questions": [{"question_number": "2", "bbox": {"x": 10, "y": 20, "w": 100, "h": 50}, "marks": null, "continued": false, "continued_from": false}]}'
    response = '"' + inner + '"'  # Wrapped in quotes
    questions = parse_detection_response(response, 800, 600)
    assert len(questions) == 1
    assert questions[0].question_number == "2"


def test_parse_markdown_wrapped_json():
    """JSON inside ```json ... ``` fences should be extracted."""
    inner = '{"page": 1, "questions": [{"question_number": "3", "bbox": {"x": 0, "y": 0, "w": 50, "h": 30}, "marks": 2, "continued": false, "continued_from": false}]}'
    response = "```json\n" + inner + "\n```"
    questions = parse_detection_response(response, 800, 600)
    assert len(questions) == 1
    assert questions[0].question_number == "3"


def test_parse_bbox_array_format():
    """Bbox as [x, y, w, h] array should be handled."""
    response = '{"page": 1, "questions": [{"question_number": "4", "bbox": [10, 20, 100, 200], "marks": 3, "continued": false, "continued_from": false}]}'
    questions = parse_detection_response(response, 800, 600)
    assert len(questions) == 1
    assert questions[0].bbox.x == 10
    assert questions[0].bbox.y == 20
    assert questions[0].bbox.w == 100
    assert questions[0].bbox.h == 200


def test_parse_bbox_missing_keys():
    """Bbox with missing y/w/h keys should be fixed before parsing."""
    response = '{"page": 1, "questions": [{"question_number": "5", "bbox": {"x": 50, 100, 300, 200}, "marks": 2, "continued": false, "continued_from": false}]}'
    questions = parse_detection_response(response, 800, 600)
    assert len(questions) == 1
    assert questions[0].bbox.x == 50
    assert questions[0].bbox.y == 100
    assert questions[0].bbox.w == 300
    assert questions[0].bbox.h == 200


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


def test_parse_missing_questions_key():
    assert parse_detection_response('{"something": "else"}', 800, 600) == []


def test_parse_empty_questions():
    assert parse_detection_response('{"page": 1, "questions": []}', 800, 600) == []


def test_parse_skips_zero_area_bbox():
    response = json.dumps({
        "page": 1,
        "questions": [
            {"question_number": "bad", "bbox": {"x": 0, "y": 0, "w": 0, "h": 0},
             "marks": None, "continued": False, "continued_from": False},
            {"question_number": "good", "bbox": {"x": 10, "y": 10, "w": 100, "h": 50},
             "marks": None, "continued": False, "continued_from": False},
        ]
    })
    questions = parse_detection_response(response, 800, 600)
    assert len(questions) == 1
    assert questions[0].question_number == "good"


def test_exam1_prompt_contains_key_instructions():
    assert "Exam 1" in EXAM1_SYSTEM_PROMPT
    assert "top-level" in EXAM1_SYSTEM_PROMPT.lower()
    assert "continued" in EXAM1_SYSTEM_PROMPT.lower()
    assert "no exam questions" in EXAM1_SYSTEM_PROMPT.lower()
    assert "formula sheet" in EXAM1_SYSTEM_PROMPT.lower()


def test_exam2_prompt_contains_key_instructions():
    assert "Exam 2" in EXAM2_SYSTEM_PROMPT
    assert "Multiple Choice" in EXAM2_SYSTEM_PROMPT
    assert "Extended Response" in EXAM2_SYSTEM_PROMPT
    assert "no exam questions" in EXAM2_SYSTEM_PROMPT.lower()
    assert "formula sheet" in EXAM2_SYSTEM_PROMPT.lower()


def test_encode_image_base64(tmp_path):
    img_path = tmp_path / "test.png"
    Image.new("RGB", (100, 50), color="white").save(img_path)
    detector = Detector(base_url="http://fake", api_key="fake", model="fake")
    b64 = detector._encode_image(img_path)
    assert isinstance(b64, str)
    assert len(b64) > 0

def test_parse_marks_string_bleed():
    response = json.dumps({
        "page": 1,
        "questions": [
            {"question_number": "1", "bbox": {"x": 0, "y": 0, "w": 100, "h": 50},
             "marks": "2", "continued": False, "continued_from": False},
            {"question_number": "2", "bbox": {"x": 0, "y": 60, "w": 100, "h": 50},
             "marks": "2.5", "continued": False, "continued_from": False},
        ]
    })
    questions = parse_detection_response(response, 800, 600)
    assert questions[0].marks == 2
    assert questions[1].marks is None

def test_parse_skips_malformed_bbox_shapes():
    response = json.dumps({
        "page": 1,
        "questions": [
            {"question_number": "bad1", "bbox": [10, 20, 100, 200, 0.9, 1],
             "marks": None, "continued": False, "continued_from": False},
            {"question_number": "bad2", "bbox": 5,
             "marks": None, "continued": False, "continued_from": False},
            {"question_number": "good", "bbox": {"x": 10, "y": 10, "w": 100, "h": 50},
             "marks": None, "continued": False, "continued_from": False},
        ]
    })
    questions = parse_detection_response(response, 800, 600)
    assert len(questions) == 1
    assert questions[0].question_number == "good"