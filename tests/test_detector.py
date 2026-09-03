import json

from PIL import Image

from src.detector import (
    DETECTION_PROMPT,
    DETECTION_SCHEMA,
    Detector,
    _box_2d_to_bbox,
    parse_detection_response,
)


def test_parse_valid_response():
    response = json.dumps(
        {
            "page": 1,
            "blocks": [
                {
                    "label": "1",
                    "box_2d": [100, 50, 300, 750],
                    "marks": 5,
                    "text": "Solve for x",
                },
                {
                    "label": "2",
                    "box_2d": [310, 50, 900, 750],
                    "marks": 8,
                },
            ],
        }
    )
    blocks = parse_detection_response(response, page_width=800, page_height=1000)
    assert len(blocks) == 2
    assert blocks[0].label == "1"
    assert blocks[0].marks == 5
    assert blocks[0].text == "Solve for x"
    assert blocks[1].text == ""  # missing text defaults to empty
    # [100, 50, 300, 750] on 800x1000 -> x=40, y=100, w=560, h=200
    assert blocks[0].bbox.x == 40
    assert blocks[0].bbox.y == 100
    assert blocks[0].bbox.w == 560
    assert blocks[0].bbox.h == 200
    assert blocks[1].label == "2"


def test_parse_marks_string_bleed():
    """String marks from the VLM must not escape as str into typed code."""
    response = json.dumps(
        {
            "page": 1,
            "blocks": [
                {
                    "label": "1",
                    "box_2d": [0, 0, 100, 50],
                    "marks": "2",
                },
                {
                    "label": "2",
                    "box_2d": [60, 0, 160, 50],
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
                    "box_2d": [0, 0, 100, 50],
                    "marks": None,
                },
            ],
        }
    )
    blocks = parse_detection_response(response, page_width=200, page_height=200)
    assert len(blocks) == 1
    assert blocks[0].marks is None


def test_parse_box_2d_clamped_to_bounds():
    """Out-of-range box_2d values must not escape the page."""
    response = json.dumps(
        {
            "page": 1,
            "blocks": [
                {
                    "label": "1",
                    "box_2d": [-100, -50, 2000, 1500],
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
    inner = '{"page": 1, "blocks": [{"label": "1", "box_2d": [0, 0, 100, 50], "marks": null}]}'
    response = json.dumps(inner)  # Double-encode: a JSON string
    blocks = parse_detection_response(response, 800, 600)
    assert len(blocks) == 1
    assert blocks[0].label == "1"


def test_parse_quoted_json():
    """JSON wrapped in regular quotes should be handled."""
    inner = '{"page": 1, "blocks": [{"label": "2", "box_2d": [10, 20, 110, 50], "marks": null}]}'
    response = '"' + inner + '"'  # Wrapped in quotes
    blocks = parse_detection_response(response, 800, 600)
    assert len(blocks) == 1
    assert blocks[0].label == "2"


def test_parse_markdown_wrapped_json():
    """JSON inside ```json ... ``` fences should be extracted."""
    inner = '{"page": 1, "blocks": [{"label": "3", "box_2d": [0, 0, 100, 50], "marks": 2}]}'
    response = "```json\n" + inner + "\n```"
    blocks = parse_detection_response(response, 800, 600)
    assert len(blocks) == 1
    assert blocks[0].label == "3"


def test_parse_repairs_repeated_key_slip():
    """Gemini emits 'label': 'marks': 2 — a key repeated mid-entry."""
    response = (
        '{"page": 1, "blocks": ['
        '{"label": "a.", "box_2d": [100, 59, 500, 860], "marks": 1, "text": "a. State"}, '
        '{"label": "c.", "box_2d": [542, 59, 903, 860], "label": "marks": 2, "text": "c. Sketch"}]}'
    )
    blocks = parse_detection_response(response, 800, 600)
    assert len(blocks) == 2
    assert blocks[1].label == "c."
    assert blocks[1].marks == 2


def test_parse_skips_malformed_box_2d():
    """One malformed box_2d must not lose the rest of the page's blocks."""
    response = json.dumps(
        {
            "page": 1,
            "blocks": [
                {
                    "label": "bad1",
                    "box_2d": [10, 20, 100, 200, 0.9, 1],
                    "marks": None,
                },
                {
                    "label": "bad2",
                    "box_2d": 5,
                    "marks": None,
                },
                {
                    "label": "bad3",
                    "box_2d": [100, 100, 100, 100],  # zero area
                    "marks": None,
                },
                {
                    "label": "good",
                    "box_2d": [10, 10, 100, 50],
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
                    "box_2d": [0, 0, 100, 50],
                    "marks": None,
                },
            ],
        }
    )
    assert parse_detection_response(response, 800, 600) == []


def test_box_2d_to_bbox_scales_by_page_size():
    box = [100, 50, 300, 750]  # ymin, xmin, ymax, xmax in 0-1000
    bbox = _box_2d_to_bbox(box, page_width=800, page_height=1000)
    assert bbox.x == 40
    assert bbox.y == 100
    assert bbox.w == 560
    assert bbox.h == 200


def test_parse_missing_blocks_key():
    assert parse_detection_response('{"something": "else"}', 800, 600) == []


def test_parse_empty_blocks():
    assert parse_detection_response('{"page": 1, "blocks": []}', 800, 600) == []


def test_detection_schema_shape():
    props = DETECTION_SCHEMA["properties"]["blocks"]["items"]["properties"]
    assert set(props) == {"label", "box_2d", "marks", "text"}
    assert props["box_2d"]["minItems"] == 4
    assert props["box_2d"]["maxItems"] == 4


def test_prompt_contains_key_instructions():
    prompt = DETECTION_PROMPT.lower()
    assert "label" in prompt
    assert "extract all question/sub-part labels" in prompt
    assert "ignore non-question text" in prompt
    assert "cover page" in prompt
    assert "box_2d" in DETECTION_PROMPT


def test_encode_image_base64(tmp_path):
    img_path = tmp_path / "test.png"
    Image.new("RGB", (100, 50), color="white").save(img_path)
    detector = Detector(base_url="http://fake", api_key="fake", model="fake")
    b64 = detector._encode_image(img_path)
    assert isinstance(b64, str)
    assert len(b64) > 0
