from src.models import Bbox, QuestionBbox, PaperMeta, QuestionResult, ExamResult, parse_filename, build_index


def test_bbox_creation():
    b = Bbox(x=10, y=20, w=100, h=200)
    assert b.x == 10
    assert b.y == 20
    assert b.w == 100
    assert b.h == 200


def test_bbox_clamp_within_bounds():
    b = Bbox(x=-10, y=-5, w=200, h=200)
    clamped = b.clamp(100, 100)
    assert clamped.x == 0
    assert clamped.y == 0
    assert clamped.x + clamped.w <= 100


def test_bbox_is_valid():
    assert Bbox(0, 0, 50, 50).is_valid(100, 100) is True
    assert Bbox(0, 0, 0, 50).is_valid(100, 100) is False
    assert Bbox(-5, 0, 50, 50).is_valid(100, 100) is False


def test_question_bbox_defaults():
    b = Bbox(x=0, y=0, w=50, h=60)
    q = QuestionBbox(question_number="3", bbox=b)
    assert q.question_number == "3"
    assert q.marks is None
    assert q.continued is False
    assert q.continued_from is False


def test_parse_official_exam1():
    meta = parse_filename("2024-vcaa-exam-1.pdf")
    assert meta is not None
    assert meta.year == 2024
    assert meta.source == "vcaa"
    assert meta.exam_type == "exam1"
    assert meta.key == "2024-vcaa-exam1"
    assert meta.is_official is True


def test_parse_trial_exam2():
    meta = parse_filename("2025-heffernan-exam-2.pdf")
    assert meta is not None
    assert meta.year == 2025
    assert meta.source == "heffernan"
    assert meta.exam_type == "exam2"
    assert meta.key == "2025-heffernan-exam2"
    assert meta.is_official is False


def test_parse_invalid_filenames():
    assert parse_filename("random-file.pdf") is None
    assert parse_filename("exam-1.pdf") is None
    assert parse_filename("2024-vcaa-exam-3.pdf") is None


def test_parse_filename_with_path():
    meta = parse_filename("/some/deep/path/2023-vcaa-exam-1.pdf")
    assert meta is not None
    assert meta.year == 2023
    assert meta.key == "2023-vcaa-exam1"


def test_exam_result_to_mapping():
    q = QuestionResult(
        number="1", image="q01.png", marks=5, pages=[1],
        has_subquestions=True, cross_page=False,
    )
    result = ExamResult(
        key="2024-vcaa-exam1", year=2024, source="vcaa",
        exam_type="exam1", questions=[q],
        path="/data/2024-vcaa-exam-1.pdf",
    )
    mapping = result.to_mapping()
    assert mapping["year"] == 2024
    assert mapping["source"] == "vcaa"
    assert mapping["total_questions"] == 1
    assert len(mapping["questions"]) == 1
    assert mapping["questions"][0]["number"] == "1"
    assert mapping["questions"][0]["image"] == "q01.png"
    assert mapping["questions"][0]["marks"] == 5
    assert mapping["questions"][0]["pages"] == [1]
    assert mapping["questions"][0]["has_subquestions"] is True
    assert mapping["questions"][0]["cross_page"] is False


def test_build_index():
    q1 = QuestionResult(
        number="1", image="q01.png", marks=5, pages=[1],
        has_subquestions=False, cross_page=False,
    )
    e1 = ExamResult(
        key="2024-vcaa-exam1", year=2024, source="vcaa",
        exam_type="exam1", questions=[q1],
        path="/data/2024-vcaa-exam-1.pdf",
    )
    e2 = ExamResult(
        key="2023-vcaa-exam1", year=2023, source="vcaa",
        exam_type="exam1", questions=[],
        path="/data/2023-vcaa-exam-1.pdf",
    )

    index = build_index([e1, e2])
    assert index["total_exams"] == 2
    assert index["total_questions"] == 1
    assert len(index["exams"]) == 2
    assert index["exams"][0]["key"] == "2023-vcaa-exam1"  # sorted by year
    assert index["exams"][1]["key"] == "2024-vcaa-exam1"
    assert index["exams"][1]["input_path"] == "/data/2024-vcaa-exam-1.pdf"


def test_build_index_empty():
    index = build_index([])
    assert index["total_exams"] == 0
    assert index["total_questions"] == 0
    assert index["exams"] == []
