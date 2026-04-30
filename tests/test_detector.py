from src.inference.detector import DefectDetector


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeBox:
    cls = [Scalar(5)]
    conf = [Scalar(0.9123456)]
    xyxy = [[Scalar(120.4567), Scalar(80.3333), Scalar(230.789), Scalar(118.944)]]


class FakeResult:
    boxes = [FakeBox()]


class FakeModel:
    names = {5: "scratches"}


def test_parse_results_returns_api_shape() -> None:
    detector = DefectDetector.__new__(DefectDetector)
    detector.model = FakeModel()
    detector.class_names = None

    detections = detector.parse_results([FakeResult()])

    assert detections == [
        {
            "class_id": 5,
            "class_name": "scratches",
            "confidence": 0.912346,
            "bbox": {"x1": 120.457, "y1": 80.333, "x2": 230.789, "y2": 118.944},
        }
    ]
