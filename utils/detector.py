from __future__ import annotations

from typing import Any


class InfantDetector:
    """YOLOv8 person detector using pretrained COCO weights."""

    PERSON_CLASS_ID = 0

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.35):
        self.confidence = confidence
        self.model = None
        self.load_error = None
        try:
            from ultralytics import YOLO

            self.model = YOLO(model_name)
        except Exception as exc:  # pragma: no cover - depends on local model setup
            self.load_error = str(exc)

    def detect_people(self, frame: Any) -> list[dict[str, Any]]:
        if self.model is None:
            return []

        results = self.model.predict(
            frame,
            classes=[self.PERSON_CLASS_ID],
            conf=self.confidence,
            verbose=False,
        )
        detections: list[dict[str, Any]] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                detections.append(
                    {
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": round(float(box.conf[0]), 3),
                        "classId": self.PERSON_CLASS_ID,
                        "label": "person",
                    }
                )
        return detections
