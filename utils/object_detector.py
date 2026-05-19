from __future__ import annotations

from math import hypot
from typing import Any

import cv2
import numpy as np


SAFE_LABELS = {"pillow", "teddy bear"}
WARNING_LABELS = {
    "backpack",
    "blanket",
    "book",
    "bowl",
    "chair",
    "cup",
    "handbag",
    "remote",
    "suitcase",
    "teddy bear",
    "vase",
}
DANGEROUS_LABELS = {
    "bottle",
    "fork",
    "hair drier",
    "keyboard",
    "knife",
    "laptop",
    "mouse",
    "scissors",
    "sports ball",
    "toaster",
}


class SurroundingsObjectDetector:
    """YOLOv8 object detection activated only after a fence breach."""

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.28):
        self.confidence = confidence
        self.model = None
        self.names: dict[int, str] = {}
        self.load_error = None
        try:
            from ultralytics import YOLO

            self.model = YOLO(model_name)
            self.names = self.model.names
        except Exception as exc:  # pragma: no cover - depends on local model setup
            self.load_error = str(exc)

    def detect_nearby(
        self,
        frame: Any,
        infant_bbox: list[int],
        max_distance: int = 360,
        include_unknown: bool = False,
    ) -> list[dict[str, Any]]:
        frame_height, frame_width = frame.shape[:2]
        proximity_bbox = self._infant_proximity_region(infant_bbox, frame_width, frame_height)
        objects: list[dict[str, Any]] = []
        if self.model is not None:
            results = self.model.predict(frame, conf=self.confidence, verbose=False)
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    label = self.names.get(class_id, str(class_id))
                    if label == "person":
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                    bbox = [int(x1), int(y1), int(x2), int(y2)]
                    distance = self._bbox_distance(proximity_bbox, bbox)
                    overlap = self._overlaps(proximity_bbox, bbox)
                    if distance > max_distance and not overlap:
                        continue
                    category = self._category(label)
                    objects.append(
                        {
                            "bbox": bbox,
                            "label": label,
                            "category": category,
                            "confidence": round(float(box.conf[0]), 3),
                            "distance": round(distance, 1),
                            "overlapsInfant": overlap,
                            "contact": overlap or distance <= self._contact_distance(category),
                        }
                    )
        if include_unknown:
            objects.extend(
                self._detect_unknown_objects(
                    frame,
                    infant_bbox,
                    proximity_bbox,
                    objects,
                    max_distance,
                )
            )
        return objects

    @staticmethod
    def _category(label: str) -> str:
        if label in DANGEROUS_LABELS:
            return "Dangerous"
        if label in WARNING_LABELS:
            return "Warning"
        if label in SAFE_LABELS:
            return "Safe"
        return "Warning"

    @staticmethod
    def _bbox_distance(a: list[int], b: list[int]) -> float:
        ax = 0 if a[0] <= b[2] and b[0] <= a[2] else min(abs(a[0] - b[2]), abs(b[0] - a[2]))
        ay = 0 if a[1] <= b[3] and b[1] <= a[3] else min(abs(a[1] - b[3]), abs(b[1] - a[3]))
        return hypot(ax, ay)

    @staticmethod
    def _overlaps(a: list[int], b: list[int]) -> bool:
        return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]

    @staticmethod
    def _contact_distance(category: str) -> int:
        if category == "Dangerous":
            return 45
        if category == "Warning":
            return 35
        return 25

    def _detect_unknown_objects(
        self,
        frame: Any,
        infant_bbox: list[int],
        proximity_bbox: list[int],
        existing_objects: list[dict[str, Any]],
        max_distance: int,
    ) -> list[dict[str, Any]]:
        """Detect nearby non-COCO objects, such as a mango, as warning objects.

        This is intentionally generic. It does not claim a custom class; it only
        says an unclassified physical object is close enough to matter.
        """
        height, width = frame.shape[:2]
        search = self._expanded_region(proximity_bbox, width, height, max_distance)
        sx1, sy1, sx2, sy2 = search
        roi = frame[sy1:sy2, sx1:sx2]
        if roi.size == 0:
            return []

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Broadly capture colored physical objects under weak webcam lighting.
        saturated = cv2.inRange(hsv, np.array([5, 22, 25]), np.array([115, 255, 255]))
        # Catch dark objects too: phones, remotes, black toys, small bags, wires.
        dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 130, 110]))

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 35, 110)
        object_mask = cv2.bitwise_or(cv2.bitwise_or(saturated, dark), edges)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        unknowns: list[dict[str, Any]] = []
        frame_area = width * height
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < frame_area * 0.0012 or area > frame_area * 0.30:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w < 25 or h < 25:
                continue
            bbox = [sx1 + x, sy1 + y, sx1 + x + w, sy1 + y + h]
            if self._covered_by_existing(bbox, existing_objects):
                continue

            distance = self._bbox_distance(proximity_bbox, bbox)
            overlap = self._overlaps(proximity_bbox, bbox)
            if distance > 260 and not overlap:
                continue

            unknowns.append(
                {
                    "bbox": bbox,
                    "label": "unknown object",
                    "category": "Warning",
                    "confidence": 0.5,
                    "distance": round(distance, 1),
                    "overlapsInfant": overlap,
                    "contact": overlap or distance <= self._contact_distance("Warning"),
                    "source": "opencv-fallback",
                }
            )

        if unknowns:
            return self._dedupe_unknowns(unknowns)

        return []

    @staticmethod
    def _expanded_region(
        bbox: list[int], frame_width: int, frame_height: int, padding: int
    ) -> list[int]:
        x1, y1, x2, y2 = bbox
        return [
            max(0, x1 - padding),
            max(0, y1 - padding),
            min(frame_width, x2 + padding),
            min(frame_height, y2 + padding),
        ]

    @staticmethod
    def _infant_proximity_region(
        bbox: list[int], frame_width: int, frame_height: int
    ) -> list[int]:
        x1, y1, x2, y2 = bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        pad_x = int(width * 0.85)
        pad_top = int(height * 2.0)
        pad_bottom = int(height * 0.65)
        return [
            max(0, x1 - pad_x),
            max(0, y1 - pad_top),
            min(frame_width, x2 + pad_x),
            min(frame_height, y2 + pad_bottom),
        ]

    @staticmethod
    def _covered_by_existing(bbox: list[int], objects: list[dict[str, Any]]) -> bool:
        return any(SurroundingsObjectDetector._iou(bbox, obj["bbox"]) > 0.35 for obj in objects)

    @staticmethod
    def _dedupe_unknowns(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        for obj in sorted(objects, key=lambda item: item["distance"]):
            if all(SurroundingsObjectDetector._iou(obj["bbox"], other["bbox"]) < 0.25 for other in kept):
                kept.append(obj)
        return kept[:4]

    @staticmethod
    def _iou(a: list[int], b: list[int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter_area / (area_a + area_b - inter_area)

    @staticmethod
    def _fallback_blob_from_infant_region(frame: Any, infant_bbox: list[int]) -> list[dict[str, Any]]:
        x1, y1, x2, y2 = infant_bbox
        width = x2 - x1
        height = y2 - y1
        if width < 30 or height < 30:
            return []

        frame_height, frame_width = frame.shape[:2]
        expanded = [
            max(0, x1 - int(width * 0.2)),
            max(0, y1 - int(height * 0.2)),
            min(frame_width, x2 + int(width * 0.2)),
            min(frame_height, y2 + int(height * 0.2)),
        ]
        return [
            {
                "bbox": expanded,
                "label": "unknown object",
                "category": "Warning",
                "confidence": 0.35,
                "distance": 0,
                "overlapsInfant": True,
                "contact": True,
                "source": "opencv-fallback-region",
            }
        ]
