from __future__ import annotations

from typing import Any

import cv2


class PoseEstimator:
    """MediaPipe Pose wrapper scoped to the infant bbox."""

    def __init__(self) -> None:
        self.pose = None
        self.load_error = None
        try:
            import mediapipe as mp

            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                enable_segmentation=False,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
            )
        except Exception as exc:  # pragma: no cover - depends on local install
            self.load_error = str(exc)

    def estimate(self, frame: Any, bbox: list[int]) -> dict[str, Any] | None:
        if self.pose is None:
            return None

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = self._expand_bbox(bbox, width, height)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        if not result.pose_landmarks:
            return None

        landmarks = {}
        for name, idx in self._landmark_indices().items():
            landmark = result.pose_landmarks.landmark[idx]
            landmarks[name] = {
                "x": int(x1 + landmark.x * (x2 - x1)),
                "y": int(y1 + landmark.y * (y2 - y1)),
                "visibility": round(float(landmark.visibility), 3),
            }
        return {"landmarks": landmarks}

    @staticmethod
    def _expand_bbox(bbox: list[int], width: int, height: int) -> list[int]:
        x1, y1, x2, y2 = bbox
        pad_x = int((x2 - x1) * 0.18)
        pad_y = int((y2 - y1) * 0.18)
        return [
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(width, x2 + pad_x),
            min(height, y2 + pad_y),
        ]

    @staticmethod
    def _landmark_indices() -> dict[str, int]:
        return {
            "nose": 0,
            "leftEye": 2,
            "rightEye": 5,
            "leftEar": 7,
            "rightEar": 8,
            "leftShoulder": 11,
            "rightShoulder": 12,
            "leftElbow": 13,
            "rightElbow": 14,
            "leftWrist": 15,
            "rightWrist": 16,
            "leftHip": 23,
            "rightHip": 24,
            "leftKnee": 25,
            "rightKnee": 26,
            "leftAnkle": 27,
            "rightAnkle": 28,
        }
