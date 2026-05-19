from __future__ import annotations

from math import hypot
import time
from typing import Any


class RiskAnalyzer:
    """Rule-based infant safety risk scoring."""

    def __init__(self) -> None:
        self.outside_since: float | None = None
        self.previous_nose: dict[str, Any] | None = None
        self.previous_face_time: float | None = None

    def reset(self) -> None:
        self.outside_since = None
        self.previous_nose = None
        self.previous_face_time = None

    def reset_fence_timer(self) -> None:
        self.outside_since = None

    def update_fence_state(self, inside_fence: bool) -> bool:
        if inside_fence:
            self.outside_since = None
            return False
        if self.outside_since is None:
            self.outside_since = time.monotonic()
        return True

    def analyze(
        self,
        infant: dict[str, Any] | None,
        inside_fence: bool,
        fence_breached: bool,
        pose: dict[str, Any] | None,
        objects: list[dict[str, Any]],
    ) -> dict[str, Any]:
        score = 0
        reasons: list[str] = []

        if infant is None:
            return {"level": "LOW", "score": 0, "reasons": ["No infant selected or tracked"]}

        if not inside_fence:
            score += 25
            reasons.append("Infant crossed the geofence")
            outside_duration = self._outside_duration()
            if outside_duration >= 4:
                score += 25
                reasons.append("Infant outside safe area for prolonged duration")

        if pose:
            posture_score, posture_reasons = self._pose_risk(pose)
            score += posture_score
            reasons.extend(posture_reasons)

            face_score, face_reasons = self._face_movement_risk(pose)
            score += face_score
            reasons.extend(face_reasons)

        object_score, object_reasons, force_high = self._object_risk(objects, pose)
        if force_high:
            return {
                "level": "HIGH",
                "score": 100,
                "reasons": object_reasons + reasons,
            }
        score += object_score
        reasons.extend(object_reasons)

        level = "LOW"
        if score >= 70:
            level = "HIGH"
        elif score >= 35:
            level = "MEDIUM"

        if not reasons:
            reasons.append("Infant appears within safe monitoring bounds")

        return {"level": level, "score": min(100, score), "reasons": reasons}

    def _outside_duration(self) -> float:
        if self.outside_since is None:
            return 0.0
        return time.monotonic() - self.outside_since

    def _pose_risk(self, pose: dict[str, Any]) -> tuple[int, list[str]]:
        landmarks = pose.get("landmarks", {})
        reasons: list[str] = []
        score = 0

        nose = landmarks.get("nose")
        left_eye = landmarks.get("leftEye")
        right_eye = landmarks.get("rightEye")
        left_ear = landmarks.get("leftEar")
        right_ear = landmarks.get("rightEar")
        left_shoulder = landmarks.get("leftShoulder")
        right_shoulder = landmarks.get("rightShoulder")
        left_hip = landmarks.get("leftHip")
        right_hip = landmarks.get("rightHip")

        if nose and left_shoulder and right_shoulder:
            shoulder_y = (left_shoulder["y"] + right_shoulder["y"]) / 2
            if nose["y"] > shoulder_y + 18:
                score += 30
                reasons.append("Head appears below torso line")

            shoulder_center_x = (left_shoulder["x"] + right_shoulder["x"]) / 2
            shoulder_width = abs(left_shoulder["x"] - right_shoulder["x"])
            if shoulder_width > 20 and abs(nose["x"] - shoulder_center_x) > shoulder_width * 0.55:
                score += 12
                reasons.append("Face turned strongly away from body center")

        visible_face_points = [
            point
            for point in [nose, left_eye, right_eye, left_ear, right_ear]
            if point and point.get("visibility", 0) >= 0.35
        ]
        if nose and len(visible_face_points) <= 2:
            score += 10
            reasons.append("Limited face visibility detected")

        if left_shoulder and right_shoulder and left_hip and right_hip:
            torso_y = (left_shoulder["y"] + right_shoulder["y"]) / 2
            hip_y = (left_hip["y"] + right_hip["y"]) / 2
            if abs(hip_y - torso_y) < 24:
                score += 20
                reasons.append("Sudden fall-like horizontal posture detected")

        return score, reasons

    def _face_movement_risk(self, pose: dict[str, Any]) -> tuple[int, list[str]]:
        nose = pose.get("landmarks", {}).get("nose")
        now = time.monotonic()
        if not nose or nose.get("visibility", 0) < 0.35:
            return 0, []

        score = 0
        reasons: list[str] = []
        if self.previous_nose and self.previous_face_time:
            elapsed = max(0.05, now - self.previous_face_time)
            movement = hypot(nose["x"] - self.previous_nose["x"], nose["y"] - self.previous_nose["y"])
            speed = movement / elapsed
            if speed > 900:
                score += 12
                reasons.append("Abrupt infant face movement detected")

        self.previous_nose = nose
        self.previous_face_time = now
        return score, reasons

    def _object_risk(
        self, objects: list[dict[str, Any]], pose: dict[str, Any] | None
    ) -> tuple[int, list[str], bool]:
        reasons: list[str] = []
        score = 0
        wrists = self._visible_wrists(pose)
        for obj in objects:
            label = obj["label"]
            category = obj["category"]
            hand_distance = self._nearest_hand_distance(wrists, obj["bbox"])
            hand_contact = hand_distance is not None and hand_distance <= 40

            if category == "Dangerous" and (obj["overlapsInfant"] or hand_contact or obj.get("contact")):
                return 0, [f"Infant may be catching dangerous object: {label}"], True

            if obj["distance"] <= 160 or hand_contact:
                score += 38
                reasons.append(f"Infant reached nearby object: {label}")
            elif obj["distance"] <= 260:
                score += 22
                reasons.append(f"Object near infant: {label}")

            if category == "Dangerous" and obj["distance"] <= 260:
                score += 20
                reasons.append(f"Dangerous object within reach: {label}")

        return min(score, 65), reasons, False

    @staticmethod
    def _visible_wrists(pose: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not pose:
            return []
        landmarks = pose.get("landmarks", {})
        return [
            point
            for point in [landmarks.get("leftWrist"), landmarks.get("rightWrist")]
            if point and point.get("visibility", 0) >= 0.25
        ]

    @staticmethod
    def _nearest_hand_distance(wrists: list[dict[str, Any]], bbox: list[int]) -> float | None:
        distances = []
        for wrist in wrists:
            distances.append(RiskAnalyzer._point_bbox_distance(wrist["x"], wrist["y"], bbox))
        return min(distances) if distances else None

    @staticmethod
    def _point_bbox_distance(x: float, y: float, bbox: list[int]) -> float:
        x1, y1, x2, y2 = bbox
        dx = 0 if x1 <= x <= x2 else min(abs(x - x1), abs(x - x2))
        dy = 0 if y1 <= y <= y2 else min(abs(y - y1), abs(y - y2))
        return hypot(dx, dy)
