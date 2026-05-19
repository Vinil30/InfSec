from __future__ import annotations

from math import hypot
from typing import Any

import cv2
import numpy as np


def bbox_center(bbox: list[int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def bbox_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter_area / (area_a + area_b - inter_area)


class InfantTracker:
    """Selected-infant tracker backed by Ultralytics ByteTrack."""

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.35) -> None:
        self.confidence = confidence
        self.model = None
        self.load_error = None
        try:
            from ultralytics import YOLO

            self.model = YOLO(model_name)
        except Exception as exc:  # pragma: no cover - depends on local model setup
            self.load_error = str(exc)
        self.selected_bbox: list[int] | None = None
        self.track_id: int | None = None
        self.manual_mode = False
        self.cv_tracker = None
        self.missed_frames = 0
        self.reinit_interval = 0

    def reset(self) -> None:
        self.selected_bbox = None
        self.track_id = None
        self.manual_mode = False
        self.cv_tracker = None
        self.missed_frames = 0
        self.reinit_interval = 0

    def update(
        self,
        frame: Any,
        detections: list[dict[str, Any]],
        selection: dict[str, float] | None,
    ) -> dict[str, Any] | None:
        if self.model is None:
            return self._fallback_update(frame, detections, selection)

        tracks = self._bytetrack(frame)
        face_reacquired = self._force_face_reacquire_if_drifted(frame)
        if face_reacquired:
            return face_reacquired

        if selection and self.track_id is None:
            selected = self._pick_selected_track(tracks, selection)
            if selected:
                self.track_id = selected.get("trackId")
                self.selected_bbox = selected["bbox"]
                self.manual_mode = False
                self._init_cv_tracker(frame, self.selected_bbox)
            else:
                selected_bbox = self._pick_selected_detection(detections, selection)
                if selected_bbox is None:
                    selected_bbox = self._face_bbox_near_selection(frame, selection)
                if selected_bbox is None:
                    selected_bbox = self._manual_bbox_from_selection(frame, selection)
                if selected_bbox:
                    self.selected_bbox = selected_bbox
                    self.track_id = self._nearest_track_id(tracks, selected_bbox) or -1
                    self.manual_mode = self.track_id == -1
                    self._init_cv_tracker(frame, self.selected_bbox)

        if self.track_id is None and self.selected_bbox:
            matched_track = self._match_bbox_to_track(tracks, self.selected_bbox)
            if matched_track:
                self.track_id = matched_track["trackId"]
                self.manual_mode = False

        if self.manual_mode and self.selected_bbox:
            matched_track = self._match_bbox_to_track(tracks, self.selected_bbox)
            if matched_track:
                self.track_id = matched_track["trackId"]
                self.manual_mode = False
                self.selected_bbox = self._smooth_bbox(self.selected_bbox, matched_track["bbox"], 0.55)
                self._init_cv_tracker(frame, self.selected_bbox)
            else:
                return self._fallback_update(frame, detections, None)

        if self.track_id is None and self.selected_bbox:
            return self._fallback_update(frame, detections, None)

        if self.track_id is None:
            return None

        match = next((track for track in tracks if track.get("trackId") == self.track_id), None)
        if match is None:
            self.missed_frames += 1
            if self.missed_frames > 15:
                self.reset()
            return self._fallback_update(frame, detections, None)

        cv_bbox = self._update_cv_tracker(frame)
        source_bbox = self._blend_tracker_sources(match["bbox"], cv_bbox)
        self.selected_bbox = self._smooth_bbox(self.selected_bbox, source_bbox, 0.45)
        self._maybe_reinit_cv_tracker(frame)
        self.missed_frames = 0
        return {**match, "bbox": self.selected_bbox, "selected": True, "missedFrames": self.missed_frames}

    def status(self) -> dict[str, Any]:
        return {
            "active": self.selected_bbox is not None,
            "trackId": self.track_id,
            "missedFrames": self.missed_frames,
            "backend": "ByteTrack" if self.model is not None else "IoU fallback",
            "manualMode": self.manual_mode,
        }

    def _bytetrack(self, frame: Any) -> list[dict[str, Any]]:
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=self.confidence,
            verbose=False,
        )
        tracks: list[dict[str, Any]] = []
        for result in results:
            for box in result.boxes:
                if box.id is None:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                tracks.append(
                    {
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": round(float(box.conf[0]), 3),
                        "classId": 0,
                        "label": "person",
                        "trackId": int(box.id[0]),
                    }
                )
        return tracks

    def _fallback_update(
        self,
        frame: Any,
        detections: list[dict[str, Any]],
        selection: dict[str, float] | None,
    ) -> dict[str, Any] | None:
        if selection and not self.selected_bbox:
            self.selected_bbox = self._pick_selected_detection(detections, selection)
            if self.selected_bbox is None:
                self.selected_bbox = self._face_bbox_near_selection(frame, selection)
            if self.selected_bbox is None:
                self.selected_bbox = self._manual_bbox_from_selection(frame, selection)
            self.track_id = self.track_id or -1 if self.selected_bbox else None
            self.manual_mode = self.selected_bbox is not None
            if self.selected_bbox:
                self._init_cv_tracker(frame, self.selected_bbox)

        if not self.selected_bbox:
            return None

        face_reacquired = self._force_face_reacquire_if_drifted(frame)
        if face_reacquired:
            return face_reacquired

        match = self._match_existing_target(detections)
        if match is None:
            tracked_bbox = self._update_cv_tracker(frame)
            if tracked_bbox and self._is_valid_tracker_bbox(frame, tracked_bbox):
                self.selected_bbox = self._smooth_bbox(self.selected_bbox, tracked_bbox, 0.42)
                self.missed_frames = 0
            else:
                reacquired = self._reacquire_from_frame(frame, detections)
                if reacquired:
                    self.selected_bbox = reacquired["bbox"]
                    self._init_cv_tracker(frame, self.selected_bbox)
                    self.missed_frames = 0
                    return {
                        **reacquired,
                        "trackId": self.track_id,
                        "selected": True,
                        "missedFrames": self.missed_frames,
                        "source": "reacquired",
                    }
                self.missed_frames += 1
            return {
                "bbox": self.selected_bbox,
                "confidence": 0.5,
                "classId": 0,
                "label": "selected region",
                "trackId": self.track_id,
                "selected": True,
                "missedFrames": self.missed_frames,
                "source": "manual-click",
            }

        cv_bbox = self._update_cv_tracker(frame)
        source_bbox = self._blend_tracker_sources(match["bbox"], cv_bbox)
        self.selected_bbox = self._smooth_bbox(self.selected_bbox, source_bbox, 0.42)
        self._maybe_reinit_cv_tracker(frame)
        self.missed_frames = 0
        return {
            **match,
            "bbox": self.selected_bbox,
            "trackId": self.track_id,
            "selected": True,
            "missedFrames": self.missed_frames,
        }

    @staticmethod
    def _manual_bbox_from_selection(frame: Any, selection: dict[str, float]) -> list[int]:
        height, width = frame.shape[:2]
        click_x = int(max(0, min(width - 1, selection["x"])))
        click_y = int(max(0, min(height - 1, selection["y"])))
        box_width = int(max(70, min(width * 0.18, width * 0.12)))
        box_height = int(max(90, min(height * 0.28, height * 0.18)))
        x1 = max(0, click_x - box_width // 2)
        y1 = max(0, click_y - box_height // 2)
        x2 = min(width, x1 + box_width)
        y2 = min(height, y1 + box_height)
        if x2 - x1 < box_width:
            x1 = max(0, x2 - box_width)
        if y2 - y1 < box_height:
            y1 = max(0, y2 - box_height)
        return [int(x1), int(y1), int(x2), int(y2)]

    def _init_cv_tracker(self, frame: Any, bbox: list[int]) -> None:
        tracker = self._create_cv_tracker()
        if tracker is None:
            self.cv_tracker = None
            return
        x1, y1, x2, y2 = bbox
        tracker.init(frame, (x1, y1, max(1, x2 - x1), max(1, y2 - y1)))
        self.cv_tracker = tracker

    def _update_cv_tracker(self, frame: Any) -> list[int] | None:
        if self.cv_tracker is None:
            return None
        ok, bbox = self.cv_tracker.update(frame)
        if not ok:
            return None
        x, y, width, height = bbox
        frame_height, frame_width = frame.shape[:2]
        x1 = int(max(0, min(frame_width - 1, x)))
        y1 = int(max(0, min(frame_height - 1, y)))
        x2 = int(max(0, min(frame_width, x + width)))
        y2 = int(max(0, min(frame_height, y + height)))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    def _is_valid_tracker_bbox(self, frame: Any, bbox: list[int]) -> bool:
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        if width < 35 or height < 45:
            return False
        if x1 <= 2 or y1 <= 2 or x2 >= frame_width - 2 or y2 >= frame_height - 2:
            return False
        if self.selected_bbox:
            prev_center = bbox_center(self.selected_bbox)
            current_center = bbox_center(bbox)
            jump = hypot(current_center[0] - prev_center[0], current_center[1] - prev_center[1])
            max_jump = max(frame_width, frame_height) * 0.18
            if jump > max_jump:
                return False

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if float(np.std(gray)) < 8:
            return False
        return True

    def _reacquire_from_frame(
        self, frame: Any, detections: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        face_bbox = self._detect_face(frame)
        candidates = []
        if face_bbox:
            candidates.append(
                {
                    "bbox": self._expand_face_to_target(frame, face_bbox),
                    "confidence": 0.62,
                    "classId": 0,
                    "label": "face reacquired",
                }
            )
        candidates.extend(detections)
        if not candidates:
            return None

        frame_height, frame_width = frame.shape[:2]
        frame_center = (frame_width / 2, frame_height / 2)

        def candidate_score(det: dict[str, Any]) -> float:
            center = bbox_center(det["bbox"])
            score = float(det.get("confidence", 0.4))
            if self.selected_bbox:
                old_center = bbox_center(self.selected_bbox)
                score -= hypot(center[0] - old_center[0], center[1] - old_center[1]) / max(frame_width, frame_height)
            else:
                score -= hypot(center[0] - frame_center[0], center[1] - frame_center[1]) / max(frame_width, frame_height)
            return score

        best = max(candidates, key=candidate_score)
        return best if candidate_score(best) > -0.25 else None

    def _force_face_reacquire_if_drifted(self, frame: Any) -> dict[str, Any] | None:
        if not self.selected_bbox:
            return None
        face_bbox = self._detect_face(frame)
        if not face_bbox:
            return None

        face_target = self._expand_face_to_target(frame, face_bbox)
        current_center = bbox_center(self.selected_bbox)
        face_center = bbox_center(face_target)
        face_width = max(1, face_target[2] - face_target[0])
        face_height = max(1, face_target[3] - face_target[1])
        distance = hypot(current_center[0] - face_center[0], current_center[1] - face_center[1])
        overlap = bbox_iou(self.selected_bbox, face_target)

        if overlap >= 0.08 or distance <= max(face_width, face_height) * 0.65:
            return None
        if distance <= max(face_width, face_height) * 1.2 and self.missed_frames < 2:
            return None

        self.selected_bbox = face_target
        self.track_id = -1 if self.track_id is None else self.track_id
        self.manual_mode = True
        self.missed_frames = 0
        self._init_cv_tracker(frame, self.selected_bbox)
        return {
            "bbox": self.selected_bbox,
            "confidence": 0.68,
            "classId": 0,
            "label": "face reacquired",
            "trackId": self.track_id,
            "selected": True,
            "missedFrames": self.missed_frames,
            "source": "face-reacquired",
        }

    @staticmethod
    def _detect_face(frame: Any) -> list[int] | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(45, 45))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        return [int(x), int(y), int(x + w), int(y + h)]

    def _face_bbox_near_selection(self, frame: Any, selection: dict[str, float]) -> list[int] | None:
        face_bbox = self._detect_face(frame)
        if not face_bbox:
            return None
        click = (selection["x"], selection["y"])
        face_center = bbox_center(face_bbox)
        face_width = face_bbox[2] - face_bbox[0]
        face_height = face_bbox[3] - face_bbox[1]
        distance = hypot(click[0] - face_center[0], click[1] - face_center[1])
        if distance > max(face_width, face_height) * 1.8:
            return None
        return self._expand_face_to_target(frame, face_bbox)

    @staticmethod
    def _expand_face_to_target(frame: Any, face_bbox: list[int]) -> list[int]:
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = face_bbox
        width = x2 - x1
        height = y2 - y1
        pad_x = int(width * 0.45)
        pad_top = int(height * 0.30)
        pad_bottom = int(height * 1.10)
        return [
            max(0, x1 - pad_x),
            max(0, y1 - pad_top),
            min(frame_width, x2 + pad_x),
            min(frame_height, y2 + pad_bottom),
        ]

    def _maybe_reinit_cv_tracker(self, frame: Any) -> None:
        self.reinit_interval += 1
        if self.reinit_interval >= 8 and self.selected_bbox:
            self._init_cv_tracker(frame, self.selected_bbox)
            self.reinit_interval = 0

    @staticmethod
    def _blend_tracker_sources(det_bbox: list[int], cv_bbox: list[int] | None) -> list[int]:
        if cv_bbox is None:
            return det_bbox
        if bbox_iou(det_bbox, cv_bbox) >= 0.12:
            return InfantTracker._smooth_bbox(cv_bbox, det_bbox, 0.38)
        return cv_bbox

    @staticmethod
    def _smooth_bbox(previous: list[int] | None, current: list[int], alpha: float) -> list[int]:
        if previous is None:
            return current
        return [
            int(previous[idx] * (1 - alpha) + current[idx] * alpha)
            for idx in range(4)
        ]

    @staticmethod
    def _create_cv_tracker() -> Any:
        creators = [
            lambda: cv2.TrackerCSRT_create(),
            lambda: cv2.legacy.TrackerCSRT_create(),
            lambda: cv2.TrackerKCF_create(),
            lambda: cv2.legacy.TrackerKCF_create(),
            lambda: cv2.TrackerMIL_create(),
            lambda: cv2.legacy.TrackerMIL_create(),
        ]
        for creator in creators:
            try:
                return creator()
            except Exception:
                continue
        return None

    def _pick_selected_track(
        self, tracks: list[dict[str, Any]], selection: dict[str, float]
    ) -> dict[str, Any] | None:
        click_x = selection["x"]
        click_y = selection["y"]
        containing = [
            track for track in tracks if self._point_inside_bbox(click_x, click_y, track["bbox"])
        ]
        if containing:
            return max(containing, key=lambda track: track["confidence"])

        return None

    def _nearest_track_id(self, tracks: list[dict[str, Any]], bbox: list[int]) -> int | None:
        matched_track = self._match_bbox_to_track(tracks, bbox)
        return matched_track["trackId"] if matched_track else None

    @staticmethod
    def _match_bbox_to_track(
        tracks: list[dict[str, Any]], bbox: list[int]
    ) -> dict[str, Any] | None:
        if not tracks:
            return None
        best = max(tracks, key=lambda track: bbox_iou(track["bbox"], bbox))
        return best if bbox_iou(best["bbox"], bbox) >= 0.25 else None

    def _pick_selected_detection(
        self, detections: list[dict[str, Any]], selection: dict[str, float]
    ) -> list[int] | None:
        click_x = selection["x"]
        click_y = selection["y"]
        containing = [
            det for det in detections if self._point_inside_bbox(click_x, click_y, det["bbox"])
        ]
        if containing:
            return max(containing, key=lambda det: det["confidence"])["bbox"]

        return None

    def _match_existing_target(self, detections: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not detections or not self.selected_bbox:
            return None

        previous_center = bbox_center(self.selected_bbox)

        def score(det: dict[str, Any]) -> float:
            cx, cy = bbox_center(det["bbox"])
            distance_penalty = hypot(cx - previous_center[0], cy - previous_center[1]) / 1000
            return bbox_iou(self.selected_bbox, det["bbox"]) - distance_penalty

        best = max(detections, key=score)
        best_iou = bbox_iou(self.selected_bbox, best["bbox"])
        prev_center = bbox_center(self.selected_bbox)
        best_center = bbox_center(best["bbox"])
        center_distance = hypot(best_center[0] - prev_center[0], best_center[1] - prev_center[1])
        selected_width = max(1, self.selected_bbox[2] - self.selected_bbox[0])
        selected_height = max(1, self.selected_bbox[3] - self.selected_bbox[1])
        max_reasonable_jump = max(selected_width, selected_height) * 0.75
        if best_iou >= 0.18 or center_distance <= max_reasonable_jump:
            return best
        return None

    @staticmethod
    def _point_inside_bbox(x: float, y: float, bbox: list[int]) -> bool:
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2
