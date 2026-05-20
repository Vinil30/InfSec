from __future__ import annotations

import base64
import os
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from utils.alert import AlertManager
from utils.detector import InfantDetector
from utils.geofence import Geofence
from utils.object_detector import SurroundingsObjectDetector
from utils.pose_estimator import PoseEstimator
from utils.risk_analyzer import RiskAnalyzer
from utils.tracker import InfantTracker


app = Flask(__name__)

geofence = Geofence()
detector: InfantDetector | None = None
tracker: InfantTracker | None = None
pose_estimator: PoseEstimator | None = None
object_detector: SurroundingsObjectDetector | None = None
risk_analyzer = RiskAnalyzer()
alert_manager = AlertManager()
frame_counter = 0
cached_objects: list[dict[str, Any]] = []


def decode_frame(data_url: str) -> np.ndarray:
    """Decode a browser canvas data URL into an OpenCV BGR frame."""
    encoded = data_url.split(",", 1)[-1]
    frame_bytes = base64.b64decode(encoded)
    np_buffer = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode frame")
    return frame


def get_detector() -> InfantDetector:
    global detector
    if detector is None:
        detector = InfantDetector()
    return detector


def get_tracker() -> InfantTracker:
    global tracker
    if tracker is None:
        tracker = InfantTracker()
    return tracker


def get_pose_estimator() -> PoseEstimator:
    global pose_estimator
    if pose_estimator is None:
        pose_estimator = PoseEstimator()
    return pose_estimator


def get_object_detector() -> SurroundingsObjectDetector:
    global object_detector
    if object_detector is None:
        object_detector = SurroundingsObjectDetector()
    return object_detector


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.post("/api/reset")
def reset() -> Any:
    global frame_counter, cached_objects

    if tracker is not None:
        tracker.reset()
    risk_analyzer.reset()
    alert_manager.clear()
    frame_counter = 0
    cached_objects = []
    return jsonify({"ok": True})


@app.post("/api/process_frame")
def process_frame() -> Any:
    global frame_counter, cached_objects

    payload = request.get_json(force=True)
    frame = decode_frame(payload["frame"])
    selection = payload.get("selection")
    fence_payload = payload.get("geofence")

    active_detector = get_detector()
    active_tracker = get_tracker()

    detections = active_detector.detect_people(frame)
    tracked_infant = active_tracker.update(frame, detections, selection)
    fence = geofence.normalize(fence_payload, frame.shape[1], frame.shape[0])

    inside_fence = True
    fence_breached = False
    pose = None
    objects = []

    if tracked_infant:
        if fence is None:
            fence = geofence.auto_sphere(tracked_infant["bbox"], frame.shape[1], frame.shape[0])
        inside_fence = geofence.contains_bbox_centroid(fence, tracked_infant["bbox"])
        fence_breached = risk_analyzer.update_fence_state(inside_fence)

        pose = get_pose_estimator().estimate(frame, tracked_infant["bbox"])
        frame_counter += 1
        if frame_counter % 3 == 0 or fence_breached or not cached_objects:
            cached_objects = get_object_detector().detect_nearby(
                frame,
                tracked_infant["bbox"],
                include_unknown=fence_breached,
            )
        if not fence_breached:
            cached_objects = [
                obj for obj in cached_objects if not str(obj.get("source", "")).startswith("opencv-fallback")
            ]
        objects = cached_objects
    else:
        risk_analyzer.reset_fence_timer()
        cached_objects = []

    risk = risk_analyzer.analyze(
        infant=tracked_infant,
        inside_fence=inside_fence,
        fence_breached=fence_breached,
        pose=pose,
        objects=objects,
    )
    alert = alert_manager.update(risk)

    return jsonify(
        {
            "detections": detections,
            "infant": tracked_infant,
            "geofence": fence,
            "insideFence": inside_fence,
            "fenceBreached": fence_breached,
            "pose": pose,
            "objects": objects,
            "risk": risk,
            "alert": alert,
            "tracking": active_tracker.status(),
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True, use_reloader=False)
