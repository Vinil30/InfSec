from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class Geofence:
    """Geofence helpers for polygons and auto-created sphere zones."""

    def normalize(
        self, fence: Any, frame_width: int, frame_height: int
    ) -> dict[str, Any] | list[dict[str, int]] | None:
        if isinstance(fence, dict) and fence.get("type") == "sphere":
            center = fence.get("center", {})
            radius = max(20, int(fence.get("radius", 0)))
            return {
                "type": "sphere",
                "center": {
                    "x": int(max(0, min(frame_width, center.get("x", 0)))),
                    "y": int(max(0, min(frame_height, center.get("y", 0)))),
                },
                "radius": int(min(radius, max(frame_width, frame_height))),
                "diameter": int(radius * 2),
                "unit": "px",
            }

        points = fence if isinstance(fence, list) else []
        if len(points) >= 3:
            return [
                {
                    "x": int(max(0, min(frame_width, point["x"]))),
                    "y": int(max(0, min(frame_height, point["y"]))),
                }
                for point in points
            ]

        return None

    def auto_sphere(
        self, bbox: list[int], frame_width: int, frame_height: int
    ) -> dict[str, Any]:
        x1, y1, x2, y2 = bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        radius = int(max(width, height) * 1.15)
        radius = max(90, min(radius, int(min(frame_width, frame_height) * 0.42)))
        return {
            "type": "sphere",
            "center": {"x": center_x, "y": center_y},
            "radius": radius,
            "diameter": radius * 2,
            "unit": "px",
        }

    def contains_bbox_centroid(self, fence: Any, bbox: list[int]) -> bool:
        if not fence:
            return True

        x1, y1, x2, y2 = bbox
        centroid = ((x1 + x2) / 2, (y1 + y2) / 2)

        if isinstance(fence, dict) and fence.get("type") == "sphere":
            center = fence["center"]
            distance = np.hypot(centroid[0] - center["x"], centroid[1] - center["y"])
            return bool(distance <= fence["radius"])

        contour = np.array([[point["x"], point["y"]] for point in fence], dtype=np.int32)
        return cv2.pointPolygonTest(contour, centroid, False) >= 0
