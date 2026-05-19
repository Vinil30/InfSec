# AI-Powered Infant Safety Monitoring MVP

Flask web app that streams webcam frames from the browser to a modular OpenCV/YOLO/MediaPipe backend. The user selects the infant once, a safe geofence is monitored, and heavier pose plus surroundings analysis only activates after a boundary breach.

## Features

- Browser webcam capture with live overlay canvas.
- YOLOv8 nano pretrained person detection.
- Selected infant tracking module with a ByteTrack-ready boundary.
- Drawable polygon geofence with centroid-in-polygon checks.
- MediaPipe Pose posture checks after geofence breach.
- YOLO surroundings analysis after breach only.
- Nearby objects categorized as `Safe`, `Warning`, or `Dangerous`.
- Rule-based `LOW`, `MEDIUM`, and `HIGH` risk scoring.
- High-risk red overlay and audible frontend alarm.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads the pretrained `yolov8n.pt` weights automatically through Ultralytics. No custom model training is required.

## Run Locally

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

Then:

1. Click `Start Camera` and allow webcam access.
2. Click `Select Infant`, then click the infant in the video.
3. Use the default safe region or click `Draw Fence`, place at least three points, and click `Close Fence`.
4. When the infant crosses the fence, pose and nearby-object analysis starts.

## Notes

- The MVP uses pretrained COCO classes only. COCO does not include every real-world hazard label such as chemical or wire, so the dangerous set maps available classes like `bottle`, `knife`, `scissors`, electronics, and similar objects.
- `utils/tracker.py` isolates tracking. For HTTP snapshot processing it uses stable IoU/center matching around the selected infant; the dependency stack includes Ultralytics ByteTrack and this module is the intended replacement point for a persistent `YOLO.track(..., tracker="bytetrack.yaml")` stream loop.
- This is a safety-support MVP, not a certified medical or child-safety device. Always supervise infants directly.
