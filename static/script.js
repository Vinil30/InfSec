const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const startBtn = document.getElementById("startBtn");
const resetBtn = document.getElementById("resetBtn");
const trackingStatus = document.getElementById("trackingStatus");
const riskLevel = document.getElementById("riskLevel");
const fenceStatus = document.getElementById("fenceStatus");
const objectCount = document.getElementById("objectCount");
const alertMessage = document.getElementById("alertMessage");
const reasonList = document.getElementById("reasonList");
const objectList = document.getElementById("objectList");
const riskOverlay = document.getElementById("riskOverlay");

let stream = null;
let processing = false;
let selectionMode = false;
let selection = null;
let geofence = null;
let latestResponse = null;
let audioContext = null;
let alarmOscillator = null;
let hasSelectedInfant = false;

startBtn.addEventListener("click", startCamera);
resetBtn.addEventListener("click", resetState);
canvas.addEventListener("click", handleCanvasClick);
window.addEventListener("resize", resizeCanvas);

async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 960 }, height: { ideal: 540 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  resizeCanvas();
  processing = true;
  hasSelectedInfant = false;
  selectionMode = true;
  geofence = null;
  latestResponse = null;
  await fetch("/api/reset", { method: "POST" });
  window.setTimeout(() => {
    alert("Select the infant to begin monitoring. Click once on the infant in the camera view.");
  }, 100);
  alertMessage.textContent = "Infant selection is required. Click once on the infant in the live feed.";
  processLoop();
}

async function resetState() {
  selection = null;
  geofence = null;
  latestResponse = null;
  hasSelectedInfant = false;
  selectionMode = true;
  stopAlarm();
  await fetch("/api/reset", { method: "POST" });
  drawOverlay();
  updatePanel({
    tracking: { active: false },
    risk: { level: "LOW", reasons: ["Reset complete"] },
    alert: { message: "Infant selection is required. Click once on the infant in the live feed." },
    objects: [],
    insideFence: true,
    geofence: null,
  });
}

function resizeCanvas() {
  const rect = video.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width));
  canvas.height = Math.max(1, Math.floor(rect.height));
  drawOverlay();
}

async function handleCanvasClick(event) {
  const point = eventToVideoPoint(event);
  if (!point || !selectionMode) return;

  await fetch("/api/reset", { method: "POST" });
  selection = point;
  geofence = null;
  latestResponse = null;
  hasSelectedInfant = true;
  selectionMode = false;
  alertMessage.textContent = "Infant selected. Creating the measured safety sphere.";
  drawOverlay();
}

async function processLoop() {
  if (!processing) return;
  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    try {
      const frame = captureFrame();
      if (!hasSelectedInfant) {
        drawOverlay();
      } else {
        const response = await fetch("/api/process_frame", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ frame, selection, geofence }),
        });
        latestResponse = await response.json();
        if (latestResponse.infant) {
          selection = null;
        }
        if (latestResponse.geofence && geofence === null) {
          geofence = latestResponse.geofence;
        }
        drawOverlay();
        updatePanel(latestResponse);
        handleAlarm(latestResponse.alert?.alarm);
      }
    } catch (error) {
      alertMessage.textContent = `Processing error: ${error.message}`;
    }
  }
  window.setTimeout(processLoop, 180);
}

function captureFrame() {
  const offscreen = document.createElement("canvas");
  offscreen.width = video.videoWidth;
  offscreen.height = video.videoHeight;
  const offscreenContext = offscreen.getContext("2d");
  offscreenContext.drawImage(video, 0, 0, offscreen.width, offscreen.height);
  return offscreen.toDataURL("image/jpeg", 0.72);
}

function drawOverlay() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawGeofence(geofence);

  if (!latestResponse) return;

  for (const det of latestResponse.detections || []) {
    drawBox(det.bbox, "Person", "#65a8ff", det.confidence);
  }

  if (latestResponse.infant) {
    drawBox(latestResponse.infant.bbox, "Selected infant", "#20c997", latestResponse.infant.confidence, 4);
  }

  for (const obj of latestResponse.objects || []) {
    const color = obj.category === "Dangerous" ? "#ff4d4f" : obj.category === "Warning" ? "#f5b942" : "#20c997";
    drawBox(obj.bbox, `${obj.label} - ${obj.category}`, color, obj.confidence, 3);
  }

  drawPose(latestResponse.pose);
}

function drawGeofence(points) {
  if (!points) return;
  if (points.type === "sphere") {
    drawSphereFence(points);
    return;
  }
  if (points.length === 0) return;
  ctx.save();
  ctx.strokeStyle = "#f5b942";
  ctx.fillStyle = "rgba(245, 185, 66, 0.08)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  const first = videoToCanvasPoint(points[0]);
  ctx.moveTo(first.x, first.y);
  for (const point of points.slice(1)) {
    const mapped = videoToCanvasPoint(point);
    ctx.lineTo(mapped.x, mapped.y);
  }
  if (points.length >= 3) ctx.closePath();
  ctx.stroke();
  if (points.length >= 3) ctx.fill();
  for (const point of points) {
    const mapped = videoToCanvasPoint(point);
    ctx.beginPath();
    ctx.arc(mapped.x, mapped.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = "#f5b942";
    ctx.fill();
  }
  ctx.restore();
}

function drawSphereFence(fence) {
  const center = videoToCanvasPoint(fence.center);
  const edge = videoToCanvasPoint({ x: fence.center.x + fence.radius, y: fence.center.y });
  const radius = Math.abs(edge.x - center.x);

  ctx.save();
  ctx.strokeStyle = "#f5b942";
  ctx.fillStyle = "rgba(245, 185, 66, 0.08)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  ctx.setLineDash([8, 8]);
  ctx.globalAlpha = 0.75;
  ctx.beginPath();
  ctx.ellipse(center.x, center.y, radius, radius * 0.36, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.ellipse(center.x, center.y, radius * 0.36, radius, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.globalAlpha = 1;

  ctx.beginPath();
  ctx.moveTo(center.x, center.y);
  ctx.lineTo(center.x + radius, center.y);
  ctx.stroke();

  const label = `Safe sphere R=${fence.radius}${fence.unit || "px"} D=${fence.diameter}${fence.unit || "px"}`;
  ctx.font = "13px system-ui";
  const labelWidth = ctx.measureText(label).width + 12;
  ctx.fillStyle = "#f5b942";
  ctx.fillRect(center.x - labelWidth / 2, center.y - radius - 28, labelWidth, 22);
  ctx.fillStyle = "#05070a";
  ctx.fillText(label, center.x - labelWidth / 2 + 6, center.y - radius - 12);
  ctx.restore();
}

function drawBox(bbox, label, color, confidence = null, width = 2) {
  const p1 = videoToCanvasPoint({ x: bbox[0], y: bbox[1] });
  const p2 = videoToCanvasPoint({ x: bbox[2], y: bbox[3] });
  const labelText = confidence === null ? label : `${label} ${Math.round(confidence * 100)}%`;

  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = width;
  ctx.strokeRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
  ctx.font = "13px system-ui";
  const labelWidth = ctx.measureText(labelText).width + 10;
  ctx.fillRect(p1.x, Math.max(0, p1.y - 24), labelWidth, 22);
  ctx.fillStyle = "#05070a";
  ctx.fillText(labelText, p1.x + 5, Math.max(14, p1.y - 8));
  ctx.restore();
}

function drawPose(pose) {
  const landmarks = pose?.landmarks;
  if (!landmarks) return;
  ctx.save();
  ctx.fillStyle = "#ffffff";
  for (const point of Object.values(landmarks)) {
    if (point.visibility < 0.35) continue;
    const mapped = videoToCanvasPoint(point);
    ctx.beginPath();
    ctx.arc(mapped.x, mapped.y, 4, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function updatePanel(data) {
  trackingStatus.textContent = data.tracking?.active
    ? data.tracking.trackId && data.tracking.trackId !== -1 ? `Track #${data.tracking.trackId}` : "Selected"
    : "Waiting";
  riskLevel.textContent = data.risk?.level || "LOW";
  riskLevel.className = `risk-${(data.risk?.level || "LOW").toLowerCase()}`;
  fenceStatus.textContent = data.geofence ? (data.insideFence === false ? "Breached" : "Inside") : "Waiting";
  objectCount.textContent = (data.objects || []).length;
  alertMessage.textContent = data.alert?.message || "Monitoring normally.";
  riskOverlay.classList.toggle("active", data.risk?.level === "HIGH");

  reasonList.innerHTML = "";
  for (const reason of data.risk?.reasons || []) {
    const li = document.createElement("li");
    li.textContent = reason;
    reasonList.appendChild(li);
  }

  objectList.innerHTML = "";
  const objects = data.objects || [];
  if (objects.length === 0) {
    objectList.textContent = data.fenceBreached
      ? "No nearby objects detected."
      : "Object analysis activates after a fence breach.";
    return;
  }

  for (const obj of objects) {
    const chip = document.createElement("div");
    chip.className = "object-chip";
    chip.innerHTML = `<strong>${obj.label}</strong><span>${obj.category} - ${obj.distance}px</span>`;
    objectList.appendChild(chip);
  }
}

function handleAlarm(active) {
  if (active) {
    startAlarm();
  } else {
    stopAlarm();
  }
}

function startAlarm() {
  if (alarmOscillator) return;
  audioContext = audioContext || new AudioContext();
  alarmOscillator = audioContext.createOscillator();
  const gain = audioContext.createGain();
  alarmOscillator.type = "square";
  alarmOscillator.frequency.value = 880;
  gain.gain.value = 0.06;
  alarmOscillator.connect(gain);
  gain.connect(audioContext.destination);
  alarmOscillator.start();
}

function stopAlarm() {
  if (!alarmOscillator) return;
  alarmOscillator.stop();
  alarmOscillator.disconnect();
  alarmOscillator = null;
}

function eventToVideoPoint(event) {
  if (!video.videoWidth || !video.videoHeight) return null;
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  return canvasToVideoPoint({ x, y });
}

function videoToCanvasPoint(point) {
  const fit = objectFitContain();
  return {
    x: fit.offsetX + point.x * fit.scale,
    y: fit.offsetY + point.y * fit.scale,
  };
}

function canvasToVideoPoint(point) {
  const fit = objectFitContain();
  return {
    x: Math.round((point.x - fit.offsetX) / fit.scale),
    y: Math.round((point.y - fit.offsetY) / fit.scale),
  };
}

function objectFitContain() {
  const scale = Math.min(canvas.width / video.videoWidth, canvas.height / video.videoHeight);
  const renderedWidth = video.videoWidth * scale;
  const renderedHeight = video.videoHeight * scale;
  return {
    scale,
    offsetX: (canvas.width - renderedWidth) / 2,
    offsetY: (canvas.height - renderedHeight) / 2,
  };
}
