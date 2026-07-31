import cv2
import numpy as np
import urllib.request
import os
import time
import gc
import traceback
import gradio as gr
import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

from ultralytics import YOLO
import mediapipe as mp

# --- 1. Auto-Download MediaPipe Model ---
model_path = "face_landmarker.task"
if not os.path.exists(model_path):
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, model_path)

# --- 2. Load Models ---
yolo_model = YOLO('yolov8n.pt')
PERSON_CLASS, PHONE_CLASS = 0, 67

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
landmarker = FaceLandmarker.create_from_options(options)
LEFT_EYE_INNER, LEFT_EYE_OUTER, LEFT_IRIS_CENTER = 133, 33, 468

# Throttle heavy AI inference so the small free-tier instance does not OOM.
PROCESS_INTERVAL = 0.7
_last_result = {"time": 0.0, "frame": None, "alert": None}

# Lookaway rule: flag only when the student looks left/right for more than 5 seconds.
GAZE_LOOKAWAY_SECONDS = 5.0
_gaze_state = {"dir": None, "start": None}

def get_gaze_direction(landmarks, img_w, img_h):
    inner = np.array([landmarks[LEFT_EYE_INNER].x * img_w, landmarks[LEFT_EYE_INNER].y * img_h])
    outer = np.array([landmarks[LEFT_EYE_OUTER].x * img_w, landmarks[LEFT_EYE_OUTER].y * img_h])
    iris = np.array([landmarks[LEFT_IRIS_CENTER].x * img_w, landmarks[LEFT_IRIS_CENTER].y * img_h])

    eye_width = np.linalg.norm(inner - outer)
    iris_dist = np.linalg.norm(iris - outer)

    if eye_width == 0: return "Unknown"
    ratio = iris_dist / eye_width

    if ratio < 0.42: return "Looking Right"
    elif ratio > 0.58: return "Looking Left"
    else: return "Looking Center"

# --- 3. The Core Processing Function ---
def process_frame(frame, gallery_history, last_snap_time):
    try:
        if frame is None:
            return frame, "Waiting for camera...", gallery_history, gallery_history, last_snap_time

        now = time.time()
        if now - _last_result["time"] < PROCESS_INTERVAL and _last_result["frame"] is not None:
            return _last_result["frame"], _last_result["alert"], gallery_history, gallery_history, last_snap_time

        if not _last_result["time"]:
            print("process_frame: first real frame received, running inference...")

        # FIX: Force a writable copy of the image so OpenCV can draw on it
        frame = frame.copy()

        # FIX: Strip RGBA alpha channel if it exists (MediaPipe requires exactly 3 channels)
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]

        frame = np.ascontiguousarray(frame)
        img_h, img_w = frame.shape[:2]
        violations = []

        # YOLO Object Detection (small input size keeps memory and CPU low)
        results = yolo_model.predict(frame, classes=[PERSON_CLASS, PHONE_CLASS], imgsz=256, conf=0.4, device="cpu", verbose=False)
        person_count, phone_count = 0, 0

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls_id == PERSON_CLASS:
                person_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            elif cls_id == PHONE_CLASS:
                phone_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        del results
        gc.collect()

        if person_count == 0: violations.append("NO PERSON DETECTED")
        elif person_count > 1: violations.append(f"MULTIPLE PERSONS ({person_count})")
        if phone_count > 0: violations.append("CELL PHONE DETECTED")

        # MediaPipe Eye Tracking
        if person_count == 1:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            face_result = landmarker.detect(mp_image)

            if face_result.face_landmarks:
                landmarks = face_result.face_landmarks[0]
                gaze = get_gaze_direction(landmarks, img_w, img_h)

                # Sustained lookaway: violation only when looking left/right for > 5 seconds
                if gaze in ("Looking Left", "Looking Right"):
                    if _gaze_state["dir"] != gaze:
                        _gaze_state["dir"] = gaze
                        _gaze_state["start"] = now
                    lookaway_secs = now - _gaze_state["start"]
                    if lookaway_secs > GAZE_LOOKAWAY_SECONDS:
                        violations.append(f"{gaze.upper()} {int(lookaway_secs)}s")
                    cv2.putText(frame, f"Gaze: {gaze} ({lookaway_secs:.0f}s)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    _gaze_state["dir"] = None
                    _gaze_state["start"] = None
                    cv2.putText(frame, f"Gaze: {gaze}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # UI Alert Formatting & Evidence Capture
        current_time = time.time()
        alert_html = "<div style='background-color: #d4edda; padding: 15px; border-radius: 5px; color: #155724; font-size: 18px;'><b>✅ Status: Secure</b> - No violations detected.</div>"

        if violations:
            violation_text = " | ".join(violations)
            alert_html = f"<div style='background-color: #f8d7da; padding: 15px; border-radius: 5px; color: #721c24; font-size: 18px;'><b>🚨 VIOLATION DETECTED:</b> {violation_text}</div>"

            y_pos = 70
            for v in violations:
                cv2.putText(frame, f"FLAG: {v}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                y_pos += 30

            if current_time - last_snap_time > 3.0:
                timestamp_str = time.strftime("%H:%M:%S")
                evidence_frame = frame.copy()
                cv2.putText(evidence_frame, f"Captured: {timestamp_str}", (10, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                gallery_history.insert(0, evidence_frame)
                if len(gallery_history) > 6:
                    gallery_history.pop()
                last_snap_time = current_time

        _last_result["time"] = now
        _last_result["frame"] = frame
        _last_result["alert"] = alert_html

        return frame, alert_html, gallery_history, gallery_history, last_snap_time

    except Exception as e:
        traceback.print_exc()
        error_html = f"<div style='background-color: #fff3cd; padding: 15px; border-radius: 5px; color: #856404; font-size: 18px;'><b>⚠️ System Error:</b> {str(e)}</div>"
        return frame, error_html, gallery_history, gallery_history, last_snap_time

# --- 4. Professional Gradio Dashboard ---
custom_theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate")

with gr.Blocks(title="AI Exam Proctor") as app:
    gallery_state = gr.State(value=[])
    last_snap_state = gr.State(value=0.0)

    gr.Markdown("# 🛡️ Automated AI Exam Proctoring System")
    gr.Markdown("### Developed for maintaining academic integrity via real-time computer vision.")
    gr.Markdown("**Detects:** Cell phone in view • Multiple persons • No person present • Looking left/right for more than 5 seconds")
    gr.Markdown("**How to start:** 1) Click the webcam area and allow camera access 2) Press the red circle Record button on the webcam 3) Watch the orange timer bar start - the AI box will then update live.")

    alert_box = gr.HTML(value="<div style='background-color: #e2e3e5; padding: 15px; border-radius: 5px; color: #383d41; font-size: 18px;'>Waiting for camera feed...</div>")

    with gr.Row():
        input_cam = gr.Image(label="Student Webcam Feed", sources=["webcam"], streaming=True)
        output_cam = gr.Image(label="AI Monitoring View (Real-time Analysis)", interactive=False, streaming=True)

    gr.Markdown("### 📸 Evidence Log (Captured Violations)")
    evidence_gallery = gr.Gallery(label="Violation Screenshots", show_label=False, elem_id="gallery", columns=3, rows=1, height="auto")

    input_cam.stream(
        fn=process_frame,
        inputs=[input_cam, gallery_state, last_snap_state],
        outputs=[output_cam, alert_box, evidence_gallery, gallery_state, last_snap_state],
        time_limit=120,
        stream_every=0.1,
        concurrency_limit=1
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.launch(server_name="0.0.0.0", server_port=port, theme=custom_theme)
