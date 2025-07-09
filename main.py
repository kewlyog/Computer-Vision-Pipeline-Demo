import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from tracker import Sort

st.set_page_config(page_title="Computer Vision Pipeline Demo", layout="wide")

st.title("📷 Computer Vision Pipeline Demo")
st.markdown("""
This demo captures real-time video from your webcam and performs real-time object detection using YOLOv8.
""")

run = st.toggle("Start Camera", value=False, key="start_camera")

# Load YOLOv8 model (caches on first run)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # nano model for speed

@st.cache_resource
def load_tracker():
    return Sort()

model = load_model()
tracker = load_tracker()

FRAME_WINDOW = st.empty()

if run:
    cap = cv2.VideoCapture(0)
    st.success("Webcam started!")
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break
        # Run YOLOv8 detection
        results = model(frame)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
        confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else []
        clss = results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else []
        names = model.model.names
        # Prepare detections for tracker: [x1, y1, x2, y2, conf]
        dets = np.array([
            [*box, conf] for box, conf in zip(boxes, confs)
        ]) if len(boxes) > 0 else np.empty((0, 5))
        tracks = tracker.update(dets)
        # Draw boxes, labels, and IDs
        for (box, track_id), conf, cls in zip(tracks, confs, clss):
            x1, y1, x2, y2 = map(int, box)
            label = f"{names[int(cls)]} {conf:.2f} ID:{track_id}"
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame, channels="RGB", use_container_width=True, caption="Live Object Detection & Tracking")
        # Streamlit needs to yield control to UI
        if not st.session_state.start_camera:
            break
    cap.release()
    st.info("Webcam stopped.")
else:
    st.info("Click 'Start Camera' above to begin.")

st.markdown("---")
st.subheader("Detection & Tracking Results")
st.write("Detected objects are shown with bounding boxes, labels, and unique IDs in the video feed above.") 