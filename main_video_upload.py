import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from ultralytics import YOLO
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from tracker import Sort

# Load YOLOv8 model (caches on first run)
@st.cache_resource
def load_model():
    # return YOLO("yolov8n.pt")  # nano model for speed
    return YOLO("yolov8s.pt")  # small model
    # return YOLO("yolov8m.pt")  # medium model
    # return YOLO("yolov8l.pt")  # large model

@st.cache_resource
def load_tracker():
    return Sort()

st.set_page_config(page_title="Computer Vision Pipeline Demo (Video Upload)", layout="wide")

st.title("📷 Computer Vision Pipeline Demo (Video Upload)")
st.markdown("""
This demo processes an uploaded video for object detection and tracking using YOLOv8.
""")

# Video upload
MAX_MB = 50
MAX_DURATION = 50  # seconds
uploaded_file = st.file_uploader("Upload a video (max 50MB, 50s)", type=["mp4", "avi", "mov"])

if 'last_uploaded_file' not in st.session_state:
    st.session_state['last_uploaded_file'] = None

if uploaded_file is not None:
    if uploaded_file != st.session_state['last_uploaded_file']:
        st.session_state['last_uploaded_file'] = uploaded_file
        st.rerun()

FRAME_WINDOW = st.empty()

if uploaded_file is not None:
    # Check file size
    uploaded_file.seek(0, os.SEEK_END)
    file_size_mb = uploaded_file.tell() / (1024 * 1024)
    uploaded_file.seek(0)
    if file_size_mb > MAX_MB:
        st.error(f"File too large: {file_size_mb:.2f} MB. Please upload a file under {MAX_MB} MB.")
    else:
        # Save to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        if duration > MAX_DURATION:
            st.error(f"Video too long: {duration:.1f} seconds. Please upload a video under {MAX_DURATION} seconds.")
            cap.release()
            os.remove(tfile.name)
        else:
            st.success(f"Processing video: {file_size_mb:.2f} MB, {duration:.1f} seconds")
            # Process video
            model = load_model()
            tracker = load_tracker()
            processed_frames = []
            frame_count = 0
            progress_bar = st.progress(0)
            CONF_THRESH = 0.4
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Run YOLOv8 detection
                results = model(frame)
                boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
                confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else []
                clss = results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else []
                names = model.model.names
                # Filter by confidence threshold
                keep = confs > CONF_THRESH
                boxes, confs, clss = boxes[keep], confs[keep], clss[keep]
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
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if frame_count % int(fps) == 0:  # Show 1 frame per second
                    processed_frames.append(frame_rgb)
                frame_count += 1
                if total_frames > 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
            progress_bar.empty()
            cap.release()
            os.remove(tfile.name)
            st.success(f"Processed {frame_count} frames.")
            # Display sample frames
            st.write("### Sample Results (1 frame per second)")
            for i, f in enumerate(processed_frames):
                st.image(f, caption=f"Frame {i*int(fps)}", use_container_width=True)
else:
    st.info("Upload a video above to begin.") 