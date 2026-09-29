import streamlit as st
import cv2
import tempfile
import os
from detector import ObjectDetector
from tracker import ObjectTracker
from utils import draw_boxes, draw_stats_overlay, FPSCounter

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Object Detection & Tracking",
    page_icon="⌖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS: Modern Clean Minimalist Design (Strictly Zero Blue) ───────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        color: #18181b !important;
    }

    .stApp {
        background-color: #f7f7f8 !important;
    }

    #MainMenu, header, footer, .stDeployButton {
        display: none !important;
    }

    /* Main Container Flow */
    .main .block-container {
        max-width: 1100px !important;
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }

    /* Top Brand Navigation */
    .brand-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 1.25rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #e4e4e7;
    }
    .brand-title-wrap {
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
    }
    .brand-title {
        font-size: 1.35rem;
        font-weight: 600;
        color: #18181b;
        letter-spacing: -0.025em;
        margin: 0;
    }
    .brand-subtitle {
        font-size: 0.82rem;
        color: #71717a;
        margin: 0;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.75rem;
        font-weight: 500;
        color: #52525b;
        background: #ffffff;
        border: 1px solid #e4e4e7;
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
    }
    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #10b981;
    }

    /* Clean Stat Cards */
    .stat-card {
        background: #ffffff;
        border: 1px solid #e4e4e7;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }
    .stat-value {
        color: #18181b;
        font-size: 1.6rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        line-height: 1;
    }
    .stat-label {
        color: #71717a;
        font-size: 0.78rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e4e4e7 !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem !important;
    }
    .sidebar-heading {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #71717a;
        margin-bottom: 0.6rem;
    }

    /* Primary Action Button (Obsidian Charcoal) */
    button[data-testid="stBaseButton-primary"] {
        background-color: #18181b !important;
        color: #ffffff !important;
        border: 1px solid #18181b !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        padding: 0.6rem 1.5rem !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.15s ease !important;
    }
    button[data-testid="stBaseButton-primary"]:hover {
        background-color: #27272a !important;
        border-color: #27272a !important;
        color: #ffffff !important;
    }

    /* Container Cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff !important;
        border: 1px solid #e4e4e7 !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
        padding: 1.25rem !important;
    }

    /* File Uploader */
    div[data-testid="stFileUploader"] {
        border-radius: 8px !important;
    }

    /* Footer */
    .app-footer {
        margin-top: 3rem;
        padding-top: 1.25rem;
        border-top: 1px solid #e4e4e7;
        text-align: center;
        font-size: 0.78rem;
        color: #a1a1aa;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="brand-nav">
        <div class="brand-title-wrap">
            <h1 class="brand-title">Object Detection & Tracking</h1>
            <p class="brand-subtitle">Real-time multi-target tracking powered by YOLOv8 and Deep SORT</p>
        </div>
        <div class="status-pill">
            <span class="status-dot"></span>
            <span>Inference Pipeline Online</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar Settings ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-heading">Model Configuration</div>', unsafe_allow_html=True)

    model_choice = st.selectbox(
        "YOLOv8 Architecture",
        ["yolov8n.pt (Nano — High Speed)", "yolov8s.pt (Small — Balanced)"],
        index=0,
    )
    model_name = model_choice.split(" ")[0]

    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.45, 0.05)
    show_trails = st.checkbox("Render Motion Trails", value=True)
    process_every_n = st.slider("Process Every N Frames", 1, 5, 1, help="Skip frames for faster playback")

    st.divider()

    st.markdown('<div class="sidebar-heading">Webcam CLI Execution</div>', unsafe_allow_html=True)
    st.caption("For zero-latency local camera streaming, run:")
    st.code("python main.py --source 0", language="bash")

    st.divider()
    st.markdown(
        "<div style='text-align: center; color: #a1a1aa; font-size: 0.75rem;'>"
        "CodeAlpha AI Internship • Task 4"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Video Source Selector ────────────────────────────────────────────────────
source_mode = st.radio(
    "Video Input Source",
    ["Demo Video (Sample)", "Upload Custom Video"],
    horizontal=True,
    label_visibility="collapsed",
)

video_path = None
temp_file_to_clean = None

if source_mode == "Demo Video (Sample)":
    sample_file = "sample_video.mp4"
    if os.path.exists(sample_file):
        video_path = sample_file
    else:
        st.error("Sample video not found. Please upload a video.")
else:
    uploaded_file = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
        help="Upload an MP4, AVI, or MOV video to run tracking on.",
    )
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        temp_file_to_clean = video_path
        tfile.close()

# ── Video Processing Pipeline ────────────────────────────────────────────────
if video_path:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    duration = total_frames / video_fps if video_fps > 0 else 0
    cap.release()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div class="stat-card"><span class="stat-label">Total Frames</span><span class="stat-value">{total_frames:,}</span></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="stat-card"><span class="stat-label">Source FPS</span><span class="stat-value">{video_fps:.0f}</span></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="stat-card"><span class="stat-label">Duration</span><span class="stat-value">{duration:.1f}s</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    if st.button("Start Tracking & Detection", type="primary", use_container_width=True):
        detector = ObjectDetector(model_name=model_name, confidence=confidence)
        tracker = ObjectTracker(max_age=30, n_init=3)
        fps_counter = FPSCounter()
        trails = {} if show_trails else None

        cap = cv2.VideoCapture(video_path)
        progress_bar = st.progress(0, text="Initializing tracker...")
        frame_display = st.empty()

        all_class_counts = {}
        total_unique_tracks = set()
        frame_idx = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                fps_counter.tick()

                if frame_idx % process_every_n != 0:
                    continue

                detections = detector.detect(frame, confidence=confidence)
                tracked_objects = tracker.update(detections, frame)

                for obj in tracked_objects:
                    cls = obj["class_name"]
                    all_class_counts[cls] = all_class_counts.get(cls, 0) + 1
                    total_unique_tracks.add(obj["track_id"])

                annotated = draw_boxes(frame, tracked_objects, trails=trails)
                annotated = draw_stats_overlay(annotated, tracked_objects, fps_counter.get_fps())

                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                # Smooth refresh
                if (frame_idx // process_every_n) % 2 == 0 or frame_idx == total_frames:
                    frame_display.image(annotated_rgb, channels="RGB")

                progress = frame_idx / total_frames if total_frames > 0 else 1.0
                progress_bar.progress(
                    min(progress, 1.0),
                    text=f"Tracking frame {frame_idx}/{total_frames} ({progress:.0%})",
                )
            progress_bar.progress(1.0, text="Inference & tracking complete")
        except Exception as e:
            st.error(f"Error during video tracking at frame {frame_idx}: {e}")
        finally:
            cap.release()

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-heading">Analytics & Detection Summary</div>', unsafe_allow_html=True)

        scol1, scol2 = st.columns(2)
        with scol1:
            st.metric("Unique Tracked Entities", len(total_unique_tracks))
        with scol2:
            st.metric("Distinct Object Categories", len(all_class_counts))

        if all_class_counts:
            with st.container(border=True):
                st.markdown("**Per-Class Detection Counts:**")
                for cls, count in sorted(all_class_counts.items(), key=lambda x: -x[1]):
                    st.markdown(f"• **{cls.title()}**: {count:,} instances identified")

    if temp_file_to_clean:
        try:
            os.unlink(temp_file_to_clean)
        except Exception:
            pass
else:
    st.info("Select 'Demo Video' or upload a video file above to begin tracking.")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-footer">
        Powered by YOLOv8 (Ultralytics) & Deep SORT (Kalman Filter + ReID) • COCO 80-Class Recognition
    </div>
    """,
    unsafe_allow_html=True,
)
