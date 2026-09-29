# Object Detection & Tracking — CodeAlpha AI Internship

Real-time multi-target object detection and tracking pipeline built with **YOLOv8** (Ultralytics), **Deep SORT**, **OpenCV**, and **Streamlit**. Supports live webcam streams, video file inference, motion trajectory trails, and comprehensive detection analytics.

![Python](https://img.shields.io/badge/Python-3.9+-18181b?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-18181b)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-18181b)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-18181b)

---

## Features

- **YOLOv8 Neural Detection**: Real-time bounding box detection across 80 COCO object classes (pedestrians, vehicles, animals, accessories, etc.).
- **Deep SORT Persistent Tracking**: Assigns persistent track IDs across frames using Kalman filtering and visual appearance embeddings.
- **Dual Runtime Interfaces**:
  - **OpenCV Desktop GUI / CLI**: Zero-latency live webcam and video streaming (`main.py`).
  - **Streamlit Web Dashboard**: Clean, modern video tracking suite with live progress and class breakdown statistics (`app.py`).
- **Motion Trails & Trajectory Visualization**: Renders smooth fading motion paths following tracked entities.
- **Real-time Metrics**: Live FPS counter, active object counts, unique entity tracking, and per-class distribution summaries.
- **Sample Demo Video**: Pre-configured test clip included for zero-setup verification.

---

## Tech Stack

| Component | Technology |
|---|---|
| **Object Detection** | YOLOv8 (Ultralytics) |
| **Multi-Object Tracking** | Deep SORT (`deep-sort-realtime`) |
| **Computer Vision Engine** | OpenCV (`cv2`) |
| **Web Dashboard** | Streamlit |
| **Language** | Python 3.9+ |

---

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/berlin-22/codealpha-object_detection.git
   cd codealpha-object_detection
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Web Application Dashboard
To launch the interactive web dashboard with video processing and analytics:
```bash
streamlit run app.py
```
Open `http://localhost:8501` (or `http://localhost:8503`) in your browser, select either the **Demo Video** or upload your own file (`.mp4`, `.avi`, `.mov`), and click **Start Tracking & Detection**.

### 2. Live Webcam / Video File CLI
For real-time desktop execution:
```bash
# Default webcam (index 0)
python main.py

# Custom video file input
python main.py --source path/to/video.mp4

# Custom confidence and video output export
python main.py --source 0 --conf 0.45 --output output_annotated.mp4

# Run without motion trails
python main.py --no-trails
```

**Keyboard Controls (CLI window):**
* `q`: Quit stream.
* `s`: Save an annotated snapshot of the current frame.

---

## Project Structure

```
CodeAlpha_ObjectDetectionTracking/
├── .streamlit/
│   └── config.toml     # Minimalist theme configuration
├── app.py              # Streamlit web application & analytics dashboard
├── main.py             # OpenCV CLI application for real-time webcam/video
├── detector.py         # YOLOv8 object detection wrapper
├── tracker.py          # Deep SORT tracker integration
├── utils.py            # Drawing overlays, color generation, and FPS counter
├── sample_video.mp4    # Sample demo video clip
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules
└── README.md           # Documentation
```

---

## How It Works

```
Video Stream ──► Frame Capture (OpenCV) ──► YOLOv8 Detection ([x1, y1, x2, y2], conf, class)
                                                        │
┌───────────────────────────────────────────────────────┘
▼
Deep SORT Association ──► Kalman Filter + ReID Embedding ──► Persistent Track ID
                                                                   │
┌──────────────────────────────────────────────────────────────────┘
▼
Overlay Annotations ──► Bounding Boxes + Labels + Motion Trails + FPS Overlay
```

---

## Acknowledgements

Built for the **CodeAlpha Artificial Intelligence Internship** (Task 4).
