"""
Object Detection & Tracking — CLI Runner
=========================================
Real-time object detection (YOLOv8) + tracking (Deep SORT) via webcam or video file.

Usage:
    python main.py                          # Webcam (default)
    python main.py --source video.mp4       # Video file
    python main.py --source 0 --conf 0.4    # Webcam with custom confidence
    python main.py --source video.mp4 --output result.mp4 --no-trails
"""

import argparse
import cv2
import sys

from detector import ObjectDetector
from tracker import ObjectTracker
from utils import draw_boxes, draw_stats_overlay, FPSCounter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time Object Detection & Tracking using YOLOv8 + Deep SORT"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: webcam index (0, 1, ...) or path to video file (default: 0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLOv8 model variant: yolov8n.pt (nano), yolov8s.pt (small), etc. (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save output video (default: None, no save)",
    )
    parser.add_argument(
        "--no-trails",
        action="store_true",
        help="Disable motion trail visualization",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Initialize components ─────────────────────────────────────────────
    print(f"[INFO] Loading YOLOv8 model: {args.model}")
    detector = ObjectDetector(model_name=args.model, confidence=args.conf)

    print("[INFO] Initializing Deep SORT tracker")
    tracker = ObjectTracker(max_age=30, n_init=3)

    fps_counter = FPSCounter(window=30)
    trails = {} if not args.no_trails else None

    # ── Open video source ─────────────────────────────────────────────────
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {args.source}")
        sys.exit(1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    print(f"[INFO] Video source opened: {frame_w}x{frame_h} @ {source_fps:.0f} FPS")

    # ── Video writer (optional) ───────────────────────────────────────────
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, source_fps, (frame_w, frame_h))
        print(f"[INFO] Saving output to: {args.output}")

    # ── Main loop ─────────────────────────────────────────────────────────
    print("[INFO] Starting detection & tracking... Press 'q' to quit, 's' to screenshot")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str):
                print("[INFO] End of video file reached.")
            else:
                print("[ERROR] Failed to read frame from webcam.")
            break

        fps_counter.tick()
        frame_count += 1

        # Detect objects
        detections = detector.detect(frame, confidence=args.conf)

        # Track objects
        tracked_objects = tracker.update(detections, frame)

        # Draw visualizations
        annotated_frame = draw_boxes(frame, tracked_objects, trails=trails)
        annotated_frame = draw_stats_overlay(
            annotated_frame, tracked_objects, fps_counter.get_fps()
        )

        # Display
        cv2.imshow("Object Detection & Tracking — CodeAlpha", annotated_frame)

        # Save to output video
        if writer is not None:
            writer.write(annotated_frame)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] Quit requested by user.")
            break
        elif key == ord("s"):
            screenshot_path = f"screenshot_{frame_count}.png"
            cv2.imwrite(screenshot_path, annotated_frame)
            print(f"[INFO] Screenshot saved: {screenshot_path}")

    # ── Cleanup ───────────────────────────────────────────────────────────
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
