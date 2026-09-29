import cv2
import numpy as np
from collections import deque
import time


# Generate a fixed color palette for tracking IDs
def _generate_colors(n: int = 100) -> list[tuple[int, int, int]]:
    """Generate N visually distinct BGR colors."""
    colors = []
    for i in range(n):
        hue = int(i * 180 / n)  # Spread across hue spectrum
        color_hsv = np.array([[[hue, 200, 255]]], dtype=np.uint8)
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in color_bgr))
    return colors


COLOR_PALETTE = _generate_colors(100)


def get_color_for_id(track_id) -> tuple[int, int, int]:
    """Get a consistent color for a tracking ID."""
    try:
        idx = int(track_id)
    except (ValueError, TypeError):
        idx = hash(str(track_id))
    return COLOR_PALETTE[abs(idx) % len(COLOR_PALETTE)]


def draw_boxes(frame, tracked_objects: list[dict], trails: dict = None) -> np.ndarray:
    """
    Draw bounding boxes, labels, and tracking IDs on the frame.

    Args:
        frame: BGR image (numpy array).
        tracked_objects: List of tracked object dicts from ObjectTracker.update().
        trails: Dict mapping track_id -> deque of center points (for motion trails).

    Returns:
        Annotated frame.
    """
    annotated = frame.copy()

    for obj in tracked_objects:
        track_id = obj["track_id"]
        x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
        class_name = obj["class_name"]
        confidence = obj["confidence"]
        color = get_color_for_id(track_id)

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label text
        label = f"{class_name} | ID:{track_id} | {confidence:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1

        # Get text size for background rectangle
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        text_y = max(y1 - 8, text_h + 4)

        # Draw semi-transparent background for text
        overlay = annotated.copy()
        cv2.rectangle(
            overlay,
            (x1, text_y - text_h - 4),
            (x1 + text_w + 8, text_y + 4),
            color,
            -1,
        )
        cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)

        # Draw text
        cv2.putText(annotated, label, (x1 + 4, text_y), font, font_scale, (255, 255, 255), thickness)

        # Draw motion trail
        if trails is not None:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if track_id not in trails:
                trails[track_id] = deque(maxlen=40)
            trails[track_id].append((cx, cy))

            points = list(trails[track_id])
            for i in range(1, len(points)):
                alpha = i / len(points)  # Fade effect
                pt_color = tuple(int(c * alpha) for c in color)
                pt_thickness = max(1, int(2 * alpha))
                cv2.line(annotated, points[i - 1], points[i], pt_color, pt_thickness)

    return annotated


def draw_stats_overlay(frame, tracked_objects: list[dict], fps: float) -> np.ndarray:
    """
    Draw statistics overlay (FPS, object count) on the frame.

    Args:
        frame: BGR image (numpy array).
        tracked_objects: List of tracked objects.
        fps: Current frames per second.

    Returns:
        Frame with stats overlay.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Count objects by class
    class_counts = {}
    for obj in tracked_objects:
        cls = obj["class_name"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    total_objects = len(tracked_objects)

    # Draw stats panel background
    panel_h = 30 + 22 * min(len(class_counts), 5) + 10
    overlay = annotated.copy()
    cv2.rectangle(overlay, (10, 10), (250, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

    # FPS
    fps_color = (0, 255, 0) if fps >= 20 else (0, 255, 255) if fps >= 10 else (0, 0, 255)
    cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)

    # Total objects
    cv2.putText(
        annotated,
        f"Objects: {total_objects}",
        (140, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    # Per-class breakdown (top 5)
    y_offset = 55
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1])[:5]:
        cv2.putText(
            annotated,
            f"  {cls}: {count}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        y_offset += 22

    return annotated


class FPSCounter:
    """Rolling FPS counter."""

    def __init__(self, window: int = 30):
        self.timestamps = deque(maxlen=window)

    def tick(self):
        self.timestamps.append(time.time())

    def get_fps(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed == 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed
