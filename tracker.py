from deep_sort_realtime.deepsort_tracker import DeepSort


class ObjectTracker:
    """
    Object tracker using Deep SORT for persistent identity tracking
    across video frames.
    """

    def __init__(self, max_age: int = 30, n_init: int = 3, max_iou_distance: float = 0.7):
        """
        Initialize the Deep SORT tracker.

        Args:
            max_age: Maximum frames to keep a track alive without detections.
            n_init: Number of consecutive detections before a track is confirmed.
            max_iou_distance: Maximum IOU distance for matching.
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
        )

    def update(self, detections: list[dict], frame) -> list[dict]:
        """
        Update tracker with new detections from the current frame.

        Args:
            detections: List of detection dicts from ObjectDetector.detect().
            frame: Current BGR frame (numpy array).

        Returns:
            List of tracked objects, each dict with keys:
              - 'track_id': Persistent tracking ID
              - 'bbox': [x1, y1, x2, y2] bounding box
              - 'class_name': Object class name
              - 'confidence': Detection confidence
              - 'is_confirmed': Whether the track is confirmed
        """
        if not detections:
            # Still update tracker to age out old tracks
            tracks = self.tracker.update_tracks([], frame=frame)
        else:
            # Convert detections to Deep SORT format: ([x1, y1, w, h], confidence, class_name)
            ds_detections = []
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                w = x2 - x1
                h = y2 - y1
                if w > 0 and h > 0:
                    ds_detections.append(
                        ([float(x1), float(y1), float(w), float(h)], float(det["confidence"]), str(det["class_name"]))
                    )

            tracks = self.tracker.update_tracks(ds_detections, frame=frame)

        tracked_objects = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            try:
                track_id = int(track.track_id)
            except (ValueError, TypeError):
                track_id = abs(hash(str(track.track_id))) % 10000
            ltrb = track.to_ltrb()  # [left, top, right, bottom]

            # Get the detection class from the track
            det_class = track.det_class if track.det_class else "unknown"
            det_conf = track.det_conf if track.det_conf else 0.0

            tracked_objects.append(
                {
                    "track_id": track_id,
                    "bbox": [float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3])],
                    "class_name": det_class,
                    "confidence": float(det_conf) if det_conf else 0.0,
                    "is_confirmed": True,
                }
            )

        return tracked_objects

    def reset(self):
        """Reset the tracker (clear all tracks)."""
        self.tracker = DeepSort(
            max_age=self.tracker.max_age,
        )
