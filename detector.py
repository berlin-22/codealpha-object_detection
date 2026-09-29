from ultralytics import YOLO


class ObjectDetector:
    """
    Object detection using YOLOv8 pre-trained on COCO dataset.
    Detects 80 object classes including person, car, dog, etc.
    """

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.5):
        """
        Initialize the YOLOv8 detector.

        Args:
            model_name: YOLOv8 model variant ('yolov8n.pt' for nano, 'yolov8s.pt' for small).
            confidence: Minimum confidence threshold for detections.
        """
        self.model = YOLO(model_name)
        self.confidence = confidence

    def detect(self, frame, confidence: float = None) -> list[dict]:
        """
        Run object detection on a single frame.

        Args:
            frame: BGR image (numpy array from OpenCV).
            confidence: Override confidence threshold (optional).

        Returns:
            List of detections, each dict with keys:
              - 'bbox': [x1, y1, x2, y2] bounding box coordinates
              - 'class_name': detected class name (e.g., 'person')
              - 'class_id': numeric class ID
              - 'confidence': detection confidence score
        """
        conf = confidence if confidence is not None else self.confidence

        results = self.model(frame, conf=conf, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf_score = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[class_id]

                detections.append(
                    {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "class_name": class_name,
                        "class_id": class_id,
                        "confidence": conf_score,
                    }
                )

        return detections

    def get_class_names(self) -> dict:
        """Return the COCO class names mapping."""
        return self.model.names
