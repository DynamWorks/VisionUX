import cv2
import numpy as np
from typing import Dict, Any
from .stream_subscriber import StreamSubscriber, Frame

class EdgeDetectionSubscriber(StreamSubscriber):
    """Subscriber for edge detection processing"""
    
    def __init__(self):
        self.enabled = False
        self.params = {
            'low_threshold': 100,
            'high_threshold': 200,
            'kernel_size': 5
        }

    def on_frame(self, frame: Frame) -> None:
        """Process frame for edge detection"""
        if not self.enabled:
            return

        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame.data, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(
                gray, 
                (self.params['kernel_size'], self.params['kernel_size']), 
                0
            )
            
            # Detect edges
            edges = cv2.Canny(
                blurred,
                self.params['low_threshold'],
                self.params['high_threshold']
            )
            
            # Convert back to BGR
            frame.data = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            frame.metadata['edge_detection'] = True
            
        except Exception as e:
            print(f"Edge detection error: {e}")

class MotionDetectionSubscriber(StreamSubscriber):
    """Subscriber for motion detection processing"""
    
    def __init__(self):
        self.enabled = False
        self.prev_frame = None
        self.min_area = 500

    def on_frame(self, frame: Frame) -> None:
        """Process frame for motion detection"""
        if not self.enabled:
            return

        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame.data, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if self.prev_frame is None:
                self.prev_frame = gray
                return

            # Compute difference
            frame_delta = cv2.absdiff(self.prev_frame, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

            # Dilate threshold image
            thresh = cv2.dilate(thresh, None, iterations=2)

            # Find contours
            contours, _ = cv2.findContours(
                thresh.copy(),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # Draw motion regions
            for contour in contours:
                if cv2.contourArea(contour) < self.min_area:
                    continue

                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(
                    frame.data,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

            self.prev_frame = gray
            frame.metadata['motion_detection'] = True

        except Exception as e:
            print(f"Motion detection error: {e}")

class ObjectDetectionSubscriber(StreamSubscriber):
    """Subscriber for object detection processing"""
    
    def __init__(self):
        self.enabled = False
        self.model = None
        self.confidence_threshold = 0.5

    def on_frame(self, frame: Frame) -> None:
        """Process frame for object detection"""
        if not self.enabled:
            return

        try:
            if self.model is None:
                from ultralytics import YOLO
                self.model = YOLO('yolov8n.pt')

            # Run inference
            results = self.model(frame.data, conf=self.confidence_threshold)

            # Draw detections
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0].tolist()  # get box coordinates
                    c = box.conf.item()
                    cls = int(box.cls.item())
                    name = r.names[cls]

                    # Draw bounding box
                    cv2.rectangle(
                        frame.data,
                        (int(b[0]), int(b[1])),
                        (int(b[2]), int(b[3])),
                        (0, 255, 0),
                        2
                    )

                    # Add label
                    cv2.putText(
                        frame.data,
                        f'{name} {c:.2f}',
                        (int(b[0]), int(b[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

            frame.metadata['object_detection'] = True

        except Exception as e:
            print(f"Object detection error: {e}")
