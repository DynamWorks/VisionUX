import cv2
import numpy as np
from .stream_subscriber import StreamSubscriber, Frame
import logging
from typing import Dict, Any
import time

class ObjectDetectionSubscriber(StreamSubscriber):
    """Subscriber for object detection processing"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.enabled = True
        self.logger = logging.getLogger(__name__)
        
    def on_frame(self, frame: Frame) -> None:
        if not self.enabled:
            return
            
        try:
            # Initialize model if needed
            if self.model is None:
                from ultralytics import YOLO
                self.model = YOLO('yolov8n.pt')
                
            # Run detection
            results = self.model(frame.data, conf=self.confidence_threshold)
            
            # Process results
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0].tolist()  # Get box coordinates
                    c = box.conf.item()  # Confidence
                    cls = int(box.cls.item())  # Class
                    name = r.names[cls]  # Class name
                    
                    detections.append({
                        'bbox': b,
                        'confidence': c,
                        'class': name
                    })
                    
            # Update frame metadata
            frame.metadata = frame.metadata or {}
            frame.metadata['detections'] = detections
            
        except Exception as e:
            self.logger.error(f"Object detection error: {e}")

class EdgeDetectionSubscriber(StreamSubscriber):
    """Subscriber for edge detection processing"""
    
    def __init__(self, low_threshold: int = 100, high_threshold: int = 200, overlay_mode: bool = True):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.enabled = True
        self.overlay_mode = overlay_mode
        self.logger = logging.getLogger(__name__)
        
    def on_frame(self, frame: Frame) -> None:
        if not self.enabled:
            return
            
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame.data, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Canny edge detection
            edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold)
            
            # Convert edges to BGR and color them green
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            edges_colored[edges > 0] = [0, 255, 0]  # Color edges green
            
            # Blend with original frame
            alpha = 0.7  # Original frame weight
            beta = 0.3   # Edge overlay weight
            blended_frame = cv2.addWeighted(frame.data, alpha, edges_colored, beta, 0)
            
            # Update frame with blended result
            frame.data = blended_frame
            frame.metadata = frame.metadata or {}
            frame.metadata.update({
                'edge_detection': True,
                'edge_params': {
                    'low_threshold': self.low_threshold,
                    'high_threshold': self.high_threshold,
                    'blend_alpha': alpha,
                    'blend_beta': beta
                }
            })
            
            # Get stream manager instance and publish blended frame
            from .stream_manager import StreamManager
            stream_manager = StreamManager()
            stream_manager.publish_frame(frame)
            
        except Exception as e:
            self.logger.error(f"Edge detection error: {e}")

class MotionDetectionSubscriber(StreamSubscriber):
    """Subscriber for motion detection processing"""
    
    def __init__(self, min_area: int = 500):
        self.min_area = min_area
        self.prev_frame = None
        self.enabled = True
        self.logger = logging.getLogger(__name__)
        
    def on_frame(self, frame: Frame) -> None:
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
            
            # Dilate threshold image to fill in holes
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process motion regions
            motion_regions = []
            for contour in contours:
                if cv2.contourArea(contour) < self.min_area:
                    continue
                    
                (x, y, w, h) = cv2.boundingRect(contour)
                motion_regions.append({
                    'bbox': [x, y, x+w, y+h],
                    'area': cv2.contourArea(contour)
                })
                
            # Create visualization
            motion_frame = frame.data.copy()
            for region in motion_regions:
                x1, y1, x2, y2 = region['bbox']
                cv2.rectangle(motion_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Create new frame with motion detection
            motion_result = Frame(
                data=motion_frame,
                timestamp=time.time(),
                frame_number=frame.frame_number,
                metadata={
                    'type': 'motion_detection',
                    'original_frame': frame.frame_number,
                    'motion_regions': motion_regions
                }
            )
            
            # Get stream manager instance and publish processed frame
            from .stream_manager import StreamManager
            stream_manager = StreamManager()
            stream_manager.publish_frame(motion_result)
            
            # Update previous frame
            self.prev_frame = gray
            
        except Exception as e:
            self.logger.error(f"Motion detection error: {e}")
            
    def cleanup(self) -> None:
        """Clean up resources"""
        self.prev_frame = None
        self.enabled = False
