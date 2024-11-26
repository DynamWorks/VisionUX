import cv2
import numpy as np
from ..frame_processor import FrameProcessor
import logging
from typing import Dict, Any
import time

class ObjectDetectionSubscriber(FrameProcessor):
    """Subscriber for object detection processing"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.enabled = True
        
    def process_frame(self, frame_data: Dict[str, Any]):
        if not self.enabled:
            return
            
        try:
            frame = frame_data['frame']
            if frame is None:
                return
                
            # Initialize model if needed
            if self.model is None:
                from ultralytics import YOLO
                self.model = YOLO('yolov8n.pt')
                
            # Run detection
            results = self.model(frame, conf=self.confidence_threshold)
            
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
                    
            # Add detections to frame metadata
            frame_data['metadata'] = frame_data.get('metadata', {})
            frame_data['metadata']['detections'] = detections
            
        except Exception as e:
            self.logger.error(f"Object detection error: {e}")

class EdgeDetectionSubscriber(FrameProcessor):
    """Subscriber for edge detection processing"""
    
    def __init__(self, low_threshold: int = 100, high_threshold: int = 200):
        super().__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.enabled = True
        
    def process_frame(self, frame_data: Dict[str, Any]):
        if not self.enabled:
            return
            
        try:
            frame = frame_data['frame']
            if frame is None:
                return
                
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Canny edge detection
            edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold)
            
            # Add edges to frame metadata
            frame_data['metadata'] = frame_data.get('metadata', {})
            frame_data['metadata']['edges'] = edges
            
        except Exception as e:
            self.logger.error(f"Edge detection error: {e}")

class MotionDetectionSubscriber(FrameProcessor):
    """Subscriber for motion detection processing"""
    
    def __init__(self, min_area: int = 500):
        super().__init__()
        self.min_area = min_area
        self.prev_frame = None
        self.enabled = True
        
    def process_frame(self, frame_data: Dict[str, Any]):
        if not self.enabled:
            return
            
        try:
            frame = frame_data['frame']
            if frame is None:
                return
                
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
                
            # Update frame metadata
            frame_data['metadata'] = frame_data.get('metadata', {})
            frame_data['metadata']['motion'] = {
                'regions': motion_regions,
                'threshold': thresh
            }
            
            # Update previous frame
            self.prev_frame = gray
            
        except Exception as e:
            self.logger.error(f"Motion detection error: {e}")
