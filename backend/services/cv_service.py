import cv2
import numpy as np
import logging
import time
import threading
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from collections import defaultdict
from shapely.geometry import Polygon
from shapely.geometry.point import Point
import mediapipe as mp
from ..utils.video_streaming.stream_subscriber import StreamSubscriber, Frame

class CVService:
    """Service for computer vision processing"""
    
    def __init__(self):
        """Initialize CV service"""
        self.logger = logging.getLogger(__name__)
        self._initialized = False
        self._lock = threading.Lock()
        self._model_ready = threading.Event()
        
        # Initialize YOLO
        from ultralytics import YOLO
        self.detector = YOLO('yolov8n.pt')  # Load the YOLOv8 nano model

                # Initialize tracking components with thread safety
        self.track_history = defaultdict(list)
        self.next_object_id = 0
        self.tracked_objects = {}
        
        # Initialize trackers
        self.trackers = {}
        if int(cv2.__version__.split('.')[0]) >= 4:
            self.multi_tracker = cv2.legacy.MultiTracker_create()
        else:
            self.multi_tracker = cv2.MultiTracker_create()
        self.tracking_history = {}

        # Initialize counting regions
        self.counting_regions = [{
            "name": "Full Frame Region",
            "polygon": None,  # Will be set based on frame dimensions
            "counts": defaultdict(int),
            "total_counts": defaultdict(int)
        }]

        # Edge detection parameters
        self.edge_detection_params = {
            'low_threshold': 100,
            'high_threshold': 200,
            'overlay_mode': False,
            'blur_size': 5,
            'blur_sigma': 0,
            'track_objects': True,  # Enable object tracking
            'min_object_area': 500  # Minimum contour area to track
        }

        # Motion detection parameters
        self.motion_detection_params = {
            'min_area': 500,
            'prev_frame': None,
            'threshold': 25,
            'dilate_iterations': 2
        }
        

    def _load_model(self):
        """Initialize MediaPipe detector"""
        try:
            self._model_ready.set()
            self._initialized = True
            self.logger.info("MediaPipe detector initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize MediaPipe detector: {e}")
            self._initialized = False
            

                
    def detect_objects(self, frame: np.ndarray) -> Dict:
        """Detect objects in frame"""
        try:
            if not isinstance(frame, np.ndarray):
                raise ValueError("Input frame must be a numpy array")
                
            # Initialize model if needed
            if not self.model:
                # Start model loading in background thread
                self._load_model_thread = threading.Thread(target=self._load_model)
                self._load_model_thread.daemon = True
                self._load_model_thread.start()

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with YOLO
            results = self.detector(frame)[0]  # Get detection results
            
            # Initialize counting region if needed
            if self.counting_regions[0]["polygon"] is None:
                height, width = frame.shape[:2]
                self.counting_regions[0]["polygon"] = Polygon([
                    (0, 0), (width, 0), (width, height), (0, height)
                ])

            detections = []
            if len(results.boxes) > 0:
                for i, detection in enumerate(results.boxes):
                    box = detection.xyxy[0].cpu().numpy()  # Get box coordinates
                    conf = float(detection.conf[0].cpu().numpy())  # Get confidence
                    cls = int(detection.cls[0].cpu().numpy())  # Get class
                    class_name = results.names[cls]  # Get class name
                    
                    # Convert coordinates to integers
                    xmin, ymin, xmax, ymax = map(int, box)
                    
                    # Create detection entry
                    bbox = [xmin, ymin, xmax, ymax]
                    track_id = i + 1
                    confidence = conf
                    
                    # Calculate center point
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    
                    # Update tracking history
                    track = self.track_history[track_id]
                    track.append((float(center_x), float(center_y)))
                    if len(track) > 30:  # Keep last 30 points
                        track.pop(0)
                    
                    # Check regions and update counts
                    for region in self.counting_regions:
                        if region["polygon"].contains(Point(center_x, center_y)):
                            region["counts"][class_name] += 1
                            region["total_counts"][class_name] += 1
                    
                    detection = {
                        'bbox': bbox,
                        'confidence': confidence,  # Use the confidence from detection.score[0]
                        'class': class_name,
                        'track_id': track_id,
                        'track_history': track.copy()
                    }
                    detections.append(detection)
            
            return {
                'detections': detections,
                'timestamp': time.time(),
                'regions': [{
                    'name': region['name'],
                    'counts': dict(region['counts']),
                    'total_counts': dict(region['total_counts'])
                } for region in self.counting_regions]
            }
            
        except Exception as e:
            self.logger.error(f"Object detection error: {e}")
            return {'error': str(e)}

    def detect_edges(self, frame: np.ndarray) -> Dict:
        """Detect edges and track objects in frame"""
        try:
            if not isinstance(frame, np.ndarray):
                raise ValueError("Invalid frame format")
                
            if frame.size == 0 or len(frame.shape) != 3:
                raise ValueError("Invalid frame dimensions")
                
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blur_size = self.edge_detection_params['blur_size']
            if blur_size % 2 == 0:
                blur_size += 1
            blurred = cv2.GaussianBlur(
                gray, 
                (blur_size, blur_size), 
                self.edge_detection_params['blur_sigma']
            )
            
            # Apply Canny edge detection
            edges = cv2.Canny(
                blurred,
                self.edge_detection_params['low_threshold'],
                self.edge_detection_params['high_threshold']
            )
            
            # Find contours for object detection
            contours, _ = cv2.findContours(
                edges,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Process contours and prepare for async tracking
            tracked_objects = []
            result = frame.copy()
            
            if self.edge_detection_params['track_objects']:
                import threading
                
                def track_object(contour, frame, obj_id):
                    area = cv2.contourArea(contour)
                    if area < self.edge_detection_params['min_object_area']:
                        return None
                        
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Initialize tracker based on OpenCV version
                    tracker_type = 'KCF'  # Using KCF tracker for better performance
                    if int(cv2.__version__.split('.')[1]) < 3:
                        tracker = cv2.Tracker_create(tracker_type)
                    else:
                        tracker = cv2.TrackerKCF_create()
                    
                    success = tracker.init(frame, (x, y, w, h))
                    if success:
                        return {
                            'id': obj_id,
                            'tracker': tracker,
                            'bbox': [x, y, w, h],
                            'center': (x + w//2, y + h//2),
                            'tracking_info': {
                                'type': tracker_type,
                                'confidence': 1.0,
                                'frames_tracked': 0,
                                'last_update': time.time()
                            }
                        }
                    return None
                
                # Start tracking threads
                tracking_threads = []
                tracking_results = []
                
                for contour in contours:
                    thread = threading.Thread(
                        target=lambda c=contour: tracking_results.append(
                            track_object(c, frame.copy(), self.next_object_id)
                        )
                    )
                    tracking_threads.append(thread)
                    thread.start()
                    self.next_object_id += 1
                
                # Wait for all tracking threads to complete
                for thread in tracking_threads:
                    thread.join()
                
                # Process tracking results
                for track_result in tracking_results:
                    if track_result:
                        self.trackers[track_result['id']] = track_result['tracker']
                        tracked_objects.append({
                            'id': track_result['id'],
                            'bbox': track_result['bbox'],
                            'center': track_result['center']
                        })
                        
                        # Draw tracking boxes with IDs
                        x, y, w, h = track_result['bbox']
                        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        # Draw ID text with better visibility
                        text = f"ID: {track_result['id']}"
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        # Draw background rectangle for text
                        cv2.rectangle(result, (x, y-25), (x + text_size[0], y), (0, 255, 0), -1)
                        # Draw text
                        cv2.putText(result, text, (x, y-5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                        # Save tracking data
                        from backend.services.tracking_analysis import TrackingAnalysis
                        tracking_analyzer = TrackingAnalysis()
                        tracking_path = tracking_analyzer.save_tracking_data(
                            str(Path(frame.filename).stem) if hasattr(frame, 'filename') else 'edge_detection',
                            tracked_objects
                        )
            
            # Draw edges
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            edges_colored[edges > 0] = [255, 0, 255]  # Bright magenta
            
            if self.edge_detection_params['overlay_mode']:
                alpha = 0.7
                beta = 0.3
                result = cv2.addWeighted(result, alpha, edges_colored, beta, 0)
            else:
                result = edges_colored
                
            return {
                'frame': result,
                'edges': edges,
                'tracked_objects': tracked_objects,
                'params': self.edge_detection_params.copy(),
                'tracking_analysis': tracking_path if 'tracking_path' in locals() else None
            }
            
        except Exception as e:
            self.logger.error(f"Edge detection error: {e}")
            return {'error': str(e)}

    def detect_motion(self, frame: np.ndarray) -> Dict:
        """Detect motion in frame"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if self.motion_detection_params['prev_frame'] is None:
                self.motion_detection_params['prev_frame'] = gray
                return {'motion_regions': [], 'is_first_frame': True}
                
            # Compute difference
            frame_delta = cv2.absdiff(
                self.motion_detection_params['prev_frame'],
                gray
            )
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            
            # Dilate threshold image
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(
                thresh.copy(),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Process motion regions
            motion_regions = []
            for contour in contours:
                if cv2.contourArea(contour) < self.motion_detection_params['min_area']:
                    continue
                    
                (x, y, w, h) = cv2.boundingRect(contour)
                motion_regions.append({
                    'bbox': [x, y, x+w, y+h],
                    'area': cv2.contourArea(contour)
                })
                
            # Update previous frame
            self.motion_detection_params['prev_frame'] = gray
            
            return {
                'motion_regions': motion_regions,
                'frame_delta': frame_delta,
                'threshold': thresh
            }
            
        except Exception as e:
            self.logger.error(f"Motion detection error: {e}")
            return {'error': str(e)}

    def set_edge_detection_params(self, params: Dict) -> None:
        """Update edge detection parameters"""
        self.edge_detection_params.update(params)

    def set_motion_detection_params(self, params: Dict) -> None:
        """Update motion detection parameters"""
        self.motion_detection_params.update(params)

    def reset(self) -> None:
        """Reset service state"""
        self.cleanup()
        self.edge_detection_params = {
            'low_threshold': 100,
            'high_threshold': 200,
            'overlay_mode': True,
            'blur_size': 5,
            'blur_sigma': 0
        }
        self.motion_detection_params = {
            'min_area': 500,
            'prev_frame': None,
            'threshold': 25,
            'dilate_iterations': 2
        }
        
    def cleanup(self) -> None:
        """Clean up resources"""
        if self.object_detection_model is not None:
            try:
                del self.object_detection_model
                self.object_detection_model = None
                self.is_initialized = False
            except Exception as e:
                self.logger.error(f"Error cleaning up model: {e}")
        
        if self.motion_detection_params['prev_frame'] is not None:
            self.motion_detection_params['prev_frame'] = None
