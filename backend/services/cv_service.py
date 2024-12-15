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
from ..utils.video_streaming.stream_subscriber import StreamSubscriber, Frame

class CVService:
    """Service for computer vision processing"""
    
    def __init__(self):
        """Initialize CV service"""
        self.logger = logging.getLogger(__name__)
        self._initialized = False
        self._lock = threading.Lock()
        self._model_ready = threading.Event()
        
        # Load COCO class names
        self.classNames = []
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        classFile = models_dir / 'coco.names'
        if not classFile.exists():
            self.logger.error("coco.names file not found in models directory")
            raise FileNotFoundError("Required coco.names file not found in models directory")

        # Load class names
        with open(classFile, 'rt') as f:
            self.classNames = f.read().rstrip('\n').split('\n')
        
        # Initialize TensorFlow model
        self._init_model()

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
            'min_object_area': 500
        }

        # Motion detection parameters
        self.motion_detection_params = {
            'min_area': 500,
            'prev_frame': None,
            'threshold': 25,
            'dilate_iterations': 2
        }
        

    def _init_model(self):
        """Initialize OpenCV DNN model"""
        try:
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            configPath = models_dir / 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
            weightsPath = models_dir / 'frozen_inference_graph.pb'
            
            if not configPath.exists() or not weightsPath.exists():
                self.logger.error("Model files not found in config_files directory")
                raise FileNotFoundError("Required model files not found in config_files directory")
            
            # Initialize DNN model
            self.net = cv2.dnn_DetectionModel(str(weightsPath), str(configPath))
            self.net.setInputSize(320, 320)
            self.net.setInputScale(1.0/127.5)
            self.net.setInputMean((127.5, 127.5, 127.5))
            self.net.setInputSwapRB(True)
            
            self._model_ready.set()
            self._initialized = True
            self.logger.info("OpenCV DNN model initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenCV DNN model: {e}")
            self._initialized = False
            self.net = None

    def _load_model(self):
        """Initialize model if not already loaded"""
        if not self._initialized:
            self._init_model()

                
    def detect_objects(self, frame: np.ndarray) -> Dict:
        """Detect objects in frame"""
        try:
            if not isinstance(frame, np.ndarray):
                raise ValueError("Input frame must be a numpy array")
                
            # Initialize model if needed
            if not self._initialized:
                self._init_model()

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # # Prepare input for TensorFlow model
            # input_tensor = tf.convert_to_tensor(frame[np.newaxis, ...])
            
            # Run detection using OpenCV DNN
            classIds, confidences, boxes = self.net.detect(frame, confThreshold=0.5)
            

            detections = []
            
            # Process detections
            if len(classIds) > 0:
                classIds = classIds.flatten()
                confidences = confidences.flatten()
            
            for i in range(len(classIds)):
                confidence = float(confidences[i])
                if confidence > 0.5:  # Confidence threshold
                    # Get bounding box coordinates
                    box = boxes[i]
                    xmin, ymin = int(box[0]), int(box[1])
                    xmax, ymax = int(xmin + box[2]), int(ymin + box[3])
                    
                    # Get class name from loaded classes
                    class_id = int(classIds[i])
                    class_name = self.classNames[class_id - 1] if 0 <= class_id - 1 < len(self.classNames) else f"class_{class_id}"
                    
                    # Create detection entry
                    bbox = [xmin, ymin, xmax, ymax]
                    track_id = i + 1
                    # confidence already set above
                    
                    detection = {
                        'bbox': bbox,
                        'confidence': confidence,
                        'class': class_name
                    }
                    detections.append(detection)
            
            return {
                'detections': detections,
                'timestamp': time.time()
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
            
            # Process contours
            result = frame.copy()
            
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
