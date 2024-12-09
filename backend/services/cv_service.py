import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional, List
from ultralytics import YOLO
from ..utils.video_streaming.stream_subscriber import StreamSubscriber, Frame

class CVService:
    """Service for computer vision processing"""
    
    def __init__(self, model_path: str = None):
        self.logger = logging.getLogger(__name__)
        self.object_detection_model = None
        # Use models directory from ContentManager
        from ..content_manager import ContentManager
        content_manager = ContentManager()
        self.model_path = model_path or str(content_manager.models_dir / 'yolov8n.pt')
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
        self.is_initialized = False
        
    def detect_objects(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> Dict:
        """Detect objects in frame"""
        try:
            if not isinstance(frame, np.ndarray):
                raise ValueError("Invalid frame format")
                
            if frame.size == 0 or len(frame.shape) != 3:
                raise ValueError("Invalid frame dimensions")
                
            if self.object_detection_model is None:
                try:
                    self.object_detection_model = YOLO(self.model_path)
                    self.is_initialized = True
                except Exception as e:
                    self.logger.error(f"Failed to load YOLO model: {e}")
                    return {'error': f"Model initialization failed: {str(e)}"}
                
            results = self.object_detection_model(frame, conf=confidence_threshold)
            
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0].tolist()
                    c = box.conf.item()
                    cls = int(box.cls.item())
                    name = r.names[cls]
                    
                    detections.append({
                        'bbox': b,
                        'confidence': c,
                        'class': name
                    })
                    
            return {
                'detections': detections,
                'timestamp': frame.timestamp if hasattr(frame, 'timestamp') else None
            }
            
        except Exception as e:
            self.logger.error(f"Object detection error: {e}")
            return {'error': str(e)}

    def detect_edges(self, frame: np.ndarray) -> Dict:
        """Detect edges in frame"""
        try:
            if not isinstance(frame, np.ndarray):
                raise ValueError("Invalid frame format")
                
            if frame.size == 0 or len(frame.shape) != 3:
                raise ValueError("Invalid frame dimensions")
                
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur with configurable parameters
            blur_size = self.edge_detection_params['blur_size']
            if blur_size % 2 == 0:  # Ensure odd kernel size
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
            
            # Convert edges to BGR and color them green
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            edges_colored[edges > 0] = [255, 0, 255]  # Bright magenta
            
            # Blend with original frame if overlay mode
            if self.edge_detection_params['overlay_mode']:
                alpha = 0.7  # Original frame weight
                beta = 0.3   # Edge overlay weight
                result = cv2.addWeighted(frame, alpha, edges_colored, beta, 0)
            else:
                result = edges_colored
                
            return {
                'frame': result,
                'edges': edges,
                'params': self.edge_detection_params.copy()
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
