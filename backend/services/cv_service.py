import cv2
import numpy as np
import logging
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
from collections import defaultdict
from shapely.geometry import Polygon
from shapely.geometry.point import Point
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
        
        # Create models directory if it doesn't exist
        content_manager.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Default to YOLOv8n model
        self.model_path = model_path or str(content_manager.models_dir / 'yolov8n.pt')
        
        # Download model if it doesn't exist
        if not Path(self.model_path).exists():
            try:
                from ultralytics import YOLO
                YOLO('yolov8n').download()
            except Exception as e:
                self.logger.error(f"Failed to download YOLO model: {e}")
                raise
        
        # Initialize trackers dictionary
        self.trackers = {}
        self.tracked_objects = {}
        self.next_object_id = 0
        # Initialize multi-tracker based on OpenCV version
        if int(cv2.__version__.split('.')[0]) >= 4:
            self.multi_tracker = cv2.legacy.MultiTracker_create()
        else:
            self.multi_tracker = cv2.MultiTracker_create()
        self.tracking_history = {}
        
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
        self.motion_detection_params = {
            'min_area': 500,
            'prev_frame': None,
            'threshold': 25,
            'dilate_iterations': 2
        }
        self.is_initialized = False
        
    def __init__(self, model_path: str = None):
        self.logger = logging.getLogger(__name__)
        self.object_detection_model = None
        # Use models directory from ContentManager
        from ..content_manager import ContentManager
        content_manager = ContentManager()
        
        # Create models directory if it doesn't exist
        content_manager.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Default to YOLOv8n model
        self.model_path = model_path or str(content_manager.models_dir / 'yolov8n.pt')
        
        # Download model if it doesn't exist
        if not Path(self.model_path).exists():
            try:
                from ultralytics import YOLO
                YOLO('yolov8n').download()
            except Exception as e:
                self.logger.error(f"Failed to download YOLO model: {e}")
                raise
        
        # Initialize tracking components
        self.track_history = defaultdict(list)
        self.next_object_id = 0
        self.tracked_objects = {}
        
        # Initialize counting regions
        self.counting_regions = [{
            "name": "Full Frame Region",
            "polygon": None,  # Will be set based on frame dimensions
            "counts": defaultdict(int),
            "total_counts": defaultdict(int)
        }]
        
        self.is_initialized = False
                try:
                    import torch
                    import gc
                    
                    # Force garbage collection
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    # Configure PyTorch to handle segfaults more gracefully
                    torch.multiprocessing.set_start_method('spawn', force=True)
                    
                    # Set memory management options
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = True
                    
                    # Load model with reduced memory footprint
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    if device == 'cuda':
                        torch.cuda.set_per_process_memory_fraction(0.7)  # Limit GPU memory usage
                    
                    # Load model in eval mode directly
                    self.object_detection_model = YOLO(self.model_path)
                    self.object_detection_model.to(device).eval()
                    
                    # Clear unnecessary memory
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                    
                    self.is_initialized = True
                    self.logger.info(f"Successfully loaded YOLO model on {device} with memory optimizations")
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
