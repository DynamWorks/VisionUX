import os
import torch
import logging
from PIL import Image
import cv2
import numpy as np
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import easyocr
from collections import defaultdict

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

class AnalysisModule(ABC):
    """Base class for analysis modules"""
    
    @abstractmethod
    def analyze(self, frame: np.ndarray, **kwargs) -> Dict:
        """Analyze a single frame"""
        pass

class ClipModule(AnalysisModule):
    """CLIP-based analysis module"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
    def analyze(self, frame: np.ndarray, text_queries: List[str], **kwargs) -> Dict:
        inputs = self.processor(
            text=text_queries,
            images=Image.fromarray(frame),
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
            
        return {
            'matches': [
                {
                    'query': text_queries[i],
                    'confidence': float(prob)
                }
                for i, prob in enumerate(probs[0])
            ]
        }

class ObjectDetectionModule(AnalysisModule):
    """YOLO-based object detection module"""
    
    def __init__(self, model_path: str = "yolov8x.pt"):
        self.model = YOLO(model_path)
        
    def analyze(self, frame: np.ndarray, **kwargs) -> Dict:
        results = self.model(frame)[0]
        return {
            'detections': [
                {
                    'class': results.names[int(cls)],
                    'confidence': float(conf),
                    'bbox': (int(x1), int(y1), int(x2), int(y2))
                }
                for x1, y1, x2, y2, conf, cls in results.boxes.data
            ]
        }

class ClipVideoAnalyzer:
    """Modular video analyzer supporting multiple analysis types"""
    
    # Default model names
    DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
    DEFAULT_YOLO_MODEL = "yolov8x.pt"
    DEFAULT_TRAFFIC_SIGN_MODEL = "yolov8n.pt"
    
    def __init__(self, config: Optional[dict] = None, analysis_types: Optional[List[str]] = None):
        """
        Initialize models based on requested analysis types
        
        Args:
            config: Configuration dictionary
            analysis_types: List of analysis types to enable ('clip', 'object', 'signs', 'text', 'lanes')
                          If None, only CLIP analysis will be enabled
        """
        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.frame_height = None
        self.frame_width = None
        self.sahi_params = None
        self.analysis_types = analysis_types or ['clip']
        
        # Initialize default config if none provided
        self.config = config or {
            'models': {
                'clip': {
                    'name': self.DEFAULT_CLIP_MODEL,
                    'local_path': 'video_analytics/models/clip'
                },
                'yolo': {
                    'name': self.DEFAULT_YOLO_MODEL,
                    'local_path': 'video_analytics/models/yolo'
                },
                'traffic_signs': {
                    'name': self.DEFAULT_TRAFFIC_SIGN_MODEL,
                    'local_path': 'video_analytics/models/traffic_signs'
                }
            }
        }
        
        # Initialize CLIP
        try:
            model_path = self.config['models']['clip']['local_path']
            self.clip_model = CLIPModel.from_pretrained(model_path).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_path)
        except:
            print("Local CLIP model not found, downloading from hub...")
            model_name = self.config['models']['clip']['name']
            self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            # Save locally
            self.clip_model.save_pretrained('video_analytics/models/clip')
            self.processor.save_pretrained('video_analytics/models/clip')
        
        # Initialize YOLO with SAHI
        yolo_model = self.config['models']['yolo']['name']
        yolo_path = os.path.join(self.config['models']['yolo']['local_path'], os.path.basename(yolo_model))
        if not os.path.exists(yolo_path):
            print("Local YOLO model not found, downloading...")
            os.makedirs(os.path.dirname(yolo_path), exist_ok=True)
            yolo_path = yolo_model
            
        try:
            self.yolo = YOLO(yolo_path)
            self.detection_model = AutoDetectionModel.from_pretrained(
                model_type='yolov8',
                model_path=yolo_path,
                confidence_threshold=0.3,
                device=self.device
            )
        except Exception as e:
            print(f"Error initializing YOLO model: {e}")
            print("Falling back to default YOLOv8n model...")
            self.yolo = YOLO('yolov8n.pt')
            self.detection_model = AutoDetectionModel.from_pretrained(
                model_type='yolov8',
                model_path='yolov8n.pt',
                confidence_threshold=0.3,
                device=self.device
            )
        
        # Initialize traffic sign detection
        traffic_sign_model = self.config['models']['traffic_signs']['name']
        sign_path = os.path.join(self.config['models']['traffic_signs']['local_path'], os.path.basename(traffic_sign_model))
        if not os.path.exists(sign_path):
            print("Local traffic sign model not found, downloading...")
            os.makedirs(os.path.dirname(sign_path), exist_ok=True)
            sign_path = traffic_sign_model
            
        self.traffic_sign_model = YOLO(sign_path)
        
        # Initialize text recognition
        self.reader = easyocr.Reader(['en'])
        
        # Initialize lane detection
        self.lane_detector = cv2.createLineSegmentDetector(0)
        
        # Initialize tracking
        self.tracker = cv2.TrackerKCF_create
        self.tracked_objects = {}
        self.object_ids = set()
        self.scene_objects = defaultdict(int)
        self.frame_objects = defaultdict(set)

    def detect_lanes(self, frame: np.ndarray) -> List[Dict]:
        """Detect lane markings in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = self.lane_detector.detect(edges)[0]
        
        if lines is None:
            return []
        
        lane_info = []
        for line in lines:
            x1, y1, x2, y2 = map(int, line[0])
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            if 20 < abs(angle) < 80:
                lane_info.append({
                    'coordinates': ((x1, y1), (x2, y2)),
                    'angle': angle,
                    'length': np.sqrt((x2-x1)**2 + (y2-y1)**2)
                })
        
        return lane_info

    def detect_text(self, frame: np.ndarray) -> List[Dict]:
        """Detect and recognize text in the frame"""
        results = self.reader.readtext(frame)
        return [
            {'text': text, 'confidence': conf, 'bbox': bbox}
            for bbox, text, conf in results
            if conf > 0.5
        ]

    def detect_traffic_signs(self, frame: np.ndarray) -> List[Dict]:
        """Detect traffic signs in the frame"""
        results = self.traffic_sign_model(frame)[0]
        return [
            {
                'class': results.names[int(cls)],
                'confidence': float(conf),
                'bbox': (int(x1), int(y1), int(x2), int(y2))
            }
            for x1, y1, x2, y2, conf, cls in results.boxes.data
        ]

    def analyze_frame(self, frame: np.ndarray, text_queries: List[str] = None,
                     confidence_threshold: float = 0.5, scene_analysis: bool = False) -> Dict:
        """Analyze a single frame"""
        try:
            # Validate frame
            if frame is None or frame.size == 0 or len(frame.shape) != 3:
                self.logger.warning("Received invalid frame")
                return self._get_default_result()
                
            # Ensure frame is in BGR format before conversion
            if frame.shape[2] != 3:
                self.logger.warning("Invalid color channels in frame")
                return self._get_default_result()
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Validate converted frame
            if frame_rgb is None or frame_rgb.size == 0:
                self.logger.warning("Frame conversion failed")
                return self._get_default_result()
                
            # Perform frame analysis
            result = self._analyze_frame_content(frame_rgb, text_queries, confidence_threshold)
            if result is None:
                self.logger.warning("Frame analysis failed")
                return self._get_default_result()
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing frame: {e}")
            return self._get_default_result()
            
        return self._analyze_frame_content(frame_rgb, text_queries, confidence_threshold)
        
    def _get_default_result(self) -> Dict:
        """Get default result structure"""
        return {
            'detections': {
                'segments': [],
                'lanes': [],
                'text': [],
                'signs': [],
                'tracking': {'current': 0, 'total': 0}
            },
            'scene_analysis': {}
        }

    def analyze_scene(self, image: np.ndarray) -> Dict:
        """Analyze a single image for scene understanding"""
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            # Get basic detections
            result = self._analyze_frame_content(image_rgb, [], 0.3)
            
            # Enhance with VILA analysis
            vila_result = self._analyze_clip(image_rgb, [
                "urban scene", "highway", "intersection",
                "pedestrian area", "parking lot"
            ])
            
            # Add scene analysis
            result['scene_analysis'] = self.vila_processor.analyze_scene(
                result, image=image_rgb
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Scene analysis failed: {str(e)}")
            return self._get_default_result()

    def _analyze_frame_content(self, frame_rgb: np.ndarray, text_queries: List[str], 
                             confidence_threshold: float) -> Dict:
        """Analyze frame content with all enabled analysis types"""
        results = self._get_default_result()
        
        try:
            # Validate frame dimensions and content
            if frame_rgb is None or frame_rgb.size == 0:
                self.logger.error("Invalid frame content")
                return results
                
            if len(frame_rgb.shape) != 3:
                self.logger.error(f"Invalid frame shape: {frame_rgb.shape}")
                return results
                
            # Initialize frame dimensions
            if self.frame_height is None:
                self.frame_height, self.frame_width = frame_rgb.shape[:2]
                
            # Use ThreadPoolExecutor for parallel processing when multiple analysis types are requested
            with ThreadPoolExecutor(max_workers=len(self.analysis_types)) as executor:
                futures = []
                analysis_tasks = {
                    'clip': (self._analyze_clip, [frame_rgb, text_queries]),
                    'object': (self._analyze_objects, [frame_rgb]),
                    'signs': (self.detect_traffic_signs, [frame_rgb]),
                    'text': (self.detect_text, [frame_rgb]),
                    'lanes': (self.detect_lanes, [frame_rgb])
                }
                
                # Submit analysis tasks based on requested types
                for analysis_type in self.analysis_types:
                    if analysis_type in analysis_tasks:
                        func, args = analysis_tasks[analysis_type]
                        futures.append((analysis_type, executor.submit(func, *args)))
                
                # Collect results
                for analysis_type, future in futures:
                    try:
                        result = future.result()
                        if result is not None:
                            results.update(result)
                        else:
                            self.logger.warning(f"{analysis_type} analysis returned None")
                    except Exception as e:
                        self.logger.error(f"Error in {analysis_type} analysis: {str(e)}")
                        
            return results
            
        except Exception as e:
            self.logger.error(f"Frame analysis failed: {str(e)}")
            return results
        
    def _analyze_clip(self, frame_rgb: np.ndarray, text_queries: List[str]) -> Dict:
        """Perform CLIP analysis on full frame"""
        inputs = self.processor(
            text=text_queries,
            images=Image.fromarray(frame_rgb),
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
            
        return {
            'clip_analysis': [
                {
                    'query': text_queries[i],
                    'confidence': float(prob)
                }
                for i, prob in enumerate(probs[0])
            ]
        }
        
    def _analyze_objects(self, frame_rgb: np.ndarray) -> Dict:
        """Perform object detection with YOLO"""
        if not hasattr(self, 'detection_model'):
            return {'segments': []}
            
        result = get_sliced_prediction(
            frame_rgb,
            self.detection_model,
            slice_height=self.frame_height,
            slice_width=self.frame_width,
            overlap_height_ratio=0,
            overlap_width_ratio=0
        )
        
        segments = [
            {
                'bbox': pred.bbox.to_xyxy(),
                'class': pred.category.name,
                'confidence': float(pred.score.value)
            }
            for pred in result.object_prediction_list
        ]
        
        # Update tracking
        current_objects = self._update_tracking(frame_rgb, result)
        
        return {
            'segments': segments,
            'lanes': self.detect_lanes(frame_rgb),
            'text': self.detect_text(frame_rgb),
            'signs': self.detect_traffic_signs(frame_rgb),
            'tracking': {
                'current': len(current_objects),
                'total': len(self.object_ids)
            }
        }

    def _update_tracking(self, frame, detections):
        """Update object tracking"""
        current_objects = set()
        
        # Update existing trackers
        for obj_id in list(self.tracked_objects.keys()):
            success, bbox = self.tracked_objects[obj_id].update(frame)
            if success:
                current_objects.add(obj_id)
            else:
                del self.tracked_objects[obj_id]
        
        # Add new trackers
        for pred in detections.object_prediction_list:
            bbox = pred.bbox.to_xyxy()
            x1, y1, x2, y2 = map(int, bbox)
            
            if self._is_new_object(current_objects, frame, (x1, y1, x2-x1, y2-y1)):
                tracker = self.tracker()
                tracker.init(frame, (x1, y1, x2-x1, y2-y1))
                new_id = len(self.object_ids)
                self.tracked_objects[new_id] = tracker
                self.object_ids.add((x1, y1, x2-x1, y2-y1))
                current_objects.add(new_id)
        
        return current_objects

    def _is_new_object(self, current_objects, frame, new_bbox):
        """Check if bbox represents a new object"""
        x1, y1, w, h = new_bbox
        for obj_id in current_objects:
            success, (ex, ey, ew, eh) = self.tracked_objects[obj_id].update(frame)
            if success:
                intersection = max(0, min(x1+w, ex+ew) - max(x1, ex)) * \
                             max(0, min(y1+h, ey+eh) - max(y1, ey))
                union = w*h + ew*eh - intersection
                if intersection/union > 0.5:
                    return False
        return True
    def _set_sahi_params(self):
        """Set SAHI parameters based on frame resolution"""
        # For HD resolution (1920x1080) or lower
        if self.frame_width <= 1920:
            self.sahi_params = {
                'slice_height': 384,
                'slice_width': 384,
                'overlap_height_ratio': 0.2,
                'overlap_width_ratio': 0.2
            }
        # For 2K resolution
        elif self.frame_width <= 2560:
            self.sahi_params = {
                'slice_height': 512,
                'slice_width': 512,
                'overlap_height_ratio': 0.3,
                'overlap_width_ratio': 0.3
            }
        # For 4K resolution
        else:
            self.sahi_params = {
                'slice_height': 640,
                'slice_width': 640,
                'overlap_height_ratio': 0.4,
                'overlap_width_ratio': 0.4
            }
