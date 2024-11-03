import os
import torch
from PIL import Image
import cv2
import numpy as np
from typing import List, Dict
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import easyocr
from collections import defaultdict

class ClipVideoAnalyzer:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 yolo_model: str = "yolov8x.pt",
                 traffic_sign_model: str = "yolov8n.pt"):
        """Initialize all models and preprocessing pipeline"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize CLIP
        try:
            model_path = self.config['models']['clip']['local_path']
            self.clip_model = CLIPModel.from_pretrained(model_path).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_path)
        except:
            print("Local CLIP model not found, downloading from hub...")
            self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            # Save locally
            self.clip_model.save_pretrained('video_analytics/models/clip')
            self.processor.save_pretrained('video_analytics/models/clip')
        
        # Initialize YOLO with SAHI
        yolo_path = os.path.join(self.config['models']['yolo']['local_path'], os.path.basename(yolo_model))
        if not os.path.exists(yolo_path):
            print("Local YOLO model not found, downloading...")
            os.makedirs(os.path.dirname(yolo_path), exist_ok=True)
            yolo_path = yolo_model
            
        self.yolo = YOLO(yolo_path)
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=yolo_path,
            confidence_threshold=0.3,
            device=self.device
        )
        
        # Initialize traffic sign detection
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

    def analyze_frame(self, frame: np.ndarray, text_queries: List[str],
                     confidence_threshold: float = 0.5) -> Dict:
        """Analyze a single frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get predictions
        result = get_sliced_prediction(
            frame_rgb,
            self.detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
        
        # Process detections with CLIP
        segments_info = []
        for pred in result.object_prediction_list:
            bbox = pred.bbox.to_xyxy()
            x1, y1, x2, y2 = map(int, bbox)
            segment = frame_rgb[y1:y2, x1:x2]
            
            inputs = self.processor(
                text=text_queries,
                images=Image.fromarray(segment),
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)
            
            segments_info.append({
                'bbox': tuple(map(int, bbox)),
                'class': text_queries[probs.argmax().item()],
                'confidence': float(probs.max())
            })
        
        # Update tracking
        current_objects = self._update_tracking(frame, result)
        
        return {
            'segments': segments_info,
            'lanes': self.detect_lanes(frame),
            'text': self.detect_text(frame),
            'signs': self.detect_traffic_signs(frame),
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
