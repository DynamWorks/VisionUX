import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel
from typing import List, Dict
import logging

class VILAProcessor:
    """Vision-Language Analysis processor for enhanced scene understanding"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.logger = logging.getLogger(__name__)
        
    def analyze_scene(self, frame_result: Dict, image: np.ndarray = None) -> Dict:
        """
        Enhance frame analysis with VILA scene understanding
        
        Args:
            frame_result: Dictionary containing detection results
            image: Optional raw image array for additional analysis
        """
        try:
            # Extract scene elements
            detections = frame_result.get('detections', {})
            objects = [d['class'] for d in detections.get('segments', [])]
            signs = [d['class'] for d in detections.get('signs', [])]
            texts = [d['text'] for d in detections.get('text', [])]
            lanes = detections.get('lanes', [])

            # Perform additional image analysis if provided
            if image is not None:
                # Process image with CLIP
                inputs = self.processor(
                    images=Image.fromarray(image),
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
            
            # Create structured scene description
            scene_analysis = {
                'objects': self._analyze_objects(objects),
                'infrastructure': self._analyze_infrastructure(signs, lanes),
                'text_content': self._analyze_text(texts),
                'scene_type': self._determine_scene_type(objects, signs, lanes),
                'activities': self._analyze_activities(frame_result)
            }
            
            return scene_analysis
            
        except Exception as e:
            self.logger.error(f"VILA scene analysis error: {str(e)}")
            return {}
            
    def _analyze_objects(self, objects: List[str]) -> Dict:
        """Analyze detected objects and their relationships"""
        obj_counts = {}
        for obj in objects:
            obj_counts[obj] = obj_counts.get(obj, 0) + 1
            
        return {
            'counts': obj_counts,
            'primary_objects': [obj for obj, count in obj_counts.items() if count >= 2],
            'total_objects': len(objects)
        }
        
    def _analyze_infrastructure(self, signs: List[str], lanes: List) -> Dict:
        """Analyze road infrastructure elements"""
        return {
            'traffic_signs': list(set(signs)),
            'has_lanes': bool(lanes),
            'lane_count': len(lanes),
            'infrastructure_type': self._determine_infrastructure_type(signs, lanes)
        }
        
    def _analyze_text(self, texts: List[str]) -> Dict:
        """Analyze detected text content"""
        return {
            'text_elements': texts,
            'text_count': len(texts),
            'has_signage': any(len(text) > 3 for text in texts)
        }
        
    def _determine_scene_type(self, objects: List[str], signs: List[str], lanes: List) -> str:
        """Determine the type of scene based on detected elements"""
        if not objects and not signs and not lanes:
            return "unknown"
            
        # Scene type heuristics
        if len(lanes) > 2:
            return "highway"
        elif "traffic light" in signs or "stop sign" in signs:
            return "intersection"
        elif any(obj in ["car", "truck", "bus"] for obj in objects):
            if len(objects) > 5:
                return "busy_street"
            return "street"
        elif any(obj in ["person", "bicycle"] for obj in objects):
            return "urban_area"
            
        return "general_road"
        
    def _determine_infrastructure_type(self, signs: List[str], lanes: List) -> str:
        """Determine infrastructure type from road elements"""
        if not signs and not lanes:
            return "unknown"
            
        if len(lanes) > 2:
            return "major_road"
        elif signs and lanes:
            return "controlled_road"
        elif lanes:
            return "marked_road"
        elif signs:
            return "signed_area"
            
        return "basic_road"
        
    def _analyze_activities(self, frame_result: Dict) -> List[str]:
        """Analyze ongoing activities in the scene"""
        activities = []
        detections = frame_result.get('detections', {})
        
        # Movement analysis from tracking
        tracking = detections.get('tracking', {})
        if tracking.get('current', 0) > 0:
            activities.append("moving_traffic")
            
        # Object-based activity inference
        objects = [d['class'] for d in detections.get('segments', [])]
        if "person" in objects:
            activities.append("pedestrian_activity")
        if "bicycle" in objects:
            activities.append("cycling_activity")
        if any(v in objects for v in ["car", "truck", "bus"]):
            activities.append("vehicular_traffic")
            
        return activities
