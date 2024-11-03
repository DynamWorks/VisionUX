import torch
from PIL import Image
import cv2
import numpy as np
from typing import List, Tuple, Dict, Set
import time
from transformers import CLIPProcessor, CLIPModel
from collections import defaultdict
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

class ClipVideoAnalyzer:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 yolo_model: str = "yolov8x.pt"):
        """Initialize CLIP model, YOLO, and preprocessing pipeline"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize CLIP
        self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Initialize YOLO with SAHI
        self.yolo = YOLO(yolo_model)
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=yolo_model,
            confidence_threshold=0.3,
            device=self.device
        )
        
        # Initialize object tracker
        self.tracker = cv2.TrackerKCF_create
        self.tracked_objects = {}
        self.object_ids = set()
        self.scene_objects = defaultdict(int)
        self.frame_objects = defaultdict(set)
        print(f"Models loaded on {self.device}")
        
    def analyze_frame(self, frame: np.ndarray, text_queries: List[str], 
                     confidence_threshold: float = 0.5) -> Dict:
        """
        Analyze a single frame using SAM2 for segmentation, CLIP for identification,
        and OpenCV for tracking
        
        Args:
            frame: numpy array of the frame (BGR format from OpenCV)
            text_queries: list of text descriptions to match against
            confidence_threshold: minimum confidence score to consider
            
        Returns:
            Dictionary containing scene analysis and object tracking results
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get YOLO + SAHI predictions
        result = get_sliced_prediction(
            frame_rgb,
            self.detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
        
        # Process each detection with CLIP
        segments_info = []
        for pred in result.object_prediction_list:
            bbox = pred.bbox.to_xyxy()
            # Extract segment
            x1, y1, x2, y2 = map(int, bbox)
            segment = frame_rgb[y1:y2, x1:x2]
            segment_pil = Image.fromarray(segment)
        
            # Process segment with CLIP
            inputs = self.processor(
                text=text_queries,
                images=segment_pil,
                return_tensors="pt",
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # Get model predictions
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Get segment bbox
            segment_bbox = cv2.boundingRect(mask.astype(np.uint8))
            
            # Store segment info
            segments_info.append({
                'bbox': segment_bbox,
                'class': text_queries[probs.argmax().item()],
                'confidence': float(probs.max())
            })
        
        # Update object tracking
        current_objects = set()
        removed_objects = set()
        
        # Update existing trackers
        for obj_id in list(self.tracked_objects.keys()):
            success, bbox = self.tracked_objects[obj_id].update(frame)
            if success:
                current_objects.add(obj_id)
            else:
                removed_objects.add(obj_id)
                del self.tracked_objects[obj_id]
        
        # Detect new objects (using simple difference detection for demo)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Add new trackers for detected objects
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                if (x, y, w, h) not in self.object_ids:
                    tracker = self.tracker()
                    tracker.init(frame, (x, y, w, h))
                    new_id = len(self.object_ids)
                    self.tracked_objects[new_id] = tracker
                    self.object_ids.add((x, y, w, h))
                    current_objects.add(new_id)
        
        # Convert scene analysis to dictionary
        scene_results = {
            query: float(score) 
            for query, score in zip(text_queries, probs[0].cpu().numpy())
            if float(score) >= confidence_threshold
        }
        
        # Update scene statistics
        for obj_id in current_objects:
            self.scene_objects[obj_id] += 1
        
        results = {
            'scene_analysis': scene_results,
            'current_objects': len(current_objects),
            'new_objects': len(current_objects - self.frame_objects['previous']),
            'exited_objects': len(removed_objects),
            'total_tracked_objects': len(self.object_ids)
        }
        
        # Update frame objects history
        self.frame_objects['previous'] = current_objects
        
        return results

def process_video(video_path: str, text_queries: List[str], 
                 output_path: str = None, sample_rate: int = 1,
                 time_interval: float = 1.0):
    """
    Process a video file using CLIP model with object tracking
    
    Args:
        video_path: path to input video file
        text_queries: list of text descriptions to detect
        output_path: path to save the analysis results
        sample_rate: process every nth frame
        time_interval: time interval in seconds for object counting
    """
    analyzer = ClipVideoAnalyzer()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frame_count = 0
    results = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % sample_rate != 0:
            continue
            
        # Analyze frame
        frame_results = analyzer.analyze_frame(frame, text_queries)
        
        # Add timestamp and interval statistics
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # convert to seconds
        interval_idx = int(timestamp / time_interval)
        
        frame_results['timestamp'] = timestamp
        frame_results['interval'] = interval_idx
        results.append(frame_results)
        
        print(f"Processed frame {frame_count} at {timestamp:.2f}s:")
        print(f"Scene analysis: {frame_results['scene_analysis']}")
        print(f"Objects in frame: {frame_results['current_objects']}")
        print(f"New objects: {frame_results['new_objects']}")
        print(f"Exited objects: {frame_results['exited_objects']}")
    
    cap.release()
    
    # Save results if output path is specified
    if output_path:
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
    
    return results

if __name__ == "__main__":
    # Example usage
    video_path = "sample_video.mp4"
    text_queries = [
        "person walking",
        "car driving",
        "traffic jam",
        "bicycle",
        "pedestrian crossing",
        "traffic light",
        "car", "truck", "bus", "motorcycle", "vehicle"
    ]
    
    results = process_video(
        video_path=video_path,
        text_queries=text_queries,
        output_path="clip_analysis_results.json",
        sample_rate=3  # Process every 30th frame
    )
