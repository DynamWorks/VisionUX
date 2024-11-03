import os
import torch
from PIL import Image
import logging
import cv2
import numpy as np
from typing import List, Tuple, Dict, Set
import time
from transformers import CLIPProcessor, CLIPModel
from collections import defaultdict
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import concurrent.futures
from queue import Queue
import threading

class ProcessingQueue:
    def __init__(self, max_size=10):
        self.frame_queue = Queue(maxsize=max_size)
        self.result_queue = Queue(maxsize=max_size)
        self.is_running = False
        
    def start(self):
        self.is_running = True
        
    def stop(self):
        self.is_running = False
        
    def put_frame(self, frame):
        self.frame_queue.put(frame)
        
    def get_result(self):
        return self.result_queue.get()
        
    def put_result(self, result):
        self.result_queue.put(result)
        
    def get_frame(self):
        return self.frame_queue.get()

class ClipVideoAnalyzer:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 yolo_model: str = "yolov8x.pt",
                 traffic_sign_model: str = "yolov8n.pt"):
        """Initialize all models and preprocessing pipeline"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize CLIP
        self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Initialize YOLO with SAHI for general object detection
        self.yolo = YOLO(yolo_model)
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=yolo_model,
            confidence_threshold=0.3,
            device=self.device
        )
        
        # Initialize traffic sign detection model
        self.traffic_sign_model = YOLO(traffic_sign_model)
        
        # Initialize text recognition
        import easyocr
        self.reader = easyocr.Reader(['en'])
        
        # Initialize lane detection parameters
        self.lane_detector = cv2.createLineSegmentDetector(0)
        
        # Initialize object tracker
        self.tracker = cv2.TrackerKCF_create
        self.tracked_objects = {}
        self.object_ids = set()
        self.scene_objects = defaultdict(int)
        self.frame_objects = defaultdict(set)
        print(f"Models loaded on {self.device}")
        
    def detect_lanes(self, frame: np.ndarray) -> List[Dict]:
        """Detect lane markings in the frame"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect line segments
        lines = self.lane_detector.detect(edges)[0]
        
        if lines is None:
            return []
        
        lane_info = []
        for line in lines:
            x1, y1, x2, y2 = map(int, line[0])
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            # Filter for likely lane markings (near vertical lines)
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
        text_detections = []
        
        for bbox, text, conf in results:
            if conf > 0.5:  # Confidence threshold
                text_detections.append({
                    'text': text,
                    'confidence': conf,
                    'bbox': bbox,
                })
        
        return text_detections

    def detect_traffic_signs(self, frame: np.ndarray) -> List[Dict]:
        """Detect traffic signs in the frame"""
        results = self.traffic_sign_model(frame)[0]
        signs = []
        
        for r in results.boxes.data:
            x1, y1, x2, y2, conf, cls = r
            signs.append({
                'class': results.names[int(cls)],
                'confidence': float(conf),
                'bbox': (int(x1), int(y1), int(x2), int(y2))
            })
        
        return signs

    def parallel_process_frame(self, frame: np.ndarray, text_queries: List[str],
                             confidence_threshold: float = 0.5) -> Dict:
        """Process different aspects of frame analysis in parallel"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all tasks
            lanes_future = executor.submit(self.detect_lanes, frame)
            text_future = executor.submit(self.detect_text, frame)
            signs_future = executor.submit(self.detect_traffic_signs, frame)
            
            # Convert BGR to RGB for YOLO and CLIP
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Submit YOLO+SAHI detection
            detection_future = executor.submit(
                get_sliced_prediction,
                frame_rgb,
                self.detection_model,
                slice_height=512,
                slice_width=512,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                perform_standard_pred=True,
                postprocess_type="NMS",
                postprocess_match_metric="IOU",
                postprocess_match_threshold=0.5,
                postprocess_class_agnostic=True
            )
            
            # Get results as they complete
            lanes_info = lanes_future.result()
            text_detections = text_future.result()
            traffic_signs = signs_future.result()
            result = detection_future.result()
            
            return {
                'lanes': lanes_info,
                'text_detections': text_detections,
                'traffic_signs': traffic_signs,
                'detections': result
            }

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
        
        # Get YOLO + SAHI predictions with automatic merging
        result = get_sliced_prediction(
            frame_rgb,
            self.detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            perform_standard_pred=True,
            postprocess_type="NMS",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=0.5,
            postprocess_class_agnostic=True
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
            
            # Store segment info
            segments_info.append({
                'bbox': tuple(map(int, bbox)),
                'class': text_queries[probs.argmax().item()],
                'confidence': float(probs.max())
            })
        
        # Update object tracking using SAHI predictions
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
        
        # Add new trackers for SAHI detections
        for pred in result.object_prediction_list:
            bbox = pred.bbox.to_xyxy()
            x1, y1, x2, y2 = map(int, bbox)
            w, h = x2 - x1, y2 - y1
            
            # Check if this bbox overlaps significantly with any existing tracker
            is_new_object = True
            for obj_id in current_objects:
                success, existing_bbox = self.tracked_objects[obj_id].update(frame)
                if success:
                    ex, ey, ew, eh = existing_bbox
                    # Calculate IoU
                    intersection = max(0, min(x1+w, ex+ew) - max(x1, ex)) * \
                                 max(0, min(y1+h, ey+eh) - max(y1, ey))
                    union = w*h + ew*eh - intersection
                    if intersection/union > 0.5:  # IoU threshold
                        is_new_object = False
                        break
            
            if is_new_object:
                tracker = self.tracker()
                tracker.init(frame, (x1, y1, w, h))
                new_id = len(self.object_ids)
                self.tracked_objects[new_id] = tracker
                self.object_ids.add((x1, y1, w, h))
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
        
        # Detect additional elements
        lane_info = self.detect_lanes(frame)
        text_detections = self.detect_text(frame)
        traffic_signs = self.detect_traffic_signs(frame)
        
        results = {
            'scene_analysis': scene_results,
            'current_objects': len(current_objects),
            'new_objects': len(current_objects - self.frame_objects['previous']),
            'exited_objects': len(removed_objects),
            'total_tracked_objects': len(self.object_ids),
            'lanes': {
                'count': len(lane_info),
                'details': lane_info
            },
            'text_detections': text_detections,
            'traffic_signs': traffic_signs,
            'segments': segments_info
        }
        
        # Update frame objects history
        self.frame_objects['previous'] = current_objects
        
        return results

def process_video(video_path: str, text_queries: List[str], 
                 output_path: str = None, sample_rate: int = 1,
                 time_interval: float = 1.0, buffer_size: int = 10):
    """
    Process a video file using CLIP model with object tracking
    
    Args:
        video_path: path to input video file
        text_queries: list of text descriptions to detect
        output_path: path to save the analysis results
        sample_rate: process every nth frame
        time_interval: time interval in seconds for object counting
        
    Raises:
        ValueError: If video file cannot be opened or is invalid
        FileNotFoundError: If video file does not exist
    """
    # Validate input file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    try:
        analyzer = ClipVideoAnalyzer()
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get total frame count for progress tracking
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Processing video with {total_frames} frames at {fps} FPS")
        
        frame_count = 0
        processed_count = 0
        results = []
    
    # Create thread pool for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Create a list to store future objects
        future_results = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % sample_rate != 0:
                continue
                
            # Submit frame for parallel processing
            future = executor.submit(analyzer.parallel_process_frame, frame, text_queries)
            future_results.append((frame_count, future))
            
            # Process completed frames
            for frame_idx, future in future_results[:]:
                if future.done():
                    frame_results = future.result()
                    timestamp = (frame_idx / fps)
                    interval_idx = int(timestamp / time_interval)
                    
                    frame_results['timestamp'] = timestamp
                    frame_results['interval'] = interval_idx
                    results.append(frame_results)
                    
                    processed_count += 1
                    future_results.remove((frame_idx, future))
                    
                    # Print progress
                    progress = (frame_count * 100) / total_frames
                    print(f"\rProgress: {progress:.1f}% - Processed {processed_count} frames", end="")
            
        # Wait for remaining frames to be processed
        print("\nFinalizing processing...")
        for frame_idx, future in future_results:
            frame_results = future.result()
            timestamp = (frame_idx / fps)
            interval_idx = int(timestamp / time_interval)
            
            frame_results['timestamp'] = timestamp
            frame_results['interval'] = interval_idx
            results.append(frame_results)
            
            processed_count += 1
            print(f"\rFinishing: {processed_count} frames processed", end="")
    
    cap.release()
    
    # Save results if output path is specified
    if output_path:
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
    
    return results

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
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
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        results = process_video(
            video_path=video_path,
            text_queries=text_queries,
            output_path="clip_analysis_results.json",
            sample_rate=30  # Process every 30th frame
        )
        
    except FileNotFoundError as e:
        logging.error(f"File error: {e}")
    except ValueError as e:
        logging.error(f"Processing error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
