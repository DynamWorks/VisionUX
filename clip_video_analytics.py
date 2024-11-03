import torch
from PIL import Image
import cv2
import numpy as np
from typing import List, Tuple, Dict
import time
from transformers import CLIPProcessor, CLIPModel

class ClipVideoAnalyzer:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """Initialize CLIP model and preprocessing pipeline"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        print(f"Model loaded on {self.device}")
        
    def analyze_frame(self, frame: np.ndarray, text_queries: List[str], 
                     confidence_threshold: float = 0.5) -> Dict[str, float]:
        """
        Analyze a single frame using CLIP model
        
        Args:
            frame: numpy array of the frame (BGR format from OpenCV)
            text_queries: list of text descriptions to match against
            confidence_threshold: minimum confidence score to consider
            
        Returns:
            Dictionary of text queries and their confidence scores
        """
        # Convert BGR to RGB and then to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # Process inputs
        inputs = self.processor(
            text=text_queries,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get model predictions
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
        # Convert to dictionary
        results = {
            query: float(score) 
            for query, score in zip(text_queries, probs[0].cpu().numpy())
            if float(score) >= confidence_threshold
        }
        
        return results

def process_video(video_path: str, text_queries: List[str], 
                 output_path: str = None, sample_rate: int = 1):
    """
    Process a video file using CLIP model
    
    Args:
        video_path: path to input video file
        text_queries: list of text descriptions to detect
        output_path: path to save the analysis results
        sample_rate: process every nth frame
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
        
        # Add timestamp
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # convert to seconds
        frame_results['timestamp'] = timestamp
        results.append(frame_results)
        
        print(f"Processed frame {frame_count}: {frame_results}")
    
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
        "traffic light","a car", "a truck", "a bus", "a motorcycle", "a vehicle"
    ]
    
    results = process_video(
        video_path=video_path,
        text_queries=text_queries,
        output_path="clip_analysis_results.json",
        sample_rate=30  # Process every 30th frame
    )
