import argparse
import json
import time
from pathlib import Path
import cv2
import torch
from PIL import Image
import numpy as np
from ..core.analyzer import ClipVideoAnalyzer
from ..utils.visualizer import ResultVisualizer
from ..utils.config import Config

def analyze_video_with_clip(video_path: str, api_url: str = "http://localhost:8001"):
    """
    Perform comprehensive CLIP analysis on a video file
    
    Args:
        video_path: Path to video file
        api_url: Base URL of the API server (for future remote processing)
    """
    # Initialize analyzer with config
    config = Config()
    analyzer = ClipVideoAnalyzer(config=config.config)
    
    # Predefined categories for analysis
    analysis_categories = {
        'scene_type': [
            "indoor scene", "outdoor scene", "urban scene", "rural scene",
            "highway scene", "residential area", "commercial area",
            "intersection", "parking lot", "construction zone"
        ],
        'weather_conditions': [
            "sunny day", "rainy weather", "cloudy conditions", "foggy conditions",
            "snowy weather", "clear weather", "dark conditions", "bright daylight"
        ],
        'traffic_conditions': [
            "heavy traffic", "light traffic", "traffic jam", "free flowing traffic",
            "congested road", "empty road", "rush hour traffic"
        ],
        'road_features': [
            "straight road", "curved road", "intersection", "roundabout",
            "highway", "local street", "bridge", "tunnel", "construction zone"
        ],
        'activities': [
            "cars driving", "people walking", "cyclists riding",
            "vehicles turning", "pedestrians crossing", "parking maneuver",
            "loading/unloading", "road work", "emergency vehicle"
        ]
    }
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    print(f"\nAnalyzing video: {Path(video_path).name}")
    print(f"Duration: {duration:.1f}s ({total_frames} frames at {fps:.1f} FPS)")
    
    # Analysis results
    results = []
    sample_rate = 30  # Analyze every 30th frame (adjust as needed)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % sample_rate != 0:
            continue
            
        # Convert frame for CLIP
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp = frame_count / fps
        
        # Analyze frame with all categories
        frame_results = {
            'frame_number': frame_count,
            'timestamp': timestamp,
            'analysis': {}
        }
        
        for category, queries in analysis_categories.items():
            # Get CLIP embeddings for frame
            inputs = analyzer.processor(
                images=Image.fromarray(frame_rgb),
                text=queries,
                return_tensors="pt",
                padding=True
            ).to(analyzer.device)
            
            with torch.no_grad():
                outputs = analyzer.clip_model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)[0]
                
                # Get top 3 matches for each category
                top_k = 3
                top_probs, top_idx = torch.topk(probs, top_k)
                
                frame_results['analysis'][category] = [
                    {
                        'description': queries[idx],
                        'confidence': float(prob)
                    }
                    for prob, idx in zip(top_probs, top_idx)
                    if float(prob) > 0.1  # Filter low confidence
                ]
        
        # Get object detections
        detections = analyzer.analyze_frame(frame, queries)
        frame_results['detections'] = detections
        
        results.append(frame_results)
        
        # Progress update
        progress = (frame_count * 100) / total_frames
        print(f"\rProgress: {progress:.1f}%", end='')
    
    cap.release()
    print("\nAnalysis complete!")
    
    # Save results
    timestamp = int(time.time())
    output_path = f"clip_analysis_{timestamp}"
    
    with open(f"{output_path}.json", 'w') as f:
        json.dump({
            'video_info': {
                'path': video_path,
                'frames': total_frames,
                'fps': fps,
                'duration': duration
            },
            'analysis_results': results
        }, f, indent=2)
    
    # Generate visualizations
    visualizer = ResultVisualizer(f"{output_path}.json")
    
    # Create summary of dominant categories over time
    category_summary = {}
    for category in analysis_categories:
        timestamps = []
        descriptions = []
        confidences = []
        
        for frame in results:
            if frame['analysis'].get(category):
                top_match = frame['analysis'][category][0]
                timestamps.append(frame['timestamp'])
                descriptions.append(top_match['description'])
                confidences.append(top_match['confidence'])
        
        category_summary[category] = {
            'timestamps': timestamps,
            'descriptions': descriptions,
            'confidences': confidences
        }
    
    # Save category summary
    with open(f"{output_path}_categories.json", 'w') as f:
        json.dump(category_summary, f, indent=2)
    
    print(f"\nResults saved to: {output_path}.json")
    print(f"Category summary saved to: {output_path}_categories.json")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Comprehensive CLIP Video Analysis')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--api-url', default="http://localhost:8001",
                       help='API server URL')
    
    args = parser.parse_args()
    
    try:
        results = analyze_video_with_clip(
            args.video_path,
            args.api_url
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
