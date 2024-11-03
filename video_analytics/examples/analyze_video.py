import requests
import json
import argparse
import time
from pathlib import Path

def analyze_video(video_path: str, text_queries: list, api_url: str = "http://localhost:8001"):
    """
    Send a video analysis request to the API
    
    Args:
        video_path: Path to the video file
        text_queries: List of text descriptions to detect
        api_url: Base URL of the API server
    """
    # Ensure video file exists
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    # Prepare request payload
    payload = {
        "video_path": str(Path(video_path).absolute()),
        "text_queries": text_queries,
        "sample_rate": 5,  # Process every frame
        "max_workers": 4   # Number of parallel workers
    }
    
    # Check API health
    try:
        health_response = requests.get(f"{api_url}/api/health")
        health_response.raise_for_status()
        print("API server is healthy")
    except Exception as e:
        raise ConnectionError(f"API server not available: {str(e)}")
    
    # Send analysis request
    print(f"Sending analysis request for {video_path}")
    try:
        response = requests.post(
            f"{api_url}/api/analyze",
            json=payload
        )
        response.raise_for_status()
        
        # Save results
        results = response.json()
        output_path = f"analysis_results_{int(time.time())}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Analysis complete! Results saved to {output_path}")
        return results
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Video Analysis API Client')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--queries', nargs='+', default=[
        "person walking", "car driving", "traffic jam",
        "bicycle", "pedestrian crossing", "traffic light",
        "car", "truck", "bus", "motorcycle", "vehicle"
    ], help='Text queries to detect')
    parser.add_argument('--api-url', default="http://localhost:8001",
                       help='API server URL')
    
    args = parser.parse_args()
    
    try:
        results = analyze_video(
            args.video_path,
            args.queries,
            args.api_url
        )
        total_frames = results.get('total_frames', 0)
        total_detections = sum(len(frame.get('detections', {}).get('segments', [])) 
                             for frame in results.get('results', []))
        print(f"Processed {total_frames} frames")
        print(f"Detected {total_detections} objects across all frames")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
