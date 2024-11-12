import argparse
import json
import time
from pathlib import Path
from ..utils.client import VideoAnalyticsClient
from ..utils.visualizer import ResultVisualizer

def analyze_video(video_path: str, text_queries: list, api_url: str = "http://localhost:5000"):
    """
    Basic video analysis using object detection and tracking.
    Provides general-purpose analysis based on user-provided queries.
    Focuses on:
    - Object detection and tracking
    - Basic scene analysis
    - Visualization of results
    
    This function:
    1. Sends a single API request with user-provided text queries
    2. Processes all frames in one batch
    3. Creates general visualizations (timeline, types, frames)
    4. Outputs basic CSV summaries
    
    Args:
        video_path: Path to the video file
        text_queries: List of text descriptions to detect
        api_url: Base URL of the API server
    """
    # Initialize API client
    client = VideoAnalyticsClient(api_url)
    
    # Send analysis request
    print(f"Analyzing video: {video_path}")
    try:
        results = client.analyze_video(
            video_path,
            text_queries,
            sample_rate=30,  # Process every 30th frame
            max_workers=4    # Use 4 parallel workers
        )
        
        # Save results
        output_path = f"analysis_results_{int(time.time())}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Analysis complete! Results saved to {output_path}")
        
        # Generate visualizations
        visualizer = ResultVisualizer(output_path)
        visualizer.plot_detections_over_time()
        visualizer.plot_detection_types()
        visualizer.create_summary_csv()
        
        print("\nGenerated visualization files:")
        print("- detections_timeline.png")
        print("- detection_types.png") 
        print("- detection_summary.csv")
        print("- tracking_summary.csv")
        
        return results
        
    except ConnectionError as e:
        print(f"Error: {e}")
        print("Make sure the API server is running with: streamlit run video_analytics/frontend/app.py")
        raise
    except Exception as e:
        print(f"Error during analysis: {e}")
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
    parser.add_argument('--parallel', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    
    args = parser.parse_args()
    
    try:
        results = analyze_video(
            args.video_path,
            args.queries,
            args.api_url
        )
        
        # Save results
        output_path = f"analysis_results_{int(time.time())}"
        with open(f"{output_path}.json", 'w') as f:
            json.dump(results, f, indent=2)
            
        # Create visualizations
        from ..utils.visualizer import ResultVisualizer
        visualizer = ResultVisualizer(f"{output_path}.json")
        
        # Generate plots and summaries
        visualizer.plot_detections_over_time()
        visualizer.plot_detection_types()
        visualizer.create_summary_csv()
        
        # Visualize sample frames
        total_frames = results.get('total_frames', 0)
        if total_frames > 0:
            # Visualize first, middle and last frames
            frame_indices = [0, total_frames//2, total_frames-1]
            for idx in frame_indices:
                visualizer.visualize_frame(idx, args.video_path)
        
        print(f"\nAnalysis complete!")
        print(f"Processed {total_frames} frames")
        print(f"Results saved to: {output_path}.json")
        print(f"Summaries saved to: detection_summary.csv, tracking_summary.csv")
        print(f"Visualizations saved to: detections_timeline.png, detection_types.png")
        print(f"Sample frames saved as: frame_*_visualized.jpg")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
