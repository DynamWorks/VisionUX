import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from ..utils.visualizer import ResultVisualizer

def visualize_analysis_results(results_path: str, video_path: str = None):
    """
    Create visualizations from video analysis results
    
    Args:
        results_path: Path to JSON results file
        video_path: Optional path to original video for frame visualization
    """
    print(f"\nVisualizing results from: {results_path}")
    
    # Initialize visualizer
    visualizer = ResultVisualizer(results_path)
    
    # Create output directory in tmp_content
    output_dir = Path('video_analytics/backend/tmp_content/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timeline plots
    print("Generating detection timeline...")
    visualizer.plot_detections_over_time()
    plt.savefig(output_dir / 'detections_timeline.png')
    plt.close()
    
    print("Generating detection type distribution...")
    visualizer.plot_detection_types()
    plt.savefig(output_dir / 'detection_types.png')
    plt.close()
    
    # Create summary CSVs
    print("Creating summary files...")
    visualizer.create_summary_csv()
    
    # Visualize sample frames if video provided
    if video_path:
        print("Generating frame visualizations...")
        results = visualizer.results
        
        if isinstance(results, dict) and 'results' in results:
            total_frames = len(results['results'])
        elif isinstance(results, list):
            total_frames = len(results)
        else:
            total_frames = 0
            
        if total_frames > 0:
            # Visualize start, middle and end frames
            frame_indices = [0, total_frames//2, total_frames-1]
            for idx in frame_indices:
                visualizer.visualize_frame(idx, video_path)
                print(f"Saved visualization for frame {idx}")
    
    print("\nVisualization complete!")
    print("Generated files:")
    print("- visualizations/detections_timeline.png")
    print("- visualizations/detection_types.png")
    print("- detection_summary.csv")
    print("- tracking_summary.csv")
    if video_path:
        print("- frame_*_visualized.jpg")

def main():
    parser = argparse.ArgumentParser(description='Visualize Video Analysis Results')
    parser.add_argument('results_path', help='Path to JSON results file')
    parser.add_argument('--video', help='Path to original video file (optional)')
    
    args = parser.parse_args()
    
    try:
        visualize_analysis_results(args.results_path, args.video)
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
