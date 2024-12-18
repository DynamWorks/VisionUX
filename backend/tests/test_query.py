import argparse
import requests
import json
import sseclient
from typing import List, Dict, Generator
import sys
sys.path.append("../../")  # Add parent directory to path

def test_query(query: str, video_path: str, api_url: str = "http://localhost:8001", 
               max_results: int = 5, threshold: float = 0.2, filters: Dict = None,
               stream: bool = True) -> Generator[Dict, None, None]:
    """
    Test the query API endpoint with advanced filtering
    
    Args:
        query: Text query to search for
        video_path: Path to video file
        api_url: Base URL of the API server
        max_results: Maximum number of results to return
        threshold: Minimum similarity threshold
        filters: Optional filters dictionary with keys:
                - time_range: [start_time, end_time] in seconds
                - object_types: List of object classes to filter for
                - min_confidence: Minimum detection confidence
        
    Returns:
        Query results dictionary
    """
    # Ensure API URL is properly formatted
    api_url = api_url.rstrip('/')
    
    # First analyze the video with streaming support
    print("\nAnalyzing video first...")
    analyze_payload = {
        "video_path": video_path,
        "text_queries": [query],  # Use the query as one of the detection targets
        "sample_rate": 30,  # Process every 30th frame for speed
        "max_workers": 4
    }
    
    try:
        if stream:
            response = requests.post(
                f"{api_url}/api/analyze",
                json=analyze_payload,
                stream=True
            )
            response.raise_for_status()
            
            client = sseclient.SSEClient(response)
            for event in client.events():
                try:
                    frame_result = json.loads(event.data)
                    yield frame_result
                except json.JSONDecodeError:
                    continue
                    
            print("Video analysis complete!")
        else:
            response = requests.post(
                f"{api_url}/api/analyze",
                json=analyze_payload
            )
            response.raise_for_status()
            print("Video analysis complete!")
            
    except Exception as e:
        print(f"Error analyzing video: {str(e)}")
        return
    
    # Now query the analyzed frames
    print("\nQuerying analysis results...")
    query_payload = {
        "query": query,
        "max_results": max_results,
        "video_path": video_path,
        "threshold": threshold
    }
    
    # Add optional filters if provided
    if filters:
        query_payload["filters"] = filters
    
    try:
        # Send query request
        response = requests.post(
            f"{api_url}/api/query",
            json=query_payload
        )
        response.raise_for_status()
        
        results = response.json()
        
        # Print results in readable format
        print(f"\nQuery: {query}")
        print(f"Found {len(results.get('results', []))} matches\n")
        
        for i, match in enumerate(results.get('results', []), 1):
            print(f"Match {i}:")
            print(f"Frame: {match.get('frame_number', 'unknown')}")
            print(f"Timestamp: {match.get('timestamp', 0):.2f}s")
            print(f"Similarity: {match.get('similarity', 0):.3f}")
            
            # Print detection details
            detections = match.get('detections', {})
            if detections:
                print("Detections:")
                for det_type, dets in detections.items():
                    if dets:
                        print(f"- {det_type}: {len(dets)} items")
            
            # Print VILA analysis
            vila_analysis = match.get('vila_analysis', {})
            if vila_analysis:
                print("\nScene Analysis:")
                print(f"- Scene Type: {vila_analysis.get('scene_type', 'unknown')}")
                print(f"- Activities: {', '.join(vila_analysis.get('activities', []))}")
                
                # Print infrastructure details
                infra = vila_analysis.get('infrastructure', {})
                if infra:
                    print(f"- Infrastructure: {infra.get('infrastructure_type', 'unknown')}")
                    if infra.get('traffic_signs'):
                        print(f"  - Signs: {', '.join(infra['traffic_signs'])}")
                    
                # Print object analysis
                objects = vila_analysis.get('objects', {})
                if objects:
                    print(f"- Objects: {objects.get('total_objects', 0)} total")
                    if objects.get('primary_objects'):
                        print(f"  - Primary: {', '.join(objects['primary_objects'])}")
            print()
            
        return results
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API server")
        print("Make sure the server is running with: python -m video_analytics.main")
        return {}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {}

def main():
    parser = argparse.ArgumentParser(description='Test Video Analytics Query API')
    parser.add_argument('--query', default="Show me cars",
                       help='Query string to search for')
    parser.add_argument('--api-url', default="http://localhost:8001",
                       help='API server URL')
    parser.add_argument('--max-results', type=int, default=5,
                       help='Maximum number of results to return')
    parser.add_argument('--video-path', required=True,
                       help='Path to the video file to query')
    parser.add_argument('--threshold', type=float, default=0.2,
                       help='Minimum similarity threshold (0-1)')
    parser.add_argument('--time-start', type=float,
                       help='Start time in seconds for filtering')
    parser.add_argument('--time-end', type=float,
                       help='End time in seconds for filtering')
    parser.add_argument('--object-types', nargs='+',
                       help='Object types to filter for (e.g., car truck)')
    parser.add_argument('--min-confidence', type=float,
                       help='Minimum detection confidence (0-1)')
    parser.add_argument('--scene-type', type=str,
                       help='Filter by scene type (e.g., highway, intersection, street)')
    parser.add_argument('--infrastructure', type=str,
                       help='Filter by infrastructure type (e.g., major_road, controlled_road)')
    parser.add_argument('--activity', type=str,
                       help='Filter by activity type (e.g., moving_traffic, pedestrian_activity)')
    
    args = parser.parse_args()
    
    try:
        # Build filters dictionary from args
        filters = {}
        if args.time_start is not None and args.time_end is not None:
            filters['time_range'] = [args.time_start, args.time_end]
        if args.object_types:
            filters['object_types'] = args.object_types
        if args.min_confidence is not None:
            filters['min_confidence'] = args.min_confidence
        if args.scene_type:
            filters['scene_type'] = args.scene_type
        if args.infrastructure:
            filters['infrastructure'] = args.infrastructure
        if args.activity:
            filters['activity'] = args.activity
            
        for result in test_query(
            query=args.query,
            video_path=args.video_path,
            api_url=args.api_url,
            max_results=args.max_results,
            threshold=args.threshold,
            filters=filters if filters else None,
            stream=True
        ):
            # Print frame results in real-time
            print(f"\rFrame {result.get('frame_number', 0)}: "
                  f"{len(result.get('detections', {}).get('segments', []))} detections", 
                  end='')
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
