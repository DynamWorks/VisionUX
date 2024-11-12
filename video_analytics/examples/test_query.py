import argparse
import requests
import json
from typing import List, Dict

def test_query(query: str, video_path: str, api_url: str = "http://localhost:8001", 
               max_results: int = 5, threshold: float = 0.2, filters: Dict = None) -> Dict:
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
    
    # First analyze the video
    print("\nAnalyzing video first...")
    analyze_payload = {
        "video_path": video_path,
        "text_queries": [query],  # Use the query as one of the detection targets
        "sample_rate": 30,  # Process every 30th frame for speed
        "max_workers": 4
    }
    
    try:
        response = requests.post(
            f"{api_url}/api/analyze",
            json=analyze_payload
        )
        response.raise_for_status()
        print("Video analysis complete!")
    except Exception as e:
        print(f"Error analyzing video: {str(e)}")
        return {}
    
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
            
        test_query(
            query=args.query,
            video_path=args.video_path,
            api_url=args.api_url,
            max_results=args.max_results,
            threshold=args.threshold,
            filters=filters if filters else None
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
