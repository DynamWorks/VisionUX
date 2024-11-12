import argparse
import requests
import json
from typing import List, Dict

def test_query(query: str, api_url: str = "http://localhost:8001", max_results: int = 5) -> Dict:
    """
    Test the query API endpoint
    
    Args:
        query: Text query to search for
        api_url: Base URL of the API server
        max_results: Maximum number of results to return
        
    Returns:
        Query results dictionary
    """
    # Ensure API URL is properly formatted
    api_url = api_url.rstrip('/')
    
    # Prepare request payload
    payload = {
        "query": query,
        "max_results": max_results
    }
    
    try:
        # Send query request
        response = requests.post(
            f"{api_url}/api/query",
            json=payload
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
    
    args = parser.parse_args()
    
    try:
        test_query(
            query=args.query,
            api_url=args.api_url,
            max_results=args.max_results
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
