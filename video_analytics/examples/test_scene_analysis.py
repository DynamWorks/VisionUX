import argparse
import requests
import base64
import json
from pathlib import Path

def test_scene_analysis(image_path: str, api_url: str = "http://localhost:8001", context: str = None):
    """
    Test the scene analysis API endpoint
    
    Args:
        image_path: Path to image file
        api_url: Base URL of the API server
        context: Optional context about the scene
    """
    # Read and encode image
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    # Prepare request payload
    payload = {
        "frame": image_data,
        "context": context,
        "stream_type": "test"
    }
    
    try:
        # Send request to API
        response = requests.post(
            f"{api_url}/api/v1/analyze_scene",
            json=payload
        )
        response.raise_for_status()
        
        # Print results
        results = response.json()
        print("\nScene Analysis Results:")
        print("-" * 50)
        
        if 'scene_analysis' in results:
            print("\nScene Analysis:")
            print(results['scene_analysis'])
            
        if 'suggested_pipeline' in results:
            print("\nSuggested Pipeline:")
            for step in results['suggested_pipeline']:
                print(f"- {step}")
                
        # Save results to file
        output_path = "scene_analysis.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\nFull results saved to: {output_path}")
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API server")
        print("Make sure the server is running with: python -m video_analytics.main")
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Test Scene Analysis API')
    parser.add_argument('image_path', help='Path to image file')
    parser.add_argument('--api-url', default="http://localhost:8001",
                       help='API server URL')
    parser.add_argument('--context', type=str,
                       help='Optional context about the scene')
    
    args = parser.parse_args()
    
    test_scene_analysis(
        args.image_path,
        api_url=args.api_url,
        context=args.context
    )

if __name__ == "__main__":
    main()
