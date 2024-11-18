import argparse
import requests
import base64
import json
import time
from pathlib import Path
import sys
sys.path.append("../../")  # Add parent directory to path

def test_scene_analysis(image_path: str, api_url: str = "http://localhost:8001", context: str = None):
    """
    Test the scene analysis API endpoint
    
    Args:
        image_path: Path to image file
        api_url: Base URL of the API server
        context: Optional context about the scene
    """
    # Prepare request payload
    payload = {
        "image_path": image_path,
        "context": context,
        "stream_type": "test"
    }
    
    try:
        # Validate image path
        if not Path(image_path).exists():
            print(f"Error: Image file not found: {image_path}")
            return
            
        # Ensure absolute path
        abs_image_path = str(Path(image_path).resolve())
        
        # Update payload with absolute path
        payload = {
            "image_path": abs_image_path,
            "context": context,
            "stream_type": "test"
        }
        
        # Send request to API
        print(f"\nSending request to {api_url}/api/v1/analyze_scene")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(
            f"{api_url}/api/v1/analyze_scene",
            json=payload
        )
        
        if response.status_code != 200:
            print(f"\nError {response.status_code}: {response.text}")
            return
        
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
                
        # Save results to tmp_content
        output_path = Path('tmp_content/scene_analysis') / f"scene_analysis_{int(time.time())}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
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
