import argparse
import requests
import json
from pathlib import Path
import time

def test_chat_service(video_path: str, prompt: str, api_url: str = "http://localhost:8001"):
    """
    Test the chat-based video analysis service
    
    Args:
        video_path: Path to video file
        prompt: Chat prompt/query
        api_url: Base URL of the API server
    """
    try:
        # Validate video path
        if not Path(video_path).exists():
            print(f"Error: Video file not found: {video_path}")
            return
            
        # Ensure absolute path
        abs_video_path = str(Path(video_path).resolve())
        
        # Prepare request payload
        payload = {
            "video_path": abs_video_path,
            "prompt": prompt
        }
        
        print(f"\nSending chat request to {api_url}/api/v1/chat")
        print(f"Prompt: {prompt}")
        
        # Send request to chat endpoint
        response = requests.post(
            f"{api_url}/api/v1/chat",
            json=payload,
            stream=True  # Enable streaming response
        )
        
        if response.status_code != 200:
            print(f"\nError {response.status_code}: {response.text}")
            return
            
        # Process streaming response
        print("\nChat Response:")
        print("-" * 50)
        
        for line in response.iter_lines():
            if line:
                try:
                    result = json.loads(line.decode().replace('data: ', ''))
                    
                    # Print chat response
                    if "response" in result:
                        print(f"\nAssistant: {result['response']}")
                    
                    # Print analysis results
                    if "results" in result:
                        print("\nAnalysis Results:")
                        print(json.dumps(result["results"], indent=2))
                        
                except json.JSONDecodeError:
                    continue
                    
        # Save results
        output_dir = Path('tmp_content/chat_results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"chat_response_{int(time.time())}.json"
        with open(output_path, 'w') as f:
            json.dump({
                "prompt": prompt,
                "video_path": abs_video_path,
                "timestamp": time.time()
            }, f, indent=2)
            
        print(f"\nChat session saved to: {output_path}")
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API server")
        print("Make sure the server is running with: python -m video_analytics.main")
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Test Chat-based Video Analysis')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--prompt', default="What's happening in this video?",
                       help='Chat prompt/query')
    parser.add_argument('--api-url', default="http://localhost:8001",
                       help='API server URL')
    
    args = parser.parse_args()
    
    test_chat_service(
        args.video_path,
        args.prompt,
        api_url=args.api_url
    )

if __name__ == "__main__":
    main()
