import requests
import time
from typing import List, Dict
from pathlib import Path

class VideoAnalyticsClient:
    """Client for interacting with the Video Analytics API"""
    
    def __init__(self, api_url: str = "http://localhost:5000"):
        self.api_url = api_url.rstrip('/')
        
    def check_server(self) -> bool:
        """Check if API server is running"""
        try:
            response = requests.get(f"{self.api_url}/api/health")
            return response.status_code == 200
        except:
            return False
            
    def analyze_video(self, video_path: str, text_queries: List[str],
                     sample_rate: int = 1, max_workers: int = 4,
                     timeout: int = 300) -> Dict:
        """
        Send video analysis request to API
        
        Args:
            video_path: Path to video file
            text_queries: List of text descriptions to detect
            sample_rate: Process every nth frame
            max_workers: Number of parallel workers
            
        Returns:
            Analysis results dictionary
            
        Raises:
            ConnectionError: If API server is not running
            FileNotFoundError: If video file does not exist
            ValueError: If text_queries is empty
            requests.exceptions.RequestException: If API request fails
        """
        if not self.check_server():
            raise ConnectionError("API server is not running. Start it with: python -m video_analytics.main")
            
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        if not video_path.is_file():
            raise ValueError(f"Path is not a file: {video_path}")
            
        if not text_queries:
            raise ValueError("text_queries cannot be empty")
            
        payload = {
            "video_path": str(Path(video_path).absolute()),
            "text_queries": text_queries,
            "sample_rate": sample_rate,
            "max_workers": max_workers
        }
        
        # Validate video format
        valid_formats = ['.mp4', '.avi', '.mov', '.mkv']
        if not any(str(video_path).lower().endswith(fmt) for fmt in valid_formats):
            raise ValueError(f"Unsupported video format. Supported formats: {valid_formats}")
            
        response = requests.post(
            f"{self.api_url}/api/analyze",
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        
        return response.json()
