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
                     sample_rate: int = 1, max_workers: int = 4) -> Dict:
        """
        Send video analysis request to API
        
        Args:
            video_path: Path to video file
            text_queries: List of text descriptions to detect
            sample_rate: Process every nth frame
            max_workers: Number of parallel workers
            
        Returns:
            Analysis results dictionary
        """
        if not self.check_server():
            raise ConnectionError("API server is not running. Start it with: python -m video_analytics.main")
            
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        payload = {
            "video_path": str(Path(video_path).absolute()),
            "text_queries": text_queries,
            "sample_rate": sample_rate,
            "max_workers": max_workers
        }
        
        response = requests.post(
            f"{self.api_url}/api/analyze",
            json=payload
        )
        response.raise_for_status()
        
        return response.json()
