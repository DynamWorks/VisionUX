"""URL configuration"""
from typing import Dict
import os

def get_urls() -> Dict[str, str]:
    """Get URL configuration from environment variables with defaults"""
    return {
        'api': os.getenv('VIDEO_ANALYTICS_API_URL', 'http://localhost:8000'),
        'ws': os.getenv('VIDEO_ANALYTICS_WS_URL', 'ws://localhost:8001'),
        'rerun_web': os.getenv('VIDEO_ANALYTICS_RERUN_WEB_URL', 'http://localhost:9090'),
        'rerun_ws': os.getenv('VIDEO_ANALYTICS_RERUN_WS_URL', 'ws://localhost:4321')
    }
