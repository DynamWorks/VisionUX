"""URL configuration"""
from typing import Dict
import os

def get_urls() -> Dict[str, str]:
    """Get URL configuration from environment variables with defaults"""
    # Get host IP from environment or use default
    host_ip = os.getenv('VIDEO_ANALYTICS_HOST_IP', '127.0.0.1')
    
    return {
        'api': os.getenv('VIDEO_ANALYTICS_API_URL', f'http://{host_ip}:8000'),
        'ws': os.getenv('VIDEO_ANALYTICS_WS_URL', f'ws://{host_ip}:8001'),
        'rerun_web': os.getenv('VIDEO_ANALYTICS_RERUN_WEB_URL', f'http://{host_ip}:9090'),
        'rerun_ws': os.getenv('VIDEO_ANALYTICS_RERUN_WS_URL', f'ws://{host_ip}:4321')
    }
