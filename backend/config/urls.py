"""URL configuration"""
from typing import Dict
import os

def get_urls() -> Dict[str, str]:
    """Get URL configuration from environment variables with defaults"""
    from ..config import Config
    config = Config()
    
    # Get host IP and ports from config with environment variable overrides
    host_ip = os.getenv('VIDEO_ANALYTICS_HOST_IP', config.get('api', {}).get('host', '127.0.0.1'))
    api_port = os.getenv('VIDEO_ANALYTICS_API_PORT', config.get('api', {}).get('port', 8000))
    ws_port = os.getenv('VIDEO_ANALYTICS_WS_PORT', config.get('websocket', {}).get('port', 8001))
    rerun_web_port = os.getenv('VIDEO_ANALYTICS_RERUN_WEB_PORT', config.get('rerun', {}).get('web_port', 9090))
    rerun_ws_port = os.getenv('VIDEO_ANALYTICS_RERUN_WS_PORT', config.get('rerun', {}).get('ws_port', 4321))
    
    return {
        'api': os.getenv('VIDEO_ANALYTICS_API_URL', f'http://{host_ip}:{api_port}'),
        'ws': os.getenv('VIDEO_ANALYTICS_WS_URL', f'ws://{host_ip}:{ws_port}'),
        'rerun_web': os.getenv('VIDEO_ANALYTICS_RERUN_WEB_URL', f'http://{host_ip}:{rerun_web_port}'),
        'rerun_ws': os.getenv('VIDEO_ANALYTICS_RERUN_WS_URL', f'ws://{host_ip}:{rerun_ws_port}')
    }
