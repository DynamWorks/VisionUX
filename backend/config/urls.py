"""URL configuration for video analytics services"""
from typing import Dict, Optional
import os
import logging
from urllib.parse import urlparse
from ..utils.config import Config

logger = logging.getLogger(__name__)

def validate_url(url: str, scheme: str) -> bool:
    """Validate URL format and scheme"""
    try:
        parsed = urlparse(url)
        return parsed.scheme == scheme and bool(parsed.netloc)
    except Exception:
        return False

def get_urls() -> Dict[str, str]:
    """
    Get URL configuration from environment variables with defaults
    
    Returns:
        Dict containing URLs for API, WebSocket and Rerun services
        
    Raises:
        ValueError: If required configuration is missing or invalid
    """
    try:
        config = Config()
        
        # Get host IP and ports with validation
        host_ip = os.getenv('VIDEO_ANALYTICS_HOST_IP', 
                           config.get('api', 'host', default='127.0.0.1'))
        
        if not host_ip:
            raise ValueError("Host IP not configured")
            
        # Get port numbers with validation
        def get_port(env_var: str, config_path: str, default: int) -> int:
            port = os.getenv(env_var, config.get(*config_path.split('.'), default=default))
            try:
                port_num = int(port)
                if not (1024 <= port_num <= 65535):
                    raise ValueError
                return port_num
            except ValueError:
                raise ValueError(f"Invalid port number for {env_var}")
                
        api_port = get_port('VIDEO_ANALYTICS_API_PORT', 'api.port', 8000)
        ws_port = get_port('VIDEO_ANALYTICS_WS_PORT', 'websocket.port', 8001)
        rerun_web_port = get_port('VIDEO_ANALYTICS_RERUN_WEB_PORT', 'rerun.web_port', 9090)
        rerun_ws_port = get_port('VIDEO_ANALYTICS_RERUN_WS_PORT', 'rerun.ws_port', 4321)
        
        # Construct URLs
        urls = {
            'api': os.getenv('VIDEO_ANALYTICS_API_URL', 
                            f'http://{host_ip}:{api_port}'),
            'ws': os.getenv('VIDEO_ANALYTICS_WS_URL',
                           f'ws://{host_ip}:{ws_port}'),
            'rerun_web': os.getenv('VIDEO_ANALYTICS_RERUN_WEB_URL',
                                  f'http://{host_ip}:{rerun_web_port}'),
            'rerun_ws': os.getenv('VIDEO_ANALYTICS_RERUN_WS_URL',
                                 f'ws://{host_ip}:{rerun_ws_port}')
        }
        
        # Validate URL formats
        if not validate_url(urls['api'], 'http'):
            raise ValueError(f"Invalid API URL: {urls['api']}")
        if not validate_url(urls['ws'], 'ws'):
            raise ValueError(f"Invalid WebSocket URL: {urls['ws']}")
        if not validate_url(urls['rerun_web'], 'http'):
            raise ValueError(f"Invalid Rerun web URL: {urls['rerun_web']}")
        if not validate_url(urls['rerun_ws'], 'ws'):
            raise ValueError(f"Invalid Rerun WebSocket URL: {urls['rerun_ws']}")
            
        return urls
        
    except Exception as e:
        logger.error(f"Error getting URLs: {str(e)}")
        raise ValueError(f"Failed to get URLs: {str(e)}")
