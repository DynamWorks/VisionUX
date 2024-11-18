from flask import Flask
import threading
import asyncio
from pathlib import Path
from .content_manager import ContentManager
from video_analytics.utils.websocket_handler import WebSocketHandler
import logging

class BackendApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.server = None
        self.content_manager = None
        self.models_loaded = False
        self.websocket_handler = WebSocketHandler()

    def is_ready(self):
        """Check if backend is ready"""
        return True

    def start(self, port=8502, config=None):
        """Start the backend server in a separate thread"""
        def run_server():
            try:
                # Initialize content manager
                if config and 'backend' in config:
                    content_config = config['backend'].get('content_storage', {})
                    self.content_manager = ContentManager(
                        base_path=content_config.get('base_path', 'tmp_content')
                    )
                    logging.info("Content manager initialized")
                    
                # Ensure model directories exist
                Path("backend/models/yolo").mkdir(parents=True, exist_ok=True)
                Path("backend/models/clip").mkdir(parents=True, exist_ok=True)
                Path("backend/models/traffic_signs").mkdir(parents=True, exist_ok=True)
                
                # Start WebSocket server
                asyncio.run(self.websocket_handler.start_server(port=port))
            except Exception as e:
                raise RuntimeError(f"Failed to start backend server: {e}")
        
        if self.server is None:
            self.server = threading.Thread(target=run_server, daemon=True)
            self.server.start()

app = BackendApp()

def is_ready():
    """Check if backend is ready"""
    return app.is_ready()

def start(port=8502):
    """Start the backend server"""
    app.start(port=port)
