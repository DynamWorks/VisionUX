from flask import Flask, send_from_directory
import threading
import asyncio
from pathlib import Path
from content_manager import ContentManager
from utils.websocket_handler import WebSocketHandler
from utils.rerun_server import RerunServer
import logging
import os

class BackendApp:
    def __init__(self):
        self.app = Flask(__name__, static_folder='../frontend/build')
        self.server = None
        self.content_manager = None
        self.models_loaded = False
        self.websocket_handler = WebSocketHandler()
        self.rerun_server = RerunServer()
        
        # Add routes
        self.setup_routes()
        
    def setup_routes(self):
        @self.app.route('/', defaults={'path': ''})
        @self.app.route('/<path:path>')
        def serve(path):
            if path and os.path.exists(self.app.static_folder + '/' + path):
                return send_from_directory(self.app.static_folder, path)
            return send_from_directory(self.app.static_folder, 'index.html')

    def is_ready(self):
        """Check if backend is ready"""
        return True

    def run(self, host='localhost', port=8000, debug=False):
        """Run the Flask application"""
        self.app.run(host=host, port=port, debug=debug)

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
                
                # Start both WebSocket and Rerun servers
                async def start_servers():
                    await asyncio.gather(
                        self.websocket_handler.start_server(port=port),
                        self.rerun_server.start()
                    )
                asyncio.run(start_servers())
            except Exception as e:
                raise RuntimeError(f"Failed to start backend server: {e}")
        
        if self.server is None:
            self.server = threading.Thread(target=run_server, daemon=True)
            self.server.start()

app = BackendApp()

def is_ready():
    """Check if backend is ready"""
    return app.is_ready()

def start(port=8000):
    """Start the backend server"""
    app.start(port=port)
