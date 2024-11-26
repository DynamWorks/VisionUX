from flask import Flask, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from api.routes import api
import threading
import logging
import time
from pathlib import Path
from content_manager import ContentManager
from utils.socket_handler import SocketHandler
from utils.config import Config
import os
import yaml

class BackendApp:
    def __init__(self):
        # Create logs directory
        Path('logs').mkdir(exist_ok=True)
        
        # Load config
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        self.config = Config(config_path)
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        # Get frontend path from config
        frontend_path = self.config.get('frontend', 'build_path', default='../frontend/build')
        self.app = Flask(__name__, static_folder=frontend_path)
        self.wsgi_app = self.app.wsgi_app  # Expose WSGI app
        
        # Initialize rate limiter
        self.limiter = Limiter(
            app=self.app,
            key_func=get_remote_address,
            default_limits=["200 per day", "50 per hour"],
            storage_uri="memory://"
        )
        
        # Get CORS settings from config
        cors_origins = self.config.get('api', 'cors_origins', default="*")
        CORS(self.app, resources={r"/api/*": {"origins": cors_origins}})
        logging.info(f"Flask app initialized with CORS origins: {cors_origins}")
        self.server = None
        self.content_manager = None
        self.models_loaded = False
        self.socket_handler = SocketHandler(self.app)
        
        # Register blueprints and routes
        self.app.register_blueprint(api, url_prefix='/api/v1')
        self.setup_routes()
    
    def setup_logging(self):
        """Initialize logging configuration"""
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        log_config = config.get('logging', {})
        log_file = log_config.get('file', 'video_analytics.log')
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Create logs directory if it doesn't exist
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / log_file
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Create formatters and handlers
        formatter = logging.Formatter(log_format)
        
        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        
        # Add handlers to root logger
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        logging.info("Logging system initialized")
        logging.info(f"Log file created at: {log_path}")
        
    def setup_routes(self):
        @self.app.route('/', defaults={'path': ''})
        @self.app.route('/<path:path>')
        def serve(path):
            if path and os.path.exists(self.app.static_folder + '/' + path):
                return send_from_directory(self.app.static_folder, path)
            return send_from_directory(self.app.static_folder, 'index.html')

        @self.app.route('/favicon.ico')
        def favicon():
            return send_from_directory(
                self.app.static_folder,
                'favicon.ico',
                mimetype='image/vnd.microsoft.icon'
            )

    def is_ready(self):
        """Check if backend is ready"""
        return True

    def run(self, host='localhost', port=8000, debug=False):
        """Run the Flask application with Socket.IO"""
        try:

            # # Start Rerun server in a separate thread
            # rerun_thread = threading.Thread(
            #     target=self.rerun_manager.start_web_server_sync,
            #     daemon=True
            # )
            # rerun_thread.start()
            # self.logger.info("Started Rerun web server thread")

            # # Start Rerun server in a separate thread
            # rerun_thread = threading.Thread(
            #     target=self.rerun_manager.start_web_server_sync,
            #     daemon=True
            # )
            # rerun_thread.start()
            # self.logger.info("Started Rerun web server thread")

            # # Wait for Rerun server to be ready
            # max_retries = 5
            # retry_delay = 2
            # for attempt in range(max_retries):
            #     try:
            #         import requests
            #         response = requests.get(f"http://{self.rerun_manager._web_host}:{self.rerun_manager._web_port}/health")
            #         if response.status_code == 200:
            #             self.logger.info(f"Rerun server verified running on port {self.rerun_manager._web_port}")
            #             break
            #     except Exception as e:
            #         if attempt == max_retries - 1:
            #             self.logger.error(f"Failed to verify Rerun server: {e}")
            #             raise
            #         self.logger.warning(f"Waiting for Rerun server (attempt {attempt + 1}/{max_retries})")
            #         time.sleep(retry_delay)
            
            # Get host/port from config with fallbacks
            host = self.config.get('api', 'host', default='0.0.0.0')
            port = self.config.get('api', 'port', default=port)
            debug = self.config.get('api', 'debug', default=debug)
            
            # Configure CORS for both REST and WebSocket
            CORS(self.app, resources={
                r"/api/*": {
                    "origins": "*",
                    "allow_headers": ["Content-Type"],
                    "methods": ["GET", "POST", "OPTIONS"]
                },
                r"/socket.io/*": {
                    "origins": "*",
                    "allow_headers": ["Content-Type"],
                    "methods": ["GET", "POST", "OPTIONS"]
                }
            })

            try:
                # Initialize eventlet
                import eventlet
                eventlet.monkey_patch()
            
                # Run Flask-SocketIO app with proper configuration
                self.socket_handler.socketio.init_app(
                    self.app,
                    cors_allowed_origins="*",
                    ping_timeout=60,
                    ping_interval=25,
                    async_mode='eventlet',  # Use eventlet for better WebSocket support
                    engineio_logger=True,
                    logger=True,
                    reconnection=True,
                    reconnection_attempts=10,
                    reconnection_delay=1000,
                    reconnection_delay_max=5000,
                    http_compression=True,
                    transports=['websocket'],  # WebSocket only, no polling
                    upgrade_timeout=20000,
                    max_http_buffer_size=100 * 1024 * 1024,
                    manage_session=False,  # Disable session management
                    allow_upgrades=True
                )
            
                self.logger.info("WebSocket server initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize WebSocket server: {e}")
                raise RuntimeError(f"WebSocket initialization failed: {e}")
            # Initialize without SSL for development
            self.socket_handler.socketio.run(
                self.app,
                host=host,
                port=port,
                debug=debug,
                allow_unsafe_werkzeug=True,  # Required for production
                use_reloader=False,  # Disable reloader in production
                log_output=True
            )
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            raise

    def start(self, default_port=8000, config=None):
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
                
                # Get server config with default port from parameter
                host = self.config.get('api', 'host', default='0.0.0.0')
                configured_port = self.config.get('api', 'port')
                port = configured_port if configured_port is not None else default_port
                debug = self.config.get('api', 'debug', default=False)
                
                # Start the Flask-SocketIO server
                self.socket_handler.socketio.run(
                    self.app,
                    host=host,
                    port=port, 
                    debug=debug,
                    allow_unsafe_werkzeug=True  # Required for production
                )
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
