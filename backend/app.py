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

# Initialize eventlet and monkey patch at the very beginning
import eventlet
eventlet.monkey_patch(socket=True, select=True, thread=True)

class BackendApp:
    def __init__(self):
        # Create logs directory
        Path('logs').mkdir(exist_ok=True)
        
        # Load config
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        self.config = Config(config_path)
        if not self.config._config:
            self.config.reset()
        
        # Set debug mode from config
        self.debug = self.config.get('api', 'debug', default=False)
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Get frontend path from config
        frontend_path = self.config.get('frontend', 'build_path', default='../frontend/build')
        self.flask_app = Flask(__name__, static_folder=frontend_path)
        
        # Initialize rate limiter with proper format
        with self.flask_app.app_context():
            default_limits = ["1000 per day"]
            rate_limits = self.config.get('api', 'rate_limits', default={})
            if isinstance(rate_limits, dict):
                if 'default' in rate_limits:
                    default_limits = [rate_limits['default']]
                # Don't add upload limit to default_limits
                    
            self.limiter = Limiter(
                app=self.flask_app,
                key_func=get_remote_address,
                default_limits=default_limits,
                storage_uri="memory://"
            )
        
        # Setup CORS
        CORS(self.flask_app, 
             resources={
                 r"/api/*": {"origins": self.config.get('api', 'cors_origins', default="*")},
                 r"/socket.io/*": {
                     "origins": "*",
                     "allow_headers": ["Content-Type"],
                     "methods": ["GET", "POST", "OPTIONS"],
                     "supports_credentials": True
                 }
             },
             supports_credentials=True)
        
        # Initialize Socket.IO with proper configuration
        self.socketio = SocketIO(
            self.flask_app,
            cors_allowed_origins="*",
            async_mode='eventlet',
            logger=True,
            engineio_logger=True,
            ping_timeout=self.config.get('websocket', 'ping_timeout', default=60),
            ping_interval=self.config.get('websocket', 'ping_interval', default=25),
            max_http_buffer_size=self.config.get('websocket', 'max_buffer_size', default=100 * 1024 * 1024),
            path='/socket.io/',
            always_connect=True,
            transports=['websocket'],
            manage_session=True
        )
        
        # Initialize core services
        self.init_services()
        
        # Register blueprints and routes
        self.flask_app.register_blueprint(api, url_prefix='/api/v1')
        self.setup_routes()
        
        self.server = None
        logging.info("Backend app initialized")
    
    def init_services(self):
        """Initialize core services"""
        try:
            # Initialize content manager with config
            base_path = self.config.get('content_storage', 'base_path', default='tmp_content')
            self.content_manager = ContentManager(base_path=base_path)
            
            # Initialize socket handler
            with self.flask_app.app_context():
                self.socket_handler = SocketHandler(self.flask_app)
            
            # Create required directories
            Path(base_path).mkdir(parents=True, exist_ok=True)
            Path(base_path, "uploads").mkdir(parents=True, exist_ok=True)
            
            self.logger.info("Core services initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize services: {e}")
            raise
    
    def setup_logging(self):
        """Initialize logging configuration"""
        log_config = self.config.get('logging', default={
            'level': 'INFO',
            'file': 'video_analytics.log',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        })
        
        log_file = log_config.get('file', 'video_analytics.log')
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / log_file
        
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        formatter = logging.Formatter(log_format)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        logging.info("Logging system initialized")
        logging.info(f"Log file created at: {log_path}")
        
    def setup_routes(self):
        """Setup basic application routes"""
        @self.flask_app.route('/', defaults={'path': ''})
        @self.flask_app.route('/<path:path>')
        def serve(path):
            if path and os.path.exists(self.flask_app.static_folder + '/' + path):
                return send_from_directory(self.flask_app.static_folder, path)
            return send_from_directory(self.flask_app.static_folder, 'index.html')

        @self.flask_app.route('/favicon.ico')
        def favicon():
            return send_from_directory(
                self.flask_app.static_folder,
                'favicon.ico',
                mimetype='image/vnd.microsoft.icon'
            )

    def run(self, host='127.0.0.1', port=8000, debug=False):
        """Run the Flask application with Socket.IO"""
        try:
            port = self.config.get('api', 'port', default=port)
            debug = self.config.get('api', 'debug', default=debug)

            ssl_enabled = self.config.get('websocket', 'ssl_enabled', default=False)
            ssl_args = {}
            
            if ssl_enabled:
                keyfile = self.config.get('websocket', 'keyfile')
                certfile = self.config.get('websocket', 'certfile')
                
                if keyfile and certfile:
                    ssl_args = {
                        'keyfile': keyfile,
                        'certfile': certfile,
                        'ssl_protocol': 'TLSv1_2'
                    }
                    self.logger.info("Enabling SSL for WebSocket server")
                else:
                    self.logger.warning("SSL enabled but key/cert files not configured")

            self.socketio.run(
                self.flask_app,
                host=host,
                port=port,
                debug=debug,
                use_reloader=False,
                log_output=True,
                **ssl_args
            )

        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            raise

    def start(self, default_port=8000):
        """Start the backend server in a separate thread"""
        def run_server():
            try:
                port = self.config.get('api', 'port', default=default_port)
                debug = self.config.get('api', 'debug', default=False)
                self.run(host='127.0.0.1', port=port, debug=debug)
            except Exception as e:
                raise RuntimeError(f"Failed to start backend server: {e}")
        
        if self.server is None:
            self.server = threading.Thread(target=run_server, daemon=True)
            self.server.start()

    def is_ready(self):
        """Check if backend is ready"""
        return True

# Create a single instance of the application
backend_app = BackendApp()
