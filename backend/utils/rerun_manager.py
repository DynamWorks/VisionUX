import rerun as rr
import logging
from typing import Optional
import asyncio
from aiohttp import web
import time
import cv2
import os
from pathlib import Path

class RerunManager:
    """Singleton class to manage Rerun initialization and state"""
    _instance = None
    _initialized = False
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.logger = logging.getLogger(__name__)
            
            # Load config
            from ..utils.config import Config
            self.config = Config()
            self._config = self.config._config  # Store config dict directly
            
            # Initialize state
            self._initialized = False
            self._server_started = False
            self._active_connections = 0
            self._shutdown_event = None
            
    def _verify_environment(self) -> bool:
        """Verify required environment settings"""
        try:
            # Initialize ports from config if not already set
            if not hasattr(self, '_ws_port'):
                self._ws_port = int(os.getenv('VIDEO_ANALYTICS_RERUN_WS_PORT', 4321))
            if not hasattr(self, '_web_port'):
                self._web_port = int(os.getenv('VIDEO_ANALYTICS_RERUN_WEB_PORT', 9090))

            # Create required directories first
            for path in ['tmp_content', 'tmp_content/uploads', 'tmp_content/analysis']:
                Path(path).mkdir(parents=True, exist_ok=True)
                
            # Check if ports are already in use
            import socket
            for port in [self._ws_port, self._web_port]:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    # Set socket options for immediate reuse
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind(('localhost', port))
                    sock.close()
                except OSError as e:
                    # Log but don't fail - port might be from previous instance
                    self.logger.warning(f"Port {port} may be in use: {e}")
                    continue
                
            # Set URLs after port verification
            self._ws_host = 'localhost'
            self._web_host = 'localhost'
                
            return True
            
        except Exception as e:
            self.logger.error(f"Environment verification failed: {e}")
            return False
            
            # Get URLs from config with environment variable overrides
            import os
            self._ws_port = int(os.getenv('VIDEO_ANALYTICS_RERUN_WS_PORT', 
                                        self.config.get('rerun', 'ws_port', default=4321)))
            self._web_port = int(os.getenv('VIDEO_ANALYTICS_RERUN_WEB_PORT',
                                         self.config.get('rerun', 'web_port', default=9090)))
            self._host = os.getenv('VIDEO_ANALYTICS_HOST_IP',
                                 self.config.get('rerun', 'host', default='localhost'))
            
            # Initialize web server components
            self._app = web.Application()
            self._runner = web.AppRunner(self._app)
            self._site = None
            self._keep_alive_task = None
            self._active_connections = 0
            self._initialized = False
            self._shutdown_event = asyncio.Event()
            
            # Initialize web server components
            self._app = web.Application()
            self._runner = web.AppRunner(self._app)
            self._site = None
            self._keep_alive_task = None
            self._active_connections = 0
            self._initialized = False
            self._shutdown_event = asyncio.Event()
    
    async def _keep_alive(self):
        """Keep Rerun connection alive with periodic heartbeats"""
        retry_count = 0
        max_retries = 3
        while True:
            try:
                await asyncio.sleep(5)
                if hasattr(rr, '_recording'):
                    # Send periodic heartbeat only if already initialized
                    rr.log("world/heartbeat", rr.Timestamp(time.time_ns()))
                    retry_count = 0  # Reset retry count on successful heartbeat
            except ConnectionResetError:
                retry_count += 1
                self.logger.warning(f"Connection reset during heartbeat (attempt {retry_count}/{max_retries})")
                if retry_count >= max_retries:
                    self.logger.error("Max heartbeat retries reached, reinitializing Rerun")
                    await self._reinitialize()
                    retry_count = 0
                await asyncio.sleep(min(2 * retry_count, 30))  # Exponential backoff with 30s max
            except Exception as e:
                self.logger.error(f"Rerun keep-alive error: {e}")
                if retry_count >= max_retries:
                    self.logger.error("Max error retries reached, attempting reinitialization")
                    await self._reinitialize()
                    retry_count = 0
                else:
                    retry_count += 1
                await asyncio.sleep(min(2 * retry_count, 30))

    def initialize(self, clear_existing: bool = True):
        """Initialize Rerun recording and server with error handling"""
        try:
            # Initialize logger first
            self.logger = logging.getLogger(__name__)
            
            if hasattr(self, '_initialized') and self._initialized:
                if clear_existing:
                    try:
                        rr.log("world", rr.Clear(recursive=True))
                        self.logger.info("Cleared existing Rerun recording")
                    except Exception as e:
                        self.logger.warning(f"Failed to clear Rerun recording: {e}")
                return

            self.logger.info("Initializing Rerun...")
            
            # Verify environment
            if not self._verify_environment():
                raise RuntimeError("Environment verification failed")
            
            # Initialize recording
            rr.init("video_analytics", spawn=False)
            self.logger.info("Created new Rerun recording")
                
            # Load blueprint configuration from config
            blueprint_config = self._config.get('rerun', {}).get('blueprint', {})
            views = []
            
            for view_config in blueprint_config['views']:
                if view_config['type'] == 'spatial2d':
                    views.append(rr.blueprint.Spatial2DView(
                        origin=view_config['origin'],
                        name=view_config['name']
                    ))
                elif view_config['type'] == 'text_log':
                    views.append(rr.blueprint.TextLogView(
                        origin=view_config['origin'],
                        name=view_config['name']
                    ))
                    
            if blueprint_config['layout'] == 'vertical':
                blueprint = rr.blueprint.Vertical(*views)
            else:
                blueprint = rr.blueprint.Horizontal(*views)
            
            # Get URLs from config
            ws_url = self.urls['rerun_ws']
            web_url = self.urls['rerun_web']
            
            # Parse URLs to get host and port
            from urllib.parse import urlparse
            ws_parsed = urlparse(ws_url)
            web_parsed = urlparse(web_url)
            
            self._ws_port = ws_parsed.port or 4321  # Default WS port
            self._web_port = web_parsed.port or 9090  # Default web port
            self._ws_host = ws_parsed.hostname or 'localhost'
            self._web_host = web_parsed.hostname or 'localhost'
            
            # Start Rerun server
            rr.serve(
                open_browser=False,
                ws_port=self._ws_port,
                web_port=self._web_port,
                default_blueprint=blueprint
            )
            self.logger.info(f"Started Rerun server - WS: {self._ws_port}, Web: {self._web_port}")
            
            self._initialized = True
            self._server_started = True
            
            # Start keep-alive task
            loop = asyncio.get_event_loop()
            if self._keep_alive_task is None or self._keep_alive_task.done():
                self._keep_alive_task = loop.create_task(self._keep_alive())
                self.logger.info("Started Rerun keep-alive task")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Rerun: {e}")
            raise

    def start_web_server_sync(self):
        """Start the web server for Rerun viewer (synchronous version)"""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            if not self._runner:
                self._runner = web.AppRunner(self._app)
                loop.run_until_complete(self._runner.setup())
                self._site = web.TCPSite(self._runner, self._web_host, self._web_port)
                loop.run_until_complete(self._site.start())
                self.logger.info(f"Rerun web server started on port {self._web_port}")
                
                # Keep the server running
                loop.run_forever()
        except Exception as e:
            self.logger.error(f"Failed to start Rerun web server: {e}")
            raise
        finally:
            loop.close()

    async def cleanup(self):
        """Clean up resources and stop servers"""
        self.logger.info("Cleaning up Rerun manager...")
        
        # Signal keep-alive task to stop
        self._shutdown_event.set()
        
        # Stop keep-alive task
        if self._keep_alive_task and not self._keep_alive_task.done():
            self._keep_alive_task.cancel()
            try:
                await self._keep_alive_task
            except asyncio.CancelledError:
                pass
                
        # Stop web server
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
            
        self.logger.info("Rerun manager cleanup complete")
    
    def register_connection(self):
        """Register a new frontend connection"""
        if not hasattr(self, '_active_connections'):
            self._active_connections = 0
        self._active_connections += 1
        self.logger.info(f"Registered new connection. Active connections: {self._active_connections}")
        
        # Ensure Rerun is initialized with the connection
        if not hasattr(rr, '_recording'):
            self.initialize(clear_existing=True)
        
    def unregister_connection(self):
        """Unregister a frontend connection"""
        self._active_connections = max(0, self._active_connections - 1)
        self.logger.info(f"Unregistered connection. Active connections: {self._active_connections}")
        
    async def _reinitialize(self):
        """Reinitialize Rerun connection after failure"""
        try:
            # Clear existing state
            if hasattr(rr, '_recording'):
                delattr(rr, '_recording')
            
            # Stop existing server if running
            if hasattr(self, '_server_started'):
                delattr(self, '_server_started')
            
            # Reinitialize
            self.initialize()
            self.logger.info("Successfully reinitialized Rerun connection")
        except Exception as e:
            self.logger.error(f"Failed to reinitialize Rerun: {e}")
            raise

    def log_frame(self, frame, frame_number=None, source=None):
        """Log a frame to Rerun with metadata"""
        try:
            # Convert BGR to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame

            # Set frame sequence and log frame
            rr.set_time_sequence("frame_sequence", frame_number if frame_number is not None else 0)
            rr.log("world/video/stream", rr.Image(frame_rgb))
            
            # Log source change event if provided
            if source:
                rr.log("world/events", 
                      rr.TextLog(f"Stream source changed to: {source}"),
                      timeless=False)
                
            # Force flush to ensure frame is displayed
            rr.flush()
            
        except Exception as e:
            self.logger.error(f"Error logging frame to Rerun: {e}")
            raise

    def reset(self):
        """Clear Rerun data"""
        try:
            rr.log("world", rr.Clear(recursive=True))
            rr.log("world/events", 
                  rr.TextLog("Stream reset"),
                  timeless=False)
            self.logger.info("Rerun data cleared successfully")
        except Exception as e:
            self.logger.error(f"Failed to reset Rerun: {e}")
            raise

    @property
    def ws_url(self) -> str:
        """Get the WebSocket URL for Rerun"""
        return self.urls['rerun_ws']

    @property
    def web_url(self) -> str:
        """Get the web viewer URL"""
        return self.urls['rerun_web']
