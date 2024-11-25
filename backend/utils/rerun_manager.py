import rerun as rr
import logging
from typing import Optional
import asyncio
from aiohttp import web
import time
import cv2
import os
import json
from pathlib import Path
from .config import Config

class RerunManager:
    """Singleton class to manage Rerun initialization and state"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.logger = logging.getLogger(__name__)
            
            # Load config
            self.config = Config()
            self._config = self.config._config  # Store config dict directly
            
            # Initialize state
            self._server_started = False
            self._active_connections = 0
            self._shutdown_event = None
            self._initialized = True
            self._runner = None
            self._site = None
            self._app = None
            self._keep_alive_task = None
            
            # Initialize web server attributes
            rerun_config = self._config.get('rerun', {})
            self._web_host = rerun_config.get('web_host', 'localhost')
            self._web_port = int(rerun_config.get('web_port', 9090))
            self._ws_host = rerun_config.get('ws_host', 'localhost')
            self._ws_port = int(rerun_config.get('ws_port', 4321))

    async def check_health(self):
        """Explicit health check method"""
        try:
            return {
                'status': 'healthy',
                'service': 'rerun-viewer',
                'web_port': self._web_port,
                'ws_port': self._ws_port,
                'timestamp': time.time()
            }
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    async def _handle_root(self, request):
        """Handle root path request"""
        from aiohttp import web
        return web.Response(text='Rerun server is running', headers={'Content-Type': 'text/plain'})
            
    def _verify_environment(self) -> bool:
        """Verify required environment settings"""
        try:
            import os  # Import os at the start of the method
            
            # Initialize ports from config if not already set
            if not hasattr(self, '_ws_port'):
                self._ws_port = int(os.getenv('VIDEO_ANALYTICS_RERUN_WS_PORT', 4321))
            if not hasattr(self, '_web_port'):
                self._web_port = int(os.getenv('VIDEO_ANALYTICS_RERUN_WEB_PORT', 9090))

            # Create required directories first
            base_dir = Path('tmp_content')
            base_dir.mkdir(parents=True, exist_ok=True)
            
            for subdir in ['uploads', 'analysis', 'chat_history']:
                (base_dir / subdir).mkdir(exist_ok=True)
                
            # Get host values from config/env
            from backend.utils.config import Config
            config = Config()
            
            self._ws_host = os.getenv('VIDEO_ANALYTICS_RERUN_WS_HOST', 
                                    config.get('rerun', 'host', default='localhost'))
            self._web_host = os.getenv('VIDEO_ANALYTICS_RERUN_WEB_HOST',
                                     config.get('rerun', 'host', default='localhost'))
            
            # Don't check ports - Rerun will handle port conflicts
            self.logger.info(f"Environment verified. WS Port: {self._ws_port}, Web Port: {self._web_port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Environment verification failed: {e}")
            return False
    
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
            
            # if hasattr(self, '_initialized') and self._initialized:
            #     if clear_existing:
            #         try:
            #             rr.log("world", rr.Clear(recursive=True))
            #             self.logger.info("Cleared existing Rerun recording")
            #         except Exception as e:
            #             self.logger.warning(f"Failed to clear Rerun recording: {e}")
            #     return

            if hasattr(self, '_initialized') and self._initialized and not clear_existing:
                self.logger.info("Rerun already initialized")
                return
                
            self.logger.info("Initializing Rerun...")
            
            # Load and verify config first
            if not hasattr(self, '_config') or not self._config:
                self.config = Config()
                self._config = self.config._config
            
            # Get config values with validation
            rerun_config = self._config.get('rerun', {})
            if not rerun_config:
                raise ValueError("Missing 'rerun' section in config")
                
            # Set required attributes from config with validation
            try:
                self._ws_port = int(rerun_config.get('ws_port', 4321))
                self._web_port = int(rerun_config.get('web_port', 9090))
                if not (1024 <= self._ws_port <= 65535 and 1024 <= self._web_port <= 65535):
                    raise ValueError("Port numbers must be between 1024 and 65535")
            except ValueError as e:
                raise ValueError(f"Invalid port configuration: {str(e)}")
                
            self._ws_host = rerun_config.get('ws_host', 'localhost')
            self._web_host = rerun_config.get('web_host', 'localhost')
            
            # Verify environment after config is loaded
            if not self._verify_environment():
                raise RuntimeError("Environment verification failed")
            
            # Initialize recording if not already done
            if not hasattr(rr, '_recording'):
                rr.init("video_analytics")
                rr.set_time_sequence("frame_sequence", sequence=0)  # Initialize sequence at 0
                rr.set_time_seconds("frame_sequence", seconds=time.time())  # Set current time with timeline
                rr.set_time_playback_mode("live")  # Start in live mode
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
            from backend.config.urls import get_urls
            urls = get_urls()
            
            # Parse URLs to get host and port
            from urllib.parse import urlparse
            ws_parsed = urlparse(urls['rerun_ws'])
            web_parsed = urlparse(urls['rerun_web'])
            
            self._ws_port = ws_parsed.port or self._ws_port  # Use existing port if URL parsing fails
            self._web_port = web_parsed.port or self._web_port
            self._ws_host = ws_parsed.hostname or self._ws_host
            self._web_host = web_parsed.hostname or self._web_host
            
            # Start Rerun server with SDK
            rr.init("video_analytics")
            rr.serve(
                open_browser=False,
                ws_port=self._ws_port,
                web_port=self._web_port,
                default_blueprint=blueprint,
                default_playback_mode="live",  # Always start in live mode
                time_sequence_id="frame_sequence"  # Use our frame sequence
            )
            self.logger.info(f"Started Rerun server - WS: {self._ws_port}, Web: {self._web_port}")
            
            self._initialized = True
            self._server_started = True
            rr.log("world", rr.Clear(recursive=True))
            
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
            from aiohttp import web
            
            # Create the web application if not already created
            if not self._app:
                self._app = web.Application()
                self._app.router.add_route('GET', '/', self._handle_root)
                self._app.router.add_route('OPTIONS', '/health', self._health_check)
                self._app.router.add_route('GET', '/health', self._health_check)
            
            async def start_site():
                if not self._runner:
                    self._runner = web.AppRunner(self._app)
                    await self._runner.setup()
                    self._site = web.TCPSite(self._runner, self._web_host, self._web_port)
                    await self._site.start()
                    self.logger.info(f"Rerun web server started on port {self._web_port}")
            
            # Set up and run event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Start the site
            loop.run_until_complete(start_site())
            
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
        
        try:
            # Signal keep-alive task to stop
            if hasattr(self, '_shutdown_event'):
                self._shutdown_event.set()
            
            # Stop keep-alive task
            if hasattr(self, '_keep_alive_task') and self._keep_alive_task and not self._keep_alive_task.done():
                self._keep_alive_task.cancel()
                try:
                    await asyncio.wait_for(self._keep_alive_task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    self.logger.warning("Keep-alive task cleanup timed out")
                    
            # Stop web server with timeout
            if hasattr(self, '_site') and self._site:
                try:
                    await asyncio.wait_for(self._site.stop(), timeout=5.0)
                except asyncio.TimeoutError:
                    self.logger.warning("Web server stop timed out")
                    
            if hasattr(self, '_runner') and self._runner:
                try:
                    await asyncio.wait_for(self._runner.cleanup(), timeout=5.0)
                except asyncio.TimeoutError:
                    self.logger.warning("Runner cleanup timed out")
            
            # Reset initialization flag
            self._initialized = False
            
            # Clear Rerun recording
            try:
                rr.log("world", rr.Clear(recursive=True))
            except Exception as e:
                self.logger.warning(f"Failed to clear Rerun recording: {e}")
                
            self.logger.info("Rerun manager cleanup complete")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            raise
    
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
            #self.initialize()
            self.logger.info("Successfully reinitialized Rerun connection")
        except Exception as e:
            self.logger.error(f"Failed to reinitialize Rerun: {e}")
            raise

    def log_frame(self, frame, frame_number=None, source=None):
        """Log a frame to Rerun with metadata"""
        try:
            if frame is None:
                self.logger.warning("Received None frame")
                return

            if not hasattr(rr, '_recording'):
                self.logger.warning("Rerun not initialized, initializing now")
                self.initialize(clear_existing=False)

            # Input validation
            if not isinstance(frame_number, (type(None), int)):
                raise ValueError(f"Invalid frame_number type: {type(frame_number)}")

            # Convert BGR to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except cv2.error as e:
                    self.logger.error(f"Color conversion failed: {e}")
                    return
            else:
                frame_rgb = frame

            # Set frame sequence and log frame
            try:
                rr.set_time_sequence("frame_sequence", frame_number if frame_number is not None else 0)
                rr.log("world/video/stream", rr.Image(frame_rgb))
            except Exception as e:
                self.logger.error(f"Failed to log frame to Rerun: {e}")
                return
            
            # Log source change event if provided
            if source:
                try:
                    rr.log("world/events", 
                          rr.TextLog(f"Stream source changed to: {source}"),
                          timeless=False)
                except Exception as e:
                    self.logger.warning(f"Failed to log source change event: {e}")
            
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
