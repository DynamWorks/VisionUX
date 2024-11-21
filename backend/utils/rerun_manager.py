import rerun as rr
import logging
from typing import Optional
import asyncio
from aiohttp import web
import time

class RerunManager:
    """Singleton class to manage Rerun initialization and state"""
    _instance = None
    _initialized = False
    def __init__(self):
        from ..config.urls import get_urls
        urls = get_urls()
        self._ws_port = int(urls['rerun_ws'].split(':')[-1])  # Extract port from WS URL
        self._web_port = int(urls['rerun_web'].split(':')[-1])  # Extract port from web URL
    _app: Optional[web.Application] = None
    _runner: Optional[web.AppRunner] = None
    _site: Optional[web.TCPSite] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = logging.getLogger(__name__)
            self._initialized = True
            self._app = web.Application()
            self._keep_alive_task = None
            self._active_connections = 0
    
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

    def initialize(self):
        """Initialize Rerun recording and server"""
        try:
            if hasattr(self, '_initialized') and self._initialized:
                return

            self.logger.info("Initializing Rerun...")
            
            # Initialize recording if not already done
            if not hasattr(rr, '_recording'):
                self.logger.info("Creating new Rerun recording")
                rr.init("video_analytics", spawn=False)  # Don't spawn viewer
                
            # Configure default blueprint
            blueprint = rr.blueprint.Vertical(
                rr.blueprint.Spatial2DView(origin="world/video/stream", name="Video Feed"),
                rr.blueprint.TextLogView(entity="world/events", name="Events")
            )
            
            # Start server if not already running
            if not hasattr(self, '_server_started'):
                try:
                    # Start Rerun server
                    rr.serve(
                        open_browser=False,
                        ws_port=self._ws_port,
                        web_port=self._web_port,
                        default_blueprint=blueprint
                    )
                    # Wait for server to be ready
                    start_time = time.time()
                    while time.time() - start_time < 10:  # 10 second timeout
                        try:
                            import requests
                            response = requests.get(f"http://localhost:{self._web_port}/health")
                            if response.status_code == 200:
                                self._server_started = True
                                self._initialized = True
                                self.logger.info(f"Rerun initialized successfully on ports - WS: {self._ws_port}, Web: {self._web_port}")
                                break
                        except:
                            time.sleep(0.5)
                    
                    if not self._server_started:
                        raise TimeoutError("Rerun server failed to start within timeout")
                        
                except Exception as e:
                    self.logger.error(f"Failed to start Rerun server: {e}")
                    raise
                
                # Start keep-alive task in the current event loop
                loop = asyncio.get_event_loop()
                if self._keep_alive_task is None or self._keep_alive_task.done():
                    self._keep_alive_task = loop.create_task(self._keep_alive())
                
        except Exception as e:
            self.logger.error(f"Failed to initialize/clear Rerun: {e}")
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
                self._site = web.TCPSite(self._runner, 'localhost', self._web_port)
                loop.run_until_complete(self._site.start())
                self.logger.info(f"Rerun web server started on port {self._web_port}")
                
                # Keep the server running
                loop.run_forever()
        except Exception as e:
            self.logger.error(f"Failed to start Rerun web server: {e}")
            raise
        finally:
            loop.close()

    async def stop_web_server(self):
        """Stop the web server"""
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        self.logger.info("Rerun web server stopped")
    
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
        return f"ws://localhost:{self._ws_port}"

    @property
    def web_url(self) -> str:
        """Get the web viewer URL"""
        return f"http://localhost:{self._web_port}"
