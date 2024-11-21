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
    # Static ports for Rerun server
    WS_PORT = 4321  # WebSocket port for Rerun server
    WEB_PORT = 9090  # Web viewer port for Rerun UI
    _ws_port = WS_PORT  # WebSocket port for data streaming
    _web_port = WEB_PORT  # HTTP port for web viewer
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

    def initialize(self, clear_existing=True):
        """Initialize Rerun and optionally clear existing data"""
        try:
            self.logger.info("Initializing Rerun...")
            if not hasattr(rr, '_recording'):
                self.logger.info("Creating new Rerun recording")
                rr.init("video_analytics")#, spawn=True)
                
            if clear_existing:
                self.logger.info("Clearing existing Rerun data")
                rr.log("world", rr.Clear(recursive=True))
            
            # Always ensure server is started
            if not hasattr(self, '_server_started'):
                rr.serve(
                    open_browser=False,
                    ws_port=self._ws_port,
                    web_port=self._web_port,
                    default_blueprint=rr.blueprint.Vertical(
                        rr.blueprint.Spatial2DView(origin="world/video/stream", name="Video Feed")
                    ),
                    host="0.0.0.0"  # Allow external connections
                )
                time.sleep(2)  # Allow server to start
                self._server_started = True
                self.logger.info(f"Rerun initialized successfully on ports - WS: {self._ws_port}, Web: {self._web_port}")
                
                # Start keep-alive task in the current event loop
                loop = asyncio.get_event_loop()
                if self._keep_alive_task is None or self._keep_alive_task.done():
                    self._keep_alive_task = loop.create_task(self._keep_alive())
                
        except Exception as e:
            self.logger.error(f"Failed to initialize/clear Rerun: {e}")
            raise

    async def start_web_server(self):
        """Start the web server for Rerun viewer"""
        try:
            if not self._runner:
                self._runner = web.AppRunner(self._app)
                await self._runner.setup()
                self._site = web.TCPSite(self._runner, 'localhost', self._web_port)
                await self._site.start()
                self.logger.info(f"Rerun web server started on port {self._web_port}")
        except Exception as e:
            self.logger.error(f"Failed to start Rerun web server: {e}")
            raise

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

    def reset(self):
        """Clear Rerun data and reinitialize"""
        try:
            rr.log("world", rr.Clear(recursive=True))
            self.logger.info("Rerun data cleared successfully")
            # Force reinitialization
            asyncio.create_task(self._reinitialize())
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
