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
    WS_PORT = 9090  # WebSocket port
    WEB_PORT = 9091  # Web viewer port
    _ws_port = WS_PORT
    _web_port = WEB_PORT
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
        """Keep Rerun connection alive as long as frontend exists"""
        while True:
            try:
                if not hasattr(rr, '_recording'):
                    self.initialize()
                # Check connection status more frequently
                await asyncio.sleep(5)  # Check every 5 seconds
                
                # If no active WebSocket connections, clear logs but keep server running
                if not hasattr(self, '_active_connections') or self._active_connections == 0:
                    rr.Clear(recursive=True)
                    self.logger.debug("No active connections, cleared Rerun logs")
                    
            except Exception as e:
                self.logger.error(f"Rerun keep-alive error: {e}")
                await asyncio.sleep(2)  # Brief pause before retry

    def initialize(self):
        """Initialize Rerun if not already initialized"""
        try:
            if not hasattr(rr, '_recording'):
                rr.init("video_analytics", spawn=True)  # Spawn new process to keep server alive
                try:
                    if not hasattr(self, '_server_started'):
                        rr.serve(
                            open_browser=False,
                            ws_port=self._ws_port,
                            default_blueprint=rr.blueprint.Vertical(
                                rr.blueprint.Spatial2DView(origin="world/video", name="Video Stream")
                            ))
                        # Add delay to ensure server is ready
                        time.sleep(2)
                        self._server_started = True
                        self.logger.info(f"Rerun initialized successfully on port {self._ws_port}")
                except Exception as port_error:
                    if "address already in use" in str(port_error).lower():
                        self.logger.info("Rerun server already running, continuing with existing instance")
                        self._server_started = True
                    else:
                        raise
            else:
                # Just clear the recording state without restarting server
                rr.Clear(recursive=True)
                self.logger.debug("Cleared Rerun logs while maintaining existing connection")
        except Exception as e:
            self.logger.error(f"Failed to initialize Rerun: {e}")
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
        
        # Ensure Rerun is initialized with new connection
        if not hasattr(rr, '_recording'):
            self.initialize()
        
    def unregister_connection(self):
        """Unregister a frontend connection"""
        self._active_connections = max(0, self._active_connections - 1)
        self.logger.info(f"Unregistered connection. Active connections: {self._active_connections}")
        
    def reset(self):
        """Clear Rerun data and reinitialize"""
        try:
            rr.Clear(recursive=True)
            self.logger.info("Rerun data cleared successfully")
            # Force reinitialization
            if hasattr(rr, '_recording'):
                delattr(rr, '_recording')
            self.initialize()
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
