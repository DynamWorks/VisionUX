import rerun as rr
import logging
from typing import Optional
import asyncio
from aiohttp import web

class RerunManager:
    """Singleton class to manage Rerun initialization and state"""
    _instance = None
    _initialized = False
    _ws_port = 4321
    _web_port = 9090
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
    
    async def _keep_alive(self):
        """Keep Rerun connection alive"""
        while True:
            try:
                if not hasattr(rr, '_recording'):
                    self.initialize()
                await asyncio.sleep(5)  # Check every 5 seconds
            except Exception as e:
                self.logger.error(f"Rerun keep-alive error: {e}")
                await asyncio.sleep(1)  # Brief pause before retry

    def initialize(self):
        """Initialize Rerun if not already initialized"""
        try:
            if not hasattr(rr, '_recording'):
                rr.init("video_analytics", spawn=True)  # Use spawn=True for better process isolation
                rr.serve(
                    open_browser=False,
                    ws_port=self._ws_port,  # Use fixed port
                    default_blueprint=rr.blueprint.Vertical(
                        rr.blueprint.Spatial2DView(origin="world/video", name="Video Stream")
                    )
                )
                
                # Port is fixed, just log it
                self.logger.info(f"Rerun initialized successfully on port {self._ws_port}")
            else:
                self.logger.debug("Rerun already initialized")
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
    
    def reset(self):
        """Clear Rerun data and reinitialize"""
        try:
            rr.clear()
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
