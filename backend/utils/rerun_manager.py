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
    
    def initialize(self):
        """Initialize Rerun if not already initialized"""
        try:
            if not hasattr(rr, '_recording'):
                rr.init("video_analytics")
                rr.serve(
                    open_browser=False,
                    ws_port=self._ws_port,
                    default_blueprint=rr.blueprint.Vertical(
                        rr.blueprint.Spatial2DView(origin="world/video", name="Video Stream")
                    )
                )
                self.logger.info("Rerun initialized successfully")
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
        """Clear Rerun data without reinitializing"""
        try:
            rr.clear()
            self.logger.info("Rerun data cleared successfully")
        except Exception as e:
            self.logger.error(f"Failed to clear Rerun data: {e}")
            raise

    @property
    def ws_url(self) -> str:
        """Get the WebSocket URL for Rerun"""
        return f"ws://localhost:{self._ws_port}"

    @property
    def web_url(self) -> str:
        """Get the web viewer URL"""
        return f"http://localhost:{self._web_port}"
