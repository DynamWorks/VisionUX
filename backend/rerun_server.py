import asyncio
import rerun as rr
from aiohttp import web
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RerunServer:
    def __init__(self, port=9090):
        self.port = port
        self.app = web.Application()
        self.runner = None
        self.site = None
        
        # Initialize Rerun
        rr.init("video_analytics")
        rr.serve(open_browser=False, ws_port=4321, skip_welcome=True)
        
    async def start(self):
        try:
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            self.site = web.TCPSite(self.runner, 'localhost', self.port)
            await self.site.start()
            logger.info(f"Rerun server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start Rerun server: {e}")
            raise

    async def stop(self):
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        logger.info("Rerun server stopped")

def run_server(port=9090):
    server = RerunServer(port)
    
    async def start_server():
        await server.start()
        try:
            await asyncio.Future()  # run forever
        finally:
            await server.stop()
    
    asyncio.run(start_server())

if __name__ == "__main__":
    run_server()
