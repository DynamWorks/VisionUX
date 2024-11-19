import rerun as rr
import logging

class RerunManager:
    """Singleton class to manage Rerun initialization and state"""
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = logging.getLogger(__name__)
            self._initialized = True
    
    def initialize(self):
        """Initialize or reinitialize Rerun"""
        try:
            rr.init("video_analytics")
            rr.serve(
                open_browser=False,
                ws_port=4321,
                default_blueprint=rr.blueprint.Vertical(
                    rr.blueprint.Spatial2DView(origin="world/video", name="Video Stream")
                )
            )
            self.logger.info("Rerun initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Rerun: {e}")
            raise
    
    def reset(self):
        """Reset Rerun state"""
        try:
            rr.init("video_analytics")
            rr.clear()
            self.initialize()
            self.logger.info("Rerun reset successfully")
        except Exception as e:
            self.logger.error(f"Failed to reset Rerun: {e}")
            raise
