from .config import Config
from .rerun_manager import RerunManager
from .custom_viewer import CustomViewer
import logging

class ViewerFactory:
    @staticmethod
    def get_viewer():
        config = Config()
        viewer_type = config.get('api', 'viewer', default='rerun')
        logger = logging.getLogger(__name__)
        
        logger.info(f"Creating viewer of type: {viewer_type}")
        
        if viewer_type == 'rerun':
            viewer = RerunManager()
            viewer.initialize()
            return viewer
        elif viewer_type == 'custom':
            viewer = CustomViewer()
            viewer.initialize()
            # Register WebSocket handler for custom viewer
            from .socket_handler import SocketHandler
            if hasattr(config, 'app') and config.app:
                socket_handler = SocketHandler(config.app)
                socket_handler.register_custom_viewer(viewer)
            return viewer
        else:
            logger.warning(f"Unknown viewer type: {viewer_type}, defaulting to RerunManager")
            viewer = RerunManager()
            viewer.initialize()
            return viewer
