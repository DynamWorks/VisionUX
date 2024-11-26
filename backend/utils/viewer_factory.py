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
        else:
            viewer = CustomViewer()
            viewer.initialize()
            return viewer
