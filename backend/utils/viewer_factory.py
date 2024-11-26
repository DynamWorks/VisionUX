from .config import Config
from .rerun_manager import RerunManager
from .custom_viewer import CustomViewer

class ViewerFactory:
    @staticmethod
    def get_viewer():
        config = Config()
        viewer_type = config.get('api', 'viewer', default='rerun')
        
        if viewer_type == 'rerun':
            return RerunManager()
        else:
            return CustomViewer()
