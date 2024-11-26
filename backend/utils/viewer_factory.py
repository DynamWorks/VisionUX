from .custom_viewer import CustomViewer
import logging

class ViewerFactory:
    """Factory for creating frame viewers"""
    
    _instance = None
    
    @classmethod
    def get_viewer(cls):
        if cls._instance is None:
            cls._instance = CustomViewer()
            cls._instance.initialize()
        return cls._instance
