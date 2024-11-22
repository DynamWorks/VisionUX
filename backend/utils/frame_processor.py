from abc import ABC, abstractmethod
import logging
from typing import Dict, Any

class FrameProcessor(ABC):
    """Base class for frame processors that can subscribe to video streams"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    @abstractmethod
    def process_frame(self, frame_data: Dict[str, Any]):
        """Process a frame from the stream"""
        pass
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        
    def cleanup(self):
        """Clean up resources"""
        pass
