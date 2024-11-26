from abc import ABC, abstractmethod
import logging
from typing import Dict, Any

class FrameProcessor(ABC):
    """Base class for frame processing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.enabled = True
        
    @abstractmethod
    def process_frame(self, frame_data: Dict[str, Any]) -> None:
        """Process a single frame"""
        pass
        
    def cleanup(self) -> None:
        """Clean up resources"""
        pass
