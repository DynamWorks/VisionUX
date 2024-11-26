from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
from dataclasses import dataclass

@dataclass
class Frame:
    data: np.ndarray
    timestamp: float
    frame_number: int
    metadata: Dict = None

class StreamPublisher(ABC):
    """Base class for video stream publishers"""
    
    @abstractmethod
    def publish_frame(self, frame: Frame) -> None:
        """Publish a frame to connected clients"""
        pass
        
    def cleanup(self) -> None:
        """Clean up resources"""
        pass
