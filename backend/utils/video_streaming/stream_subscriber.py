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

class StreamSubscriber(ABC):
    """Base class for video stream subscribers"""
    
    @abstractmethod
    def on_frame(self, frame: Frame) -> None:
        """Called when a new frame is available"""
        pass
        
    def cleanup(self) -> None:
        """Clean up resources"""
        pass
