import logging
from typing import Dict, Any, Optional
import time
import numpy as np
from dataclasses import dataclass, field

@dataclass
class Frame:
    """Frame data container"""
    data: np.ndarray
    timestamp: float = field(default_factory=time.time)
    frame_number: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class StreamPublisher:
    """Base class for stream publishers"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.frame_count = 0
        self.start_time = None
        self.is_publishing = False
        self.last_frame_time = None
        self.fps = 0

    def start_publishing(self) -> None:
        """Start publishing frames"""
        self.is_publishing = True
        self.start_time = time.time()
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.fps = 0
        self.logger.info("Started publishing")

    def stop_publishing(self) -> None:
        """Stop publishing frames"""
        self.is_publishing = False
        self.logger.info("Stopped publishing")

    def publish_frame(self, frame: Frame) -> None:
        """Publish a frame - to be implemented by subclasses"""
        if not self.is_publishing:
            self.start_publishing()

        self.frame_count += 1
        current_time = time.time()

        # Update FPS calculation
        if self.last_frame_time:
            time_diff = current_time - self.last_frame_time
            if time_diff >= 1.0:  # Update FPS every second
                self.fps = self.frame_count / time_diff
                self.frame_count = 0
                self.last_frame_time = current_time

        # Add performance metrics to frame metadata
        frame.metadata.update({
            'publisher_fps': round(self.fps, 2),
            'frame_count': self.frame_count,
            'publish_time': current_time
        })

        self._publish_impl(frame)

    def _publish_impl(self, frame: Frame) -> None:
        """Implementation specific publishing - to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement _publish_impl")

    def get_stats(self) -> Dict[str, Any]:
        """Get publisher statistics"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time if self.start_time else 0

        return {
            'is_publishing': self.is_publishing,
            'frame_count': self.frame_count,
            'fps': round(self.fps, 2),
            'elapsed_time': round(elapsed_time, 2),
            'last_frame_time': self.last_frame_time,
            'timestamp': current_time
        }

    def reset_stats(self) -> None:
        """Reset publisher statistics"""
        self.frame_count = 0
        self.start_time = time.time() if self.is_publishing else None
        self.last_frame_time = time.time() if self.is_publishing else None
        self.fps = 0
