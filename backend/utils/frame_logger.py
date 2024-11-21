import cv2
import rerun as rr
import logging
from typing import Optional, Dict, Any
import numpy as np

class FrameLogger:
    """Handles unified frame logging for both camera and video streams"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def log_frame(self, frame: np.ndarray, frame_number: int, source: str = "unknown"):
        """
        Log a frame to Rerun with metadata
        
        Args:
            frame: Input frame in BGR format
            frame_number: Sequential frame number
            source: Source identifier (e.g. "camera" or "video")
        """
        try:
            # Convert BGR to RGB for visualization
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Log frame to Rerun
            rr.set_time_sequence("frame_sequence", frame_number)
            rr.log("world/video/stream", 
                  rr.Image(frame_rgb))
            
            # Log source change event if provided
            rr.log("world/events", 
                  rr.TextLog(f"Frame from: {source}"),
                  timeless=False)
                
        except Exception as e:
            self.logger.error(f"Error logging frame: {e}", exc_info=True)
            raise
