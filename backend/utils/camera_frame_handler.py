import logging
import numpy as np
import cv2
import time
import rerun as rr
from .rerun_manager import RerunManager

class CameraFrameHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def handle_frame(self, frame_data):
        """Process incoming camera frame data"""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                self.logger.debug(f"Received camera frame: {frame.shape}")  # Changed to debug to reduce log spam
                # Convert BGR to RGB for Rerun
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Log to Rerun using camera topic
                timestamp = time.time_ns()  # Use nanosecond precision
                rr.log("world/video", 
                      rr.Image(frame_rgb),
                      timeless=False,
                      timestamp=timestamp)
                
                # Send frame received acknowledgment
                return True
            else:
                self.logger.error("Failed to decode camera frame")
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing camera frame: {e}")
            return False
