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
                self.logger.info(f"Received camera frame: {frame.shape}")
                # Convert BGR to RGB for Rerun
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Log to Rerun using same topic as video stream
                timestamp = time.time()
                rr.log("world/video", 
                      rr.Image(frame_rgb),
                      timeless=False,
                      timestamp=timestamp)
            else:
                self.logger.error("Failed to decode camera frame")
                
        except Exception as e:
            self.logger.error(f"Error processing camera frame: {e}")
