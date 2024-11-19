import logging
import numpy as np
import cv2
import rerun as rr
import time

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
                
                # Log to Rerun
                timestamp = time.time()
                rr.log("camera/original", 
                      rr.Image(frame_rgb),
                      timeless=False,
                      timestamp=timestamp)
            else:
                self.logger.error("Failed to decode camera frame")
                
        except Exception as e:
            self.logger.error(f"Error processing camera frame: {e}")
