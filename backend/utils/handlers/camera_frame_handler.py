import logging
import numpy as np
import cv2
import time
import rerun as rr
from .base_handler import BaseMessageHandler

class CameraFrameHandler(BaseMessageHandler):
    """Handles processing of camera frame data"""
    
    def __init__(self):
        super().__init__()
        
    async def handle(self, websocket, message_data):
        """Process incoming camera frame data"""
        try:
            if isinstance(message_data, bytes):
                success = await self.handle_frame(message_data)
                if not success:
                    await self.send_error(websocket, "Failed to process camera frame")
            else:
                self.logger.warning("Received non-binary data for camera frame")
                await self.send_error(websocket, "Invalid camera frame data")
                
        except Exception as e:
            self.logger.error(f"Error handling camera frame: {e}")
            await self.send_error(websocket, str(e))
            
    async def handle_frame(self, frame_data, metadata=None):
        """Process incoming camera frame data with metadata"""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                self.logger.debug(f"Received camera frame: {frame.shape}")
                
                # Convert BGR to RGB for Rerun
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Get timestamp from metadata or use current time
                timestamp = metadata.get('timestamp', time.time_ns()) if metadata else time.time_ns()
                
                # Log frame with metadata
                rr.log("world/video", 
                      rr.Image(frame_rgb),
                      timeless=False,
                      timestamp=timestamp)
                
                # Log additional debug info if provided
                if metadata:
                    rr.log("camera/info",
                          rr.TextLog(f"Frame size: {metadata.get('width')}x{metadata.get('height')}"),
                          timestamp=timestamp)
                
                return True
            else:
                self.logger.error("Failed to decode camera frame")
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing camera frame: {e}", exc_info=True)
            return False
