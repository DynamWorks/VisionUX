from .base_handler import BaseMessageHandler
import json
import logging
import cv2
import numpy as np
import base64
import rerun as rr
import time

class CameraStreamHandler(BaseMessageHandler):
    """Handles camera stream messages"""
    
    def __init__(self):
        super().__init__()
        
    async def handle(self, websocket, message_data):
        """Handle camera frame data"""
        try:
            if isinstance(message_data, dict) and message_data.get('type') == 'camera_frame':
                # Wait for the binary frame data
                frame_data = await websocket.recv()
                if isinstance(frame_data, bytes):
                    # Convert bytes to numpy array
                    nparr = np.frombuffer(frame_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Convert BGR to RGB for Rerun
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Log frame to Rerun
                        timestamp = message_data.get('timestamp', time.time_ns())
                        rr.log("world/video", 
                              rr.Image(frame_rgb),
                              timeless=False,
                              timestamp=timestamp)
                    else:
                        self.logger.error("Failed to decode camera frame")
                        
        except Exception as e:
            self.logger.error(f"Error handling camera frame: {e}")
            await self.send_error(websocket, str(e))
