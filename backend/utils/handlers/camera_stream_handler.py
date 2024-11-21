from .base_handler import BaseMessageHandler
import json
import logging
import cv2
import numpy as np
import base64
import rerun as rr
import time
from typing import Optional

class CameraStreamHandler(BaseMessageHandler):
    """Handles camera stream messages"""
    
    def __init__(self):
        super().__init__()
        self.last_frame_time = 0
        self.min_frame_interval = 1/30  # Limit to 30 FPS
        
    async def handle(self, websocket, message_data):
        """Handle camera frame data"""
        try:
            if isinstance(message_data, dict) and message_data.get('type') == 'camera_frame':
                # Rate limiting
                current_time = time.time()
                if current_time - self.last_frame_time < self.min_frame_interval:
                    return
                self.last_frame_time = current_time
                
                # Wait for the binary frame data with timeout
                try:
                    frame_data = await websocket.recv()
                except Exception as e:
                    self.logger.error(f"Failed to receive frame data: {e}")
                    return
                
                if not isinstance(frame_data, bytes):
                    self.logger.error(f"Invalid frame data type: {type(frame_data)}")
                    await self.send_error(websocket, "Invalid frame data format")
                    return
                
                frame = self._decode_frame(frame_data)
                if frame is None:
                    await self.send_error(websocket, "Failed to decode frame")
                    return
                
                # Process and log frame
                await self._process_frame(frame, message_data)
                
        except Exception as e:
            self.logger.error(f"Error handling camera frame: {e}", exc_info=True)
            await self.send_error(websocket, str(e))
            
    def _decode_frame(self, frame_data: bytes) -> Optional[np.ndarray]:
        """Decode binary frame data to numpy array"""
        try:
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                self.logger.error("Failed to decode frame data")
                return None
            return frame
        except Exception as e:
            self.logger.error(f"Error decoding frame: {e}")
            return None
            
    async def _process_frame(self, frame: np.ndarray, metadata: dict):
        """Process and log frame to Rerun"""
        try:
            # Convert BGR to RGB for Rerun
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get frame metadata
            timestamp = metadata.get('timestamp', time.time_ns())
            width = metadata.get('width', frame.shape[1])
            height = metadata.get('height', frame.shape[0])
            
            # Log frame and metadata to Rerun
            rr.log("world/video", 
                  rr.Image(frame_rgb),
                  timeless=False,
                  timestamp=timestamp)
            
            # Log frame metadata
            rr.log("camera/info",
                  rr.TextLog(f"Frame size: {width}x{height}"),
                  timestamp=timestamp)
            
            self.logger.debug(f"Processed frame: {width}x{height} at {timestamp}")
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            raise
