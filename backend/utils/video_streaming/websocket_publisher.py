import cv2
import numpy as np
from .stream_publisher import StreamPublisher, Frame
import logging
import json
import asyncio
from typing import Optional, Dict, Any

class WebSocketPublisher(StreamPublisher):
    """Publishes frames to WebSocket clients"""
    
    def __init__(self, websocket=None):
        self.websocket = websocket
        self.logger = logging.getLogger(__name__)
        self.enabled = True
        self._frame_count = 0
        
    def set_websocket(self, websocket):
        """Set or update WebSocket connection"""
        self.websocket = websocket
        
    async def publish_frame(self, frame: Frame) -> None:
        """Publish frame to WebSocket clients"""
        if not self.enabled or not self.websocket:
            return
            
        try:
            # Convert frame to JPEG
            success, buffer = cv2.imencode('.jpg', frame.data)
            if not success:
                raise ValueError("Failed to encode frame")
                
            # Send frame metadata
            metadata = {
                'type': 'frame',
                'timestamp': frame.timestamp,
                'frame_number': frame.frame_number,
                'metadata': frame.metadata
            }
            
            try:
                await self.websocket.send(json.dumps(metadata))
            except Exception as e:
                self.logger.error(f"Error sending frame metadata: {e}")
                return
                
            # Send binary frame data
            try:
                await self.websocket.send(buffer.tobytes())
            except Exception as e:
                self.logger.error(f"Error sending frame data: {e}")
                return
                
            self._frame_count += 1
            
            # Log streaming stats periodically
            if self._frame_count % 100 == 0:
                self.logger.info(f"Streamed {self._frame_count} frames")
                
        except Exception as e:
            self.logger.error(f"Error publishing frame: {e}")
            
    def cleanup(self) -> None:
        """Clean up resources"""
        self.enabled = False
        self.websocket = None
