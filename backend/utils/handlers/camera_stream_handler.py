from .base_handler import BaseMessageHandler
import json
import logging
import cv2
import numpy as np
import base64
import rerun as rr
import time
import asyncio
from typing import Optional, Dict, Any
from collections import deque
from dataclasses import dataclass
from datetime import datetime

@dataclass
class FrameMetrics:
    """Tracks frame processing metrics"""
    timestamp: float
    process_time: float
    size: int
    width: int 
    height: int

class CameraStreamHandler(BaseMessageHandler):
    """Handles camera stream messages with rate limiting and metrics"""
    
    def __init__(self, target_fps: int = 30, metrics_window: int = 100):
        super().__init__()
        self.last_frame_time = 0
        self.target_fps = target_fps
        self.min_frame_interval = 1/target_fps
        self.frame_metrics = deque(maxlen=metrics_window)
        self.frame_count = 0
        self.start_time = time.time()
        
    async def handle(self, websocket, message_data: Dict[str, Any]) -> None:
        """
        Handle camera frame data with rate limiting and metrics
        
        Args:
            websocket: WebSocket connection
            message_data: Frame metadata dictionary
        """
        try:
            if not isinstance(message_data, dict) or message_data.get('type') != 'camera_frame':
                self.logger.warning(f"Invalid message type: {type(message_data)}")
                return

            # Adaptive rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_frame_time
            
            if time_since_last < self.min_frame_interval:
                # Skip frame if too soon
                self.logger.debug(f"Skipping frame - interval {time_since_last:.3f}s < {self.min_frame_interval:.3f}s")
                return
                
            self.last_frame_time = current_time
            process_start = time.time()

            # Receive frame data with timeout
            try:
                frame_data = await asyncio.wait_for(websocket.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                self.logger.error("Timeout waiting for frame data")
                await self.send_error(websocket, "Frame data timeout")
                return
            except Exception as e:
                self.logger.error(f"Failed to receive frame data: {e}")
                return

            # Validate frame data
            if not isinstance(frame_data, bytes):
                self.logger.error(f"Invalid frame data type: {type(frame_data)}")
                await self.send_error(websocket, "Invalid frame data format")
                return

            if len(frame_data) == 0:
                self.logger.error("Empty frame data received")
                await self.send_error(websocket, "Empty frame data")
                return

            # Decode and process frame
            frame = self._decode_frame(frame_data)
            if frame is None:
                await self.send_error(websocket, "Failed to decode frame")
                return

            # Track metrics
            self.frame_count += 1
            process_time = time.time() - process_start
            
            metrics = FrameMetrics(
                timestamp=current_time,
                process_time=process_time,
                size=len(frame_data),
                width=frame.shape[1],
                height=frame.shape[0]
            )
            self.frame_metrics.append(metrics)

            # Calculate FPS
            elapsed = current_time - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            
            # Log metrics periodically
            if self.frame_count % 100 == 0:
                avg_process_time = sum(m.process_time for m in self.frame_metrics) / len(self.frame_metrics)
                self.logger.info(
                    f"Processed {self.frame_count} frames, "
                    f"FPS: {fps:.1f}, "
                    f"Avg process time: {avg_process_time*1000:.1f}ms"
                )

            # Process frame with metrics
            await self._process_frame(frame, message_data, metrics)
                
        except Exception as e:
            self.logger.error(f"Error handling camera frame: {e}", exc_info=True)
            await self.send_error(websocket, str(e))
            
    def _decode_frame(self, frame_data: bytes) -> Optional[np.ndarray]:
        """
        Decode binary frame data to numpy array with validation
        
        Args:
            frame_data: Raw binary frame data
            
        Returns:
            Decoded frame as numpy array or None if decoding fails
            
        Raises:
            ValueError: If frame data is invalid
        """
        try:
            # Basic validation
            if len(frame_data) < 100:  # Arbitrary min size for valid frame
                raise ValueError("Frame data too small")
                
            # Decode frame
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise ValueError("Failed to decode frame data")
                
            # Validate frame properties
            if frame.size == 0 or len(frame.shape) != 3:
                raise ValueError(f"Invalid frame shape: {frame.shape}")
                
            if not np.isfinite(frame).all():
                raise ValueError("Frame contains invalid values")
                
            return frame
            
        except Exception as e:
            self.logger.error(f"Error decoding frame: {e}")
            return None
            
    async def _process_frame(self, frame: np.ndarray, metadata: Dict[str, Any], metrics: FrameMetrics) -> None:
        """
        Process and log frame to Rerun with enhanced metrics
        
        Args:
            frame: Decoded frame as numpy array
            metadata: Frame metadata dictionary
            metrics: Frame processing metrics
        """
        try:
            # Memory optimization - work with frame view when possible
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            timestamp = metadata.get('timestamp', time.time_ns())
            
            # Enhanced logging with metrics
            rr.log("world/video", 
                  rr.Image(frame_rgb),
                  timeless=False,
                  timestamp=timestamp)
            
            # Log detailed metrics
            rr.log("camera/metrics",
                  rr.TextLog(
                      f"Frame: {self.frame_count}\n"
                      f"Size: {metrics.width}x{metrics.height}\n"
                      f"Process time: {metrics.process_time*1000:.1f}ms\n"
                      f"Data size: {metrics.size/1024:.1f}KB"
                  ),
                  timestamp=timestamp)
            
            # Log frame quality metrics
            quality_metrics = {
                "brightness": float(np.mean(frame)),
                "contrast": float(np.std(frame)),
                "blur": float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            }
            
            rr.log("camera/quality",
                  rr.TextLog(
                      f"Brightness: {quality_metrics['brightness']:.1f}\n"
                      f"Contrast: {quality_metrics['contrast']:.1f}\n"
                      f"Blur: {quality_metrics['blur']:.1f}"
                  ),
                  timestamp=timestamp)
            
            self.logger.debug(
                f"Processed frame {self.frame_count}: "
                f"{metrics.width}x{metrics.height} in {metrics.process_time*1000:.1f}ms"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}", exc_info=True)
            raise
