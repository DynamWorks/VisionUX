from .base_handler import BaseMessageHandler
import json
import logging
import cv2
from pathlib import Path
from ..video_stream import VideoStream
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
        self.frame_handler = None  # Will be initialized when needed
        
    async def handle(self, websocket, message_data: Dict[str, Any]) -> None:
        """Handle incoming camera frame data"""
        try:
            message_type = message_data.get('type')

            if message_type == 'stop_video_stream':
                if hasattr(self, 'video_stream') and self.video_stream:
                    self.logger.info("Stopping video stream")
                    self.video_stream.stop()
                    delattr(self, 'video_stream')
                    await websocket.send(json.dumps({
                        'type': 'video_stream_stopped',
                        'status': 'success'
                    }))
                    self.logger.info("Video stream stopped successfully")
                return
                
            elif message_type == 'pause_video_stream':
                if hasattr(self, 'video_stream') and self.video_stream:
                    self.logger.info("Pausing video stream")
                    self.video_stream.pause()
                    await websocket.send(json.dumps({
                        'type': 'video_stream_paused',
                        'status': 'success'
                    }))
                    self.logger.info("Video stream paused successfully")
                return
                
            elif message_type == 'resume_video_stream':
                if hasattr(self, 'video_stream') and self.video_stream:
                    self.logger.info("Resuming video stream")
                    self.video_stream.resume()
                    await websocket.send(json.dumps({
                        'type': 'video_stream_resumed',
                        'status': 'success'
                    }))
                    self.logger.info("Video stream resumed successfully")
                return
            elif message_type == 'start_video_stream':
                # Handle start video stream request
                filename = message_data.get('filename')
                if not filename:
                    await self.send_error(websocket, "No filename provided")
                    return
                    
                file_path = Path("tmp_content/uploads") / filename
                if not file_path.exists():
                    await self.send_error(websocket, f"Video file not found: {filename}")
                    return
                    
                # Verify file exists and is readable
                try:
                    # Try to open video file first to verify it's valid
                    cap = cv2.VideoCapture(str(file_path))
                    if not cap.isOpened():
                        await self.send_error(websocket, f"Could not open video file: {filename}")
                        return
                    
                    # Get video properties
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    if total_frames <= 0 or fps <= 0:
                        await self.send_error(websocket, f"Invalid video file format: {filename}")
                        return
                        
                    cap.release()
                    
                    # Initialize video stream with Rerun setup
                    self.logger.info(f"Initializing video stream for {file_path}")
                    
                    # Initialize Rerun first
                    from ..rerun_manager import RerunManager
                    rerun_manager = RerunManager()
                    rerun_manager.initialize()  # Ensure Rerun is initialized
                    
                    self.video_stream = VideoStream(str(file_path))
                    self.video_stream.start()
                    self.logger.info("Video stream started successfully")
                    
                    # Log stream start event
                    rr.log("world/events",
                          rr.TextLog(f"Started streaming: {filename}"),
                          timeless=False)
                    
                    # Send success response with video properties
                    await self.send_response(websocket, {
                        'type': 'video_stream_started',
                        'filename': filename,
                        'properties': {
                            'frames': total_frames,
                            'fps': fps,
                            'width': width,
                            'height': height
                        }
                    })
                    self.logger.info(f"Started streaming video: {filename} ({total_frames} frames at {fps} FPS)")
                except Exception as e:
                    self.logger.error(f"Error starting video stream: {e}")
                    await self.send_error(websocket, f"Failed to start video stream: {str(e)}")
                return
                
            elif message_type == 'pause_video_stream':
                if hasattr(self, 'video_stream') and self.video_stream:
                    self.video_stream.pause()
                    await self.send_response(websocket, {
                        'type': 'video_stream_paused'
                    })
                return
                
            elif message_type == 'resume_video_stream':
                if hasattr(self, 'video_stream') and self.video_stream:
                    self.video_stream.resume()
                    await self.send_response(websocket, {
                        'type': 'video_stream_resumed'
                    })
                return
                
            elif message_type == 'camera_frame':
                if not isinstance(message_data, dict):
                    self.logger.warning(f"Invalid message type: {type(message_data)}")
                    return

            # Get frame data with timeout
            try:
                frame_data = await asyncio.wait_for(websocket.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                self.logger.error("Timeout waiting for frame data")
                await self.send_error(websocket, "Frame data timeout")
                return

            # Get current time for metrics
            current_time = time.time()
            process_start = current_time
                
            # Decode frame
            frame = self._decode_frame(frame_data)
            if frame is None:
                await self.send_error(websocket, "Failed to decode frame")
                return
                
            # Calculate processing time
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
            rr.log("world/video/stream", 
                  rr.Image(frame_rgb),
                  timeless=False)
            
            # Log detailed metrics
            rr.log("world/metrics/camera",
                  rr.TextLog(
                      f"Frame: {self.frame_count}\n"
                      f"Size: {metrics.width}x{metrics.height}\n"
                      f"Process time: {metrics.process_time*1000:.1f}ms\n"
                      f"Data size: {metrics.size/1024:.1f}KB"
                  ))
            
            # Log frame quality metrics
            quality_metrics = {
                "brightness": float(np.mean(frame)),
                "contrast": float(np.std(frame)),
                "blur": float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            }
            
            rr.log("world/metrics/quality",
                  rr.TextLog(
                      f"Brightness: {quality_metrics['brightness']:.1f}\n"
                      f"Contrast: {quality_metrics['contrast']:.1f}\n"
                      f"Blur: {quality_metrics['blur']:.1f}"
                  ))
            
            self.logger.debug(
                f"Processed frame {self.frame_count}: "
                f"{metrics.width}x{metrics.height} in {metrics.process_time*1000:.1f}ms"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}", exc_info=True)
            raise
