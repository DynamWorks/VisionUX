from .base_handler import BaseMessageHandler
import json
import cv2
from pathlib import Path
import numpy as np
import time
import asyncio
from typing import Optional, Dict, Any
from collections import deque
from dataclasses import dataclass
from ..video_stream import VideoStream
import rerun as rr


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
                    
                    # Get RerunManager instance and terminate/restart
                    from ..rerun_manager import RerunManager
                    rerun_manager = RerunManager()
                    
                    # Cleanup existing Rerun instance
                    await rerun_manager.cleanup()
                    
                    # Start fresh Rerun instance
                    rerun_manager.initialize(clear_existing=True)
                    
                    # Send stop confirmation first
                    await websocket.send(json.dumps({
                        'type': 'video_stream_stopped',
                        'status': 'success'
                    }))
                    self.logger.info("Video stream stopped and Rerun reinitialized")
                    
                    # Wait briefly then send refresh command
                    await asyncio.sleep(1)
                    await websocket.send(json.dumps({
                        'type': 'force_refresh',
                        'timestamp': time.time()
                    }))
                    self.logger.info("Sent force refresh command")
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
                            
                    # Initialize video stream
                    self.logger.info(f"Initializing video stream for {file_path}")
                    
                    # Get RerunManager instance and reset it completely
                    from ..rerun_manager import RerunManager
                    rerun_manager = RerunManager()
                    
                    # Stop any existing recording and clear memory
                    if hasattr(rerun_manager, '_recording'):
                        delattr(rerun_manager, '_recording')
                    
                    # Reinitialize Rerun with clean state
                    rerun_manager.initialize(clear_existing=True)
                    rerun_manager.reset()
                    
                    # Initialize and start video stream
                    self.video_stream = VideoStream(str(file_path))
                    self.video_stream.start()
                    self.logger.info("Video stream started successfully")
                    
                    # Ensure video is loaded from the beginning
                    cap = cv2.VideoCapture(str(file_path))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Log initial frame
                        rerun_manager.log_frame(
                            frame=frame_rgb,
                            frame_number=0,
                            source=f"Started streaming: {filename}"
                        )
                        # Start streaming from beginning
                        self.video_stream.frame_count = 0
                        self.video_stream.add_frame({
                            'frame': frame,
                            'timestamp': time.time(),
                            'frame_number': 0
                        })
                    cap.release()
                    
                    # Log stream start event using RerunManager instance
                    rerun_manager.log_frame(
                        frame=None,
                        frame_number=0,
                        source=f"Started streaming: {filename}"
                    )
                    
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
                
                
            elif message_type == 'stop_camera_stream':
                self.logger.info("Stopping camera stream")
                await self.send_response(websocket, {
                    'type': 'camera_stream_stopped'
                })
                return
                
            elif message_type == 'camera_frame':
                try:
                    frame_data = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    if not frame_data:
                        self.logger.error("Empty frame data received")
                        return
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

            # Get RerunManager instance
            from ..rerun_manager import RerunManager
            rerun_manager = RerunManager()

            # Convert BGR to RGB for visualization
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Log frame to Rerun
            rerun_manager.log_frame(
                frame=frame_rgb,
                frame_number=self.frame_count,
                source="camera_stream"
            )

            self.frame_count += 1

            # Log basic frame info periodically
            if self.frame_count % 100 == 0:
                self.logger.info(f"Processed {self.frame_count} frames")

        except Exception as e:
            self.logger.warning(f"Failed to process frame: {e}")
            self.logger.debug(f"Error details: {str(e)}", exc_info=True)
            raise  # Re-raise to be caught by outer exception handler

        except Exception as e:
            self.logger.error(f"Error handling camera frame: {e}", exc_info=True)
            await self.send_error(websocket, str(e))
            
    def _decode_frame(self, frame_data: bytes) -> Optional[np.ndarray]:
        """
        Decode binary frame data to numpy array with validation and error handling
        
        Args:
            frame_data: Raw binary frame data
            
        Returns:
            Decoded frame as numpy array or None if decoding fails
            
        Raises:
            ValueError: If frame data is invalid or corrupted
            TypeError: If frame_data is not bytes
        """
        try:
            # Type validation
            if not isinstance(frame_data, bytes):
                raise TypeError("Frame data must be bytes")

            # Size validation with meaningful threshold
            min_frame_size = 1024  # 1KB minimum for a valid compressed frame
            if len(frame_data) < min_frame_size:
                raise ValueError(f"Frame data too small ({len(frame_data)} bytes < {min_frame_size} bytes minimum)")
                
            # Decode frame with error handling
            try:
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except ValueError as e:
                raise ValueError(f"Failed to decode frame data: {e}")
            
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
            
            # Get RerunManager instance for frame logging
            from ..rerun_manager import RerunManager
            rerun_manager = RerunManager()
            
            # Log frame using RerunManager instance
            rerun_manager.log_frame(
                frame=frame_rgb,
                frame_number=self.frame_count,
                source="camera_stream"
            )
            
            self.logger.debug(
                f"Processed frame {self.frame_count}: "
                f"{metrics.width}x{metrics.height} in {metrics.process_time*1000:.1f}ms"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}", exc_info=True)
            raise
