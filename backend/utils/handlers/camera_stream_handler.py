import logging
import time
import cv2
import numpy as np
from typing import Any, Dict, Optional
from .base_handler import BaseHandler
from .progress_handler import ProgressHandler
from ..video_streaming.stream_manager import StreamManager
from ..video_streaming.stream_publisher import Frame

class CameraStreamHandler(BaseHandler):
    """Handler for camera streaming"""
    
    def __init__(self, progress_handler: Optional[ProgressHandler] = None):
        super().__init__("camera_stream", "stream")
        self.progress_handler = progress_handler
        self.stream_manager = StreamManager()
        self.current_stream = None
        self.frame_count = 0
        self.start_time = None
        self.logger.info("Initialized CameraStreamHandler")

    def _handle_impl(self, data: Any) -> Dict:
        """Handle incoming frame data"""
        try:
            if not self.stream_manager.is_streaming:
                self.stream_manager.start_streaming()
                self.start_time = time.time()
                self.frame_count = 0

            # Convert binary frame data to numpy array
            if isinstance(data, (bytes, bytearray)):
                # Decode JPEG frame
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            elif isinstance(data, np.ndarray):
                frame = data
            else:
                raise ValueError("Invalid frame data type")

            if frame is None:
                raise ValueError("Failed to decode frame")

            # Create frame object
            frame_obj = Frame(
                data=frame,
                timestamp=time.time(),
                frame_number=self.frame_count,
                metadata={
                    'source': 'camera',
                    'resolution': f"{frame.shape[1]}x{frame.shape[0]}",
                    'channels': frame.shape[2] if len(frame.shape) > 2 else 1
                }
            )

            # Publish frame
            self.stream_manager.publish_frame(frame_obj)
            self.frame_count += 1

            # Calculate FPS
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0

            return {
                'status': 'success',
                'frame_count': self.frame_count,
                'fps': round(fps, 2),
                'timestamp': time.time()
            }

        except Exception as e:
            self._log_error(e, "Error handling camera frame")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }

    def validate(self, data: Any) -> bool:
        """Validate frame data"""
        return data is not None

    def start_stream(self) -> Dict:
        """Start camera stream"""
        try:
            if not self.stream_manager.is_streaming:
                self.stream_manager.start_streaming()
                self.start_time = time.time()
                self.frame_count = 0
                self._set_active(True)
                return {
                    'status': 'success',
                    'message': 'Stream started',
                    'timestamp': time.time()
                }
            return {
                'status': 'warning',
                'message': 'Stream already active',
                'timestamp': time.time()
            }
        except Exception as e:
            self._log_error(e, "Error starting stream")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }

    def stop_stream(self) -> Dict:
        """Stop camera stream"""
        try:
            if self.stream_manager.is_streaming:
                self.stream_manager.stop_streaming()
                self._set_active(False)
                return {
                    'status': 'success',
                    'message': 'Stream stopped',
                    'timestamp': time.time()
                }
            return {
                'status': 'warning',
                'message': 'No active stream',
                'timestamp': time.time()
            }
        except Exception as e:
            self._log_error(e, "Error stopping stream")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }

    def pause_stream(self) -> Dict:
        """Pause camera stream"""
        try:
            if self.stream_manager.is_streaming:
                self.stream_manager.pause_streaming()
                return {
                    'status': 'success',
                    'message': 'Stream paused',
                    'timestamp': time.time()
                }
            return {
                'status': 'warning',
                'message': 'No active stream',
                'timestamp': time.time()
            }
        except Exception as e:
            self._log_error(e, "Error pausing stream")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }

    def resume_stream(self) -> Dict:
        """Resume camera stream"""
        try:
            if self.stream_manager.is_paused:
                self.stream_manager.resume_streaming()
                return {
                    'status': 'success',
                    'message': 'Stream resumed',
                    'timestamp': time.time()
                }
            return {
                'status': 'warning',
                'message': 'Stream not paused',
                'timestamp': time.time()
            }
        except Exception as e:
            self._log_error(e, "Error resuming stream")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }

    def cleanup(self) -> None:
        """Cleanup handler resources"""
        super().cleanup()
        if self.stream_manager.is_streaming:
            self.stream_manager.stop_streaming()

    def get_stream_stats(self) -> Dict:
        """Get streaming statistics"""
        if not self.stream_manager.is_streaming:
            return {
                'status': 'inactive',
                'timestamp': time.time()
            }

        elapsed_time = time.time() - self.start_time if self.start_time else 0
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0

        return {
            'status': 'active',
            'frame_count': self.frame_count,
            'elapsed_time': round(elapsed_time, 2),
            'fps': round(fps, 2),
            'is_paused': self.stream_manager.is_paused,
            'timestamp': time.time()
        }
