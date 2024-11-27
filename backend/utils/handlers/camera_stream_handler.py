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
    
    def __init__(self, progress_handler: Optional[ProgressHandler] = None, camera_index: int = 0):
        super().__init__("camera_stream", "stream")
        self.progress_handler = progress_handler
        self.stream_manager = StreamManager()
        self.camera_index = camera_index
        self.available_cameras = self._get_available_cameras()
        self.capture = None
        self.frame_count = 0
        self.start_time = None
        self.client_keepalive = {}  # Track client keepalive status
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

    def _get_available_cameras(self) -> Dict[int, str]:
        """Get list of available camera devices"""
        available_cameras = {}
        for i in range(10):  # Check first 10 indexes
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    # Try to get the camera name/description
                    name = f"Camera {i}"
                    try:
                        name = cap.getBackendName()
                    except:
                        pass
                    available_cameras[i] = name
                cap.release()
        return available_cameras

    def get_available_cameras(self) -> Dict[int, str]:
        """Return list of available cameras"""
        return self.available_cameras

    def set_camera_device(self, device_index: int) -> Dict:
        """Set the camera device to use"""
        if device_index not in self.available_cameras:
            return {
                'status': 'error',
                'message': f'Camera device {device_index} not available',
                'available_devices': self.available_cameras
            }
        
        was_streaming = self.stream_manager.is_streaming
        if was_streaming:
            self.stop_stream()
            
        self.camera_index = device_index
        
        if was_streaming:
            return self.start_stream()
        return {
            'status': 'success',
            'message': f'Camera device set to {device_index}',
            'device_name': self.available_cameras[device_index]
        }

    def start_stream(self) -> Dict:
        """Start camera stream"""
        try:
            if not self.stream_manager.is_streaming:
                # Initialize video capture
                self.capture = cv2.VideoCapture(self.camera_index)
                if not self.capture.isOpened():
                    available = self._get_available_cameras()
                    raise RuntimeError(
                        f"Failed to open camera {self.camera_index}. "
                        f"Available cameras: {available}"
                    )
                
                # Set camera properties
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.capture.set(cv2.CAP_PROP_FPS, 30)
                self.capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                
                self.stream_manager.start_streaming()
                self.start_time = time.time()
                self.frame_count = 0
                self._set_active(True)
                
                # Start capture loop in a separate thread
                import threading
                self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
                self.capture_thread.start()
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
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def add_client(self, client_id: str) -> None:
        """Add a new streaming client"""
        self.stream_manager.add_client(client_id)
        self.client_keepalive[client_id] = time.time()
        
    def remove_client(self, client_id: str) -> None:
        """Remove a streaming client"""
        self.stream_manager.remove_client(client_id)
        self.client_keepalive.pop(client_id, None)
        
    def update_client_keepalive(self, client_id: str) -> None:
        """Update client keepalive timestamp"""
        self.client_keepalive[client_id] = time.time()
        
    def cleanup_stale_clients(self) -> None:
        """Remove clients that haven't sent keepalive in 2 minutes"""
        current_time = time.time()
        stale_clients = [
            client_id for client_id, last_seen in self.client_keepalive.items()
            if current_time - last_seen > 120  # 2 minutes timeout
        ]
        for client_id in stale_clients:
            self.remove_client(client_id)
            
    def _capture_loop(self):
        """Continuous capture loop for camera frames"""
        while self.stream_manager.is_streaming and self.capture and self.capture.isOpened():
            if not self.stream_manager.is_paused:
                ret, frame = self.capture.read()
                if ret:
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
                    self.stream_manager.publish_frame(frame_obj)
                    self.frame_count += 1
                else:
                    self.logger.error("Failed to read frame from camera")
                    break
            time.sleep(1/30)  # Limit to ~30 FPS
            
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
