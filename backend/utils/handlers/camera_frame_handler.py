import logging
import numpy as np
import cv2
import time
import rerun as rr
from typing import Optional, Dict, Tuple
from collections import deque
from dataclasses import dataclass

@dataclass
class FrameMetrics:
    """Tracks frame processing metrics"""
    timestamp: float
    process_time: float
    size: int
    width: int 
    height: int

class CameraFrameHandler:
    """Handles frame processing and metrics"""
    
    def __init__(self, target_fps: int = 30, metrics_window: int = 100):
        self.logger = logging.getLogger(__name__)
        self.last_frame_time = 0
        self.target_fps = target_fps
        self.min_frame_interval = 1/target_fps
        self.frame_metrics = deque(maxlen=metrics_window)
        self.frame_count = 0
        self.start_time = time.time()

    def _decode_frame(self, frame_data: bytes) -> Optional[np.ndarray]:
        """Decode binary frame data to numpy array"""
        try:
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Failed to decode frame")
            return frame
        except Exception as e:
            self.logger.error(f"Frame decode error: {e}")
            return None

    def process_frame(self, frame_data: bytes, metadata: Optional[Dict] = None) -> Tuple[bool, Dict]:
        """Process a single frame and return metrics"""
        try:
            # Rate limiting check
            current_time = time.time()
            time_since_last = current_time - self.last_frame_time
            
            if time_since_last < self.min_frame_interval:
                return False, {"reason": "frame_rate_limited"}
                
            self.last_frame_time = current_time
            process_start = time.time()

            # Decode frame
            frame = self._decode_frame(frame_data)
            if frame is None:
                return False, {"reason": "decode_failed"}

            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Update metrics
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

            # Log to Rerun
            rr.log("world/video/stream", 
                  rr.Image(frame_rgb),
                  timeless=False)

            if metadata:
                rr.log("world/camera/info",
                      rr.TextLog(f"Frame size: {frame.shape[1]}x{frame.shape[0]}"),
                      timestamp=int(current_time * 1e9))

            return True, {
                'frame': frame_rgb,
                'metrics': metrics,
                'fps': fps,
                'frame_number': self.frame_count
            }

        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            return False, {"reason": str(e)}
