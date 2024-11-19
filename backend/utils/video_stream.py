import cv2
import time
from threading import Thread, Event
from queue import Queue
import logging
from typing import Optional, Generator
import rerun as rr

class VideoStream:
    """Handles video streaming, either from file or camera"""
    
    def __init__(self, source, loop: bool = True, buffer_size: int = 30):
        self.source = source
        self.loop = loop and isinstance(source, str)  # Only loop for file sources
        self.buffer = Queue(maxsize=buffer_size)
        
        self.stop_event = Event()
        self.frame_count = 0
        self.current_frame = None
        self.logger = logging.getLogger(__name__)
        self._stream_thread = None
        
    def start(self):
        """Start video streaming in a separate thread"""
        self._stream_thread = Thread(target=self._stream_frames, daemon=True)
        self._stream_thread.start()
        
    def _stream_frames(self):
        """Stream frames from video source"""
        while not self.stop_event.is_set():
            # Handle both string paths and VideoCapture objects
            cap = cv2.VideoCapture(self.source) if isinstance(self.source, str) else self.source
            if not cap.isOpened():
                self.logger.error(f"Failed to open video source: {self.source}")
                break
                
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    if self.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break
                        
                # Update current frame
                self.current_frame = frame
                self.frame_count += 1
                
                # Add to buffer, dropping oldest frame if full
                if self.buffer.full():
                    try:
                        self.buffer.get_nowait()
                    except:
                        pass
                # Convert BGR to RGB for rerun
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Log video to rerun
                try:
                    if not hasattr(self, '_rerun_initialized'):
                        rr.init("video_analytics", spawn=False)
                        rr.connect()
                        self._rerun_initialized = True
                        
                        if isinstance(self.source, str):
                            # For video files, use AssetVideo with looping enabled
                            video_asset = rr.AssetVideo(path=self.source)
                            rr.log("world/video", video_asset, static=True)
                            
                            # Get frame timestamps and send them
                            frame_timestamps_ns = video_asset.read_frame_timestamps_ns()
                            rr.send_columns(
                                "world/video",
                                times=[rr.TimeNanosColumn("video_time", frame_timestamps_ns)],
                                components=[
                                    rr.VideoFrameReference.indicator(),
                                    rr.components.VideoTimestamp.nanoseconds(frame_timestamps_ns)
                                ],
                            )
                    else:
                        # For live camera feed, log frames directly
                        timestamp = time.time_ns()
                        rr.log("world/video",
                              rr.Image(frame_rgb),
                              timeless=False,)
                              #timestamp=timestamp)
                except Exception as e:
                    self.logger.warning(f"Failed to log to Rerun: {e}")
                    self.logger.debug(f"Error details: {str(e)}", exc_info=True)
                
                self.buffer.put({
                    'frame': frame,
                    'timestamp': time.time(),
                    'frame_number': self.frame_count
                })
                
                # Control frame rate
                time.sleep(1/30)  # 30 FPS
                
            cap.release()
            if not self.loop:
                break
                
    def read(self) -> Optional[dict]:
        """Read the next frame"""
        if self.buffer.empty():
            return None
        return self.buffer.get()
        
    def get_frames(self, max_frames: int = 30) -> Generator[dict, None, None]:
        """Get a sequence of frames"""
        frames = []
        while len(frames) < max_frames and not self.buffer.empty():
            frames.append(self.buffer.get())
        for frame in frames:
            yield frame
            
    def stop(self):
        """Stop video streaming"""
        self.stop_event.set()
        if self._stream_thread:
            self._stream_thread.join()
