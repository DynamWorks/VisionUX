import cv2
import time
from threading import Thread, Event
from queue import Queue
import logging
from typing import Optional, Generator
import rerun as rr

class VideoStream:
    """Handles video streaming, either from file or camera"""
    
    def __init__(self, source, loop: bool = True, buffer_size: int = 10):
        self.source = source
        self.loop = loop and isinstance(source, str)  # Only loop for file sources
        self.buffer = Queue(maxsize=buffer_size)
        self.subscribers = set()  # For frame processors
        
        self.stop_event = Event()
        self.pause_event = Event()
        self.frame_count = 0
        self.current_frame = None
        self.paused_position = 0  # Store frame position when paused
        self.logger = logging.getLogger(__name__)
        self._stream_thread = None
        self._cap = None  # Store VideoCapture object
        
    def start(self):
        """Start video streaming in a separate thread"""
        self._stream_thread = Thread(target=self._stream_frames, daemon=True)
        self._stream_thread.start()
        
    def _stream_frames(self):
        """Stream frames from video source"""
        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                time.sleep(0.1)  # Reduce CPU usage while paused
                continue
            try:
                # Handle both string paths and VideoCapture objects
                self._cap = cv2.VideoCapture(self.source) if isinstance(self.source, str) else self.source
                if not self._cap.isOpened():
                    self.logger.error(f"Failed to open video source: {self.source}")
                    # Try with different backend
                    self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                    if not self._cap.isOpened():
                        self.logger.error("Failed to open video with FFMPEG backend")
                        break
                        
                # Verify video file is valid
                total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames <= 0:
                    self.logger.error(f"Invalid video file - no frames detected: {self.source}")
                    break
                    
                self.logger.info(f"Successfully opened video with {total_frames} frames")
                
                # Process frames
                while not self.stop_event.is_set():
                    ret, frame = self._cap.read()
                    if not ret:
                        if self.loop:
                            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
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
                    
                    # Only log frames when actively streaming (not paused or stopped)
                    if not self.pause_event.is_set() and not self.stop_event.is_set():
                        try:
                            # Convert BGR to RGB for visualization
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            
                            # Get RerunManager instance
                            from .rerun_manager import RerunManager
                            rerun_manager = RerunManager()
                            
                            # Log frame using RerunManager
                            rerun_manager.log_frame(
                                frame=frame_rgb,
                                frame_number=self.frame_count,
                                source=str(self.source)
                            )
                            
                            # Log streaming event
                            rr.log("world/events", 
                                  rr.TextLog(f"Streaming frame {self.frame_count}"),
                                  timeless=False)
                            
                        except Exception as e:
                            self.logger.warning(f"Failed to log frame: {e}")
                            self.logger.debug(f"Error details: {str(e)}", exc_info=True)
                    
                    self.buffer.put({
                        'frame': frame,
                        'timestamp': time.time(),
                        'frame_number': self.frame_count
                    })
                    
                    # Control frame rate and add delay if buffer is getting full
                    if self.buffer.qsize() > self.buffer.maxsize * 0.8:  # Buffer over 80% full
                        time.sleep(1/15)  # Reduce to 15 FPS when buffer filling up
                    else:
                        time.sleep(1/30)  # Normal 30 FPS
                
                # Release capture when done with this loop iteration
                self._cap.release()
                if not self.loop:
                    break

            except Exception as e:
                self.logger.error(f"Error in video streaming: {e}")
                if not self.loop:
                    break
                time.sleep(1)  # Wait before retry

    def read(self) -> Optional[dict]:
        """Read the next frame"""
        if self.buffer.empty():
            return None
        frame_data = self.buffer.get()
        
        # Notify subscribers
        for subscriber in self.subscribers:
            try:
                subscriber.process_frame(frame_data)
            except Exception as e:
                self.logger.error(f"Subscriber error: {e}")
                
        return frame_data
        
    def add_subscriber(self, subscriber):
        """Add a frame processing subscriber"""
        self.subscribers.add(subscriber)
        
    def remove_subscriber(self, subscriber):
        """Remove a frame processing subscriber"""
        self.subscribers.discard(subscriber)
        
    def add_frame(self, frame_data: dict):
        """Add a frame to the stream"""
        if self.buffer.full():
            try:
                self.buffer.get_nowait()
            except:
                pass
        self.buffer.put(frame_data)
    
    def get_frames(self, max_frames: int = 30) -> Generator[dict, None, None]:
        """Get a sequence of frames"""
        frames = []
        while len(frames) < max_frames and not self.buffer.empty():
            frames.append(self.buffer.get())
        for frame in frames:
            yield frame
        
    def stop(self):
        """Stop video streaming and reset to beginning"""
        self.logger.info("Stopping video stream...")
        self.stop_event.set()
        self.pause_event.clear()
        self.paused_position = 0
        
        # Wait for thread to finish with timeout
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=2.0)
            if self._stream_thread.is_alive():
                self.logger.warning("Stream thread did not stop within timeout")
            
        # Clear buffer
        while not self.buffer.empty():
            try:
                self.buffer.get_nowait()
            except:
                pass
                
        # Get RerunManager instance
        from .rerun_manager import RerunManager
        rerun_manager = RerunManager()
        
        # Clear Rerun recording and reinitialize
        rerun_manager.initialize(clear_existing=True)
        
        # Reset video capture to beginning
        if self._cap and self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
        self.logger.info("Video stream stopped and Rerun buffer cleared")
            
