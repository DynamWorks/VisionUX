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
        
        self.stop_event = Event()
        self.pause_event = Event()
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
            if self.pause_event.is_set():
                time.sleep(0.1)  # Reduce CPU usage while paused
                continue
            try:
                # Handle both string paths and VideoCapture objects
                cap = cv2.VideoCapture(self.source) if isinstance(self.source, str) else self.source
                if not cap.isOpened():
                    self.logger.error(f"Failed to open video source: {self.source}")
                    # Try with different backend
                    cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                    if not cap.isOpened():
                        self.logger.error("Failed to open video with FFMPEG backend")
                        break
                        
                # Verify video file is valid
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames <= 0:
                    self.logger.error(f"Invalid video file - no frames detected: {self.source}")
                    break
                    
                self.logger.info(f"Successfully opened video with {total_frames} frames")
                
                # Process frames
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
                    
                    # Convert BGR to RGB for visualization
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Only log to Rerun if not paused
                    if not self.pause_event.is_set():
                        try:
                            # Initialize Rerun through manager if needed
                            from .rerun_manager import RerunManager
                            rerun_manager = RerunManager()
                            rerun_manager.initialize()  # Ensure Rerun is initialized
                            
                            self.logger.debug(f"Logging frame to Rerun, shape: {frame_rgb.shape}")
                            
                            # Calculate stream time in nanoseconds
                            stream_time = int(cap.get(cv2.CAP_PROP_POS_MSEC) * 1e6)  # ms to ns
                            
                            # Log frame as a sequence with stream time
                            rr.log("world/video/stream",
                                  rr.Image(frame_rgb),
                                  timeless=False,
                                  timestamp=stream_time,
                                  sequence=self.frame_count
                                  )
                            
                            # Log frame metadata with same timestamp
                            rr.log("world/video/metadata",
                                  rr.TextLog(f"Frame {self.frame_count} - Time: {stream_time/1e9:.3f}s"),
                                  timeless=False,
                                  timestamp=stream_time,
                                  sequence=self.frame_count
                                  )
                            
                            # Force flush to ensure frames are displayed
                            rr.flush()
                        except Exception as e:
                            self.logger.warning(f"Failed to log to Rerun: {e}")
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
                cap.release()
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
        return self.buffer.get()
    
    def get_frames(self, max_frames: int = 30) -> Generator[dict, None, None]:
        """Get a sequence of frames"""
        frames = []
        while len(frames) < max_frames and not self.buffer.empty():
            frames.append(self.buffer.get())
        for frame in frames:
            yield frame
        
    def stop(self):
        """Stop video streaming and clean up resources"""
        self.logger.info("Stopping video stream...")
        self.stop_event.set()
        self.pause_event.clear()
        
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
                
        try:
            # Log stop event without clearing visualization
            rr.log("world/events", 
                  rr.TextLog("Video stream stopped"),
                  timeless=False)
            self.logger.info("Video stream stopped successfully")
        except Exception as e:
            self.logger.error(f"Error logging stop event: {e}")
            
    def pause(self):
        """Pause video streaming"""
        self.pause_event.set()
        self.logger.info("Video stream paused")
        try:
            # Log pause event to Rerun
            rr.log("world/events", 
                  rr.TextLog("Video stream paused"),
                  timeless=False)
        except Exception as e:
            self.logger.error(f"Error logging pause event: {e}")
        
    def resume(self):
        """Resume video streaming"""
        self.pause_event.clear()
        self.logger.info("Video stream resumed")
        try:
            # Log resume event to Rerun
            rr.log("world/events", 
                  rr.TextLog("Video stream resumed"),
                  timeless=False)
        except Exception as e:
            self.logger.error(f"Error logging resume event: {e}")
