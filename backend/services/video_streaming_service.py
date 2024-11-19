import cv2
import numpy as np
import rerun as rr
import time
import logging
import asyncio
from pathlib import Path
from queue import Queue
from threading import Thread, Event

class VideoStreamingService:
    def __init__(self, buffer_size: int = 30):
        self.current_video = None
        self.is_streaming = False
        self.frame_count = 0
        self.buffer = Queue(maxsize=buffer_size)
        self.stop_event = Event()
        
    async def stream_video(self, video_path):
        """Stream video file to rerun-io in a loop"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Failed to open video file: {video_path}")
            
            self.current_video = video_path    
            self.is_streaming = True
            self.frame_count = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            delay = 1 / fps if fps > 0 else 0.033  # Default to ~30fps
            
            while self.is_streaming and not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    # Reset to beginning of video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                self.frame_count += 1
                
                # Convert BGR to RGB for rerun
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Add to buffer, dropping oldest frame if full
                if self.buffer.full():
                    try:
                        self.buffer.get_nowait()
                    except:
                        pass
                
                self.buffer.put({
                    'frame': frame_rgb,
                    'timestamp': time.time(),
                    'frame_number': self.frame_count
                })
                
                # Log frame to rerun
                timestamp = time.time()
                rr.log("video/playback",
                       rr.Image(frame_rgb),
                       timeless=False,
                       timestamp=timestamp)
                
                # Delay to maintain original video fps
                await asyncio.sleep(delay)
                
        except Exception as e:
            logging.error(f"Error streaming video: {e}")
            raise
        finally:
            if cap is not None:
                cap.release()
            self.is_streaming = False
            self.current_video = None
            self.stop_event.clear()
    
    def get_current_frame(self):
        """Get the latest frame from the buffer"""
        if not self.buffer.empty():
            return self.buffer.get()
        return None
    
    def get_frame_count(self):
        """Get the total number of frames processed"""
        return self.frame_count
    
    def stop_streaming(self):
        """Stop the current video stream"""
        self.stop_event.set()
        self.is_streaming = False
