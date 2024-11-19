import cv2
import numpy as np
import rerun as rr
import time
import logging
import asyncio
from pathlib import Path

class VideoStreamingService:
    def __init__(self):
        self.current_video = None
        self.is_streaming = False
        
    async def stream_video(self, video_path):
        """Stream video file to rerun-io in a loop"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Failed to open video file: {video_path}")
                
            self.is_streaming = True
            fps = cap.get(cv2.CAP_PROP_FPS)
            delay = 1 / fps if fps > 0 else 0.033  # Default to ~30fps
            
            while self.is_streaming:
                ret, frame = cap.read()
                if not ret:
                    # Reset to beginning of video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                    
                # Convert BGR to RGB for rerun
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
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
            
    def stop_streaming(self):
        """Stop the current video stream"""
        self.is_streaming = False
