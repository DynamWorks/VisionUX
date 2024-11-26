import cv2
import numpy as np
from .frame_processor import FrameProcessor
import rerun as rr

class EdgeDetectionSubscriber(FrameProcessor):
    """Processes frames for edge detection visualization"""
    
    def __init__(self, low_threshold=100, high_threshold=200):
        super().__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.enabled = False
        
    def process_frame(self, frame_data):
        """Process frame for edge detection"""
        if not self.enabled:
            return
            
        try:
            frame = frame_data['frame']
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Canny edge detection
            edges = cv2.Canny(blurred, 
                            self.low_threshold, 
                            self.high_threshold)
                            
            # Convert back to RGB for visualization
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            
            # Log to Rerun
            rr.log("world/video/edges", 
                   rr.Image(edges_rgb),
                   timeless=False)
                   
        except Exception as e:
            self.logger.error(f"Edge detection error: {e}")
            
    def toggle(self):
        """Toggle edge detection on/off"""
        self.enabled = not self.enabled
        self.logger.info(f"Edge detection {'enabled' if self.enabled else 'disabled'}")
