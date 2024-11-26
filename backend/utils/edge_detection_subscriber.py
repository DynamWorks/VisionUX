import cv2
import numpy as np
from .frame_processor import FrameProcessor

class EdgeDetector(FrameProcessor):
    """Real-time edge detection processor"""
    
    def __init__(self):
        super().__init__()
        self.low_threshold = 100
        self.high_threshold = 200
        
    def process_frame(self, frame_data: dict) -> None:
        if not self.enabled or 'frame' not in frame_data:
            return
            
        try:
            frame = frame_data['frame']
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold)
            
            frame_data['metadata'] = frame_data.get('metadata', {})
            frame_data['metadata']['edges'] = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            
        except Exception as e:
            self.logger.error(f"Edge detection failed: {e}")
