from .frame_processor import FrameProcessor
import cv2
from .rerun_manager import RerunManager

class RerunProcessor(FrameProcessor):
    """Processes frames and logs them to Rerun"""
    
    def __init__(self):
        super().__init__()
        self.rerun_manager = RerunManager()
        self.frame_count = 0
        
    def process_frame(self, frame_data):
        """Log frame to Rerun"""
        try:
            frame = frame_data['frame']
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Use frame number from data if available, otherwise increment local counter
            frame_number = frame_data.get('frame_number', self.frame_count)
            
            self.rerun_manager.log_frame(
                frame=frame_rgb,
                frame_number=frame_number,
                source=frame_data.get('source', 'unknown')
            )
            
            self.frame_count = frame_number + 1
            
        except Exception as e:
            self.logger.error(f"Error logging to Rerun: {e}")
