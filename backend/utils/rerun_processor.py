from .frame_processor import FrameProcessor
import cv2
from .rerun_manager import RerunManager

class RerunProcessor(FrameProcessor):
    """Processes frames and logs them to Rerun"""
    
    def __init__(self):
        super().__init__()
        self.rerun_manager = RerunManager()
        
    def process_frame(self, frame_data):
        """Log frame to Rerun"""
        try:
            frame = frame_data['frame']
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            self.rerun_manager.log_frame(
                frame=frame_rgb,
                frame_number=frame_data.get('frame_number', 0),
                source=frame_data.get('source', 'unknown')
            )
        except Exception as e:
            self.logger.error(f"Error logging to Rerun: {e}")
