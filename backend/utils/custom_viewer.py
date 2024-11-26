import cv2
import numpy as np
import logging
from typing import Optional, Dict, Any
import time
from pathlib import Path
import json

class CustomViewer:
    """Custom viewer implementation alternative to Rerun"""
    
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.logger = logging.getLogger(__name__)
            self.frame_count = 0
            self.output_dir = Path("tmp_content/viewer_output")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._initialized = True
            self._recording = True
            
    def initialize(self, clear_existing: bool = True):
        """Initialize viewer"""
        if clear_existing:
            for f in self.output_dir.glob("*"):
                f.unlink()
        self._recording = True
        self.frame_count = 0
        
    def log_frame(self, frame: np.ndarray, frame_number: Optional[int] = None, 
                  source: Optional[str] = None):
        """Log frame with metadata"""
        try:
            if not self._recording:
                return
                
            if frame is None:
                self.logger.warning("Received None frame")
                return
                
            timestamp = time.time()
            frame_num = frame_number if frame_number is not None else self.frame_count
            
            # Save frame as image
            frame_path = self.output_dir / f"frame_{frame_num:06d}.jpg"
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            # Save metadata
            metadata = {
                "frame_number": frame_num,
                "timestamp": timestamp,
                "source": source or "unknown",
                "shape": frame.shape
            }
            
            meta_path = self.output_dir / f"frame_{frame_num:06d}.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f)
                
            self.frame_count += 1
            
        except Exception as e:
            self.logger.error(f"Error logging frame: {e}")
            
    def reset(self):
        """Reset viewer state"""
        self.initialize(clear_existing=True)
        
    async def cleanup(self):
        """Clean up resources"""
        self._recording = False
        
    def register_connection(self):
        """Register new connection"""
        pass
        
    def unregister_connection(self):
        """Unregister connection"""
        pass
