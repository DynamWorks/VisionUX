from pathlib import Path
import json
import time
from typing import Dict, List, Optional
import logging

class MemoryManager:
    """Manages frame memory and persistence"""
    
    def __init__(self, content_manager=None):
        self.frames = []
        self.content_manager = content_manager
        self.logger = logging.getLogger(__name__)
        
    def add_frame(self, frame_data: Dict) -> None:
        """Add frame data to memory"""
        self.frames.append(frame_data)
        
        # If we have a content manager, persist the frame
        if self.content_manager:
            try:
                self.content_manager.save_analysis(
                    frame_data,
                    f"frame_{frame_data.get('frame_number', len(self.frames))}"
                )
            except Exception as e:
                self.logger.warning(f"Failed to persist frame: {e}")
    
    def get_frames(self, start: Optional[int] = None, end: Optional[int] = None) -> List[Dict]:
        """Get frames within range"""
        if start is None:
            start = 0
        if end is None:
            end = len(self.frames)
        return self.frames[start:end]
    
    def clear(self) -> None:
        """Clear frame memory"""
        self.frames = []
        
    def save_state(self) -> None:
        """Save current memory state"""
        if self.content_manager:
            try:
                self.content_manager.save_analysis(
                    {"frames": self.frames},
                    f"memory_state_{int(time.time())}"
                )
            except Exception as e:
                self.logger.error(f"Failed to save memory state: {e}")
                
    def load_state(self, state_id: str) -> bool:
        """Load memory state from storage"""
        if not self.content_manager:
            return False
            
        try:
            state = self.content_manager.get_recent_analysis(state_id)
            if state and "frames" in state[0]:
                self.frames = state[0]["frames"]
                return True
        except Exception as e:
            self.logger.error(f"Failed to load memory state: {e}")
        return False
