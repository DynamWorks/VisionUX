import cv2
from typing import Optional, Dict, List
import platform
import logging

class CameraManager:
    """Manages camera device access across different platforms"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._system = platform.system().lower()
        
    def get_available_cameras(self) -> List[Dict]:
        """Get list of available camera devices"""
        cameras = []
        
        try:
            if self._system in ['darwin', 'linux']:
                # For macOS and Linux, try the first few device indices
                for i in range(10):
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        cameras.append({
                            'id': i,
                            'name': f'Camera {i}',
                            'system': self._system
                        })
                        cap.release()
                        
            elif self._system == 'windows':
                # For Windows, use DirectShow backend
                for i in range(10):
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                    if cap.isOpened():
                        cameras.append({
                            'id': i,
                            'name': f'Camera {i}',
                            'system': self._system
                        })
                        cap.release()
            
            # For mobile devices, the main camera is usually index 0
            if not cameras:
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    cameras.append({
                        'id': 0,
                        'name': 'Default Camera',
                        'system': self._system
                    })
                    cap.release()
                    
        except Exception as e:
            self.logger.error(f"Error detecting cameras: {str(e)}")
            
        return cameras
        
    def open_camera(self, camera_id: int) -> Optional[cv2.VideoCapture]:
        """Open specified camera with platform-specific settings"""
        try:
            if self._system == 'windows':
                cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(camera_id)
                
            if not cap.isOpened():
                self.logger.error(f"Failed to open camera {camera_id}")
                return None
                
            # Set common properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            return cap
            
        except Exception as e:
            self.logger.error(f"Error opening camera: {str(e)}")
            return None
