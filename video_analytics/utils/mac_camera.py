import subprocess
import json
from typing import List, Dict
import cv2
import logging
import re

logger = logging.getLogger(__name__)

def get_mac_cameras() -> List[Dict]:
    """Get available cameras on macOS using multiple detection methods"""
    cameras = {}  # Use dict to avoid duplicates
    
    # Method 1: system_profiler
    try:
        result = subprocess.run(
            ['system_profiler', 'SPCameraDataType', '-json'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if 'SPCameraDataType' in data:
                for device in data['SPCameraDataType']:
                    if '_items' in device:
                        for camera in device['_items']:
                            unique_id = camera.get('unique_id', '')
                            cameras[unique_id] = {
                                'id': len(cameras),
                                'name': camera.get('_name', f'Camera {len(cameras)}'),
                                'model': camera.get('model_id', ''),
                                'unique_id': unique_id,
                                'system': 'darwin'
                            }
    except Exception as e:
        logger.warning(f"system_profiler detection failed: {e}")

    # Method 2: AVFoundation devices
    try:
        result = subprocess.run(
            ['system_profiler', 'SPUSBDataType', '-json'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if 'SPUSBDataType' in data:
                for controller in data['SPUSBDataType']:
                    _parse_usb_devices(controller, cameras)
    except Exception as e:
        logger.warning(f"USB detection failed: {e}")

    # Method 3: Direct device testing - only test indices 0 and 1 on macOS
    for i in range(2):  # Only test first 2 indices on macOS
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to read a frame to verify camera works
                ret, _ = cap.read()
                if ret:
                    # Get device info
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    # Generate a unique ID based on resolution and index
                    unique_id = f"camera_{i}_{width}x{height}"
                    
                    if unique_id not in cameras:
                        cameras[unique_id] = {
                            'id': i,
                            'name': f'Camera {i} ({width}x{height})',
                            'model': 'Unknown',
                            'unique_id': unique_id,
                            'system': 'darwin',
                            'resolution': f'{width}x{height}'
                        }
                        logger.info(f"Found working camera at index {i}: {width}x{height}")
            cap.release()
        except Exception as e:
            logger.warning(f"Error testing camera index {i}: {e}")

    return list(cameras.values())

def _parse_usb_devices(device: Dict, cameras: Dict):
    """Recursively parse USB devices to find cameras"""
    if '_items' in device:
        for item in device['_items']:
            if isinstance(item, dict):
                # Check if device is a camera
                if any(cam_hint in str(item).lower() for cam_hint in 
                      ['camera', 'webcam', 'facetime', 'isight']):
                    unique_id = item.get('serial_num', '')
                    if unique_id and unique_id not in cameras:
                        cameras[unique_id] = {
                            'id': len(cameras),
                            'name': item.get('_name', f'USB Camera {len(cameras)}'),
                            'model': item.get('manufacturer', 'Unknown'),
                            'unique_id': unique_id,
                            'system': 'darwin'
                        }
                # Recurse into sub-items
                _parse_usb_devices(item, cameras)
