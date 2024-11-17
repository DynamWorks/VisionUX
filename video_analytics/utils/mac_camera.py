import subprocess
from typing import List, Dict

def get_mac_cameras() -> List[Dict]:
    """Get available cameras on macOS using system_profiler"""
    cameras = []
    try:
        # Run system_profiler to get camera info
        result = subprocess.run(
            ['system_profiler', 'SPCameraDataType', '-json'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            
            # Parse camera data
            if 'SPCameraDataType' in data:
                for device in data['SPCameraDataType']:
                    if '_items' in device:
                        for camera in device['_items']:
                            cameras.append({
                                'id': len(cameras),  # Use index as ID
                                'name': camera.get('_name', f'Camera {len(cameras)}'),
                                'model': camera.get('model_id', ''),
                                'unique_id': camera.get('unique_id', ''),
                                'system': 'darwin'
                            })
    except Exception as e:
        print(f"Error detecting macOS cameras: {e}")
        
    return cameras
