import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import logging
import numpy as np

class TrackingAnalysis:
    """Analyzes and stores object tracking data"""
    
    def __init__(self, base_path: str = "tmp_content/analysis"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def save_tracking_data(self, video_name: str, tracked_objects: List[Dict]) -> str:
        """Save tracking data for a video"""
        try:
            timestamp = int(time.time())
            filename = f"{video_name}_tracking_{timestamp}.json"
            output_path = self.base_path / filename
            
            # Process tracking data
            analysis = {
                'video_name': video_name,
                'timestamp': timestamp,
                'tracked_objects': tracked_objects,
                'statistics': self._calculate_statistics(tracked_objects),
                'metadata': {
                    'total_objects': len(tracked_objects),
                    'analysis_version': '1.0'
                }
            }
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(analysis, f, indent=2, cls=self.NumpyEncoder)
                
            self.logger.info(f"Saved tracking analysis to {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error saving tracking data: {e}")
            raise
            
    def _calculate_statistics(self, tracked_objects: List[Dict]) -> Dict:
        """Calculate statistics for tracked objects"""
        stats = {
            'object_counts': len(tracked_objects),
            'avg_track_length': 0,
            'movement_patterns': [],
            'interaction_zones': []
        }
        
        if not tracked_objects:
            return stats
            
        # Calculate average track length
        track_lengths = []
        for obj in tracked_objects:
            if 'first_frame' in obj and 'last_frame' in obj:
                track_lengths.append(obj['last_frame'] - obj['first_frame'])
        
        if track_lengths:
            stats['avg_track_length'] = sum(track_lengths) / len(track_lengths)
            
        # Analyze movement patterns
        for obj in tracked_objects:
            if 'trajectory' in obj and len(obj['trajectory']) > 1:
                pattern = self._analyze_movement_pattern(obj['trajectory'])
                stats['movement_patterns'].append({
                    'object_id': obj['id'],
                    'pattern': pattern
                })
                
        # Find interaction zones
        stats['interaction_zones'] = self._find_interaction_zones(tracked_objects)
        
        return stats
        
    def _analyze_movement_pattern(self, trajectory: List[Dict]) -> str:
        """Analyze movement pattern from trajectory"""
        if len(trajectory) < 2:
            return "stationary"
            
        # Calculate overall displacement
        start = trajectory[0]['position']
        end = trajectory[-1]['position']
        
        # Calculate total path length
        path_length = 0
        for i in range(len(trajectory) - 1):
            p1 = trajectory[i]['position']
            p2 = trajectory[i + 1]['position']
            path_length += np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
        # Compare path length to displacement
        displacement = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        
        if path_length < displacement * 1.2:
            return "linear"
        elif path_length > displacement * 2:
            return "circular"
        else:
            return "complex"
            
    def _find_interaction_zones(self, tracked_objects: List[Dict]) -> List[Dict]:
        """Find zones where multiple objects interact"""
        zones = []
        
        # Group trajectories by frame ranges
        frame_ranges = {}
        for obj in tracked_objects:
            if 'first_frame' in obj and 'last_frame' in obj:
                frame_range = range(obj['first_frame'], obj['last_frame'] + 1)
                frame_ranges[obj['id']] = frame_range
                
        # Find overlapping frame ranges
        for id1, range1 in frame_ranges.items():
            for id2, range2 in frame_ranges.items():
                if id1 >= id2:
                    continue
                    
                # Check for temporal overlap
                overlap = set(range1) & set(range2)
                if overlap:
                    # Check for spatial proximity in overlapping frames
                    zone = self._check_spatial_proximity(
                        tracked_objects, id1, id2, min(overlap), max(overlap)
                    )
                    if zone:
                        zones.append(zone)
                        
        return zones
        
    def _check_spatial_proximity(self, tracked_objects: List[Dict], 
                               id1: int, id2: int, start_frame: int, 
                               end_frame: int, threshold: float = 50.0) -> Optional[Dict]:
        """Check if two objects are spatially close in given frame range"""
        obj1 = next(obj for obj in tracked_objects if obj['id'] == id1)
        obj2 = next(obj for obj in tracked_objects if obj['id'] == id2)
        
        # Get positions in frame range
        positions1 = [t for t in obj1.get('trajectory', []) 
                     if start_frame <= t.get('frame', 0) <= end_frame]
        positions2 = [t for t in obj2.get('trajectory', []) 
                     if start_frame <= t.get('frame', 0) <= end_frame]
        
        if not positions1 or not positions2:
            return None
            
        # Check distances
        min_distance = float('inf')
        interaction_point = None
        
        for p1 in positions1:
            for p2 in positions2:
                if p1['frame'] == p2['frame']:
                    dist = np.sqrt(
                        (p1['position'][0] - p2['position'][0])**2 +
                        (p1['position'][1] - p2['position'][1])**2
                    )
                    if dist < min_distance:
                        min_distance = dist
                        interaction_point = {
                            'frame': p1['frame'],
                            'position': [
                                (p1['position'][0] + p2['position'][0]) / 2,
                                (p1['position'][1] + p2['position'][1]) / 2
                            ]
                        }
                        
        if min_distance <= threshold:
            return {
                'objects': [id1, id2],
                'start_frame': start_frame,
                'end_frame': end_frame,
                'min_distance': min_distance,
                'center': interaction_point
            }
            
        return None
        
    class NumpyEncoder(json.JSONEncoder):
        """Custom JSON encoder for numpy types"""
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import logging
import numpy as np

class TrackingAnalysis:
    """Analyzes and stores object tracking data"""
    
    def __init__(self, base_path: str = "tmp_content/analysis"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def save_tracking_data(self, video_name: str, tracked_objects: List[Dict]) -> str:
        """Save tracking data for a video"""
        try:
            timestamp = int(time.time())
            filename = f"{video_name}_tracking_{timestamp}.json"
            output_path = self.base_path / filename
            
            # Process tracking data
            analysis = {
                'video_name': video_name,
                'timestamp': timestamp,
                'tracked_objects': tracked_objects,
                'statistics': self._calculate_statistics(tracked_objects),
                'metadata': {
                    'total_objects': len(tracked_objects),
                    'analysis_version': '1.0'
                }
            }
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(analysis, f, indent=2, cls=self.NumpyEncoder)
                
            self.logger.info(f"Saved tracking analysis to {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error saving tracking data: {e}")
            raise
            
    def _calculate_statistics(self, tracked_objects: List[Dict]) -> Dict:
        """Calculate statistics for tracked objects"""
        stats = {
            'object_counts': len(tracked_objects),
            'avg_track_length': 0,
            'movement_patterns': [],
            'interaction_zones': []
        }
        
        if not tracked_objects:
            return stats
            
        # Calculate average track length
        track_lengths = []
        for obj in tracked_objects:
            if 'first_frame' in obj and 'last_frame' in obj:
                track_lengths.append(obj['last_frame'] - obj['first_frame'])
        
        if track_lengths:
            stats['avg_track_length'] = sum(track_lengths) / len(track_lengths)
            
        # Analyze movement patterns
        for obj in tracked_objects:
            if 'trajectory' in obj and len(obj['trajectory']) > 1:
                pattern = self._analyze_movement_pattern(obj['trajectory'])
                stats['movement_patterns'].append({
                    'object_id': obj['id'],
                    'pattern': pattern
                })
                
        # Find interaction zones
        stats['interaction_zones'] = self._find_interaction_zones(tracked_objects)
        
        return stats
        
    def _analyze_movement_pattern(self, trajectory: List[Dict]) -> str:
        """Analyze movement pattern from trajectory"""
        if len(trajectory) < 2:
            return "stationary"
            
        # Calculate overall displacement
        start = trajectory[0]['position']
        end = trajectory[-1]['position']
        
        # Calculate total path length
        path_length = 0
        for i in range(len(trajectory) - 1):
            p1 = trajectory[i]['position']
            p2 = trajectory[i + 1]['position']
            path_length += np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
        # Compare path length to displacement
        displacement = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        
        if path_length < displacement * 1.2:
            return "linear"
        elif path_length > displacement * 2:
            return "circular"
        else:
            return "complex"
            
    def _find_interaction_zones(self, tracked_objects: List[Dict]) -> List[Dict]:
        """Find zones where multiple objects interact"""
        zones = []
        
        # Group trajectories by frame ranges
        frame_ranges = {}
        for obj in tracked_objects:
            if 'first_frame' in obj and 'last_frame' in obj:
                frame_range = range(obj['first_frame'], obj['last_frame'] + 1)
                frame_ranges[obj['id']] = frame_range
                
        # Find overlapping frame ranges
        for id1, range1 in frame_ranges.items():
            for id2, range2 in frame_ranges.items():
                if id1 >= id2:
                    continue
                    
                # Check for temporal overlap
                overlap = set(range1) & set(range2)
                if overlap:
                    # Check for spatial proximity in overlapping frames
                    zone = self._check_spatial_proximity(
                        tracked_objects, id1, id2, min(overlap), max(overlap)
                    )
                    if zone:
                        zones.append(zone)
                        
        return zones
        
    def _check_spatial_proximity(self, tracked_objects: List[Dict], 
                               id1: int, id2: int, start_frame: int, 
                               end_frame: int, threshold: float = 50.0) -> Optional[Dict]:
        """Check if two objects are spatially close in given frame range"""
        obj1 = next(obj for obj in tracked_objects if obj['id'] == id1)
        obj2 = next(obj for obj in tracked_objects if obj['id'] == id2)
        
        # Get positions in frame range
        positions1 = [t for t in obj1.get('trajectory', []) 
                     if start_frame <= t.get('frame', 0) <= end_frame]
        positions2 = [t for t in obj2.get('trajectory', []) 
                     if start_frame <= t.get('frame', 0) <= end_frame]
        
        if not positions1 or not positions2:
            return None
            
        # Check distances
        min_distance = float('inf')
        interaction_point = None
        
        for p1 in positions1:
            for p2 in positions2:
                if p1['frame'] == p2['frame']:
                    dist = np.sqrt(
                        (p1['position'][0] - p2['position'][0])**2 +
                        (p1['position'][1] - p2['position'][1])**2
                    )
                    if dist < min_distance:
                        min_distance = dist
                        interaction_point = {
                            'frame': p1['frame'],
                            'position': [
                                (p1['position'][0] + p2['position'][0]) / 2,
                                (p1['position'][1] + p2['position'][1]) / 2
                            ]
                        }
                        
        if min_distance <= threshold:
            return {
                'objects': [id1, id2],
                'start_frame': start_frame,
                'end_frame': end_frame,
                'min_distance': min_distance,
                'center': interaction_point
            }
            
        return None
        
    class NumpyEncoder(json.JSONEncoder):
        """Custom JSON encoder for numpy types"""
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
