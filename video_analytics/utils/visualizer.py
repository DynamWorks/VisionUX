import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class ResultVisualizer:
    def __init__(self, results_path: str):
        """Initialize visualizer with results path"""
        self.results_path = Path(results_path)
        self.results = self._load_results()
        
    def _load_results(self):
        """Load and parse results file"""
        import json
        if self.results_path.suffix == '.json':
            with open(self.results_path) as f:
                return json.load(f)
        elif self.results_path.suffix == '.csv':
            return pd.read_csv(self.results_path)
        else:
            raise ValueError(f"Unsupported file format: {self.results_path.suffix}")
            
    def plot_detections_over_time(self):
        """Plot object detections over time"""
        plt.figure(figsize=(12, 6))
        
        # Convert results to proper format for plotting
        timestamps = []
        detection_counts = []
        
        results_list = self.results.get('results', [])
        if not results_list:
            results_list = [self.results] if isinstance(self.results, dict) else self.results
            
        for result in results_list:
            if isinstance(result, dict):
                timestamps.append(result.get('timestamp', 0))
                detections = result.get('detections', {})
                if isinstance(detections, dict):
                    detection_counts.append(len(detections.get('segments', [])))
                else:
                    detection_counts.append(0)
        
        plt.plot(timestamps, detection_counts)
        plt.title('Object Detections Over Time')
        plt.xlabel('Timestamp (s)')
        plt.ylabel('Number of Detections')
        plt.grid(True)
        plt.savefig('detections_timeline.png')
        
    def plot_detection_types(self):
        """Plot distribution of detection types"""
        detection_types = []
        
        results_list = self.results.get('results', [])
        if not results_list:
            results_list = [self.results] if isinstance(self.results, dict) else self.results
            
        for result in results_list:
            if isinstance(result, dict):
                detections = result.get('detections', {})
                if isinstance(detections, dict):
                    segments = detections.get('segments', [])
                    for det in segments:
                        if isinstance(det, dict) and 'class' in det:
                            detection_types.append(det['class'])
        
        if detection_types:
            counts = pd.Series(detection_types).value_counts()
            plt.figure(figsize=(10, 6))
            counts.plot(kind='bar')
            plt.title('Distribution of Detection Types')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('detection_types.png')
        else:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No detection data available', 
                    horizontalalignment='center', verticalalignment='center')
            plt.savefig('detection_types.png')
        
    def create_summary_csv(self):
        """Create summary CSV files for different aspects"""
        # Get results list
        results_list = self.results.get('results', [])
        if not results_list:
            results_list = [self.results] if isinstance(self.results, dict) else self.results
            
        # Detections summary
        detections_data = []
        tracking_data = []
        
        for result in results_list:
            if isinstance(result, dict):
                detections = result.get('detections', {})
                detections_data.append({
                    'timestamp': result.get('timestamp', 0),
                    'frame_number': result.get('frame_number', 0),
                    'num_detections': len(detections.get('segments', [])),
                    'num_lanes': len(detections.get('lanes', [])),
                    'num_signs': len(detections.get('signs', [])),
                    'num_text': len(detections.get('text', []))
                })
                
                tracking = detections.get('tracking', {})
                tracking_data.append({
                    'timestamp': result.get('timestamp', 0),
                    'frame_number': result.get('frame_number', 0),
                    'current_objects': tracking.get('current', 0),
                    'total_tracked': tracking.get('total', 0)
                })
        
        # Create and save DataFrames
        if detections_data:
            detections_df = pd.DataFrame(detections_data)
            detections_df.to_csv('detection_summary.csv', index=False)
            
        if tracking_data:
            tracking_df = pd.DataFrame(tracking_data)
            tracking_df.to_csv('tracking_summary.csv', index=False)
        
    def visualize_frame(self, frame_number: int, video_path: str):
        """Visualize detections on a specific frame"""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not read frame {frame_number}")
            
        frame_result = self.results[self.results['frame_number'] == frame_number].iloc[0]
        
        # Draw detections
        for det in frame_result['detections']['segments']:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{det['class']}: {det['confidence']:.2f}",
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Draw lanes
        for lane in frame_result['detections']['lanes']:
            pts = np.array(lane['coordinates'], np.int32)
            cv2.polylines(frame, [pts], False, (255, 0, 0), 2)
            
        # Save visualized frame
        cv2.imwrite(f'frame_{frame_number}_visualized.jpg', frame)
