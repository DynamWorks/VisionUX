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
        if self.results_path.suffix == '.json':
            return pd.read_json(self.results_path)
        elif self.results_path.suffix == '.csv':
            return pd.read_csv(self.results_path)
        else:
            raise ValueError(f"Unsupported file format: {self.results_path.suffix}")
            
    def plot_detections_over_time(self):
        """Plot object detections over time"""
        plt.figure(figsize=(12, 6))
        self.results.groupby('timestamp')['detections.segments'].apply(len).plot()
        plt.title('Object Detections Over Time')
        plt.xlabel('Timestamp (s)')
        plt.ylabel('Number of Detections')
        plt.grid(True)
        plt.savefig('detections_timeline.png')
        
    def plot_detection_types(self):
        """Plot distribution of detection types"""
        detection_types = []
        for segments in self.results['detections.segments']:
            for det in segments:
                detection_types.append(det['class'])
                
        counts = pd.Series(detection_types).value_counts()
        plt.figure(figsize=(10, 6))
        counts.plot(kind='bar')
        plt.title('Distribution of Detection Types')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('detection_types.png')
        
    def create_summary_csv(self):
        """Create summary CSV files for different aspects"""
        # Detections summary
        detections_df = pd.DataFrame([
            {
                'timestamp': row['timestamp'],
                'frame_number': row['frame_number'],
                'num_detections': len(row['detections']['segments']),
                'num_lanes': len(row['detections']['lanes']),
                'num_signs': len(row['detections']['signs']),
                'num_text': len(row['detections']['text'])
            }
            for _, row in self.results.iterrows()
        ])
        detections_df.to_csv('detection_summary.csv', index=False)
        
        # Object tracking summary
        tracking_df = pd.DataFrame([
            {
                'timestamp': row['timestamp'],
                'frame_number': row['frame_number'],
                'current_objects': row['detections']['tracking']['current'],
                'total_tracked': row['detections']['tracking']['total']
            }
            for _, row in self.results.iterrows()
        ])
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
