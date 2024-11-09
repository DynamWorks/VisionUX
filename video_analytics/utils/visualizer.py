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
                data = json.load(f)
                # Handle API response format
                if isinstance(data, dict) and 'results' in data:
                    return data['results']
                return data
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
        
        # Handle both direct results and nested results format
        if isinstance(self.results, list):
            results_list = self.results
        else:
            results_list = [self.results] if isinstance(self.results, dict) else []
            
        for result in results_list:
            if isinstance(result, dict):
                timestamps.append(result.get('timestamp', 0))
                detections = result.get('detections', {})
                if isinstance(detections, dict):
                    # Sum all types of detections
                    total_detections = (
                        len(detections.get('segments', [])) +
                        len(detections.get('lanes', [])) +
                        len(detections.get('signs', [])) +
                        len(detections.get('text', []))
                    )
                    detection_counts.append(total_detections)
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
        detection_types = {
            'Objects': [],
            'Lanes': [],
            'Signs': [],
            'Text': []
        }
        
        results_list = self.results.get('results', [])
        if not results_list:
            if isinstance(self.results, dict):
                results_list = [self.results]
            else:
                results_list = self.results
            
        for result in results_list:
            if isinstance(result, dict):
                detections = result.get('detections', {})
                if isinstance(detections, dict):
                    # Collect all types of detections
                    for det in detections.get('segments', []):
                        if isinstance(det, dict) and 'class' in det:
                            detection_types['Objects'].append(det['class'])
                    
                    detection_types['Lanes'].extend(['Lane'] * len(detections.get('lanes', [])))
                    detection_types['Signs'].extend([det.get('class', 'Sign') for det in detections.get('signs', [])])
                    detection_types['Text'].extend([det.get('text', 'Text') for det in detections.get('text', [])])
        
        # Create subplot for each detection category
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Distribution of Detection Types')
        
        for (category, items), ax in zip(detection_types.items(), axes.flat):
            if items:
                counts = pd.Series(items).value_counts()
                if not counts.empty:
                    counts.plot(kind='bar', ax=ax)
                    ax.set_title(f'{category} Distribution')
                    ax.set_xlabel('Class')
                    ax.set_ylabel('Count')
                    ax.tick_params(axis='x', rotation=45)
                else:
                    ax.text(0.5, 0.5, f'No {category.lower()} detected',
                           horizontalalignment='center',
                           verticalalignment='center')
            else:
                ax.text(0.5, 0.5, f'No {category.lower()} detected',
                       horizontalalignment='center',
                       verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig('detection_types.png')
        
    def create_summary_csv(self):
        """Create summary CSV files for different aspects"""
        # Get results list
        if isinstance(self.results, dict) and 'results' in self.results:
            results_list = self.results['results']
        elif isinstance(self.results, list):
            results_list = self.results
        else:
            results_list = [self.results] if isinstance(self.results, dict) else []
            
        # Detections summary
        detections_data = []
        tracking_data = []
        
        for result in results_list:
            if isinstance(result, dict):
                # Handle nested detections structure
                detections = result.get('detections', {})
                if not detections and 'segments' in result:
                    # Handle flat structure
                    detections = result
                
                frame_data = {
                    'timestamp': float(result.get('timestamp', 0)),
                    'frame_number': int(result.get('frame_number', 0)),
                    'num_detections': len(detections.get('segments', [])),
                    'num_lanes': len(detections.get('lanes', [])),
                    'num_signs': len(detections.get('signs', [])),
                    'num_text': len(detections.get('text', []))
                }
                
                # Ensure we have at least one detection
                if any(v > 0 for k, v in frame_data.items() if k.startswith('num_')):
                    detections_data.append(frame_data)
                
                tracking = detections.get('tracking', {})
                if tracking:
                    tracking_data.append({
                        'timestamp': float(result.get('timestamp', 0)),
                        'frame_number': int(result.get('frame_number', 0)),
                        'current_objects': int(tracking.get('current', 0)),
                        'total_tracked': int(tracking.get('total', 0))
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
