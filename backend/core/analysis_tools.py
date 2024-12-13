from langchain.tools import BaseTool
from typing import Dict, Any, Optional, Union, Type
from pathlib import Path
import cv2
import time
import logging
import json
import numpy as np

logger = logging.getLogger(__name__)

class SceneAnalysisTool(BaseTool):
    name: str = "scene_analysis"
    description: str = "Analyze the current video scene"
    scene_service: Any = None
    
    def __init__(self, chat_service=None):
        super().__init__()
        # Import here to avoid circular imports
        from backend.services import SceneAnalysisService
        self.scene_service = SceneAnalysisService()
        
    def _run(self, video_path: Path = None, **kwargs) -> str:
        """Run scene analysis on video file or frames"""
        try:
            if video_path:
                if not self.scene_service:
                    from backend.services import SceneAnalysisService
                    self.scene_service = SceneAnalysisService()
                
                # Get video path
                video_file = Path("tmp_content/uploads") / video_path.name
                if not video_file.exists():
                    raise ValueError(f"Video file not found: {video_file}")
                    
                # Create video capture
                cap = cv2.VideoCapture(str(video_file))
                if not cap.isOpened():
                    raise ValueError("Failed to open video file")
                    
                try:
                    # Sample frames for analysis
                    frames = []
                    frame_numbers = []
                    timestamps = []
                    
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    interval = max(1, total_frames // 8)  # Sample 8 frames
                    
                    for i in range(8):
                        frame_pos = i * interval
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                        ret, frame = cap.read()
                        if ret:
                            frames.append(frame)
                            frame_numbers.append(frame_pos)
                            timestamps.append(frame_pos / fps if fps > 0 else 0)
                finally:
                    cap.release()
                    
                if not frames:
                    raise ValueError("Failed to capture frames for analysis")
                    
                # Perform scene analysis
                analysis = self.scene_service.analyze_scene(
                    frames,
                    context=str({
                        'video_file': video_path.name,
                        'source_type': 'video',
                        'timestamp': time.time()
                    }),
                    frame_numbers=frame_numbers,
                    timestamps=timestamps
                )
                
                if not analysis:
                    raise ValueError("Analysis failed")
                    
                return f"Scene analysis completed: {analysis['scene_analysis']['description']}"
                
            else:
                # Handle direct frame analysis
                result = self.scene_service.analyze_scene(**kwargs)
                return f"Scene analysis completed: {result['scene_analysis']['description']}"
                
        except Exception as e:
            logger.error(f"Scene analysis error: {e}")
            raise

class ObjectDetectionTool(BaseTool):
    name: str = "object_detection" 
    description: str = "Detect objects in the current video"
    detection_service: Any = None
    
    def __init__(self, detection_service):
        super().__init__()
        self.detection_service = detection_service
        
    def _run(self, video_path: Path = None, **kwargs) -> str:
        """Run object detection on video file or frames"""
        try:
            if video_path:
                if not video_path.exists():
                    raise ValueError(f"Video file not found: {video_path}")
                
                # Create video capture
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    raise ValueError("Failed to open video file")
                
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Setup video writer for visualization
                vis_path = Path("tmp_content/visualizations")
                vis_path.mkdir(parents=True, exist_ok=True)
                output_video = vis_path / f"{video_path.stem}_objects.mp4"
                writer = cv2.VideoWriter(
                    str(output_video),
                    cv2.VideoWriter_fourcc(*'avc1'),
                    fps,
                    (width, height)
                )
                
                detections_list = []
                frame_number = 0
                
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                            
                        # Run detection on frame
                        result = self.detection_service.detect_objects(frame)
                        if 'error' in result:
                            logger.warning(f"Frame {frame_number} detection error: {result['error']}")
                            continue
                            
                        # Add frame number to detections
                        result['frame_number'] = frame_number
                        detections_list.append(result)
                        
                        # Draw detections on frame
                        for det in result.get('detections', []):
                            bbox = det['bbox']
                            cv2.rectangle(frame, 
                                (int(bbox[0]), int(bbox[1])), 
                                (int(bbox[2]), int(bbox[3])), 
                                (255, 0, 255), 2)  # Bright magenta
                            cv2.putText(frame, 
                                f"{det['class']}: {det['confidence']:.2f}", 
                                (int(bbox[0]), int(bbox[1])-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        writer.write(frame)
                        frame_number += 1
                        
                finally:
                    cap.release()
                    writer.release()
                
                # Save detection results
                analysis_path = Path("tmp_content/analysis")
                analysis_path.mkdir(parents=True, exist_ok=True)
                results_file = analysis_path / f"{video_path.stem}_objects.json"
                
                analysis_results = {
                    'video_file': video_path.name,
                    'total_frames': total_frames,
                    'fps': fps,
                    'detections': detections_list,
                    'visualization': str(output_video),
                    'timestamp': time.time()
                }
                
                with open(results_file, 'w') as f:
                    json.dump(analysis_results, f, indent=2)
                
                return f"Object detection completed on video. Found objects across {frame_number} frames. Results saved to {results_file}"
                
            else:
                # Handle single frame detection
                result = self.detection_service.detect_objects(**kwargs)
                return f"Object detection completed: {result['detections']}"
                
        except Exception as e:
            logger.error(f"Object detection error: {e}")
            raise

class EdgeDetectionTool(BaseTool):
    name: str = "edge_detection"
    description: str = "Detect edges in the current video"
    edge_service: Any = None
    
    def __init__(self, edge_service):
        super().__init__()
        self.edge_service = edge_service
        
    def _run(self, video_path: Path = None, enabled: bool = True, **kwargs) -> str:
        """Run edge detection on video file or frames"""
        try:
            if video_path and enabled:
                if not video_path.exists():
                    raise ValueError(f"Video file not found: {video_path}")
                
                # Create video capture
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    raise ValueError("Failed to open video file")
                
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Setup video writer for visualization
                vis_path = Path("tmp_content/visualizations")
                vis_path.mkdir(parents=True, exist_ok=True)
                output_video = vis_path / f"{video_path.stem}_edges.mp4"
                writer = cv2.VideoWriter(
                    str(output_video),
                    cv2.VideoWriter_fourcc(*'avc1'),
                    fps,
                    (width, height)
                )
                
                edge_results = []
                frame_number = 0
                
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                            
                        # Run edge detection on frame
                        result = self.edge_service.detect_edges(frame)
                        if 'error' in result:
                            logger.warning(f"Frame {frame_number} edge detection error: {result['error']}")
                            continue
                            
                        # Add frame number to results
                        result['frame_number'] = frame_number
                        edge_results.append(result)
                        
                        # Write processed frame
                        writer.write(result['frame'])
                        frame_number += 1
                        
                finally:
                    cap.release()
                    writer.release()
                
                # Save edge detection results
                analysis_path = Path("tmp_content/analysis")
                analysis_path.mkdir(parents=True, exist_ok=True)
                results_file = analysis_path / f"{video_path.stem}_edges.json"
                
                analysis_results = {
                    'video_file': video_path.name,
                    'total_frames': total_frames,
                    'fps': fps,
                    'edge_detection_params': self.edge_service.edge_detection_params,
                    'frames_processed': frame_number,
                    'visualization': str(output_video),
                    'timestamp': time.time()
                }
                
                with open(results_file, 'w') as f:
                    json.dump(analysis_results, f, indent=2)
                
                return f"Edge detection completed on video. Processed {frame_number} frames. Results saved to {results_file}"
                
            else:
                # Handle single frame or enable/disable
                if enabled:
                    result = self.edge_service.detect_edges(**kwargs)
                    return f"Edge detection completed on frame"
                else:
                    self.edge_service.stop_edge_detection(**kwargs)
                    return "Edge detection stopped"
                    
        except Exception as e:
            logger.error(f"Edge detection error: {e}")
            raise

class ChatTool(BaseTool):
    name: str = "chat"
    description: str = "Chat about the video analysis results"
    chat_service: Any = None
    
    def __init__(self, chat_service):
        super().__init__()
        self.chat_service = chat_service
        
    def _run(self, query: str, **kwargs) -> str:
        try:
            result = self.chat_service.process_chat(query, **kwargs)
            return result.get('rag_response', {}).get('answer', 'No response available')
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise
