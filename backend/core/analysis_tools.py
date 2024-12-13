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
        
    def _run(self, video_path: Path = None, **kwargs) -> Dict:
        """Run object detection on video file or frames"""
        try:
            if not video_path or not video_path.exists():
                return {'error': f"Video file not found: {video_path}"}

            # Create video capture
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return {'error': 'Failed to open video file'}

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Setup video writer
            vis_path = Path("tmp_content/visualizations")
            vis_path.mkdir(parents=True, exist_ok=True)
            output_video = vis_path / f"{video_path.stem}_objects.mp4"
            writer = cv2.VideoWriter(
                str(output_video),
                cv2.VideoWriter_fourcc(*'avc1'),
                fps,
                (width, height)
            )

            detections = []
            frame_count = 0

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Run detection on frame
                    result = self.detection_service.detect_objects(frame)
                    if 'error' in result:
                        continue

                    # Add frame number and timestamp
                    result['frame_number'] = frame_count
                    result['timestamp'] = frame_count / fps
                    detections.append(result)

                    # Draw detections on frame
                    for det in result.get('detections', []):
                        bbox = det['bbox']
                        cv2.rectangle(frame,
                            (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])),
                            (0, 255, 0), 2)
                        cv2.putText(frame,
                            f"{det['class']}: {det['confidence']:.2f}",
                            (int(bbox[0]), int(bbox[1])-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    writer.write(frame)
                    frame_count += 1

            finally:
                cap.release()
                writer.release()

            if not detections:
                return {'error': 'No objects detected'}

            # Save results
            analysis_id = f"object_detection_{int(time.time())}"
            results = {
                'video_file': video_path.name,
                'frame_count': frame_count,
                'detections': detections,
                'visualization': str(output_video),
                'timestamp': time.time()
            }

            # Save analysis results
            from backend.content_manager import ContentManager
            content_manager = ContentManager()
            saved_path = content_manager.save_analysis(results, analysis_id)

            return {
                'analysis_id': analysis_id,
                'detections': detections,
                'frame_count': frame_count,
                'storage_path': str(saved_path),
                'visualization': str(output_video)
            }

        except Exception as e:
            logger.error(f"Object detection error: {e}")
            return {'error': str(e)}

class EdgeDetectionTool(BaseTool):
    name: str = "edge_detection"
    description: str = "Detect edges in the current video"
    edge_service: Any = None
    
    def __init__(self, edge_service):
        super().__init__()
        self.edge_service = edge_service
        
    def _run(self, video_path: Path = None, save_analysis: bool = True, **kwargs) -> Dict:
        """Run edge detection on video file or frames"""
        try:
            if not video_path or not video_path.exists():
                return {'error': f"Video file not found: {video_path}"}

            # Create video capture
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return {'error': 'Failed to open video file'}

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Setup video writer
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
            frame_count = 0

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Run edge detection on frame
                    result = self.edge_service.detect_edges(frame)
                    if 'error' in result:
                        continue

                    # Add frame number and timestamp
                    result['frame_number'] = frame_count
                    result['timestamp'] = frame_count / fps
                    edge_results.append(result)

                    # Write processed frame
                    if 'frame' in result:
                        writer.write(result['frame'])
                    frame_count += 1

            finally:
                cap.release()
                writer.release()

            if not edge_results:
                return {'error': 'Edge detection failed'}

            response_data = {
                'frame_count': frame_count,
                'visualization': str(output_video)
            }

            if save_analysis:
                analysis_id = f"edge_detection_{int(time.time())}"
                
                # Convert edge_results to compressed format
                compressed_results = []
                for result in edge_results:
                    if 'edges' in result:
                        edges = np.array(result['edges'])
                        non_zero = np.nonzero(edges)
                        compressed_result = {
                            'frame_number': result['frame_number'],
                            'timestamp': result['timestamp'],
                            'shape': edges.shape,
                            'positions': list(zip(non_zero[0].tolist(), non_zero[1].tolist()))
                        }
                    else:
                        compressed_result = {
                            'frame_number': result['frame_number'],
                            'timestamp': result['timestamp']
                        }
                    compressed_results.append(compressed_result)

                # Prepare results
                results = {
                    'video_file': video_path.name,
                    'frame_count': frame_count,
                    'edge_results': compressed_results,
                    'visualization': str(output_video),
                    'timestamp': time.time(),
                    'format': 'sparse',
                    'tracked_objects': []
                }

                # Extract tracked objects data
                tracked_objects = {}
                for result in edge_results:
                    if 'tracked_objects' in result:
                        for obj in result['tracked_objects']:
                            obj_id = obj['id']
                            if obj_id not in tracked_objects:
                                tracked_objects[obj_id] = {
                                    'id': obj_id,
                                    'first_frame': result['frame_number'],
                                    'last_frame': result['frame_number'],
                                    'trajectory': [],
                                    'bbox_history': []
                                }
                            tracked_obj = tracked_objects[obj_id]
                            tracked_obj['last_frame'] = result['frame_number']
                            if 'center' in obj:
                                tracked_obj['trajectory'].append({
                                    'frame': result['frame_number'],
                                    'position': obj['center']
                                })
                            if 'bbox' in obj:
                                tracked_obj['bbox_history'].append({
                                    'frame': result['frame_number'],
                                    'bbox': obj['bbox']
                                })

                # Add tracked objects to results
                results['tracked_objects'] = list(tracked_objects.values())

                # Save analysis results
                from backend.content_manager import ContentManager
                content_manager = ContentManager()
                saved_path = content_manager.save_analysis(results, analysis_id)
                response_data.update({
                    'analysis_id': analysis_id,
                    'storage_path': str(saved_path)
                })
            else:
                # When save_analysis is disabled, only include tracked objects
                response_data['tracked_objects'] = results.get('tracked_objects', [])

            return response_data

        except Exception as e:
            logger.error(f"Edge detection error: {e}")
            return {'error': str(e)}

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
