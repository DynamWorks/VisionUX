from langchain.tools import BaseTool
from typing import Dict, Any, Optional, Union, Type
from pathlib import Path
import cv2
import time
import logging

logger = logging.getLogger(__name__)

class SceneAnalysisTool(BaseTool):
    name: str = "scene_analysis"
    description: str = "Analyze the current video scene"
    scene_service: Any = None
    
    def __init__(self, scene_service):
        super().__init__()
        self.scene_service = scene_service
        
    def _run(self, **kwargs) -> str:
        try:
            result = self.scene_service.analyze_scene(**kwargs)
            return f"Scene analysis completed: {result['scene_analysis']['description']}"
        except Exception as e:
            logger.error(f"Scene analysis error: {e}")
            raise
            
    def analyze_video(self, video_path: Path) -> bool:
        """Analyze video file and generate scene analysis"""
        try:
            # Import here to avoid circular imports
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
            
            return True if analysis else False
            
        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            return False

class ObjectDetectionTool(BaseTool):
    name: str = "object_detection" 
    description: str = "Detect objects in the current video"
    detection_service: Any = None
    
    def __init__(self, detection_service):
        super().__init__()
        self.detection_service = detection_service
        
    def _run(self, **kwargs) -> str:
        try:
            result = self.detection_service.detect_objects(**kwargs)
            return f"Object detection completed: {result['objects']}"
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
        
    def _run(self, enabled: bool = True, **kwargs) -> str:
        try:
            if enabled:
                self.edge_service.start_edge_detection(**kwargs)
                return "Edge detection started"
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
