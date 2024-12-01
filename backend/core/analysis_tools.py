from langchain.tools import BaseTool
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SceneAnalysisTool(BaseTool):
    name = "scene_analysis"
    description = "Analyze the current video scene"
    
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

class ObjectDetectionTool(BaseTool):
    name = "object_detection"
    description = "Detect objects in the current video"
    
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
    name = "edge_detection"
    description = "Detect edges in the current video"
    
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
    name = "chat"
    description = "Chat about the video analysis results"
    
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
