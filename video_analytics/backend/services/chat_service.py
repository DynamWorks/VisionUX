import json
import time
from pathlib import Path
import logging
from typing import Dict, List, Optional
from ..content_manager import ContentManager
from ...core.swarm_agents import SwarmCoordinator
from .rag_service import RAGService
from ...utils.memory_manager import MemoryManager
import numpy as np

class ChatService:
    """Service for handling chat interactions with video analysis context"""
    
    def __init__(self):
        self.content_manager = ContentManager()
        self.swarm_coordinator = SwarmCoordinator()
        self.logger = logging.getLogger(__name__)
        
    def _get_context_from_tmp(self) -> List[Dict]:
        """Gather context from tmp_content directory"""
        context = []
        
        # Get recent analysis results
        analysis_files = list(Path('tmp_content/analysis').glob('*.json'))
        for file in sorted(analysis_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
            try:
                with open(file) as f:
                    context.append(json.load(f))
            except Exception as e:
                self.logger.warning(f"Failed to load context from {file}: {e}")
                
        return context
        
    def _extract_function_calls(self, query: str) -> List[str]:
        """Extract required function calls from query"""
        function_keywords = {
            'object_detection': ['detect', 'find', 'locate', 'show', 'identify'],
            'face_analysis': ['face', 'person', 'people', 'identity'],
            'traffic_analysis': ['traffic', 'vehicle', 'car', 'speed', 'flow']
        }
        
        required_functions = []
        query_lower = query.lower()
        
        for func, keywords in function_keywords.items():
            if any(kw in query_lower for kw in keywords):
                required_functions.append(func)
                
        return required_functions
        
    def __init__(self):
        self.content_manager = ContentManager()
        self.swarm_coordinator = SwarmCoordinator()
        self.rag_service = RAGService()
        self.logger = logging.getLogger(__name__)
        self._current_chain = None
        
    def process_chat(self, query: str, video_path: str) -> Dict:
        """Process chat query with RAG and execute required functions"""
        try:
            # Get latest analysis results
            analysis_files = list(Path('tmp_content/analysis').glob('*.json'))
            if not analysis_files:
                return {"error": "No analysis results found"}
                
            latest_results = max(analysis_files, key=lambda p: p.stat().st_mtime)
            
            # Create or get knowledge base
            if not self._current_chain:
                vectordb = self.rag_service.create_knowledge_base(latest_results)
                if not vectordb:
                    return {"error": "Failed to create knowledge base"}
                self._current_chain = self.rag_service.get_retrieval_chain(vectordb)
            
            # Query knowledge base
            rag_response = self.rag_service.query_knowledge_base(query, self._current_chain)
            
            # Initialize response data
            response_data = {
                "query": query,
                "rag_response": rag_response,
                "results": {}
            }
            
            # Extract required functions for additional processing
            required_functions = self._extract_function_calls(query)
            
            # Execute required functions using swarm
            if required_functions:
                import cv2
                cap = cv2.VideoCapture(video_path)
                frames = []
                frame_numbers = []
                timestamps = []
                
                while len(frames) < 8:  # Sample up to 8 frames
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                    frame_numbers.append(len(frames))
                    timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
                
                cap.release()
                
                if frames:
                    swarm_results = self.swarm_coordinator.analyze_frame_batch(
                        frames=frames,
                        frame_numbers=frame_numbers,
                        timestamps=timestamps
                    )
                    response_data["results"] = swarm_results
                    
            # Save chat results
            results_path = self.content_manager.save_analysis(
                response_data,
                f"chat_response_{int(time.time())}"
            )
            
            return response_data
            
        except Exception as e:
            self.logger.error(f"Chat processing failed: {e}")
            return {
                "error": str(e),
                "query": query
            }
