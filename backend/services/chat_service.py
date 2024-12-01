from pathlib import Path
import logging
from typing import Dict, List, Optional
from backend.content_manager import ContentManager
from backend.core.swarm_agents import SwarmCoordinator
from .rag_service import RAGService
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import json
import time
import numpy as np

class ChatService:
    """Service for handling chat interactions with video analysis context"""
    
    def __init__(self, user_id: str = None, project_id: str = None):
        self.content_manager = ContentManager()
        self.swarm_coordinator = SwarmCoordinator()
        self.rag_service = RAGService(user_id=user_id, project_id=project_id)
        self.logger = logging.getLogger(__name__)
        self._current_chain = None
        self.system_message = SystemMessage(
            content="""You are an AI assistant powered by a RAG system.
            When answering questions about video content:
            1. Only use information from the provided context
            2. If the context doesn't contain enough information, clearly state that
            3. Cite specific frames and timestamps when possible
            4. Keep responses clear and concise
            5. If you're unsure about something, express that uncertainty
            6. Never make up information that isn't in the context"""
        )
        
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
        
    def process_chat(self, query: str, video_path: str, use_swarm: bool = False) -> Dict:
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
            
            # Get recent chat history
            chat_history = []
            chat_file = Path('tmp_content/chat_history') / f"{video_path}_chat.json"
            if chat_file.exists():
                try:
                    with open(chat_file) as f:
                        chat_history = json.load(f)
                except Exception as e:
                    self.logger.warning(f"Failed to load chat history: {e}")

            # Query knowledge base with chat context
            rag_response = self.rag_service.query_knowledge_base(
                query, 
                self._current_chain,
                chat_history=chat_history[-5:] if chat_history else None  # Pass last 5 messages
            )
            
            # Initialize response data
            response_data = {
                "query": query,
                "rag_response": rag_response,
                "results": {}
            }
            
            # Extract required functions for additional processing
            required_functions = self._extract_function_calls(query)
            
            # Execute required functions using swarm if enabled
            if required_functions and use_swarm:
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
