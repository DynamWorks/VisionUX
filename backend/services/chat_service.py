from pathlib import Path
import logging
from typing import Dict, List, Optional
from backend.content_manager import ContentManager
from .rag_service import RAGService
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import json
import time
import numpy as np

class ChatService:
    """Service for handling chat interactions with video analysis context"""
    
    def __init__(self, user_id: str = None, project_id: str = None):
        self.content_manager = ContentManager()
        self.rag_service = RAGService(user_id=user_id, project_id=project_id)
        self.logger = logging.getLogger(__name__)
        self._current_chain = None
        
        # Initialize agent with tools
        from backend.core.analysis_tools import (
            SceneAnalysisTool, ObjectDetectionTool, 
            EdgeDetectionTool, ChatTool
        )
        from backend.core.agent_framework import VideoAnalysisAgent
        
        self.tools = [
            SceneAnalysisTool(self),
            ObjectDetectionTool(self),
            EdgeDetectionTool(self),
            ChatTool(self)
        ]
        
        self.agent = VideoAnalysisAgent(
            llm=self.rag_service.llm,
            tools=self.tools
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
        
    def process_chat(self, query: str, video_path: str, use_swarm: bool = False, confirmed: bool = False) -> Dict:
        """Process chat query using agent framework"""
        try:
            # Process query through agent
            result = self.agent.process_query(
                query,
                confirmed=confirmed,
                video_path=video_path,
                use_swarm=use_swarm
            )
            
            if result.get('error'):
                return {"error": result['error']}
                
            # If action requires confirmation and not confirmed, return confirmation request
            if result.get('requires_confirmation'):
                return {
                    "requires_confirmation": True,
                    "pending_action": result['pending_action'],
                    "response": result['response']
                }
            
            # Create or get knowledge base
            if not self._current_chain:
                # Get latest analysis results
                analysis_files = list(Path('tmp_content/analysis').glob('*.json'))
                if not analysis_files:
                    return {"error": "No analysis results found"}
                    
                latest_results = max(analysis_files, key=lambda p: p.stat().st_mtime)
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
            
            # Execute required functions if needed
            if required_functions:
                response_data["results"] = {}
                    
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
