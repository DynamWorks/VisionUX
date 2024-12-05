from pathlib import Path
import logging
import hashlib
from typing import Dict, List, Optional
from backend.content_manager import ContentManager
from .rag_service import RAGService
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableSequence
import json
import time
import numpy as np
import cv2

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
        """Process chat query using RAG and handle tool execution"""
        try:
            # Handle tool confirmation 
            if confirmed and hasattr(self, '_pending_tool'):
                return self._execute_confirmed_tool(self._pending_tool)

            # Create or update knowledge base
            if not self._current_chain:
                self._current_chain = self.rag_service.create_knowledge_base(Path(video_path))
                if not self._current_chain:
                    return {
                        "rag_response": {
                            "answer": "I need to analyze the video first. Would you like me to run a scene analysis?",
                            "sources": []
                        },
                        "requires_analysis": True
                    }

            # Get chat history
            chat_history = self._get_chat_history(video_path)

            # Query knowledge base
            try:
                # Ensure we're using the new LCEL chain correctly
                rag_response = self.rag_service.query_knowledge_base(
                    query=query,
                    chain=self._current_chain,
                    chat_history=chat_history[-5:] if chat_history else None,
                    tools=self.tools,
                    results_path=Path(video_path)
                )

                import pdb; pdb.set_trace()
                
                # Check for tool calls in response
                if 'tool_calls' in rag_response:
                    tool_name = rag_response['tool_calls'][0]['function']['name']
                    tool_args = rag_response['tool_calls'][0]['function']['arguments']
                    
                    # Store pending tool
                    self._pending_tool = {
                        'name': tool_name,
                        'args': tool_args
                    }
                    
                    # Update response to request confirmation
                    rag_response['answer'] += f"\n\nWould you like me to execute {tool_name}?"
                    rag_response['requires_confirmation'] = True

                # # Check for tool suggestions
                # tool_suggestions = self._analyze_for_tools(query, rag_response.get('answer', ''))
                # if tool_suggestions:
                #     rag_response['answer'] += f"\n\n{tool_suggestions}"
                #     rag_response.update({
                #         "requires_confirmation": True,
                #         "pending_action": tool_suggestions.split()[0]
                #     })

                # Save chat history
                self._save_chat_message(video_path, query, rag_response.get('answer', ''))
                self._save_chat_message(video_path, query, rag_response )
                return {
                    "query": query,
                    "timestamp": time.time(),
                    "rag_response": rag_response
                }

            except Exception as e:
                self.logger.error(f"RAG query error: {e}")
                return self._format_error_response(str(e), query)

        except Exception as e:
            self.logger.error(f"Chat processing failed: {e}")
            return self._format_error_response(str(e), query)

    def _execute_confirmed_tool(self, tool_info: Dict) -> Dict:
        """Execute a confirmed tool action"""
        tool_name = tool_info['name']
        tool_args = tool_info.get('args', {})
        
        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    result = tool.run(**tool_args)
                    # Clear pending tool after execution
                    self._pending_tool = None
                    return {
                        "rag_response": {
                            "answer": f"Tool execution complete: {result}",
                            "sources": []
                        },
                        "action_executed": {
                            "action": tool_name,
                            "args": tool_args,
                            "status": "completed",
                            "timestamp": time.time()
                        }
                    }
                except Exception as e:
                    self._pending_tool = None
                    return {"error": f"Tool execution failed: {str(e)}"}
        return {"error": f"Tool {tool_name} not found"}

    def _get_chat_history(self, video_path: str) -> List[Dict]:
        """Get chat history for video"""
        try:
            chat_file = Path('tmp_content/chat_history') / f"{video_path}_chat.json"
            if chat_file.exists():
                with open(chat_file) as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load chat history: {e}")
        return []

    def _save_chat_message(self, video_path: str, query: str, response: str) -> None:
        """Save chat message to history"""
        self.content_manager.save_chat_history([
            {'role': 'user', 'content': query},
            {'role': 'assistant', 'content': response}
        ], video_path)

    def _format_error_response(self, error: str, query: str) -> Dict:
        """Format error response"""
        error_msg = f"I encountered an error: {error}. Please try again or contact support if the issue persists."
        return {
            "error": error_msg,
            "query": query,
            "rag_response": {
                "answer": error_msg,
                "sources": []
            }
        }

    def _analyze_for_tools(self, query: str, current_answer: str) -> Optional[str]:
        """Analyze query and current answer to suggest relevant tools"""
        tool_patterns = {
            'object_detection': ['object', 'detect', 'find', 'identify', 'locate', 'spot'],
            'scene_analysis': ['describe', 'analyze', 'understand', 'explain', 'what is happening'],
            'edge_detection': ['edge', 'boundary', 'outline', 'shape', 'contour']
        }
        
        query_lower = query.lower()
        suggestions = []
        
        for tool, patterns in tool_patterns.items():
            if any(p in query_lower for p in patterns):
                if tool == 'object_detection':
                    suggestions.append("I can run object detection to identify and locate specific objects in the video.")
                elif tool == 'scene_analysis':
                    suggestions.append("I can perform a detailed scene analysis to better understand what's happening.")
                elif tool == 'edge_detection':
                    suggestions.append("I can enable edge detection to highlight object boundaries and shapes.")
                    
        if suggestions:
            return "\n\nWould you like me to " + " Or ".join(suggestions[:-1] + ["?" if len(suggestions) == 1 else " or " + suggestions[-1] + "?"])
        return None
