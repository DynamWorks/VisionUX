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
        
    def process_chat(self, query: str, video_path: str, confirmed: bool = False, tool_input: Dict = None) -> Dict:
        """Process chat query using RAG and handle tool execution"""
        try:
            # Handle tool confirmation and execution
            if confirmed and hasattr(self, '_pending_tool'):
                result = self._execute_confirmed_tool(self._pending_tool)
                self._pending_tool = None  # Clear pending tool after execution
                return result

            # Create or update knowledge base
            if not self._current_chain:
                self._current_chain = self.rag_service.create_knowledge_base(Path(video_path))
                if not self._current_chain:
                    return {
                        "answer": "I need to analyze the video first. Would you like me to run a scene analysis?",
                        "requires_analysis": True,
                        "suggested_tool": "scene_analysis",
                        "requires_confirmation": True,
                        "chat_messages": [
                            {"role": "assistant", "content": "I need to analyze the video first. Would you like me to run a scene analysis?"}
                        ]
                    }

            # Get chat history
            chat_history = self._get_chat_history(video_path)

            # Query knowledge base with tool awareness
            try:
                rag_response = self.rag_service.query_knowledge_base(
                    query=query,
                    chain=self._current_chain,
                    chat_history=chat_history[-5:] if chat_history else None,
                    tools=self.tools,
                    results_path=Path(video_path)
                )

                # Check for tool suggestions
                tool_suggestion = self._analyze_for_tools(query, rag_response.get('answer', ''))
                
                response = {
                    "answer": rag_response.get('answer', ''),
                    "sources": rag_response.get('sources', []),
                    "timestamp": time.time(),
                    "chat_messages": [
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": rag_response.get('answer', '')}
                    ]
                }

                if tool_suggestion:
                    self._pending_tool = tool_suggestion
                    response.update({
                        "suggested_tool": tool_suggestion['name'],
                        "tool_description": tool_suggestion['description'],
                        "requires_confirmation": True,
                        "answer": f"{response['answer']}\n\nWould you like me to {tool_suggestion['description']}?"
                    })
                    response['chat_messages'][-1]['content'] = response['answer']

                return response

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
                    response = {
                        "answer": f"I've completed the {tool_name}: {result}",
                        "tool_executed": {
                            "name": tool_name,
                            "description": tool_info['description'],
                            "result": result
                        },
                        "timestamp": time.time(),
                        "chat_messages": [
                            {"role": "system", "content": f"Executing {tool_name}..."},
                            {"role": "assistant", "content": f"I've completed the {tool_name}: {result}"}
                        ]
                    }
                    
                    # Check if we need another tool
                    next_tool = self._analyze_for_tools(result, "")
                    if next_tool:
                        self._pending_tool = next_tool
                        response.update({
                            "suggested_tool": next_tool['name'],
                            "tool_description": next_tool['description'],
                            "requires_confirmation": True,
                            "answer": f"{response['answer']}\n\nWould you like me to {next_tool['description']}?"
                        })
                        
                    return response
                    
                except Exception as e:
                    error_msg = f"Tool execution failed: {str(e)}"
                    return {
                        "error": error_msg,
                        "chat_messages": [
                            {"role": "system", "content": f"Error executing {tool_name}"},
                            {"role": "assistant", "content": error_msg}
                        ]
                    }
                    
        return {
            "error": f"Tool {tool_name} not found",
            "chat_messages": [
                {"role": "system", "content": f"Tool {tool_name} not found"},
                {"role": "assistant", "content": "I apologize, but I couldn't find the requested tool. Please try a different action."}
            ]
        }

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
