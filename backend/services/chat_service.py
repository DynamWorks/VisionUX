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
        self.state_dir = Path("tmp_content/agent_state")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize RAG service first to get retriever and LLM
        self.rag_service = RAGService(user_id=user_id, project_id=project_id)
        
        # Initialize tools after RAG service
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
        
        # Initialize agent with RAG components
        self.agent = VideoAnalysisAgent(
            llm=self.rag_service.llm,
            tools=self.tools,
            retriever=self.rag_service.retriever
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
        
    def _load_agent_state(self, state_path: Path) -> Dict:
        """Load or initialize agent state"""
        if state_path.exists():
            try:
                with open(state_path) as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load state, initializing new: {e}")

        return {
            'messages': [],
            'current_query': '',
            'retriever_result': None,
            'suggested_tool': None,
            'tool_input': None,
            'requires_confirmation': False,
            'confirmed': False,
            'final_response': None,
            'state_id': hashlib.md5(str(time.time()).encode()).hexdigest()[:8],
            'last_checkpoint': time.time(),
            'execution_history': []
        }

    def _save_agent_state(self, state_path: Path, state: Dict) -> None:
        """Save agent state to disk"""
        try:
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save agent state: {e}")

    def process_chat(self, query: str, video_path: str, confirmed: bool = False, tool_input: Dict = None) -> Dict:
        """Process chat query through agent workflow with state persistence"""
        try:
            # Load or initialize agent state
            state_path = Path("tmp_content/agent_state") / f"{video_path}_state.json"
            state = self._load_agent_state(state_path)
            
            # Update state with new query
            state.update({
                'current_query': query,
                'confirmed': confirmed,
                'tool_input': tool_input,
                'video_path': video_path
            })
            
            # Get chat history for context
            chat_history = self._get_chat_history(video_path)
            state['messages'] = chat_history
            
            # Initialize in-memory checkpointer for persistence
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
            
            # Run agent workflow with state persistence and thread ID
            config = {
                'configurable': {
                    'thread_id': f"chat_{video_path}_{int(time.time())}",
                    'checkpoint_ns': video_path
                }
            }
            
            result = self.agent.workflow.invoke(
                state,
                config=config,
                checkpointer=checkpointer
            )

            # Process workflow result
            response = {
                "answer": result.get('final_response', ''),
                "sources": result.get('retriever_result', []),
                "timestamp": time.time(),
                "chat_messages": result.get('messages', [])
            }

            # Add tool suggestion if present
            if result.get('suggested_tool'):
                response.update({
                    "suggested_tool": result['suggested_tool'],
                    "tool_description": result.get('tool_description', ''),
                    "requires_confirmation": True,
                    "answer": f"{response['answer']}\n\nWould you like me to {result.get('tool_description', '')}?"
                })
                if response['chat_messages']:
                    response['chat_messages'][-1]['content'] = response['answer']

            return response

        except Exception as e:
            self.logger.error(f"Chat processing error: {e}")
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
