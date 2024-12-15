import sqlite3
import hashlib
from pathlib import Path
import logging
from typing import Dict, List, Optional
from backend.content_manager import ContentManager
from .rag_service import RAGService
from langgraph.checkpoint.sqlite import SqliteSaver
import json
import time

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
            EdgeDetectionTool(self)#,
            #ChatTool(self)
        ]
        
        # Initialize agent with RAG components
        self.agent = VideoAnalysisAgent(
            llm=self.rag_service.llm,
            tools=self.tools,
            retriever=self.rag_service.retriever
        )
        
        

    def process_chat(self, query: str, video_path: str, confirmed: bool = False, tool_input: Dict = None) -> Dict:
        """Process chat query through agent workflow with state persistence"""
        try:
            # Initialize state
            state = {
                'messages': [],
                'current_query': query,
                'retriever_result': None,
                'suggested_tool': None,
                'tool_input': tool_input,
                'requires_confirmation': True,
                'confirmed': confirmed,
                'final_response': None,
                'video_path': video_path,
                'state_id': hashlib.md5(str(time.time()).encode()).hexdigest()[:8],
                'last_checkpoint': time.time(),
                'execution_history': [],
                'retrieve_info': True
            }
            
            # Get chat history for context
            chat_history = self._get_chat_history(video_path)
            state['messages'] = chat_history
            
            # # Initialize SQLite checkpointer
            # db_path = Path("tmp_content/agent_state/checkpoints.sqlite")
            # db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # # Create SqliteSaver instance
            # from langgraph.checkpoint.sqlite import SqliteSaver
            # checkpointer = SqliteSaver.from_conn_string(str(db_path))
            
            # Generate unique thread ID and namespace
            # thread_id = f"chat_{video_path}_{int(time.time())}"
            thread_id = f"video_{hashlib.md5(video_path.encode()).hexdigest()[:8]}"
            
            # Configure checkpointing
            config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

            
            # Compile and run workflow with checkpointing
            app = self.agent.workflow.compile()
            result = app.invoke(state, config=config)
            
            # # Save final state checkpoint
            # checkpoint_data = {
            #     "thread_id": thread_id,
            #     "thread_ts": time.strftime('%Y-%m-%dT%H:%M:%S'),
            #     "checkpoint": {
            #         "state": result
            #     },
            #     "metadata": {
            #         "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
            #     }
            # }
            
            # # Save using SqliteSaver
            # checkpointer.put(
            #     config={"configurable": {"thread_id": thread_id, , "checkpoint_ns": ""}},
            #     checkpoint=checkpoint_data["checkpoint"],
            #     metadata=checkpoint_data["metadata"],
            #     new_versions={}
            # )
            
            # Process workflow result
            vis_path = Path("tmp_content/visualizations")
            video_path = Path(result.get('video_path', ''))
            video_stem = video_path.stem if isinstance(video_path, Path) else Path(str(video_path)).stem
            
            visualization = None
            if result.get('suggested_tool') == 'edge_detection':
                visualization = str(vis_path / f"{video_stem}_edges.mp4")
            elif result.get('suggested_tool') == 'object_detection':
                visualization = str(vis_path / f"{video_stem}_objects.mp4")
                
            response = {
                "answer": result.get('final_response', ''),
                "sources": result.get('retriever_result', []),
                "timestamp": time.time(),
                "chat_messages": result.get('messages', []),
                "suggested_tool": result.get('suggested_tool', ''),
                "confirmed": result.get('confirmed', False),
                "visualization": visualization
            }

            return response

        except Exception as e:
            self.logger.error(f"Chat processing error: {e}")
            return self._format_error_response(str(e), query)

        except Exception as e:
            self.logger.error(f"Chat processing failed: {e}")
            return self._format_error_response(str(e), query)


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

