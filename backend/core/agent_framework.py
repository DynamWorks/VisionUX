from typing import Dict, List, Optional, Any
from langchain.agents import AgentExecutor
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools import BaseTool
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
import re
import logging

logger = logging.getLogger(__name__)

class VideoAnalysisAgent:
    """Agent for handling video analysis queries and actions"""
    
    def __init__(self, llm, tools: List[Tool]):
        self.llm = llm
        self.tools = tools
        self.conversation_history = []
        self.pending_action = None
        self.action_confirmed = False
        
        # Define system prompt
        self.system_prompt = """You are an AI assistant that helps users analyze videos.
You can have natural conversations and help execute various video analysis functions.

When handling user queries:
1. Carefully analyze if they want information or want to execute an action
2. For information requests, provide clear explanations
3. For action requests, ask for confirmation before executing
4. Suggest relevant analysis functions when appropriate
5. Be clear about what actions you're about to take
6. Express uncertainty when needed

Available tools:
{tools}

Example interactions:
User: "Tell me about object detection"
Assistant: I can explain how object detection works. Would you also like me to demonstrate by running object detection on the current video?

User: "Run object detection"
Assistant: I'll help you run object detection analysis. Would you like me to proceed with object detection on the current video?

User: "What objects are in the video?"
Assistant: To identify objects in the video, I'll need to run object detection analysis. Would you like me to do that now?
"""

    def _create_graph(self) -> StateGraph:
        """Create the agent workflow graph"""
        
        # Define graph nodes
        def analyze_query(state):
            """Analyze user query to determine intent"""
            query = state['query']
            
            # Check for action keywords
            action_keywords = {
                'run': ['run', 'execute', 'perform', 'start', 'do'],
                'analyze': ['analyze', 'detect', 'find', 'identify'],
                'info': ['explain', 'tell', 'what is', 'how does']
            }
            
            intent = 'info'  # Default to information intent
            for action, keywords in action_keywords.items():
                if any(kw in query.lower() for kw in keywords):
                    intent = action
                    break
                    
            return {'intent': intent, **state}
            
        def handle_info_request(state):
            """Handle information request"""
            response = self.llm.predict(
                self.system_prompt + f"\nUser: {state['query']}\nAssistant:"
            )
            return {'response': response, **state}
            
        def confirm_action(state):
            """Get confirmation for action"""
            if state.get('confirmed'):
                return {'response': 'Action confirmed', **state}
                
            tool_name = self._get_relevant_tool(state['query'])
            confirmation_msg = f"Would you like me to run {tool_name} analysis now?"
            return {'response': confirmation_msg, 'pending_action': tool_name, **state}
            
        def execute_action(state):
            """Execute confirmed action"""
            if not state.get('confirmed'):
                return {'response': 'Action not confirmed', **state}
                
            tool = self._get_tool(state['pending_action'])
            if not tool:
                return {
                    'response': f"Sorry, {state['pending_action']} is not available",
                    **state
                }
                
            try:
                result = tool.run(state.get('tool_input', {}))
                return {'response': f"Action completed: {result}", **state}
            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                return {'response': f"Error executing action: {str(e)}", **state}
                
        # Create graph
        workflow = StateGraph()
        
        # Add nodes
        workflow.add_node("analyze_query", analyze_query)
        workflow.add_node("handle_info_request", handle_info_request) 
        workflow.add_node("confirm_action", confirm_action)
        workflow.add_node("execute_action", execute_action)
        
        # Add edges
        workflow.add_edge("analyze_query", "handle_info_request", lambda x: x['intent'] == 'info')
        workflow.add_edge("analyze_query", "confirm_action", lambda x: x['intent'] in ['run', 'analyze'])
        workflow.add_edge("confirm_action", "execute_action", lambda x: x.get('confirmed', False))
        workflow.add_edge("confirm_action", END, lambda x: not x.get('confirmed', False))
        workflow.add_edge("handle_info_request", END)
        workflow.add_edge("execute_action", END)
        
        # Set entry point
        workflow.set_entry_point("analyze_query")
        
        return workflow

    def _get_relevant_tool(self, query: str) -> Optional[str]:
        """Get most relevant tool for query"""
        tool_keywords = {
            'object_detection': ['object', 'detect', 'find', 'identify'],
            'scene_analysis': ['scene', 'analyze', 'describe'],
            'edge_detection': ['edge', 'boundary', 'outline'],
            'chat': ['chat', 'talk', 'discuss']
        }
        
        query_lower = query.lower()
        for tool, keywords in tool_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return tool
        return None
        
    def _get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get tool by name"""
        return next((t for t in self.tools if t.name == tool_name), None)
        
    def process_query(self, query: str, **kwargs) -> Dict:
        """Process user query and return response"""
        try:
            # Initialize state
            state = {
                'query': query,
                'confirmed': kwargs.get('confirmed', False),
                'tool_input': kwargs
            }
            
            # Run workflow
            workflow = self._create_graph()
            result = workflow.run(state)
            
            # Update conversation history
            self.conversation_history.append({
                'role': 'user',
                'content': query
            })
            self.conversation_history.append({
                'role': 'assistant',
                'content': result['response']
            })
            
            return {
                'response': result['response'],
                'pending_action': result.get('pending_action'),
                'requires_confirmation': bool(result.get('pending_action')) and not result.get('confirmed'),
                'conversation_history': self.conversation_history[-5:]  # Last 5 messages
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'error': str(e),
                'response': "I encountered an error processing your request."
            }
