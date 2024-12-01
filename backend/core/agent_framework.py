from typing import Dict, List, Optional, Any, TypedDict, Literal, Annotated, Union
from typing_extensions import TypeAlias
from pathlib import Path
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

# Define state types
class WorkflowState(TypedDict):
    query: str
    intent: Optional[Literal['info', 'run', 'analyze']]
    confirmed: bool
    response: str
    pending_action: str
    tool_input: dict

# Define state value types using Annotated
StateValue: TypeAlias = Union[str, bool, dict, None]
AgentState: TypeAlias = Dict[str, Annotated[StateValue, "state_value"]]

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
        def analyze_query(state: AgentState) -> AgentState:
            """Analyze user query to determine intent"""
            # Check for action keywords
            action_keywords = {
                'run': ['run', 'execute', 'perform', 'start', 'do'],
                'analyze': ['analyze', 'detect', 'find', 'identify'],
                'info': ['explain', 'tell', 'what is', 'how does']
            }
            
            intent = 'info'  # Default to information intent
            for action, keywords in action_keywords.items():
                if any(kw in state['query'].lower() for kw in keywords):
                    intent = action
                    break
                    
            return {**state, "intent": intent}
            
        def handle_info_request(state: AgentState) -> AgentState:
            """Handle information request"""
            # Create messages for the LLM with context
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=state['query'])
            ]
            
            # Add context about available analysis if any
            analysis_files = list(Path("tmp_content/analysis").glob("*.json"))
            if not analysis_files:
                context_msg = "\nNote: No previous analysis results available. Consider suggesting scene analysis."
                messages.append(SystemMessage(content=context_msg))
            
            # Use invoke instead of predict
            response = self.llm.invoke(messages).content
            return {**state, "response": response, "has_analysis": bool(analysis_files)}
            
        def confirm_action(state: AgentState) -> AgentState:
            """Get confirmation for action"""
            if state.get('confirmed'):
                return {**state, "response": 'Action confirmed'}
                
            tool_name = self._get_relevant_tool(state['query'])
            confirmation_msg = f"Would you like me to run {tool_name} analysis now?"
            return {
                **state,
                "response": confirmation_msg,
                "pending_action": tool_name
            }
            
        def execute_action(state: AgentState) -> AgentState:
            """Execute confirmed action"""
            if not state.get('confirmed'):
                return {**state, "response": 'Action not confirmed'}
                
            tool = self._get_tool(state['pending_action'])
            if not tool:
                return {
                    **state,
                    "response": f"Sorry, {state['pending_action']} is not available"
                }
                
            try:
                result = tool.run(state.get('tool_input', {}))
                return {**state, "response": f"Action completed: {result}"}
            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                return {**state, "response": f"Error executing action: {str(e)}"}
                
        # Create graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_query", analyze_query)
        workflow.add_node("handle_info_request", handle_info_request) 
        workflow.add_node("confirm_action", confirm_action)
        workflow.add_node("execute_action", execute_action)
        
        # Define conditional edges
        def route_query(state: AgentState) -> str:
            """Route to appropriate handler based on intent"""
            return "handle_info_request" if state.get('intent') == 'info' else "confirm_action"
            
        def route_action(state: AgentState) -> str:
            """Route action based on confirmation"""
            return "execute_action" if state.get('confirmed', False) else str(END)
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "analyze_query",
            lambda x: route_query(x)
        )
        workflow.add_conditional_edges(
            "confirm_action",
            lambda x: route_action(x)
        )
        
        # Add final edges
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
            # Initialize state with all required fields
            state: AgentState = {
                'query': query,
                'confirmed': kwargs.get('confirmed', False),
                'tool_input': kwargs,
                'intent': None,
                'response': '',
                'pending_action': '',
                'action_executed': False,
                'executed_action': None
            }
            
            # Create and compile workflow
            workflow = self._create_graph()
            app = workflow.compile()
            
            # Run workflow using invoke
            result = app.invoke(state)
            
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
