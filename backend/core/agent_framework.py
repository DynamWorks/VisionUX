from typing import Dict, List, Optional, Any, TypedDict, Literal, Annotated, Union
from typing_extensions import TypeAlias
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
                    
            return {"intent": intent}
            
        def handle_info_request(state: AgentState) -> AgentState:
            """Handle information request"""
            response = self.llm.predict(
                self.system_prompt + f"\nUser: {state['query']}\nAssistant:"
            )
            return {"response": response}
            
        def confirm_action(state: AgentState) -> AgentState:
            """Get confirmation for action"""
            if state.get('confirmed'):
                return {"response": 'Action confirmed'}
                
            tool_name = self._get_relevant_tool(state['query'])
            confirmation_msg = f"Would you like me to run {tool_name} analysis now?"
            return {
                "response": confirmation_msg,
                "pending_action": tool_name
            }
            
        def execute_action(state: AgentState) -> AgentState:
            """Execute confirmed action"""
            if not state.get('confirmed'):
                return {"response": 'Action not confirmed'}
                
            tool = self._get_tool(state['pending_action'])
            if not tool:
                return {
                    "response": f"Sorry, {state['pending_action']} is not available"
                }
                
            try:
                result = tool.run(state.get('tool_input', {}))
                return {"response": f"Action completed: {result}"}
            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                return {"response": f"Error executing action: {str(e)}"}
                
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
            if state.get('intent') == 'info':
                return "handle_info_request"
            return "confirm_action"
            
        def route_action(state: AgentState) -> str:
            """Route action based on confirmation"""
            if state.get('confirmed', False):
                return "execute_action"
            return "end"
        
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
        if not query or not isinstance(query, str):
            return {
                'error': 'Invalid query format',
                'response': 'Please provide a valid text query.'
            }

        try:
            # Initialize state
            state: AgentState = {
                'query': query,
                'confirmed': kwargs.get('confirmed', False),
                'tool_input': kwargs,
                'intent': None,
                'response': '',
                'pending_action': ''
            }
            
            # Create and compile workflow
            workflow = self._create_graph()
            app = workflow.compile()
            
            # Run workflow using invoke
            try:
                result = app.invoke(state)
            except Exception as workflow_error:
                logger.error(f"Workflow execution error: {workflow_error}")
                return {
                    'error': str(workflow_error),
                    'response': "I encountered an error in processing your request. Please try rephrasing or simplifying your query."
                }
            
            # Update conversation history
            self.conversation_history.append({
                'role': 'user',
                'content': query
            })
            self.conversation_history.append({
                'role': 'assistant',
                'content': result.get('response', 'No response generated')
            })
            
            return {
                'response': result.get('response', 'No response generated'),
                'pending_action': result.get('pending_action'),
                'requires_confirmation': bool(result.get('pending_action')) and not result.get('confirmed'),
                'conversation_history': self.conversation_history[-5:]  # Last 5 messages
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {
                'error': str(e),
                'response': "I encountered an unexpected error. Please try again or contact support if the issue persists."
            }
