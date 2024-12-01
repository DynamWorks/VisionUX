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
        self.logger = logging.getLogger(__name__)
        
        # Initialize RAG service and chain
        from backend.services.rag_service import RAGService
        self.rag_service = RAGService()
        self._current_chain = None
        
        # Define system prompt
        self.system_prompt = """You are a helpful AI assistant that analyzes videos and chats about them using RAG (Retrieval Augmented Generation).

Your primary role is to:
1. Use RAG to provide informed responses about video content
2. Suggest relevant analysis functions when they could provide more insights
3. Always get explicit confirmation before executing any actions
4. Maintain a natural, conversational tone while being precise

When responding:
- Default to using RAG to answer questions about video content
- If RAG data is insufficient, suggest running appropriate analysis
- For action requests, clearly explain what will happen and get confirmation
- Provide specific suggestions based on the conversation context

Available tools:
{tools}

Example interactions:
User: "What's happening in the video?"
Assistant: Let me check the analysis data... [provides RAG-based response]. I notice some interesting activity - would you like me to run a fresh scene analysis for more details?

User: "Are there any cars?"
Assistant: From the existing analysis, [RAG-based response about vehicles]. I could run object detection specifically focused on vehicles if you'd like more precise information. Would you like me to do that?

User: "Yes, run the detection"
Assistant: I'll run object detection to identify vehicles and other objects. This will analyze the current video frames. Would you like me to proceed with this analysis now?
"""

    def _create_graph(self) -> StateGraph:
        """Create the agent workflow graph"""
        
        # Define graph nodes
        def analyze_query(state: AgentState) -> AgentState:
            """Analyze user query to determine intent"""
            # Check if this is a follow-up query
            if state['follow_up'] and state['last_action']:
                # Continue with previous action context
                return {**state, "intent": state['last_action']['intent']}
                
            # Default to RAG intent unless explicit action requested
            intent = 'rag'
            
            # Check for action keywords
            action_keywords = {
                'run': ['run', 'execute', 'perform', 'start', 'do'],
                'analyze': ['analyze', 'detect', 'find', 'identify'],
                'info': ['explain', 'tell', 'what is', 'how does'],
                'continue': ['continue', 'proceed', 'go ahead', 'yes', 'ok', 'sure'],
                'rag': ['what', 'how', 'why', 'when', 'where', 'who', 'describe', 'tell me about']
            }
            
            # Check for continuation of previous action
            if state['last_action'] and any(kw in state['query'].lower() for kw in action_keywords['continue']):
                return {**state, "intent": state['last_action']['intent']}
            
            # Check for explicit action requests first
            query_lower = state['query'].lower()
            for action in ['run', 'analyze']:
                if any(kw in query_lower for kw in action_keywords[action]):
                    intent = action
                    break
                    
            # If no explicit action, use RAG by default
            if intent == 'rag':
                # Add suggestion for relevant analysis if appropriate
                if any(kw in query_lower for kw in ['object', 'person', 'activity', 'movement']):
                    state['suggest_analysis'] = True
                    
            return {**state, "intent": intent}
            
        def handle_rag_request(state: AgentState) -> AgentState:
            """Handle RAG-based query"""
            try:
                # Get RAG response
                rag_response = self.rag_service.query_knowledge_base(
                    state['query'],
                    self._current_chain,
                    chat_history=state.get('conversation_context', [])
                )
                
                response = rag_response.get('answer', '')
                
                # Add analysis suggestion if flagged
                if state.get('suggest_analysis'):
                    response += "\n\nI could run additional analysis to get more specific information about this. Would you like me to do that?"
                    state['pending_action'] = self._get_relevant_tool(state['query'])
                
                return {**state, "response": response, "rag_response": rag_response}
                
            except Exception as e:
                self.logger.error(f"RAG query error: {e}")
                return {**state, "response": "I encountered an error accessing the analysis data. Would you like me to run a new analysis?"}
            
            # Add context about available analysis if any
            analysis_files = list(Path("tmp_content/analysis").glob("*.json"))
            if not analysis_files:
                response = "No previous analysis results available. Would you like me to run a scene analysis?"
            else:
                response = self.llm.invoke([
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=state['query'])
                ]).content
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
        workflow.add_node("handle_rag_request", handle_rag_request)
        workflow.add_node("confirm_action", confirm_action)
        workflow.add_node("execute_action", execute_action)
        
        # Define conditional edges
        def route_query(state: AgentState) -> str:
            """Route to appropriate handler based on intent"""
            return "handle_rag_request" if state.get('intent') == 'rag' else "confirm_action"
            
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
        workflow.add_edge("handle_rag_request", END)
        workflow.add_edge("execute_action", END)
        
        # Set entry point
        workflow.set_entry_point("analyze_query")
        
        return workflow

    def _is_follow_up(self, query: str) -> bool:
        """Detect if query is a follow-up to previous conversation"""
        follow_up_indicators = [
            'what about',
            'and then',
            'after that',
            'also',
            'what else',
            'continue',
            'go ahead',
            'yes',
            'ok',
            'proceed'
        ]
        return any(indicator in query.lower() for indicator in follow_up_indicators)

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
                'executed_action': None,
                'conversation_context': kwargs.get('conversation_context', []),
                'last_action': kwargs.get('last_action'),
                'follow_up': self._is_follow_up(query)
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
