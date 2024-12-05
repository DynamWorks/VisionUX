from typing import Dict, List, Optional, Any, TypedDict, Literal, Annotated, Union, Tuple, Type
from typing_extensions import TypeAlias
from pathlib import Path
import operator
import time
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.tools import BaseTool
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
import re
import json
import logging

logger = logging.getLogger(__name__)

# Define state types
class AgentState(TypedDict):
    messages: List[Dict[str, str]]  # Chat history
    current_query: str              # Current user query
    retriever_result: Optional[str] # Result from retriever
    suggested_tool: Optional[str]   # Tool suggested for execution
    tool_input: Optional[Dict]      # Input parameters for tool
    requires_confirmation: bool     # Whether tool execution needs confirmation
    confirmed: bool                # Whether user confirmed tool execution
    final_response: Optional[str]   # Final response to return to user

class VideoAnalysisAgent:
    """Agent for handling video analysis queries and actions"""
    
    def __init__(self, llm, tools: List[Tool], retriever):
        self.llm = llm
        self.tools = tools
        self.retriever = retriever
        self.logger = logging.getLogger(__name__)
        
        # Initialize tool executor
        self.tool_executor = ToolExecutor(tools)
        
        # Create the graph
        self.workflow = self._create_workflow()
        
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

    def _create_workflow(self) -> StateGraph:
        """Create the agent workflow graph"""
        
        # Create workflow graph
        workflow = StateGraph(AgentState)
        
        # Define nodes
        workflow.add_node("retrieve", self._retrieve_info)
        workflow.add_node("suggest_tool", self._suggest_tool)
        workflow.add_node("execute_tool", self._execute_tool)
        workflow.add_node("generate_response", self._generate_response)
        
        # Define edges
        workflow.add_edge("retrieve", "suggest_tool")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "suggest_tool",
            self._route_after_suggestion,
            {
                "execute": "execute_tool",
                "respond": "generate_response"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_tool",
            self._route_after_execution,
            {
                "respond": "generate_response",
                "suggest": "suggest_tool"
            }
        )
        
        # Add edge to end
        workflow.add_edge("generate_response", END)
        
        # Set entry point
        workflow.set_entry_point("retrieve")
        
        return workflow
            
    def _retrieve_info(self, state: AgentState) -> AgentState:
        """Use retriever to get initial response"""
        query = state["current_query"]
        chat_history = state.get("messages", [])
        
        # Get retriever response
        result = self.retriever.get_relevant_documents(query)
        
        return {
            **state,
            "retriever_result": result[0].page_content if result else None,
            "messages": chat_history + [
                {"role": "user", "content": query}
            ]
        }
        
    def _suggest_tool(self, state: AgentState) -> AgentState:
        """Analyze query and suggest appropriate tool"""
        # Create tool suggestion prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze the query and retriever result to suggest an appropriate tool if needed."),
            MessagesPlaceholder(variable_name="messages"),
            ("user", "Query: {query}\nRetriever result: {result}\n\nShould I suggest a tool?")
        ])
        
        # Get suggestion from LLM
        chain = prompt | self.llm | JsonOutputParser()
        suggestion = chain.invoke({
            "messages": state["messages"],
            "query": state["current_query"],
            "result": state["retriever_result"]
        })
        
        return {
            **state,
            "suggested_tool": suggestion.get("tool"),
            "tool_input": suggestion.get("input"),
            "requires_confirmation": bool(suggestion.get("tool")),
            "confirmed": False
        }
        
    def _execute_tool(self, state: AgentState) -> AgentState:
        """Execute suggested tool if confirmed"""
        if not state["confirmed"]:
            return {
                **state,
                "final_response": "Please confirm if you want to execute this tool."
            }
            
        tool = state["suggested_tool"]
        tool_input = state["tool_input"]
        
        # Execute tool
        result = self.tool_executor.invoke({
            "tool": tool,
            "tool_input": tool_input
        })
        
        return {
            **state,
            "tool_result": result,
            "messages": state["messages"] + [
                {"role": "assistant", "content": f"Executed {tool} with result: {result}"}
            ]
        }
        
    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate final response combining retriever and tool results"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate a helpful response using the retriever result and any tool outputs."),
            MessagesPlaceholder(variable_name="messages"),
            ("user", "Generate response for: {query}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "messages": state["messages"],
            "query": state["current_query"]
        })
        
        return {
            **state,
            "final_response": response,
            "messages": state["messages"] + [
                {"role": "assistant", "content": response}
            ]
        }
            
    def _route_after_suggestion(self, state: AgentState) -> str:
        """Route to next node after tool suggestion"""
        if state["requires_confirmation"]:
            return "execute" if state["confirmed"] else "respond"
        return "respond"
        
    def _route_after_execution(self, state: AgentState) -> str:
        """Route to next node after tool execution"""
        # Check if we need to suggest another tool
        if state.get("tool_result", {}).get("suggest_next_tool"):
            return "suggest"
        return "respond"
        
    def process_query(self, query: str, chat_history: List[Dict] = None) -> Dict:
        """Process user query through the workflow"""
        # Initialize state
        state = AgentState(
            messages=chat_history or [],
            current_query=query,
            retriever_result=None,
            suggested_tool=None,
            tool_input=None,
            requires_confirmation=False,
            confirmed=False,
            final_response=None
        )
        
        # Run workflow
        result = self.workflow.invoke(state)
        
        return {
            "response": result["final_response"],
            "suggested_tool": result.get("suggested_tool"),
            "requires_confirmation": result.get("requires_confirmation", False),
            "chat_history": result["messages"]
        }
            
        def confirm_action(state: AgentState) -> AgentState:
            """Get confirmation for action with tool description"""
            if not isinstance(state, dict):
                logger.error("Invalid state type")
                return {**state, "response": "Internal error: invalid state"}

            if state.get('confirmed'):
                return {**state, "response": 'Action confirmed'}
                
            tool_name = self._get_relevant_tool(state.get('query', ''))
            if not tool_name:
                return {**state, "response": "No relevant tool found for this query"}
                
            tool = self._get_tool(tool_name)
            if not tool:
                return {**state, "response": f"Tool {tool_name} is not available"}
                
            confirmation_msg = (
                f"Would you like me to run {tool_name} analysis now?\n"
                f"This will: {tool.description}"
            )
            
            return {
                **state,
                "response": confirmation_msg,
                "pending_action": tool_name,
                "tool_description": tool.description
            }
            
        def execute_action(state: AgentState) -> AgentState:
            """Execute confirmed action with input validation"""
            if not isinstance(state, dict):
                logger.error("Invalid state type")
                return {**state, "response": "Internal error: invalid state"}

            if not state.get('confirmed'):
                return {**state, "response": 'Action not confirmed'}
                
            tool_name = state.get('pending_action')
            if not tool_name:
                logger.error("No pending action in state")
                return {**state, "response": "No action specified"}
                
            tool = self._get_tool(tool_name)
            if not tool:
                logger.error(f"Tool not found: {tool_name}")
                return {
                    **state,
                    "response": f"Sorry, {tool_name} is not available"
                }
                
            try:
                # Validate tool inputs
                tool_input = state.get('tool_input', {})
                if hasattr(tool, 'validate_inputs'):
                    tool_input = tool.validate_inputs(tool_input)
                
                # Execute tool
                result = tool.run(tool_input)
                
                # Update state with result
                return {
                    **state, 
                    "response": f"Action completed: {result}",
                    "tool_result": result,
                    "execution_timestamp": time.time()
                }
            except ValueError as ve:
                logger.warning(f"Tool input validation error: {ve}")
                return {**state, "response": f"Invalid input: {str(ve)}"}
            except Exception as e:
                logger.error(f"Tool execution error: {e}", exc_info=True)
                return {
                    **state, 
                    "response": f"Error executing action: {str(e)}",
                    "error": str(e)
                }
                
        return self.workflow

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
